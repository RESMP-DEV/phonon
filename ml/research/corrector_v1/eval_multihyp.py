"""Evaluate S1/M3/M4 correctors, v0 adapter, raw v2, and oracles on holdout gates."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_ROOT,
    LETTERS,
    LETTER_MODEL,
    MISSING,
    PARAKEET_JSONL,
    QWEN_ID,
    RESEARCH_ROOT,
    V0_ADAPTER,
    apply_chat_template,
    chat_messages,
    ensure_data_dirs,
    format_user,
    heartbeat,
    hf_token,
    load_v0_common,
    max_new_tokens_for,
    option0_pred_path,
    read_jsonl,
    set_hf_env,
    write_jsonl,
)

SET_ORDER = ("wispr_holdout120", "wispr_edit25")


def load_option0_hyps(split: str) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for key in LETTER_MODEL.values():
        path = option0_pred_path(split, key)
        for row in read_jsonl(path):
            by_id.setdefault(row["id"], {})[key] = row.get("hypothesis") or ""
    return by_id


def eval_rows() -> dict[str, list[dict]]:
    parakeet = {row["id"]: row for row in read_jsonl(PARAKEET_JSONL)}
    sets: dict[str, list[dict]] = {name: [] for name in SET_ORDER}
    holdout_hyps = load_option0_hyps("wispr_holdout120")
    edit_hyps = load_option0_hyps("wispr_edit25")
    v0 = load_v0_common()
    for row in v0.load_wispr_pairs():
        if row.get("split") != "wispr_holdout120":
            continue
        row_id = row["id"]
        hyps = holdout_hyps.get(row_id, {})
        pk = parakeet.get(row_id, {})
        if "parakeet_v2" not in hyps:
            hyps = dict(hyps)
            hyps["parakeet_v2"] = pk.get("parakeet_raw") or ""
        sets["wispr_holdout120"].append(
            {
                "id": row_id,
                "reference": row.get("target") or "",
                "hyps": hyps,
                "v2": hyps.get("parakeet_v2") or "",
            }
        )
    for row in v0.load_july_rows():
        row_id = row["id"]
        hyps = edit_hyps.get(row_id, {})
        pk = parakeet.get(row_id, {})
        if "parakeet_v2" not in hyps:
            hyps = dict(hyps)
            hyps["parakeet_v2"] = pk.get("parakeet_raw") or ""
        sets["wispr_edit25"].append(
            {
                "id": row_id,
                "reference": row.get("edited") or row.get("target") or "",
                "hyps": hyps,
                "v2": hyps.get("parakeet_v2") or "",
            }
        )
    return sets


def letters_user(row: dict, letters: tuple[str, ...], drop: str | None) -> str:
    hyps = {}
    for letter in letters:
        key = LETTER_MODEL[letter]
        text = row["hyps"].get(key) or ""
        if drop == letter:
            text = MISSING
        hyps[letter] = text
    return format_user(hyps, letters)


def oracle_hyp(row: dict, letters: tuple[str, ...]) -> str:
    v0 = load_v0_common()
    best = None
    best_wer = None
    for letter in letters:
        hyp = row["hyps"].get(LETTER_MODEL[letter]) or ""
        wer = v0.pair_wer(row["reference"], hyp, v0.fair_norm)
        if best_wer is None or wer < best_wer:
            best_wer = wer
            best = hyp
    return best or ""


def score_hyps(rows: list[dict], hyps: list[str], v2_inputs: list[str]) -> dict[str, Any]:
    v0 = load_v0_common()
    refs = [r["reference"] for r in rows]
    scores = v0.score_lists(refs, hyps)
    worse = 0
    for row, hyp, v2 in zip(rows, hyps, v2_inputs, strict=True):
        in_wer = v0.pair_wer(row["reference"], v2, v0.fair_norm)
        out_wer = v0.pair_wer(row["reference"], hyp, v0.fair_norm)
        if out_wer - in_wer > 1e-9:
            worse += 1
    scores["worse_frac"] = worse / len(rows) if rows else 0.0
    scores["n"] = len(rows)
    return scores


def load_stack(model_id: str, adapter: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    token = hf_token()
    processor = None
    last = None
    src = str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else model_id
    for loader in (AutoProcessor, AutoTokenizer):
        try:
            processor = loader.from_pretrained(src, token=token, trust_remote_code=True)
            break
        except Exception as exc:
            last = exc
    if processor is None:
        raise RuntimeError(f"processor load failed: {last}")
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "token": token,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, processor, tokenizer


def generate_one(
    model,
    processor,
    tokenizer,
    user_text: str,
    letters: tuple[str, ...],
    max_new: int,
    extra_gen: dict | None = None,
) -> dict[str, Any]:
    import torch

    messages = chat_messages(user_text, letters=letters)
    prompt = apply_chat_template(processor, messages, add_generation_prompt=True)
    if hasattr(processor, "__call__") and processor is not tokenizer:
        try:
            inputs = processor(text=prompt, return_tensors="pt")
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    eos_ids: list[int] = []
    for value in (getattr(tokenizer, "eos_token_id", None), pad_id):
        if isinstance(value, list):
            eos_ids.extend(int(v) for v in value if v is not None)
        elif value is not None:
            eos_ids.append(int(value))
    unk = getattr(tokenizer, "unk_token_id", None)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            eos_ids.append(tid)
    eos_ids = list(dict.fromkeys(eos_ids))
    gen_kwargs = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "pad_token_id": pad_id,
        "eos_token_id": eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
    }
    if extra_gen:
        gen_kwargs.update(extra_gen)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = None
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - t0
    new_tokens = out[0, input_len:]
    n_new = int(new_tokens.shape[0])
    v0 = load_v0_common()
    text = v0.strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return {
        "text": text,
        "prompt_tokens": input_len,
        "new_tokens": n_new,
        "hit_cap": n_new >= max_new,
        "max_new_tokens": max_new,
        "elapsed_seconds": elapsed,
    }


def run_condition(
    label: str,
    model,
    processor,
    tokenizer,
    sets: dict[str, list[dict]],
    letters: tuple[str, ...],
    drop: str | None,
    single_v2: bool,
    limit: int,
) -> dict[str, Any]:
    result = {"label": label, "letters": list(letters), "drop": drop, "sets": {}}
    last_beat = time.perf_counter()
    for set_name in SET_ORDER:
        rows = sets[set_name]
        if limit:
            rows = rows[:limit]
        outputs = []
        texts = []
        for index, row in enumerate(rows, start=1):
            if single_v2:
                user = row["v2"]
                use_letters: tuple[str, ...] = ("A",)
            else:
                user = letters_user(row, letters, drop)
                use_letters = letters
            max_new = max_new_tokens_for(user)
            gen = generate_one(model, processor, tokenizer, user, use_letters, max_new)
            gen["id"] = row["id"]
            gen["input"] = user
            gen["reference"] = row["reference"]
            outputs.append(gen)
            texts.append(gen["text"])
            if index % 20 == 0:
                print(f"{label} {set_name} {index}/{len(rows)}", flush=True)
            now = time.perf_counter()
            if index == 1 or now - last_beat >= 600:
                heartbeat()
                last_beat = now
        v2_inputs = [r["v2"] for r in rows]
        scores = score_hyps(rows, texts, v2_inputs)
        scores["mean_latency_seconds"] = (
            sum(g["elapsed_seconds"] for g in outputs) / len(outputs) if outputs else 0.0
        )
        scores["mean_prompt_tokens"] = (
            sum(g["prompt_tokens"] for g in outputs) / len(outputs) if outputs else 0.0
        )
        scores["mean_new_tokens"] = (
            sum(g["new_tokens"] for g in outputs) / len(outputs) if outputs else 0.0
        )
        scores["hit_cap"] = sum(1 for g in outputs if g["hit_cap"])
        scores["max_new_tokens_used"] = max((g["max_new_tokens"] for g in outputs), default=0)
        result["sets"][set_name] = {**scores, "predictions": outputs}
        pred_path = DATA_ROOT / "eval_predictions" / f"{label}_{set_name}.jsonl"
        write_jsonl(
            pred_path,
            [
                {
                    "id": g["id"],
                    "input": g["input"],
                    "output": g["text"],
                    "reference": g["reference"],
                    "prompt_tokens": g["prompt_tokens"],
                    "new_tokens": g["new_tokens"],
                    "hit_cap": g["hit_cap"],
                    "elapsed_seconds": g["elapsed_seconds"],
                }
                for g in outputs
            ],
        )
    return result


def cpu_baselines(sets: dict[str, list[dict]], limit: int) -> list[dict[str, Any]]:
    results = []

    def pack(label: str, getter) -> dict[str, Any]:
        block = {"label": label, "sets": {}}
        for set_name in SET_ORDER:
            rows = sets[set_name][:limit] if limit else sets[set_name]
            hyps = [getter(row) for row in rows]
            v2_inputs = [r["v2"] for r in rows]
            block["sets"][set_name] = score_hyps(rows, hyps, v2_inputs)
            block["sets"][set_name]["mean_prompt_tokens"] = None
            block["sets"][set_name]["mean_latency_seconds"] = None
            block["sets"][set_name]["hit_cap"] = 0
        return block

    results.append(pack("raw v2", lambda row: row["v2"]))
    results.append(pack("oracle-3", lambda row: oracle_hyp(row, ("A", "B", "C"))))
    results.append(pack("oracle-4", lambda row: oracle_hyp(row, LETTERS)))
    return results


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def pick_set(models: list[dict], label: str, set_name: str) -> dict | None:
    for model in models:
        if model.get("label") == label:
            return (model.get("sets") or {}).get(set_name)
    return None


def build_markdown(payload: dict) -> str:
    models = payload.get("models") or []
    rows_spec = payload.get("table_rows") or []
    lines = [
        "# Corrector v1 results",
        "",
        "Multi-hypothesis LoRA corrector on Wispr holdouts. Fair WER is Whisper",
        "`EnglishTextNormalizer` then jiwer corpus WER. Strict lowercase WER lowercases",
        "both sides and otherwise leaves digits and punctuation. Worse-frac is the",
        "fraction of holdout120 clips where the system is worse than Parakeet v2 raw.",
        "tokens/clip is mean prompt tokens on holdout120. Generation cap is",
        "`max(256, 3 * input_word_count)` per row.",
        "",
        f"Seed {payload.get('seed')}. Max prompt tokens in training: "
        f"{payload.get('max_prompt_tokens')}. Train seq len: {payload.get('max_seq_len')}.",
        f"Outputs that hit the generation cap: {payload.get('hit_cap_total')}.",
        "",
        "| system | holdout120 fair | holdout120 strict | edit25 fair | edit25 strict | "
        "worse-frac holdout | tokens/clip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in rows_spec:
        h = pick_set(models, label, "wispr_holdout120") or {}
        e = pick_set(models, label, "wispr_edit25") or {}
        lines.append(
            f"| {label} | {fmt(h.get('fair_wer'))} | {fmt(h.get('strict_lc_wer'))} | "
            f"{fmt(e.get('fair_wer'))} | {fmt(e.get('strict_lc_wer'))} | "
            f"{fmt(h.get('worse_frac'))} | {fmt(h.get('mean_prompt_tokens'))} |"
        )
    lines += ["", "## wispr_train recognizers (new transcripts vs Wispr target)", ""]
    lines += [
        "| model | n | fair WER | strict lc WER | RTF | audio seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, block in (payload.get("train_asr") or {}).items():
        if block.get("missing"):
            lines.append(f"| {key} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {key} | {block.get('n')} | {fmt(block.get('fair_wer'))} | "
            f"{fmt(block.get('strict_lc_wer'))} | {fmt(block.get('realtime_factor'))} | "
            f"{fmt(block.get('audio_seconds'))} |"
        )
    lat = payload.get("latency") or {}
    lines += [
        "",
        "## latency (3090, mean per clip)",
        "",
        f"S1 holdout120: {fmt(lat.get('s1_holdout'))} s. "
        f"M4 holdout120: {fmt(lat.get('m4_holdout'))} s. "
        f"S1 edit25: {fmt(lat.get('s1_edit'))} s. "
        f"M4 edit25: {fmt(lat.get('m4_edit'))} s.",
        "",
        "## what the numbers say",
        "",
    ]
    for sentence in payload.get("sentences") or []:
        lines.append(sentence)
        lines.append("")
    notes = payload.get("notes") or []
    if notes:
        lines += ["## notes", ""]
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def table_row_order() -> list[str]:
    rows = ["raw v2", "oracle-3", "oracle-4", "v0 adapter", "S1", "M3", "M4"]
    for letter in ("A", "B", "C"):
        rows.append(f"M3 minus {letter}")
    for letter in LETTERS:
        rows.append(f"M4 minus {letter}")
    return rows


def make_sentences(payload: dict) -> list[str]:
    models = payload.get("models") or []
    h_s1 = pick_set(models, "S1", "wispr_holdout120") or {}
    h_m3 = pick_set(models, "M3", "wispr_holdout120") or {}
    h_m4 = pick_set(models, "M4", "wispr_holdout120") or {}
    h_v0 = pick_set(models, "v0 adapter", "wispr_holdout120") or {}
    h_raw = pick_set(models, "raw v2", "wispr_holdout120") or {}
    h_o3 = pick_set(models, "oracle-3", "wispr_holdout120") or {}
    h_o4 = pick_set(models, "oracle-4", "wispr_holdout120") or {}
    e_s1 = pick_set(models, "S1", "wispr_edit25") or {}
    e_m4 = pick_set(models, "M4", "wispr_edit25") or {}
    train = payload.get("train_asr") or {}
    coh = train.get("cohere") or {}
    uni = train.get("parakeet_unified") or {}
    drops = []
    for letter in LETTERS:
        block = pick_set(models, f"M4 minus {letter}", "wispr_holdout120") or {}
        drops.append((letter, block.get("fair_wer")))
    present = [(letter, wer) for letter, wer in drops if wer is not None]
    worst = max(present, key=lambda item: item[1]) if present else ("?", None)
    s1 = h_s1.get("fair_wer")
    m4 = h_m4.get("fair_wer")
    delta = None if s1 is None or m4 is None else m4 - s1
    noise = 0.01
    if delta is None:
        beat = "M4 and S1 holdout fair WER are incomplete."
    elif delta < -noise:
        beat = (
            f"M4 beats S1 on holdout120 fair WER by {-delta:.4f} absolute "
            f"({fmt(m4)} vs {fmt(s1)}), larger than a 1-point noise band."
        )
    elif delta > noise:
        beat = (
            f"M4 does not beat S1 on holdout120 fair WER; it is {delta:.4f} worse "
            f"({fmt(m4)} vs {fmt(s1)})."
        )
    else:
        beat = (
            f"M4 and S1 are within 1 WER point on holdout120 fair WER "
            f"({fmt(m4)} vs {fmt(s1)}); treat that as noise, not a win."
        )
    coh_rtf = coh.get("realtime_factor")
    uni_rtf = uni.get("realtime_factor")
    ratio = None
    if coh_rtf and uni_rtf:
        ratio = coh_rtf / uni_rtf if uni_rtf else None
    m3 = h_m3.get("fair_wer")
    coh_worth = (
        f"Adding Cohere (M4 vs M3) changes holdout120 fair WER from {fmt(m3)} to {fmt(m4)}; "
        f"Cohere train RTF is {fmt(coh_rtf)} vs unified {fmt(uni_rtf)}"
        + (f" ({ratio:.1f}x slower)." if ratio else ".")
    )
    if m3 is not None and m4 is not None and ratio is not None:
        gain = m3 - m4
        if gain < 0.005 and ratio >= 2:
            coh_worth += " That gain does not pay for the extra recognizer cost on this gate."
        elif gain >= 0.01:
            coh_worth += " The extra recognizer moves WER enough to keep in the mix."
    drop_s = (
        f"Dropping hypothesis {worst[0]} hurts M4 holdout120 the most "
        f"(fair WER {fmt(worst[1])} vs full M4 {fmt(m4)})."
    )
    oracle_s = (
        f"Raw Parakeet v2 holdout120 fair WER is {fmt(h_raw.get('fair_wer'))}; "
        f"oracle-3 is {fmt(h_o3.get('fair_wer'))} and oracle-4 is {fmt(h_o4.get('fair_wer'))}. "
        f"The v0 adapter with the per-row generation cap is {fmt(h_v0.get('fair_wer'))}."
    )
    edit_s = (
        f"On wispr_edit25, S1 fair WER is {fmt(e_s1.get('fair_wer'))} and M4 is "
        f"{fmt(e_m4.get('fair_wer'))}; holdout worse-frac is {fmt(h_s1.get('worse_frac'))} "
        f"for S1 and {fmt(h_m4.get('worse_frac'))} for M4."
    )
    return [beat, coh_worth, drop_s, oracle_s, edit_s]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--out-json", type=Path, default=RESEARCH_ROOT / "results.json")
    parser.add_argument("--out-md", type=Path, default=RESEARCH_ROOT / "results.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_hf_env()
    ensure_data_dirs()
    sets = eval_rows()
    for name, rows in sets.items():
        empty = sum(1 for r in rows if not r["reference"] or not r["v2"])
        print(f"eval set {name}: n={len(rows)} empty_ref_or_v2={empty}", flush=True)
    models = cpu_baselines(sets, args.limit)
    notes = []
    if not args.skip_generate:
        import torch

        jobs = [
            ("v0 adapter", V0_ADAPTER, ("A",), True, [None]),
            ("S1", DATA_ROOT / "adapters" / "S1", ("A",), True, [None]),
            ("M3", DATA_ROOT / "adapters" / "M3", ("A", "B", "C"), False, [None, "A", "B", "C"]),
            ("M4", DATA_ROOT / "adapters" / "M4", LETTERS, False, [None, "A", "B", "C", "D"]),
        ]
        for base_label, adapter, letters, single_v2, drops in jobs:
            if not adapter.exists() or not (adapter / "adapter_config.json").exists():
                notes.append(f"missing adapter {adapter}")
                print(f"missing adapter {adapter}", flush=True)
                continue
            print(f"load {base_label} {adapter}", flush=True)
            heartbeat()
            model, processor, tokenizer = load_stack(QWEN_ID, adapter)
            try:
                for drop in drops:
                    if drop is None:
                        label = base_label
                    else:
                        label = f"{base_label} minus {drop}"
                    print(f"eval {label}", flush=True)
                    models.append(
                        run_condition(
                            label,
                            model,
                            processor,
                            tokenizer,
                            sets,
                            letters,
                            drop,
                            single_v2,
                            args.limit,
                        )
                    )
            finally:
                del model
                torch.cuda.empty_cache()
    build_stats_path = DATA_ROOT / "build_stats.json"
    build_stats = json.loads(build_stats_path.read_text()) if build_stats_path.exists() else {}
    train_asr = build_stats.get("train_asr") or {}
    tokens = build_stats.get("tokens") or {}
    hit_cap_total = 0
    for model in models:
        for block in (model.get("sets") or {}).values():
            hit_cap_total += int(block.get("hit_cap") or 0)
    s1_h = pick_set(models, "S1", "wispr_holdout120") or {}
    m4_h = pick_set(models, "M4", "wispr_holdout120") or {}
    s1_e = pick_set(models, "S1", "wispr_edit25") or {}
    m4_e = pick_set(models, "M4", "wispr_edit25") or {}
    payload = {
        "seed": build_stats.get("seed"),
        "max_prompt_tokens": tokens.get("max_tokens"),
        "max_seq_len": tokens.get("seq_len"),
        "token_stats": tokens,
        "train_asr": train_asr,
        "build": {k: v for k, v in build_stats.items() if k in {"s1", "m3", "m4"}},
        "hit_cap_total": hit_cap_total,
        "table_rows": table_row_order(),
        "latency": {
            "s1_holdout": s1_h.get("mean_latency_seconds"),
            "m4_holdout": m4_h.get("mean_latency_seconds"),
            "s1_edit": s1_e.get("mean_latency_seconds"),
            "m4_edit": m4_e.get("mean_latency_seconds"),
        },
        "models": [
            {
                k: v
                for k, v in model.items()
                if k != "sets"
            }
            | {
                "sets": {
                    sname: {kk: vv for kk, vv in sblock.items() if kk != "predictions"}
                    for sname, sblock in model.get("sets", {}).items()
                }
            }
            for model in models
        ],
        "notes": notes,
    }
    payload["sentences"] = make_sentences(payload)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.out_md.write_text(build_markdown(payload) + "\n")
    print(f"wrote {args.out_md} and {args.out_json}", flush=True)
    print(build_markdown(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
