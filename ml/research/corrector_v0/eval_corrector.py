"""Evaluate baseline, base LM, and LoRA correctors on the three holdout sets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_ROOT,
    GEMMA_ID,
    HOLDOUT_SPLITS,
    PARAKEET_JSONL,
    QWEN_FALLBACK_ID,
    QWEN_ID,
    RESEARCH_ROOT,
    SYSTEM_PROMPT,
    TTS_PAIRS,
    chat_messages,
    fair_norm,
    heartbeat,
    load_july_rows,
    load_wispr_pairs,
    pair_wer,
    read_jsonl,
    record_failure,
    score_lists,
    set_hf_env,
    strip_thinking,
)

SET_ORDER = ("wispr_holdout120", "wispr_edit25", "wispr_text_holdout")


def eval_rows() -> dict[str, list[dict]]:
    parakeet = {row["id"]: row for row in read_jsonl(PARAKEET_JSONL)}
    sets: dict[str, list[dict]] = {name: [] for name in SET_ORDER}
    for row in load_wispr_pairs():
        split = row.get("split")
        if split == "wispr_holdout120":
            pk = parakeet.get(row["id"], {})
            sets[split].append(
                {
                    "id": row["id"],
                    "input": pk.get("parakeet_raw") or "",
                    "reference": row.get("target") or "",
                    "wispr_asr": row.get("asr") or "",
                }
            )
        elif split == "wispr_text_holdout":
            sets[split].append(
                {
                    "id": row["id"],
                    "input": row.get("asr") or "",
                    "reference": row.get("target") or "",
                    "wispr_asr": row.get("asr") or "",
                }
            )
    for row in load_july_rows():
        pk = parakeet.get(row["id"], {})
        sets["wispr_edit25"].append(
            {
                "id": row["id"],
                "input": pk.get("parakeet_raw") or "",
                "reference": row.get("edited") or row.get("target") or "",
                "wispr_asr": row.get("asr") or "",
            }
        )
    return sets


def apply_prompt(processor, user_text: str) -> str:
    messages = chat_messages(user_text)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    for extra in ({"enable_thinking": False}, {}):
        try:
            text = processor.apply_chat_template(messages, **kwargs, **extra)
            if isinstance(text, list):
                text = text[0]
            return str(text)
        except TypeError:
            continue
        except Exception:
            mm = []
            for msg in messages:
                content = msg["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                mm.append({"role": msg["role"], "content": content})
            try:
                text = processor.apply_chat_template(
                    mm, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if isinstance(text, list):
                    text = text[0]
                return str(text)
            except Exception:
                pass
    return SYSTEM_PROMPT + "\n\n" + user_text + "\n"


def load_stack(model_id: str, adapter: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from common import hf_token

    token = hf_token()
    processor = None
    last = None
    for loader in (AutoProcessor, AutoTokenizer):
        try:
            processor = loader.from_pretrained(
                str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else model_id,
                token=token,
                trust_remote_code=True,
            )
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
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, processor, tokenizer


def generate_one(model, processor, tokenizer, user_text: str, max_new_tokens: int) -> str:
    import torch

    prompt = apply_prompt(processor, user_text)
    if hasattr(processor, "__call__") and processor is not tokenizer:
        try:
            inputs = processor(text=prompt, return_tensors="pt")
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
    }
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = None
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking(text)


def generate_batch(model, processor, tokenizer, texts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy generation for a batch of prompts with left padding; same outputs as generate_one."""
    import torch

    if len(texts) == 1:
        return [generate_one(model, processor, tokenizer, texts[0], max_new_tokens)]
    prompts = [apply_prompt(processor, t) for t in texts]
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
    }
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = None
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    outs = []
    for i in range(out.shape[0]):
        text = tokenizer.decode(out[i, input_len:], skip_special_tokens=True)
        outs.append(strip_thinking(text))
    return outs


def worse_examples(rows: list[dict], outputs: list[str], k: int = 10) -> tuple[float, list[dict]]:
    scored = []
    worse = 0
    for row, hyp in zip(rows, outputs, strict=True):
        in_wer = pair_wer(row["reference"], row["input"], fair_norm)
        out_wer = pair_wer(row["reference"], hyp, fair_norm)
        delta = out_wer - in_wer
        if delta > 1e-9:
            worse += 1
        scored.append(
            {
                "id": row["id"],
                "input": row["input"],
                "output": hyp,
                "reference": row["reference"],
                "input_fair_wer": in_wer,
                "output_fair_wer": out_wer,
                "delta_fair_wer": delta,
            }
        )
    scored.sort(key=lambda item: item["delta_fair_wer"], reverse=True)
    frac = worse / len(rows) if rows else 0.0
    return frac, scored[:k]


def run_model(
    label: str,
    model_id: str,
    adapter: Path | None,
    sets: dict[str, list[dict]],
    max_new_tokens: int,
    limit: int,
    batch_size: int = 16,
) -> dict:
    import torch

    t0 = time.perf_counter()
    model, processor, tokenizer = load_stack(model_id, adapter)
    result = {"label": label, "model_id": model_id, "adapter": str(adapter) if adapter else None, "sets": {}}
    try:
        for set_name in SET_ORDER:
            rows = sets[set_name]
            if limit:
                rows = rows[:limit]
            # Batch by prompt length so padding is small; results go back in row order.
            order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]))
            hyps: list[str] = [""] * len(rows)
            done = 0
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                outs = generate_batch(model, processor, tokenizer, [rows[i]["input"] for i in idx], max_new_tokens)
                for i, hyp in zip(idx, outs, strict=True):
                    hyps[i] = hyp
                done += len(idx)
                if done % 100 < len(idx) or done == len(rows):
                    print(f"{label} {set_name} {done}/{len(rows)}", flush=True)
                    heartbeat()
            refs = [r["reference"] for r in rows]
            ins = [r["input"] for r in rows]
            worse_frac, worst = worse_examples(rows, hyps)
            result["sets"][set_name] = {
                "n": len(rows),
                "corrector": score_lists(refs, hyps),
                "baseline": score_lists(refs, ins),
                "worse_frac": worse_frac,
                "worst": worst,
                "predictions": [
                    {"id": r["id"], "input": r["input"], "output": h, "reference": r["reference"]}
                    for r, h in zip(rows, hyps, strict=True)
                ],
            }
    finally:
        del model
        torch.cuda.empty_cache()
    result["seconds"] = time.perf_counter() - t0
    return result


def discover_models(explicit: list[str]) -> list[tuple[str, str, Path | None]]:
    """Return (label, model_id, adapter_or_none)."""
    found: list[tuple[str, str, Path | None]] = []
    if explicit:
        for item in explicit:
            label, model_id, adapter = item.split(":", 2)
            found.append((label, model_id, Path(adapter) if adapter else None))
        return found
    mapping = {
        "qwen3-0.6b": QWEN_ID,
        "qwen3-0.6b-base": QWEN_ID,
        "gemma-4-e2b-it": GEMMA_ID,
        "gemma-4-e2b-it-base": GEMMA_ID,
        "qwen3-1.7b": QWEN_FALLBACK_ID,
        "qwen3-1.7b-base": QWEN_FALLBACK_ID,
    }
    adapters = DATA_ROOT / "adapters"
    if adapters.exists():
        for path in sorted(adapters.iterdir()):
            if not (path / "adapter_config.json").exists():
                continue
            name = path.name
            model_id = mapping.get(name)
            if model_id is None:
                meta = path / "train_meta.json"
                if meta.exists():
                    model_id = json.loads(meta.read_text()).get("model")
            if not model_id:
                continue
            found.append((f"{name}-lora", model_id, path))
            found.append((f"{name}-base", model_id, None))
    return found


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_markdown(payload: dict) -> str:
    lines = [
        "# Corrector v0 results",
        "",
        "Personal edit-pass LoRA on Wispr pairs plus TTS-augmented Parakeet pairs.",
        "Fair WER uses whisper_normalizer EnglishTextNormalizer then jiwer.",
        "Strict lowercase WER lowercases both sides and otherwise leaves digits/punctuation.",
        "",
        f"Train rows: {payload.get('train_rows')}. Dev rows: {payload.get('dev_rows')}.",
        "",
    ]
    asr = payload.get("asr_scores") or {}
    if asr:
        lines += ["## Parakeet v2 vs Wispr ASR (recognition vs formatting)", ""]
        lines += [
            "| split | n | parakeet fair WER | wispr asr fair WER | parakeet==target |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for split, block in (asr.get("splits") or {}).items():
            p = block["parakeet_vs_target"]
            w = block["wispr_asr_vs_target"]
            lines.append(
                f"| {split} | {block['n']} | {p['fair_wer']:.4f} | {w['fair_wer']:.4f} | "
                f"{block['parakeet_eq_target_fair']:.4f} |"
            )
        lines.append("")
    for set_name in SET_ORDER:
        lines += [f"## {set_name}", ""]
        lines += [
            "| system | n | fair WER | strict lc WER | worse than input |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        baseline_written = False
        for model in payload.get("models", []):
            block = (model.get("sets") or {}).get(set_name)
            if not block:
                continue
            if not baseline_written:
                b = block["baseline"]
                lines.append(
                    f"| raw input unchanged | {int(b['n'])} | {fmt(b['fair_wer'])} | "
                    f"{fmt(b['strict_lc_wer'])} | 0.0000 |"
                )
                baseline_written = True
            c = block["corrector"]
            lines.append(
                f"| {model['label']} | {int(c['n'])} | {fmt(c['fair_wer'])} | "
                f"{fmt(c['strict_lc_wer'])} | {block['worse_frac']:.4f} |"
            )
        lines.append("")
        # worst examples from first LoRA row if present
        lora_model = next(
            (m for m in payload.get("models", []) if "lora" in m.get("label", "") and set_name in (m.get("sets") or {})),
            None,
        )
        if lora_model:
            worst = lora_model["sets"][set_name].get("worst") or []
            lines.append(f"Ten worst {lora_model['label']} rows on {set_name} (largest fair-WER increase):")
            lines.append("")
            for i, ex in enumerate(worst, start=1):
                lines.append(
                    f"{i}. id={ex['id']} delta={ex['delta_fair_wer']:.3f} "
                    f"in={ex['input_fair_wer']:.3f} out={ex['output_fair_wer']:.3f}"
                )
                lines.append(f"   INPUT: {ex['input'][:400]}")
                lines.append(f"   OUTPUT: {ex['output'][:400]}")
                lines.append(f"   REFERENCE: {ex['reference'][:400]}")
                lines.append("")
    failures = payload.get("failures") or []
    lines += ["## Failures", ""]
    if not failures:
        lines.append("None recorded.")
    else:
        for item in failures:
            lines.append(f"- `{item.get('step')}` attempt {item.get('attempt')}: {item.get('error')}")
    lines.append("")
    notes = payload.get("notes") or []
    if notes:
        lines += ["## Notes", ""]
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def load_counts() -> tuple[int, int]:
    train = DATA_ROOT / "train.jsonl"
    dev = DATA_ROOT / "dev.jsonl"
    def count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    return count(train), count(dev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16, help="prompts per generate call; 1 = old per-clip path")
    parser.add_argument("--model", action="append", default=[], help="label:model_id:adapter_or_empty")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--out-json", type=Path, default=RESEARCH_ROOT / "results.json")
    parser.add_argument("--out-md", type=Path, default=RESEARCH_ROOT / "results.md")
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Keep already-evaluated labels from --out-json and only run new models.",
    )
    args = parser.parse_args()
    set_hf_env()
    sets = eval_rows()
    for name, rows in sets.items():
        empty = sum(1 for r in rows if not r["input"] or not r["reference"])
        print(f"eval set {name}: n={len(rows)} empty_in_or_ref={empty}", flush=True)
        if name in HOLDOUT_SPLITS and empty and name != "wispr_text_holdout":
            print(f"warning: {name} missing parakeet_raw on some rows", flush=True)

    asr_path = DATA_ROOT / "asr_scores.json"
    asr_scores = json.loads(asr_path.read_text()) if asr_path.exists() else {}
    train_rows, dev_rows = load_counts()
    models_out = []
    notes = []
    results_json = args.out_json
    seen_labels: set[str] = set()
    if args.reuse_results and results_json.exists():
        prev = json.loads(results_json.read_text())
        for model in prev.get("models") or []:
            label = model.get("label")
            if not label or label == "none":
                continue
            models_out.append(model)
            seen_labels.add(label)
        notes.extend(prev.get("notes") or [])
        print(f"reusing {len(seen_labels)} eval labels from {results_json}", flush=True)
    if not args.skip_generate:
        specs = discover_models(args.model)
        if not specs and not models_out:
            notes.append("No adapters found; reporting baseline only.")
        for label, model_id, adapter in specs:
            if label in seen_labels:
                print(f"skip existing eval {label}", flush=True)
                continue
            try:
                print(f"eval {label} {model_id} adapter={adapter}", flush=True)
                heartbeat()
                models_out.append(
                    run_model(label, model_id, adapter, sets, args.max_new_tokens, args.limit, args.batch_size)
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                record_failure(f"eval:{label}", err, 1)
                notes.append(f"eval {label} failed: {err}")
                print(err, flush=True)

    if not models_out:
        # baseline-only payload so results.md still has the required sets
        dummy = {"label": "none", "sets": {}}
        for name, rows in sets.items():
            use = rows[: args.limit] if args.limit else rows
            refs = [r["reference"] for r in use]
            ins = [r["input"] for r in use]
            dummy["sets"][name] = {
                "n": len(use),
                "corrector": score_lists(refs, ins),
                "baseline": score_lists(refs, ins),
                "worse_frac": 0.0,
                "worst": [],
            }
        models_out.append(dummy)

    failures = []
    fail_path = DATA_ROOT / "failures.jsonl"
    if fail_path.exists():
        failures = read_jsonl(fail_path)
    tts_n = sum(1 for _ in read_jsonl(TTS_PAIRS)) if TTS_PAIRS.exists() else 0
    payload = {
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "tts_pairs": tts_n,
        "asr_scores": asr_scores,
        "models": [
            {k: v for k, v in model.items() if k != "sets"}
            | {
                "sets": {
                    sname: {k: v for k, v in sblock.items() if k != "predictions"}
                    for sname, sblock in model.get("sets", {}).items()
                }
            }
            for model in models_out
        ],
        "failures": failures,
        "notes": notes,
    }
    pred_dir = DATA_ROOT / "eval_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for model in models_out:
        for sname, sblock in model.get("sets", {}).items():
            preds = sblock.get("predictions")
            if preds:
                (pred_dir / f"{model['label']}_{sname}.jsonl").write_text(
                    "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
                )
    results_md = args.out_md
    results_json.write_text(json.dumps(payload, indent=2) + "\n")
    results_md.write_text(build_markdown(payload) + "\n")
    print(f"wrote {results_md} and {results_json}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        record_failure("eval_corrector", f"{type(exc).__name__}: {exc}", 1)
        raise
