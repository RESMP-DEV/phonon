"""Filter lexicon/utterances, recorrupt at real rates, keep TTS, write synth_pairs_v1."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrupt import corrupt_text  # noqa: E402
from normalize import chat_messages, normalize_target  # noqa: E402
from paths import (  # noqa: E402
    CLEAN_PATH,
    DATA_ROOT,
    FILTER_STATS,
    LEXICON_PATH,
    LOG_DIR,
    PAIRS_PATH,
    REAL_WERS,
    SYSTEM_PROMPT,
    V0_RANKED,
    V0_TTS,
)
from pollution import classify_term, has_identifier_dump  # noqa: E402
from corrupt import _load_v0  # noqa: E402

get_error_model = _load_v0("errors").get_error_model


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_lexicon() -> tuple[list[dict], dict[str, str], dict]:
    rows = []
    status: dict[str, str] = {}
    class_n = Counter()
    kind_keep = Counter()
    examples = {}
    watch = {
        "PersimmonModel",
        "Blip2VisionConfig",
        "FlightGear",
        "KOOPA",
        "GatherGetCollByteCount",
        "Grok",
        "KernelBench",
        "Codex",
        "Fable",
        "Gpubox",
    }
    for row in iter_jsonl(V0_RANKED):
        cls = classify_term(row)
        status[row["term"]] = cls
        class_n[cls] += 1
        if cls in {"keep", "weak"}:
            rows.append(row)
            if cls == "keep":
                kind_keep[row.get("kind") or "?"] += 1
        if row["term"] in watch:
            examples[row["term"]] = {
                "class": cls,
                "kind": row.get("kind"),
                "count": row.get("count"),
                "source_file": row.get("source_file"),
                "files": (row.get("files") or [])[:5],
            }
    rows.sort(key=lambda r: (-float(r.get("rank_score") or 0), -int(r.get("count") or 0)))
    write_jsonl(LEXICON_PATH, rows)
    stats = {
        "ranked_in": sum(class_n.values()),
        "keep": class_n["keep"],
        "weak": class_n["weak"],
        "polluted": class_n["polluted"],
        "lexicon_out": len(rows),
        "keep_kinds": dict(kind_keep),
        "examples": examples,
        "lexicon_path": str(LEXICON_PATH),
    }
    return rows, status, stats


def term_state(terms: list[str], status: dict[str, str]) -> str:
    if not terms:
        return "none"
    states = [status.get(t, "polluted") for t in terms]
    if all(s == "polluted" for s in states):
        return "all_polluted"
    if all(s in {"keep", "weak"} for s in states):
        return "all_clean"
    return "mixed"


def keep_utterance(text: str, terms: list[str], status: dict[str, str]) -> tuple[bool, str]:
    target = normalize_target(text)
    if not target:
        return False, "empty_target"
    if has_identifier_dump(target):
        return False, "identifier_dump"
    state = term_state(terms, status)
    if state == "all_polluted":
        return False, "all_polluted"
    return True, state


def load_real_wers() -> list[float]:
    if REAL_WERS.exists():
        blob = json.loads(REAL_WERS.read_text())
        wers = [float(x) for x in blob.get("wers") or []]
        if wers:
            return wers
    # fallback: zeros and a few small values matching the known mix
    return [0.0] * 38 + [0.05] * 12 + [0.084] * 12 + [0.21] * 8 + [0.35] * 2


def subsample_wer(pairs: list[dict], n: int = 500) -> dict[str, float]:
    if not pairs:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "n": 0}
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "corrector_v0_common",
        Path("/home/user/phonon/research/corrector_v0/common.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("corrector_v0 common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fair_norm = mod.fair_norm
    pair_wer = mod.pair_wer

    step = max(len(pairs) // n, 1)
    sample = pairs[::step][:n]
    wers = []
    for row in sample:
        try:
            wers.append(pair_wer(row["target"], row["input"], fair_norm))
        except Exception:
            continue
    if not wers:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "n": 0}
    ys = sorted(wers)
    k = (len(ys) - 1) * 0.90
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    p90 = ys[f] if f == c else ys[f] * (c - k) + ys[c] * (k - f)
    mid = ys[len(ys) // 2] if len(ys) % 2 else 0.5 * (ys[len(ys) // 2 - 1] + ys[len(ys) // 2])
    return {"mean": sum(ys) / len(ys), "median": mid, "p90": float(p90), "n": len(ys)}


def build_error_model_pairs(
    clean: list[dict],
    status: dict[str, str],
    substitutions: dict,
    real_wers: list[float],
    rng: random.Random,
    hit_scale: float,
) -> tuple[list[dict], dict]:
    kept = []
    drop_reason = Counter()
    for row in clean:
        text = row.get("text") or ""
        terms = list(row.get("terms") or [])
        ok, reason = keep_utterance(text, terms, status)
        if not ok:
            drop_reason[reason] += 1
            continue
        target = normalize_target(text)
        clean_terms = [t for t in terms if status.get(t) in {"keep", "weak"}]
        target_wer = rng.choice(real_wers)
        raw, applied = corrupt_text(
            target, rng, clean_terms, substitutions, target_wer, hit_scale=hit_scale
        )
        if not raw.strip():
            drop_reason["empty_raw"] += 1
            continue
        uid = row.get("id") or f"{len(kept):05d}"
        kept.append(
            {
                "id": f"synth_v1_em_{uid}",
                "source": "synth_error_model",
                "input": raw,
                "target": target,
                "messages": chat_messages(raw, target, SYSTEM_PROMPT),
                "route": "error_model",
                "terms": clean_terms,
                "error_classes_applied": applied,
                "prompt_id": row.get("prompt_id"),
                "setting": row.get("setting"),
                "target_wer_sampled": target_wer,
                "term_filter": reason,
            }
        )
        drop_reason["kept"] += 1
    stats = {
        "in": len(clean),
        "kept": len(kept),
        "drop_reason": dict(drop_reason),
        "hit_scale": hit_scale,
    }
    return kept, stats


def build_tts_pairs(status: dict[str, str]) -> tuple[list[dict], dict]:
    kept = []
    drop_reason = Counter()
    n_in = 0
    for row in iter_jsonl(V0_TTS):
        n_in += 1
        target = normalize_target(row.get("target") or "")
        terms = list(row.get("terms") or [])
        ok, reason = keep_utterance(target, terms, status)
        if not ok:
            drop_reason[reason] += 1
            continue
        raw = (row.get("input") or "").strip()
        if not raw or not target:
            drop_reason["empty"] += 1
            continue
        clean_terms = [t for t in terms if status.get(t) in {"keep", "weak"}]
        out = dict(row)
        out["id"] = str(row.get("id") or "").replace("synth_tts_", "synth_v1_tts_", 1)
        if not str(out["id"]).startswith("synth_v1_"):
            out["id"] = f"synth_v1_tts_{out['id']}"
        out["target"] = target
        out["input"] = raw
        out["messages"] = chat_messages(raw, target, SYSTEM_PROMPT)
        out["route"] = "tts_asr"
        out["terms"] = clean_terms
        out["term_filter"] = reason
        kept.append(out)
        drop_reason["kept"] += 1
    stats = {"in": n_in, "kept": len(kept), "drop_reason": dict(drop_reason)}
    return kept, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--hit-scale", type=float, default=1.0)
    parser.add_argument("--calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", action="store_false", dest="calibrate")
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    lex_rows, status, lex_stats = filter_lexicon()
    print(
        f"lexicon keep={lex_stats['keep']} weak={lex_stats['weak']} "
        f"polluted={lex_stats['polluted']} out={lex_stats['lexicon_out']}",
        flush=True,
    )
    print("examples", {k: v["class"] for k, v in lex_stats["examples"].items()}, flush=True)

    model = get_error_model(prefer_sibling=True)
    substitutions = model.get("substitutions") or {}
    print(f"error_model source={model.get('source')} rates={model.get('rates')}", flush=True)

    real_wers = load_real_wers()
    clean = load_jsonl(CLEAN_PATH)
    rng = random.Random(args.seed)
    hit_scale = args.hit_scale
    em, em_stats = build_error_model_pairs(clean, status, substitutions, real_wers, rng, hit_scale)

    if args.calibrate and em:
        probe = subsample_wer(em, 600)
        print(f"pre-calibrate subsample n={probe['n']} mean={probe['mean']:.4f} median={probe['median']:.4f} p90={probe['p90']:.4f}", flush=True)
        target_mean = 0.0844
        if probe["mean"] > 0.01 and abs(probe["mean"] - target_mean) > 0.012:
            hit_scale = hit_scale * (target_mean / probe["mean"])
            hit_scale = max(0.35, min(2.5, hit_scale))
            print(f"re-corrupt with hit_scale={hit_scale:.3f}", flush=True)
            rng = random.Random(args.seed)
            em, em_stats = build_error_model_pairs(
                clean, status, substitutions, real_wers, rng, hit_scale
            )
            probe2 = subsample_wer(em, 600)
            print(
                f"post-calibrate subsample n={probe2['n']} mean={probe2['mean']:.4f} "
                f"median={probe2['median']:.4f} p90={probe2['p90']:.4f}",
                flush=True,
            )
            em_stats["pre_calibrate"] = probe
            em_stats["post_calibrate"] = probe2
            em_stats["hit_scale"] = hit_scale
        else:
            em_stats["pre_calibrate"] = probe

    tts, tts_stats = build_tts_pairs(status)
    print(f"em kept={len(em)} tts kept={len(tts)} total={len(em)+len(tts)}", flush=True)

    seen: set[tuple[str, str]] = set()
    pairs = []
    for row in em + tts:
        key = (row["input"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(row)
    write_jsonl(PAIRS_PATH, pairs)

    stats = {
        "lexicon": lex_stats,
        "error_model": em_stats,
        "tts": tts_stats,
        "n_pairs": len(pairs),
        "n_em": sum(1 for r in pairs if r.get("route") == "error_model"),
        "n_tts": sum(1 for r in pairs if r.get("route") == "tts_asr"),
        "pairs_path": str(PAIRS_PATH),
        "seed": args.seed,
        "hit_scale": hit_scale,
        "filler_weights": {"like": 9743, "um": 736, "uh": 430, "you know": 465},
    }
    FILTER_STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in ("n_pairs", "n_em", "n_tts", "hit_scale")}, indent=2))
    print(f"wrote {PAIRS_PATH} and {FILTER_STATS}", flush=True)
    if len(pairs) < 6000:
        print(f"WARNING: only {len(pairs)} pairs survive; LLM regen threshold is 6000", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
