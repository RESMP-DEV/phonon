"""Step 3: corrupt clean_v3 utterances with the v1-calibrated error model -> synth_pairs_v3_em."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

V1 = Path("/home/user/phonon/research/synth_v1")
sys.path.insert(0, str(V1))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corrupt import _load_v0, corrupt_text  # noqa: E402
from normalize import chat_messages, normalize_target  # noqa: E402
from paths import SYSTEM_PROMPT  # noqa: E402
from pollution import has_identifier_dump  # noqa: E402
from register import read_jsonl  # noqa: E402

get_error_model = _load_v0("errors").get_error_model

CLEAN = Path("/data/phonon_synth_v3/clean_v3.jsonl")
OUT = Path("/data/phonon_synth_v3/synth_pairs_v3_em.jsonl")
STATS = Path("/data/phonon_synth_v3/pairs_stats_v3.json")
REAL_WERS = Path("/data/phonon_synth_v1/tmp/real_wers.json")
LEXICON = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
TARGET_MEAN_WER = 0.0844


def load_real_wers() -> list[float]:
    if REAL_WERS.exists():
        wers = [float(x) for x in json.loads(REAL_WERS.read_text()).get("wers") or []]
        if wers:
            return wers
    return [0.0] * 38 + [0.05] * 12 + [0.084] * 12 + [0.21] * 8 + [0.35] * 2


def _common():
    spec = importlib.util.spec_from_file_location(
        "corrector_v0_common", "/home/user/phonon/research/corrector_v0/common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def subsample_wer(pairs: list[dict], n: int = 600) -> dict:
    if not pairs:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "n": 0}
    m = _common()
    step = max(len(pairs) // n, 1)
    wers = []
    for row in pairs[::step][:n]:
        try:
            wers.append(m.pair_wer(row["target"], row["input"], m.fair_norm))
        except Exception:
            continue
    ys = sorted(wers) or [0.0]
    k = (len(ys) - 1) * 0.90
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    p90 = ys[f] if f == c else ys[f] * (c - k) + ys[c] * (k - f)
    mid = ys[len(ys) // 2] if len(ys) % 2 else 0.5 * (ys[len(ys) // 2 - 1] + ys[len(ys) // 2])
    return {"mean": sum(ys) / len(ys), "median": mid, "p90": float(p90), "n": len(ys)}


def build(clean: list[dict], allowed: set[str], subs: dict, real_wers: list[float],
          seed: int, hit_scale: float) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    kept: list[dict] = []
    drop = Counter()
    for row in clean:
        target = normalize_target(row.get("text") or "")
        if not target:
            drop["empty_target"] += 1
            continue
        if has_identifier_dump(target):
            drop["identifier_dump"] += 1
            continue
        terms = [t for t in (row.get("terms") or []) if t in allowed]
        wer = rng.choice(real_wers)
        raw, applied = corrupt_text(target, rng, terms, subs, wer, hit_scale=hit_scale)
        if not raw.strip():
            drop["empty_raw"] += 1
            continue
        kept.append({
            "id": f"synth_v3_em_{row.get('id') or len(kept)}",
            "source": "synth_error_model",
            "input": raw,
            "target": target,
            "messages": chat_messages(raw, target, SYSTEM_PROMPT),
            "route": "error_model",
            "terms": terms,
            "error_classes_applied": applied,
            "topic_id": row.get("topic_id"),
            "register": row.get("register"),
            "target_wer_sampled": wer,
        })
        drop["kept"] += 1
    return kept, {"in": len(clean), "kept": len(kept), "drop_reason": dict(drop),
                  "hit_scale": hit_scale}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--hit-scale", type=float, default=1.0)
    ap.add_argument("--clean", type=Path, default=CLEAN)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clean = list(read_jsonl(args.clean))
    allowed = {r["term"] for r in read_jsonl(LEXICON)}
    model = get_error_model(prefer_sibling=True)
    subs = model.get("substitutions") or {}
    print(f"clean={len(clean)} error_model source={model.get('source')}", flush=True)
    real_wers = load_real_wers()

    hs = args.hit_scale
    pairs, stats = build(clean, allowed, subs, real_wers, args.seed, hs)
    probe = subsample_wer(pairs)
    print(f"pre-calibrate n={probe['n']} mean={probe['mean']:.4f} median={probe['median']:.4f} "
          f"p90={probe['p90']:.4f}", flush=True)
    stats["pre_calibrate"] = probe
    if probe["mean"] > 0.01 and abs(probe["mean"] - TARGET_MEAN_WER) > 0.012:
        hs = max(0.35, min(2.5, hs * (TARGET_MEAN_WER / probe["mean"])))
        print(f"re-corrupt with hit_scale={hs:.3f}", flush=True)
        pairs, stats = build(clean, allowed, subs, real_wers, args.seed, hs)
        probe2 = subsample_wer(pairs)
        print(f"post-calibrate n={probe2['n']} mean={probe2['mean']:.4f} "
              f"median={probe2['median']:.4f} p90={probe2['p90']:.4f}", flush=True)
        stats["pre_calibrate"] = probe
        stats["post_calibrate"] = probe2
        stats["hit_scale"] = hs

    seen = set()
    out_rows = []
    for r in pairs:
        key = (r["input"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        out_rows.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["n_pairs"] = len(out_rows)
    stats["out"] = str(args.out)
    cls = Counter()
    for r in out_rows:
        for c in r.get("error_classes_applied") or []:
            cls[c] += 1
    stats["error_classes"] = dict(cls.most_common())
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in ("n_pairs", "hit_scale", "error_classes")}, indent=1))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
