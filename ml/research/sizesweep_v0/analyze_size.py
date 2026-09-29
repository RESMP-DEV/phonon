"""sizesweep_v0: metrics for every adapter evaluated into /data/phonon_sizesweep_v0/refined.

Metric definitions are imported unchanged from research/bigrun_v0/analyze_bigrun.py (which
imports them from research/vocab_v1/analyze_vocab1.py). Writes research/sizesweep_v0/results.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
from analyze_bigrun import (  # noqa: E402
    COND_FILE, REAL_FILE, CONDS, REAL_CONDS, real_block, term_block,
)
from analyze_vocab1 import fair, read_jsonl, row_wer  # noqa: E402

REF = Path("/data/phonon_sizesweep_v0/refined")
ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
OUT = Path("/home/user/phonon/research/sizesweep_v0")
LAT = Path("/data/phonon_sizesweep_v0/latency.json")

# label -> (pretty model, base id, params_b)
MODELS = {
    "bigrun_big_350m": ("LFM2.5-350M", "LiquidAI/LFM2.5-350M", 0.35),
    "size_qwen3-0.6b": ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 0.6),
    "bigrun_mid_r16": ("LFM2.5-1.2B", "LiquidAI/LFM2.5-1.2B-Instruct", 1.2),
    "size_qwen3-1.7b": ("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 1.7),
    "size_gemma4-e2b": ("gemma-4-E2B-it", "google/gemma-4-E2B-it", 2.0),
    "size_lfm350_mid": ("LFM2.5-350M (train_mid)", "LiquidAI/LFM2.5-350M", 0.35),
}
ORDER = ["raw", "bigrun_big_350m", "size_lfm350_mid", "size_qwen3-0.6b", "bigrun_mid_r16",
         "size_qwen3-1.7b", "size_gemma4-e2b"]


def dir_mb(p: Path) -> float:
    try:
        out = subprocess.run(["du", "-sm", str(p)], capture_output=True, text=True).stdout
        return float(out.split()[0])
    except Exception:
        return 0.0


def adapter_mb(p: Path) -> float:
    tot = 0
    for f in ("adapter_model.safetensors", "adapter_model.bin"):
        q = p / f
        if q.exists():
            tot += q.stat().st_size
    return round(tot / 1e6, 1)


def main() -> int:
    res: dict = {"terms": {}, "real": {}, "models": {}}
    base = {k: read_jsonl(p) for k, p in COND_FILE.items() if p.exists()}

    prep = {}
    for k, rows in base.items():
        refs = [r["reference"] for r in rows]
        raws = [r["hyp"] for r in rows]
        terms = [r["term"] for r in rows]
        ret_has = [r["retrieved_has_term"] for r in rows]
        vr = [r["vocab_retrieved"] for r in rows]
        vo = [r["vocab_oracle"] for r in rows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
        prep[k] = (rows, refs, raws, terms, ret_has, vr, vo, raw_rw)
        res["terms"].setdefault("raw", {})[k] = {
            "none": term_block(refs, raws, terms, raws, vr, raw_rw, ret_has, classes=True)}

    labels = sorted({p.name.split("__")[0] for p in REF.glob("*__*.jsonl")})
    for lab in labels:
        for k, (rows, refs, raws, terms, ret_has, vr, vo, raw_rw) in prep.items():
            p = REF / f"{lab}__{k}.jsonl"
            if not p.exists():
                continue
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["terms"].setdefault(lab, {})[k] = {}
            for cond in CONDS:
                hyps = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rows]
                vocabs = vo if cond == "oracle" else vr
                blk = term_block(refs, raws, terms, hyps, vocabs, raw_rw, ret_has,
                                 classes=(cond == "retrieved"))
                blk["guard_fired"] = sum(by_id.get(r["id"], {}).get(f"fired_{cond}", 0)
                                         for r in rows)
                blk["guard_capped"] = sum(by_id.get(r["id"], {}).get(f"capped_{cond}", 0)
                                          for r in rows)
                res["terms"][lab][k][cond] = blk

    if REAL_FILE.exists():
        rrows = read_jsonl(REAL_FILE)
        refs = [r["reference"] for r in rrows]
        ins = [r["input"] for r in rrows]
        vocs = [r["vocab_retrieved"] for r in rrows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, ins)]
        res["real"]["raw"] = {"none": real_block(refs, ins, ins, vocs, raw_rw)}
        res["real"]["raw"]["none"].update({"win": 0, "tie": len(rrows), "loss": 0,
                                           "damage_rate": 0.0})
        for lab in labels:
            p = REF / f"{lab}__real.jsonl"
            if not p.exists():
                continue
            by_id = {r["id"]: r for r in read_jsonl(p)}
            res["real"][lab] = {}
            for cond in REAL_CONDS:
                hy = [by_id.get(r["id"], {}).get(f"out_{cond}", "") for r in rrows]
                blk = real_block(refs, ins, hy, vocs, raw_rw)
                blk["guard_fired"] = sum(by_id.get(r["id"], {}).get(f"fired_{cond}", 0)
                                         for r in rrows)
                blk["guard_capped"] = sum(by_id.get(r["id"], {}).get(f"capped_{cond}", 0)
                                          for r in rrows)
                res["real"][lab][cond] = blk

    for lab in labels:
        d = ADAPTERS / lab
        meta = {}
        mp = d / "train_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        pretty, base_id, params = MODELS.get(lab, (lab, meta.get("model", "?"), None))
        rows = meta.get("train_rows", 0)
        secs = meta.get("seconds", 0.0)
        ep = meta.get("epochs", 0)
        res["models"][lab] = {
            "pretty": pretty, "base": meta.get("model", base_id), "params_b": params,
            "train_rows": rows, "epochs": ep, "train_s": secs,
            "train_min": round(secs / 60.0, 1) if secs else None,
            "rows_per_s": round(rows * ep / secs, 1) if secs and rows and ep else None,
            "batch_size": meta.get("batch_size"), "grad_accum": meta.get("grad_accum"),
            "grad_checkpointing": meta.get("grad_checkpointing"),
            "lora_r": meta.get("lora_r"), "lora_dropout": meta.get("lora_dropout"),
            "peak_vram_gb": round(meta.get("peak_vram_bytes", 0) / 1e9, 1) or None,
            "median_step_s": meta.get("median_step_s"),
            "train_tokens_per_s": meta.get("real_tokens_per_s"),
            "adapter_mb": adapter_mb(d), "adapter_dir_mb": dir_mb(d),
        }
    # ---- retrieval v2 lists (heldout_new / heldout_old unseen, retrieved only) ----
    V2 = Path("/data/phonon_retrieval_v2")
    v2cond = {"new": V2 / "cond_new_v2.jsonl", "unseen": V2 / "cond_unseen_v2.jsonl"}
    res["terms_v2"] = {}
    for k, cp in v2cond.items():
        if not cp.exists():
            continue
        rows = read_jsonl(cp)
        refs = [r["reference"] for r in rows]
        raws = [r["hyp"] for r in rows]
        terms_ = [r["term"] for r in rows]
        ret_has = [r["retrieved_has_term"] for r in rows]
        vr = [r["vocab_retrieved"] for r in rows]
        raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
        res["terms_v2"].setdefault("recall", {})[k] = sum(1 for x in ret_has if x) / len(rows)
        for lab in sorted({p.name.split("__")[0] for p in (V2 / "refined").glob("*__*.jsonl")}):
            fp = V2 / "refined" / f"{lab}__{k}.jsonl"
            if not fp.exists():
                continue
            by_id = {r["id"]: r for r in read_jsonl(fp)}
            hyps = [by_id.get(r["id"], {}).get("out_retrieved", "") for r in rows]
            res["terms_v2"].setdefault(lab, {})[k] = term_block(
                refs, raws, terms_, hyps, vr, raw_rw, ret_has)

    if LAT.exists():
        res["latency"] = json.loads(LAT.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"labels": labels,
                      "models": {k: v["pretty"] for k, v in res["models"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
