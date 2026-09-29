"""Teacher-forced numerics against a cached bf16 reference.

The reference run (backend `hf`, bf16, same base+adapter) stores, per numerics_500 row:
  * the top-K log-probabilities and ids at every target position (teacher forced on the
    reference text), K = 64 by default;
  * the greedy output under the same guard as the generation stage;
  * the last-prompt-position hidden state of every layer, for the first --hidden-rows rows.
It lands in /data/phonon_bench_v0/ref/<key>/ and later runs pay nothing.

A candidate run reports its own log-probabilities at the reference's top-K ids, so no run ever
has to materialise a [tokens, 65536] tensor.  KL is over that truncated support, renormalised:
  KL = sum_k p_ref(k) * (log p_ref(k) - log p_cand(k)),  p renormalised over the K ids.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .metrics import fair, wer

TOPK = 64


def ref_key(base: str, adapter: str | None, set_name: str) -> str:
    a = Path(adapter).name if adapter else "noadapter"
    b = str(base).rstrip("/").replace("/", "_")
    return f"{b}__{a}__{set_name}"


def _flatten(per_row):
    offs = np.cumsum([0] + [r["top_ids"].shape[0] for r in per_row]).astype("int64")
    return {
        "offsets": offs,
        "top_ids": np.concatenate([r["top_ids"] for r in per_row]).astype("int32"),
        "top_logprobs": np.concatenate([r["top_logprobs"] for r in per_row]).astype("float16"),
        "top1": np.concatenate([r["top1"] for r in per_row]).astype("int32"),
        "target_ids": np.concatenate([r["target_ids"] for r in per_row]).astype("int32"),
        "target_logprob": np.concatenate(
            [r["target_logprob"] for r in per_row]).astype("float16"),
    }


def build_reference(backend, rows, out_dir: Path, topk: int = TOPK, hidden_rows: int = 100,
                    log=print) -> dict:
    """Teacher-forced top-K + hidden states for the reference backend. Returns the cache dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row, ids, skipped = [], [], []
    hidden = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        prompt = backend.build_prompt(r["input"], r.get("vocab_retrieved") or [])
        tf = backend.teacher_forced(prompt, r["reference"], topk)
        if tf is None:
            skipped.append(r["id"])
            continue
        per_row.append(tf)
        ids.append(r["id"])
        if i < hidden_rows and backend.supports_hidden:
            h = backend.hidden_last_prompt(prompt)
            if h is not None:
                hidden.append(h.astype("float16"))
        if log and (i + 1) % 100 == 0:
            log(f"  numerics reference {i + 1}/{len(rows)} [{time.perf_counter() - t0:.0f}s]")
    flat = _flatten(per_row)
    np.savez_compressed(out_dir / "teacher_forced.npz", **flat,
                        hidden=np.stack(hidden) if hidden else np.zeros((0,), dtype="float16"))
    (out_dir / "ids.json").write_text(json.dumps({"ids": ids, "skipped": skipped,
                                                  "topk": topk, "hidden_rows": len(hidden)}))
    return {"n_rows": len(ids), "n_tokens": int(flat["offsets"][-1]), "skipped": len(skipped),
            "hidden_rows": len(hidden), "seconds": time.perf_counter() - t0}


def save_reference_greedy(out_dir: Path, ids, outputs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "greedy.json").write_text(json.dumps(dict(zip(ids, outputs))))


def load_reference(out_dir: Path):
    z = np.load(out_dir / "teacher_forced.npz")
    meta = json.loads((out_dir / "ids.json").read_text())
    greedy = {}
    p = out_dir / "greedy.json"
    if p.exists():
        greedy = json.loads(p.read_text())
    return z, meta, greedy


def compare(backend, rows, ref_dir: Path, hidden_rows: int = 100, log=print) -> dict:
    """Candidate against the cached reference. Returns the numerics block."""
    z, meta, greedy = load_reference(ref_dir)
    offs = z["offsets"]
    ref_ids = z["top_ids"]
    ref_lp = z["top_logprobs"].astype("float32")
    ref_top1 = z["top1"]
    ref_hidden = z["hidden"]
    by_id = {r["id"]: r for r in rows}
    kls, agree, tok_n = [], 0, 0
    per_row_kl = []
    missing = 0
    t0 = time.perf_counter()
    hid_max, hid_rel = [], []
    for i, rid in enumerate(meta["ids"]):
        r = by_id.get(rid)
        if r is None:
            missing += 1
            continue
        a, b = int(offs[i]), int(offs[i + 1])
        prompt = backend.build_prompt(r["input"], r.get("vocab_retrieved") or [])
        got = backend.logprobs_at(prompt, r["reference"], ref_ids[a:b])
        if got is None:
            missing += 1
            continue
        cand_lp, cand_top1 = got
        t = cand_lp.shape[0]
        rl = ref_lp[a:a + t]
        # renormalise both over the reference's top-K support
        pr = np.exp(rl - rl.max(axis=-1, keepdims=True))
        pr /= pr.sum(axis=-1, keepdims=True)
        lr = np.log(np.maximum(pr, 1e-12))
        cl = cand_lp - cand_lp.max(axis=-1, keepdims=True)
        cl = cl - np.log(np.maximum(np.exp(cl).sum(axis=-1, keepdims=True), 1e-12))
        kl = (pr * (lr - cl)).sum(axis=-1)
        kls.append(kl)
        per_row_kl.append(float(kl.mean()))
        agree += int((cand_top1 == ref_top1[a:a + t]).sum())
        tok_n += t
        if i < hidden_rows and i < ref_hidden.shape[0] and backend.supports_hidden:
            h = backend.hidden_last_prompt(prompt)
            if h is not None:
                hr = ref_hidden[i].astype("float32")
                n = min(h.shape[0], hr.shape[0])
                d = np.abs(h[:n] - hr[:n])
                hid_max.append(d.max(axis=-1))
                hid_rel.append((d / (np.abs(hr[:n]) + 1e-6)).mean(axis=-1))
        if log and (i + 1) % 100 == 0:
            log(f"  numerics compare {i + 1}/{len(meta['ids'])} "
                f"[{time.perf_counter() - t0:.0f}s]")
    out = {"rows_scored": len(per_row_kl), "rows_missing": missing, "tokens": tok_n,
           "topk": meta["topk"], "seconds": time.perf_counter() - t0}
    if kls:
        allk = np.concatenate(kls)
        out.update({"kl_mean": float(allk.mean()), "kl_p99": float(np.percentile(allk, 99)),
                    "kl_max": float(allk.max()),
                    "kl_row_mean_p99": float(np.percentile(per_row_kl, 99)),
                    "top1_agreement": agree / max(1, tok_n)})
    if hid_max:
        hm = np.stack(hid_max).mean(axis=0)
        hr = np.stack(hid_rel).mean(axis=0)
        out["hidden_layers"] = int(hm.shape[0])
        out["hidden_max_abs_by_layer"] = [float(x) for x in hm]
        out["hidden_rel_by_layer"] = [float(x) for x in hr]
        out["hidden_max_abs"] = float(hm.max())
        out["hidden_rel_mean"] = float(hr.mean())
    return out


def greedy_agreement(ref_dir: Path, ids, outputs) -> dict:
    """Word agreement and exact match of this run's greedy outputs against the reference's."""
    _, _, greedy = load_reference(ref_dir)
    if not greedy:
        return {"note": "no cached reference greedy outputs"}
    pairs = [(greedy[i], o) for i, o in zip(ids, outputs) if i in greedy]
    if not pairs:
        return {"note": "no overlapping ids"}
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    return {"n": len(pairs),
            "word_agreement_vs_ref": 1.0 - wer(a, b, fair),
            "exact_vs_ref": sum(1 for x, y in pairs if fair(x) == fair(y)) / len(pairs)}
