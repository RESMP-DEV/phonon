"""STEP 3: refine heldout_new / heldold-unseen with v2 lists, with and without the
`ASR alternatives:` line. Length guard always on; the guard compares against the bare
1-best hypothesis, never against the prompt that carries the alternatives line."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for p in ("/home/user/phonon/research/corrector_v0", "/home/user/phonon/research/vocab_v0",
          "/home/user/phonon/research/vocab_v1", "/home/user/phonon/research/bigrun_v0",
          "/home/user/phonon/research/nbest_v0"):
    sys.path.insert(0, p)

from altline import line_for, load_nbest, with_alt_line  # noqa: E402
from refine_bigrun import generate_capped, guard_cap, guard_post  # noqa: E402
from vocab_common import read_jsonl  # noqa: E402

ADAPTERS = Path("/data/phonon_corrector_v0/adapters")
OUT = Path("/data/phonon_nbest_v0/refined")
LFM12 = "LiquidAI/LFM2.5-1.2B-Instruct"
SETS = {
    "new": (Path("/data/phonon_retrieval_v2/cond_new_v2.jsonl"),
            Path("/data/phonon_nbest_v0/nbest_heldout_new.jsonl")),
    "unseen": (Path("/data/phonon_retrieval_v2/cond_unseen_v2.jsonl"),
               Path("/data/phonon_nbest_v0/nbest_unseen.jsonl")),
}
REAL = (Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl"),
        Path("/data/phonon_nbest_v0/nbest_real.jsonl"))
CONDS = ("retrieved", "oracle")
VARIANTS = ("noalt", "alt")
STATS = Path("/data/phonon_nbest_v0/refine_stats.json")


def run(model, processor, tokenizer, jobs, bs, mnt, label, stats):
    from refine_vocab1 import apply_prompt_vocab
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["text"]) + 4 * len(jobs[i]["vocab"]))
    outs, fired, capped = [""] * len(jobs), [0] * len(jobs), [0] * len(jobs)
    ptoks = 0
    t0 = time.perf_counter()
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        prompts = [apply_prompt_vocab(processor, jobs[i]["text"], jobs[i]["vocab"]) for i in idx]
        caps = [guard_cap(tokenizer, jobs[i]["text"], mnt) for i in idx]
        ptoks += sum(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        res, cap_hit = generate_capped(model, tokenizer, prompts, caps)
        for i, h, ch in zip(idx, res, cap_hit):
            txt, fb = guard_post(jobs[i]["base"], h)
            outs[i], fired[i], capped[i] = txt, int(fb), int(ch)
        done = min(start + bs, len(order))
        if (start // bs) % 20 == 0 or done == len(order):
            el = time.perf_counter() - t0
            print(f"  {label} {done}/{len(order)} {done/max(el,1e-9):.1f} rows/s "
                  f"fired={sum(fired)} capped={sum(capped)}", flush=True)
    el = time.perf_counter() - t0
    stats[label] = {"rows": len(jobs), "wall_s": round(el, 1),
                    "rows_per_s": round(len(jobs) / max(el, 1e-9), 2),
                    "mean_prompt_tokens": round(ptoks / max(len(jobs), 1), 1),
                    "guard_fired": sum(fired), "capped": sum(capped)}
    return outs, fired, capped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[])
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--real-batch-size", type=int, default=16)
    ap.add_argument("--real-max-new-tokens", type=int, default=2048)
    ap.add_argument("--sets", default="new,unseen")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from eval_corrector import load_stack

    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads(STATS.read_text()) if STATS.exists() else {}
    want = [s for s in args.sets.split(",") if s]
    data = {}
    for k in want:
        if k == "real":
            continue
        cond_path, nb_path = SETS[k]
        rows = read_jsonl(cond_path)
        nb = load_nbest(nb_path)
        for r in rows:
            r["_alt"] = line_for(nb.get(r["id"]), r["hyp"])
        data[k] = rows
        print(f"{k}: rows={len(rows)} with_line={sum(1 for r in rows if r['_alt'])}", flush=True)
    real_rows = []
    if "real" in want:
        real_rows = read_jsonl(REAL[0])
        nb = load_nbest(REAL[1])
        for r in real_rows:
            r["_alt"] = line_for(nb.get(r["id"]), r["input"])
        print(f"real: rows={len(real_rows)} with_line={sum(1 for r in real_rows if r['_alt'])}",
              flush=True)

    for spec in args.adapter:
        parts = spec.split(":")
        label = parts[0]
        adapter = ADAPTERS / (parts[1] if len(parts) > 1 and parts[1] else label)
        if not (adapter / "adapter_config.json").exists():
            print(f"MISSING adapter {adapter}; skipping {label}", flush=True)
            continue
        todo = [k for k in data if args.force or not (OUT / f"{label}__{k}.jsonl").exists()]
        want_real = bool(real_rows) and (args.force or not (OUT / f"{label}__real.jsonl").exists())
        if not todo and not want_real:
            print(f"skip {label} (complete)", flush=True)
            continue
        print(f"=== {label} adapter={adapter}", flush=True)
        t0 = time.perf_counter()
        model, processor, tokenizer = load_stack(LFM12, adapter)
        try:
            for key in todo:
                rows = data[key]
                res = {}
                for cond in CONDS:
                    vkey = "vocab_oracle" if cond == "oracle" else "vocab_retrieved"
                    for var in VARIANTS:
                        jobs = [{"base": r["hyp"],
                                 "text": with_alt_line(r["hyp"], r["_alt"]) if var == "alt"
                                         else r["hyp"],
                                 "vocab": r[vkey]} for r in rows]
                        res[(cond, var)] = run(model, processor, tokenizer, jobs,
                                               args.batch_size, args.max_new_tokens,
                                               f"{label}/{key}/{cond}/{var}", stats)[0]
                        STATS.write_text(json.dumps(stats, indent=1) + "\n")
                tmp = (OUT / f"{label}__{key}.jsonl").with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as h:
                    for i, r in enumerate(rows):
                        rec = {"id": r["id"], "term": r["term"], "kind": r["kind"],
                               "voice": r["voice"], "reference": r["reference"], "hyp": r["hyp"],
                               "alt_line": r["_alt"],
                               "retrieved_has_term": r["retrieved_has_term"],
                               "vocab_oracle": r["vocab_oracle"],
                               "vocab_retrieved": r["vocab_retrieved"]}
                        for (cond, var), outs in res.items():
                            rec[f"out_{cond}_{var}"] = outs[i]
                        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.rename(OUT / f"{label}__{key}.jsonl")
                print(f"  wrote {label}__{key}.jsonl [{time.perf_counter()-t0:.0f}s]", flush=True)
            if want_real:
                res = {}
                for cond in CONDS[:1] + ("none",):
                    for var in VARIANTS:
                        jobs = [{"base": r["input"],
                                 "text": with_alt_line(r["input"], r["_alt"]) if var == "alt"
                                         else r["input"],
                                 "vocab": r["vocab_retrieved"] if cond == "retrieved" else []}
                                for r in real_rows]
                        res[(cond, var)] = run(model, processor, tokenizer, jobs,
                                               args.real_batch_size, args.real_max_new_tokens,
                                               f"{label}/real/{cond}/{var}", stats)[0]
                        STATS.write_text(json.dumps(stats, indent=1) + "\n")
                tmp = (OUT / f"{label}__real.jsonl").with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as h:
                    for i, r in enumerate(real_rows):
                        rec = {"id": r["id"], "input": r["input"], "reference": r["reference"],
                               "true_terms": r["true_terms"], "alt_line": r["_alt"],
                               "vocab_retrieved": r["vocab_retrieved"]}
                        for (cond, var), outs in res.items():
                            rec[f"out_{cond}_{var}"] = outs[i]
                        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.rename(OUT / f"{label}__real.jsonl")
                print(f"  wrote {label}__real.jsonl [{time.perf_counter()-t0:.0f}s]", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"{label} done wall_s={time.perf_counter()-t0:.0f}", flush=True)
    STATS.write_text(json.dumps(stats, indent=1) + "\n")
    print("REFINE NBEST DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
