"""One run: load once, init once, retrieve on CPU while the GPU loads, generate every set in one
batched pass, then metrics, numerics and latency."""
from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import numerics as NUM
from . import retrieval as R
from .backends import make as make_backend
from .errors import set_english_words
from .fixtures import BENCH, SETS_DIR, load_manifest, load_set
from .metrics import (fair, false_inserts, hit_norm, read_jsonl, real_block, row_wer, term_block,
                      write_jsonl)
from .prompts import guard_cap, guard_post
from .report import render_md

DEFAULT_SETS = ["real_580", "term_old_seen", "term_old_unseen", "term_new", "term_pool2",
                "term_pool3"]
RETRIEVAL_SETS = ["real_580", "term_old_seen", "term_old_unseen", "term_new"]
NOISE = {"fair_wer": 0.002, "term_hit_norm": 0.01}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- generation
def run_set(backend, rows, cond: str, max_batch: int, hard_cap: int, label: str):
    """Length-sorted batched greedy with the per-row token cap and the word-count post-check."""
    key = {"retrieved": "vocab_retrieved", "oracle": "vocab_oracle", "none": None}[cond]
    jobs = [{"text": r["input"], "vocab": [] if key is None else (r.get(key) or [])}
            for r in rows]
    n = len(jobs)
    bs = min(n, max_batch)
    order = sorted(range(n), key=lambda i: len(jobs[i]["text"]) + 4 * len(jobs[i]["vocab"]))
    ntoks = backend.count_tokens([j["text"] for j in jobs])
    caps = [guard_cap(t, hard_cap) for t in ntoks]
    outs = [""] * n
    fired = [0] * n
    capped = [0] * n
    gen_tokens = 0
    t0 = time.perf_counter()
    for start in range(0, n, bs):
        idx = order[start:start + bs]
        prompts = [backend.build_prompt(jobs[i]["text"], jobs[i]["vocab"]) for i in idx]
        texts, cap_hit, nt = backend.generate_batch(prompts, [caps[i] for i in idx])
        gen_tokens += nt
        for i, h, ch in zip(idx, texts, cap_hit):
            txt, fb = guard_post(jobs[i]["text"], h)
            outs[i] = txt
            fired[i] = int(fb)
            capped[i] = int(ch)
        done = min(start + bs, n)
        log(f"  {label} {done}/{n} {done / max(time.perf_counter() - t0, 1e-9):.1f} rows/s "
            f"fired={sum(fired)} capped={sum(capped)}")
    dt = time.perf_counter() - t0
    return {"outs": outs, "fired": fired, "capped": capped, "seconds": dt,
            "rows_per_s": n / max(dt, 1e-9), "gen_tokens": gen_tokens,
            "tokens_per_s": gen_tokens / max(dt, 1e-9), "batch": bs,
            "mean_cap": sum(caps) / n}


# --------------------------------------------------------------------- retrieval
def retrieval_prepare(sets_dir: Path, lexicon: str, two_stage: bool, procs: int):
    """Lexicon, phoneme cache, index, retriever and the rerank worker pool.

    Called from the MAIN thread before the backend loads: the pool is forked, so it must not
    be created while another thread is initialising CUDA."""
    t0 = time.perf_counter()
    terms, rank, kind = R.load_lexicon(sets_dir / lexicon)
    ph = R.Phoneme(sets_dir / "g2p_cache.json")
    prior = json.loads((sets_dir / "kind_prior.json").read_text())
    t_index = time.perf_counter()
    ix = R.Index(terms, rank, kind, ph=ph, nreal=5)
    t_index = time.perf_counter() - t_index
    ret = R.RetrieverV2(ix, kind_prior=prior.get("kind_lo", {}), alpha=prior.get("alpha", 0.25),
                        stage1=(0.5, 0.5) if two_stage else None,
                        rerank=400 if two_stage else 0)
    by_squash: dict[str, list[str]] = {}
    for t in terms:
        s = R.squash(t)
        if len(s) >= 3:
            by_squash.setdefault(s, []).append(t)
    nprocs = R.auto_procs(procs)
    pool = R.start_pool(ret, nprocs)
    return {"ret": ret, "pool": pool, "by_squash": by_squash, "pool_terms": len(terms),
            "index_seconds": t_index, "lexicon": lexicon, "two_stage": two_stage,
            "procs": nprocs, "prepare_seconds": time.perf_counter() - t0}


def retrieval_stage(set_names, sets_dir: Path, ctx: dict, limit: int, workers: int):
    t0 = time.perf_counter()
    out = {k: ctx[k] for k in ("lexicon", "pool_terms", "index_seconds", "two_stage", "procs",
                               "prepare_seconds")}
    out["sets"] = {}
    ret, mp_pool, by_squash = ctx["ret"], ctx["pool"], ctx["by_squash"]
    for name in set_names:
        rows = load_set(name, sets_dir, limit=limit)
        if not rows:
            continue
        kind_of = "term" if name.startswith("term_") else "real"
        ts = time.perf_counter()
        tops = ret.topk([r["input"] for r in rows], k=100, workers=workers, pool=mp_pool)
        dt = time.perf_counter() - ts
        rep = R.recall_report(rows, tops, kind_of, by_squash=by_squash)
        rep.update({"rows": len(rows), "seconds": dt, "clips_per_s": len(rows) / max(dt, 1e-9)})
        out["sets"][name] = rep
        log(f"  retrieval {name}: r@10={rep['recall@10']:.4f} r@30={rep['recall@30']:.4f} "
            f"r@budget={rep['recall@budget']:.4f} len={rep['budget_mean_len']:.1f} "
            f"[{dt:.0f}s]")
    out["seconds"] = time.perf_counter() - t0
    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()
    return out


# --------------------------------------------------------------------- scoring
def score_set(name: str, rows, gen: dict, cond: str):
    refs = [r["reference"] for r in rows]
    raws = [r["input"] for r in rows]
    hyps = gen["outs"]
    key = {"retrieved": "vocab_retrieved", "oracle": "vocab_oracle",
           "none": "vocab_retrieved"}[cond]
    vocabs = [r.get(key) or [] for r in rows]
    raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, raws)]
    if name.startswith("term_"):
        terms = [r["term"] for r in rows]
        ret_has = [r["retrieved_has_term"] for r in rows]
        raw_unrel = sum(1 for v, h, rf, rw, tt in zip(vocabs, raws, refs, raws, terms)
                        if false_inserts(v, h, rf, rw, tt)[2]) / len(rows)
        blk = term_block(refs, raws, terms, hyps, vocabs, raw_rw, ret_has, raw_fi=raw_unrel)
        blk["raw_term_hit_norm"] = sum(hit_norm(t, h) for t, h in zip(terms, raws)) / len(rows)
    else:
        raw_unrel = sum(1 for v, h, rf in zip(vocabs, raws, refs)
                        if false_inserts(v, h, rf, h)[2]) / len(rows)
        blk = real_block(refs, raws, hyps, vocabs, raw_rw, raw_fi=raw_unrel)
    from .metrics import wer as _wer
    from .metrics import strict_lc
    blk["raw_fair_wer"] = _wer(refs, raws, fair)
    blk["raw_strict_lc_wer"] = _wer(refs, raws, strict_lc)
    blk["guard_fired"] = sum(gen["fired"])
    blk["guard_capped"] = sum(gen["capped"])
    blk["condition"] = cond
    blk["gen"] = {k: gen[k] for k in
                  ("seconds", "rows_per_s", "gen_tokens", "tokens_per_s", "batch", "mean_cap")}
    return blk


def cap_for(name, args):
    return args.real_max_new_tokens if name.startswith("real") else args.max_new_tokens


# --------------------------------------------------------------------- latency
def latency_stage(backend, rows, hard_cap: int, n: int = 20):
    import statistics as st

    sel = rows[:n]
    ttfts, tps, totals = [], [], []
    for r in sel:
        prompt = backend.build_prompt(r["input"], r.get("vocab_retrieved") or [])
        cap = guard_cap(backend.count_tokens([r["input"]])[0], hard_cap)
        _, ttft, total, ntok = backend.generate_timed(prompt, cap)
        ttfts.append(ttft)
        totals.append(total)
        if ntok:
            tps.append(ntok / max(total, 1e-9))
    fin = [x for x in ttfts if x == x]
    return {"n": len(sel),
            "ttft_ms_median": (1000 * st.median(fin)) if fin else None,
            "ttft_ms_p90": (1000 * sorted(fin)[int(0.9 * (len(fin) - 1))]) if fin else None,
            "row_ms_median": 1000 * st.median(totals) if totals else None,
            "tokens_per_s_batch1": st.median(tps) if tps else None}


# --------------------------------------------------------------------- main
def run(args) -> int:
    sets_dir = Path(args.sets_dir)
    set_english_words(sets_dir / "english_words.txt")
    man = load_manifest(sets_dir)
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    label = args.label or (Path(args.adapter).name if args.adapter else args.backend)
    out_dir = Path(args.runs_dir) / f"{stamp}_{label}_{args.backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_names = [s for s in (args.sets.split(",") if args.sets else DEFAULT_SETS) if s]

    res = {
        "label": label, "backend": args.backend, "condition": args.condition,
        "started_utc": started.isoformat(), "run_dir": str(out_dir),
        "host": platform.node(), "cmd": vars(args) | {"func": None},
        "fixtures": {k: {"rows": v["rows"], "sha256": v["sha256"][:16]}
                     for k, v in man["sets"].items() if k in set_names},
        "stages": {}, "sets": {}, "noise_floor": NOISE,
    }
    t_run = time.perf_counter()

    # ---- retrieval on CPU while the GPU loads -------------------------------
    pool = ThreadPoolExecutor(max_workers=1)
    fut = None
    if args.retrieval:
        rsets = [s for s in set_names if s in RETRIEVAL_SETS]
        log(f"retrieval stage (thread): {rsets} over {args.lexicon}")
        t_prep = time.perf_counter()
        rctx = retrieval_prepare(sets_dir, args.lexicon, args.two_stage,
                                 getattr(args, "retrieval_procs", 0))
        res["stages"]["retrieval_prepare"] = time.perf_counter() - t_prep
        log(f"retrieval prepare {res['stages']['retrieval_prepare']:.1f}s "
            f"procs={rctx['procs']}")
        fut = pool.submit(retrieval_stage, rsets, sets_dir, rctx, args.retrieval_limit,
                          args.retrieval_workers)

    # ---- load ---------------------------------------------------------------
    backend = make_backend(args.backend, args)
    t0 = time.perf_counter()
    info = backend.load()
    res["stages"]["load"] = time.perf_counter() - t0
    res["engine"] = info
    log(f"load {res['stages']['load']:.1f}s :: {info}")

    # ---- init / warmup ------------------------------------------------------
    t0 = time.perf_counter()
    backend.warmup()
    res["stages"]["init"] = time.perf_counter() - t0
    log(f"init {res['stages']['init']:.1f}s")

    # ---- generation ---------------------------------------------------------
    t0 = time.perf_counter()
    gens = {}
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for name in set_names:
        rows = load_set(name, sets_dir, limit=args.limit)
        if not rows:
            log(f"  set {name} empty; skipping")
            continue
        real = name.startswith("real")
        cap = args.real_max_new_tokens if real else args.max_new_tokens
        bs = (getattr(args, "real_max_batch", 0) or args.max_batch) if real \
            else args.max_batch
        gen = run_set(backend, rows, args.condition, bs, cap, name)
        gens[name] = (rows, gen)
        write_jsonl(preds_dir / f"{name}.jsonl",
                    [{"id": r["id"], "input": r["input"], "reference": r["reference"],
                      "term": r.get("term", ""), "output": o, "guard_fired": f, "capped": c}
                     for r, o, f, c in zip(rows, gen["outs"], gen["fired"], gen["capped"])])
    res["stages"]["generation"] = time.perf_counter() - t0
    log(f"generation {res['stages']['generation']:.1f}s over {len(gens)} sets")

    # ---- metrics ------------------------------------------------------------
    t0 = time.perf_counter()
    for name, (rows, gen) in gens.items():
        res["sets"][name] = score_set(name, rows, gen, args.condition)
    res["stages"]["metrics"] = time.perf_counter() - t0
    log(f"metrics {res['stages']['metrics']:.1f}s")

    # ---- numerics -----------------------------------------------------------
    if args.numerics:
        t0 = time.perf_counter()
        nset = getattr(args, "numerics_set", "numerics_500")
        nrows = load_set(nset, sets_dir, limit=args.numerics_limit)
        key = args.ref_key or NUM.ref_key(args.base, args.adapter, nset)
        ref_dir = Path(args.ref_dir) / key
        block = {"reference_key": key, "reference_dir": str(ref_dir), "set": nset}
        try:
            if args.build_reference or not (ref_dir / "teacher_forced.npz").exists():
                if not backend.supports_numerics:
                    raise RuntimeError(f"backend {args.backend} cannot build a reference")
                log(f"numerics: building bf16 reference at {ref_dir}")
                block["build"] = NUM.build_reference(backend, nrows, ref_dir, hidden_rows=(
                    args.hidden_rows if backend.supports_hidden else 0), log=log)
                g = run_set(backend, nrows, args.condition, args.max_batch,
                            cap_for(nset, args), f"{nset}/ref")
                NUM.save_reference_greedy(ref_dir, [r["id"] for r in nrows], g["outs"])
                block["is_reference"] = True
                block["greedy"] = {"word_agreement_vs_ref": 1.0, "exact_vs_ref": 1.0}
            else:
                if backend.supports_numerics:
                    block.update(NUM.compare(backend, nrows, ref_dir,
                                             hidden_rows=args.hidden_rows, log=log))
                else:
                    block["note"] = f"backend {args.backend} exposes no logprobs"
                g = run_set(backend, nrows, args.condition, args.max_batch,
                            cap_for(nset, args), nset)
                block["greedy"] = NUM.greedy_agreement(ref_dir, [r["id"] for r in nrows],
                                                       g["outs"])
        except Exception as exc:
            block["error"] = f"{type(exc).__name__}: {exc}"
            log(f"numerics FAILED: {block['error']}")
        res["numerics"] = block
        res["stages"]["numerics"] = time.perf_counter() - t0
        log(f"numerics {res['stages']['numerics']:.1f}s")

    # ---- latency ------------------------------------------------------------
    if args.latency:
        t0 = time.perf_counter()
        lrows = load_set(set_names[0], sets_dir, limit=0)
        try:
            res["latency"] = latency_stage(backend, lrows, args.real_max_new_tokens
                                           if set_names[0].startswith("real")
                                           else args.max_new_tokens, n=args.latency_rows)
        except Exception as exc:
            res["latency"] = {"error": f"{type(exc).__name__}: {exc}"}
        best = max((v["gen"]["tokens_per_s"] for v in res["sets"].values()), default=0.0)
        res["latency"]["tokens_per_s_set_batch"] = best
        res["latency"]["peak_vram_gb"] = backend.peak_vram_gb()
        res["stages"]["latency"] = time.perf_counter() - t0
        log(f"latency {res['stages']['latency']:.1f}s :: {res['latency']}")

    # ---- collect the retrieval thread --------------------------------------
    if fut is not None:
        t0 = time.perf_counter()
        try:
            res["retrieval"] = fut.result()
        except Exception as exc:
            res["retrieval"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"retrieval FAILED: {res['retrieval']['error']}")
        res["stages"]["retrieval"] = (res.get("retrieval", {}).get("seconds")
                                      or (time.perf_counter() - t0))
        res["stages"]["retrieval_blocked"] = time.perf_counter() - t0
    pool.shutdown(wait=False)

    backend.close()
    res["stages"]["wall"] = time.perf_counter() - t_run
    res["finished_utc"] = datetime.now(timezone.utc).isoformat()

    (out_dir / "run.json").write_text(json.dumps(res, indent=1) + "\n")
    (out_dir / "run.md").write_text(render_md(res))
    line = {"ts": stamp, "label": label, "backend": args.backend,
            "condition": args.condition, "run_dir": str(out_dir),
            "wall_s": res["stages"]["wall"],
            "stages": {k: round(v, 2) for k, v in res["stages"].items()},
            "sets": {k: {m: v.get(m) for m in
                         ("fair_wer", "strict_lc_wer", "term_hit_norm", "damage_rate",
                          "hit_term_retrieved", "false_insert_rate_unrelated", "guard_fired")
                         if v.get(m) is not None}
                     for k, v in res["sets"].items()}}
    for k, v in res["sets"].items():
        line["sets"][k]["ENTITY_per_1k"] = v["classes"]["ENTITY"]
    if "numerics" in res:
        line["numerics"] = {k: res["numerics"].get(k)
                            for k in ("kl_mean", "kl_p99", "top1_agreement")}
        line["numerics"]["greedy"] = res["numerics"].get("greedy")
    if "latency" in res:
        line["latency"] = res["latency"]
    bench = Path(args.bench_jsonl)
    bench.parent.mkdir(parents=True, exist_ok=True)
    with bench.open("a") as h:
        h.write(json.dumps(line) + "\n")

    print()
    print(render_md(res))
    log(f"wrote {out_dir}/run.json and run.md; appended {bench}")
    return 0
