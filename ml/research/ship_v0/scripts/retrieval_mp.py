"""retrieval_mp: prove the multiprocess two-stage rerank is bit-identical and time it.

Three retrievers are compared on the same rows:
  old   - the pre-patch retrieval.py, restored from the backup taken before the edit
  new1  - the patched code with pool=None (the refactored in-process path)
  newN  - the patched code over a fork pool of N workers
Bit-identical means the (term, score) list of every row matches exactly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

SETS = Path("/data/phonon_bench_v0/sets")
OUT = Path("/data/phonon_ship_v0/retrieval_mp.json")
BENCH_SRC = Path("/home/user/phonon/research/bench_v0/src")
RESEARCH = Path("/home/user/phonon/research")
TIMED_SETS = ["real_580", "term_old_seen", "term_old_unseen", "term_new"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rows_of(name: str, limit: int = 0):
    out = []
    with (SETS / f"{name}.jsonl").open() as h:
        for line in h:
            if line.strip():
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def build(mod, procs: int):
    terms, rank, kind = mod.load_lexicon(SETS / "lexicon_big.jsonl")
    ph = mod.Phoneme(SETS / "g2p_cache.json")
    prior = json.loads((SETS / "kind_prior.json").read_text())
    ix = mod.Index(terms, rank, kind, ph=ph, nreal=5)
    ret = mod.RetrieverV2(ix, kind_prior=prior.get("kind_lo", {}),
                          alpha=prior.get("alpha", 0.25), stage1=(0.5, 0.5), rerank=400)
    pool = mod.start_pool(ret, procs) if procs > 1 and hasattr(mod, "start_pool") else None
    return ret, pool, len(terms)


def same(a, b) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for (ta, sa), (tb, sb) in zip(ra, rb):
            if ta != tb or sa != sb:
                return False
    return True


def main() -> int:
    res = {"host": os.uname().nodename, "cpu_count": os.cpu_count(), "identity": {},
           "timing": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------------------------------------------------------- identity, 1000 rows
    sample = rows_of("term_new", 600) + rows_of("real_580", 400)
    texts = [r["input"] for r in sample]
    res["identity"]["rows"] = len(texts)

    sys.path.insert(0, str(BENCH_SRC))
    bak = Path("/data/phonon_ship_v0/retrieval_old.py")
    shutil.copy("/data/phonon_ship_v0/retrieval.py.bak", bak)
    old_mod = load_module("retrieval_old", bak)
    ret_old, _, nterms = build(old_mod, 1)
    t = time.perf_counter()
    out_old = ret_old.topk(texts, k=100)
    t_old = time.perf_counter() - t
    del ret_old, old_mod
    sys.modules.pop("retrieval_old", None)

    new_mod = load_module("retrieval_new", BENCH_SRC / "phonon_bench" / "retrieval.py")
    ret1, _, _ = build(new_mod, 1)
    t = time.perf_counter()
    out_new1 = ret1.topk(texts, k=100)
    t_new1 = time.perf_counter() - t
    procs = new_mod.auto_procs(0)
    pool = new_mod.start_pool(ret1, procs)
    t = time.perf_counter()
    out_newN = ret1.topk(texts, k=100, pool=pool)
    t_newN = time.perf_counter() - t
    pool.close()
    pool.join()
    res["identity"]["bench"] = {
        "pool_terms": nterms, "procs": procs,
        "old_vs_new_inprocess": same(out_old, out_new1),
        "old_vs_new_multiprocess": same(out_old, out_newN),
        "seconds": {"old": round(t_old, 2), "new_1proc": round(t_new1, 2),
                    f"new_{procs}proc": round(t_newN, 2)},
    }
    print("identity(bench):", res["identity"]["bench"], flush=True)
    del ret1, out_old, out_new1, out_newN

    # ---------------------------------------------------------------- research copy
    sys.path.insert(0, str(RESEARCH / "retrieval_v2"))
    v2_bak = Path("/data/phonon_ship_v0/retrieve2_old.py")
    shutil.copy("/data/phonon_ship_v0/retrieve2.py.bak", v2_bak)
    kw = dict(nmax=8, w_char=0.35, w_meta=0.35, w_ph=0.30, stage1=(0.5, 0.5), rerank=400)
    prior = json.loads((SETS / "kind_prior.json").read_text())
    kp = {k: 0.25 * v for k, v in prior.get("kind_lo", {}).items()}

    def build_v2(mod):
        terms, rank, kind = new_mod.load_lexicon(SETS / "lexicon_big.jsonl")
        ph = mod.Phoneme(SETS / "g2p_cache.json")
        ix = mod.Index(terms, rank, kind, ph=ph, nreal=5)
        return mod.RetrieverV2(ix, kind_prior=kp, **kw)

    m_old = load_module("retrieve2_old", v2_bak)
    r_old = build_v2(m_old)
    o_old = r_old.topk(texts, k=100)
    del r_old, m_old
    m_new = load_module("retrieve2_new", RESEARCH / "retrieval_v2" / "retrieve2.py")
    r_new = build_v2(m_new)
    o_new1 = r_new.topk(texts, k=100)
    p2 = m_new.start_pool(r_new, procs)
    o_newN = r_new.topk(texts, k=100, pool=p2)
    p2.close()
    p2.join()
    res["identity"]["retrieve2"] = {
        "old_vs_new_inprocess": same(o_old, o_new1),
        "old_vs_new_multiprocess": same(o_old, o_newN),
    }
    print("identity(retrieve2):", res["identity"]["retrieve2"], flush=True)
    del r_new, o_old, o_new1, o_newN

    # ---------------------------------------------------------------- timing, the bench sets
    all_rows = {n: rows_of(n) for n in TIMED_SETS}
    res["timing"]["rows"] = {n: len(v) for n, v in all_rows.items()}
    ret, _, _ = build(new_mod, 1)
    for p in (1, 8, 16, 32, 48):
        if p > 1 and p > (os.cpu_count() or 8):
            continue
        mp_pool = new_mod.start_pool(ret, p) if p > 1 else None
        per_set = {}
        t0 = time.perf_counter()
        for n, rws in all_rows.items():
            ts = time.perf_counter()
            ret.topk([r["input"] for r in rws], k=100, pool=mp_pool)
            per_set[n] = round(time.perf_counter() - ts, 2)
        total = time.perf_counter() - t0
        if mp_pool is not None:
            mp_pool.close()
            mp_pool.join()
        res["timing"][f"procs_{p}"] = {"total_s": round(total, 2), "per_set": per_set}
        print(f"procs={p} total={total:.1f}s {per_set}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))

    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    ok = all(res["identity"][k][x] for k in ("bench", "retrieve2")
             for x in ("old_vs_new_inprocess", "old_vs_new_multiprocess"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
