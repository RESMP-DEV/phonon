"""STEP 4: what a per-user index buys - full union pool vs the term's own repo / source slice.

Both conditions use the same scorer: 5 spoken realisations, n-grams to 8 words,
0.5 char / 0.5 metaphone, kind log-odds prior (no g2p phonemes: the per-user pools bring
hundreds of thousands of uncached words and g2p warming them is a 25-minute tax that would
not change the A/B).  Budget rule and threshold from STEP 2.
"""
from __future__ import annotations
import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/phonon/research/retrieval_v2")
from retrieve2 import Index  # noqa: E402
from common import read_jsonl, pool_union, load_pool2_rows, ASR_NEW  # noqa: E402
import variants as V  # noqa: E402

D = Path("/data/phonon_index_v0")
PR = json.loads(Path("/data/phonon_retrieval_v2/prior.json").read_text())
KLO, ALPHA = PR["kind_lo"], PR["alpha"]
THR, MULT, CAP = 65, 3, 80
PER_USER_CAP = 0  # 0 = the whole repo slice
log = lambda s: print(s, flush=True)
T0 = time.perf_counter()

KIND_MAP = {"module": "module", "file": "file", "flag": "flag"}


def kind_of(term, hint):
    k = KIND_MAP.get(hint, "")
    if k:
        return k
    if term.startswith("--"):
        return "flag"
    if re.fullmatch(r"[A-Z]{2,8}", term):
        return "acronym"
    if "/" in term or term.endswith((".py", ".rs", ".cu", ".go", ".ts", ".md", ".h", ".cpp")):
        return "file"
    return "identifier"


def budget(nw):
    return min(CAP, max(30, MULT * nw))


def score_rows(terms, rank, kind, texts, workers=-1):
    ix = Index(terms, rank, kind, use_phonemes=False, nreal=5)
    block = max(64, min(2048, int(2.5e8 / max(1, len(ix.reals)))))
    best, _ = V.run_cfgs(ix, texts, [("x", 0.5, 0.5, 0.0)], nmax=8, workers=workers, block=block)
    B = best["x"]
    prior = np.array([ALPHA * KLO.get(k, 0.0) for k in ix.kind], dtype=np.float32)
    bonus = 1e-4 * np.log1p(ix.rank)
    out = []
    for ti in range(len(texts)):
        row = B[ti].astype(np.float32) + bonus + prior
        kk = min(100, len(terms))
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.argsort(-row[idx], kind="stable")]
        out.append([(terms[int(j)], float(B[ti][j])) for j in idx])
    return out


def stats(lists, nwords, golds, pool_sizes):
    agg = {"pairs": 0, "r10": 0, "r30": 0, "rb": 0, "len": [], "rank": [], "covered": 0,
           "cov_pairs": 0, "rb_cov": 0, "pool": []}
    for lst, nw, g, ps in zip(lists, nwords, golds, pool_sizes):
        b = budget(nw)
        terms = [t for t, s in lst]
        keep = [t for t, s in lst[:b] if s >= THR]
        agg["len"].append(len(keep))
        agg["pool"].append(ps)
        pos = {t: i for i, t in enumerate(terms)}
        inpool = set(terms[:0])
        for t in g:
            agg["pairs"] += 1
            r = pos.get(t)
            if r is not None:
                if r < 10:
                    agg["r10"] += 1
                if r < 30:
                    agg["r30"] += 1
                agg["rank"].append(r)
            if t in keep:
                agg["rb"] += 1
    n = max(1, agg["pairs"])
    return {"pairs": agg["pairs"], "recall@10": agg["r10"] / n, "recall@30": agg["r30"] / n,
            "recall@budget": agg["rb"] / n,
            "mean_len": sum(agg["len"]) / max(1, len(agg["len"])),
            "mean_pool": sum(agg["pool"]) / max(1, len(agg["pool"])),
            "mean_gold_rank": (sum(agg["rank"]) / len(agg["rank"])) if agg["rank"] else None}


RES = {"scorer": "a8_c50m50 + kind prior", "threshold": THR, "budget": f"min({CAP},max(30,{MULT}w))",
       "per_user_cap": PER_USER_CAP}

# --------------------------------------------------------------- condition (a)
uterms, urank, ukind = pool_union()
log(f"union pool={len(uterms)}")
sets = {}
rows_new = read_jsonl(ASR_NEW)
sets["heldout_new"] = (rows_new, [r["hyp"] for r in rows_new], [{r["term"]} for r in rows_new])
rows_p2 = load_pool2_rows()
sets["heldout_pool2"] = (rows_p2, [r["hyp"] for r in rows_p2], [{r["term"]} for r in rows_p2])

lists_a_new = None
for name, (rows, texts, golds) in sets.items():
    lists = score_rows(uterms, urank, ukind, texts)
    if name == "heldout_new":
        lists_a_new = lists
    st = stats(lists, [len(t.split()) for t in texts], golds, [len(uterms)] * len(texts))
    RES.setdefault(name, {})["full_union"] = st
    log(f"[{name}] (a) full union: r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
        f"r@budget={st['recall@budget']:.4f} len={st['mean_len']:.1f} "
        f"pool={st['mean_pool']:.0f} [{time.perf_counter()-T0:.0f}s]")
    if name != "heldout_new":
        del lists

# --------------------------------------------------------------- condition (b)
# --- heldout_new: group clips by the gold term's own repo, two index tiers
t2r = json.loads((D / "heldnew_term2repo.json").read_text())
groups = defaultdict(list)
for i, r in enumerate(rows_new):
    groups[t2r.get(r["term"], "?")].append(i)
log(f"heldout_new repo groups={len(groups)} "
    f"{sorted(((k, len(v)) for k, v in groups.items()), key=lambda x: -x[1])[:6]}")
texts_new = sets["heldout_new"][1]
golds_new = sets["heldout_new"][2]
nw_new = [len(t.split()) for t in texts_new]

for tier, byfile, ixfile in (("definitions", "by_repo.json", "symbols_local.jsonl"),
                             ("mentions", "by_repo_mentions.json", "symbols_local_mentions.jsonl")):
    by_repo = json.loads((D / byfile).read_text())
    ix_rows = {}
    for l in (D / ixfile).open(encoding="utf-8"):
        r = json.loads(l)
        ix_rows[r["term"]] = (r["count"], r["kind"])
    log(f"[{tier}] index rows={len(ix_rows)} repos={len(by_repo)}")
    lists_b = [None] * len(rows_new)
    pool_b = [0] * len(rows_new)
    for repo, idxs in sorted(groups.items(), key=lambda kv: len(by_repo.get(kv[0], []))):
        terms = list(by_repo.get(repo, []))
        if PER_USER_CAP:
            terms = sorted(terms, key=lambda t: -ix_rows.get(t, (0, ""))[0])[:PER_USER_CAP]
        if not terms:
            for i in idxs:
                lists_b[i] = []
            continue
        rank = {t: float(ix_rows.get(t, (0, ""))[0]) for t in terms}
        kind = {t: kind_of(t, ix_rows.get(t, (0, ""))[1]) for t in terms}
        sub = score_rows(terms, rank, kind, [texts_new[i] for i in idxs])
        for j, i in enumerate(idxs):
            lists_b[i] = sub[j]
            pool_b[i] = len(terms)
        log(f"  [{tier}] {repo}: pool={len(terms)} clips={len(idxs)} [{time.perf_counter()-T0:.0f}s]")
    present = [i for i in range(len(rows_new)) if pool_b[i]]
    cov = [i for i in present if rows_new[i]["term"] in set(by_repo.get(t2r.get(rows_new[i]["term"], "?"), []))]
    for label, sel in (("all_clips", list(range(len(rows_new)))), ("repo_present", present),
                       ("gold_in_index", cov)):
        if not sel:
            continue
        st = stats([lists_b[i] for i in sel], [nw_new[i] for i in sel],
                   [golds_new[i] for i in sel], [pool_b[i] for i in sel])
        st["clips"] = len(sel)
        RES["heldout_new"][f"per_user_{tier}_{label}"] = st
        log(f"[heldout_new] (b {tier}/{label}, {len(sel)} clips): r@10={st['recall@10']:.4f} "
            f"r@30={st['recall@30']:.4f} r@budget={st['recall@budget']:.4f} "
            f"len={st['mean_len']:.1f} pool={st['mean_pool']:.0f}")
        if label != "all_clips":
            stA = stats([lists_a_new[i] for i in sel], [nw_new[i] for i in sel],
                        [golds_new[i] for i in sel], [len(uterms)] * len(sel))
            stA["clips"] = len(sel)
            RES["heldout_new"][f"full_union_{tier}_{label}"] = stA
            log(f"[heldout_new] (a union, same {len(sel)} clips): r@30={stA['recall@30']:.4f} "
                f"r@budget={stA['recall@budget']:.4f}")
    del lists_b, by_repo, ix_rows

# --- heldout_pool2: group clips by the gold term's own pool-2 source
p2_src = {}
for p in (Path("/data/phonon_pool2_v0/terms_pool2.jsonl"),
          Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl")):
    for r in read_jsonl(p):
        p2_src.setdefault(r["source"], []).append(r["term"])
held_src = {r["term"]: r["source"] for r in
            read_jsonl(Path("/data/phonon_pool2_v0/terms_heldout_pool2.jsonl"))}
groups2 = defaultdict(list)
for i, r in enumerate(rows_p2):
    groups2[held_src.get(r["term"], "?")].append(i)
log(f"heldout_pool2 source groups={len(groups2)}")
texts_p2 = sets["heldout_pool2"][1]
golds_p2 = sets["heldout_pool2"][2]
lists_c = [None] * len(rows_p2)
pool_c = [0] * len(rows_p2)
for source, idxs in sorted(groups2.items()):
    terms = p2_src.get(source, [])
    if not terms:
        for i in idxs:
            lists_c[i] = []
        continue
    rank = {t: 1.0 for t in terms}
    kind = {t: kind_of(t, "") for t in terms}
    sub = score_rows(terms, rank, kind, [texts_p2[i] for i in idxs])
    for j, i in enumerate(idxs):
        lists_c[i] = sub[j]
        pool_c[i] = len(terms)
    log(f"  [pool2] {source}: pool={len(terms)} clips={len(idxs)} [{time.perf_counter()-T0:.0f}s]")
st = stats(lists_c, [len(t.split()) for t in texts_p2], golds_p2, pool_c)
st["gold_in_restricted_pool"] = len(rows_p2)
RES["heldout_pool2"]["per_user_index"] = st
log(f"[heldout_pool2] (b) per-source: r@10={st['recall@10']:.4f} r@30={st['recall@30']:.4f} "
    f"r@budget={st['recall@budget']:.4f} len={st['mean_len']:.1f} pool={st['mean_pool']:.0f}")

(D / "peruser.json").write_text(json.dumps(RES, indent=1, default=float) + "\n")
log(f"STEP 4 done [{time.perf_counter()-T0:.0f}s]")
