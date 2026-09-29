"""Pool rows + retrieval lists -> training rows (the v1 recipe).

Same numerics as research/bigrun_v0/step5c_mix.py and research/pool3_v0/step4c_mix3.py:

  acoustic row: rng = seeded(<seed_tag>, id); u = rng.random()
    u < p_noline                    -> no vocabulary line, target = corrected target
    u < p_noline + p_restraint      -> keep-as-heard target, gold term removed from the list
    else                            -> top-k list with the gold term forced in at the last slot
  real row:     id = <id>#r<rep>; rng = seeded(<real seed_tag>, id); p_noline of them lose the list
  replay:       for each replay file, rows whose source == match keep with probability `fraction`
                under random.Random(seed), in file order
  assembly:     new rows + replay rows + real rows, random.Random(seed).shuffle

keep_as_heard and chat_messages are imported from research/vocab_v1 and research/vocab_v0.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..paths import research_syspath
from ..util import iter_jsonl, log, write_json


def _imports(lexicon: str | None):
    research_syspath()
    import vocab_common

    if lexicon:
        vocab_common.LEXICON_PATH = Path(lexicon)
    from build_data_v1 import keep_as_heard  # research/vocab_v1
    from vocab_common import Lexicon, chat_messages, seeded

    return keep_as_heard, Lexicon, chat_messages, seeded


class RowBuilder:
    """Builds one training row at a time; the only place branch numerics live."""

    def __init__(self, cfg: dict, held: set[str], lexicon: str | None = None,
                 need_lexicon: bool = True):
        self.keep_as_heard, Lexicon, self.chat_messages, self.seeded = _imports(lexicon)
        self.cfg = cfg
        self.topk = int(cfg.get("topk", 30))
        self.p_noline = float(cfg.get("p_noline", 0.20))
        self.p_restraint = float(cfg.get("p_restraint", 0.30))
        self.ac = cfg.get("acoustic") or {}
        self.real_cfg = cfg.get("real") or {}
        self.held = held
        self.n = Counter()
        self.lex = None
        if need_lexicon and self.real_cfg:
            self.lex = Lexicon(exclude=held)
            log(f"lexicon_rows={len(self.lex.rows)} held={len(held)}")

    # ---------------- acoustic ----------------
    def acoustic(self, row: dict, terms: list[str]) -> dict:
        rng = self.seeded(self.ac.get("seed_tag", "v1ac"), row["id"])
        term = row.get("term") or ""
        u = rng.random()
        if u < self.p_noline:
            vocab, target, branch = [], row["target"], "noline"
            self.n["ac_noline"] += 1
        elif u < self.p_noline + self.p_restraint:
            kept = self.keep_as_heard(row["input"], row["target"], term)
            if kept is None:
                kept = row["input"]
                self.n["ac_restraint_failed"] += 1
            vocab = [t for t in terms if t != term][:self.topk]
            rng.shuffle(vocab)
            target, branch = kept, "restraint"
            self.n["ac_restraint"] += 1
        else:
            vocab = list(terms[:self.topk])
            if term and term not in vocab:
                vocab = vocab[:self.topk - 1] + [term]
                self.n["ac_forced_term"] += 1
            rng.shuffle(vocab)
            target, branch = row["target"], "list"
            self.n["ac_normal"] += 1
        return {"id": row["id"], "source": self.ac.get("source", "tts_parakeet"),
                "branch": branch, "stage": row.get("stage"), "term": term,
                "kind": row.get("kind"), "voice": row["voice"],
                "sentence_idx": row["sentence_idx"],
                "input": row["input"], "target": target,
                "true_terms": [term] if term else [], "vocab": vocab,
                "messages": self.chat_messages(row["input"], vocab, target)}

    # ---------------- real ----------------
    def real(self, row: dict, rep: int, terms: list[str]) -> dict:
        rid = f"{row['id']}#r{rep}"
        rng = self.seeded(self.real_cfg.get("seed_tag", "v1real"), rid)
        true_terms = [t for t in self.lex.find_terms(row["target"]) if t not in self.held]
        if rng.random() < self.p_noline:
            vocab = []
            self.n["real_noline"] += 1
        else:
            k = self.real_cfg.get("topk")
            vocab = list(terms[:int(k)] if k else terms)
            rng.shuffle(vocab)
            self.n["real_list"] += 1
        return {"id": rid, "source": row.get("source", "real"),
                "branch": "noline" if not vocab else "list",
                "input": row["input"], "target": row["target"],
                "true_terms": true_terms, "vocab": vocab,
                "messages": self.chat_messages(row["input"], vocab, row["target"])}


def _held_terms(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for p in paths or []:
        out |= {r["term"] for r in iter_jsonl(p)}
    return out


def _lists(path: str | None, ids: set[str] | None = None) -> dict[str, list[str]]:
    if not path:
        return {}
    out = {}
    for r in iter_jsonl(path):
        if ids is None or r["id"] in ids:
            out[r["id"]] = r["terms"]
    return out


def build_mix(cfg: dict, out_dir: Path, held_paths: list[str], lexicon: str | None) -> dict:
    """Full build: acoustic rows + replay + real repeats -> one shuffled jsonl."""
    held = _held_terms(held_paths)
    rb = RowBuilder(cfg, held, lexicon)
    ac_cfg = cfg.get("acoustic") or {}
    real_cfg = cfg.get("real") or {}
    seed = int(cfg.get("seed", 918))

    new_rows: list[dict] = []
    if ac_cfg:
        L_ac = _lists(ac_cfg.get("lists"))
        keep = (ac_cfg.get("filter") or {}).get("stage")
        for row in iter_jsonl(ac_cfg["pool"]):
            if keep and row.get("stage") != keep:
                continue
            new_rows.append(rb.acoustic(row, L_ac.get(row["id"], [])))
        log(f"acoustic rows={len(new_rows)}")

    replay_rows: list[dict] = []
    replay_stats = []
    for i, rp in enumerate(cfg.get("replay") or []):
        rng = random.Random(int(rp.get("seed", seed)))
        kept, avail = [], 0
        real_from_replay: list[dict] = []
        reps = int((rp.get("take_real") or {}).get("repeats", 0))
        for r in iter_jsonl(rp["path"]):
            if r.get("source") == rp["match_source"]:
                avail += 1
                if rng.random() < float(rp["fraction"]):
                    if rp.get("rename_source"):
                        r["source"] = rp["rename_source"]
                    kept.append(r)
            elif reps:
                tail = r["id"].rsplit("#r", 1)[-1]
                if tail.isdigit() and int(tail) < reps:
                    real_from_replay.append(r)
        replay_rows += kept + real_from_replay
        replay_stats.append({"path": rp["path"], "match": rp["match_source"],
                             "fraction": rp["fraction"], "kept": len(kept), "available": avail,
                             "real_rows": len(real_from_replay)})
        log(f"replay[{i}] {len(kept)}/{avail} real={len(real_from_replay)}")

    real_rows: list[dict] = []
    if real_cfg.get("rows"):
        L_real = _lists(real_cfg.get("lists"))
        base = list(iter_jsonl(real_cfg["rows"]))
        for rep in range(int(real_cfg.get("repeats", 1))):
            for row in base:
                real_rows.append(rb.real(row, rep, L_real.get(row["id"], [])))
        log(f"real rows={len(real_rows)} unique={len(base)}")

    rows = new_rows + replay_rows + real_rows
    random.Random(seed).shuffle(rows)
    out = Path(cfg["out"]) if "/" in str(cfg.get("out", "")) else out_dir / cfg.get(
        "out", "train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"out": str(out), "rows": len(rows), "acoustic_rows": len(new_rows),
             "replay_rows": len(replay_rows), "real_rows": len(real_rows),
             "replay": replay_stats,
             "branches": dict(Counter(r.get("branch") for r in rows)),
             "sources": dict(Counter(r.get("source") for r in rows)),
             "branch_counters": dict(rb.n)}
    write_json(out_dir / "mix_stats.json", stats)
    return stats


def verify_mix(cfg: dict, out_dir: Path, held_paths: list[str], lexicon: str | None) -> dict:
    """Rebuild the first N rows of an existing corpus and diff field by field."""
    v = cfg["verify"]
    ref_path = v["reference"]
    n_rows = int(v.get("rows", 2000))
    ref: list[dict] = []
    for r in iter_jsonl(ref_path):
        ref.append(r)
        if len(ref) >= n_rows:
            break
    log(f"reference slice {len(ref)} rows from {ref_path}")

    ac_ids = {r["id"] for r in ref if "#r" not in r["id"]}
    real_ids = {r["id"].rsplit("#r", 1)[0] for r in ref if "#r" in r["id"]}
    ac_cfg = cfg.get("acoustic") or {}
    real_cfg = cfg.get("real") or {}

    pool = {r["id"]: r for r in iter_jsonl(ac_cfg["pool"]) if r["id"] in ac_ids} if ac_cfg else {}
    L_ac = _lists(ac_cfg.get("lists"), ac_ids) if ac_cfg else {}
    base = {r["id"]: r for r in iter_jsonl(real_cfg["rows"])
            if r["id"] in real_ids} if real_cfg.get("rows") else {}
    L_real = _lists(real_cfg.get("lists"), real_ids) if real_cfg.get("rows") else {}
    log(f"loaded pool={len(pool)}/{len(ac_ids)} real={len(base)}/{len(real_ids)}")

    held = _held_terms(held_paths)
    rb = RowBuilder(cfg, held, lexicon, need_lexicon=bool(base))

    same = 0
    missing = 0
    diffs: list[dict] = []
    field_diffs: Counter = Counter()
    by_branch: Counter = Counter()
    for r in ref:
        rid = r["id"]
        if "#r" in rid:
            bid, rep = rid.rsplit("#r", 1)
            src = base.get(bid)
            got = rb.real(src, int(rep), L_real.get(bid, [])) if src else None
        else:
            src = pool.get(rid)
            got = rb.acoustic(src, L_ac.get(rid, [])) if src else None
        if got is None:
            missing += 1
            continue
        by_branch[r.get("branch")] += 1
        if json.dumps(got, sort_keys=True, ensure_ascii=False) == json.dumps(
                r, sort_keys=True, ensure_ascii=False):
            same += 1
            continue
        d = {"id": rid, "fields": []}
        for k in sorted(set(got) | set(r)):
            if got.get(k) != r.get(k):
                d["fields"].append(k)
                field_diffs[k] += 1
        if len(diffs) < 20:
            d["ref"] = {k: r.get(k) for k in d["fields"]}
            d["rebuilt"] = {k: got.get(k) for k in d["fields"]}
            diffs.append(d)
    checked = len(ref) - missing
    stats = {"reference": ref_path, "rows_checked": checked, "identical": same,
             "different": checked - same, "missing_source_row": missing,
             "identical_fraction": same / max(checked, 1),
             "branches_in_slice": dict(by_branch),
             "fields_differing": dict(field_diffs), "examples": diffs}
    write_json(out_dir / "verify_mix.json", stats)
    return stats
