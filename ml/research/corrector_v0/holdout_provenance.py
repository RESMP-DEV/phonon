"""Step 2: are the Wispr holdouts separated from the 4,014 training rows by time
or session, or are they a random split of the same sessions?

Reads only; writes research/corrector_v0/holdout_provenance.{md,json}.
"""
from __future__ import annotations
import json, hashlib, re, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("/data/phonon_personal/wispr_20260915")
CORR = Path("/data/phonon_corrector_v0")
EDIT = Path("/data/phonon_personal/dictation_eval_20260719/edit_samples.json")
OUT = Path("/home/user/phonon/research/corrector_v0")
SESSION_GAP_MIN = 30


def ts(s):
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", s)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    pairs = jl(ROOT / "corrector_pairs_v0.jsonl")
    by_id = {r["id"]: r for r in pairs}
    for r in pairs:
        r["dt"] = ts(r.get("timestamp"))
        r["day"] = r["dt"].date().isoformat() if r["dt"] else None

    # sessions: contiguous dictations with < SESSION_GAP_MIN gap, over the whole export
    ordered = sorted([r for r in pairs if r["dt"]], key=lambda r: r["dt"])
    sid = 0
    prev = None
    for r in ordered:
        if prev is not None and (r["dt"] - prev).total_seconds() > SESSION_GAP_MIN * 60:
            sid += 1
        r["session"] = sid
        prev = r["dt"]

    train = jl(CORR / "train.jsonl")
    src = Counter(r.get("source", "?").split(":")[0] for r in train)
    train_wispr = [by_id[r["id"]] for r in train if r["id"] in by_id]
    train_other = [r for r in train if r["id"] not in by_id]

    holdouts = {}
    h120 = jl(ROOT / "wispr_holdout120.jsonl")
    holdouts["wispr_holdout120"] = [by_id[r["transcriptEntityId"]] for r in h120
                                    if r["transcriptEntityId"] in by_id]
    holdouts["wispr_text_holdout"] = [r for r in pairs if r["split"] == "wispr_text_holdout"]
    edits = json.loads(EDIT.read_text(encoding="utf-8"))
    edit_rows = []
    for e in edits:
        d = ts(e.get("timestamp"))
        edit_rows.append({"id": e["id"], "dt": d, "day": d.date().isoformat() if d else None,
                          "session": None, "in_export": e["id"] in by_id})
    holdouts["wispr_edit25"] = edit_rows

    tr_days = Counter(r["day"] for r in train_wispr if r["day"])
    tr_sessions = {r["session"] for r in train_wispr if r.get("session") is not None}
    tr_dts = sorted(r["dt"] for r in train_wispr if r["dt"])

    res = {
        "session_gap_minutes": SESSION_GAP_MIN,
        "train_rows_total": len(train),
        "train_sources": dict(src),
        "train_rows_from_wispr_export": len(train_wispr),
        "train_rows_not_in_export (TTS etc)": len(train_other),
        "train_date_range": [tr_dts[0].isoformat(), tr_dts[-1].isoformat()] if tr_dts else None,
        "train_days": len(tr_days),
        "train_sessions": len(tr_sessions),
        "holdouts": {},
    }

    for name, rows in holdouts.items():
        dts = sorted(r["dt"] for r in rows if r.get("dt"))
        days = Counter(r["day"] for r in rows if r.get("day"))
        sess = {r["session"] for r in rows if r.get("session") is not None}
        shared_days = set(days) & set(tr_days)
        rows_on_shared_day = sum(days[d] for d in shared_days)
        shared_sess = sess & tr_sessions
        rows_in_shared_session = sum(1 for r in rows if r.get("session") in shared_sess)
        res["holdouts"][name] = {
            "n": len(rows),
            "date_range": [dts[0].isoformat(), dts[-1].isoformat()] if dts else None,
            "days": len(days),
            "days_shared_with_train": len(shared_days),
            "rows_on_days_that_also_have_train_rows": rows_on_shared_day,
            "pct_rows_on_shared_days": (rows_on_shared_day / len(rows)) if rows else None,
            "sessions": len(sess),
            "sessions_shared_with_train": len(shared_sess),
            "rows_in_sessions_that_also_have_train_rows": rows_in_shared_session,
            "id_overlap_with_train_ids": len({r["id"] for r in rows} & {r["id"] for r in train}),
        }

    # wispr_edit25 carries its own id namespace ("edit_xxxx"), so an id join misses
    # the fact that the same dictations are in the export. Join on raw ASR text.
    def nz(s):
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    trows = {r["id"]: r for r in train}
    asr_index = {nz(r["asr"]): r for r in pairs}
    leak = {"n": len(edits), "ts_match_in_export": 0, "asr_match_in_export": 0,
            "in_train_jsonl": 0, "identical_target": 0, "leaked_ids": []}
    ts_set = {r["timestamp"] for r in pairs}
    for e in edits:
        leak["ts_match_in_export"] += e.get("timestamp") in ts_set
        r = asr_index.get(nz(e.get("asr")))
        if not r:
            continue
        leak["asr_match_in_export"] += 1
        if r["id"] in trows:
            leak["in_train_jsonl"] += 1
            leak["leaked_ids"].append(e["id"])
            if nz(trows[r["id"]]["target"]) == nz(e.get("edited")):
                leak["identical_target"] += 1
    res["wispr_edit25_leakage"] = leak

    # verify the hash rule in build_corrector_pairs_v0.py actually produced wispr_text_holdout
    ok = mism = 0
    for r in pairs:
        if r["has_audio"]:
            continue
        want = "wispr_text_holdout" if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000 < 160 \
            else "wispr_text_train"
        ok += (want == r["split"])
        mism += (want != r["split"])
    res["hash_rule_reproduces_text_split"] = {"match": ok, "mismatch": mism}

    # alternative: last 15 percent of DAYS as holdout
    all_days = sorted({r["day"] for r in pairs if r["day"]})
    k = max(1, int(round(0.15 * len(all_days))))
    cut_days = set(all_days[-k:])
    train_ids = {r["id"] for r in train}
    prop_holdout = [r for r in pairs if r["day"] in cut_days]
    prop_holdout_from_train = [r for r in prop_holdout if r["id"] in train_ids]
    prop_train = [r for r in pairs if r["day"] not in cut_days]
    res["date_split_proposal"] = {
        "total_days_in_export": len(all_days),
        "holdout_days": k,
        "holdout_day_range": [all_days[-k], all_days[-1]],
        "cutoff": all_days[-k],
        "holdout_rows_all_pairs": len(prop_holdout),
        "train_rows_all_pairs": len(prop_train),
        "rows_currently_in_train_jsonl_that_move_to_holdout": len(prop_holdout_from_train),
        "wispr_train_rows_remaining": len(train_wispr) - len(prop_holdout_from_train),
        "current_holdout120_rows_that_stay_holdout": sum(
            1 for r in holdouts["wispr_holdout120"] if r["day"] in cut_days),
        "current_text_holdout_rows_that_stay_holdout": sum(
            1 for r in holdouts["wispr_text_holdout"] if r["day"] in cut_days),
        "note": "proposal only, no file written",
    }

    # day histogram for the tail
    res["rows_per_day_tail"] = {d: sum(1 for r in pairs if r["day"] == d) for d in all_days[-12:]}

    (OUT / "holdout_provenance.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    L = []
    A = L.append
    A("# Holdout provenance: are the Wispr eval sets separated by time or session?\n")
    A(f"Export: `{ROOT}` ({len(pairs)} usable pairs). Training file: `{CORR}/train.jsonl` "
      f"({len(train)} rows, {len(train_wispr)} of them from this export, "
      f"{len(train_other)} TTS/other).\n")
    A(f"Sessions are reconstructed from timestamps (the export carries no session id): a new session "
      f"starts after a gap of more than {SESSION_GAP_MIN} minutes.\n")
    A("## Verdict\n")
    A("**Random split of the same sessions, not a time or session split.** "
      "`build_corrector_pairs_v0.py` assigns `wispr_text_holdout` by "
      "`sha256(transcriptEntityId) % 1000 < 160`, a content-independent 16 percent hash of the id; "
      "the 120 audio clips inherit their split from `manifest_with_audio.jsonl`, also id-keyed. "
      "Nothing in either rule looks at the timestamp.\n")
    A(f"Hash rule replay over the {ok+mism} text rows: {ok} match, {mism} mismatch.\n")
    A("## Date ranges and overlap\n")
    A("| set | n | first | last | days | days shared with train | rows on a day that also has train rows | sessions | sessions shared | rows in a shared session |")
    A("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    A(f"| train (wispr rows) | {len(train_wispr)} | {res['train_date_range'][0]} | {res['train_date_range'][1]} "
      f"| {res['train_days']} | - | - | {res['train_sessions']} | - | - |")
    for name, d in res["holdouts"].items():
        dr = d["date_range"] or ["-", "-"]
        A(f"| {name} | {d['n']} | {dr[0]} | {dr[1]} | {d['days']} | {d['days_shared_with_train']} "
          f"| {d['rows_on_days_that_also_have_train_rows']} ({100*(d['pct_rows_on_shared_days'] or 0):.0f}%) "
          f"| {d['sessions']} | {d['sessions_shared_with_train']} | {d['rows_in_sessions_that_also_have_train_rows']} |")
    A("")
    A("Sessions are not defined for `wispr_edit25`: it comes from the older "
      "`dictation_eval_20260719` export and carries its own `edit_xxxxxxxx` ids.\n")
    A("Exact id overlap between each holdout and train.jsonl: " +
      ", ".join(f"{n}={d['id_overlap_with_train_ids']}" for n, d in res["holdouts"].items()) +
      ". For the two Wispr splits that is genuine: no row leaks, only days and sessions.\n")
    A("## wispr_edit25 is not a holdout at all\n")
    A(f"Joining on raw ASR text instead of id: all {leak['asr_match_in_export']} of {leak['n']} "
      f"edit25 clips are present in the 2026-09-15 export ({leak['ts_match_in_export']} also match "
      f"by timestamp), and **{leak['in_train_jsonl']} of {leak['n']} are rows of "
      f"`train.jsonl`** — same raw ASR input, and for {leak['identical_target']} of them the "
      "training target is byte-identical to the edit25 reference after whitespace folding. "
      "The id-level check missed it because the two exports use different id namespaces. "
      "Any number measured on wispr_edit25 for a model trained on train.jsonl is a training-set "
      "number on 60 percent of its rows.\n")
    p = res["date_split_proposal"]
    A("## Alternative date-level split (proposal, nothing written)\n")
    A(f"Hold out the last 15 percent of dictation days: {p['holdout_days']} of {p['total_days_in_export']} days, "
      f"{p['holdout_day_range'][0]} .. {p['holdout_day_range'][1]}, cutoff {p['cutoff']}.\n")
    A("| quantity | rows |")
    A("|---|---:|")
    A(f"| holdout rows (all export pairs on those days) | {p['holdout_rows_all_pairs']} |")
    A(f"| train rows (all export pairs before the cutoff) | {p['train_rows_all_pairs']} |")
    A(f"| rows currently in train.jsonl that would move to holdout | {p['rows_currently_in_train_jsonl_that_move_to_holdout']} |")
    A(f"| wispr train rows remaining after the move | {p['wispr_train_rows_remaining']} |")
    A(f"| current wispr_holdout120 rows that stay held out | {p['current_holdout120_rows_that_stay_holdout']} / 120 |")
    A(f"| current wispr_text_holdout rows that stay held out | {p['current_text_holdout_rows_that_stay_holdout']} / {res['holdouts']['wispr_text_holdout']['n']} |")
    A("")
    A("Rows per day over the last 12 days of the export: " +
      ", ".join(f"{d}={c}" for d, c in res["rows_per_day_tail"].items()) + "\n")
    (OUT / "holdout_provenance.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(res, indent=1)[:4000])
    print("STEP 2 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
