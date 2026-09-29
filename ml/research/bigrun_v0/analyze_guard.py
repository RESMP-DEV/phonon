"""bigrun_v0 length-guard eval: the same metrics as analyze_bigrun.py, guard off vs guard on.

Reads /data/phonon_bigrun_v0/refined (guard off) and .../refined_guard (guard on) for the three
adapters and writes research/bigrun_v0/results_guard.md + results_guard.json. Metric definitions
are imported unchanged from research/vocab_v1/analyze_vocab1.py via analyze_bigrun.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/phonon/research/vocab_v1")
sys.path.insert(0, "/home/user/phonon/research/bigrun_v0")
from analyze_vocab1 import fair, read_jsonl, row_wer  # noqa: E402
from analyze_bigrun import real_block, term_block  # noqa: E402
from refine_bigrun import guard_post  # noqa: E402

D = Path("/data/phonon_bigrun_v0")
DIRS = {"off": D / "refined", "on": D / "refined_guard"}
NEW_FILE = D / "eval_conditions_new.jsonl"
REAL_FILE = Path("/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl")
MODES = ["off", "post", "on"]
LABELS = ["bigrun_big_350m", "bigrun_mid_r16", "bigrun_xl_r16"]
BASE = {"bigrun_big_350m": "LFM2.5-350M", "bigrun_mid_r16": "LFM2.5-1.2B",
        "bigrun_xl_r16": "LFM2.5-1.2B"}
REAL_CONDS = ["none", "retrieved"]
TERM_CONDS = ["none", "oracle", "retrieved"]


READ = """
## Read

The word post-check is the whole guard on real audio: applied to the unchanged generations it
takes `bigrun_big_350m`'s retrieved fair WER from 0.1796 to 0.1382 by replacing 3 of 580 rows,
and `bigrun_xl_r16` from 0.1187 to 0.1120 by catching the two rows it had collapsed to 9 words.
The token cap almost never fires on its own - 1 row out of 3,480 real-audio decodes stopped at
the cap without an EOS - but it is what bounds worst-case latency, and on the 350M's one runaway
(a 328-word input) it holds the output to 498 words instead of 1,612. The guard costs the 1.2B
nothing: `bigrun_mid_r16`'s shipping condition (real audio, retrieved list) comes back
byte-identical with the guard on - 0 rows fired, 0.1127 fair WER, 329/218/33 win/tie/loss - and
heldout_new term hit is 0.811 either way, while `bigrun_xl_r16` gains 0.007 WER and loses its two
truncated rows. It does not make the 350M shippable: its best guarded number is 0.1382 against
`bigrun_mid_r16`'s 0.1127, 23 percent worse, with a higher damage rate (0.069 against 0.057) - the
guard removes the catastrophe, not the deficit. Two caveats: re-decoding with per-row caps changes
the batch's `max_new_tokens` and batched greedy decode on this stack is not bit-reproducible, so
46-50 of the 580 rows drift for the 350M (1-2 for the 1.2B) and its `on` number of 0.1515 is that
drift rather than the guard; and the 0.4x collapse rule has margin on this holdout (no reference
is shorter than 0.6x its input) but it is the rule to watch for a user whose dictation compresses
harder than this one's.
"""


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main() -> int:
    res = {"real": {}, "new": {}, "guard": {}}

    rrows = read_jsonl(REAL_FILE)
    refs = [r["reference"] for r in rrows]
    ins = [r["input"] for r in rrows]
    vocs = [r["vocab_retrieved"] for r in rrows]
    raw_rw = [row_wer(a, b, fair) for a, b in zip(refs, ins)]
    res["real"]["raw"] = {"none": real_block(refs, ins, ins, vocs, raw_rw)}
    res["real"]["raw"]["none"].update({"win": 0, "tie": len(rrows), "loss": 0, "damage_rate": 0.0})

    nrows = read_jsonl(NEW_FILE)
    n_refs = [r["reference"] for r in nrows]
    n_raws = [r["hyp"] for r in nrows]
    n_terms = [r["term"] for r in nrows]
    n_ret = [r["retrieved_has_term"] for r in nrows]
    n_vr = [r["vocab_retrieved"] for r in nrows]
    n_vo = [r["vocab_oracle"] for r in nrows]
    n_raw_rw = [row_wer(a, b, fair) for a, b in zip(n_refs, n_raws)]
    res["new"]["raw"] = {"none": term_block(n_refs, n_raws, n_terms, n_raws, n_vr, n_raw_rw, n_ret)}

    for g, root in DIRS.items():
        for lab in LABELS:
            key = f"{lab}|{g}"
            p = root / f"{lab}__real.jsonl"
            if p.exists():
                by = {r["id"]: r for r in read_jsonl(p)}
                res["real"][key] = {}
                for c in REAL_CONDS:
                    hy = [by.get(r["id"], {}).get(f"out_{c}", "") for r in rrows]
                    res["real"][key][c] = real_block(refs, ins, hy, vocs, raw_rw)
                    res["guard"][f"real|{key}|{c}"] = {
                        "fired": sum(by.get(r["id"], {}).get(f"fired_{c}", 0) for r in rrows),
                        "capped": sum(by.get(r["id"], {}).get(f"capped_{c}", 0) for r in rrows)}
                if g == "off":
                    pk = f"{lab}|post"
                    res["real"][pk] = {}
                    for c in REAL_CONDS:
                        hy = [by.get(r["id"], {}).get(f"out_{c}", "") for r in rrows]
                        pairs = [guard_post(a, b) for a, b in zip(ins, hy)]
                        res["real"][pk][c] = real_block(refs, ins, [x[0] for x in pairs],
                                                        vocs, raw_rw)
                        res["guard"][f"real|{pk}|{c}"] = {
                            "fired": sum(1 for x in pairs if x[1]), "capped": "-"}
            p = root / f"{lab}__new.jsonl"
            if p.exists():
                by = {r["id"]: r for r in read_jsonl(p)}
                res["new"][key] = {}
                for c in TERM_CONDS:
                    hy = [by.get(r["id"], {}).get(f"out_{c}", "") for r in nrows]
                    vv = n_vo if c == "oracle" else n_vr
                    res["new"][key][c] = term_block(n_refs, n_raws, n_terms, hy, vv,
                                                    n_raw_rw, n_ret)
                    res["guard"][f"new|{key}|{c}"] = {
                        "fired": sum(by.get(r["id"], {}).get(f"fired_{c}", 0) for r in nrows),
                        "capped": sum(by.get(r["id"], {}).get(f"capped_{c}", 0) for r in nrows)}
                if g == "off":
                    pk = f"{lab}|post"
                    res["new"][pk] = {}
                    for c in TERM_CONDS:
                        hy = [by.get(r["id"], {}).get(f"out_{c}", "") for r in nrows]
                        vv = n_vo if c == "oracle" else n_vr
                        pairs = [guard_post(a, b) for a, b in zip(n_raws, hy)]
                        res["new"][pk][c] = term_block(n_refs, n_raws, n_terms,
                                                       [x[0] for x in pairs], vv, n_raw_rw, n_ret)
                        res["guard"][f"new|{pk}|{c}"] = {
                            "fired": sum(1 for x in pairs if x[1]), "capped": "-"}

    out = Path("/home/user/phonon/research/bigrun_v0")
    (out / "results_guard.json").write_text(json.dumps(res, indent=1) + "\n")

    L = []
    L.append("# bigrun_v0: decode-time length guard\n")
    L.append("Guard: `max_new_tokens = 1.5 x input tokens + 32` per row, then fall back to the "
             "raw input when the output is over 2x or (for inputs over 8 words) under 0.4x the "
             "input word count. Same adapters, same 580-row real-audio date-split holdout "
             "(`/data/phonon_corrector_v0/datesplit/holdout_date_audio.jsonl`, retrieved lists "
             "from `/data/phonon_term_eval_v1/real/eval_holdout_date_audio.jsonl`) and the "
             "2,700-clip `heldout_new` set. Metrics are the ones in `analyze_bigrun.py`; "
             "`fired` counts rows replaced by the raw input, `cap` counts rows whose decode hit "
             "the token cap without emitting EOS.\n")
    L.append("\nThree modes. `off` is the unguarded baseline. `post` applies only the word-count "
             "post-check to those exact generations, so it isolates the rule with zero decode "
             "noise. `on` is the shipped guard: per-row token cap at decode time plus the "
             "post-check. `on` regenerates, and batched decode on this stack is not bit-identical "
             "when the batch's max_new_tokens changes - 46-50 of the 580 rows drift for the 350M "
             "(1-2 for the 1.2B), so read `post` for the rule's effect and `on` for what ships.\n")

    L.append("\n## Real audio, 580 rows\n")
    L.append("| adapter | guard | cond | fair WER | strict WER | damage | win | tie | loss | "
             "fired | cap | runaway | trunc |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    b = res["real"]["raw"]["none"]
    L.append(f"| raw Parakeet | - | - | {f4(b['fair_wer'])} | {f4(b['strict_lc_wer'])} | 0.000 | "
             f"0 | {b['n']} | 0 | - | - | 0 | 0 |")
    for lab in LABELS:
        for c in REAL_CONDS:
            for g in MODES:
                k = f"{lab}|{g}"
                if k not in res["real"]:
                    continue
                r = res["real"][k][c]
                gu = res["guard"].get(f"real|{k}|{c}", {})
                L.append(f"| {lab} | {g} | {c} | {f4(r['fair_wer'])} | {f4(r['strict_lc_wer'])} | "
                         f"{f3(r['damage_rate'])} | {r['win']} | {r['tie']} | {r['loss']} | "
                         f"{gu.get('fired', '-')} | {gu.get('capped', '-')} | "
                         f"{r['runaway_rows']} | {r['truncated_rows']} |")

    L.append("\n## heldout_new, 2,700 clips\n")
    L.append("| adapter | guard | cond | term hit | fair WER | strict WER | damage | "
             "fired | cap |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    b = res["new"]["raw"]["none"]
    L.append(f"| raw Parakeet | - | - | {f3(b['term_hit_norm'])} | {f4(b['fair_wer'])} | "
             f"{f4(b['strict_lc_wer'])} | 0.000 | - | - |")
    for lab in LABELS:
        for c in TERM_CONDS:
            for g in MODES:
                k = f"{lab}|{g}"
                if k not in res["new"]:
                    continue
                r = res["new"][k][c]
                gu = res["guard"].get(f"new|{k}|{c}", {})
                L.append(f"| {lab} | {g} | {c} | {f3(r['term_hit_norm'])} | {f4(r['fair_wer'])} | "
                         f"{f4(r['strict_lc_wer'])} | {f3(r['damage_rate'])} | "
                         f"{gu.get('fired', '-')} | {gu.get('capped', '-')} |")

    L.append(READ)
    (out / "results_guard.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
