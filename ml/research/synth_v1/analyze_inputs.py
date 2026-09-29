"""CPU analysis: pollution survival, dumps, real filler extras. No GPU."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pollution import (  # noqa: E402
    classify_term,
    has_identifier_dump,
    is_third_party_path,
    is_user_authored_path,
)

RANKED = Path("/data/phonon_synth_v0/lexicon_ranked.jsonl")
CLEAN = Path("/data/phonon_synth_v0/clean_utterances.jsonl")
TTS = Path("/data/phonon_synth_v0/tts_pairs.jsonl")
REAL = Path("/data/phonon_personal/wispr_20260915/corrector_pairs_v0.jsonl")
OUT = Path("/data/phonon_synth_v1/tmp/analyze_inputs.json")

EXAMPLES = [
    "PersimmonModel",
    "Blip2VisionConfig",
    "FlightGear",
    "KOOPA",
    "GatherGetCollByteCount",
    "cublasCreate_v2",
    "Grok",
    "KernelBench",
    "Wispr",
    "Qwen",
    "Gpubox",
    "Parakeet",
]

FILLER_UNI = ("um", "uh", "uhh", "umm", "like")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ranked = list(load_jsonl(RANKED))
    by_class = Counter()
    example_status = {}
    keep_terms = set()
    polluted_terms = set()
    weak_terms = set()
    keep_kinds = Counter()
    polluted_path_examples = Counter()
    for row in ranked:
        cls = classify_term(row)
        by_class[cls] += 1
        term = row["term"]
        if cls == "keep":
            keep_terms.add(term)
            keep_kinds[row.get("kind") or "?"] += 1
        elif cls == "weak":
            weak_terms.add(term)
        else:
            polluted_terms.add(term)
            src = row.get("source_file") or ""
            polluted_path_examples[is_third_party_path(src)] += 1
        if term in EXAMPLES:
            files = row.get("files") or []
            example_status[term] = {
                "class": cls,
                "kind": row.get("kind"),
                "count": row.get("count"),
                "n_files": len(files),
                "n_user": sum(1 for f in files if is_user_authored_path(f)),
                "n_third": sum(1 for f in files if is_third_party_path(f)),
                "files": files[:5],
            }

    clean = list(load_jsonl(CLEAN))
    dump_n = 0
    all_polluted = 0
    mixed = 0
    all_clean = 0
    no_terms = 0
    survive_strict = 0  # not dump, not all-polluted
    survive_keep_weak = 0
    survive_no_polluted_term = 0
    for row in clean:
        terms = list(row.get("terms") or [])
        text = row.get("text") or ""
        dump = has_identifier_dump(text)
        if dump:
            dump_n += 1
        if not terms:
            no_terms += 1
            term_cls = "none"
        else:
            states = []
            for t in terms:
                if t in keep_terms:
                    states.append("keep")
                elif t in weak_terms:
                    states.append("weak")
                else:
                    states.append("polluted")
            if all(s == "polluted" for s in states):
                all_polluted += 1
                term_cls = "all_polluted"
            elif all(s in {"keep", "weak"} for s in states):
                all_clean += 1
                term_cls = "all_clean"
            else:
                mixed += 1
                term_cls = "mixed"
        if not dump and term_cls != "all_polluted":
            survive_strict += 1
        if not dump and term_cls in {"all_clean", "mixed", "none"}:
            survive_keep_weak += 1
        if not dump and terms and any(t in keep_terms for t in terms) and not any(
            t in polluted_terms for t in terms
        ):
            survive_no_polluted_term += 1
        elif not dump and not terms:
            survive_no_polluted_term += 1

    tts = list(load_jsonl(TTS))
    tts_dump = 0
    tts_all_polluted = 0
    tts_survive = 0
    for row in tts:
        terms = list(row.get("terms") or [])
        target = row.get("target") or ""
        dump = has_identifier_dump(target)
        if dump:
            tts_dump += 1
        if terms and all((t not in keep_terms and t not in weak_terms) for t in terms):
            tts_all_polluted += 1
            all_p = True
        else:
            all_p = False
        if not dump and not all_p:
            tts_survive += 1

    # real filler extras
    extra = Counter()
    n_ref_words = 0
    n_pairs = 0
    you_know_in = 0
    you_know_tgt = 0
    like_in = 0
    like_tgt = 0
    for row in load_jsonl(REAL):
        raw = (row.get("asr") or row.get("input") or "").lower()
        tgt = (row.get("target") or "").lower()
        if not tgt.strip() or not raw.strip():
            continue
        n_pairs += 1
        rw = WORD_RE.findall(raw)
        tw = WORD_RE.findall(tgt)
        n_ref_words += len(tw)
        rc, tc = Counter(rw), Counter(tw)
        for f in FILLER_UNI:
            extra[f] += max(0, rc[f] - tc[f])
        # you know bigram
        def bigrams(ws):
            return sum(1 for i in range(len(ws) - 1) if ws[i] == "you" and ws[i + 1] == "know")

        yi, yt = bigrams(rw), bigrams(tw)
        extra["you know"] += max(0, yi - yt)
        you_know_in += yi
        you_know_tgt += yt
        like_in += rc["like"]
        like_tgt += tc["like"]

    filler_rates = {k: extra[k] / n_ref_words for k in list(FILLER_UNI) + ["you know"]}

    out = {
        "ranked_n": len(ranked),
        "term_class": dict(by_class),
        "keep_kinds": dict(keep_kinds),
        "example_status": example_status,
        "clean_n": len(clean),
        "dump_n": dump_n,
        "all_polluted": all_polluted,
        "mixed": mixed,
        "all_clean": all_clean,
        "no_terms": no_terms,
        "survive_strict": survive_strict,
        "survive_keep_weak": survive_keep_weak,
        "survive_no_polluted_term": survive_no_polluted_term,
        "tts_n": len(tts),
        "tts_dump": tts_dump,
        "tts_all_polluted": tts_all_polluted,
        "tts_survive": tts_survive,
        "real_n_pairs": n_pairs,
        "real_n_ref_words": n_ref_words,
        "filler_extra_counts": dict(extra),
        "filler_extra_per_word": filler_rates,
        "you_know_input": you_know_in,
        "you_know_target": you_know_tgt,
        "like_input": like_in,
        "like_target": like_tgt,
        "keep_n": len(keep_terms),
        "weak_n": len(weak_terms),
        "polluted_n": len(polluted_terms),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2)[:6000])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
