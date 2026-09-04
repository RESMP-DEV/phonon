"""After the Kokoro oracle: attach spoken forms, drop non-seed terms the recognizer already gets right ("same"),
keep the top N by prior, and split a stratified held-out set (every 5th) for gating."""
import json
import re
import sys

inp, cache_path, out_path, held_path = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
cands = json.load(open(inp))
cache = {}
for line in open(cache_path):
    try:
        d = json.loads(line)
        cache[d["term"]] = d["voices"]
    except Exception:
        pass


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def summarize(term, voices):
    """Mirror of profile_miner.oracle.summarize: same / case / format / phonetic plus distinct heard forms."""
    forms, diff = [], "same"
    for v, heard in voices.items():
        if heard == term:
            continue
        forms.append(heard)
        if heard.lower() == term.lower():
            d = "case"
        elif norm(heard) == norm(term):
            d = "format"
        else:
            d = "phonetic"
        order = ["same", "case", "format", "phonetic"]
        if order.index(d) > order.index(diff):
            diff = d
    return diff, list(dict.fromkeys(forms))


out, dropped_same, no_oracle = [], 0, 0
for c in cands:
    if c.get("diff") is None:
        v = cache.get(c["term"])
        if v is None:
            no_oracle += 1
            c["spoken_forms"], c["diff"] = None, "missing"
        else:
            c["diff"], c["spoken_forms"] = summarize(c["term"], v)
    if c["diff"] == "same" and not c["seed"]:
        dropped_same += 1
        continue
    c = dict(c)
    c["id"] = len(out)
    out.append(c)
    if len(out) >= N:
        break
held = [c["id"] for c in out if c["id"] % 5 == 2]
json.dump(out, open(out_path, "w"), ensure_ascii=False)
json.dump(held, open(held_path, "w"))
print(f"[prep] {len(out)} candidates ({dropped_same} dropped as 'same', {no_oracle} without oracle), held-out {len(held)}",
      file=sys.stderr)
