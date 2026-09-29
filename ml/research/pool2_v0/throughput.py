"""pool2: parse the stage logs into /data/phonon_pool2_v0/throughput.json."""
from __future__ import annotations
import json, re
from pathlib import Path

D = Path("/data/phonon_pool2_v0")
LOG = D / "logs"

GEN = re.compile(r"wrote (\d+) sentences for (\d+) terms .*wall_min=([\d.]+)")
TTS = re.compile(r"TTS done clips=(\d+) failed=(\d+) wall_s=([\d.]+) ([\d.]+) clips/s.*-> (\S+)")
ASR = re.compile(r"ASR done clips=(\d+) wall_s=([\d.]+) ([\d.]+) clips/s.*-> (\S+)")


def main() -> int:
    stages = []

    # ---- sentences ------------------------------------------------------
    for label, files in (("sentences, 4 per term (both GPUs)", ["gen2_h0.log", "gen2_h1.log"]),
                         ("held-out sentences, 3 per term", ["gen2_heldout.log"])):
        n = 0
        wall = 0.0
        for f in files:
            p = LOG / f
            if not p.exists():
                continue
            for m in GEN.finditer(p.read_text(errors="ignore")):
                n += int(m.group(1))
                wall = max(wall, float(m.group(3)))
        if n:
            stages.append({"stage": label, "n": n, "failed": 0, "wall_min": wall,
                           "rate": f"{n / (wall * 60):.1f} sent/s"})

    # ---- TTS / ASR ------------------------------------------------------
    tts: dict[str, list] = {"pool": [], "heldout": []}
    asr: dict[str, list] = {"pool": [], "heldout": []}
    for p in sorted(LOG.glob("*.log")):
        txt = p.read_text(errors="ignore")
        for m in TTS.finditer(txt):
            tag = "heldout" if "heldout" in m.group(5) else "pool"
            rec = (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)),
                   m.group(5))
            if rec not in tts[tag]:
                tts[tag].append(rec)
        for m in ASR.finditer(txt):
            tag = "heldout" if "heldout" in m.group(4) else "pool"
            rec = (int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
            if rec not in asr[tag]:
                asr[tag].append(rec)

    for tag, label in (("heldout", "TTS held-out (3 unseen voices)"),
                       ("pool", "TTS pool 2 (9 training voices)")):
        rows = tts[tag]
        if not rows:
            continue
        stages.append({"stage": label, "n": sum(r[0] for r in rows),
                       "failed": sum(r[1] for r in rows),
                       "wall_min": max(r[2] for r in rows) / 60.0,
                       "rate": f"{sum(r[3] for r in rows):.1f} clips/s (both GPUs)"})
    for tag, label in (("heldout", "ASR held-out"), ("pool", "ASR pool 2")):
        rows = asr[tag]
        if not rows:
            continue
        stages.append({"stage": label, "n": sum(r[0] for r in rows), "failed": "-",
                       "wall_min": max(r[1] for r in rows) / 60.0,
                       "rate": f"{sum(r[2] for r in rows):.1f} clips/s (both GPUs)"})

    out = {"stages": stages,
           "tts_detail": {k: [list(r) for r in v] for k, v in tts.items()},
           "asr_detail": {k: [list(r) for r in v] for k, v in asr.items()}}
    (D / "throughput.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"stages": stages}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
