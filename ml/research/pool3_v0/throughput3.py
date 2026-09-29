"""pool3: collect stage throughput from the logs into /data/phonon_pool3_v0/throughput.json."""
from __future__ import annotations
import json, re
from pathlib import Path

D = Path("/data/phonon_pool3_v0")
L = D / "logs"


def gen(log: Path, label: str):
    if not log.exists():
        return None
    m = re.search(r"wrote (\d+) sentences for (\d+) terms .*wall_min=([\d.]+)",
                  log.read_text(errors="ignore"))
    if not m:
        return None
    n, _t, w = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return {"stage": label, "n": n, "failed": 0, "wall_min": w,
            "rate": f"{n / max(w * 60, 1e-9):.1f} sent/s (1 GPU)"}


def tts(tag: str, label: str, shards: list[int]):
    tot = fail = 0
    wall = 0.0
    det = []
    for h in shards:
        p = L / f"tts_{tag}_s{h}.log"
        if not p.exists():
            continue
        m = re.search(r"TTS done clips=(\d+) failed=(\d+) wall_s=([\d.]+) ([\d.]+) clips/s",
                      p.read_text(errors="ignore"))
        if not m:
            continue
        tot += int(m.group(1)); fail += int(m.group(2))
        wall = max(wall, float(m.group(3)))
        det.append([int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))])
    if not det:
        return None, det
    return {"stage": label, "n": tot, "failed": fail, "wall_min": wall / 60,
            "rate": f"{tot / max(wall, 1e-9):.1f} clips/s (1 GPU, {len(det)} processes)"}, det


def asr(tag: str, label: str, shards: list[int]):
    tot = 0
    wall = 0.0
    det = []
    for h in shards:
        p = L / f"asr_{tag}_s{h}.log"
        if not p.exists():
            continue
        txt = p.read_text(errors="ignore")
        m = None
        for m in re.finditer(r"asr (\d+)/(\d+) ([\d.]+) clips/s elapsed_min=([\d.]+)", txt):
            pass
        if not m:
            continue
        n = int(m.group(2))
        el = float(m.group(4)) * 60
        tot += n
        wall = max(wall, el)
        det.append([n, round(el, 1), float(m.group(3))])
    if not det:
        return None, det
    return {"stage": label, "n": tot, "failed": "-", "wall_min": wall / 60,
            "rate": f"{tot / max(wall, 1e-9):.1f} clips/s (1 GPU, {len(det)} processes)"}, det


def main() -> int:
    stages = []
    for s in (gen(L / "gen_pool.log", "sentences, 2 per term (1 GPU)"),
              gen(L / "gen_heldout.log", "held-out sentences, 3 per term")):
        if s:
            stages.append(s)
    t_pool, d_pool = tts("pool", "TTS pool 3 (2 of the 9 training voices per term)", [0, 1])
    t_held, d_held = tts("heldout", "TTS held-out (3 unseen voices)", [0])
    a_pool, ad_pool = asr("pool", "ASR pool 3 (Parakeet TDT 0.6b v2)", [0, 1])
    a_held, ad_held = asr("heldout", "ASR held-out", [0])
    for s in (t_held, t_pool, a_held, a_pool):
        if s:
            stages.append(s)
    out = {"stages": stages,
           "tts_detail": {"pool": d_pool, "heldout": d_held},
           "asr_detail": {"pool": ad_pool, "heldout": ad_held}}
    (D / "throughput.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
