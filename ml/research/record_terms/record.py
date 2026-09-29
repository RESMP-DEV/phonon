#!/usr/bin/env python3
"""Read the 300 held-out-term sentences aloud. Enter starts, Enter stops; r = redo, s = skip, q = quit. Resumes."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
script = [json.loads(l) for l in open(os.path.join(HERE, "read_script.jsonl"))]
out_path = os.path.join(HERE, "recordings.jsonl"); wav_dir = os.path.join(HERE, "wav"); os.makedirs(wav_dir, exist_ok=True)
done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
todo = [r for r in script if r["id"] not in done]
print(f"{len(done)} done, {len(todo)} to go. Speak naturally, as if dictating to a coding agent.\n")
for i, r in enumerate(todo, 1):
    while True:
        print(f"[{len(done)+i}/{len(script)}] term: {r['term']}\n\n    {r['sentence']}\n")
        k = input("Enter = record, s = skip, q = quit: ").strip().lower()
        if k == "q": sys.exit(0)
        if k == "s": break
        wav = os.path.join(wav_dir, r["id"] + ".wav")
        p = subprocess.Popen(["sox", "-q", "-d", "-r", "16000", "-c", "1", "-b", "16", wav])
        t0 = time.time(); input("recording... Enter = stop: "); p.terminate(); p.wait(); dur = time.time() - t0
        k = input(f"{dur:.1f}s  Enter = keep, r = redo: ").strip().lower()
        if k == "r": continue
        with open(out_path, "a") as f:
            f.write(json.dumps({**r, "wav": wav, "seconds": round(dur, 2)}) + "\n")
        break
print("all done")
