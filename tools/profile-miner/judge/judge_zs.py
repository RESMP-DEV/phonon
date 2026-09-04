"""Zero-shot batch judge through a vLLM OpenAI endpoint (base Qwen3.5-4B, thinking off).
usage: python judge_zs.py judge_final.json out.json http://localhost:8398/v1 base [base|strict]"""
import json
import sys
import time
import urllib.request

from prompts import BASE, STRICT, fmt_batch_item

inp, out, endpoint, model = sys.argv[1:5]
variant = sys.argv[5] if len(sys.argv) > 5 else "base"
SYSTEM = STRICT if variant == "strict" else BASE
BATCH = 20
cands = json.load(open(inp))
results = {}
t0 = time.time()
for i in range(0, len(cands), BATCH):
    batch = cands[i:i + BATCH]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Candidates:\n" + "\n".join(fmt_batch_item(c) for c in batch)}]
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(endpoint + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
            s, e = txt.find("["), txt.rfind("]")
            for r in json.loads(txt[s:e + 1]):
                results[int(r["id"])] = bool(r["keep"])
            break
        except Exception as ex:
            print(f"batch {i} attempt {attempt}: {ex}", file=sys.stderr)
    if (i // BATCH) % 10 == 0:
        print(f"[zs-{variant}] {min(i + BATCH, len(cands))}/{len(cands)} {time.time() - t0:.0f}s", file=sys.stderr)
json.dump({str(k): v for k, v in sorted(results.items())}, open(out, "w"))
print(f"[zs-{variant}] {sum(results.values())} keep of {len(results)} judged ({len(cands)} in) in {time.time() - t0:.0f}s",
      file=sys.stderr)
