"""Spoken-register sentences for a term list, LFM2.5-1.2B-Instruct, batched, GPU."""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
TRAIN = Path("/data/phonon_corrector_v0/train.jsonl")

SYSTEM = (
    "You are a software developer dictating out loud to a coding agent or a teammate. "
    "Reply with the requested numbered sentences only. No markdown, no quotes, no commentary, "
    "no headings, never mention these instructions."
)

USER = """This is how I actually talk when I dictate:
{exemplars}

Write {n} different sentences in that same spoken style. Every sentence must contain {term} exactly once, spelled exactly like that, used naturally inside the sentence and never as a label at the front.
Rules: first person, contractions are fine, one complete sentence per line, 8 to 30 words each, about real engineering work, no lists or headings.
Output exactly {n} numbered lines and nothing else."""

PREFILL = [
    "1. I'm looking at {term} right now and",
    "1. Let me check whether {term}",
    "1. Yeah, the thing is {term}",
    "1. Can you see if {term}",
    "1. I think {term}",
    "1. We should probably test {term}",
    "1. Honestly {term}",
    "1. The last run showed {term}",
    "1. I keep hitting a wall with {term}",
    "1. Go ahead and wire {term}",
    "1. It turns out {term}",
    "1. My guess is {term}",
    "1. Before we ship, double check {term}",
    "1. I just rebuilt {term}",
    "1. So the plan is to move {term}",
    "1. Quick note to self, {term}",
    "1. Not sure why {term}",
    "1. Can you pull up {term}",
    "1. I'd rather keep {term}",
    "1. Looks like {term}",
]

DEMO_TERM = "cuBLASLt"
DEMO_ASSISTANT = """1. I'm pretty sure cuBLASLt is picking a bad heuristic for that shape, so let me pin the algo.
2. Can you check whether the fallback path still routes through cuBLASLt when the batch is uneven?
3. Yeah the numbers moved once I stopped fighting cuBLASLt and just used the grouped call."""

STRICT_USER = """You are a developer talking out loud about your work with {term}.
Say {n} different things, one per line, numbered. Each line mentions {term} once, partway through the line, never at the start.
Each line is one sentence, 8 to 30 words, first person, about real engineering work. Nothing else."""

MD_RE = re.compile(r"[#*`\[\]|]{2,}|```|^[-*]\s")
META_RE = re.compile(
    r"\b(guidelines?|instructions?|word (?:limit|count|placement)|numbered|as an ai|spoken style|"
    r"exactly like that|exactly in the middle|exact string|the term|sure,? here|output only|"
    r"sentences?|first person|contractions?)\b", re.I)
NUM_RE = re.compile(r"^\s*(?:\(?\d{1,2}[\.\)\:]|[-*\u2022])\s*")
BARE_NUM_RE = re.compile(r"^\s*\d{1,2}\s+(?=[A-Za-z])")

def heartbeat(last: float, agent: str) -> float:
    if time.time() - last < 20 * 60:
        return last
    subprocess.run(["overnight-compute", "heartbeat", "--agent", agent, "--ttl", "30m"], check=False)
    return time.time()


def load_exemplar_pool(seed: int) -> list[str]:
    pool = []
    with TRAIN.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("source", "")).startswith("wispr"):
                continue
            t = (row.get("target") or "").strip()
            n = len(t.split())
            if 10 <= n <= 35 and "\n" not in t:
                pool.append(t)
    random.Random(seed).shuffle(pool)
    return pool


def splice_case(sentence: str, term: str) -> str | None:
    """Return the sentence with the term in canonical casing, or None if absent."""
    idx = sentence.lower().find(term.lower())
    if idx < 0:
        return None
    return sentence[:idx] + term + sentence[idx + len(term):]


def clean(line: str) -> str:
    text = NUM_RE.sub("", line).strip()
    text = BARE_NUM_RE.sub("", text).strip()
    text = text.strip('"').strip("'").strip()
    text = " ".join(text.split())
    return text


def accept(text: str, term: str, relax: int = 0) -> str | None:
    if not text or MD_RE.search(text) or META_RE.search(text):
        return None
    if any(ch in text for ch in "{}<>"):
        return None
    if re.search(r"[.!?][\"')\]]?\s+\S", text):
        return None
    n = len(text.split())
    if n < 8 or n > 30:
        return None
    count = text.lower().count(term.lower())
    if count == 0 or (count != 1 and relax < 2):
        return None
    if relax < 1 and text.lower().startswith(term.lower()):
        return None
    return splice_case(text, term)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-term", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--agent", default="opus-term")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    terms = [json.loads(l) for l in args.terms.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        terms = terms[: args.limit]
    pool = load_exemplar_pool(args.seed)
    print(f"terms={len(terms)} exemplar_pool={len(pool)}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, local_files_only=True, device_map={"": 0})
    model.eval()
    print(f"model loaded {time.perf_counter()-t0:.1f}s", flush=True)

    kept: dict[str, list[str]] = {t["term"]: [] for t in terms}
    seen: dict[str, set[str]] = {t["term"]: set() for t in terms}
    hb = heartbeat(0.0, args.agent)
    t_start = time.perf_counter()
    attempted = 0

    for rnd in range(args.rounds):
        todo = [t for t in terms if len(kept[t["term"]]) < args.per_term]
        if not todo:
            break
        temp = 0.7 + 0.15 * rnd
        print(f"round={rnd} todo_terms={len(todo)} temp={temp:.2f}", flush=True)
        rng = random.Random(args.seed + 1000 * rnd)
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            prompts = []
            prefills = []
            for item in batch:
                need = args.per_term - len(kept[item["term"]])
                ask = max(need, 2) + 1
                ex = rng.sample(pool, 6)
                exemplars = "\n".join(f"- {e}" for e in ex)
                if rnd % 3 == 2:
                    messages = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": STRICT_USER.format(n=ask, term=item["term"])},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(exemplars=exemplars, n=ask, term=item["term"])},
                    ]
                chat = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                pre = PREFILL[(rng.randrange(len(PREFILL)))].format(term=item["term"])
                prefills.append(pre)
                prompts.append(chat + pre)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1400)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=temp, top_k=50, repetition_penalty=1.05,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            in_len = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            texts = [pre + t for pre, t in zip(prefills, texts)]
            for item, raw in zip(batch, texts):
                term = item["term"]
                for line in raw.splitlines():
                    attempted += 1
                    if len(kept[term]) >= args.per_term:
                        break
                    cand = accept(clean(line), term, relax=max(0, rnd - 2))
                    if cand is None:
                        continue
                    key = cand.lower()
                    if key in seen[term]:
                        continue
                    seen[term].add(key)
                    kept[term].append(cand)
            hb = heartbeat(hb, args.agent)
            done = start + len(batch)
            if (start // args.batch_size) % 5 == 0 or done >= len(todo):
                have = sum(len(v) for v in kept.values())
                el = time.perf_counter() - t_start
                print(f"round={rnd} {done}/{len(todo)} terms, sentences={have} "
                      f"elapsed_min={el/60:.1f} {have/max(el,1e-9):.2f} sent/s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    full = 0
    with args.out.open("w", encoding="utf-8") as h:
        for item in terms:
            sents = kept[item["term"]]
            if len(sents) >= args.per_term:
                full += 1
            for j, s in enumerate(sents):
                rec = {
                    "id": f"{item.get('term_id') or 'x'}_s{j}",
                    "term": item["term"],
                    "kind": item.get("kind"),
                    "sentence": s,
                    "n_words": len(s.split()),
                }
                for k in ("in_synth_v1", "in_synth_v2", "in_train", "rank_score", "mangle_score", "decile"):
                    if k in item:
                        rec[k] = item[k]
                h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    el = time.perf_counter() - t_start
    print(f"wrote {n} sentences for {len(terms)} terms ({full} with full quota), "
          f"attempted_lines={attempted} wall_min={el/60:.1f} -> {args.out}", flush=True)
    empty = [t["term"] for t in terms if not kept[t["term"]]]
    print(f"terms with zero sentences: {len(empty)} {empty[:30]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
