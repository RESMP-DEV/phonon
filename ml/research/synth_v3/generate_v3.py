"""Step 2a: generate spoken-register candidate utterances with LFM2.5-1.2B-Instruct.

Few-shot prompt: 8 real accepted targets per batch, stratified by length, so the model
copies the user's register instead of writing prose. Topics from synth_v0, 1-2 lexicon
terms per utterance. Writes every cleaned generation (with a pre_ok flag) so the register
filter can report an honest keep rate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register import read_jsonl  # noqa: E402

HF_HOME = Path("/data/hf")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
REAL = Path("/data/phonon_corrector_v0/train.jsonl")
TOPICS = Path("/data/phonon_synth_v0/topics.jsonl")
LEXICON = Path("/data/phonon_synth_v1/lexicon_ranked.jsonl")
OUT = Path("/data/phonon_synth_v3/candidates_v3.jsonl")

SYSTEM = (
    "You are the same person who dictated the example messages. "
    "You are talking out loud into a dictation app and you do not edit yourself. "
    "Reply with one spoken message only: no markdown, no headings, no bullet lists, "
    "no quotes, no preamble, no mention of these instructions."
)

PROMPT = """Messages this person dictated, verbatim:

{examples}

Now dictate one new message in exactly that voice, about this:
{topic}

Rules:
- You MUST actually say {terms} in the message. Say the name out loud, exactly as written.
- First person, spoken connectives (so, and, but, then, like, basically, actually, I mean).
- Ramble the way people actually talk out loud: leave in all the little words (and, so, that, the,
  to, of, going to, need to, want to, kind of, a bit of). Do not tighten it up or make it punchy.
- One topic only. No headings, no bullet lists, no colon-and-list constructions.
- Between 25 and 60 words. Prefer one long run-on sentence held together with commas and "and"
  over several short punchy ones.
- Speech only, nothing else.

Say {terms} and nothing about these rules."""

META_RE = re.compile(
    r"\b(guidelines?|instructions?|word limit|word count|as an ai|verbatim|"
    r"output only|these instructions|the examples? above|dictation app|"
    r"in that voice|here(?:'s| is) (?:a|the|my) (?:new )?(?:message|dictation))\b",
    re.I,
)
MD_RE = re.compile(r"[#*`\[\]|]{2,}|```|^\s*[-*•]\s|^\s*\d+[.)]\s", re.M)
TERM_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|[_\-./]")
TOPIC_SKIP_RE = re.compile(
    r"copyright|licensed under|permitted to copy|all rights reserved|spdx-|"
    r"\bdco\b|apache license|gnu general public|mit license",
    re.I,
)
KEEP_SHORT = {"gpu", "cpu", "wer", "tts", "asr", "api", "llm", "lora", "moe", "cu", "sm",
              "nv", "ptx", "sass", "itn", "pnc", "sft", "rl"}


def heartbeat(last: float, agent: str) -> float:
    if time.time() - last < 22 * 60:
        return last
    subprocess.run(["overnight-compute", "heartbeat", "--agent", agent, "--ttl", "30m"],
                   check=False, capture_output=True)
    return time.time()


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (text or "").strip().lower()))


def ngrams(text: str, n: int = 8) -> set[str]:
    ws = norm_key(text).split()
    return {" ".join(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))}


def clean(raw: str) -> str:
    t = raw.strip()
    t = re.sub(r"<\|im_end\|>.*", "", t, flags=re.S)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = t.strip().strip('"').strip("'")
    t = re.sub(r"^(message|utterance|output|here(?:'s| is)[^:\n]*|sure)\s*[:\-]\s*", "", t, flags=re.I)
    return " ".join(t.split())


def term_in(term: str, text: str) -> bool:
    low = text.lower()
    if term.lower() in low:
        return True
    parts = [p for p in TERM_SPLIT.split(term) if len(p) > 1]
    return bool(parts) and all(p.lower() in low for p in parts)


def pre_check(text: str, terms: list[str], ex_grams: set[str]) -> str:
    if not text:
        return "empty"
    n = len(text.split())
    if n < 5:
        return "too_short"
    if n > 70:
        return "too_long"
    if MD_RE.search(text):
        return "markdown"
    if META_RE.search(text):
        return "meta"
    if any(ch in text for ch in "{}<>"):
        return "braces"
    if not any(term_in(t, text) for t in terms):
        return "no_term"
    if ngrams(text) & ex_grams:
        return "copies_exemplar"
    return "ok"


def term_weight(row: dict) -> float:
    t = row["term"]
    w = max(0.1, float(row.get("mangle_score") or row.get("rank_score") or 0) + 1.0)
    w *= 1.0 + (row.get("count") or 1) ** 0.3
    if (row.get("kind") or "") in {"product", "library", "cli_tool", "person", "machine"}:
        w *= 4.0
    if re.search(r"[a-z][A-Z]", t) or "_" in t:
        w *= 1.5
    if t.isupper() and 2 <= len(t) <= 3:
        w *= 0.2
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--agent", default="opus-register")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    real = [r.get("target") or "" for r in read_jsonl(REAL)]
    real = [t for t in real if 4 <= len(t.split()) <= 70]
    bins: list[list[str]] = [[], [], [], []]
    for t in real:
        n = len(t.split())
        bins[0 if n < 10 else 1 if n < 22 else 2 if n < 40 else 3].append(t)
    print(f"real exemplars={len(real)} bins={[len(b) for b in bins]}", flush=True)
    real_keys = {norm_key(t) for t in real}

    topics = [r for r in read_jsonl(TOPICS)
              if r.get("text") and not TOPIC_SKIP_RE.search(r["text"])
              and not any(p in (r.get("source_file") or "").lower()
                          for p in ("/license", "copying", "code_of_conduct"))]
    lex = [r for r in read_jsonl(LEXICON)
           if (len(r["term"]) > 2 or r["term"].lower() in KEEP_SHORT)
           and len(r["term"]) <= 24
           and len([p for p in TERM_SPLIT.split(r["term"]) if p]) <= 3]
    weights = [term_weight(r) for r in lex]
    print(f"topics={len(topics)} lexicon={len(lex)}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True, device_map={"": "cuda:0"})
    model.eval()
    print(f"loaded {MODEL_ID} in {time.perf_counter() - t0:.1f}s", flush=True)

    n_total = args.smoke or args.n
    bs = max(1, args.batch_size)
    seen: set[str] = set(real_keys)
    kept = 0
    attempted = 0
    reasons: dict[str, int] = {}
    hb = heartbeat(0.0, args.agent)
    t_gen = time.perf_counter()

    with OUT.open("w", encoding="utf-8") as fh, torch.inference_mode():
        for start in range(0, n_total, bs):
            batch_n = min(bs, n_total - start)
            # 8 exemplars, stratified by length, shared by the whole batch
            ex = []
            for b in bins:
                if b:
                    ex += rng.sample(b, min(2, len(b)))
            rng.shuffle(ex)
            ex_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(ex))
            ex_grams: set[str] = set()
            for t in ex:
                ex_grams |= ngrams(t)

            jobs = []
            prompts = []
            for j in range(batch_n):
                terms = []
                guard = 0
                want = 1 if rng.random() < 0.7 else 2
                while len(terms) < want and guard < 40:
                    guard += 1
                    t = rng.choices(lex, weights=weights, k=1)[0]["term"]
                    if t not in terms:
                        terms.append(t)
                if not terms:
                    terms = ["CUDA"]
                topic = topics[rng.randrange(len(topics))]
                user = PROMPT.format(
                    examples=ex_block,
                    topic=(topic.get("text") or "")[:180],
                    terms=" and ".join(terms) if len(terms) > 1 else terms[0],
                )
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True))
                jobs.append({"terms": terms, "topic_id": topic.get("id"),
                             "topic": (topic.get("text") or "")[:180]})

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1400)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95, top_k=50,
                repetition_penalty=1.05, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            for job, raw in zip(jobs, texts, strict=True):
                attempted += 1
                text = clean(raw)
                reason = pre_check(text, job["terms"], ex_grams)
                if reason == "ok":
                    key = norm_key(text)
                    if key in seen:
                        reason = "duplicate"
                    else:
                        seen.add(key)
                reasons[reason] = reasons.get(reason, 0) + 1
                if reason == "ok":
                    kept += 1
                fh.write(json.dumps({
                    "id": f"cand_{attempted:06d}", "text": text, "pre_ok": reason == "ok",
                    "pre_reason": reason, "terms": job["terms"], "topic_id": job["topic_id"],
                    "n_words": len(text.split()),
                }, ensure_ascii=False) + "\n")
            fh.flush()
            hb = heartbeat(hb, args.agent)
            if (start // bs) % 5 == 0 or start + bs >= n_total:
                el = time.perf_counter() - t_gen
                print(f"attempted={attempted} pre_ok={kept} {attempted / max(el, 1e-6):.1f} utt/s "
                      f"elapsed_min={el / 60:.1f} eta_min={(n_total - attempted) / max(attempted / max(el, 1e-6), 1e-6) / 60:.1f}",
                      flush=True)
    el = time.perf_counter() - t_gen
    print(f"DONE attempted={attempted} pre_ok={kept} wall_min={el / 60:.1f} out={OUT}", flush=True)
    print("pre_reasons " + json.dumps(dict(sorted(reasons.items(), key=lambda kv: -kv[1]))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
