"""Generate spoken dictation utterances with LFM2.5-1.2B-Instruct. GPU 0, small footprint."""
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
from paths import CLEAN_PATH, DATA_ROOT, HF_HOME, LEXICON_PATH, MODEL_ID, TOPICS_PATH  # noqa: E402

RANKED_LEXICON = DATA_ROOT / "lexicon_ranked.jsonl"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SETTINGS = [
    "dictating a task to a coding agent about a repo you are working in",
    "talking to a teammate on Slack about a bug, GPU, or deploy",
    "speaking a git commit message out loud",
    "speaking a DEVLOG or AGENTS note to yourself",
    "asking yourself a debugging question out loud while looking at a stack trace",
]

SYSTEM = (
    "You are a software developer thinking out loud to a coworker or coding agent. "
    "Reply with one spoken utterance only. No quotes, no markdown, no lists. "
    "Never mention instructions or that you were asked to speak. Do not list identifiers."
)

PROMPT = """You are {setting}.
What's going on: {topic}
Talk about the work using {term1} and {term2} the way you'd actually say them.
If it fits, also say: {other_terms}
First person, contractions, optional uh or wait-no. 15-120 words. Speech only.
"""

META_RE = re.compile(
    r"\b(guidelines?|instructions?|word limit|word count|exact (?:phrases?|terms?|forms?)|"
    r"as an ai|i(?:'m| am) going to make sure|include the required|stay focused on the task|"
    r"output only|these instructions|repo terms?|mention these|I need to mention|"
    r"keep track of those|using these terms)\b",
    re.I,
)

MD_RE = re.compile(r"[#*`\[\]|]{2,}|```|^[-*]\s", re.M)
TERM_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|[_\-./]")


TOPIC_SKIP_RE = re.compile(
    r"copyright|licensed under|permitted to copy|all rights reserved|spdx-|"
    r"\bdco\b|apache license|gnu general public|mit license",
    re.I,
)


def heartbeat(last: float) -> float:
    if time.time() - last < 25 * 60:
        return last
    subprocess.run(
        ["overnight-compute", "heartbeat", "--agent", "grok-synth", "--ttl", "30m"],
        check=False,
    )
    return time.time()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_topics(path: Path) -> list[dict]:
    rows = []
    for row in load_jsonl(path):
        text = row.get("text") or ""
        src = (row.get("source_file") or "").lower()
        if TOPIC_SKIP_RE.search(text):
            continue
        if any(part in src for part in ("/license", "copying", "code_of_conduct")):
            continue
        rows.append(row)
    return rows


def load_done(path: Path) -> tuple[set[str], int]:
    seen: set[str] = set()
    n = 0
    if not path.exists():
        return seen, 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            t = norm_key(row.get("text") or row.get("target") or "")
            if t:
                seen.add(t)
    return seen, n


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def term_in_text(term: str, text: str) -> bool:
    if not term:
        return False
    low = text.lower()
    t = term.lower()
    if t in low:
        return True
    parts = [p for p in TERM_SPLIT.split(term) if len(p) > 1]
    if parts and all(p.lower() in low for p in parts):
        return True
    return False


def clean_utterance(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.S)
    text = re.sub(r"</?think>.*?</think>", "", text, flags=re.S)
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"^(utterance|output|here(?:'s| is)|sure)[:\-\s]+", "", text, flags=re.I)
    text = " ".join(text.split())
    return text


def acceptable(text: str, terms: list[str]) -> bool:
    if not text:
        return False
    n = len(text.split())
    if n < 15 or n > 120:
        return False
    if MD_RE.search(text):
        return False
    if META_RE.search(text):
        return False
    if "gpubox oomed" in text.lower() or "dropping batch to eight" in text.lower():
        return False
    if any(ch in text for ch in "{}<>"):
        return False
    return any(term_in_text(t, text) for t in terms)


KEEP_SHORT = {
    "gpu", "cpu", "wer", "tts", "asr", "api", "llm", "rlhf", "lora", "moe",
    "cu", "sm", "nv", "ptx", "sass", "itn", "pnc", "sft", "rl",
}


def term_weight(row: dict) -> float:
    t = row["term"]
    w = max(0.1, float(row.get("mangle_score") or row.get("rank_score") or 0) + 1.0)
    w *= 1.0 + (row.get("count") or 1) ** 0.3
    kind = row.get("kind") or ""
    if kind in {"product", "library", "cli_tool", "person", "machine"}:
        w *= 4.0
    if re.search(r"[a-z][A-Z]", t) or "_" in t:
        w *= 1.5
    if t.isupper() and 2 <= len(t) <= 3:
        w *= 0.2
    return w


def usable_lexicon(lex: list[dict]) -> tuple[list[dict], list[float]]:
    usable, weights = [], []
    for row in lex:
        t = row["term"]
        if len(t) <= 2 and t.lower() not in KEEP_SHORT:
            continue
        if t.lower() in {"software", "implied", "limited", "copyright"}:
            continue
        usable.append(row)
        weights.append(term_weight(row))
    if not usable:
        return lex, [1.0] * len(lex)
    return usable, weights


def pick_terms(usable: list[dict], cumw: list[float], rng: random.Random, n: int) -> list[str]:
    import bisect

    total = cumw[-1]
    out = []
    seen = set()
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        x = rng.random() * total
        i = min(bisect.bisect_left(cumw, x), len(usable) - 1)
        t = usable[i]["term"]
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def build_jobs(lex: list[dict], topics: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    usable, weights = usable_lexicon(lex)
    cumw: list[float] = []
    s = 0.0
    for w in weights:
        s += w
        cumw.append(s)
    jobs = []
    t0 = time.perf_counter()
    for i in range(n):
        n_terms = rng.randint(3, 8)
        terms = pick_terms(usable, cumw, rng, n_terms)
        if len(terms) < 2:
            terms = (terms + ["CUDA", "gpubox"])[:3]
        topic = topics[rng.randrange(len(topics))] if topics else {"text": "working on GPU kernels and ASR", "id": "fallback"}
        setting = SETTINGS[i % len(SETTINGS)]
        jobs.append(
            {
                "prompt_id": f"p{i:05d}",
                "terms": terms,
                "topic_id": topic.get("id"),
                "topic": (topic.get("text") or "")[:280],
                "setting": setting,
                "term1": terms[0],
                "term2": terms[1],
                "other_terms": ", ".join(terms[2:]),
            }
        )
    return jobs


def gpu_mem() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        return f"alloc={a:.2f}G reserved={r:.2f}G"
    except Exception:
        return "?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=14000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--target-min", type=int, default=8000)
    parser.add_argument("--target-max", type=int, default=12000)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    lex_path = RANKED_LEXICON if RANKED_LEXICON.exists() else LEXICON_PATH
    lex = load_jsonl(lex_path)
    lex = [r for r in lex if float(r.get("mangle_score") or r.get("rank_score") or 0) >= 1.0]
    if len(lex) < 50:
        lex = load_jsonl(lex_path)
    print(f"lexicon file={lex_path}", flush=True)
    topics = load_topics(TOPICS_PATH)
    print(f"lexicon usable={len(lex)} topics={len(topics)}", flush=True)
    hb = heartbeat(0.0)

    seen, n_done = load_done(CLEAN_PATH)
    if n_done >= args.target_min and not args.smoke:
        print(f"already have {n_done} utterances at {CLEAN_PATH}", flush=True)
        return 0

    n_jobs = 64 if args.smoke else args.n
    jobs = build_jobs(lex, topics, n_jobs, args.seed)
    start_i = n_done if not args.smoke else 0
    jobs = jobs[start_i:]
    print(f"jobs remaining={len(jobs)} already={n_done}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s {gpu_mem()}", flush=True)

    mode = "w" if args.smoke else "a"
    kept = n_done
    attempted = 0
    dropped = 0
    t_gen = time.perf_counter()
    bs = max(1, args.batch_size)
    with CLEAN_PATH.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(jobs), bs):
            if not args.smoke and kept >= args.target_max:
                break
            batch = jobs[start : start + bs]
            prompts = []
            for job in batch:
                user = PROMPT.format(
                    setting=job["setting"],
                    topic=job["topic"],
                    term1=job.get("term1") or job["terms"][0],
                    term2=job.get("term2") or (job["terms"][1] if len(job["terms"]) > 1 else job["terms"][0]),
                    other_terms=job.get("other_terms") or ", ".join(job["terms"][2:]),
                )
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ]
                chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(chat)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
            if args.temperature <= 0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update(do_sample=True, temperature=args.temperature, top_k=50)
            out = model.generate(**enc, **gen_kwargs)
            in_len = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for job, raw in zip(batch, texts, strict=True):
                attempted += 1
                text = clean_utterance(raw)
                if not acceptable(text, job["terms"]):
                    dropped += 1
                    continue
                key = norm_key(text)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                kept += 1
                row = {
                    "id": f"utt_{kept:05d}",
                    "text": text,
                    "prompt_id": job["prompt_id"],
                    "terms": job["terms"],
                    "topic_id": job["topic_id"],
                    "topic": job["topic"],
                    "setting": job["setting"],
                    "n_words": len(text.split()),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            hb = heartbeat(hb)
            if (start // bs) % 10 == 0 or start + bs >= len(jobs):
                elapsed = time.perf_counter() - t_gen
                rate = attempted / max(elapsed, 1e-6)
                print(
                    f"gen attempted={attempted} kept={kept} dropped={dropped} "
                    f"{rate:.2f} utt/s {gpu_mem()} elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
            if args.smoke and kept >= args.smoke:
                break
    print(
        f"done kept={kept} attempted={attempted} dropped={dropped} "
        f"wall_s={time.perf_counter() - t_gen:.1f} out={CLEAN_PATH}",
        flush=True,
    )
    if args.smoke:
        return 0
    if kept < args.target_min:
        print(f"WARNING kept={kept} < target_min={args.target_min}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
