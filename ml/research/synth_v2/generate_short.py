"""Generate 2-25 word spoken utterances with LFM2.5-1.2B-Instruct. GPU 0, <10 GB."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    CLEAN_PATH,
    DATA_ROOT,
    HF_HOME,
    KIND_MIX,
    LEXICON_PATH,
    LOG_DIR,
    MODEL_ID,
    REAL_PAIRS,
    TOPICS_PATH,
)

V1_POLLUTION = Path("/home/user/phonon/research/synth_v1")
if str(V1_POLLUTION) not in sys.path:
    sys.path.append(str(V1_POLLUTION))
from pollution import is_third_party_path, is_user_authored_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STARTERS = {
    "question": [
        "Can you",
        "Could you",
        "Why is",
        "What happened to",
        "How do I",
        "Where did",
        "Did the",
        "Is gpubox",
        "Should we",
        "Who owns",
    ],
    "instruction": [
        "Rerun",
        "Check",
        "Open",
        "Post",
        "Pull",
        "Delete",
        "Run",
        "Bump",
        "Move",
        "Ping",
    ],
    "ack": [
        "Yep",
        "Yeah",
        "Okay",
        "Got it",
        "Sounds good",
        "Cool",
        "Alright",
        "Sure",
        "LGTM",
        "Fine",
    ],
    "slack_commit": [
        "Fix",
        "Add",
        "Wire",
        "Drop",
        "Land",
        "Note:",
        "FYI",
        "WIP",
        "Ship",
        "Clean up",
    ],
    "self_correction": [
        "Wait no",
        "Wait, I mean",
        "No wait",
        "Actually",
        "Scratch that",
        "Hang on",
        "I meant",
        "Wait, not",
        "Sorry, I mean",
        "Nah, I mean",
    ],
}

KIND_PROMPTS = {
    "question": "Output a question to a coding agent or teammate. It must be a question, ending with a question mark. Do not answer it.",
    "instruction": "Output a one-line spoken instruction to an agent or teammate. Imperative, not a question.",
    "ack": "Output a short spoken acknowledgement or reply about the work, then what to do next.",
    "slack_commit": "Output a short Slack line or commit subject spoken out loud.",
    "self_correction": "Output a spoken self-correction mid-sentence. Must include wait, I mean, actually, or scratch that.",
}

SYSTEM = (
    "You dictate one short line into a speech-to-text mic. "
    "Output that line only. No quotes, no markdown, no lists, no labels. "
    "Plain ASCII. Spoken register. Start with the given starter words."
)

PROMPT = """{kind_instruction}
Start with exactly: {starter}
Use the term {term1} naturally.
Speak {n_words} words, not fewer. Topic: {topic}
One spoken line only.
"""

BANNED_EXACT = {
    "yep, ship it.",
    "yep, ship it",
    "okay great, do that.",
    "okay great, do that",
    "okay, ship it.",
    "okay, got it.",
    "yep, got it.",
    "can you check why the sidecar ooms on gpubox?",
    "rerun the sweep with batch 16 and post the table",
    "wait no, i meant the v2 checkpoint, not v1",
}

META_RE = re.compile(
    r"\b(word limit|word count|exact (?:phrases?|terms?|forms?)|"
    r"as an ai|i(?:'m| am) going to make sure|include the required|stay focused on the task|"
    r"output only|these instructions|repo terms?|mention these|I need to mention|"
    r"keep track of those|using these terms|here is the utterance|next beat|"
    r"handling your request|topic noted|one spoken line|about \d+ words)\b",
    re.I,
)
MD_RE = re.compile(r"[#*`\[\]|]{2,}|```|^[-*]\s", re.M)
TERM_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|[_\-./]")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
Q_START = re.compile(
    r"^(can|could|would|should|will|what|why|how|who|where|which|is|are|do|does|did|"
    r"am|was|were|have|has|had|any|anyone)\b",
    re.I,
)
SELF_RE = re.compile(
    r"\b(wait(?:\s+no)?|no wait|i mean|actually|scratch that|not \w+,?\s+\w+|wait,)\b",
    re.I,
)
KEEP_SHORT = {
    "gpu", "cpu", "wer", "tts", "asr", "api", "llm", "rlhf", "lora", "moe",
    "cu", "sm", "nv", "ptx", "sass", "itn", "pnc", "sft", "rl",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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
    text = re.sub(r"^(utterance|output|here(?:'s| is)|sure|okay so)[:\-\s]+", "", text, flags=re.I)
    text = " ".join(text.split())
    return text


def kind_ok(kind: str, text: str) -> bool:
    n = len(text.split())
    if kind == "question":
        return ("?" in text) or bool(Q_START.search(text))
    if kind == "ack":
        return n <= 18
    if kind == "self_correction":
        return bool(SELF_RE.search(text))
    return True


def reject_reason(
    text: str,
    terms: list[str],
    kind: str,
    require_term: bool,
    min_words: int = 2,
    max_words: int = 25,
) -> str | None:
    if not text:
        return "empty"
    n = len(text.split())
    if n < min_words:
        return "short"
    if n > max_words:
        return "long"
    if NON_ASCII.search(text):
        return "non_ascii"
    if MD_RE.search(text):
        return "md"
    if META_RE.search(text):
        return "meta"
    if any(ch in text for ch in "{}<>"):
        return "braces"
    if norm_key(text) in BANNED_EXACT:
        return "banned"
    if not kind_ok(kind, text):
        return "kind"
    if require_term and terms and not any(term_in_text(t, text) for t in terms):
        return "term"
    return None


def acceptable(
    text: str,
    terms: list[str],
    kind: str,
    require_term: bool,
    min_words: int = 2,
    max_words: int = 25,
) -> bool:
    return reject_reason(text, terms, kind, require_term, min_words, max_words) is None


def term_weight(row: dict) -> float:
    t = row["term"]
    w = max(0.1, float(row.get("mangle_score") or row.get("rank_score") or 0) + 1.0)
    w *= 1.0 + (row.get("count") or 1) ** 0.3
    kind = row.get("kind") or ""
    if kind in {"product", "library", "cli_tool", "person", "machine"}:
        w *= 4.0
    if kind == "acronym":
        w *= 1.8
    if re.search(r"[a-z][A-Z]", t) or "_" in t:
        w *= 1.3
    if t.isupper() and 2 <= len(t) <= 3:
        w *= 0.2
    if len(t) > 24:
        w *= 0.25
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


def load_topics(path: Path) -> list[dict]:
    rows = []
    for row in load_jsonl(path):
        text = row.get("text") or ""
        src = row.get("source_file") or ""
        if is_third_party_path(src) and not is_user_authored_path(src):
            continue
        if not is_user_authored_path(src):
            continue
        if len(text) < 20:
            continue
        rows.append(row)
    return rows


def sample_kind(rng: random.Random) -> str:
    labels, weights = zip(*KIND_MIX)
    return rng.choices(list(labels), weights=list(weights), k=1)[0]


def sample_n_words(kind: str, empirical: list[int], rng: random.Random, lo: int = 2, hi: int = 25) -> int:
    if kind == "ack":
        n = rng.randint(2, 10)
    elif kind == "self_correction":
        n = rng.randint(8, 25)
    elif kind == "question":
        n = rng.choice(empirical) if empirical else rng.randint(4, 18)
    elif kind == "instruction":
        n = rng.randint(6, 22)
    elif kind == "slack_commit":
        n = rng.randint(5, 20)
    else:
        n = rng.choice(empirical) if empirical else rng.randint(4, 20)
    n = max(lo, min(hi, n))
    if n < lo:
        n = lo
    if lo >= 11:
        # second pass: force the long-short band regardless of kind
        n = rng.randint(lo, hi)
    return n


def real_wordcounts() -> list[int]:
    out = []
    for row in load_jsonl(REAL_PAIRS):
        n = len((row.get("target") or "").split())
        if 2 <= n <= 25:
            out.append(n)
    return out or list(range(2, 26))


def real_keys() -> set[str]:
    keys = set()
    for row in load_jsonl(REAL_PAIRS):
        t = norm_key(row.get("target") or "")
        if t:
            keys.add(t)
    return keys


def build_jobs(
    lex: list[dict],
    topics: list[dict],
    n: int,
    seed: int,
    empirical: list[int],
    min_words: int = 2,
    max_words: int = 25,
) -> list[dict]:
    rng = random.Random(seed)
    usable, weights = usable_lexicon(lex)
    cumw: list[float] = []
    s = 0.0
    for w in weights:
        s += w
        cumw.append(s)
    jobs = []
    for i in range(n):
        kind = sample_kind(rng)
        n_words = sample_n_words(kind, empirical, rng, min_words, max_words)
        n_terms = 1 if n_words <= 10 else rng.choice([1, 1, 2])
        if kind == "ack":
            n_terms = 1 if rng.random() < 0.45 else 0
        terms = pick_terms(usable, cumw, rng, max(n_terms, 1)) if n_terms else []
        topic = topics[rng.randrange(len(topics))] if topics else {"text": "working on GPU kernels", "id": "fallback"}
        topic_text = re.sub(r"\s+", " ", (topic.get("text") or ""))[:140]
        term1 = terms[0] if terms else "CUDA"
        term2 = terms[1] if len(terms) > 1 else ""
        starter = rng.choice(STARTERS[kind])
        # Acks still get a term in the prompt so they are not generic yep/okay.
        require_term = bool(terms) and kind in {"question", "instruction", "slack_commit", "self_correction"} and n_words >= 6
        jobs.append(
            {
                "prompt_id": f"s{i:05d}",
                "kind": kind,
                "n_words": n_words,
                "terms": terms,
                "topic_id": topic.get("id"),
                "topic": topic_text,
                "term1": term1,
                "term2": term2,
                "starter": starter,
                "require_term": require_term,
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
    parser.add_argument("--n", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--target-min", type=int, default=7000)
    parser.add_argument("--target-max", type=int, default=11000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=25)
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        global CLEAN_PATH
        CLEAN_PATH = DATA_ROOT / "tmp" / "smoke_short.jsonl"

    lex = load_jsonl(LEXICON_PATH)
    topics = load_topics(TOPICS_PATH)
    empirical = real_wordcounts()
    banned = real_keys()
    print(f"lexicon={len(lex)} user_topics={len(topics)} empirical_2_25={len(empirical)}", flush=True)

    seen, n_done = load_done(CLEAN_PATH)
    if n_done >= args.target_min and not args.smoke and args.min_words <= 2:
        print(f"already have {n_done} utterances at {CLEAN_PATH}", flush=True)
        return 0

    n_jobs = 96 if args.smoke else args.n
    jobs = build_jobs(lex, topics, n_jobs, args.seed, empirical, args.min_words, args.max_words)
    # same-seed resume only when generating the default 2-25 band
    start_i = n_done if (not args.smoke and args.min_words <= 2) else 0
    jobs = jobs[start_i:]
    print(f"jobs remaining={len(jobs)} already={n_done} min_words={args.min_words}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mem_frac = min(0.10, args.max_memory_gb / 96.0)
    try:
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    except Exception as exc:
        print(f"memory_fraction skip: {exc}", flush=True)

    print(f"loading {MODEL_ID} mem_frac={mem_frac:.3f} ...", flush=True)
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
    if args.smoke:
        kept = 0
        seen = set()
        n_done = 0
    else:
        kept = n_done
    attempted = 0
    dropped = 0
    drop_reason: dict[str, int] = {}
    t_gen = time.perf_counter()
    bs = max(1, args.batch_size)
    with CLEAN_PATH.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(jobs), bs):
            if not args.smoke and kept >= args.target_max:
                break
            batch = jobs[start : start + bs]
            prompts = []
            max_new = 8
            for job in batch:
                user = PROMPT.format(
                    kind_instruction=KIND_PROMPTS[job["kind"]],
                    starter=job.get("starter") or STARTERS[job["kind"]][0],
                    n_words=job["n_words"],
                    topic=job["topic"],
                    term1=job.get("term1") or "CUDA",
                )
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ]
                chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(chat)
                max_new = max(max_new, min(56, int(job["n_words"]) * 3 + 10))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_kwargs = dict(
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.08,
                do_sample=True,
                temperature=args.temperature,
                top_k=50,
                top_p=0.92,
            )
            out = model.generate(**enc, **gen_kwargs)
            in_len = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for job, raw in zip(batch, texts, strict=True):
                attempted += 1
                text = clean_utterance(raw)
                reason = reject_reason(
                    text,
                    job["terms"],
                    job["kind"],
                    job.get("require_term", False),
                    args.min_words,
                    args.max_words,
                )
                if args.smoke and attempted <= 40:
                    print(f"RAW kind={job['kind']} starter={job.get('starter')!r} n={len(text.split())} reason={reason} :: {text[:180]!r}", flush=True)
                if reason:
                    dropped += 1
                    drop_reason[reason] = drop_reason.get(reason, 0) + 1
                    continue
                key = norm_key(text)
                if key in seen or key in banned:
                    dropped += 1
                    drop_reason["dup"] = drop_reason.get("dup", 0) + 1
                    continue
                seen.add(key)
                kept += 1
                row = {
                    "id": f"short_{kept:05d}",
                    "text": text,
                    "prompt_id": job["prompt_id"],
                    "terms": job["terms"],
                    "topic_id": job["topic_id"],
                    "topic": job["topic"],
                    "setting": job["kind"],
                    "kind": job["kind"],
                    "n_words": len(text.split()),
                    "target_n_words": job["n_words"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (start // bs) % 8 == 0 or start + bs >= len(jobs):
                elapsed = time.perf_counter() - t_gen
                rate = attempted / max(elapsed, 1e-6)
                print(
                    f"gen attempted={attempted} kept={kept} dropped={dropped} "
                    f"{rate:.2f} utt/s {gpu_mem()} elapsed_min={elapsed / 60:.1f} "
                    f"drop={drop_reason}",
                    flush=True,
                )
            if args.smoke and (kept - n_done) >= args.smoke:
                break
    print(
        f"done kept={kept} attempted={attempted} dropped={dropped} "
        f"wall_s={time.perf_counter() - t_gen:.1f} out={CLEAN_PATH} drop={drop_reason}",
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
