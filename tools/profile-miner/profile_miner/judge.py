"""Stage 4b: keep/drop judge over the live (non-lexicon) top-N, a LoRA classifier on a local mlx model.

One prompt per term (term, spoken forms, source counts, two evidence lines); the score is the next-token logit
margin keep minus drop after the assistant header. Prompts are right-padded and batched: every model here is
causal, so padding after a sequence cannot change its last real position. Writes mined/judge.json
{term: {"margin", "keep"}}. The adapter recipe and PER_TERM prompt live in tools/profile-miner/judge/
(trained on Opus labels over this miner's own candidates); PER_TERM below must stay identical to that file.
"""
import json
import os
import sys
import time
from pathlib import Path

from .common import out_dir, read_json, write_json

PER_TERM = """You curate a personal dictation dictionary for a software developer. Keep a candidate only if the developer would say this exact word aloud and a general speech recognizer would likely get it wrong: names of their projects, repos, machines, people, companies, models, libraries, tools, file names, and jargon spelled wrong by a general model. Drop ordinary English words in any casing, abbreviations of everyday words, tokens of three letters or fewer that are not well-known products, paths, timestamps, numbers, version tags, generic code identifiers, and shell subcommands.
You get the term, what the speech recognizer produced when it was spoken (spoken_forms), counts per source (claude/codex/grok = the developer's chat messages to coding agents; repos = repo docs and file trees; code = source files containing it), and up to two lines the developer wrote containing it.
Answer with one word: keep or drop."""

DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"   # the model Phonon already ships for polish
DEFAULT_ADAPTER = Path(__file__).resolve().parent.parent / "judge" / "adapter-gemma-4-e2b"  # gitignored, see judge/
MAX_RSS_GB = float(os.environ.get("PHONON_JUDGE_MAX_RSS_GB", "4"))  # this runs on the user's laptop: abort, never swap


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)  # macOS reports bytes


def fmt_single(c):
    return json.dumps({"term": c["term"], "spoken_forms": c.get("spoken_forms"), "sources": c["sources"],
                       "lines": c.get("evidence", [])[:2]}, ensure_ascii=False)


def prompt_text(tok, c):
    msgs = [{"role": "system", "content": PER_TERM}, {"role": "user", "content": fmt_single(c)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def last_logits(model, arr, last):
    """Logits at each sequence's last real position only. The full [B, T, vocab] output is 8 GB for a batch of
    8 at 1,024 tokens on a 262k vocabulary (Gemma), so the head is applied to the gathered hidden rows."""
    import mlx.core as mx
    lm = getattr(model, "language_model", model)   # mlx multimodal wrappers (Gemma 4, Qwen3.5)
    hidden = lm.model(arr)
    rows = hidden[mx.arange(arr.shape[0]), last]
    if getattr(lm.args, "tie_word_embeddings", False):
        out = lm.model.embed_tokens.as_linear(rows)
    else:
        out = lm.lm_head(rows)
    cap = getattr(lm.args, "final_logit_softcapping", None)
    if cap:
        out = mx.tanh(out / cap) * cap
    return out


def margins(model, tok, cands, batch=8, max_len=1024):
    """keep-minus-drop logit margin per candidate, batched with right padding."""
    import mlx.core as mx
    keep = tok.encode("keep", add_special_tokens=False)[0]
    drop = tok.encode("drop", add_special_tokens=False)[0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    encs = [tok.encode(prompt_text(tok, c))[-max_len:] for c in cands]
    order = sorted(range(len(encs)), key=lambda i: len(encs[i]))  # similar lengths together: less padding
    out = [0.0] * len(cands)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        n = max(len(encs[i]) for i in idx)
        arr = [encs[i] + [pad] * (n - len(encs[i])) for i in idx]
        last = mx.array([len(encs[i]) - 1 for i in idx])
        rows = last_logits(model, mx.array(arr), last)
        m = (rows[:, keep] - rows[:, drop]).astype(mx.float32)
        mx.eval(m)
        for i, v in zip(idx, m.tolist()):
            out[i] = v
        if rss_gb() > MAX_RSS_GB:
            raise SystemExit(f"[judge] peak RSS {rss_gb():.1f} GB over the {MAX_RSS_GB:g} GB budget after {s + len(idx)} terms; stopping")
    return out


def run(top=300, adapter=None, model_id=None, batch=8):
    od = out_dir()
    adapter = adapter or os.environ.get("PHONON_JUDGE_ADAPTER") or (str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.is_dir() else None)
    model_id = model_id or os.environ.get("PHONON_JUDGE_MODEL") or DEFAULT_MODEL
    ranked = read_json(od / "mined" / "candidates.json")
    live = [m for m in ranked if not m.get("lexicon")][:top]
    from mlx_lm import load
    t0 = time.time()
    model, tok = load(model_id, adapter_path=adapter)
    if adapter:
        from mlx.utils import tree_flatten
        n_lora = sum("lora_" in k for k, _ in tree_flatten(model.parameters()))
        if n_lora == 0:
            raise SystemExit(f"[judge] adapter {adapter} loaded no LoRA parameters (key prefix mismatch?)")
    t_load = time.time() - t0
    t0 = time.time()
    ms = margins(model, tok, live, batch=batch)
    t_run = time.time() - t0
    out = {m["term"]: {"margin": round(v, 3), "keep": v > 0} for m, v in zip(live, ms)}
    write_json(od / "mined" / "judge.json", {"model": model_id, "adapter": adapter, "top": top, "terms": out})
    kept = sum(v["keep"] for v in out.values())
    print(f"[judge] {kept} keep of {len(out)} live terms; load {t_load:.0f}s, score {t_run:.0f}s "
          f"= {t_run / max(len(out), 1):.2f} s/term (batch {batch}); model {model_id}"
          f"{' + adapter' if adapter else ' (no adapter: zero-shot)'}; peak RSS {rss_gb():.1f} GB", file=sys.stderr)
    return out
