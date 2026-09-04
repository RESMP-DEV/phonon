"""Shared judge prompts. BASE is the prompt used for the earlier 400-term Opus and 4B runs (kept verbatim so
labels are comparable); STRICT adds explicit drop rules and examples. PER_TERM is the classifier form."""
import json

BASE = """You curate a personal dictation dictionary for a software developer. The dictionary holds words that speech recognition gets wrong for this person: names of their projects, repos, machines, people, companies, models, libraries, tools, file names, and technical jargon they actually say out loud. It must NOT hold ordinary programming identifiers, shell words, log words, abbreviations of everyday words, version tags, or anything nobody would dictate as a word (env, INFO, v1, tok, diff, src, dir, e2e, w2).
For each candidate you get: the term as written, what the speech recognizer produced when the term was spoken (spoken_forms), how often it appears per source (claude/codex/grok = the developer's own chat messages to coding agents; repos = repo docs and file trees; code = number of source files containing it), and up to two lines the developer wrote containing it.
Answer with a JSON array only: [{"id": <id>, "keep": true|false}, ...] covering every id."""

STRICT = BASE + """

Rules, applied in order:
1. Keep only if the developer would say this exact word aloud AND a general speech recognizer would likely get it wrong (misspell, split, or substitute a common word). Project, repo, machine, person, company, model, library, tool, and file names qualify; so does jargon a general model spells wrong (cuBLAS, Ghostty, tracesmith, nvcc, Parakeet).
2. Drop ordinary English words in any casing or spelling (verified, Vocab, Evals, contrib, prio), abbreviations of everyday words (tok, diff, env, dir, repro, algo, util), and any alphabetic token of three letters or fewer unless it is a well-known product (npm, gcc, tmux keep; cta, dst, prv drop).
3. Drop file paths, timestamps, dates, numbers, sizes, coordinates, version tags, hashes, and identifiers with dots or brackets that are typed, not spoken (position.x, python3.14, 2026-04-30T01, state.mkgfx, 3x3, 10s).
4. Drop generic snake_case or camelCase identifiers and shell subcommands that only appear in code or command lines (npc_count, shfl_sync, git merge, tmux name), unless the developer's own chat lines show them dictating it as a name.
5. When a term is a real name but the lines show it only inside pasted logs or paths, still keep it if it is the developer's own project, machine, or person.
When unsure, drop."""

PER_TERM = """You curate a personal dictation dictionary for a software developer. Keep a candidate only if the developer would say this exact word aloud and a general speech recognizer would likely get it wrong: names of their projects, repos, machines, people, companies, models, libraries, tools, file names, and jargon spelled wrong by a general model. Drop ordinary English words in any casing, abbreviations of everyday words, tokens of three letters or fewer that are not well-known products, paths, timestamps, numbers, version tags, generic code identifiers, and shell subcommands.
You get the term, what the speech recognizer produced when it was spoken (spoken_forms), counts per source (claude/codex/grok = the developer's chat messages to coding agents; repos = repo docs and file trees; code = source files containing it), and up to two lines the developer wrote containing it.
Answer with one word: keep or drop."""


def fmt_batch_item(c):
    return json.dumps({"id": c["id"], "term": c["term"], "spoken_forms": c.get("spoken_forms"),
                       "sources": c["sources"], "lines": c["evidence"]}, ensure_ascii=False)


def fmt_single(c):
    return json.dumps({"term": c["term"], "spoken_forms": c.get("spoken_forms"), "sources": c["sources"],
                       "lines": c["evidence"]}, ensure_ascii=False)
