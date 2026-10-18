# Receipt: Exa/Jina hardware seeding and Opus review-UI redesign

Date: 2026-10-18
Host: main machine (M4 Max)
Branch: `ml/portable-harness`

## Ownership and provider boundaries

Web retrieval for dataset seeding is now Exa plus local Jina/LM Studio rerank
embeddings only. `tools/web-seed/web_seed.py` records `zcode`, `openrouter`, and
`glm` as excluded providers in every payload and never imports an LLM route. UI
design was offloaded to Claude Opus 4.6 through the Antigravity CLI (`agy`), which
read only the committed generator source and never the private queue, audio, or
transcripts.

## Web-seeded hardware term

Phrase `RTX 6000 Pro Blackwell` resolved through `web-retrieval` (Exa search with
highlights, official-domain ranking) to the canonical
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, supported by NVIDIA product,
newsroom, and architecture-pdf sources. Evidence artifact:
`~/.local/share/phonon-research/web-seed/rtx-6000-pro-blackwell.json`.

The canonical term and four spoken forms were added to the local dictionary through
`phonon dictionary add`. The dictionary entry is local personal data and is not
committed.

## Review-UI redesign

Opus produced a nine-section design specification, then implemented the core
changes in `ml/src/phonon/review_ui.py`: persistent sidebar replaced with an
on-demand drawer, fixed command bar with progress and throughput, two-column
editor/diffs workspace, collapsed metadata, verdict controls next to the editor,
keyboard command mode with Escape, play/pause, seek, speed, teacher digit
selection, drawer toggle, and auto-advance to the next pending clip on every
verdict. The export schema and browser-local `edits` schema are unchanged.

Opus credits were exhausted mid-implementation, so the parent reviewed the diff,
fixed the two stale expectations in `ml/tests/test_review_ui.py`, and regenerated
the private review page.

## Commands and results

- `agy --model claude-opus-4-6-thinking --prompt ...` (design and implementation)
- `uvx ruff check tools/web-seed ml/src/phonon/review_ui.py ml/tests/test_review_ui.py`: pass
- `uvx ruff format --check ...`: pass
- `PYTHONPATH=ml/src uv run --project ml --frozen pytest ml/tests/test_review_ui.py -q`: 3 passed
- `PYTHONPATH=tools/web-seed uv run --no-project --python 3.12 --with pytest pytest -q tools/web-seed/tests`: 2 passed
- `node --check` on the generated private review JavaScript: pass
- `phonon report review-ui` regenerated the private page: 100 rows, 0 missing audio

## Non-claims

- The redesign is implemented and statically verified, not yet measured for
  reviewer throughput.
- The web-seed evidence covers one phrase and does not qualify a global dictionary
  policy.
- Antigravity Opus credits were exhausted; further Opus design work requires a
  quota reset.
