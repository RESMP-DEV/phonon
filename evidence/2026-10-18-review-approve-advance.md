# Receipt: review approval advances to next clip

**Date:** 2026-10-18
**Agent:** Codex

## Change

The label review UI now saves the current canonical text and verdict, then advances
to the next clip whenever `Approve` or the `a` keyboard shortcut is used. It does
not advance past the final row. Regenerating the same queue now reuses its prior
review identity when the existing JSON payload has the same queue path and row
IDs, so browser-local decisions survive a rebuild.

## Verification

In `/Users/kearm/phonon/.worktrees/train-adapt-v4`:

- `uvx ruff check ml/src/phonon/review_ui.py ml/tests/test_review_ui.py`: pass.
- `uvx ruff format --check ml/src/phonon/review_ui.py ml/tests/test_review_ui.py`: pass.
- `PYTHONPATH=ml/src uv run --project ml --frozen pytest ml/tests/test_review_ui.py -q`: 2 tests pass.
- Rebuilt the private 100-row Aqua/Cohere review bundle with 0 missing audio rows.

The private queue and generated review files remain under
`~/.local/share/phonon-research/aqua-current-v1/`; no transcript content is
committed.

## Non-claims

This verifies static UI generation and local review rebuilding. It does not add
web retrieval or claim anything about model quality.
