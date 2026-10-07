# Receipt: Bacon Phonon prompt v1

Date: 2026-10-07
Agent: Codex
Prompt path: `prompts/bacon_phonon_v1.txt`
SHA-256: `593726da81277a485df4b3a0619a3a465a1b33c706e94b40840fe579d34e935c`

## Scope

Created a product-owned prompt template for the external correction layer
called Bacon. The template is generic and contains no personal transcript data.
It defines:

- Bacon's role as Phonon's local correction layer.
- Input placeholders for raw ASR, historical corrections, dictionary terms,
  and screen context.
- A correction policy that prioritizes meaning, ordering, technical entities,
  and developer spelling.
- Explicit instructions to use history only as evidence, never as phrases to
  copy.
- A conservative fail-closed rule: when uncertain, leave the raw wording
  unchanged.
- Output-only behavior with no explanations or alternatives.

## Non-claims

This is a prompt asset only. It is not wired into the product runtime, not
evaluated on personal data, and not yet paired with a specific Bacon model or
API. The template intentionally does not include real history examples or any
owner transcript text.
