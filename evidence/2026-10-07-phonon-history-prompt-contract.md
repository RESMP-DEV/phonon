# Receipt: Phonon history prompt contract and prompt-baked study

Date: 2026-10-07
Agent: Codex

## Correction made

An earlier turn created a prompt for an external correction layer named
"Bacon". That was a mishearing. The prompt asset, its receipt, and its
architecture entry were removed in commit `4f9d13d` rather than left in place.

## Phonon-owned prompts

`sidecar/phonon_prompts.py` now owns two dictation prompts:

| Prompt ID | SHA-256 | Purpose |
| --- | --- | --- |
| `prose_dictation_v1` | `3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5` | Static dictation contract, unchanged |
| `prose_dictation_history_v1` | `25ec25666727faea3f7d77c07df4df61b37fc4cdb9e18233f8c8f725028f566d` | History-grounded dictation contract |

The history-grounded prompt instructs the model to use the user's historical
corrections as evidence for recurring spellings, formatting, and correction
tendencies, never as phrases to copy; to prefer the history-supported spelling
only when the audio is ambiguous; to preserve meaning, ordering, negation,
technical entities, punctuation, and developer spelling; and to leave the
acoustically plausible wording unchanged when uncertain.

No personal transcript text is committed with either prompt.

## Prompt-baked training

Two pack layouts now exist for training on history:

| Builder | Layout | Rows |
| --- | --- | --- |
| `build_aqua_history_prompt_pack.py` | per-row dynamic history selection | 12,855 |
| `build_aqua_fixed_card_pack.py` | one byte-identical fixed card for every row | opt-in |

The dynamic pack at
`/home/kearm/salm-lora/build/phonon-history-pack-v2/pack` carries prompt
`prose_history_dictation_v2` with prompt SHA-256
`664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6`, manifest
SHA-256
`9378e4532e54c8816ea962a515ac833d60efbab0ec1e1d102b0296e81d490f08`,
history-contract SHA-256
`21eccc7415d93548a347f816aa8b71ef18709ae2b45ef0f77da9c2441d623053`, 13 Arrow
shards, and context 768.

## New Optuna study

Because the audio pack changed, the earlier learning-rate study no longer
applies. A new named study was started under the new contract:

- Study root: `/home/kearm/salm-lora/build/optuna/phonon-history-sft-lr-v1`
- Audio pack: `phonon-history-pack-v2/pack`
- Same fixed contract otherwise: rank 16, 4:1:1 mix, 768/768/1024 contexts,
  600 steps per trial, four trials, logarithmic LR search from `1e-5` to
  `1.5e-4`, vision-forgetting penalty.

## Non-claims

The new study was still running when this receipt was written; no trial result
is claimed here. The fixed-card builder was committed but its full pack was not
built in this turn.
