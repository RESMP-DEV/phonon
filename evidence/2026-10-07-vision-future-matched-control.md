# Receipt: fused Phonon history model and audio-only GRPO future control

Date: 2026-10-07
Host: B550

## Corrected fused standalone evaluation

The first fused-model score of `0.31552052658929497` fair WER was invalid because
the evaluator silently skipped PEFT `.base_layer.weight` keys. The invalid
hypotheses and score are preserved under
`/home/kearm/salm-lora/build/phonon-history-sft-2000-fused-eval-v1/failed-miskey/`.
After key normalization, 148 of 332 merged language tensors were applied and 766
audio-model keys remained unchanged.

Fused standalone model results:

| Split | Fair WER | Strict WER | Exact | Hypotheses | Score |
| --- | ---: | ---: | ---: | --- | --- |
| Selection 500 | 0.08548415035510133 | 0.11474067333939945 | 0.374 | `selection-hyps.jsonl` | `selection-score.json` |
| Future 750 | 0.11902256518180755 | 0.16334341626026297 | 0.228 | `future-hyps.jsonl` | `future-score.json` |

Vision evaluation on the fused model used 100 held-out screenshots:

| Condition | WER | Exact |
| --- | ---: | ---: |
| Real | 0.025369178341537298 | 0.750 |
| White-image control | 0.9750094661113214 | 0.000 |

This demonstrates retained pixel reading, not recitation.

## Audio-only GRPO future control

The first native control attempt failed on the future schema because
`transcribe_lfm25audio.py` hardcodes `ts`; a second attempt exposed the same
issue for `dur`. Both failures are preserved as
`native-salm-grpo-future-v1-keyerror` and
`native-salm-grpo-future-v2-keyerror-dur`. A normalized temporary slice added
only synthetic metadata fields and reran the same model.

Audio-only GRPO v1, 750-row future control:

| Fair WER | Strict WER | Exact | Hypotheses | Score |
| ---: | ---: | ---: | --- | --- |
| 0.1191394832222612 | 0.16630656213346504 | 0.22666666666666666 | `hyps.jsonl` | `score.json` |

## Result

The fused Phonon history model scored 0.11902256518180755 versus 0.1191394832222612
for audio-only GRPO, a 0.0001169 fair-WER advantage. The difference is small and
does not establish meaningful superiority, but the fused model is clearly not
worse on the untouched future split and retains vision.

