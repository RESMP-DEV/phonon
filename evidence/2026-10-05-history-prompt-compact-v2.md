# Receipt: Aqua compact-history prompt pack v2

Date: 2026-10-05
Agent: Codex
Source revision: `c7d4eec8213761b5acf4b0ac452d5c0636877606`
Host: B550 (`/home/kearm/phonon`)
Output: `/home/kearm/salm-lora/build/history-prompt-compact-v2/pack`

## Contract

- Prompt ID: `prose_history_dictation_v2`
- Prompt schema SHA-256:
  `664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6`
- Retrieval selector and candidate pool are unchanged from v1:
  `idf-cosine-v1`, two distinct raw-to-corrected examples.
- The rendered prompt preserves `prose_dictation_v1` exactly as its prefix and
  appends compact `<example>` pairs plus an explicit anti-copy instruction.
- The current target remains excluded by audio ID and its corrected text is not
  used for retrieval or rendering.

Local and B550 verification:

```text
ruff: passed
focused history/train tests: 7 passed
git diff --check: passed
```

## Pack receipt

- Started: `2026-10-05T16:36:05-07:00`
- Completed: `2026-10-05T16:51:13-07:00`
- Wall time: 908 seconds
- Manifest SHA-256:
  `9378e4532e54c8816ea962a515ac833d60efbab0ec1e1d102b0296e81d490f08`
- History-contract SHA-256:
  `21eccc7415d93548a347f816aa8b71ef18709ae2b45ef0f77da9c2441d623053`
- Contract targets: 12,855
- Dynamic prompt hashes: 12,855 unique values
- Packed Arrow examples: 12,855
- Arrow shards: 13
- Pack size: `6113806478` bytes
- Per-file checksum manifest SHA-256:
  `82a82b0257aaa9e3f27cc407acd2d349f2e58d8906c68a88fcd902b602444673`

All 12,855 targets passed target self-exclusion and the two-distinct-history
invariant. Unlike XML-history v1, no target overflowed context 768.

## Non-claims

- The pack receipt itself proves structure, not model quality.
- Aqua audio and transcript contents remain personal, B550-controlled data.

## Matched 4,000-step pilot

- Training root:
  `/home/kearm/salm-lora/build/history-compact-training-v1/aqua-4000-compact-v1`
- Source: trial-0100 public checkpoint SHA-256
  `511c6f1ecae7281c039c63fda52a5cd83b692b1e09909506ad0fbfadf0d9bb20`
- Additional Aqua steps: 4,000
- Rank/context: 32/768
- Wall time: 613.3 seconds
- Adapter SHA-256:
  `16ab70ac07c49749835a7613e888d58a9f1b45c04ece23f87a9799921e5d58d8`
- Offline W&B run: `history-compact-aqua-4000-v1`

Reference-kernel 500-row result:

| Adapter | Fair WER | Strict WER | Exact |
| --- | ---: | ---: | ---: |
| Static prose, 4,000 steps | 0.11761649055950113 | 0.154049135577798 | 0.316 |
| XML history v1, 4,000 steps | 0.17218084184999133 | 0.20937215650591448 | 0.278 |
| Compact history v2, 4,000 steps | 0.1627403429759224 | 0.2002729754322111 | 0.280 |

Hypothesis SHA-256:
`3299a7b762fb76da09db549d9047d762eb74df0ffb9de05e485b1e25a0eeb608`.
Score SHA-256:
`9df25ac7fd5c41ee2191ab1370b91649e41e891f74792d2941a3755bd2e756e7`.
Paired static comparison: 85 rows improved, 275 tied, and 140 worsened;
median row delta remained zero.

Unchanged trial-0100 controls under the same dynamic evaluators were also
scored. XML v1 scored 0.8041746059241296 fair WER, 0.8413102820746133 strict
WER, and 0.128 exact. Compact v2 scored 0.7024943703447081 fair WER,
0.7403093721565059 strict WER, and 0.128 exact. These controls show that both
history prompt layouts are major distribution shifts and that 4,000 steps
substantially recover, but do not yet reach, static-prompt quality.

## Full compact-history epoch

- Training root:
  `/home/kearm/salm-lora/build/history-compact-full-epoch-v1/aqua-full-epoch-v1`
- Additional Aqua steps: 12,855
- Total steps from public resume: 22,855
- Wall time: 1,829.2 seconds
- Adapter SHA-256:
  `44a379ee1762461077319fa28173d2da782ad1ea5f29b0303c8fde3218e033a4`
- Offline W&B run: `history-compact-full-epoch-v1`

| Adapter | Split | Fair WER | Strict WER | Exact |
| --- | --- | ---: | ---: | ---: |
| Static full epoch | Selection 500 | 0.11077429412783649 | 0.14285714285714285 | 0.352 |
| Compact-history full epoch | Selection 500 | 0.11345920665165425 | 0.14522292993630573 | 0.350 |
| Static full epoch | Future 750 | 0.14521220624342335 | 0.19340700043212544 | 0.18666666666666668 |
| Compact-history full epoch | Future 750 | 0.14445223898047468 | 0.19297487499228347 | 0.20533333333333334 |

Selection hypothesis SHA-256:
`5f1e3d13b5125c9d023ccf41dd84e9a413806125522a5c213836ebf26d976ccb`.
Future hypothesis SHA-256:
`c95b82f01663e8b2e491e42943e4d933db5508e59b7ad471619704a8b98a7a58`.
The canonical remote receipt is
`/home/kearm/salm-lora/build/history-compact-full-epoch-v1/aqua-full-epoch-v1/receipt.json`.

The full history adapter wins future fair/strict WER and exact over static full
epoch, while remaining slightly worse on the repeated selection split. Against
static full, it wins 104 rows, loses 85, and ties 311. It still does not beat
Aqua raw and is not promoted to the product engine. An aborted future wrapper
that would have overwritten the selection score is preserved as
`FUTURE-LAUNCH-V1-FAILED`; it was stopped before scoring and resumed with a
separate `future-score.json`.

## Adapter soup controls

Two equal-weight soups were materialized with matched tensor keys, shapes, and
dtypes:

- static full plus history full, SHA-256
  `4dbec79b772d9ca97f60d6b763e683e88e23a1409b1f95e6714f8e3a03b54367`
- static 8k plus static full plus history full, SHA-256
  `71803dfabbf777979ddcb27358bc88cfcf90dbab65dc7fda36088fb8ed8404fa`

The two-way soup scored 0.13554477741209076 fair WER,
0.16560509554140126 strict WER, and 0.356 exact on selection 500. Its exact
score improved, but its WER was much worse than either full checkpoint, so it
was not sent to future evaluation. The three-way soup has a bounded selection
control scored 0.208470465962238 fair WER, 0.24340309372156507 strict WER, and
0.310 exact. Its hypothesis SHA-256 is
`d55d7bbf7cee0383262ae6d34fd6aa71b7efcbb472f6e1c0d43f071f22203205`; neither
soup is promoted.

## Optimized-runtime parity

On 2026-10-06, the full compact-history adapter was re-evaluated with Torch
`2.14.0+cu130`, FlashAttention `2.8.3.post1`, and causal-conv1d `1.7.0`. It
matched every reference aggregate metric exactly:

| Split | Fair WER | Strict WER | Exact | Mean generation |
| --- | ---: | ---: | ---: | ---: |
| Selection 500 | 0.11345920665165425 | 0.14522292993630573 | 0.350 | 0.9088 s |
| Future 750 | 0.14445223898047468 | 0.19297487499228347 | 0.20533333333333334 | 0.9295 s |

Selection hypothesis SHA-256:
`4db68a869286f3bcbee7ca55ba1ac30daaf457fed117a5ac24949be8d653ea97`.
Future hypothesis SHA-256:
`b204249805221730d80902431d97405452a4f3b7278aab73d570e61c5b5cb998`.
The canonical receipt is
`/home/kearm/salm-lora/build/history-compact-full-epoch-v1/optimized-runtime-v1/receipt.json`.
These generation timings are research-evaluator timings, not resident product
protocol, macOS end-to-end, or streaming latency claims.

## No-oracle self-retrieval control

The oracle-free evaluator uses one resident reverse model for two passes:

1. generate a draft with `prose_dictation_v1`;
2. use that draft as the only history-retrieval query;
3. retrieve two historical examples and generate the final transcript with
   `prose_history_dictation_v2`.

The target row's Aqua raw text and accepted text are not passed to retrieval.
Every output records `oracle_query: false`.

| Stage | Fair WER | Strict WER | Exact |
| --- | ---: | ---: | ---: |
| First-pass draft | 0.1268837692707431 | 0.16041856232939036 | 0.322 |
| Second-pass self-retrieval final | 0.123592586177031 | 0.15668789808917197 | 0.324 |
| Oracle-query history final | 0.11345920665165425 | 0.14522292993630573 | 0.350 |

The final hypothesis SHA-256 is
`b0ec7f04d46a52d3d525346f5f4693e0ed62b36bde1a490486e2ddaddd998edd`; the final
score SHA-256 is
`f71c2f819e0886e963d010136023a1ae4ed81a25152346c3035c384fda0a3988`. Mean
draft, retrieval, final, and total times were 0.8963 s, 0.0617 s, 0.9031 s,
and 1.8611 s. A peer two-GPU Qwen calibration was active, so those timings are
not an exclusive-load latency receipt. The canonical receipt is
`/home/kearm/salm-lora/build/history-compact-full-epoch-v1/self-retrieval-v2/receipt.json`.
The first launch failed before generation on an invalid aggregate `.eval()` call
and is preserved under `self-retrieval-v1-launch-failed`.
