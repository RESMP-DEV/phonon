# Ming ASR-tiny matched slice scores, 2026-10-19

## Scope

This receipt records the recovered, repaired, and completed matched evaluation of
`Ming-3.0-asr-tiny` on B550: the dependency repair that unblocked it, a 25-row
pilot, and then the **full registered 500-row gate** for two context arms
(no-context and no-leak hotwords) plus the Aqua raw control on the identical
rows. The 500-row gate supersedes the 25-row pilot as the decisive result, and
it corrects the pilot's misleading impression. It also supersedes the "pilot
crashed at import, zero rows" status recorded in
`evidence/2026-10-19-ming-asr-survey.md`.

It is **aggregate-only by design**: no transcript text, no row IDs, no audio
paths, no dictionary/lexicon term values, and no per-row private content appear
here. The per-row `hyps.jsonl` and the 797-term lexicon remain on the B550 research
share (under the same `build/asr-teachers/` tree named in `docs/architecture.md`)
and were not copied into Git or sent to any rented provider. Only sanitized score JSON (WER/exactness/counts) was
transferred to the main machine's ignored `build/` and is hash-pinned below.

## Dependency repair

The first pilot pass crashed at import because the pinned B550 environment (the
`ming-asr` env: torch 2.11.0+cu130, transformers 5.12.1, torchaudio 2.11.0+cu130,
fla-core 0.5.2) lacked two runtime dependencies of the hotword retriever
`asr_teachers/vocab_retrieval.py`, which imports both at module load:

- `metaphone` (DoubleMetaphone)
- `rapidfuzz`

Neither is in the model's `requirements.txt`, so the original install did not
fetch them. They were installed into the same pinned env and the teacher module
was import-smoke-tested (`transcribe_ming_asr.py --help` import path) before any
run. This is an environment repair, not a model or protocol change; the pinned
torch/transformers versions were left untouched.

## Pilot execution

Both arms ran on B550 GPU 0 (`cuda:0`, one RTX 3090 Ti) inside detached tmux
sessions using the resumeable teacher `asr_teachers/transcribe_ming_asr.py`,
greedy decoding (`do_sample=False`), `max-new-tokens 256`,
`language-hint en`, BF16/SDPA. Audio is read with `soundfile` from the frozen
slice's FLAC clips (16 kHz). The frozen slice is the registered 500-row
evaluation slice (path recorded in `docs/architecture.md`); the first 25 rows
were selected with `--limit 25`.

| Arm | context-mode | Selected rows | Completed | Wall time | Peak GPU |
| --- | --- | ---:| ---:| ---:| ---:|
| no-context | `none` | 25 | 25 | ~68 s | ~18.8 GiB |
| no-leak hotwords | `hotwords` (top-10) | 25 | 25 | ~66 s | ~18.8 GiB |

The hotwords arm derives candidate terms **only** from the Aqua raw hypothesis
plus the local 797-term lexicon (`aqua-lexicon-v2.json`), never from the
accepted reference, per the no-leak contract. Mean clip duration is 7.84 s
(196.02 s of audio across 25 rows). Generation real-time factor was about 2.9
(Ming generation metadata).

## Matched scores (identical 25 rows, accepted reference as target)

Scored with the frozen `score_hyps.py` (jiwer + `whisper_normalizer` English
normalizer) in the B550 `salm-lora` scoring env. Lower WER is better.

| Lane | fair WER | strict-LC WER | fair exact |
| --- | ---:| ---:| ---:|
| Aqua raw (control) | 0.0556 | 0.0932 | 0.72 |
| Ming ASR-tiny, no-context | 0.0573 | 0.2761 | 0.56 |
| Ming ASR-tiny, no-leak hotwords | 0.0920 | 0.2614 | 0.48 |

Per-row win/tie/loss against the accepted reference, no-context Ming versus Aqua
raw (by edit distance): **2 Ming win / 14 tie / 9 Aqua win**; total edit distance
33 (Ming) versus 32 (Aqua) across the 25 rows. Hotwords versus no-context
(same rows): hotwords changed the output on **6 of 25** rows, and on the accepted
reference were **2 better / 19 tie / 4 worse**.

Aggregate score JSON SHA-256 (main-machine copies under ignored
`build/asr-teachers/ming-pilot25b/`):

| File | SHA-256 |
| --- | --- |
| `score-ming.json` | `b476130ed15c527a49d29752419ef98f6b23b54140e4ad9c4b67551c74e77b0d` |
| `score-aqua-raw.json` | `ae63d29a6efa24aa75a734b87639ad127b56b15096d095f7e8e2f3dc5569a688` |
| `score-ming-hotwords.json` | `4330b104ddefbbb89415780c53c52254c179d45acf39acf0f3a65b2ddb5df26a` |

Note on the timing columns: the Aqua-raw score is computed from the same
`hyps.jsonl` using its `raw_aqua` field, so its `mean_gen_s`/`rtf` columns are
inherited from Ming's generation metadata and are **not** an Aqua latency
measurement. Only the WER/exactness columns are comparable across lanes.

## Full registered 500-row gate (decisive result)

After the pilot, both arms were run over the complete registered 500-row frozen
slice (`--limit 0`), in parallel on the two idle RTX 3090 Ti GPUs: no-context on
`cuda:0`, no-leak hotwords on `cuda:1`. Both arms completed 500/500 rows
(3806.83 s of audio each, about 21 minutes per arm, roughly 18.8-19.0 GiB per
GPU, real-time factor about 2.9).

Scored with the same frozen scorer and the same accepted reference. The Aqua
raw control reproduces the previously registered baseline exactly
(fair WER 0.032912, strict-LC 0.055232), which validates the harness and the
row set against the documented frozen-slice baselines.

| Lane | fair WER | strict-LC WER | fair exact |
| --- | ---: | ---: | ---: |
| Aqua raw (control) | 0.0329 | 0.0552 | 0.674 |
| Ming ASR-tiny, no-context | 0.0901 | 0.2758 | 0.358 |
| Ming ASR-tiny, no-leak hotwords | 0.1051 | 0.2639 | 0.302 |

Per-row win/tie/loss against the accepted reference on all 500 rows:

| Comparison | Ming win | Tie | Aqua win | Total edits (Ming vs Aqua) |
| --- | ---: | ---: | ---: | --- |
| Ming no-context vs Aqua raw | 4 | 233 | 263 | 1040 vs 380 |
| Ming no-leak hotwords vs Aqua raw | 8 | 195 | 297 | 1213 vs 380 |

Hotwords versus no-context on the same 500 rows changed the output on **227**
rows and was net-negative: 77 better / 295 tie / 128 worse.

| File | SHA-256 |
| --- | --- |
| `ming-asr-tiny-full500-none-score-aqua-raw.json` | `b22b44f8e6e9ea3bf11689b4c6310c1e7e17cc2257427b8a5e513ead0d54c11d` |
| `ming-asr-tiny-full500-none-score-ming-none.json` | `8b9fa492da66e0ceb30c9e615abae4773dd88309626d5d329318002693287868` |
| `ming-asr-tiny-full500-hotwords-score-ming-hotwords.json` | `189df78d5bb4baea7eb1ece54068c389830edd8e8aab66b0835200af1963c5d6` |

**The 25-row pilot was misleading and the full gate corrects it.** The pilot
showed Ming no-context at fair WER 0.0573 against Aqua raw 0.0556, an apparent
near-tie. On the full 500 rows the same lane is 0.0901 against 0.0329, and the
per-row split is 4 win / 233 tie / 263 Aqua win with 1040 total edits against
380. The first 25 rows of the slice are not representative of the slice: Aqua
raw's advantage is concentrated in rows the pilot did not sample. Any conclusion
drawn from the pilot subset, including the "effectively tied" reading and the
"consistent with the judgment lane" interpretation, is superseded by the
registered gate.

**Gate verdict.** On the registered matched protocol, `Ming-3.0-asr-tiny` loses
decisively to the Aqua raw acoustic lane (roughly 2.7x the edit distance) and
the no-leak hotword context makes it worse, not better (0.1051 versus 0.0901).
Ming ASR-tiny is therefore **not promotable** as a Phonon raw acoustic engine,
and the hotword-context mechanism is **not** supported on this slice. This
directly contradicts the directional estimate in
`evidence/2026-10-19-ming-asr-survey.md` (which projected roughly 4.5-6.5% from
public-to-private degradation factors) and shows that cross-protocol
extrapolation from public benchmarks was not reliable for this model on this
distribution.

## Interpretation

- **Pilot reading (superseded by the full gate above).** On the 25-row pilot,
  no-context Ming (0.0573) showed a similar fair WER to Aqua raw (0.0556) and
  the per-row split was 2/14/9. The full 500-row gate overturns this: the pilot
  subset was unrepresentative. This bullet is retained only to record what the
  subset suggested and why it must not be cited.
- **Pilot reading (superseded by the full gate above).** On the pilot subset the
  strict-LC gap (0.2761 versus 0.0932) was read as *consistent with* a
  judgment-lane signal, since fair normalization removes casing, punctuation,
  and filler differences that strict-LC retains, without establishing that
  cause. On the full 500 rows the fair-WER gap is also large (0.0901 versus
  0.0329), so the gate does not isolate a formatting-only distance: the model
  also loses on the normalized acoustic lane. The judgment-lane hypothesis is
  therefore not supported by this evidence, and is retained only as an open
  architectural question for a different teacher, not as a finding about Ming.
- **No-leak hotwords did not help (confirmed on the full gate).** On the pilot
  subset, top-10 Aqua-raw-retrieved hotwords made fair WER worse (0.0920 versus
  0.0573) and were net-negative per row (2 better / 4 worse), with candidate
  explanations noted. The full gate confirms and sharpens this: hotwords are
  worse on fair WER (0.1051 versus 0.0901), changed 227 of 500 rows, and were
  77 better / 295 tie / 128 worse. The hotword-context mechanism is not
  supported for this model on this slice, and the retrieved-hotword lane should
  not be promoted on the strength of the no-leak design alone.

## Verification

Recorded UTC: 2026-10-19T23:05Z-23:15Z. Platforms: macOS main machine (control)
and B550 Linux/CUDA (execution). Source revision: this PR head; worktree dirty
state: false at commit. Private slice text, row IDs, audio paths, and lexicon
term values were not printed in this tracked receipt; only aggregate scores were
transferred off B550.

| Check | Sanitized command or procedure | Result |
| --- | --- | --- |
| Dependency repair | `pip install metaphone rapidfuzz` into the pinned `ming-asr` env; import smoke of `vocab_retrieval` and `transcribe_ming_asr` | exit 0; `doublemetaphone` and `rapidfuzz` import; teacher `--help` path loads |
| Audio readability | `soundfile.read` on a slice FLAC | exit 0; 16 kHz mono confirmed |
| No-context arm | detached tmux `transcribe_ming_asr.py --context-mode none --limit 25 --device cuda:0` | 25/25 rows, status complete, ~68 s |
| Hotwords arm | detached tmux `transcribe_ming_asr.py --context-mode hotwords --lexicon aqua-lexicon-v2.json --max-hotwords 10 --limit 25` | 25/25 rows, status complete, ~66 s |
| Scoring | `score_hyps.py <hyps> --hyp-field {hyp,raw_aqua} --ref-field ref` in the `salm-lora` scoring env | exit 0; aggregates above |
| Per-row comparison | jiwer `process_words` edit-distance win/tie/loss over identical rows | exit 0; counts above |
| Aggregate transfer | `scp` of the three `score-*.json` to main `build/` (no transcripts) | exit 0; SHA-256 recorded above |
| Full no-context arm | detached tmux `transcribe_ming_asr.py --context-mode none --limit 0 --device cuda:0` | 500/500 rows, status complete, ~21 min, ~18.8 GiB |
| Full hotwords arm | detached tmux `transcribe_ming_asr.py --context-mode hotwords --limit 0 --device cuda:1` | 500/500 rows, status complete, ~21 min, ~19.0 GiB |
| Full-slice scoring | `score_hyps.py` over both 500-row outputs, `--hyp-field {hyp,raw_aqua}` | exit 0; Aqua control reproduces the registered 0.032912 baseline; aggregates above |
| Full-slice per-row comparison | jiwer `process_words` edit-distance win/tie/loss over 500 identical rows | exit 0; counts above |
| Repository gate | `just quality` at this PR head | exit 0, PASS |

## Non-claims

The 500-row gate is the registered matched-slice comparison for the **raw
acoustic lane** only, under this exact protocol (greedy, no context or top-10
no-leak hotwords, English hint, BF16/SDPA, one 3090 Ti per arm). It rejects
Ming ASR-tiny as a raw acoustic engine on this slice and rejects the retrieved
hotword arm for this model on this slice. It does not measure Aqua latency (the
timing columns are Ming's generation metadata), does not qualify any model as a
daily-driver engine, does not prove streaming behavior, and does not authorize
any private-data upload to a rented provider. It does not evaluate the
judgment / correction-teacher hypothesis, which needs a teacher that is
competitive on the raw lane first; that lane remains an open architectural
question. No quantized Ming model was run; all numbers here are the BF16
checkpoint. The earlier 25-row pilot numbers are retained above for provenance
but are not a valid basis for any conclusion.
