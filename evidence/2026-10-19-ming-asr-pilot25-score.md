# Ming ASR-tiny matched 25-row pilot score, 2026-10-19

## Scope

This receipt records the recovered and completed matched 25-row frozen-slice
pilot for `Ming-3.0-asr-tiny` on B550, the dependency repair that unblocked it,
and the aggregate scores for two context arms (no-context and no-leak hotwords)
plus the Aqua raw control on the identical rows. It supersedes the
"pilot crashed at import, zero rows" status recorded in
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

## Interpretation

- **Raw acoustic lane: effectively tied.** On fair WER, no-context Ming (0.0573)
  is statistically indistinguishable from Aqua raw (0.0556) on these 25 rows; the
  per-row split (2/14/9) leans slightly to Aqua but the total edit distance is
  33 versus 32. This is a 25-row subset, not the 500-row gate, so it is
  directional only. It already refutes any assumption that a specialized open
  ASR model automatically beats the Aqua raw acoustic lane on this short
  technical-dictation distribution.
- **The strict-LC gap is the judgment-lane signal, not an acoustic defect.** Ming
  fair WER (0.0573) is close to Aqua, but strict-LC WER diverges sharply (0.2761
  versus 0.0932, about 3x). Fair normalization removes casing, punctuation, and
  filler/formatting differences; strict-LC retains them. The gap therefore
  measures exactly the intended-text cleanup distance between Ming's verbatim
  acoustic output and Aqua's *accepted* text. This is direct empirical support
  for the architecture's judgment / correction-teacher framing: Ming's value is
  not lower raw WER, it is the correction lane that must close this formatting
  and preference distance.
- **No-leak hotwords did not help on this subset.** Injecting the top-10
  Aqua-raw-retrieved hotwords made fair WER worse (0.0920 versus 0.0573), changed
  only 6 of 25 rows, and was net-negative per row (2 better / 4 worse). Candidate
  explanations to test on the full slice: the top-10 retrieval injects
  near-miss terms that bias decoding on short clips; the 25-row subset is
  hotword-sparse; or the contiguous-suffix prompt format is not optimal. This is
  a negative result for the hotword arm at this size, recorded as such, not
  hidden.

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
| Repository gate | `just quality` at this PR head | exit 0, PASS |

## Non-claims

This is a **25-row subset**, not the registered 500-row gate; it does not promote
or reject Ming as an engine and does not establish a full-slice fair/strict WER.
It does not measure Aqua latency (the timing columns are Ming's). It does not
prove the judgment/correction teacher improves intended text; it only measures
the raw-acoustic and formatting distance that motivates that lane. It does not
qualify a daily-driver engine, prove streaming behavior, or authorize any
private-data upload to a rented provider. The hotword negative result is specific
to top-10 retrieval on 25 rows and must be re-tested on the full slice before any
conclusion about the hotword mechanism itself. No quantized Ming model was run;
this pilot used the BF16 checkpoint.
