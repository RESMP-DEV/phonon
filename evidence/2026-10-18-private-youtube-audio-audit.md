# Private YouTube audio audit, 2026-10-18

## Scope

This receipt records a private-Hugging-Face inventory and permission audit of
the non-owner, non-Aqua technical YouTube mirrors. No data was uploaded, no
training or ASR pass was launched, and no Deepgram quota was spent. Aggregate
outputs stay in the local research share; this tracked receipt deliberately
omits row IDs, source URLs, source paths, speaker identities, terms, notes, and
all transcript-shaped text.

## Pinned local integrity audit

Two highest-value private mirrors were downloaded with the authenticated native
`hf` CLI to `~/.local/share/phonon-research/hf-private-datasets/`:

- `EVA-UNIT-01/youtube_technical_v2_gpumode_latest_20260525b` at revision
  `b0bbdb9d5dad89286dc70d0cb3183fdb9bfc064f`
- `EVA-UNIT-01/youtube_technical_consensus_extended_v0` at revision
  `412fdeec4bf119297a66e66ce2775b5af8b18b6f`

The new privacy-safe CLI verified every embedded byte:

- Rows observed: 4,710
- Unique row IDs: 4,452
- Focused-set cross-dataset duplicate IDs or audio hashes: 0
- Audio hash mismatches: 0
- Metadata-versus-container sample-rate mismatches: 0
- Invalid WAVs: 0
- Duration mismatches: 0
- Clip-interval mismatches: 0
- Noncanonical streams: 0
- Embedded streams: 4,710 x mono PCM16 at 16 kHz
- Aggregate report:
  `~/.local/share/phonon-research/hf-private-datasets/private-audio-audit-v1.json`

The GPU Mode latest snapshot has 3,010 unique rows and 90,115.013 seconds
(25.03 hours), but no nonempty reference labels and no teacher transcripts. The
consensus extended snapshot exposes 1,700 row observations over 1,442 unique
IDs; its 258 extra observations are duplicate IDs and audio hashes. It has
43,217.138 unique-audio seconds (12.01 hours), nonempty labels on every
observation, and seven teacher transcripts on every observation. All duplicate
metadata was equal, but split membership overlaps across `auto_dev`, `hard`,
and `speed`.

Every one of these 4,710 rows is explicitly marked
`train_allowed=false,eval_allowed=false`. Therefore the focused combined set
has 37.04 hours of technically relevant real audio, but zero rows are currently
usable for training. The latest snapshot is useful for teacher-disagreement
mining; the consensus snapshot is useful for comparison against its existing
seven teachers. Neither may be promoted without a separate permission and
evaluation-contamination decision.

## Broad private-set permission probe

A metadata-only remote Parquet probe read permission, identity-count,
duration, split, and label-status columns for four more private YouTube sets.
It did not download audio or read transcript text:

- `youtube_dev_tutorials_real_audio_v0`: 14,910 rows, 123.62 hours, 164 leakage groups
- `youtube_freecodecamp_real_dev_courses_v0`: 20,787 rows, 173.09 hours, 38 leakage groups
- `youtube_technical_v2_gpumode_seed_20260525`: 534 rows, 4.44 hours, 3 leakage groups
- `youtube_technical_v2_gpumode_wide_20260525a`: 1,459 rows, 12.14 hours, 8 leakage groups

Across these 37,690 rows, every row is also marked
`train_allowed=false,eval_allowed=false`. The sanitized aggregate is
`~/.local/share/phonon-research/hf-private-datasets/remote-permission-audit-v1.json`.
This broad probe did not verify embedded-audio hashes or WAV containers and is
not an integrity claim.

## Verification

The first implementation round passed five focused tests and the aggregate
audio audit at the then-current dirty tree. Review probes then found and repaired
the repository-root miscalculation, metadata/container sample-rate disagreement,
duplicate-audio eligibility, arbitrary provenance fields, dataset/file basename
disclosure, output/input aliasing, non-finite durations, and corrupt-Parquet
failure handling. Follow-up reviews added coverage for nested teacher records
and canonical nullable whole-file clip intervals. The repaired focused tool suite has
eleven tests, and the
aggregate command again returned `PASS` over all 4,710 embedded audio rows.
The corrupt-Parquet probe returned exit code 2 and wrote only an
`ArrowInvalid` failure marker, not a traceback or private path. The final common
gate passed `just quality`: repository contracts, diff hygiene,
Ruff lint/format, all repository Python tests, and shell checks at a clean
branch head. The machine receipt records the exact validated head at
`build/reports/quality/receipt.json`.

## Non-claims

This work does not prove that every private HF dataset in the account was
enumerated, does not verify audio integrity in the four remotely probed broad
sets, and does not authorize any row for training. It does not compare source
groups against every frozen evaluation manifest because the relevant original
gate queues were not present on B550 at probe time. No model, WER, latency, or
daily-driver readiness claim is made.
