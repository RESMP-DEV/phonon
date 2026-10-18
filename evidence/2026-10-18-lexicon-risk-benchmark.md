# Receipt: technical-lexicon difficulty benchmark

Date: 2026-10-18
Hosts: main machine and B550
Branch/worktree: `work/lexicon-local-sources`
Base revision: `2cce0af6f41dd4ab284c1dccd4afe33c5847a1b6` (dirty during receipted development)

## Question

Create benchmark triage data from obscure but useful technical terms beyond the Aqua dictionary/replacement surface, using local toolbelt and system inventories as well as the existing public technical lexicon.

## Method and privacy boundary

The shipped baseline uses `tools/profile-miner/lexicon` and its precomputed two-voice Kokoro-to-Parakeet oracle. Categories are public labels covering PyPI, crates.io, Homebrew, Hugging Face, and curated GPU/CUDA/developer terms.

`tools/lexicon-benchmark/collect_local.py` inventories toolbelt entries with path fields, executable `PATH` commands, macOS manual-page command/concept names in sections 1/4/5/6/7/8, Homebrew tap leaves, native `gh extension list` and editor extensions, and package locks only when they identify the exact public crates.io, npmjs.org, or PyPI registry. It excludes editable/path packages, private registries, linked workspaces, npm publisher scopes, generic English, script filenames, and already-shipped terms. It does not read Aqua settings, dictation audio/transcripts, chat logs, source text, or the Phonon corpus.

Aggregate reports contain counts and projected manifest fields only. Ranked terms, spoken forms, and stratified review slices remain local.

## Baseline result

Over 41,948 measured public entries:

| Worst diff class | Terms |
| --- | ---: |
| same | 1,039 |
| case | 4,185 |
| format | 11,021 |
| phonetic | 25,703 |

Using `format + phonetic` as the triage-risk set gives 36,724 terms, 87.55%. All shipped rows have two non-null voices. 975 terms are multi-label; category columns are label assignments, not a partition of the terms.

Top category assignments by risky share (multi-label):

| Category | Assignments | Risky share |
| --- | ---: | ---: |
| crate | 2,687 | 90.92% |
| pypi | 14,081 | 90.78% |
| hf-model | 9,004 | 90.69% |
| cask | 6,967 | 89.61% |
| hf-org | 1,865 | 88.85% |
| brew | 7,358 | 77.47% |

The shipped-only v3 aggregate report is local at
`~/.local/share/phonon-research/lexicon-benchmark/aggregate-report-v3.json`.
The final reviewed v6 report and term exports remain under
`~/.local/share/phonon-research/lexicon-benchmark/local-fixed-v2/`.

## Local inventory collection

The reviewed collector run produced 6,170 unique local candidates:

| Source | Kept | Seen |
| --- | ---: | ---: |
| toolbelt | 13 | 28 |
| path-command | 2,677 | 3,375 |
| cargo-local | 792 | 123,585 |
| manpage | 4,006 | 7,151 |
| npm-local | 269 | 1,079 |
| brew-tap | 5 | 14 |
| pypi-local | 0 | 44 |

Terms can belong to multiple sources. `collection_failures` was empty. The 44 public PyPI observations were all already present in the shipped lexicon, as were four Cargo observations. The local terms SHA-256 is `ab638411e8020c2c2da8abb61d9614a7878fa34b7153815439ed623a0b38f89c`. Candidate, lexicon, oracle, and ranked outputs remain under `~/.local/share/phonon-research/lexicon-benchmark/local-fixed-v2/`.

The completed two-voice oracle classified those 6,170 local terms as:

| Worst diff class | Local terms |
| --- | ---: |
| same | 63 |
| case | 588 |
| format | 2,046 |
| phonetic | 3,473 |

The local risky subset is 5,519 terms (89.45%). By multi-label source:

| Source | Assignments | Risky share |
| --- | ---: | ---: |
| npm-local | 269 | 94.05% |
| toolbelt | 13 | 92.31% |
| cargo-local | 792 | 91.41% |
| manpage | 4,006 | 88.44% |
| path-command | 2,677 | 84.76% |
| brew-tap | 5 | 80.00% |

## B550 oracle run

B550 initially had GPU 0 occupied by the qwen LR study and GPU 1 idle apart from its 4.4 GiB embedding service. The first CUDA extra-index environment failed on a PyPI NVIDIA wheel timeout; a standard-PyPI oracle environment then imported torch/torchaudio/NumPy/soundfile/NeMo/Kokoro/misaki successfully.

The first xet model transfer degraded to roughly 66 KiB/s. Direct HTTP also decayed after 70–260 MiB. Segmented aria2 downloads were used instead and checksum-seeded into the B550 Hugging Face cache:

- Kokoro 82M `kokoro-v1_0.pth`: 327,212,226 bytes, SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`.
- Parakeet TDT `parakeet-tdt-0.6b-v2.nemo`: 2,472,222,720 bytes, SHA-256 `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.

Offline smoke checks loaded Kokoro on CUDA and restored Parakeet from the exact cached NeMo snapshot on CUDA. The initial 5,025-term run was pinned to GPU 1 with `HF_HUB_OFFLINE=1` and checkpointed as JSONL. Review recovered multi-name manual rows, inline uv registry metadata, and Cargo.lock TOML parsing. The resulting 360- and 788-term supplements ran on B550 CPU with the same models after unrelated H3 and training work occupied both GPUs.

Post-run validation reported 6,170 required terms, zero missing required terms, 6,170 two-voice rows, and three obsolete cached rows from terms removed by the reviewed collector. The final combined cache SHA-256 is `71b3f3cea774b2232db02f769368018020c3893179b003f778f3caba4cea7e41`.

The merged v6 report contains 48,118 complete two-voice rows: 41,948 shipped and 6,170 local. Its diff classes are 1,102 same, 4,773 case, 13,067 format, and 29,176 phonetic; 42,243 rows (87.79%) are format/phonetic-risk. All 2,565 multi-label rows are counted once per label in category tables. The 200-row stratified export has 100 shipped and 100 local unique terms, with class coverage of 56 phonetic, 53 format, 49 case, and 42 same.

## Verification

- `PYTHONPATH=tools/lexicon-benchmark uv run --no-project --python 3.12 --with pytest pytest -q tools/lexicon-benchmark/tests`: 31 passed.
- `uvx ruff check scripts tools/lexicon-benchmark`: pass.
- `uvx ruff format --check scripts tools/lexicon-benchmark tests sidecar tools/tests`: pass.
- `just quality`: pass; aggregate Python suite 84 passed.
- `shellcheck tools/profile-miner/lexicon/oracle_linux_setup.sh`: pass.
- `shfmt --diff tools/profile-miner/lexicon/oracle_linux_setup.sh`: pass.
- `git diff --check`: pass.
- Final reviewed collector run produced 6,170 terms with no collection failures.
- Final benchmark merge produced 48,118 ranked rows, 48,118 complete two-voice rows, and a 200-row unique stratified slice.
- Read-only CCR review completed; all findings were repaired: private registry leakage, missing quality-runner integration, pre-ranking `--limit`, all-null voice evidence, origin-balanced stratification and fallback, scoped npm filtering, exact toolbelt parsing, aggregate-manifest projection and provenance, multi-name manual rows, inline uv registry metadata, Cargo.lock TOML parsing, explicit UTF-8 gzip decoding, empty PATH components, negative counts, and colliding/unsafe outputs.

## Non-claims

- This does not read user dictation, Aqua history/settings, or the local Phonon corpus.
- Two-voice synthesized TTS risk is triage, not product WER.
- A high-risk term can still be corrected successfully by a dictionary.
- The 6,170 local candidates are not claimed as training data or product vocabulary.
- No correction model, adapter, or daily-driver capability is qualified by this receipt.
