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
The final merged v4 report and term exports remain under
`~/.local/share/phonon-research/lexicon-benchmark/local/`.

## Local inventory collection

The final collector run produced 5,025 unique local candidates:

| Source | Kept | Seen |
| --- | ---: | ---: |
| toolbelt | 12 | 22 |
| path-command | 2,677 | 3,375 |
| manpage | 3,503 | 7,151 |
| npm-local | 269 | 1,079 |
| brew-tap | 5 | 14 |

Terms can belong to multiple sources. `collection_failures` was empty. The local terms SHA-256 is `744786a83858a1100620e6c50f978a291e43b263c083cbcaa5e8ff0d180f3aef`. Candidate, lexicon, oracle, and ranked outputs remain under `~/.local/share/phonon-research/lexicon-benchmark/local/`.

The completed two-voice oracle classified those 5,025 local terms as:

| Worst diff class | Local terms |
| --- | ---: |
| same | 47 |
| case | 523 |
| format | 1,700 |
| phonetic | 2,755 |

The local risky subset is 4,455 terms (88.66%). By multi-label source:

| Source | Assignments | Risky share |
| --- | ---: | ---: |
| npm-local | 269 | 94.05% |
| toolbelt | 12 | 91.67% |
| manpage | 3,503 | 88.58% |
| path-command | 2,677 | 84.76% |
| brew-tap | 5 | 80.00% |

## B550 oracle run

B550 initially had GPU 0 occupied by the qwen LR study and GPU 1 idle apart from its 4.4 GiB embedding service. The first CUDA extra-index environment failed on a PyPI NVIDIA wheel timeout; a standard-PyPI oracle environment then imported torch/torchaudio/NumPy/soundfile/NeMo/Kokoro/misaki successfully.

The first xet model transfer degraded to roughly 66 KiB/s. Direct HTTP also decayed after 70–260 MiB. Segmented aria2 downloads were used instead and checksum-seeded into the B550 Hugging Face cache:

- Kokoro 82M `kokoro-v1_0.pth`: 327,212,226 bytes, SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`.
- Parakeet TDT `parakeet-tdt-0.6b-v2.nemo`: 2,472,222,720 bytes, SHA-256 `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.

Offline smoke checks loaded Kokoro on CUDA and restored Parakeet from the exact cached NeMo snapshot on CUDA. The measured 5,025-term run was pinned to GPU 1 with `HF_HUB_OFFLINE=1` and checkpointed as JSONL.

Post-run validation reported 5,025 input terms, 5,025 unique rows, zero missing terms, zero extra terms, and 5,025 two-voice rows. The oracle cache SHA-256 is `105ec54e815fa143752c22415daa767b7d3bdc02adb288aa66ce07244bb38ca1`.

The merged v4 report contains 46,973 complete two-voice rows: 41,948 shipped and 5,025 local. Its diff classes are 1,086 same, 4,708 case, 12,721 format, and 28,458 phonetic; 41,179 rows (87.67%) are format/phonetic-risk. All 2416 multi-label rows are counted once per label in category tables. The 200-row stratified export has 100 shipped and 100 local unique terms, with class coverage of 56 phonetic, 53 format, 49 case, and 42 same.

## Verification

- `PYTHONPATH=tools/lexicon-benchmark uv run --no-project --python 3.12 --with pytest pytest -q tools/lexicon-benchmark/tests`: 23 passed.
- `uvx ruff check scripts tools/lexicon-benchmark`: pass.
- `uvx ruff format --check scripts tools/lexicon-benchmark tests sidecar tools/tests`: pass.
- `just quality`: pass; aggregate Python suite 76 passed.
- `shellcheck tools/profile-miner/lexicon/oracle_linux_setup.sh`: pass.
- `shfmt --diff tools/profile-miner/lexicon/oracle_linux_setup.sh`: pass.
- `git diff --check`: pass.
- Final collector rerun reproduced the same 5,025-term SHA-256 with no collection failures.
- Final benchmark merge produced 46,973 ranked rows, 46,973 complete two-voice rows, and a 200-row unique stratified slice.
- Read-only CCR review completed; its blocking findings were repaired: private registry leakage, missing quality-runner integration, pre-ranking `--limit`, all-null voice evidence, origin-balanced stratification, scoped npm handling, exact toolbelt parsing, and aggregate-manifest projection.

## Non-claims

- This does not read user dictation, Aqua history/settings, or the local Phonon corpus.
- Two-voice synthesized TTS risk is triage, not product WER.
- A high-risk term can still be corrected successfully by a dictionary.
- The 5,025 local candidates are not claimed as training data or product vocabulary.
- No correction model, adapter, or daily-driver capability is qualified by this receipt.
