# Receipt: three-teacher filtering and initial training packs

Date: 2026-10-18
Hosts: main machine for manifests; B550 for Liquid Audio pack construction and
loader verification
Branch: `ml/portable-harness`

## Question

Determine how much of the native 48 kHz capture can be filtered automatically
using Cohere Transcribe, Qwen3-ASR, and Voxtral Small 24B MXFP4, and structure
the surviving rows into the existing SALM training-pack contract.

## Implementation

Added `tools/teacher-filter/build_dataset.py`. It:

1. aligns the three teacher outputs by audio stem;
2. computes normalized pairwise WER over all three teacher pairs;
3. selects a two-teacher majority transcript when possible, otherwise the
   low-disagreement medoid transcript;
4. assigns `auto`, `review`, or `exclude` tiers;
5. emits separate dataset, default raw pack, review, exclusion, and derived-arm
   augmentation manifests;
6. records teacher-output hashes and explicit non-claims.

The default raw lane never mixes denoised audio with raw audio. Derived arms
are emitted only to `derived-augmentation-manifest.jsonl`.

## Filtering yield

Corrected mean pairwise WER yield over all 27 ten-second segments:

| Mean pairwise WER threshold | Rows | Seconds | Percent |
| ---: | ---: | ---: | ---: |
| 0.000 | 4 | 39.936 | 14.8% |
| 0.025 | 5 | 49.920 | 18.5% |
| 0.050 | 9 | 89.856 | 33.3% |
| 0.075 | 13 | 129.792 | 48.1% |
| 0.100 | 14 | 139.861 | 51.9% |
| 0.125 | 17 | 165.738 | 63.0% |
| 0.150 | 18 | 175.807 | 66.7% |
| 0.250 | 19 | 185.876 | 70.4% |
| 0.400 | 21 | 205.929 | 77.8% |
| 0.500 | 22 | 215.913 | 81.5% |

The selected operating points are:

| Lane | Auto threshold | Auto | Review | Exclude | Derived augmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 0.075 | 13 rows / 129.792 s | 8 rows / 76.137 s | 6 rows / 59.989 s | 39 rows |
| Broad | 0.150 | 18 rows / 175.807 s | 3 rows / 30.122 s | 6 rows / 59.989 s | 54 rows |

The broad lane implements the requested data-first posture: 24 of 27 rows are
resolved automatically, only three clips remain for review, and the six clear
disagreement/silence rows stay excluded. Strict remains the conservative control
lane.

Local private manifests:

```text
~/.local/share/phonon-research/parallel-isolation-v2/structured-strict-v1/
~/.local/share/phonon-research/parallel-isolation-v2/structured-broad-v1/
~/.local/share/phonon-research/parallel-isolation-v2/filter-yield-curve.json
```

## Training packs

B550 pack root:

```text
/home/kearm/salm-lora/build/parallel-isolation-packs/
```

All packs use prompt `prose_dictation_v1`, prompt SHA-256
`3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`,
context length 768, duration bounds 2-20 seconds, and the existing
`build_aqua_prompt_pack.py` tokenizer/preprocessor.

| Pack | Rows | Skipped | Directory hash | Size |
| --- | ---: | ---: | --- | ---: |
| `strict-raw-v1` | 13 | 0 | `abb0b0b0c554169a724217ed8cf071d88ed334c71d807cb08a3369da9156649f` | 6.4 MiB |
| `broad-raw-v1` | 18 | 0 | `2ad18a89a7c923fb9a7ecd69c4e843a357c483af6f438aec3be804fc7bdd0ea0` | 8.7 MiB |
| `broad-derived-v1` | 54 | 0 | `6692857f42d2a2649e2e22b5b46c5310178e297e29c43d3eb4c2d0363fb1a3c8` | 26 MiB |

`broad-derived-v1` is separate because RNNoise and DeepFilterNet remain
unpromoted candidates; it must not be interpreted as the default training lane.

## Live verification

- `PYTHONPATH=tools/teacher-filter uv run --no-project --python 3.12 --with pytest pytest -q tools/teacher-filter/tests`: 3 passed
- `uvx ruff check tools/teacher-filter`: pass
- `uvx ruff format --check tools/teacher-filter`: pass
- `git diff --check`: pass
- Pack builds: all three complete with zero skipped rows
- Loader probes with `LFM2DataLoader(..., context_length=768)`:
  - strict raw: 13 rows, one collated `LFM2AudioModelInput`
  - broad raw: 18 rows, one collated `LFM2AudioModelInput`
  - broad derived: 54 rows, one collated `LFM2AudioModelInput`

## Non-claims

- Auto labels are model-consensus labels, not human ground truth.
- No model was trained and no quality score improved as a result of this pass.
- The three review rows still require owner or external rater decisions.
- `broad-derived-v1` is an augmentation experiment only; preprocessing is not
  enabled in runtime capture.
- Existing personal audio and transcripts remain local/private and are not
  committed.
