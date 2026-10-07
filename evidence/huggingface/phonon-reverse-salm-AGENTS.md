# AGENTS.md — loading and merging Phonon reverse SALM adapters

This repository contains research adapters for a two-model graft, not a normal
single-base LoRA checkpoint. Read this before loading, merging, evaluating, or
redistributing the files.

## Current artifacts

| Artifact | Purpose | Prompt contract |
| --- | --- | --- |
| `adapters/v0.1.0-alpha.1.safetensors` | History-conditioned alpha prototype | `prose_history_dictation_v2`, context 768 |
| `adapters/static-soup-v0.safetensors` | Static-prose control/comparison | `prose_dictation_v1`, context 512 |

Both adapters use LoRA rank 32 and contain 190 tensors:

- 184 `vl.*` tensors: PEFT LoRA A/B tensors for the grafted VL language model.
- 6 `audio_adapter.*` tensors: **full replacement weights** for the audio
  adapter, not LoRA tensors.

There are no `conformer.*` tensors in these two releases.

## Required base models

Instantiate both upstream models:

- Audio conformer and processor:
  `LiquidAI/LFM2.5-Audio-1.5B`
- VL language model and tokenizer:
  `LiquidAI/LFM2.5-VL-1.6B`

The graft attaches the audio conformer and trained audio adapter to the VL
language stack. Audio embeddings replace reserved token ID `14` in the VL input
stream. Vision remains present but unused.

Do not load either adapter into only one stock Hugging Face model. A merged VL
checkpoint without the audio model and graft code is not runnable as reverse
SALM.

## Verified runtime loading path

The verified path is Phonon branch `ml/portable-harness`:

```bash
git clone git@github.com:RESMP-DEV/phonon.git
cd phonon
git checkout ml/portable-harness

hf download RESMP-DEV/phonon-reverse-salm \
  adapters/v0.1.0-alpha.1.safetensors \
  --local-dir ../phonon-reverse-salm
```

Use the existing loader rather than reimplementing the namespace split:

```python
from pathlib import Path

import torch

from ml.research.reverse_vl_v0.reverse_audio_vl import ReverseAudioVL
from ml.research.reverse_vl_v0.transcribe_reverse_audio_vl import load_adapter

device = torch.device("cuda")
model = ReverseAudioVL.from_pretrained(device=device)
load_adapter(
    model,
    Path("../phonon-reverse-salm/adapters/v0.1.0-alpha.1.safetensors"),
    rank=32,
)
```

The equivalent manual contract is:

1. Build `ReverseAudioVL.from_pretrained(...)`, which loads both upstream bases.
2. Freeze the VL and audio parameters.
3. Inject LoRA into `model.vl` with:
   - rank `32`
   - alpha `64`
   - inference dropout `0.0`
   - the target regex from `reverse_audio_vl.LORA_TARGETS`
4. Remove the `vl.` prefix and load those tensors into the injected VL model.
5. Remove the `audio_adapter.` prefix and load those six tensors into
   `model.audio.audio_adapter` with `strict=True`.
6. Return both modules to the target device and call `.eval()`.

A direct `PeftModel.from_pretrained(...)` call against this file is incorrect:
the file also contains the audio-adapter namespace and is not a standalone PEFT
directory.

## Eager LoRA merge

The runtime injection path above is the verified quality path. If an integration
requires baked VL weights, merge only after both upstream models and the graft
are constructed and the adapter namespaces are split correctly.

This adapter release uses `inject_adapter_in_model`, which modifies the VL model
in place. It does **not** wrap the model in `PeftModel`; therefore
`model.vl.merge_and_unload()` raises `AttributeError` and is the wrong API here.

Call `merge()` on every inserted PEFT target layer:

```python
load_adapter(
    model,
    Path("adapters/v0.1.0-alpha.1.safetensors"),
    rank=32,
)

merged_targets = [
    module
    for module in model.vl.modules()
    if hasattr(module, "base_layer") and hasattr(module, "lora_A")
]
assert len(merged_targets) == 92
for module in merged_targets:
    module.merge()
```

This bakes the 184 LoRA A/B deltas into the 92 target layers in place. The six
`audio_adapter` tensors are full replacement weights and must remain attached to
`model.audio.audio_adapter`; they are not LoRA and must not be merged.

A B550 smoke test using adapter SHA-256
`44a379ee1762461077319fa28173d2da782ad1ea5f29b0303c8fde3218e033a4` loaded the
graft, generated once, merged all 92 targets, generated again, and obtained an
identical non-empty 120-character output. No transcript was printed or stored.

The in-place merged VL module still contains dormant LoRA branches. Removing
those branches and producing a smaller standalone VL checkpoint requires a
separate unload/export utility and byte-identical quality receipt; do not
improvise it by discarding submodules. A runnable merged export must preserve
all of:

1. the merged VL language model;
2. the audio conformer from `LFM2.5-Audio-1.5B`;
3. the trained `audio.audio_adapter` weights;
4. the VL tokenizer/config;
5. the audio placeholder and embedding-splicing behavior;
6. the exact prompt ID, prompt bytes, context length, and rank contract.

Do not publish a merged model over this repository without a new explicit
release decision and byte-identical quality receipt.

## Prompt and history contracts

### `v0.1.0-alpha.1`

- Prompt ID: `prose_history_dictation_v2`
- Prompt schema SHA-256:
  `664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6`
- Context: 768
- Rank: 32
- Alpha: 64

The intended prototype uses one pass with a locally controlled fixed user
history card. That card contains personal transcript text and is intentionally
not in this repository. Build it locally from consented data with the Phonon
builder and keep it under the collaborator's data policy.

The alpha adapter can also be evaluated with per-row dynamic history retrieval,
but only when the same training-only history pool and deterministic selector are
available. Do not substitute a reference transcript at product inference time.

### `static-soup-v0`

- Prompt ID: `prose_dictation_v1`
- Prompt SHA-256:
  `3ccd2adee6411c68fa0126b7af9cfaf838d95cf89d87643c8f00cfd87cea11a5`
- Context: 512
- Rank: 32

Do not use the static-prose adapter with the history prompt. The history prompt
is not a generic upgrade: the static soup collapses under it.

## Do not mix contracts

Never average, soup, partially load, or layer these adapters unless all of the
following match:

- identical base-model revisions;
- identical key sets, shapes, and dtypes;
- identical prompt ID and bytes;
- identical context length;
- identical LoRA rank and alpha;
- identical training/data provenance intent.

The two published adapters intentionally use different prompt/context contracts.
They are comparison artifacts, not interchangeable weights.

## Verification checklist

Before reporting an integration as working:

1. Re-download the adapter from this private repository.
2. Verify SHA-256 against `ADAPTER_SHA256SUMS`.
3. Confirm exactly 190 tensors and the two expected top-level namespaces.
4. Confirm one consistent LoRA rank: 32.
5. Load through `load_adapter(..., rank=32)`.
6. Run at least one real audio file through the graft and confirm non-empty
   output.
7. Record the prompt ID/hash, context, rank, kernel versions, adapter hash, and
   audio ID or fixture ID without storing transcript text in public telemetry.
8. For quality comparisons, use the frozen Phonon scorer and report fair WER,
   strict WER, exact match, and row count together.

## Privacy and release rules

This is a private internal research release. Do not redistribute publicly.

Never upload any of:

- personal dictation audio;
- raw or accepted transcripts;
- fixed user-history cards;
- audio IDs or session IDs;
- dictionary values;
- screenshots;
- internal manifest files;
- telemetry containing transcript text.

Only aggregate metrics, tensor hashes, prompt hashes, and controlled fixture
identifiers belong in public or shared receipts.
