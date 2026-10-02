# Reverse VL graft v0

Direction under test: graft the audio input head from
`LiquidAI/LFM2.5-Audio-1.5B` onto the language stack of
`LiquidAI/LFM2.5-VL-1.6B`. This is the reverse of the earlier vision-into-audio
graft.

## Implementation

`reverse_audio_vl.py`:

- loads both pinned models;
- copies/retains the Audio conformer and `audio_adapter`;
- reserves harmless token id 14 for continuous audio slots;
- encodes mel frames with the Audio conformer;
- projects frames with `audio_adapter`;
- replaces reserved token embeddings in the VL language stream;
- keeps the native VL vision path intact but unused;
- supplies supervised logits and greedy generation.

`train_reverse_audio_vl.py` trains:

- VL LoRA r16 on convolution and feed-forward projections;
- the audio adapter;
- frozen VL base and frozen conformer;
- with checkpoint/resume support.

`build_granary_reverse_pack.py` streams public English
`espnet/yodas-granary` rows with `Audio(decode=False)`, filters 2-20 second
WAVs, and packs them through the normal liquid-audio mapper. The source
manifests are `nvidia/Granary` (CC-BY-4.0); the backing audio dataset is
CC-BY-3.0.

`transcribe_reverse_audio_vl.py` loads a saved graft adapter and evaluates a
JSONL slice.

## Bounded result (2026-10-01)

On the first 25 rows of the frozen 500-clip Aqua slice:

| stage | fair WER | strict WER | exact |
| --- | ---: | ---: | ---: |
| Aqua-only graft, 2k steps | 0.1458 | 0.2029 | 0.28 |
| YODAS-Granary 2k public steps | 0.1840 | 0.2322 | 0.28 |
| YODAS-Granary 2k + Aqua 1k | **0.1337** | **0.1755** | 0.40 |

The public-only stage does not transfer directly, but public pretraining plus
Aqua adaptation improves the reverse graft. A 10k-row/10k-step scaling run is
the next measurement. This lane remains far behind the native Audio student on
this slice and is research-only.

Run from the B550 research root in `~/salm-lora` using the pinned
`~/envs/salm-lora` environment; data and generated packs stay outside Git.
