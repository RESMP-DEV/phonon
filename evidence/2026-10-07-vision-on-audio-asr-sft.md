# Receipt: vision-on-audio SALM post-CPT ASR SFT

Date: 2026-10-07
Agent: Codex
Host: B550 (`/home/kearm/phonon`, branch `train-adapt-v4`)
Source revision: `dee49221` (`feat: add init-adapter and LoRA fuse for vision SALM`)

## Scope

Continue `salm-vision-cpt-v1` (audio-native SALM with a transplanted SigLIP2
tower and VL projector) with supervised fine-tuning on the Aqua dictation
pack to recover and improve the ASR capability, then fuse the LoRA into the
language weights. No GRPO was run.

## Code added

- `ml/research/salm_vision_v0/salm_vision.py`: vision-on-audio model patch
  (`_vision_prefill`, `_vision_features`, `_vision_logits`, `install_vision`),
  `VisionRow`/`VisionBatch`/`VisionDataLoader`/`vision_collator`, and
  `generate_text`. Migrated from the uncontrolled B550 workspace.
- `ml/research/salm_vision_v0/train_salm_vision_cpt.py`: CPT/SFT trainer with a
  new `--init-adapter` argument that loads an existing LoRA (and projector
  tensors, when present) before training, enabling task tuning from a prior
  stage instead of a fresh LoRA.
- `ml/research/salm_vision_v0/fuse_salm_vision.py`: merges the language LoRA
  into `lfm.*` weights by calling `.merge()` on every inserted PEFT target
  layer, then writes `merged_lfm.safetensors`, optional `projector.safetensors`,
  the vision rows, and `fuse_receipt.json` with SHA-256 hashes.
- `sidecar/salm_vision_server.py`: JSONL product-sidecar prototype accepting
  `caption`, `ping`, and `shutdown`, with an explicit image path per request and
  no image persistence.

## Verification

Local: `uvx ruff check` on the new modules passed.
B550: fuse smoke on `salm-vision-cpt-v1` produced
`merged_lfm.safetensors` SHA-256
`a0400d2db955a4bfa7d6a60d02bd2102345685d2e6947624eee329487c95ef15`, 184 LoRA
tensors folded across 92 target layers, injected-model audio cross-entropy
0.1778. Trainer `--init-adapter` smoke ran 10 steps from the CPT adapter.

## Experiment 1: mixed-lane SFT, 2,000 steps, LR 2e-5, ratio 4:1:1

Root: `/home/kearm/salm-lora/build/salm-vision-asr-sft-v1`
Init adapter: `/home/kearm/salm-lora/salm-vision-cpt-v1/cpt_adapter.safetensors`
(`a43fac977d6fa3bf8150257ac26db8a9cea39d8eb937db17d64ecc493260d6dc`)
Adapter SHA-256: `97430109f036902a46c3d610fbf847f57aed359f17e0eb26563caea8013910ea`
Wall time: 324.2 s. Lane mean last 50: audio 0.2840, corrector 0.1368,
vision 0.0385.

Fused output: `/home/kearm/salm-lora/build/salm-vision-fused-v1`
`merged_lfm.safetensors` SHA-256
`9fa33fc51c7c6ee40c1254fbe939ecc5366809da1e541b52ce93aa9c4e1c4659`.

| Evaluation | Result |
| --- | --- |
| Vision reading, 200 held-out screenshots | WER 0.0248, exact 0.795 |
| Vision blank-image control | WER 1.2973, exact 0.000 |
| Audio, 500-row selection slice | fair 0.09024770483284254, strict 0.11956323930846224, exact 0.350 |

Vision is preserved; audio is unchanged from the CPT baseline (0.0902). This
arm did not improve ASR.

## Experiment 2: audio-only SFT, 2,000 steps, LR 1e-4

Root: `/home/kearm/salm-lora/build/salm-vision-asr-sft-audio-only-v1`
Same init adapter, audio pack only (`aqua-sft-dataset-v3`), no corrector or
vision lane. Wall time 344.8 s, audio mean last 50 0.5741.

| Evaluation | Result |
| --- | --- |
| Audio, 500-row selection slice | fair 0.08531093019227438, strict 0.11637852593266607, exact 0.356 |
| Vision reading, 200 held-out screenshots | WER 0.0252, exact 0.770 |
| Vision blank-image control | WER 1.0298, exact 0.000 |

Audio improved by 0.0049 fair WER over the CPT baseline while vision reading
was retained (0.0252 WER, 77.0 percent exact) and the blank control still
collapsed to 0 percent exact. Exact match rose from 0.350 to 0.356.

## Comparison

| Model | Audio fair WER | Vision WER | Vision blank WER |
| --- | ---: | ---: | ---: |
| CPT baseline `salm-vision-cpt-v1` | 0.0902 | 0.0250 | 0.9970 |
| Mixed SFT 2e-5 | 0.0902 | 0.0248 | 1.2973 |
| Audio-only SFT 1e-4 | 0.0853 | 0.0252 | 1.0298 |
| Audio-only `native-salm-grpo-v1` (no vision) | 0.0839 | n/a | n/a |

The audio-only SFT adapter is the best vision-capable audio model so far, but
it remains behind the audio-only GRPO adapter on fair WER by 0.0014.

## Non-claims

- No future-split evaluation was run for either SFT arm.
- No macOS or product-sidecar latency was measured for the SFT adapters.
- The fused `merged_lfm.safetensors` is not yet evaluated as a standalone
  model; only the LoRA-injected path was scored.
- 2,000 steps at LR 1e-4 is one operating point, not a swept optimum.
