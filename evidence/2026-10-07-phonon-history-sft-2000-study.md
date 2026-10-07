# Receipt: Phonon history-prompt SFT 2,000-step Optuna study

Date: 2026-10-07
Study root: `/home/kearm/salm-lora/build/optuna/phonon-history-sft-2000-v1`
Source revision: `b94512ff20f826c82679bc967c7a5096284d4d1e`
Marker: `OPTUNA-DONE`

## Contract

- Audio pack: `/home/kearm/salm-lora/build/phonon-history-pack-v2/pack`
  (12,855 rows, prompt `prose_history_dictation_v2`, prompt SHA-256
  `664fdeeca49d5c43187f435cf7de4c3b2b2951fee45b9059dae2fef4e9ecb2a6`,
  manifest SHA-256
  `9378e4532e54c8816ea962a515ac833d60efbab0ec1e1d102b0296e81d490f08`)
- Initialized from `salm-vision-cpt-v1/cpt_adapter.safetensors`
- Rank 16, 4:1:1 audio/corrector/vision mix, contexts 768/768/1024
- 2,000 steps per trial, three trials
- Learning rate sampled logarithmically from `3e-5` to `1.1e-4`
- Objective: audio fair WER plus `2 * max(0, vision WER - 0.04)`
- FlashAttention 2.8.3.post1 and causal-conv1d 1.7.0 active and recorded

## Results

| Trial | LR | Objective | Audio fair WER | Audio strict WER | Audio exact | Vision WER | Vision exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0232e-4 | 0.0862636 | 0.0862636 | 0.1154686 | 0.358 | 0.0272624 | 0.770 |
| 1 | 6.2662e-5 | 0.0873896 | 0.0873896 | 0.1160146 | 0.352 | 0.0234760 | 0.780 |
| 2 | 3.4878e-5 | 0.0868699 | 0.0868699 | 0.1158326 | 0.358 | 0.0265051 | 0.810 |

Best trial: 0
Best LR: `1.0231901903347211e-04`
Best adapter SHA-256: `d0bd203510dae234839e2afa5fdd337ef382a5406bd9acd5b38ac0f1f38515f5`
Best hypothesis SHA-256: `413fd87d5669c3ddbd3247b31b0d9f3c4be1a21d12e4892929fcc287b02c4fb6`

## Fused model

The winning LoRA was fused into the language weights:

- Output: `/home/kearm/salm-lora/build/phonon-history-sft-2000-fused-v1/merged_lfm.safetensors`
- SHA-256: `e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7`
- 184 LoRA tensors merged across 92 target layers
- Injected-model cross-entropy on a history-pack batch: 0.12680479884147644

## Comparison

| Model | Audio fair WER | Audio exact | Vision WER |
| --- | ---: | ---: | ---: |
| CPT baseline | 0.0902 | 0.350 | 0.0250 |
| 600-step history study best | 0.0868 | 0.356 | 0.0227 |
| 2,000-step history study best | 0.0863 | 0.358 | 0.0273 |
| Audio-only GRPO (no vision) | 0.0839 | 0.360 | n/a |

The 2,000-step history adapter is the best measured audio score among
vision-capable models, and it is 0.0024 behind the audio-only GRPO adapter.

## Non-claims

The fused weights were not evaluated as a standalone model; only the injected
path was scored. The untouched future split was not scored. The selected
learning rate is valid only under this exact source revision, pack, prompt,
contexts, rank, initialization adapter, kernel mode, and step count.
