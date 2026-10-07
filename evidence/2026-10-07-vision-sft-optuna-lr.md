# Receipt: vision-on-audio SFT Optuna learning-rate study

Date: 2026-10-07
Agent: Codex
Host: B550
Source revision: `88f00c1ad65d2ad76f38419e38635ed8d588f13f`
Study root: `/home/kearm/salm-lora/build/optuna/vision-on-audio-asr-sft-lr-v1`
Study DB: `study.db`
Marker: `OPTUNA-DONE`

## Environment

The study manifest is `ml/research/salm_vision_v0/environment-b550.json`.

- Python 3.12
- Torch `2.14.0+cu130`
- liquid-audio `1.3.0`
- PEFT `0.21.0`
- Transformers `5.17.0`
- datasets `5.0.1`
- Optuna `5.0.0`
- FlashAttention `2.8.3.post1`
- causal-conv1d `1.7.0`
- safetensors `0.8.0`

Every trial recorded actual package versions and importability of
FlashAttention and causal-conv1d. The unrelated CUDA indexer services on B550
were not modified.

## Fixed contract

- Vision-on-audio graft, OMP rows
- Initialize from `salm-vision-cpt-v1/cpt_adapter.safetensors`
- Rank 16
- 4:1:1 audio/corrector/vision mix
- Contexts 768/768/1024
- 600 steps per trial
- Four trials
- Learning rate sampled logarithmically from `1e-5` to `1.5e-4`
- Objective: audio fair WER plus `2 * max(0, vision WER - 0.04)`

## Results

| Trial | LR | Objective | Audio fair WER | Audio strict WER | Audio exact | Vision WER | Vision exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00013597559847434336 | 0.1535041245222754 | 0.08886194353022692 | 0.1200181983621474 | 0.346 | 0.07232109049602424 | 0.780 |
| 1 | 0.00006268486284584557 | 0.08721635198337087 | 0.08721635198337087 | 0.11710646041856233 | 0.344 | 0.027262400605831124 | 0.800 |
| 2 | 0.00001346552330345776 | 0.08799584271609215 | 0.08799584271609215 | 0.1181073703366697 | 0.352 | 0.02158273381294964 | 0.800 |
| 3 | 0.000038237545987770586 | 0.0878226225532652 | 0.0878226225532652 | 0.11737943585077343 | 0.360 | 0.021204089360090876 | 0.810 |

Best trial: 1
Best LR: `6.268486284584557e-05`
Best objective: `0.08721635198337087`
Best adapter SHA-256: `996feaacbdd41ed06a5fe14b463db8f2d6658ce89bd059554d0c04f76e7d23d3`

The high-LR trial showed vision forgetting and was penalized. The lower-LR
trials retained vision; trial 3 had the best exact scores but slightly worse
audio fair WER than trial 1.

## Boundary

This is a four-trial learning-rate study at 600 steps, not a final-training
claim. It does not replace the prior 2,000-step audio-only SFT result, does not
score the untouched future split, and does not prove fused-model parity. The
chosen LR is valid only for this source revision, initialization adapter, packs,
contexts, ratio, rank, kernel mode, and step count.
