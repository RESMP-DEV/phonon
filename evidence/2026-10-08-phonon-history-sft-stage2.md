# Receipt: Phonon history-prompt SFT stage two

Date: 2026-10-08
Study root: `/home/kearm/salm-lora/build/optuna/phonon-history-sft-stage2-lr-v1`
Source revision: `dbad2981d5cd2d0a060e99ea8f17f039cf2073a1`
Marker: `OPTUNA-DONE`

## Contract

- Initialized each trial from the best 2,000-step adapter
  `d0bd203510dae234839e2afa5fdd337ef382a5406bd9acd5b38ac0f1f38515f5`.
- Audio pack: `/home/kearm/salm-lora/build/phonon-history-pack-v2/pack`
- Prompt: `prose_history_dictation_v2`
- Rank 16, 4:1:1 audio/corrector/vision mix
- Contexts 768/768/1024
- 1,000 additional steps per trial
- Three learning rates between `2e-5` and `8e-5`
- Every trial fused the LoRA and scored the standalone fused model
- FlashAttention `2.8.3.post1` and causal-conv1d `1.7.0` were active and recorded

## Results

| Trial | LR | Objective | Audio fair WER | Audio strict WER | Audio exact | Vision WER | Vision exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.0246e-5 | 0.0866967 | 0.0866967 | 0.1156506 | 0.368 | 0.0344566 | 0.760 |
| 1 | 2.2740e-5 | 0.0871297 | 0.0871297 | 0.1161965 | 0.366 | 0.0257478 | 0.760 |
| 2 | 3.9979e-5 | 0.0882557 | 0.0882557 | 0.1172884 | 0.368 | 0.0265051 | 0.750 |

Best trial: 0
Best LR: `6.0245784940505935e-05`
Adapter SHA-256: `3dc9361790c7904622990456e0daf95f49ccc67d697a99988174d746ec65c26f`

## Verdict

Stage two did **not** improve the selected fused model. The existing 2,000-step
fused checkpoint remains better:

| Model | Selection fair WER | Selection strict WER | Selection exact | Vision WER |
| --- | ---: | ---: | ---: | ---: |
| Existing 2,000-step fused | 0.08548415035510133 | 0.11474067333939945 | 0.374 | 0.025369178341537298 |
| Stage-two best | 0.08669669149489001 | 0.11565059144676978 | 0.368 | 0.03445664521014767 |

The stage-two best has slightly higher exact than some older arms, but it is
worse than the existing fused checkpoint on all decisive aggregate WER metrics
and vision WER. It is retained as a control, not promoted.

## Non-claims

- No future-split score was run because selection already failed to improve.
- This does not prove additional SFT is impossible; it rejects this specific
  continuation point, step count, LR range, and data mix.
