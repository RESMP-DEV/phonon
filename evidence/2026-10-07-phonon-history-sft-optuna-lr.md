# Receipt: Phonon history-prompt SFT Optuna study

Date: 2026-10-07
Study root: `/home/kearm/salm-lora/build/optuna/phonon-history-sft-lr-v1`
Source revision: `b94512ff`
Marker: `OPTUNA-DONE`

The study changed only the audio pack: every audio row now carries prompt
`prose_history_dictation_v2` and two dynamically selected historical
correction examples. Corrector and vision lanes, rank, initialization adapter,
contexts, 600-step trial length, and kernel mode stayed fixed.

| Trial | LR | Objective | Audio fair WER | Audio strict WER | Audio exact | Vision WER | Vision exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.2094e-4 | 0.0883423 | 0.0883423 | 0.1181984 | 0.358 | 0.0223400 | 0.750 |
| 1 | 4.8103e-5 | 0.0907674 | 0.0907674 | 0.1207461 | 0.350 | 0.0223400 | 0.810 |
| 2 | 8.6199e-5 | 0.0867833 | 0.0867833 | 0.1168335 | 0.356 | 0.0227187 | 0.780 |
| 3 | 2.4013e-5 | 0.0902477 | 0.0902477 | 0.1203822 | 0.350 | 0.0314275 | 0.800 |

Best trial: 2
Best LR: `8.619877702306163e-05`
Adapter SHA-256: `d3397f013161d8d38244b3d8ef4c77b2e0395d57681830d97c4ac733b44a664c`

All four trials ran with FlashAttention `2.8.3.post1` and causal-conv1d `1.7.0`
importable and recorded in trial user attributes.

Non-claims: four 600-step trials are not a full training run, the future split
was not scored, the fused standalone model was not scored, and this LR is valid
only under this exact pack/prompt/source/kernel contract.
