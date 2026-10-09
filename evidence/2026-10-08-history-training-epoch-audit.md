# Receipt: history-prompt training epoch audit

Date: 2026-10-08
Host: B550
Decision: do not replace the Hugging Face model artifacts with a new fused full model yet.

## Dataset sizes

| Pack | Rows |
| --- | ---: |
| Original audio CPT pack (`aqua-sft-dataset-v3`) | 35,375 |
| Corrector pack (`aqua-correction-v4`) | 17,496 |
| Vision pack (`aqua-vision-dataset-v1`) | 8,000 |
| History-prompt audio pack (`phonon-history-pack-v2/pack`) | 12,855 |

## Older vision CPT

`salm-vision-cpt-v1` ran 24,000 steps at ratio `2:1:2`. Its lane exposure was:

| Lane | Samples | Dataset rows | Effective epochs |
| --- | ---: | ---: | ---: |
| Audio | 9,600 | 35,375 | 0.271 |
| Corrector | 4,800 | 17,496 | 0.274 |
| Vision | 9,600 | 8,000 | 1.200 |

The vision lane completed more than one shuffled pass, but audio and corrector
did not complete one epoch.

## History-prompt SFT

The selected 2,000-step model and rejected 1,000-step stage-two control both
used ratio `4:1:1`.

| Run | Audio samples | Corrector samples | Vision samples |
| --- | ---: | ---: | ---: |
| 2,000-step winner | 1,334 | 333 | 333 |
| 1,000-step stage-two control | 668 | 166 | 166 |
| Combined | 2,002 | 499 | 499 |

Combined history-SFT exposure:

| Lane | Dataset rows | Exposure |
| --- | ---: | ---: |
| History audio | 12,855 | 0.156 epoch |
| Corrector | 17,496 | 0.029 epoch |
| Vision | 8,000 | 0.062 epoch |

Therefore the new history-prompt SFT has not exhausted its dataset. The older
CPT exceeded one vision epoch, but that is not a full epoch across all lanes and
is not the new history-prompt training contract.

## Upload boundary

The best fused language file is:

`/home/kearm/salm-lora/build/phonon-history-sft-2000-fused-v1/merged_lfm.safetensors`

SHA-256:
`e15bc9059d70f05341a3c83b82cee477f1307e82bfc1a16ca4617f452b821ae7`

Size: 2.3 GiB.

Because the condition of completing the new dataset was not met, this fused
file was not uploaded to Hugging Face. The private repository continues to
carry the smaller adapter artifacts and metadata. No personal transcript or
audio data is included in this receipt.
