# Receipt: history SFT overfit audit and existing Phonon deployment

Date: 2026-10-09
Agent: Codex
Host: main M4 Max and B550

## Overfitting / continuation evidence

The selected 2,000-step history-prompt model was followed by a three-trial,
1,000-step stage-two Optuna study. Every stage-two trial initialized from the
selected 2,000-step adapter and used the same 4:1:1 data mix. Training loss
decreased, but selection quality and vision quality regressed:

| Model | Training status | Selection fair WER | Selection exact | Vision WER |
| --- | --- | ---: | ---: | ---: |
| Selected 2,000-step fused model | baseline | 0.08548415035510133 | 0.374 | 0.025369178341537298 |
| Best stage-two continuation | lower final loss | 0.08669669149489001 | 0.368 | 0.03445664521014767 |

The stage-two best also had worse strict WER (`0.11565059144676978` versus
`0.11474067333939945`). This is evidence against continuing the same recipe:
additional optimization reduced training loss while worsening both held-out
audio and held-out vision. It does not prove that every longer schedule or
different mixture will overfit, but it rejects blindly extending the current
continuation point.

The new history-prompt SFT has also consumed only 2,002 of 12,855 audio rows,
499 of 17,496 corrector rows, and 499 of 8,000 vision rows. Therefore this is
not classical dataset exhaustion; it is earlier negative transfer or capability
forgetting under the tested continuation recipe.

## Existing-setup deployment verification

The existing explicit SALM path was verified on the main machine with the real
audio fixture `/tmp/phonon-real-probe.wav`.

Command family:

```bash
PHONON_ASR_ENGINE=salm cargo run -q -p phonon-cli -- engine
```

Observed events:

- LiquidAudio processor loaded.
- `LFM2.5-Audio-1.5B` weights loaded.
- Installed `lora_adapter.safetensors` loaded.
- Real-audio transcription returned non-empty text.
- ASR warmup and demo passed.
- Correction model loaded and became ready.
- Engine emitted final `ready`.

Installed adapter:

`/Users/kearm/.local/share/phonon/salm/lora_adapter.safetensors`

SHA-256:

`d157a38be7ebc801000a6bac13146b8f7333b4a89f8dcda1eb9e26ad5b54f830`

A direct `phonon-asr` real-spawn integration test also passed. The hardcoded
`starting parakeet` message in `phonon engine` is diagnostic text only; the
subsequent LiquidAudio and adapter events prove the SALM engine was selected.

## Boundary

The verified deployment is the audio-only native SALM GRPO adapter. The current
vision-capable history adapter is not deployed in the product engine because
its image sidecar and consented image protocol are not integrated. The
vision-capable fused artifact remains research-only.
