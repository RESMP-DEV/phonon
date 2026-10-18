# Receipt: continual-improvement option audit for fused SALM

**Date:** 2026-10-18
**Agent:** Codex
**Scope:** live compute inventory, selected-model residual-error aggregation, teacher-complementarity analysis, and ranked follow-on experiments. No training job or model artifact changed.

## Live compute

B550, also aliased `compute`, exposed two otherwise idle RTX 3090 Ti GPUs. Each has 24,564 MiB total; resident unrelated services used about 1.4 GiB per card. The research filesystem has 2.2 TiB available and the host has about 84 GiB available memory.

Strix also reports hostname `compute`, so it must be identified by SSH alias/address and GPU rather than hostname. It has one idle RTX 5070 Laptop GPU with 8,151 MiB, compute capability 12.0, driver 595.99.02, CUDA 13.4, 2.7 TiB storage available, and about 89 GiB available memory. It is suitable for lightweight evaluation and kernel probes, but the full 1.5B audio model plus vision tower needs a live batch-one fit test before training is claimed feasible there.

## Selected model and data state

The selected fused model remains:

- selection 500: 0.08548415035510133 fair WER / 0.11474067333939945 strict / 0.374 exact;
- untouched future 750: 0.11902256518180755 / 0.16334341626026297 / 0.228 exact;
- vision: 0.025369178341537298 WER / 0.750 exact;
- blank image: 0.9750094661113214 WER / 0 exact.

The unexhausted lanes are 12,855 history-audio rows, 17,496 corrector rows, and 8,000 vision rows. Selected plus rejected stage-two exposure was only 15.6%, 2.9%, and 6.2% respectively.

Stage two lowered final loss from 0.3671875 to 0.33984375 and lowered every lane's last-50 loss, while fair WER, strict WER, and vision WER all worsened. This rejects another continuation from the selected adapter under the same recipe; it does not reject fresh use of the remaining data.

## Private residual-error aggregation

These aggregates were calculated on B550 without printing or committing transcript content.

### Selection 500

- 187/500 fair-exact rows and no empty hypotheses.
- Dominant operation by row: 363 substitutions, 72 insertions, 65 deletions.
- Digit-sequence mismatch: 16/500.
- Per-row WER above 0.5: 16/500.
- Mean per-row WER 0.09838; median 0.05196.
- Exactness by reference length:
  - 1–4 words: 12/20, 60.0%;
  - 5–10 words: 45/77, 58.4%;
  - 11–20 words: 84/184, 45.7%;
  - 21+ words: 46/219, 21.0%.
- Of 313 non-exact rows, 55.3% have 21+ reference words. Their median reference length is 22 words and mean WER is 0.1572.

The residual is concentrated in long, dense output rather than catastrophic blanks. Digit errors alone cannot explain the gap.

### Future 750

- 171/750 fair-exact rows and no empty hypotheses.
- Dominant operation: 476 substitutions, 155 insertions, 119 deletions.
- Digit mismatch: 28/750.
- Mean per-row WER 0.13224; median 0.09091.
- Exactness declines from 50/115 at 5–10 words to 67/248 at 11–20 words and 46/374, 12.3%, at 21+ words.

The future file's duration values were not reliable for duration binning; reference length is the trustworthy split.

### Stage-two paired control

Against the selected model, best stage-two trial 0 was better on 46 rows, tied 402, and was worse on 52. Only nine rows were severe under both, and stage two recovered seven of the selected model's severe rows. It changes a narrow boundary and net-harms aggregate quality.

## Teacher complementarity

Existing full-slice outputs were compared without exposing text:

| Output | Fair WER | Beats Aqua | Ties Aqua | Loses to Aqua |
| --- | ---: | ---: | ---: | ---: |
| Aqua raw | 0.03291183093712108 | n/a | n/a | n/a |
| Cohere Transcribe 03-2026 | 0.05768231422137537 | 20 | 303 | 177 |
| Voxtral Small MXFP4 | 0.06071366707084705 | 11 | 305 | 184 |
| Qwen3 ASR 1.7B | 0.07067382643339684 | 6 | 268 | 226 |
| Selected fused SALM | 0.08548415035510133 | 44 | 203 | 253 |

A reference-only row oracle reaches 0.02494370344708124 fair WER. Winner counts are Aqua 445, selected SALM 42, Cohere 11, and Qwen 2. This is an upper bound, not a feasible selector.

Simple reference-free gates were diagnostics only: selected-when-Cohere-agrees scored 0.034903862809631044, selected-when-any-teacher-agrees scored 0.036116403949419715, and offline majority scored 0.04339165078815174. None beat Aqua raw. This is not a two-model runtime proposal; the useful signal is that disagreement rows can drive active data selection or single-model distillation.

## Ranked improvement options

1. **Deployable history-retrieval matrix.** The registered history score uses the manifest's raw Aqua transcript as retrieval query, which product inference does not initially have. Score no-history, first-pass SALM self-retrieval, raw-query control, and a compact fixed card on selection and future rows with latency. This is the cheapest decisive experiment.
2. **Full-pack private inference and active hard-pack.** Run the selected fused checkpoint over all 12,855 history-pack rows. Current mean generation is about 0.44 s/row: roughly 95 minutes on one 3090 Ti or 48 minutes split across both. Aggregate locally, then build a fresh pack weighted to long rows, substitutions, Aqua/SALM disagreement, random replay, corrector retention, and vision retention. Train from the CPT initialization, not the selected adapter. Use the 1,458-row development split for checkpoint selection and score only the best arms on future 750.
3. **Regularized full-lane study.** From the CPT initialization, sweep dropout, weight decay, LR, rank 16 versus 32, and development-set early stopping over much more of the pack. This tests whether stage two failed from optimization overfitting rather than data exhaustion.
4. **Long-utterance specialization.** Build stable silence/energy segmentation, overlap or carry-over context, punctuation reconstruction, and rejection of invented transitions. Gate on the 21+ word regime without regressing short-row latency or exactness.
5. **Joint audio-plus-image conditioning.** The patched model can receive audio and image tensors, but no pack supplies paired dictation audio plus screenshot context. Build consent-gated real pairs plus synthetic distractors. The prompt must use the screenshot as context for speech, never copy visible screen text. Start with active/cropped context and a 2,048–3,072 token window.
6. **Real screenshot generalization.** Add consented real UI layouts, fonts, scaling, dark modes, and occlusions beyond the rendered terminal/editor pack. Keep blank controls; unfreeze the projector only in a separate low-LR arm.
7. **Single-model disagreement distillation.** Use offline complementarity to add rows where SALM uniquely beats Aqua and rows where Aqua wins into a fresh SFT pack. Keep teacher metadata out of prompts and do not add a runtime ensemble.
8. **Public-audio curriculum.** YODAS/Granary scaling helped the reverse lane, but this is more expensive and less targeted than active-pack and long-utterance work. Keep it later.
9. **Strix kernel/inference probes.** SM120 kernels and memory fit can improve throughput or unlock context, but are quality-neutral until then.

## Recommended first wave

Run options 1 and 2 concurrently on B550. Put the retrieval matrix on GPU 0 and the full-pack inference audit on GPU 1, or split the 12,855 rows across both after retrieval completes. Leave Strix out of model execution until a batch-one resident-memory fit test passes. Use the resulting receipts to construct one active/replay pack and one new regularized study.

## Commands and privacy

Live state used `ssh B550`, `ssh strix`, `nvidia-smi`, `df`, `free`, dataset metadata reads, and aggregate-only Python over existing hypothesis files. No transcript, accepted text, audio list, screenshot, or personal prompt was printed or committed.

## Non-claims

- No training or model-state change occurred.
- Teacher gates are selection-slice diagnostics, not deployable future claims.
- The oracle uses references and is not a product selector.
- Strix model fit remains unmeasured.
- This receipt does not decide branch integration or Hugging Face release.
