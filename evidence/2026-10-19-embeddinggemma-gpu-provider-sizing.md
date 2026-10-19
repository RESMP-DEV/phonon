# EmbeddingGemma 2 GPU provider sizing, 2026-10-19

## Scope

This receipt records live GPU-provider inventory and on-silicon sizing for the
pinned `google/embeddinggemma-2` contrastive adaptation. It covers synthetic
probes only. No private transcript, dictionary phrase, row ID, audio file, or
private JSONL was uploaded to a rented provider during this measurement.

## Model and workload facts

Pinned model:

- `google/embeddinggemma-2`
- revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
- local weights SHA-256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`

Measured properties:

- Full checkpoint tensor count: 744,371,992 parameters, all BF16.
- Text-only load with `vision_config=null`: 576,613,664 parameters
  (586,706,208 after LoRA wrappers).
- Text tower: 24 layers, hidden size 512, four attention heads, two KV heads,
  head dimension 256, 262,144-entry vocabulary, sliding window 512.
- Private text fields: p50 103, p90 129, p99 154, maximum 178 approximate
  tokens. Sequence length 512 covers the measured private text distribution.
- Public text fields: p50 17, p90 178, p99 491, maximum 4,377 approximate
  tokens. Sequence 512 covers p99, not the public maximum.
- Combined epoch used for cost comparison: 12,478 triplets (2,482 private train
  plus 9,996 public), before train/eval repetition strategy.

The text tower uses plain projections at
`language_model.layers.N.self_attn.{q,k,v,o}_proj` and
`language_model.layers.N.mlp.{gate,up,down}_proj`. A bare `linear` target is
wrong: it matches only the audio tower's `Gemma4ClippableLinear` wrappers, and a
text-only forward then has no gradient.

## On-silicon measurements

All runs used synthetic random token IDs, the in-batch ranking objective,
BF16 weights, and AdamW. The `full` rows are **pure-BF16 full-parameter runs**:
optimizer moments also remained BF16. They are not a mixed-precision FP32-master
recipe. `steps` is a short measured sample, not a full training epoch; the
historical timing mean includes the first cold step, so provider epoch costs are
planning estimates rather than steady-state acceptance measurements.

### RTX 3090 Ti, B550

| Mode | Batch x seq | Grad ckpt | Trainable | Peak alloc | Peak reserved | sec/step | Triplets/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| full (pure BF16) | 8 x 512 | no | 576.61M | 22.441 GiB | 23.150 GiB | 0.825 | 9.70 |
| full (pure BF16) | 16 x 512 | no | 576.61M | OOM | OOM | - | - |
| LoRA r32 | 4 x 512 | no | 10.09M | 11.297 GiB | 11.738 GiB | 0.449 | 8.92 |
| LoRA r32 | 8 x 512 | no | 10.09M | 21.394 GiB | 21.715 GiB | 0.793 | 10.09 |
| LoRA r32 | 8 x 512 | yes | 10.09M | 4.659 GiB | 6.023 GiB | 1.088 | 7.35 |
| LoRA r32 | 16 x 512 | yes | 10.09M | 8.075 GiB | 10.723 GiB | 2.091 | 7.65 |
| LoRA r32 | 32 x 512 | yes | 10.09M | 14.908 GiB | 20.238 GiB | 4.114 | 7.78 |

Gradient checkpointing is the decisive memory lever: at batch 8 / sequence 512
it reduces peak allocated memory from 21.394 GiB to 4.659 GiB while retaining
roughly 73 percent of non-checkpointed throughput.

### Modal A10G, `$1.10/hour`

- Batch 8 / seq 512 / LoRA r32 / grad checkpoint:
  4.657 GiB peak allocated, 6.027 GiB reserved, 4.01 triplets/sec.
- Batch 16 / seq 512: 8.075 GiB allocated, 10.719 GiB reserved, 4.20
  triplets/sec.
- Full fine-tune at batch 8 OOMs on the 22.06 GiB device.

### Modal L4, `$0.80/hour`

- Batch 8 / seq 512 / LoRA r32 / grad checkpoint:
  4.659 GiB allocated, 6.020 GiB reserved, 2.82 triplets/sec.
- Batch 16 / seq 512: 8.073 GiB allocated, 11.266 GiB reserved, 2.32
  triplets/sec.
- Batch 32 / seq 512: 14.908 GiB allocated, 20.223 GiB reserved, 2.31
  triplets/sec.
- Full fine-tune at batch 8 OOMs on the 22.06 GiB device.

### Modal T4, `$0.59/hour`

- Batch 8 / seq 512 / LoRA r32 / grad checkpoint:
  4.657 GiB allocated, 6.262 GiB reserved, 1.11 triplets/sec.
- Batch 16 / seq 512: 8.075 GiB allocated, 11.422 GiB reserved, 1.11
  triplets/sec.
- The T4 has 14.57 GiB visible, supports BF16 operation, and fits the LoRA
  probe, but its measured cost efficiency is worse than L4.

### Modal L40S, `$1.95/hour`

Live probe: 44.42 GiB visible, compute capability 8.9, BF16 supported. It was
not needed for the LoRA memory floor. It is the cheapest surveyed Modal GPU
expected to fit the measured full fine-tune memory requirement, but full
fine-tuning has not been qualified.

## Provider inventory and cost comparison

Live provider state:

- Modal metered cost `$19.28`, credits remaining approximately `$18.54`,
  current billed cost `$0.00`.
- Prime wallet balance `$49.01`.
- RunPod balance `$0.00`.
- Vast.ai requires a provider 2FA session before account reads.
- Lambda CLI is installed but no usable token was established.

For one 12,478-triplet epoch, using measured batch-8 LoRA throughput:

| Provider | Rate | Throughput | Cost/epoch | Cost/10 epochs |
|---|---:|---:|---:|---:|
| B550 RTX 3090 Ti | owned | 7.35/s | `$0.00` | `$0.00` |
| Modal A10G | `$1.10/h` | 4.01/s | `$0.951` | `$9.508` |
| Modal L4 | `$0.80/h` | 2.82/s | `$0.983` | `$9.833` |
| Modal T4 | `$0.59/h` | 1.11/s | `$1.842` | `$18.423` |

Measured recommendation:

- Use B550 for the first real LoRA run when it is stable and reachable.
- If renting, choose **Modal A10G**, not T4: A10G is the measured price/perf
  winner (`$0.951/epoch versus `$0.983` on L4 and `$1.842` on T4).
- L4 is the fallback if A10G is unavailable and cost per epoch is close enough
  (`$0.032` higher).
- T4 fits in memory but is 3.6x slower than A10G and is not cost efficient.
- L40S is unnecessary for LoRA and is not yet qualified for full fine-tuning.
- Pure-BF16 full-parameter training at batch 8 requires at least the measured
  23.15 GiB reserved; A10G/L4/T4 do not meet that floor. This is a measurement,
  not a promotion of full fine-tuning or a claim about FP32-master training.

A viable provider image was proved with
`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, plus
`transformers==5.20.2`, `sentence-transformers==6.1.0`, `peft==0.21.0`, and
Pillow. Transformers 5.20.2 requires torch >= 2.6; the stock
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image silently disables PyTorch
and is unsuitable.

## Privacy finding and boundary

A live Modal volume audit found `salm-data:/modal-real-subset-v1` containing:

- 2,000 `AQ_*.flac` audio files,
- a 2,000-row manifest with `raw` and `corrected` transcript fields,
- session IDs and timestamps,
- a SHA-256 manifest.

This is real dictation data on a rented third-party provider. It conflicts with
the standing Phonon rule that Modal is synthetic-only, and no repo receipt or
memory entry authorizes this volume. This survey did not delete or alter it
because deletion is a separate privacy decision that must preserve evidence and
an audit trail.

No private data was uploaded by the new GPU probe. Its only inputs are synthetic
random token IDs. The existing real-data Modal volume remains a separate,
explicit cleanup decision.

## B550 stability observations

B550 rebooted repeatedly during the survey:

- boot ended 11:06:01,
- a very short boot ended 11:08:01,
- a boot ended 11:48:09,
- current boot began 11:50:53.

The previous boot ended through `systemd-reboot.service`, not a kernel panic.
Unattended-upgrades automatic reboot is disabled. CUDA also failed once after a
reboot because `nvidia_uvm` was loaded but unusable; unloading and reloading
`nvidia_uvm` repaired CUDA (`torch.cuda.is_available=true`). B550 is therefore
usable for measurement, but a long run needs detached checkpointing, automatic
resume, and reboot monitoring before it can be considered reliable.

## Verification and non-claims

The new synthetic-only probe is
`tools/embedding-finetune/measure_gpu_budget.py`. Its focused test suite passed
17 tests with one GPU-only test skipped on the Mac. Ruff passed. The complete
quality-gate result for the final committed revision is in the repository's
ignored `build/reports/quality/receipt.json`.

These measurements do not approve a private-data upload, promote the private
YouTube permission gate, qualify a model adapter, establish a full training
recipe, or prove a multi-hour provider run stable. Provider prices and balances
were live at the survey time and can change.
