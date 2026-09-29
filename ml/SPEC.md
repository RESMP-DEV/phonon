# ml: refiner spec

## Goal

A developer dictates to a coding agent and gets the text they meant, including the identifiers,
package names, flags and people in their own work, fully on device. The ASR alone gets the words
mostly right and the technical terms mostly wrong: raw Parakeet TDT v2 hits 0.41 of held-out repo
terms and 0.18 fair WER on real dictation. The refiner exists to close that gap without damaging
the ordinary words around it.

## System

Two models, both local.

1. **Context model (offline).** Walks the user's repositories and notes, extracts candidate terms
   with a deterministic tree-sitter index (definition sites, then technical-looking mentions), and
   ranks and caps them into a per-user vocabulary. An uncapped per-user index scores worse than a
   curated union pool, so ranking is part of the contract, not an optimisation.
2. **Retrieval (per utterance, CPU).** Matches the raw transcript against the vocabulary by
   characters, double metaphone and ARPAbet phonemes over n-grams up to 8 words, with several
   spoken realisations per term and a per-kind prior; two-stage rerank; list length
   `min(80, max(30, 3 x words))` above a score threshold. Budget: about 20 ms per utterance.
3. **Refiner (per utterance).** LFM2.5-1.2B-Instruct plus a LoRA adapter. Input: the system prompt,
   one vocabulary line, the raw transcript. Output: the intended text. Greedy decode with the
   word-count guard (fall back to the raw transcript on a runaway or a collapse).

**Training data.** Acoustic synthetic pairs: terms from the lexicon, sentences written around them
by LFM2.5-1.2B, spoken by 9 Kokoro voices with stable-hash degradation, recognised by Parakeet, so
the raw side carries real ASR errors. Mixed with list, no-list and restraint rows, plus the user's
real pairs repeated x12. Distinct terms drive term hit (about 0.096 per decade of terms up to 14.5k,
bending past 70k where the eval sets reach their ceilings); more rows per term and higher LoRA rank
hurt real dictation. Text-only synthetic corruption never beat raw and is not used.

**Private users.** A generic adapter trained on acoustic data alone damages real dictation until a
short second stage on the user's own pairs. That two-stage path is the default when a user's pairs
cannot join the shared corpus.

## Deployment

One architecture, one set of weights and one quantisation per format on macOS, Windows and Linux,
with identical quality across them (decided 2026-09-17). The formats are MLX `custom_attn6emb8`
(4-bit g64, 6-bit attention, 8-bit embeddings; 746 MB) and GGUF IQ4_XS with an imatrix from mixed
personal and generic calibration text (663 MB). Both agree with bf16 at about 0.98 by word.
llama.cpp is the baseline engine for testing; custom kernels are allowed where an engine falls
short. Qwen3-1.7B is benchmarked as a possible successor model for every platform; it is not a
per-OS variant. Parakeet TDT v2 stays the ASR; parakeet-redux lost on terms and real WER.

## The eval: phonon-bench

`research/bench_v0` is the ship gate. It loads weights once, runs every frozen set in one batched
pass, and writes quality, retrieval, numerics and latency to one JSON.

| set | rows | what it answers |
|---|---:|---|
| `real_580` | 580 | real dictation after the 2026-09-04 cutoff: does the refiner help ordinary speech |
| `real_holdout120` | 120 | older real holdout, used for quantisation agreement |
| `term_old_seen` / `term_old_unseen` | 2,700 each | 300 repo terms, training voices and unseen voices |
| `term_new` | 2,700 | 300 further repo terms, never-trained voices |
| `term_pool2` | 2,700 | 300 terms from outside the repo walk (packages, headers, CLIs) |
| `term_pool3` | 2,700 | 300 terms from 150 public repos |
| `numerics_500` | 500 | teacher-forced KL, top-1 agreement and hidden-state drift against a cached bf16 reference |

Metrics: fair WER (Whisper normaliser), strict lowercase WER, damage rate (rows worse than raw),
win/tie/loss, error classes per 1000 words (ENTITY, FUNCTION, ORTHOGRAPHY, DROP, INSERT), term hit
(normalised and loose), hit when the term was retrieved, retrieval recall at 10/30/budget, false
insertions of list terms, guard firings, and latency (TTFT, tokens/s, peak memory).

Proposed gate, set from the measured floors and not yet enforced in code: real fair WER and damage no worse than the reference adapter
beyond noise (+/- 0.002 WER), real ENTITY at or under raw Parakeet (22.6 per 1000), term hit within
0.01 of the reference, and an int4 build at 0.98 or better word agreement with its bf16 source.

## Eval roadmap

Ordered by what the current numbers cannot tell us.

1. **Field set from a real voice.** Every term number today is Kokoro TTS. Record the 300
   held-out-term sentences (`research/record_terms/record.py`), recognise them with Parakeet and
   freeze them as `term_field`. This is the number that says whether the synthetic term gains
   survive a real microphone, room and accent.
2. **More than one speaker.** All real audio is one person. Collect the same script from a few other
   developers and report per-speaker, so a gain that is really one voice is visible.
3. **Rolling real holdout.** Refresh `real_580` from dictations after a moving cutoff (the app's local
   corpus with intended-text labels), keep the cutoff out of training, and track the trend per
   release instead of one frozen snapshot.
4. **Contamination check in the bench.** Hash every eval row and term and refuse an adapter whose
   training corpus contains one. `wispr_edit25` was 60 percent contaminated and nobody noticed until
   a provenance audit.
5. **Retrieval at product scale.** Score recall at budget on real per-user indexes (hundreds of
   thousands of candidates, ranked and capped) and on pools above 40k terms, where recall at 30
   falls to 0.72 before v2 and the refiner is no longer the constraint.
6. **Cross-platform numerics.** Run the same fixtures on the MLX Metal build on a Mac and the GGUF
   build under llama.cpp on Windows and Linux; report agreement with bf16 and between platforms, and
   gate a release on the gap. Today the MLX-to-transformers bf16 gap is KL 6.3e-4 on the GPU box only.
7. **On-device latency.** Seconds from end of speech to final text on target hardware (Apple silicon,
   a Windows laptop, a discrete GPU), with resident memory, as a bench stage tagged by device.
8. **A seconds-scale tier.** Multiprocess retrieval so the full run is under a minute, and add a
   stratified smoke subset for inner-loop checks; keep the full sets for the gate.
9. **Deterministic gate mode.** Batched greedy decode drifts with batch shape; the gate should fix
   the batch or run batch 1 so a release number reproduces exactly.
10. **Classifier fixes.** The ORTHOGRAPHY class over-counts about 4x because it counts chunks, not
    words; fix it before using it to steer training.
11. **Move the bench into `tools/`** and run it on every release candidate.
