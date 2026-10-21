# Deepgram offline dataset-labeling probe, 2026-10-21

## Corrected scope

This receipt supersedes the interaction-oriented emphasis in the earlier
voice-intelligence survey for dataset-building decisions. Deepgram is evaluated
here as an **offline annotation front end**: a source of transcript variants,
word alignment, confidence, segmentation, weak structured labels, and
provenance for SALM training. Realtime behavior, conversational dynamics, and
open-versus-closed ownership are not selection criteria for this lane.

The parent and two additional read-only CCR workers made no repository code or
runtime changes. No personal Phonon dictation audio was uploaded. Probes used
only repository-owned synthetic speech, a synthetic text sentence, and
Deepgram's public spacewalk sample.

## Live access and reproducibility

The updated native CLI is `deepctl 0.4.0`. Its keyring credential authenticates
to project `7e2c808e-d62a-43fc-b135-0a3b5aa14e78`; the displayed credential
expiry is 2026-10-22 16:23:02 UTC. The model catalog call succeeded and listed
567 model records.

For English offline annotation, `model=nova-3&version=2025-07-31.0` resolved
reproducibly to `general-nova-3`, architecture `nova-3`, model UUID
`2187e11a-3532-4498-b076-81fa530bdd49`. Selecting that UUID directly returned
`403 INSUFFICIENT_PERMISSIONS`, so the reproducible request must use the alias
plus version and archive the returned model metadata. Repeated structure probes
kept transcript text and word timestamps stable, while confidence bits and
utterance UUIDs varied slightly; those values are observations, not stable row
identities.

## Verified annotation outputs

A synthetic 6.198-second 16 kHz WAV was generated locally with macOS Speech and
annotated with Nova-3 under `mip_opt_out=true`. The smart-formatted request
returned the exact intended sentence with:

- 16 word records containing `word`, `start`, `end`, `confidence`,
  `punctuated_word`, `speaker`, and `speaker_confidence`;
- one paragraph and sentence span;
- one utterance with confidence and its own UUID;
- custom topic `"Speech dataset annotation"` at confidence 0.22985;
- custom intents `"Preserve raw hypothesis"` at 0.78653 and
  `"Generate corrected prose for dataset"` at 0.01628.

An otherwise paired raw variant with `punctuate=false&smart_format=false`
returned lowercase unpunctuated text and recognized `pros` instead of `prose`
at 0.44873 confidence. The smart-formatted request returned `prose` at 0.96802
confidence. These are two separate provider observations, not a deterministic
local transform; both must be retained with their exact request parameters.

The public spacewalk probes additionally verified diarization v2, paragraph and
utterance segmentation, and entity spans such as `ORDINAL` and
`GENDER_SEXUALITY` with word-index bounds and confidence. Nova-3 rejected
N-best alternatives: it does not support more than one alternative. Unknown
`prompt`, `response_format=json_schema`, and `json_schema` parameters were
silently ignored and returned an ordinary transcript. Therefore Deepgram is not
an arbitrary-prompt or schema-enforced LLM teacher; it is a fixed-output TSLM
annotation service with custom topic/intent constraints.

## Dataset role

Useful records are:

| Deepgram field | Dataset role |
| --- | --- |
| Raw and smart transcript variants | Independent offline ASR hypotheses and formatting variants |
| Word time bounds | Alignment between audio slices and text spans |
| Word confidence | Review priority and weak error signal |
| Speaker and speaker confidence | Anonymous turn segmentation |
| Utterance spans | Primary segmentation for training samples |
| Paragraphs and sentences | Secondary discourse segmentation |
| Entities with word bounds | Weak span labels after time mapping |
| Custom topics/intents | Weak task/domain labels |
| Request ID, model metadata, tags, feature list | Provenance and request-family grouping |

Do not treat a Deepgram transcript as the accepted target by itself. It can
select rows, prioritize review, align spans, condition an explicitly registered
experiment, or contribute evidence to blind reconciliation. The training target
remains reviewed human/intended text. A lower Deepgram WER than another teacher
does not prove that SALM trained from its output will improve.

## Proposed append-only sidecar

Keep the canonical audio row unchanged. Store annotations in a private,
append-only JSONL sidecar keyed by stable row ID and audio SHA-256. One run
record per provider request should retain:

- opaque source ID, row ID, dataset revision, leakage group, and consent scope;
- exact audio hash, duration, channel, and sample rate;
- provider endpoint, full normalized parameter set, request ID, tag, and
  completion time;
- requested alias/version and returned model UUID/name/architecture;
- `mip_opt_out=true`, endpoint/region, and privacy decision version;
- factor condition (`base_raw`, `base_smart`, `keyterm_raw`, `keyterm_smart`);
- keyterm profile ID/hash/count/source without exposing term values in tracked
  receipts;
- raw transcript, smart transcript, words, speaker turns, utterances,
  paragraphs, entities, and document-level intelligence;
- provider latency and, where measured, end-to-end timing;
- raw response path/hash and parse/validation status;
- `supersedes_run_id` for corrections rather than mutation.

Downstream samples should be utterance-level slices with both acoustic bounds
and text provenance. Generated topics, intents, entities, and sentiment remain
`weak`; only explicit human review may promote a candidate to a training
target. Keyterms may come only from raw ASR hypotheses, local consented
dictionaries, or public inventories, never accepted/final text or the frozen
evaluation slice.

## Evaluation design

1. Freeze row IDs, audio hashes, leakage groups, and disjoint train/dev/eval
   permissions before choosing keyterms or tuning anything.
2. Run the four-arm transcription factor on identical audio bytes, storing every
   raw response.
3. Validate audio hash, duration, monotonic timestamps, word bounds, speaker
   bounds, coverage, model version, and parameter identity before ingestion.
4. Blindly reconcile Aqua raw, Deepgram raw/smart, and existing independent
   teachers without exposing accepted text. Accepted-aware outputs can triage
   but cannot prove independent improvement.
5. Prioritize human review by disagreement, low confidence, keyterm delta,
   timestamp anomalies, and rare-term/entity coverage.
6. Compare matched SALM arms: baseline; matched selection without Deepgram;
   Deepgram-feature selection; and blind Deepgram-evidence reconciliation.
   Selection must be evaluated against a non-Deepgram selection arm of the same
   size.
7. Report once on the registered full frozen 500-row accepted-target gate.
   Subset readings cannot promote a lane.

## Privacy, limits, and cost

Default STT requests participate in Model Improvement and retain audio and
transcripts. Every personal-data request must use `mip_opt_out=true`; this makes
audio, text, and transcripts zero-retention after the response. Metadata and
usage logs remain retrievable for 90 days, do not contain audio or transcript
bodies, and must use opaque tags rather than source text or personal names. A
regional endpoint plus opt-out is required for an in-region processing claim.
The live synthetic opt-out request was visible in request history with
`mip_opt_out` among its features.

Nova-3 pre-recorded permits up to 50 concurrent requests on self-serve plans.
Combined intelligence features reduce the limit to the lowest feature limit,
typically 5-10 concurrency. Separate transcription/segmentation from text
intelligence when scaling.

Published PAYG pre-recorded rates are USD 0.0043/minute for Nova-3 English,
0.0052/minute multilingual, plus USD 0.0013/minute for keyterms and
0.0017/minute for entity detection; pre-recorded diarization and smart format
are included. Summarization is token-billed. The current key can read usage but
lacks `billing:read`, so exact invoice balance was not verified. Session usage
across all public/synthetic probes totaled 0.08758 hours.

## Worker receipts

| Piece | SHA-256 | Outcome |
| --- | --- | --- |
| Deepgram dataset contract | `7b2ffe34ad78552e2ae4c10a39070ea71f4d676b283a08dbf22848c38fc2d073` | completed, exit 0 |
| Phonon dataset integration design | `daa521354c5cc7f16bbde8ae40fc677231d75c73a1047a9a19894c748d7f74eb` | completed, exit 0 |

The complete ignored worker reports and CCR run receipts remain under
`build/voice-intel-workers/` and `~/.cache/ccr-worker-runs/`.

## Non-claims

This proves authenticated catalog access, API contracts, synthetic/public
annotation behavior, output structure, rate/retention documentation, and usage
accounting. It does not prove Deepgram WER on private Phonon audio, authorize a
private-audio upload, quantify an invoice balance, train a SALM adapter, or
promote any annotation strategy. No personal transcript, private row ID,
private lexicon term, secret, or owner screen content appears here.
