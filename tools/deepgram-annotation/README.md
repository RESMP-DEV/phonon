# Deepgram annotation builder

Offline tool that builds a group-disjoint manifest and runs the four-arm
Deepgram annotation over already assigned train/eval/test rows. All output
stays local: the raw provider
payloads are archived under a content-addressed directory and the normalized
rows are appended to a JSONL sidecar. The macOS keyring is the only source of
the API key and the key is never printed, logged, or written to disk by this
tool.

## Manifest build

Source records are JSONL objects carrying at least ``row_id``,
``audio_sha256``, ``duration_seconds``, ``leakage_group``, ``source_kind``,
and ``source_id``. Audio arrives as either an extracted ``audio_path`` or
inline ``audio_bytes_base64``; audio extraction itself is handled upstream.
For the two-way ``build`` verb, rows that declare a ``split`` keep it and rows
without a declared split are grouped into whole leakage groups assigned so
train and eval each hit an exact duration target. The ``run`` verb accepts a
manifest containing any subset of ``train``, ``eval``, and ``test``; the
Phonon dataset builder supplies all three.

```sh
python tools/deepgram-annotation/cli.py build \
  --input path/to/source.jsonl \
  --output build/deepgram/manifest.jsonl \
  --report build/deepgram/report.json \
  --train-target-ms 10863457 \
  --eval-target-ms 2272178
```

The report is aggregate-only: row counts, leakage-group counts, duration
totals, and source-kind counts. It never contains row IDs, audio paths, or
transcripts.

## Annotation run

```sh
python tools/deepgram-annotation/cli.py run \
  --manifest build/deepgram/manifest.jsonl \
  --output build/deepgram/annotations.jsonl \
  --archive build/deepgram/raw \
  --receipt build/deepgram/receipt.json \
  --keyterms path/to/keyterms.json \
  --concurrency 5
```

The runner reads the API key from the deepctl macOS keyring account
``api-key.default`` under service ``com.deepgram.dx.deepctl``. It retries
HTTP 429 and 5xx responses with an exponential backoff and honours any
``Retry-After`` header. Concurrency is bounded to 5 by the provider's
combined-intelligence limit.

## Provenance and resume

Each normalized annotation row records the row and audio hash, the arm, the
exact request parameters, the full provider payload hash, the provider
request ID, the model identity the provider reported, word timestamps and
confidence, punctuated surfaces, speaker turns, utterances, paragraphs and
sentences, entities, topics, intents, sentiment, summary text, and provider
timings. Raw responses are archived under
``<archive>/<audio_sha256>/<row>_<arm>_<payload_sha256>.json``.

A row/arm identity is stable across runs, so re-running the same command
skips completed rows and only issues requests for the remainder. Rows that
fail a validation gate are not appended; the failure is recorded in the
aggregate receipt.

## Validation gates

Every normalized row is checked before it is appended. The gates verify that
the row and audio hash match the manifest, the request pins
``model=nova-3``, ``version=2025-07-31.0``, and ``mip_opt_out=true``, smart
arms request every rich feature, raw arms request none of them, word and
utterance timestamps are within the audio duration, word confidence stays in
``[0, 1]``, and the response carries a provider request ID, returned model
UUID, transcript, and non-empty payload hash. Nova can project the final word
of a clipped source beyond its physical boundary; those rows are retained and
marked in ``quality.flags`` rather than discarded.

## Privacy

Requests always set ``mip_opt_out=true``, which places them in Deepgram's
zero-retention lane. Request tags are opaque 16-hex digests; they never carry
row IDs, source identifiers, or transcript text. The receipt is
aggregate-only and the keyterm metadata records the profile ID, hash, and
term count but never the terms themselves. Never place personal audio,
transcripts, or accepted text in a tracked receipt, a PR description, or a
public artifact.

## Tests

```sh
uv run --no-project --python 3.12 --with pytest \
  pytest tools/deepgram-annotation/tests -q -p no:cacheprovider
```

The test suite covers manifest validation, group-disjoint split construction,
arm parameter contract, keyring credential lookup, HTTP retry behaviour on
429 and 5xx, response normalization against a fixture, per-arm validation
gates, idempotent resume, and aggregate-only receipts. No live provider call
is made.
