# Aqua parity baseline scorer

`tools/aqua_baseline.py` is a no-model, no-network, standard-library scorer for
explicit evaluation JSONL files. It registers the aggregate protocol only. It
never owns or copies the frozen slice; the owner keeps rows, hypotheses, audio,
and manifests on local or compute-host storage and passes paths to the CLI.

## Inputs

```bash
python tools/aqua_baseline.py \
  --rows /path/to/frozen_rows.jsonl \
  --candidate /path/to/candidate_hypotheses.jsonl \
  --baseline /path/to/baseline_hypotheses.jsonl \
  --manifest /path/to/frozen_manifest.json \
  --metadata /path/to/run_metadata.json \
  --output /path/to/receipt.json
```

`--baseline`, `--manifest`, `--metadata`, and `--schema-config` are optional.
The default contract is:

```json
{
  "schema_version": 1,
  "rows": {"id": "id", "reference": "ref"},
  "candidate": {
    "id": "id",
    "hypothesis": "hyp",
    "generation_seconds": "generation_seconds",
    "end_to_end_seconds": "end_to_end_seconds"
  },
  "baseline": {"id": "id", "hypothesis": "raw_aqua"},
  "manifest": {"id": "id"},
  "metadata": {"allowed_fields": ["model", "model_revision", "runtime", "runtime_revision", "kernel_mode", "dataset_revision", "manifest_revision"]}
}
```

`--print-schema-config` prints this object. A custom config must contain the
same keys and version. Field mappings name top-level JSON fields; timing
mappings may be `null`, and any timing missing from a row is skipped. A
manifest may be a JSON array, a `{"rows": [...]}` object, or JSONL. If it has
mapped IDs, its ID set must equal the frozen-row ID set exactly.

## Row contract

A stable row ID is a non-empty JSON string with no leading or trailing
whitespace. Prefer a namespaced zero-padded form such as `aqua:0001`; the
scorer deliberately does not infer IDs from order, audio bytes, text, or a
hash. The frozen-row ID set is authoritative. Candidate and baseline files must
have no duplicate IDs and no missing or extra IDs. A reference must be a string
containing non-whitespace text. Hypotheses may be empty.

The default mappings are row `id`/`ref`, candidate `id`/`hyp` plus
`generation_seconds` and `end_to_end_seconds`, and baseline
`id`/`raw_aqua`. Extra row fields are allowed but are neither hashed nor
emitted.

## Metrics

Fair WER and strict WER are corpus-level token edit distances divided by total
reference tokens. Fair normalization is the registered Whisper English
normalizer, vendored with its public spelling table under
`tools/aqua_baseline_data/`. Strict normalization lowercases and trims
whitespace but preserves case-sensitive punctuation and formatting otherwise.
Exact rate is the fraction where fair-normalized reference and hypothesis are
equal. Empty normalized text participates as the scorer's `<empty>` sentinel.
Latencies use linear interpolation: `p50` is the 0.50 quantile and `p95` is the
0.95 quantile.

With a baseline, wins, ties, and losses compare candidate and baseline fair WER
on each stable ID; `worse_than_input_rate` is losses divided by rows. Without
one, all four fields are `null`. Aggregate timing percentiles are returned even
when no matching values exist, with count `0` and null percentiles.

## Receipt and privacy

The receipt records UTC time, schema version and mapping, input paths and
SHA-256s, row and manifest counts, aggregate metrics, timing aggregates, and
explicit non-claims. It never emits references, hypotheses, per-row timing
values, audio, or other transcript material. Metadata is allowlisted to short
scalar provenance fields; transcript-shaped keys are rejected. A path in a
receipt is still potentially sensitive, so publish or move a receipt only after
applying the repository's evidence policy.

This tool proves only that the supplied files satisfy the row contract and
reports their aggregate scores. It does not identify the registered frozen
500-row slice, validate audio, execute a model, or authorize a product-quality
claim.
