# Private dataset audit

This tool audits local mirrors of private canonical Phonon ASR Parquets before
they can enter a training or teacher-mining workflow. It verifies embedded
audio hashes, WAV properties, metadata durations, clip intervals, permissions,
label coverage, teacher coverage, and duplicate/cross-dataset identities.

The output is aggregate-only. It never contains row IDs, transcripts, teacher
transcripts, terms, notes, speaker identities, source URLs, or original source
paths. Keep reports in the local research share or under ignored `build/`.
Never place the aggregate itself in tracked repository evidence.

Run it with PyArrow available:

```sh
uv run --no-project --python 3.12 --with pyarrow \
  tools/private-dataset-audit/private_dataset_audit.py \
  --dataset ~/.local/share/phonon-research/hf-private-datasets/example \
  --provenance ~/.local/share/phonon-research/hf-private-datasets/provenance.json \
  --out ~/.local/share/phonon-research/hf-private-datasets/audit.json
```

The `provenance.json` object maps each directory name to its private HF repo ID
and pinned revision. A passing integrity check does not authorize training:
the report separately records whether any rows are both train-permitted and
label-eligible. Rows with `train_allowed=false`, `eval_allowed=true`, duplicate
identity, missing labels, or noncanonical audio remain blocked.
