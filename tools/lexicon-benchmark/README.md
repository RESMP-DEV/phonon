# Technical lexicon benchmark

This tool ranks technical terms by a measured TTS-to-ASR difficulty signal and
keeps user dictation out of the benchmark.

## Public baseline

The shipped profile-miner lexicon contains public GPU, CUDA, developer-tool,
PyPI, Rust, Homebrew, and Hugging Face names with two-voice Kokoro to Parakeet
oracle results:

```sh
tools/lexicon-benchmark/benchmark.py \
  --out ~/.local/share/phonon-research/lexicon-benchmark/aggregate-report.json \
  --emit-terms ~/.local/share/phonon-research/lexicon-benchmark/ranked-terms.jsonl
```

The aggregate report contains counts only. Term rows and spoken forms remain
local.

## Local inventory extension

`collect_local.py` inventories the local toolbelt, executable command names,
macOS manual-page command/concept names, Homebrew taps, CLI extensions, and
registry-sourced Cargo/npm/PyPI packages. It ignores editable and path-sourced
packages, package-lock links, generic English words, script filenames, and the
shipped lexicon. It does not read chats, transcripts, source text, or Aqua
data.

```sh
local=~/.local/share/phonon-research/lexicon-benchmark/local
tools/lexicon-benchmark/collect_local.py \
  --out "$local/candidates.json" \
  --terms "$local/terms.json" \
  --lexicon "$local/lexicon.json" \
  --workspace-root /path/to/development/root
```

Run the Linux oracle from `tools/profile-miner/lexicon/oracle_linux.py` against
`terms.json`, then merge the local cache with the shipped baseline. The
collector refuses tracked-repository output paths; use the local research share
or the ignored repository `build/` root. Rows with fewer than two non-null
voices are retained as incomplete evidence but excluded from risk aggregates.

```sh
tools/lexicon-benchmark/benchmark.py \
  --out "$local/aggregate-report.json" \
  --extra-lexicon "$local/lexicon.json" \
  --extra-spoken "$local/spoken.jsonl" \
  --extra-manifest "$local/candidates.manifest.json" \
  --emit-terms "$local/ranked-terms.jsonl" \
  --emit-stratified "$local/stratified-200.jsonl" \
  --stratified-size 200
```

The stratified slice balances ASR class, origin, and category. It is intended
for local review and future correction-model evaluation, not for publication
without term-by-term consent.
