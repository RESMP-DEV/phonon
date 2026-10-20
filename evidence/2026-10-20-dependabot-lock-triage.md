# Dependabot lock triage for the ML environment, 2026-10-20

## Scope

This receipt records the triage of three Dependabot pull requests that all
targeted the single tracked `ml/uv.lock` file, the decision to merge one and
close two, and the dependency-constraint evidence behind those decisions. It
contains no personal data, private paths, transcripts, or model artifacts.

## Why triage was needed

Three `uv` dependency PRs opened within a minute of each other and all rewrote
the same lockfile. Because a lockfile rewrite can silently drop or replace the
pinned research dependency set, package-count and package-set deltas were
measured before any merge rather than trusting the PR title.

| PR | Package | Target version | Lock package count vs `main` | Decision |
| --- | --- | --- | ---: | --- |
| #34 | `aiohttp` | 3.13.5 to 3.14.4 | 206, delta 0 | Merged |
| #33 | `fsspec` | 2024.12.0 to 2026.6.0 | 181, delta -25 | Closed |
| #32 | `hydra-core` | 1.3.2 to 1.3.7 | 176, delta -30 | Closed |

## PR #34: merged

`aiohttp` 3.13.5 to 3.14.4 with its `yarl` pin 1.24.2 to 1.25.1 was the only
one of the three that did not remove packages: the lock still resolves 206
packages, identical in count and content apart from those two versions.

Acceptance run at head `cc1abed87cb9b16845f9c7a2896089d90b81160c`:

| Check | Command or procedure | Exit | Result |
| --- | --- | ---: | --- |
| Lock consistency | `uv lock --check` in `ml/` | 0 | 206 packages resolved |
| Environment install | `uv sync` in `ml/` | 0 | 54 packages installed, `aiohttp 3.14.4`, `yarl 1.25.1` |
| Runtime import | `uv run python -c 'import aiohttp, yarl'` | 0 | `aiohttp 3.14.4`, `yarl 1.25.1` |
| Project CLI | `uv run phonon --help` | 0 | CLI help rendered |
| Dependency tree | `uv tree --package aiohttp` and `uv tree --invert --package aiohttp` | 0 | dependency edges intact |
| Repository common gate | `just quality` at the PR head | 0 | PASS, clean source |

The default `uv run pytest` in `ml/` collects zero tests and exits 5, so it was
not used as acceptance evidence for a lock-only change. It remains a real gap
in the ML project's own test surface.

PR #34 merged as `27563a15b2ff330db3125e1eb715fc4986438f35`.

## PR #33: closed, constraint conflict

`nemo-toolkit 2.3.0` requires `fsspec==2024.12.0`, and `lightning 2.4.0`
requires `fsspec[http]<2026.0,>=2022.5.0`. `fsspec 2026.6.0` therefore cannot
satisfy the current graph. Forcing it with
`uv lock --upgrade-package fsspec==2026.6.0` reduced the lock from 206 to 181
packages and removed 25 existing entries, which is a dependency-set
replacement rather than a version bump. The PR was closed with that evidence
and a reopen condition.

## PR #32: closed, constraint conflict

`nemo-toolkit 2.3.0` constrains `hydra-core<=1.3.2,>1.3` for every relevant
extra, so `hydra-core 1.3.7` is not installable while that NeMo dependency set
remains. Forcing it with `uv lock --upgrade-package hydra-core==1.3.7`
reduced the lock from 206 to 176 packages and removed 41 entries, including
`bitsandbytes`, `peft`, `optuna`, `lhotse`, `lightning`, and `pydantic`. The PR
was closed with that evidence and a reopen condition.

## Verification

Recorded UTC: 2026-10-20T01:32Z-01:40Z. Platform: macOS main machine. Source
revision at final common-gate verification: `27563a15`; dirty state false.

| Check | Command or procedure | Exit | Result |
| --- | --- | ---: | --- |
| PR34 lock consistency | `uv lock --check` in `ml/` | 0 | 206 packages |
| PR34 sync and runtime | `uv sync`, `uv run python`, `uv run phonon --help` | 0 | install, import, and CLI all succeed |
| PR34 artifact hashes | `shasum -a 256 build/reports/pr34/*` | 0 | four hashes recorded |
| PR33 constraint probe | `uv lock --upgrade-package fsspec==2026.6.0` | 0 | 181 packages, 25 removals |
| PR32 constraint probe | `uv lock --upgrade-package hydra-core==1.3.7` | 0 | 176 packages, 41 removals |
| Upstream requirement read | PyPI JSON metadata for `nemo-toolkit` 2.3.0, `datasets` 5.0.1, `lightning` 2.4.0, `pytorch-lightning` 2.6.4 | 0 | exact pins and caps recorded above |
| Repository common gate | `just quality` at `27563a15` | 0 | PASS, clean source |

## Non-claims

This triage does not claim that `aiohttp 3.14.4` is free of upstream defects;
the production audit surface for the Python ML environment was not exercised.
It does not prove the ML research environment has functional test coverage,
which is a separate gap. It does not upgrade NeMo, Lightning, or the research
dependency set, so `hydra-core` and `fsspec` remain at their constrained
versions. It does not verify behavior of the optional `nemo`, `whisper`, or
`transformers-asr` extras beyond dependency resolution.
