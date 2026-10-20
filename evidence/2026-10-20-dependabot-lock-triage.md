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
package blocks. No package block is added or removed. Besides the two version
changes, the `aiohttp` 3.14.4 block adds one dependency edge to
`typing-extensions`.

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

Retained PR #34 evidence hashes:

| Artifact | SHA-256 |
| --- | --- |
| `local-validation.log` | `ff0f71b004b492b3bde40fd11da0d9e8666582dfafb86b8cb4ce48df28f4ba3f` |
| `runtime-validation.log` | `fbdf34052e439ac0c35b90598fb1be2cebaa35b4572d1443eddd814a22e64af0` |
| `local-validation.exit` | `8b56c7c303900c3e9d89641d31c8840e3e37eae33f322f4351ac6837c08b7174` |
| `runtime-validation.exit` | `19eaf43821a7660ec323a87c8457bf74823beb296c39f5e01aa8a683aa50f061` |

The install/test/lint/audit chain completed in 3.79 seconds and the runtime
import/CLI/tree chain completed in 0.999 second. Their full logs are the four
hashed artifacts above.

## PR #33: closed, constraint conflict

`nemo-toolkit 2.3.0` requires `fsspec==2024.12.0`, and `lightning 2.4.0`
requires `fsspec[http]<2026.0,>=2022.5.0`. `fsspec 2026.6.0` therefore cannot
satisfy the current graph. Forcing it with
`uv lock --upgrade-package fsspec==2026.6.0` reduced the lock from 206 to 181
packages: 25 distinct packages disappear and none enter. This is a
dependency-set replacement rather than a version bump. The retained probe log
is `build/reports/pr33/probe.log`, SHA-256
`1e24ed6d0235ece199a3fd6077322eaf8100b68ef90561746a4f2e4e027c83c5`. The PR
was closed with that evidence and a reopen condition.

## PR #32: closed, constraint conflict

`nemo-toolkit 2.3.0` constrains `hydra-core<=1.3.2,>1.3` for every relevant
extra, so `hydra-core 1.3.7` is not installable while that NeMo dependency set
remains. Forcing it with `uv lock --upgrade-package hydra-core==1.3.7`
reduces the lock from 206 to 176 packages: 40 distinct packages disappear, 10
enter, and the existing `urllib3` name is rewritten (which produces 41 removal
lines and 11 addition lines in the raw lock diff). The disappearing set
includes `bitsandbytes`, `peft`, `optuna`, `lhotse`, `lightning`, and
`pydantic`. The retained probe log is `build/reports/pr32/probe.log`,
SHA-256
`ff7b4dee0e0667d7444b92eb61e3ecb015bab0fbff04f7027ec9d4e386fd738a`. The PR
was closed with that evidence and a reopen condition.

## Verification

Recorded UTC: 2026-10-20T01:32Z-01:56Z. Platform: macOS main machine. At
documentation head `2e09375eead74eb773b909e1e94078d6e691fa2a`, `just quality`
exited 0 with `source_dirty: false`; the read-only inventory receipt is
`build/reports/dependabot-lock-triage/receipt.json`. Any later wording-only
repair commit is gated again at its own clean head before merge.

| Check | Command or procedure | Exit | Elapsed | Log |
| --- | --- | ---: | ---: | --- |
| source status | `git status --short --branch` | 0 | 18 ms | `build/reports/dependabot-lock-triage/source-status.log` |
| PR34 package count | `git show 27563a15b2ff330db3125e1eb715fc4986438f35:ml/uv.lock` | 0 | 9 ms | `build/reports/dependabot-lock-triage/pr34-package-count.log` |
| PR34 removed blocks | `git diff 944d7584143cdb496343fdb1ef148643b8615a85 27563a15b2ff330db3125e1eb715fc4986438f35 -- ml/uv.lock` | 0 | 15 ms | `build/reports/dependabot-lock-triage/pr34-package-block-removals.log` |
| PR34 added blocks | `git diff 944d7584143cdb496343fdb1ef148643b8615a85 27563a15b2ff330db3125e1eb715fc4986438f35 -- ml/uv.lock` | 0 | 14 ms | `build/reports/dependabot-lock-triage/pr34-package-block-additions.log` |
| PR34 artifact hashes | `shasum -a 256 -c build/reports/pr34/hashes.txt` | 0 | 13 ms | `build/reports/dependabot-lock-triage/pr34-hash-verify.log` |
| PR33 probe hash | `shasum -a 256 build/reports/pr33/probe.log` | 0 | 12 ms | `build/reports/dependabot-lock-triage/pr33-probe-hash.log` |
| PR32 probe hash | `shasum -a 256 build/reports/pr32/probe.log` | 0 | 13 ms | `build/reports/dependabot-lock-triage/pr32-probe-hash.log` |
| referenced paths | `test -f` for receipt, lock, and PR34 hashes | 0 | 3 ms | `build/reports/dependabot-lock-triage/referenced-paths.log` |
| privacy scan | `rg` for private paths, row IDs, transcript/audio markers | 0 | 6 ms | `build/reports/dependabot-lock-triage/privacy-scan.log` |
| diff check | `git diff --check` | 0 | 8 ms | `build/reports/dependabot-lock-triage/diff-check.log` |

The historical PR34 execution logs and the PR32/PR33 forced-lock probes are
represented by their retained artifacts and hashes above. The upstream
requirement statements were read from live PyPI JSON metadata during triage and
independently cross-checked from locally cached uv wheel metadata by the review.

## Non-claims

This triage does not claim that `aiohttp 3.14.4` is free of upstream defects;
the production audit surface for the Python ML environment was not exercised.
It does not prove the ML research environment has functional test coverage,
which is a separate gap. It does not upgrade NeMo, Lightning, or the research
dependency set, so `hydra-core` and `fsspec` remain at their constrained
versions. It does not verify behavior of the optional `nemo`, `whisper`, or
`transformers-asr` extras beyond dependency resolution.
