# Dependabot triage and postcss advisory repair, 2026-10-23

## Scope

This receipt records the triage and merge of the three open RESMP-DEV
Dependabot pull requests at the time of the work, plus the follow-up repair of
the pre-existing `postcss` override advisory that the website validation
surfaced. It contains no personal data, private paths, transcripts, or model
artifacts.

## Queue state

A live organization-wide search for open pull requests authored by
`app/dependabot` found exactly three, all in this repository:

| PR | Change | Base package set | Head package set | Decision |
| --- | --- | ---: | ---: | --- |
| #37 | `gitpython` 3.1.62 -> 3.2.1 in `ml/uv.lock` | 206 | 206 | Merged |
| #38 | `pillow` 12.2.0 -> 13.0.0 in `ml/uv.lock` | 206 | 206 | Merged |
| #39 | `next` ^16.3.8 -> ^16.4.1 in `website/` | 436 | 436 | Merged |

For #37 and #38, the complete `[[package]]` name sets were identical before and
after: no package was added or removed and only the target version changed.
This is the opposite of the 2026-10-20 lockfile rewrites that dropped 25-40
packages and were closed. For #39, the complete npm `packages` key set was
identical (436 entries); only `next` and its nine `@next/*` platform packages
changed version.

## PR #37: gitpython 3.2.1

`gitpython` is a transitive dependency of `wandb` 0.27.0 via
`nemo-toolkit` 2.3.0 under the optional `nemo` extra; no repository file
imports `git` directly, and a default `uv sync` installs 54 distributions
without gitpython. Acceptance therefore combined the lock proof with an
isolated consumer smoke.

At head `875311e6da1633351186513725bea1e0e6365293`:

| Check | Result |
| --- | --- |
| `uv lock --check` in `ml/` | exit 0, 206 packages |
| `uv sync` in `ml/` | exit 0, 54 distributions, gitpython absent from default env |
| `uv run phonon --help` | exit 0 |
| `uv tree --package gitpython` / `--invert --package gitpython` | exit 0, chain confirmed |
| Isolated smoke: `gitpython==3.2.1` + `wandb==0.27.0` | exit 0; `git.Repo` read the exact head, reported detached HEAD and clean tree |
| Lock-format compatibility | `uv lock --check` exit 0 under uv 0.7.0, 0.8.0, 0.9.0, and 0.13.1 (lock revision 3 -> 5 is readable by older uv) |
| `just quality` | exit 0, clean source |

The lockfile's `revision` bump from 3 to 5 is a side effect of Dependabot
regenerating with a newer uv; the repository pins no uv version, and the
compatibility probe above bounds the risk.

## PR #38: pillow 13.0.0

`pillow` is a transitive dependency of `matplotlib` 3.10.9, `mediapy` 1.1.6,
and `tensorboard` 2.20.0, all reached only through the optional `nemo` extra;
no repository file imports PIL directly, and the default `uv sync` does not
install pillow. Pillow 13.0.0 (2026-10-16) fixes numerous CVEs; its complete
upstream backwards-incompatible list was enumerated and audited.

At head `16dc675724f34d1077d587f15335cd2102513c92`:

| Check | Result |
| --- | --- |
| `uv lock --check` in `ml/` | exit 0, 206 packages |
| `uv sync` in `ml/` | exit 0, default env does not install pillow |
| `uv run phonon --help` | exit 0 |
| `uv tree --package pillow` / `--invert --package pillow` | exit 0, consumers confirmed |
| Isolated smoke: pillow 13 + matplotlib 3.10.9 + mediapy 1.1.6 (numpy 1.26.4) | exit 0; numpy RGB -> `Image.fromarray` -> PNG round trip pixel-exact; Agg figure -> PNG -> PIL reopen valid; mediapy conversion returned correct shape |
| Wheel installability | exit 0 on CPython 3.11 and 3.12 |
| Removed-API audit | no consumer reaches a removed Pillow 13 API |
| `just quality` | exit 0, clean source |

The removed-API audit examined installed consumer source. `mediapy`'s only
`fromarray(mode=...)` wrapper is always called with `mode=None`; matplotlib's
twelve `fromarray` calls pass no keywords; its `_show` hit is its own
`FigureManagerMac._show`, not `PIL.Image._show`. No consumer references
`get_child_images`, `ImageCmsProfile.product_name`/`product_info`, or
`ExifTags.IFD.Makernote`. A directed probe also proved the removed I-mode PNG
clipping is now a hard error, and that no consumer can create an I-mode image
(mediapy rejects int32, matplotlib uses RGB(A), tensorboard only calls
`Image.open`). The Linux GPU box that runs the `nemo` extra has glibc 2.43, so
Pillow 13's dropped `manylinux2014` wheels are not a constraint there.

## PR #39: next 16.4.1

At head `716af16db2813dcbec8f79962df41692e3dd5323`, under both the machine's
Node 26 and the declared Node 22 engine:

| Check | Node 26 | Node 22 |
| --- | ---: | ---: |
| `npm ci` | exit 0 | exit 0 |
| `npm run lint` | exit 0 | exit 0 |
| `npm run test` (build + rendered HTML) | exit 0, 1/1 pass | exit 0, 1/1 pass |
| `npm ls next` | `next@16.4.1` | `next@16.4.1` |
| `just quality` | - | exit 0, clean source |

The rendered-HTML test asserts marketing copy, download URLs, and element
ordering; no assertion is Next.js-version-specific. The validation surfaced one
pre-existing advisory: `npm audit --omit=dev` reported 2 moderate findings
because the repository's `"postcss": "8.5.21"` override downgraded postcss
below the fixed 8.5.23 that `next@16.4.1` itself requests exactly. The advisory
predated this PR; main resolved postcss 8.5.21 both before and after the Next
bump.

## Merges and assembled-main verification

All three PRs were merged with unchanged pinned heads, GitHub `MERGEABLE` /
`CLEAN`, no unresolved review threads, and CodeRabbit reporting success:

| PR | Head | Merge commit |
| --- | --- | --- |
| #37 | `875311e6da1633351186513725bea1e0e6365293` | `287d0e45194d591e50bfe632c036aefe48085200` |
| #38 | `16dc675724f34d1077d587f15335cd2102513c92` | `9b682704c00b406aaad37aa5426b7eb3db91d38e` |
| #39 | `716af16db2813dcbec8f79962df41692e3dd5323` | `bb0b769372aa4b0a0b71fca3a4e54f38eb507c47` |

After #37 merged, GitHub recomputed #38 as `MERGEABLE` / `CLEAN` before it was
merged. At assembled main `bb0b769372aa4b0a0b71fca3a4e54f38eb507c47`, in a
clean detached worktree:

- `uv lock --check` exited 0 with 206 packages, `gitpython` 3.2.1, and
  `pillow` 13.0.0.
- `just quality` exited 0 with a clean source tree.
- Under Node 22, `npm ci`, `npm run lint`, and `npm run test` exited 0
  (1/1 rendered-HTML pass) and `next@16.4.1` resolved.

The primary checkout's local `main` had already diverged from `origin/main`
with seven unpushed granite/gpumode commits touching only
`docs/`, `evidence/`, and `tools/deepgram-annotation/`. It was left untouched;
those commits do not modify either lockfile or `website/package.json`, so the
package-set proofs above are unaffected.

## Postcss override repair

Checkout state for the checks in this section: base revision
`bb0b769372aa4b0a0b71fca3a4e54f38eb507c47` with the override, lock,
architecture, and evidence changes applied as a dirty working tree
(`git status --short` showed exactly those four paths modified or added, plus
ignored build artifacts). The committed repair head is
`c980a5cd13a3025cc693e586ac036f71b9e56225`; the clean final-head re-run is
recorded in the pull-request review evidence.

The minimal repair changes `website/package.json`'s override from
`"postcss": "8.5.21"` to `"postcss": "8.5.23"`, exactly matching the pin
declared by `next@16.4.1`. The npm package set remains 436 entries with no
package added or removed; only `node_modules/postcss` changes
(8.5.21 -> 8.5.23). Under the declared Node 22 engine: `npm ci`, `npm run
lint`, and `npm run test` exited 0 (1/1 rendered-HTML pass), `npm ls postcss`
resolved 8.5.23 for both `@tailwindcss/postcss` and `next`, and
`npm audit --omit=dev` exited 0 with zero vulnerabilities.

The lock was regenerated with npm 11.20.0 to avoid an unrelated lock-format
rewrite: npm 10.9.9 removed 112 `libc` fields when regenerating, while npm
11.20.0 produced a three-line postcss-only lock delta.

## Non-claims

- The optional `nemo` extra was not installed; the full Linux GPU runtime path
  for gitpython, pillow, wandb, matplotlib, mediapy, and tensorboard was not
  exercised end to end. Isolated consumer smokes and installed-source audits
  bound the removed-API risk but do not measure NeMo behavior.
- Pillow codec coverage is PNG only; JPEG, WebP, TIFF, and mediapy's video path
  were not exercised.
- Website validation covers the declared Node 22 engine and this macOS arm64
  host. The other eight Next SWC platform binaries were resolved in the lock
  but not compiled or run.
- A clean `npm audit` is not a general security guarantee; it covers npm's
  advisory database for the audited dependency tree only.
- No Rust, Swift, permissions, packaging, model, OCR, dictation, or UI-visual
  gates were run for these dependency changes.
- Merging these updates does not claim the upstream releases are defect-free.
