# Phonon worktree prune and workflow inventory, 2026-10-20

## Scope

This receipt records the controlled pruning of stale Phonon worktrees, the
proof used to distinguish integrated work from unmerged work, the preservation
location for local-only evidence, and the closeout of the then-open Next.js
dependency PR. It contains no private transcript, row ID, per-clip audio path,
dictionary value, or lexicon term.

## Worktree decisions

| Worktree | Branch / commit | Decision proof | Result |
| --- | --- | --- | --- |
| `pr13-merge` | `merge/pr13` at `257f892` | `git merge-base --is-ancestor merge/pr13 main` passed | Integrated; checkout removed |
| `private-audio-augmentation` | `work/private-audio-augmentation` at `f86e716` | PR #25 merge commit `51f728f`; `git diff --quiet 51f728f f86e716` passed | Tree exactly integrated; checkout and stale local branch removed |
| `private-corpus-curation` | `work/private-corpus-curation` at `7642a23` | PR #26 merge commit `a79b298`; `git diff --quiet a79b298 7642a23` passed | Tree exactly integrated; checkout and stale local branch removed |
| `train-adapt-v4` | `ml/portable-harness` at `81d5b45` | Clean checkout, no process owner, and branch matched `origin/ml/portable-harness` | Checkout removed; unmerged branch retained |
| `validated-insertion` | `work/validated-insertion` at `2bef056` | Branch has one unmerged feature commit and local changes; no matching implementation symbols exist on `main` | Checkout removed only after exact patch/probe preservation; branch retained |

The stale `.worktrees/worker-receipts` directory was not a registered Git
worktree. Its receipts were copied to the legacy preservation area and the
directory was removed.

## Preservation

Local-only reports, review artifacts, differing PR13 scaffold versions, the
validated-insertion binary patch, its untracked probe, and two branch bundles
were preserved under the ignored primary-checkout path
`build/legacy-worktrees/20261020/`. Its machine manifest records 91 files,
9,244,417 bytes, and SHA-256
`4b0da7e265f469a970acf411ca3d3739b6b344cb7e4a9feefed3019482e5db84`.

The private parquet audit artifact from the PR #25 review area was preserved
locally and was not published. Compiler and package caches were not retained.

## Branch inventory after pruning

- `main` at `4187466e`: clean and synchronized with `origin/main`.
- `ml/portable-harness` at `81d5b45`: retained unmerged research/history lane,
  126 commits ahead of the then-audited `main`, with a live remote branch and
  a local bundle in the preservation area.
- `work/validated-insertion` at `2bef056`: retained focused unmerged feature
  lane. Its uncommitted repair patch and `tools/insertion-probe` are preserved
  in the legacy area; neither has been reviewed or integrated.

Integrated local branches `merge/pr13`, `work/lexicon-local-sources`,
`work/private-audio-augmentation`, `work/private-corpus-curation`, and
`feat/training-capture-ui` were deleted after ancestry or exact-tree merge
proof. The two retained branches above are not dangling work; they are explicit
resume lanes without active checkouts.

## Next.js workflow closeout

Dependabot PR #23 bumped `website` from Next.js 16.3.7 to 16.3.8. It was
validated at head `a284dcff` under the project-pinned Node runtime:

| Check | Result |
| --- | --- |
| Runtime | Node `v22.23.3`, npm `12.2.0` |
| `npm ci` | exit 0 |
| `npm test` | production Next build and rendered-HTML test pass, 1/1 |
| `npm run lint` | exit 0 |
| `npm audit --omit=dev --audit-level=high` | exit 0; no high findings |

The audit still reported two moderate PostCSS advisories inherited through
Next.js; they were not introduced by this patch and remain dependency follow-up
work. PR #23 merged as `4187466e8a2ee7835b467c0f984a70a73c94fb71`. Validation
logs and hashes remain under ignored `build/reports/pr23/`.

## Verification

Recorded UTC: 2026-10-20T01:00Z-01:06Z. Platform: macOS main machine. Source
revision at final common-gate verification: `4187466e`; dirty state false.

| Check | Command or procedure | Result |
| --- | --- | --- |
| Integrated ancestry | `git merge-base --is-ancestor <branch> main` | passed for `merge/pr13` and `work/lexicon-local-sources` |
| Exact squash-tree proof | `git diff --quiet <merge-commit> <branch-tip>` | passed for PR #25, PR #26, and PR #15 branch tips |
| Process safety | `lsof +D <worktree>` before removal | no live working-directory owner |
| Preservation integrity | hash manifest and `git bundle verify` | 91 files preserved; both bundles valid |
| Worktree cleanup | normal `git worktree remove` for five worktrees | all succeeded without force |
| Website dependency gate | Node 22.23.3 `npm ci`, `npm test`, `npm run lint`, focused audit | all exit 0 |
| Repository common gate | `just quality` at `4187466e` | exit 0, PASS, clean source |

## Non-claims

This inventory does not claim `ml/portable-harness` is obsolete or reviewed; it
remains an unmerged lane. It does not review, test, or integrate the preserved
validated-insertion patch. Preserved private artifacts remain local and are not
published by this receipt. The Next.js production audit does not clear the two
moderate PostCSS advisories. No model, ASR, latency, or daily-driver gate was
re-run by this cleanup.
