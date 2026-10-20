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
`build/legacy-worktrees/20261020/`. Its machine manifest inventories exactly 90
payload files totaling 9,228,352 bytes and intentionally excludes itself to
avoid a stale self-reference. The current manifest SHA-256 is
`fcaec719b0316085992bb7a55d7271dd1725c3ab29309c310762e9c8fec8344c`; all 90
payload size/hash entries were re-verified against the files.

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
validated at head `a284dcff` using Node `v22.23.3`, within the
repository-declared Node `22.x` range:

| Exact procedure under Node `v22.23.3` / npm `12.2.0` | Exit | Result |
| --- | ---: | --- |
| `npm ci` | 0 | 354 packages installed |
| `npm test` | 0 | production Next build passed; rendered-HTML test 1/1 |
| `npm run lint` | 0 | ESLint passed |
| `npm audit --omit=dev --audit-level=high` | 0 | no high findings |

The complete isolated Node 22 command chain completed in 24.877 seconds. Its
log, exit marker, and runtime record are hashed in
`build/reports/pr23/validation-hashes.txt`.

The audit still reported two moderate PostCSS advisories inherited through
Next.js; they were not introduced by this patch and remain dependency follow-up
work. PR #23 merged as `4187466e8a2ee7835b467c0f984a70a73c94fb71`. Validation
logs and hashes remain under ignored `build/reports/pr23/`.

## Verification

Recorded UTC: 2026-10-20T01:00Z-01:20Z. Platform: macOS main machine. The
consolidated machine receipt is
`build/reports/worktree-inventory/receipt.json`; every row below is also
written there with its exact command, exit code, elapsed time, and log path.
At documentation head `3d8d7767b64b1ddbd9c77e4687f07500a99c090d`, `just quality`
exited 0 with `source_dirty: false`; its machine receipt is
`build/reports/quality/receipt.json`. The wording-only repair commit that
follows is gated again at its own clean head, with that final-head machine
receipt replacing the same path before merge.

| Check | Exact command or procedure | Exit | Elapsed | Log |
| --- | --- | ---: | ---: | --- |
| source-status | `git status --short --branch` | 0 | 16 ms | `build/reports/worktree-inventory/source-status.log` |
| worktree-inventory | `git worktree list` | 0 | 10 ms | `build/reports/worktree-inventory/worktree-inventory.log` |
| branch-inventory | `git branch -vv` | 0 | 12 ms | `build/reports/worktree-inventory/branch-inventory.log` |
| PR13 ancestry | `git merge-base --is-ancestor 257f892bbb8078e7f69fd0fe0aeef2adb0c9dfab main` | 0 | 11 ms | `build/reports/worktree-inventory/pr13-ancestry.log` |
| lexicon ancestry | `git merge-base --is-ancestor 02acd5a68576dc210dd43458bb5fd52ba12826bf main` | 0 | 12 ms | `build/reports/worktree-inventory/lexicon-ancestry.log` |
| PR15 exact tree | `git diff --quiet e388b32a83ea74d89f91392102147b127eb1f17a 80671a1d777996cdfa21837ce9c208334b04eb67` | 0 | 10 ms | `build/reports/worktree-inventory/pr15-exact-tree.log` |
| PR25 exact tree | `git diff --quiet 51f728fa4298569962020c4759dc917e5b5d0012 f86e716252539e0a1fcfb5a5ca50242d9eca7cf4` | 0 | 11 ms | `build/reports/worktree-inventory/pr25-exact-tree.log` |
| PR26 exact tree | `git diff --quiet a79b298adbde600b9b21be18daddf6ec25c2c783 7642a2386b9e4de6c42802abc8acf83fb857d40a` | 0 | 11 ms | `build/reports/worktree-inventory/pr26-exact-tree.log` |
| preservation manifest | hash and size verify all 90 payload entries | 0 | 41 ms | `build/reports/worktree-inventory/manifest-payload-verify.log` |
| portable-harness bundle | `git bundle verify ml-portable-harness.bundle` | 0 | 9 ms | `build/reports/worktree-inventory/ml-bundle-verify.log` |
| validated-insertion bundle | `git bundle verify work-validated-insertion.bundle` | 0 | 10 ms | `build/reports/worktree-inventory/validated-bundle-verify.log` |
| PR23 artifact hashes | `shasum -a 256 -c build/reports/pr23/validation-hashes.txt` | 0 | 18 ms | `build/reports/worktree-inventory/pr23-hash-verify.log` |
| public privacy scan | `rg` for private paths, row IDs, transcript/audio markers | 0 | 9 ms | `build/reports/worktree-inventory/privacy-scan.log` |
| diff check | `git diff --check` | 0 | 12 ms | `build/reports/worktree-inventory/diff-check.log` |

Before each removal, `lsof +D <worktree>` found no live working-directory
owner. All five worktrees were then removed with normal, non-force
`git worktree remove`; the failed verifier-path attempt and its repair are
represented by the passing final machine receipt rather than hidden.

## Non-claims

This inventory does not claim `ml/portable-harness` is obsolete or reviewed; it
remains an unmerged lane. It does not review, test, or integrate the preserved
validated-insertion patch. Preserved private artifacts remain local and are not
published by this receipt. The Next.js production audit does not clear the two
moderate PostCSS advisories. No model, ASR, latency, or daily-driver gate was
re-run by this cleanup.
