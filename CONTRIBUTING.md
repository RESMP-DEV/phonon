# Contributing to Phonon

This is the shared procedure for people and agents. Start with
[README.md](README.md), the relevant section of
[architecture](docs/architecture.md), and [AGENTS.md](AGENTS.md). Product
requirements live in [SPEC.md](SPEC.md); this file owns development procedure,
code standards, checks, evidence, and review.

## Set up a work area

Start from a committed baseline. Inspect `git status --short` and choose one
coherent scope before editing. When another checkout is already present:

```bash
git fetch origin
git worktree add -b work/<description> .worktrees/<name> origin/main
```

Do not reuse or remove an existing worktree without identifying its owner.
Worktrees do not share uncommitted files, `build/`, runtime state, model caches,
or report directories. Preserve unrelated dirty work.

## Work records and receipts

Every non-trivial state change or finding meant to be trusted later needs a
persistent receipt. Keep an execution checklist in ignored
`build/<task>/work.md` while work is in progress. Consolidated evidence goes in
`build/reports/<mode>/receipt.json`; sanitized durable evidence may go to
`evidence/` when it must survive a clean build.

A receipt records:

- UTC timestamp, platform, source revision, and whether the source was dirty.
- A scope sentence and explicit non-claims.
- Each check name, command, exit code, elapsed time, log path, and result.
- Aggregate `passed`; every failed check remains represented in that same receipt.

A runner must invalidate its old public receipt before startup. Startup,
command, or publication failures must leave a failing receipt or no current
receipt, never an earlier PASS. Receipt output is local evidence; do not copy
personal text, audio, screenshots, credentials, or private paths into it.

Workers provide changed files, commands, evidence, and limitations. Their green
tests are evidence, not integration acceptance. The integrating contributor
reviews the combined diff and writes the parent receipt.

## Checks

Run from the repository root:

| Command | Scope |
| --- | --- |
| `just quality` | Instruction/ignore contracts, diff hygiene, product Python lint and format, root/product/tool/runner tests, ShellCheck, and shfmt; no GUI or model |
| `just rust` | Cargo format, Clippy with warnings denied, and all workspace tests |
| `just swift` | Swift package tests on macOS |
| `just ui` | Deterministic synthetic native renders; no owner screen capture |
| `just permissions` | Signed app bundle build plus non-prompting app-process TCC diagnostic |
| `scripts/check_permissions.py --request-screen-recording` | Explicit user-invoked Screen Recording enrollment probe; never run casually |
| `ml/research/final_sweep` commands | Research-protocol checks; see the architecture section and pinned environment |

Required coverage accumulates. Take the union of applicable rows and run each
distinct check once at the final head:

| Change | Required validation |
| --- | --- |
| Shared Rust, persistence, data, or engine contract | `just quality` and `just rust`, plus focused behavioral tests for recovery and malformed input |
| Swift app behavior, permissions, capture, or settings | `just quality`, `just swift`, affected synthetic UI or permission check, and product-scope regression cases |
| Visual design or layout | `just swift`, `just ui`, and inspection of the actual generated images |
| macOS packaging/signing | `just permissions`, package verification, and the affected native tests |
| Python product sidecar or check runner | `just quality`, plus a protocol/recovery case that exercises malformed and interrupted input |
| Research/training code | The architecture-pinned research suite, prompt/hash contract, and exact artifact/revision metadata |
| Documentation only | Path/link verification, `just quality` when shared contracts changed, and `git diff --check` |
| Privacy/consent/retention behavior | Focused synthetic data lifecycle tests plus explicit revocation, crash-interruption, and deletion assertions |

A common check does not replace a native platform gate, model-quality gate,
latency measurement, or privacy acceptance. State skipped checks as skipped;
never describe an unavailable optional reviewer or unmeasured gate as approval.

### Selecting checks and reusing results

Run the smallest decisive check while iterating. Reuse an earlier result only
when its recorded revision and input set cover the final changed files and the
check did not depend on state that has changed. Re-run any failed scope after
repair. Before handoff, run every aggregate gate that spans the change and
record which results were reused.

Do not repeatedly rerun an unchanged full suite to manufacture confidence. A
focused discriminator is better evidence than a second identical full run. The
final PR description must distinguish fresh final-head evidence from unchanged
input reuse and named non-claims.

## Code standards

| Area | Standard |
| --- | --- |
| Rust | Workspace edition and existing module style; `thiserror` for library errors, `anyhow` for application boundaries; Clippy warnings are errors |
| Swift | Native SwiftUI/AppKit idioms, small feature-owned types, testable policy objects, and short user-facing copy |
| Python | Python 3.12 for product runners, modern typing, pathlib, explicit argv lists, bounded subprocesses, and no unexplained broad exception |
| Shell | `#!/usr/bin/env bash`, ShellCheck clean, shfmt layout, safe empty-array expansion, explicit paths and cleanup |
| Native UI | Existing dark instrument language; clear state over wordy onboarding; user copy states the action, not implementation internals |
| Tests | Observable contracts, malformed input, cancellation, interruption, recovery, privacy deletion, and platform boundaries; no synthetic substitute for a required real-platform measurement |

Validate external size, schema, identity, and state before allocation or
mutation. Blocking work stays off the UI thread. Personal-data paths use
default-off consent and explicit deletion. A mock must remain labeled until its
production path is independently exercised.

## Debugging and native platforms

Prefer native CLI/JSON diagnostics, direct APIs, and deterministic fixtures.
`PhononBar --phonon-diagnostic permissions` is non-prompting by default; the
request flag is the only explicit Screen Recording enrollment path. Do not use
UI-coordinate automation or owner screen content for debugging.

macOS TCC is a user boundary. Never reset or directly mutate permission
databases. A denied state is valid test evidence when permission has not been
granted. GUI checks run serially and own the foreground only for their declared
case. Windows validation must use the documented selftest and synthetic
fixtures, never an owner's live documents.

## Pull requests

1. Keep one coherent branch per result; do not push directly to `main`.
2. Include focused tests and update `docs/architecture.md` only for contracts,
   decisions, measured evidence, or the append-only work log.
3. Run the applicable matrix and put exact commands/outcomes in the PR. Include
   clean final-head evidence plus clearly labeled reused results.
4. Trigger configured reviewers, inspect thread-aware feedback, fix valid
   findings, reply with evidence, and resolve every actionable thread.
   Optional quota/authentication failures are recorded as unavailable.
5. Merge only at a reviewed head with no conflicts, unresolved actionable
   finding, or required repository gate. Preserve the merge receipt and exact
   head SHA.

Do not include Aqua audio/transcripts, accepted text, owner screenshots,
credentials, or private absolute paths in public descriptions or logs. A local
personal-data sample is never a pull-request fixture.
