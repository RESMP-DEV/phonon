# Receipt: fail-closed active-target insertion

Date: 2026-10-18
Host: main machine
Branch: `ml/portable-harness`

## Architecture decision

Adopt the valuable Aqua integration mechanisms without adopting Electron or a
speculative helper process. Phonon insertion is now a native, fail-closed
transaction:

1. Capture an immutable target identity at dictation start.
2. Keep the most recent external target for menu-triggered passes after Phonon’s
   own panel becomes active.
3. Reactivate that process only when Phonon currently owns focus.
4. Recapture and validate process and bundle identity before touching the
   pasteboard.
5. Treat a focused accessibility element moving to another process as a target
   change when both focused-PID observations are available.
6. Mark the temporary pasteboard contents as auto-generated, concealed, and
   transient.
7. Restore the previous pasteboard after 200 ms, but only if its change count is
   unchanged.

The implementation deliberately does not add Aqua’s per-app submit profiles or a
separate system bridge. Those require a real failure ledger and measured benefit;
the fail-closed transaction is the durable product boundary.

## Implementation

- Added `bar/Sources/Insertion.swift`.
- Replaced the mutable `NSRunningApplication` insertion target with
  `InsertionTargetSnapshot`.
- Added `InsertionTargetValidation`.
- Added `PasteboardPrivacy`, `PasteboardSnapshot`, and
  `PasteboardTransaction`.
- Updated `Typer.emit` to validate before pasteboard mutation and restore the
  previous pasteboard safely.
- Added `bar/Tests/InsertionTests.swift`.

## Verification

```text
swift test --package-path bar
```

Result:

```text
Executed 73 tests, with 0 failures (0 unexpected)
```

New coverage includes:

- same process/bundle/focused-PID target passes;
- changed process or bundle fails;
- focused element moving processes fails;
- missing expected/current target fails;
- all three `org.nspasteboard` privacy markers are present;
- previous clipboard contents are restored;
- a racing user copy is not overwritten.

## Non-claims

- This does not qualify all target applications; a real target-app interaction
  probe is still required.
- App-specific Enter/Tab/submit behavior remains unimplemented.
- Background-output muting remains unimplemented and out of the default product.
- No separate bridge process was added.
- No personal dictation was recorded during this implementation pass.
