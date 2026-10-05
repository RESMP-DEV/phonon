# Common repository/product Python and shell checks; no GUI or model launch.
quality:
    python3 scripts/check_quality.py

# Compiler formatting plus all Rust workspace tests and warnings.
rust:
    python3 scripts/check_rust.py

# Native macOS unit tests; uses the Swift package's sandbox-safe cases.
swift:
    python3 scripts/check_swift.py

# Build the app bundle and run the non-prompting app-process TCC diagnostic.
permissions:
    scripts/package-bar.sh
    scripts/check_permissions.py

# Deterministic synthetic native renders; never captures the owner's screen.
ui:
    scripts/ui_snapshots.sh
