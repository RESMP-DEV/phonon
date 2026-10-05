#!/usr/bin/env python3
"""Run Phonon's Rust format, analyzer, and workspace tests with one receipt."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__:
    from scripts.check_quality import (
        ROOT,
        command_result,
        invalidate_receipt,
        read_log_result,
        source_revision,
        stdout_section,
        write_receipt,
    )
else:
    from check_quality import (
        ROOT,
        command_result,
        invalidate_receipt,
        read_log_result,
        source_revision,
        stdout_section,
        write_receipt,
    )

CHECKS: tuple[tuple[str, list[str]], ...] = (
    ("cargo-fmt", ["fmt", "--all", "--", "--check"]),
    (
        "cargo-clippy",
        ["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
    ),
    ("cargo-test", ["test", "--workspace"]),
)
SCOPE = "Compiled Rust gate: formatting, warnings, and the full workspace test suite."
NON_CLAIMS = [
    "This does not build or launch the Swift app or Python sidecars.",
    "This does not measure model quality, dictation latency, or end-to-end insertion.",
]


def run_checks(
    report_dir: Path, cargo: str, runner: Any = command_result
) -> list[dict[str, Any]]:
    """Run each Rust check even when an earlier one fails."""
    return [
        runner(
            name,
            [cargo, *arguments],
            report_dir=report_dir,
            cwd=ROOT,
            timeout=1800.0,
        )
        for name, arguments in CHECKS
    ]


def stdout_text(result: dict[str, Any], report_dir: Path) -> str:
    """Read the ignored log written by the common command runner."""
    return read_log_result(result, report_dir).strip()


def main(argv: list[str] | None = None) -> int:
    """Run the Rust gate and publish its aggregate receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "build/reports/rust",
    )
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])
    report_dir = (
        arguments.report_dir
        if arguments.report_dir.is_absolute()
        else Path.cwd() / arguments.report_dir
    )
    receipt_path = report_dir / "receipt.json"
    cargo = shutil.which("cargo") or "cargo"
    try:
        invalidate_receipt(receipt_path)
        revision, revision_result = source_revision(
            report_dir, shutil.which("git") or "git"
        )
        status_result = command_result(
            "source-status",
            [shutil.which("git") or "git", "status", "--porcelain"],
            report_dir=report_dir,
            cwd=ROOT,
            timeout=15.0,
        )
        status = stdout_text(status_result, report_dir)
        results = [revision_result, status_result, *run_checks(report_dir, cargo)]
        receipt = write_receipt(
            receipt_path,
            source_revision=revision,
            source_dirty=(
                bool(stdout_section(status)) if status_result["passed"] else None
            ),
            checks=results,
            scope=SCOPE,
            non_claims=NON_CLAIMS,
        )
        print(f"{'PASS' if receipt['passed'] else 'FAIL'} rust: {receipt_path}")
        return 0 if receipt["passed"] else 1
    except Exception as error:  # noqa: BLE001 - final CLI failure boundary
        invalidate_receipt(receipt_path)
        try:
            write_receipt(
                receipt_path,
                source_revision=None,
                source_dirty=None,
                checks=[
                    {
                        "check": "rust-runner",
                        "kind": "contract",
                        "passed": False,
                        "diagnostic": f"runner failed: {error}",
                    }
                ],
                scope=SCOPE,
                non_claims=NON_CLAIMS,
            )
        except Exception as receipt_error:  # noqa: BLE001 - publication boundary
            print(f"FAIL rust receipt: {receipt_error}", file=sys.stderr)
        print(f"FAIL rust: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
