#!/usr/bin/env python3
"""Run Phonon's native Swift package tests and publish one aggregate receipt."""

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

SCOPE = "Native Swift unit gate for the macOS application package."
NON_CLAIMS = [
    "This does not run Rust, Python sidecars, models, UI snapshots, or TCC diagnostics.",
    "Unit success does not qualify live dictation, insertion, capture, or latency behavior.",
]


def swift_command(root: Path) -> list[str]:
    """Use the real Swift package manager command from its owning checkout."""
    swift = shutil.which("swift") or "swift"
    return [swift, "test", "--package-path", str(root / "bar"), "--disable-sandbox"]


def run_checks(
    report_dir: Path, root: Path, runner: Any = command_result
) -> dict[str, Any]:
    """Run the Swift package suite once and retain its complete output."""
    return runner(
        "swift-test",
        swift_command(root),
        report_dir=report_dir,
        cwd=root,
        timeout=1800.0,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Swift gate and atomically publish its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "build/reports/swift",
    )
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])
    report_dir = (
        arguments.report_dir
        if arguments.report_dir.is_absolute()
        else Path.cwd() / arguments.report_dir
    )
    receipt_path = report_dir / "receipt.json"
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
        status_log = read_log_result(status_result, report_dir)
        results = [revision_result, status_result, run_checks(report_dir, ROOT)]
        receipt = write_receipt(
            receipt_path,
            source_revision=revision,
            source_dirty=(
                bool(stdout_section(status_log)) if status_result["passed"] else None
            ),
            checks=results,
            scope=SCOPE,
            non_claims=NON_CLAIMS,
        )
        print(f"{'PASS' if receipt['passed'] else 'FAIL'} swift: {receipt_path}")
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
                        "check": "swift-runner",
                        "kind": "contract",
                        "passed": False,
                        "diagnostic": f"runner failed: {error}",
                    }
                ],
                scope=SCOPE,
                non_claims=NON_CLAIMS,
            )
        except Exception as receipt_error:  # noqa: BLE001 - publication boundary
            print(f"FAIL swift receipt: {receipt_error}", file=sys.stderr)
        print(f"FAIL swift: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
