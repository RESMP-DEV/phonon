"""Contracts for the Rust and Swift receipt runners."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import check_rust, check_swift


def command(value: bool, timeouts: list[float] | None = None) -> Any:
    def runner(
        name: str,
        command: Sequence[str | Path],
        *,
        report_dir: Path,
        cwd: Path,
        **details: object,
    ) -> dict[str, Any]:
        if timeouts is not None:
            timeouts.append(float(details["timeout"]))
        return {
            "check": name,
            "kind": "command",
            "passed": value,
            "exit_code": 0 if value else 2,
            "command": [str(part) for part in command],
            "elapsed_ms": 0.0,
            "diagnostic": None if value else "forced failure",
            "log": f"{name}.log",
        }

    return runner


def test_rust_gate_runs_every_cold_build_check_after_failure(
    tmp_path: Path,
) -> None:
    timeouts: list[float] = []
    results = check_rust.run_checks(
        tmp_path, "/opt/bin/cargo", runner=command(False, timeouts)
    )

    assert [result["check"] for result in results] == [
        "cargo-fmt",
        "cargo-clippy",
        "cargo-test",
    ]
    assert all(result["passed"] is False for result in results)
    assert all(result["diagnostic"] == "forced failure" for result in results)
    assert timeouts == [1800.0, 1800.0, 1800.0]


def test_swift_gate_uses_package_path_and_cold_build_timeout(
    tmp_path: Path,
) -> None:
    captured: list[object] = []

    def runner(
        name: str,
        command: Sequence[str | Path],
        *,
        report_dir: Path,
        cwd: Path,
        **details: object,
    ) -> dict[str, Any]:
        captured.extend([name, command, details["timeout"]])
        return {"check": name, "passed": True}

    result = check_swift.run_checks(tmp_path, tmp_path, runner=runner)

    assert result["passed"] is True
    assert captured[0] == "swift-test"
    command = captured[1]
    assert isinstance(command, list)
    assert command[-3:] == [
        "--package-path",
        str(tmp_path / "bar"),
        "--disable-sandbox",
    ]
    assert captured[2] == 1800.0


def test_domain_runner_failure_writes_its_own_receipt(tmp_path: Path) -> None:
    report = tmp_path / "reports"
    with patch.object(
        check_swift, "source_revision", side_effect=RuntimeError("launcher gone")
    ):
        exit_code = check_swift.main(["--report-dir", str(report)])

    receipt = json.loads((report / "receipt.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert receipt["passed"] is False
    assert receipt["scope"].startswith("Native Swift")
    assert receipt["non_claims"][0].startswith("This does not run")
    assert receipt["checks"][0]["diagnostic"] == "runner failed: launcher gone"
