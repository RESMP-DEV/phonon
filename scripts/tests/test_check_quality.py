"""Behavioral regressions for Phonon's common quality runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import check_quality


def make_root(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text(
        "/build/\n.worktrees/\n.sindexer/\n", encoding="utf-8"
    )
    (tmp_path / "AGENTS.md").write_text("# instructions\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").symlink_to("AGENTS.md")
    return tmp_path


def test_instruction_link_requires_literal_relative_target(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert check_quality.check_instruction_link(root)["passed"] is True
    (root / "CLAUDE.md").unlink()
    (root / "CLAUDE.md").symlink_to(root / "AGENTS.md")
    result = check_quality.check_instruction_link(root)
    assert result["passed"] is False
    assert "target must be AGENTS.md" in result["diagnostic"]


def test_ignore_contract_requires_all_machine_state_lines(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert check_quality.check_ignore_contract(root)["passed"] is True
    (root / ".gitignore").write_text("/build/\n", encoding="utf-8")
    result = check_quality.check_ignore_contract(root)
    assert result["passed"] is False
    assert ".worktrees/" in result["diagnostic"]
    assert ".sindexer/" in result["diagnostic"]


def test_repository_arguments_are_rendered_without_machine_paths() -> None:
    assert (
        check_quality._render_argument(check_quality.ROOT / "scripts/example.py")
        == "scripts/example.py"
    )
    assert (
        check_quality._render_argument(shutil.which("git") or "/usr/bin/git") == "git"
    )
    assert check_quality._render_argument("tests") == "tests"


def test_missing_executable_is_named_in_the_failure_diagnostic(
    tmp_path: Path,
) -> None:
    executable = "phonon-definitely-missing-executable"
    result = check_quality.command_result(
        "missing-tool",
        [executable],
        report_dir=tmp_path,
        cwd=check_quality.ROOT,
    )

    assert result["passed"] is False
    assert executable in result["command"][0]
    assert executable in result["diagnostic"]


def test_common_checks_have_stable_order_and_explicit_paths(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    scripts = root / "scripts"
    scripts.mkdir()
    shell = scripts / "example.sh"
    shell.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    checker = scripts / "check_example.py"
    checker.write_text("", encoding="utf-8")
    report = tmp_path / "reports"
    tools = {
        name: f"/usr/bin/{name}" for name in ("git", "uv", "uvx", "shellcheck", "shfmt")
    }
    calls: list[tuple[str, list[str]]] = []

    def runner(
        name: str,
        command: Sequence[str | Path],
        *,
        report_dir: Path,
        cwd: Path,
        **_: object,
    ) -> dict[str, object]:
        rendered = [str(part) for part in command]
        calls.append((name, rendered))
        return {
            "check": name,
            "kind": "command",
            "passed": True,
            "exit_code": 0,
            "command": rendered,
            "elapsed_ms": 0.0,
            "diagnostic": None,
            "log": f"{name}.log",
        }

    with patch.object(check_quality, "command_result", runner):
        results = check_quality.quality_checks(root, report, tools)

    assert [result["check"] for result in results] == [
        "instruction-symlink",
        "ignore-contract",
        "git-diff-check",
        "ruff-check",
        "ruff-format-check",
        "python-tests",
        "shellcheck",
        "shfmt",
    ]
    commands = dict(calls)
    python_test_command = commands["python-tests"]
    assert "--no-project" in python_test_command
    assert "--with" in python_test_command
    assert "pytest" in python_test_command
    assert "numpy" in python_test_command
    assert str(scripts) in commands["ruff-check"]
    assert str(checker) in commands["ruff-format-check"]
    assert str(shell) in commands["shellcheck"]
    assert str(shell) in commands["shfmt"]


def test_status_output_parser_handles_empty_and_trimmed_logs() -> None:
    assert check_quality.stdout_section("STDOUT:\n\nSTDERR:\n") == ""
    assert check_quality.stdout_section("STDOUT:\n\nSTDERR:") == ""
    assert check_quality.stdout_section("STDOUT:\n M file\nSTDERR:") == "M file"


def test_source_revision_rejects_successful_but_malformed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def successful_wrong_output(
        name: str,
        command: Sequence[str | Path],
        *,
        report_dir: Path,
        cwd: Path,
        **_: object,
    ) -> dict[str, object]:
        log = report_dir / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("STDOUT:\nnot-a-revision\nSTDERR:\n", encoding="utf-8")
        return {"check": name, "passed": True, "diagnostic": None}

    monkeypatch.setattr(check_quality, "command_result", successful_wrong_output)
    revision, result = check_quality.source_revision(tmp_path, "/usr/bin/git")

    assert revision is None
    assert result["passed"] is False
    assert result["diagnostic"] == "git did not return a 40-hex revision"


def test_runner_exception_does_not_hide_a_failed_receipt(tmp_path: Path) -> None:
    report = tmp_path / "reports"
    with patch.object(
        check_quality,
        "source_revision",
        side_effect=RuntimeError("launcher disappeared"),
    ):
        exit_code = check_quality.main(["--report-dir", str(report)])

    receipt_path = report / "receipt.json"
    assert exit_code == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["passed"] is False
    assert receipt["checks"][0]["diagnostic"] == "runner failed: launcher disappeared"


def test_receipt_replaces_a_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "private.json"
    target.write_text('{"private": true}\n', encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.symlink_to(target)
    check_quality.invalidate_receipt(receipt_path)
    receipt = check_quality.write_receipt(
        receipt_path,
        source_revision="0" * 40,
        source_dirty=False,
        checks=[{"check": "test", "passed": True}],
    )

    assert receipt["passed"] is True
    assert not receipt_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {"private": True}
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_failed_receipt_publication_cleans_its_temporary_file(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    with (
        patch.object(Path, "replace", side_effect=OSError("publication blocked")),
        pytest.raises(OSError, match="publication blocked"),
    ):
        check_quality.write_receipt(
            receipt_path,
            source_revision=None,
            source_dirty=None,
            checks=[{"check": "test", "passed": True}],
        )

    assert not receipt_path.exists()
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_direct_help_works_outside_the_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(check_quality.__file__), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
