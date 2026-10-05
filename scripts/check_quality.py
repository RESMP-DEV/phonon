#!/usr/bin/env python3
"""Run Phonon's non-GUI product-quality checks and write one aggregate receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_VERSION = 1
REQUIRED_IGNORE_LINES = ("/build/", ".worktrees/", ".sindexer/")
PYTHON_LINT_PATHS = (
    "scripts",
    "sidecar",
    "tests",
    "tools/aqua_baseline.py",
    "tools/tests",
)
PYTHON_FORMAT_PATHS = ("sidecar", "tests", "tools/tests")
REVISION_LENGTH = 40


def contract_result(name: str, passed: bool, diagnostic: str | None) -> dict[str, Any]:
    """Return a compact result for a repository contract check."""
    return {
        "check": name,
        "kind": "contract",
        "passed": passed,
        "diagnostic": diagnostic,
    }


def check_instruction_link(root: Path) -> dict[str, Any]:
    """Require CLAUDE.md to be the literal relative instruction symlink."""
    path = root / "CLAUDE.md"
    target = root / "AGENTS.md"
    if not path.is_symlink():
        return contract_result(
            "instruction-symlink", False, "CLAUDE.md must link to AGENTS.md"
        )
    try:
        link_target = Path(str(path.readlink()))
    except OSError as error:
        return contract_result(
            "instruction-symlink", False, f"cannot inspect CLAUDE.md: {error}"
        )
    if link_target != Path("AGENTS.md"):
        return contract_result(
            "instruction-symlink",
            False,
            f"CLAUDE.md target must be AGENTS.md, got {link_target}",
        )
    if not target.is_file():
        return contract_result("instruction-symlink", False, "AGENTS.md is missing")
    return contract_result("instruction-symlink", True, None)


def check_ignore_contract(root: Path) -> dict[str, Any]:
    """Keep machine, worktree, and semantic-index state out of Git."""
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return contract_result(
            "ignore-contract", False, f"cannot read .gitignore: {error}"
        )
    missing = [line for line in REQUIRED_IGNORE_LINES if line not in lines]
    if missing:
        return contract_result(
            "ignore-contract", False, f"missing required lines: {missing}"
        )
    return contract_result("ignore-contract", True, None)


TOOL_COMMAND_NAMES = frozenset(
    {"git", "uv", "uvx", "shellcheck", "shfmt", "cargo", "swift"}
)


def _render_argument(part: str | Path) -> str:
    """Render repository paths and standard tools without machine-local prefixes."""
    path = Path(str(part))
    if not path.is_absolute():
        return str(part)
    if path.name in TOOL_COMMAND_NAMES and str(path) == shutil.which(path.name):
        return path.name
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_result(
    name: str,
    command: Sequence[str | Path],
    *,
    report_dir: Path,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run one bounded command and persist output without copying it into JSON."""
    rendered = [_render_argument(part) for part in command]
    started = time.monotonic()
    if not rendered or not rendered[0]:
        return _command_failure(
            name, rendered, report_dir, cwd, "missing required executable", None
        )
    try:
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            env=environment,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        diagnostic = (
            None
            if completed.returncode == 0
            else f"{name} failed with exit code {completed.returncode}"
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        exit_code = None
        stdout = ""
        stderr = repr(error)
        diagnostic = f"unable to execute {name}: {error}"
    log = _write_log(name, rendered, report_dir, cwd, stdout, stderr)
    return {
        "check": name,
        "kind": "command",
        "passed": diagnostic is None,
        "exit_code": exit_code,
        "command": rendered,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "diagnostic": diagnostic,
        "log": log,
    }


def _command_failure(
    name: str,
    command: list[str],
    report_dir: Path,
    cwd: Path,
    diagnostic: str,
    exit_code: int | None,
) -> dict[str, Any]:
    """Record launcher failures before any executable is available."""
    log = _write_log(name, command, report_dir, cwd, "", diagnostic)
    return {
        "check": name,
        "kind": "command",
        "passed": False,
        "exit_code": exit_code,
        "command": command,
        "elapsed_ms": 0.0,
        "diagnostic": diagnostic,
        "log": log,
    }


def _write_log(
    name: str,
    command: list[str],
    report_dir: Path,
    cwd: Path,
    stdout: str,
    stderr: str,
) -> str:
    """Write ignored process output and return its repository-relative path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.log"
    rendered_cwd = "." if cwd == ROOT else str(cwd)
    content = (
        f"CWD: {rendered_cwd}\nCommand: {shlex.join(command)}\n\n"
        f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n"
    )
    path.write_text(content, encoding="utf-8")
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def read_log_result(result: dict[str, Any], report_dir: Path) -> str:
    """Read command output used for receipt metadata, never for success."""
    try:
        return (report_dir / f"{result['check']}.log").read_text(encoding="utf-8")
    except (KeyError, OSError, UnicodeError):
        return ""


def source_revision(report_dir: Path, git: str) -> tuple[str | None, dict[str, Any]]:
    """Return the current 40-hex revision and its bounded command result."""
    result = command_result(
        "source-revision",
        [git, "rev-parse", "HEAD"],
        report_dir=report_dir,
        cwd=ROOT,
        timeout=15.0,
    )
    match = next(
        (
            line
            for line in read_log_result(result, report_dir).splitlines()
            if len(line) == REVISION_LENGTH
        ),
        "",
    )
    valid_revision = len(match) == REVISION_LENGTH and all(
        character in "0123456789abcdef" for character in match.lower()
    )
    if result["passed"] and not valid_revision:
        result["passed"] = False
        result["diagnostic"] = "git did not return a 40-hex revision"
        return None, result
    return (match.lower() if result["passed"] else None), result


def quality_checks(
    root: Path, report_dir: Path, tools: dict[str, str]
) -> list[dict[str, Any]]:
    """Run every common non-GUI check; later receipt aggregation retains failures."""
    script_paths = sorted(path for path in (root / "scripts").glob("*.sh"))
    dynamic_python = sorted((root / "scripts").glob("check_*.py"))
    lint_paths = [str(root / path) for path in PYTHON_LINT_PATHS]
    format_paths = [
        *(str(root / path) for path in PYTHON_FORMAT_PATHS),
        *(str(path) for path in dynamic_python),
    ]
    python_environment = {
        **dict(os.environ),
        "PYTHONPATH": f"{root}{os.pathsep}{root / 'tools'}",
    }
    shell_paths = [str(path) for path in script_paths]
    checks = [
        check_instruction_link(root),
        check_ignore_contract(root),
    ]
    commands = [
        (
            "git-diff-check",
            [tools["git"], "diff", "--check", "HEAD"],
            {},
        ),
        (
            "ruff-check",
            [tools["uvx"], "ruff", "check", *lint_paths],
            {},
        ),
        (
            "ruff-format-check",
            [tools["uvx"], "ruff", "format", "--check", *format_paths],
            {},
        ),
        (
            "python-tests",
            [
                tools["uv"],
                "run",
                "--no-project",
                "--python",
                "3.12",
                "--with",
                "pytest",
                "--with",
                "numpy",
                "pytest",
                "tests",
                "tools/tests",
                "scripts/tests",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            python_environment,
        ),
    ]
    if shell_paths:
        commands.extend(
            [
                ("shellcheck", [tools["shellcheck"], *shell_paths], {}),
                ("shfmt", [tools["shfmt"], "--diff", *shell_paths], {}),
            ]
        )
    else:
        checks.append(
            contract_result("shell-scripts", False, "no shell scripts were discovered")
        )
    for name, command, environment in commands:
        checks.append(
            command_result(
                name,
                command,
                report_dir=report_dir,
                cwd=root,
                environment=environment or None,
            )
        )
    return checks


def invalidate_receipt(path: Path) -> None:
    """Remove a stale public receipt without following a final symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def write_receipt(
    path: Path,
    *,
    source_revision: str | None,
    source_dirty: bool | None,
    checks: list[dict[str, Any]],
    scope: str | None = None,
    non_claims: list[str] | None = None,
) -> dict[str, Any]:
    """Atomically publish the aggregate receipt with explicit non-claims."""
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "scope": scope
        or (
            "Non-GUI product-quality gate: repository contracts, diff hygiene, "
            "Python lint/format, product Python tests, and shell checks."
        ),
        "non_claims": non_claims
        or [
            "This does not build or launch the Swift app.",
            "This does not run Rust, model, OCR, dictation, corpus, or UI-visual checks.",
            "Green script tests do not qualify a model or research adapter for product use.",
        ],
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
    try:
        temporary_path.replace(path)
    finally:
        # After a successful rename this is a no-op; after a publication
        # failure it prevents an invocation-owned temporary receipt leak.
        temporary_path.unlink(missing_ok=True)
    return receipt


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the common quality invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "build/reports/quality",
        help="receipt and log directory (default: build/reports/quality)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the common gate and return its aggregate exit status."""
    arguments = parse_arguments(argv if argv is not None else sys.argv[1:])
    report_dir = (
        arguments.report_dir
        if arguments.report_dir.is_absolute()
        else Path.cwd() / arguments.report_dir
    )
    receipt_path = report_dir / "receipt.json"
    tools = {
        name: shutil.which(name) or name
        for name in ("git", "uv", "uvx", "shellcheck", "shfmt")
    }
    try:
        invalidate_receipt(receipt_path)
        revision, revision_result = source_revision(report_dir, tools["git"])
        status_result = command_result(
            "source-status",
            [tools["git"], "status", "--porcelain"],
            report_dir=report_dir,
            cwd=ROOT,
            timeout=15.0,
        )
        status_text = read_log_result(status_result, report_dir)
        source_dirty = (
            bool(stdout_section(status_text)) if status_result["passed"] else None
        )
        results = [
            revision_result,
            status_result,
            *quality_checks(ROOT, report_dir, tools),
        ]
        receipt = write_receipt(
            receipt_path,
            source_revision=revision,
            source_dirty=source_dirty,
            checks=results,
        )
        outcome = "PASS" if receipt["passed"] else "FAIL"
        print(f"{outcome} quality: {receipt_path}", flush=True)
        return 0 if receipt["passed"] else 1
    except Exception as error:  # noqa: BLE001 - final CLI failure boundary
        failure = contract_result("quality-runner", False, f"runner failed: {error}")
        try:
            write_receipt(
                receipt_path,
                source_revision=None,
                source_dirty=None,
                checks=[failure],
            )
        except Exception as receipt_error:  # noqa: BLE001 - report publication boundary
            print(f"FAIL quality receipt: {receipt_error}", file=sys.stderr, flush=True)
        print(f"FAIL quality: {error}", file=sys.stderr, flush=True)
        return 1


def stdout_section(log: str) -> str:
    """Extract the bounded stdout section written by ``_write_log``."""
    body = log.partition("STDOUT:\n")[2]
    return body.split("\nSTDERR:", 1)[0].strip()


if __name__ == "__main__":
    raise SystemExit(main())
