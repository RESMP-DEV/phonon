#!/usr/bin/env python3
"""Check Phonon's macOS permission diagnostics from the real app process."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Invocation:
    app: Path
    request_screen_recording: bool
    require_granted: bool
    receipt: Path
    json_output: bool


def parse_arguments(argv: list[str]) -> Invocation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app",
        type=Path,
        default=ROOT / "bar/dist/Phonon.app",
        help="signed Phonon.app bundle to inspect (default: bar/dist/Phonon.app)",
    )
    parser.add_argument(
        "--request-screen-recording",
        action="store_true",
        help=(
            "explicitly perform Phonon's one-pixel, cursor-free ScreenCaptureKit "
            "enrollment attempt; the image is discarded"
        ),
    )
    parser.add_argument(
        "--require-granted",
        action="store_true",
        help="fail if Screen Recording is not granted after the diagnostic",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "build/reports/permissions/receipt.json",
    )
    parser.add_argument("--json", action="store_true", help="print the receipt as JSON")
    arguments = parser.parse_args(argv)
    receipt = arguments.receipt
    if not receipt.is_absolute():
        receipt = Path.cwd() / receipt
    return Invocation(
        app=arguments.app.resolve(),
        request_screen_recording=arguments.request_screen_recording,
        require_granted=arguments.require_granted,
        # Keep the literal final component. Resolving a preexisting receipt
        # symlink would redirect unlink/write operations to its target.
        receipt=receipt,
        json_output=arguments.json,
    )


def command_result(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: float = 30.0,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "check": name,
            "command": command,
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "check": name,
            "command": command,
            "passed": False,
            "exit_code": None,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "stdout": "",
            "stderr": repr(error),
        }


def source_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    revision = command_result(
        "source-revision", ["git", "rev-parse", "HEAD"], timeout=5.0
    )
    status = command_result(
        "source-status", ["git", "status", "--porcelain"], timeout=5.0
    )
    revision["passed"] = revision["passed"] and len(revision["stdout"]) == 40
    return (
        revision,
        status,
        {
            "check": "source-dirty",
            "passed": status["passed"],
            "exit_code": status["exit_code"],
            "dirty": bool(status["stdout"]),
        },
    )


def validate_diagnostic(
    result: dict[str, Any], invocation: Invocation
) -> dict[str, Any]:
    errors: list[str] = []
    diagnostic: dict[str, Any] = {}
    try:
        diagnostic = json.loads(result["stdout"])
    except (TypeError, json.JSONDecodeError) as error:
        errors.append(f"diagnostic is not JSON: {error}")
    if not isinstance(diagnostic, dict):
        errors.append("diagnostic must be a JSON object")
        diagnostic = {}

    expected_keys = {
        "schemaVersion",
        "diagnostic",
        "bundleIdentifier",
        "isAppBundle",
        "accessibilityGranted",
        "inputMonitoringGranted",
        "screenRecordingGranted",
        "microphoneStatus",
        "screenRecordingRequest",
    }
    if set(diagnostic) != expected_keys:
        errors.append(f"unexpected diagnostic keys: {sorted(diagnostic)}")
    if diagnostic.get("schemaVersion") != RECEIPT_SCHEMA_VERSION:
        errors.append("diagnostic schema version mismatch")
    if diagnostic.get("diagnostic") != "permissions":
        errors.append("diagnostic mode mismatch")
    if diagnostic.get("bundleIdentifier") != "com.infatoshi.phonon":
        errors.append("unexpected bundle identifier")
    if diagnostic.get("isAppBundle") is not True:
        errors.append("diagnostic did not run from an app bundle")
    request = diagnostic.get("screenRecordingRequest")
    if not isinstance(request, dict):
        errors.append("screenRecordingRequest must be a JSON object")
    elif request.get("performed") is not invocation.request_screen_recording:
        errors.append("request flag was not honored")
    if (
        invocation.require_granted
        and diagnostic.get("screenRecordingGranted") is not True
    ):
        errors.append("Screen Recording is not granted")

    return {
        "check": "diagnostic-schema",
        "command": [],
        "passed": not errors,
        "exit_code": result["exit_code"] if not errors else 1,
        "errors": errors,
        "diagnostic": diagnostic,
    }


def write_receipt(
    path: Path,
    *,
    invocation: Invocation,
    checks: list[dict[str, Any]],
    source_revision: str | None,
    source_dirty: bool | None,
) -> dict[str, Any]:
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "app": str(invocation.app),
        "request_screen_recording": invocation.request_screen_recording,
        "scope": (
            "App-process macOS TCC diagnostics and bundle identity. Default does "
            "not request permissions or read screen content; an explicit flag "
            "performs only a discarded one-pixel enrollment capture."
        ),
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
        "non_claims": [
            "This does not exercise dictation, OCR text extraction, corpus attachment, or insertion.",
            "A false permission value proves current state only; it is not a prompt or repair.",
            "A true Screen Recording value does not by itself prove a full-quality capture transaction.",
        ],
    }
    payload = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish from the same directory so an interrupted receipt cannot appear at
    # the public path while the complete JSON is still being written.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
    temporary_path.replace(path)
    temporary_path.unlink(missing_ok=True)
    return receipt


def run(invocation: Invocation) -> tuple[int, dict[str, Any]]:
    # Path.unlink operates on the link itself, never its target. This safely
    # invalidates a stale or hostile receipt symlink without following it.
    invocation.receipt.unlink(missing_ok=True)
    revision, status, dirty = source_state()
    checks: list[dict[str, Any]] = [revision, status, dirty]

    executable = invocation.app / "Contents/MacOS/PhononBar"
    app_contract = {
        "check": "app-contract",
        "command": [str(executable)],
        "passed": invocation.app.is_dir() and executable.is_file(),
        "exit_code": None,
        "errors": []
        if invocation.app.is_dir() and executable.is_file()
        else ["app bundle or executable missing"],
    }
    checks.append(app_contract)
    if app_contract["passed"]:
        signature = command_result(
            "codesign",
            [
                "codesign",
                "--verify",
                "--deep",
                "--strict",
                "--verbose=2",
                str(invocation.app),
            ],
        )
        bundle_id = command_result(
            "bundle-id",
            [
                "plutil",
                "-extract",
                "CFBundleIdentifier",
                "raw",
                "-o",
                "-",
                str(invocation.app / "Contents/Info.plist"),
            ],
        )
        bundle_id["passed"] = (
            bundle_id["passed"] and bundle_id["stdout"] == "com.infatoshi.phonon"
        )
        if not bundle_id["passed"]:
            bundle_id["stderr"] = bundle_id["stderr"] or "unexpected bundle identifier"
        checks.extend([signature, bundle_id])

        diagnostic_command = [
            str(executable),
            "--phonon-diagnostic",
            "permissions",
        ]
        if invocation.request_screen_recording:
            diagnostic_command.append("--request-screen-recording")
        may_execute = signature["passed"] and bundle_id["passed"]
        if may_execute:
            diagnostic = command_result(
                "permission-diagnostic", diagnostic_command, timeout=60.0
            )
            checks.append(diagnostic)
            checks.append(validate_diagnostic(diagnostic, invocation))
        else:
            checks.append(
                {
                    "check": "permission-diagnostic",
                    "command": diagnostic_command,
                    "passed": False,
                    "exit_code": None,
                    "errors": ["not run because signing or bundle identity failed"],
                }
            )
    else:
        checks.append(
            {
                "check": "permission-diagnostic",
                "command": [str(executable), "--phonon-diagnostic", "permissions"],
                "passed": False,
                "exit_code": None,
                "errors": ["not run because app contract failed"],
            }
        )

    revision_text = revision["stdout"] if revision["passed"] else None
    receipt = write_receipt(
        invocation.receipt,
        invocation=invocation,
        checks=checks,
        source_revision=revision_text,
        source_dirty=dirty["dirty"] if dirty["passed"] else None,
    )
    if invocation.json_output:
        print(json.dumps(receipt, indent=2))
    else:
        outcome = "PASS" if receipt["passed"] else "FAIL"
        print(f"{outcome} permissions: {invocation.receipt}")
    return 0 if receipt["passed"] else 1, receipt


def main(argv: list[str] | None = None) -> int:
    invocation = parse_arguments(argv if argv is not None else sys.argv[1:])
    exit_code, _ = run(invocation)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
