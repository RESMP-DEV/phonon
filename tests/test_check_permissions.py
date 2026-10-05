import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_permissions.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_permissions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_diagnostic_is_non_prompting(checker) -> None:
    invocation = checker.parse_arguments([])
    assert invocation.app == ROOT / "bar/dist/Phonon.app"
    assert invocation.request_screen_recording is False


def test_diagnostic_schema_rejects_identity_and_flag_drift(checker) -> None:
    invocation = checker.parse_arguments([])
    diagnostic = {
        "schemaVersion": 1,
        "diagnostic": "permissions",
        "bundleIdentifier": "com.example.other",
        "isAppBundle": True,
        "accessibilityGranted": False,
        "inputMonitoringGranted": False,
        "screenRecordingGranted": False,
        "microphoneStatus": "not_determined",
        "screenRecordingRequest": {"performed": False},
    }
    result = checker.validate_diagnostic(
        {"stdout": json.dumps(diagnostic), "exit_code": 0}, invocation
    )
    assert result["passed"] is False
    assert "unexpected bundle identifier" in result["errors"]


def test_diagnostic_schema_reports_non_objects_instead_of_crashing(checker) -> None:
    invocation = checker.parse_arguments([])
    list_result = checker.validate_diagnostic(
        {"stdout": "[]", "exit_code": 0}, invocation
    )
    null_request = {
        "schemaVersion": 1,
        "diagnostic": "permissions",
        "bundleIdentifier": "com.infatoshi.phonon",
        "isAppBundle": True,
        "accessibilityGranted": False,
        "inputMonitoringGranted": False,
        "screenRecordingGranted": False,
        "microphoneStatus": "not_determined",
        "screenRecordingRequest": None,
    }
    null_result = checker.validate_diagnostic(
        {"stdout": json.dumps(null_request), "exit_code": 0}, invocation
    )

    assert list_result["passed"] is False
    assert "diagnostic must be a JSON object" in list_result["errors"]
    assert null_result["passed"] is False
    assert "screenRecordingRequest must be a JSON object" in null_result["errors"]


def test_invalidated_app_contract_writes_a_failing_receipt(
    checker, tmp_path: Path
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"passed": true, "stale": true}\n')
    invocation = checker.Invocation(
        app=tmp_path / "missing/Phonon.app",
        request_screen_recording=False,
        require_granted=False,
        receipt=receipt,
        json_output=False,
    )
    exit_code, output = checker.run(invocation)

    assert exit_code == 1
    assert receipt.exists()
    persisted = json.loads(receipt.read_text())
    assert output["passed"] is False
    assert persisted["passed"] is False
    assert "stale" not in persisted
    assert any(
        check["check"] == "permission-diagnostic" for check in persisted["checks"]
    )
    assert any(
        check.get("errors") == ["not run because app contract failed"]
        for check in persisted["checks"]
    )


def test_runner_never_adds_the_request_flag_by_default(
    checker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "Phonon.app"
    executable = app / "Contents/MacOS/PhononBar"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    (app / "Contents/Info.plist").write_text("{}\n")
    commands: list[tuple[str, list[str]]] = []

    def fake_command_result(
        name: str, command: list[str], **_: object
    ) -> dict[str, object]:
        commands.append((name, command))
        stdout = "e" * 40 if name == "source-revision" else ""
        if name == "bundle-id":
            stdout = "com.infatoshi.phonon"
        if name == "permission-diagnostic":
            stdout = json.dumps(
                {
                    "schemaVersion": 1,
                    "diagnostic": "permissions",
                    "bundleIdentifier": "com.infatoshi.phonon",
                    "isAppBundle": True,
                    "accessibilityGranted": False,
                    "inputMonitoringGranted": False,
                    "screenRecordingGranted": False,
                    "microphoneStatus": "not_determined",
                    "screenRecordingRequest": {"performed": False},
                }
            )
        return {
            "check": name,
            "command": command,
            "passed": True,
            "exit_code": 0,
            "elapsed_ms": 0.0,
            "stdout": stdout,
            "stderr": "",
        }

    monkeypatch.setattr(checker, "command_result", fake_command_result)
    invocation = checker.Invocation(
        app=app,
        request_screen_recording=False,
        require_granted=False,
        receipt=tmp_path / "receipt.json",
        json_output=False,
    )
    exit_code, receipt = checker.run(invocation)
    diagnostic = next(
        command for name, command in commands if name == "permission-diagnostic"
    )

    assert exit_code == 0
    assert receipt["passed"] is True
    assert diagnostic[-1] != "--request-screen-recording"
