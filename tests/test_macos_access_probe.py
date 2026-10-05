import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "macos_access_probe.swift"
FORBIDDEN_KEYS = {
    "bundle_path",
    "bundle_id",
    "path",
    "title",
    "text",
    "value",
    "window_title",
    "selected_text",
}


@pytest.fixture(scope="module")
def binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("macos_access_probe")
    binary = build_dir / "macos_access_probe"
    subprocess.run(
        [
            "swiftc",
            "-swift-version",
            "5",
            "-O",
            "-module-cache-path",
            str(build_dir / "module-cache"),
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def collect_keys(report: object) -> set[str]:
    found: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            found.update(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(report)
    return found


def test_probe_fixture_is_deterministic_redacted_json(binary: Path) -> None:
    command = [str(binary), "--fixture"]
    runs = [subprocess.run(command, check=True, capture_output=True) for _ in range(2)]
    assert all(result.stderr == b"" for result in runs)
    assert runs[0].returncode == runs[1].returncode == 0
    assert runs[0].stdout == runs[1].stdout

    report = json.loads(runs[0].stdout)
    assert report["schema_version"] == 1
    assert report["mode"] == "fixture"
    assert report["screen_capture"]["pixels_read"] is False
    assert set(report["privacy"].values()) == {False}
    assert collect_keys(report).isdisjoint(FORBIDDEN_KEYS)
    assert str(ROOT).encode() not in runs[0].stdout
    assert str(Path.home()).encode() not in runs[0].stdout


def test_probe_live_output_is_schema_valid_and_redacted(binary: Path) -> None:
    result = subprocess.run([str(binary)], check=True, capture_output=True)
    assert result.stderr == b""

    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["mode"] == "live"
    assert isinstance(report["permissions"]["accessibility_granted"], bool)
    assert isinstance(report["permissions"]["screen_recording_preflight_granted"], bool)
    assert isinstance(report["permissions"]["input_monitoring_preflight_granted"], bool)
    assert report["screen_capture"]["pixels_read"] is False
    assert set(report["privacy"].values()) == {False}
    assert report["focused_ui"]["action_names_reported"] is False

    identity = report["focused_application"].get("pid_sha256_12")
    assert identity is None or re.fullmatch(r"[0-9a-f]{12}", identity)
    for semantic_name in (
        report["focused_ui"].get("role"),
        report["focused_ui"].get("subrole"),
    ):
        assert semantic_name is None or re.fullmatch(r"AX[A-Za-z0-9]+", semantic_name)

    assert collect_keys(report).isdisjoint(FORBIDDEN_KEYS)
    assert str(ROOT).encode() not in result.stdout
    assert str(Path.home()).encode() not in result.stdout


def test_probe_source_has_no_request_capture_network_or_mutation_calls() -> None:
    source = SOURCE.read_text()
    for forbidden_call in (
        "AXIsProcessTrustedWithOptions",
        "AXUIElementSetAttributeValue",
        "AXUIElementPerformAction",
        "CGRequestScreenCaptureAccess",
        "CGRequestListenEventAccess",
        "requestAccess",
        "SCScreenshotManager",
        "SCShareableContent",
        "URLSession",
    ):
        assert forbidden_call not in source
