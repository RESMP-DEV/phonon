import XCTest
import ScreenCaptureKit
@testable import PhononBar

final class PermissionTests: XCTestCase {
    func testDiagnosticSerializationDoesNotBridgeThroughSharedMutableState() throws {
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/PermissionDiagnostics.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        XCTAssertFalse(source.contains("DispatchSemaphore"))
        XCTAssertFalse(source.contains("Task.detached"))
    }

    func testScreenRecordingEnrollmentCapturesOneCursorFreePixel() {
        let configuration = ScreenRecordingPermission.enrollmentConfiguration()
        XCTAssertEqual(configuration.width, 1)
        XCTAssertEqual(configuration.height, 1)
        XCTAssertEqual(configuration.showsCursor, false)
    }

    func testPrivacyPanesUseExactSystemSettingsAnchors() {
        let expected = [
            PrivacyPane.microphone: "Privacy_Microphone",
            PrivacyPane.accessibility: "Privacy_Accessibility",
            PrivacyPane.inputMonitoring: "Privacy_ListenEvent",
            PrivacyPane.screenRecording: "Privacy_ScreenCapture",
        ]
        XCTAssertEqual(Set(PrivacyPane.allCases), Set(expected.keys))
        for (pane, anchor) in expected {
            XCTAssertEqual(
                pane.settingsURL,
                URL(
                    string:
                        "x-apple.systempreferences:com.apple.preference.security?\(anchor)")!)
        }
    }

    func testScreenRecordingGuideIsShortAndNamesTheExactPane() {
        let guide = PermissionGuide.screenRecording
        XCTAssertEqual(guide.pane, .screenRecording)
        XCTAssertEqual(guide.pane.openTitle, "Open Screen Recording Settings")
        XCTAssertEqual(guide.detail, "Phonon reads visible text to improve spelling.")
        XCTAssertEqual(guide.steps, ["Turn Phonon on.", "Choose Screen Recording.", "Restart Phonon."])
        XCTAssertLessThan(guide.steps.count, 4)
    }

    func testInputMonitoringGuideUsesItsOwnPane() {
        let guide = PermissionGuide.inputMonitoring
        XCTAssertEqual(guide.pane, .inputMonitoring)
        XCTAssertEqual(guide.detail, "This makes the shortcut work in other apps.")
        XCTAssertEqual(guide.steps, ["Turn Phonon on.", "Restart Phonon."])
    }

    func testPermissionDiagnosticParsesOnlyTheExplicitNonPromptingMode() {
        let absent = try? PermissionDiagnostics.parseArguments(["PhononBar"])
        guard let absent else {
            return XCTFail("arguments without a diagnostic flag must parse")
        }
        XCTAssertEqual(absent.mode, .none)
        XCTAssertEqual(absent.requestScreenRecording, false)

        let present = try? PermissionDiagnostics.parseArguments([
            "PhononBar", "--phonon-diagnostic", "permissions",
        ])
        XCTAssertEqual(present?.mode, .permissions)
        XCTAssertEqual(present?.requestScreenRecording, false)

        let requesting = try? PermissionDiagnostics.parseArguments([
            "PhononBar", "--phonon-diagnostic", "permissions",
            "--request-screen-recording",
        ])
        XCTAssertEqual(requesting?.mode, .permissions)
        XCTAssertEqual(requesting?.requestScreenRecording, true)

        XCTAssertThrowsError(
            try PermissionDiagnostics.parseArguments([
                "PhononBar", "--phonon-diagnostic", "capture-pixels",
            ]))
        XCTAssertThrowsError(
            try PermissionDiagnostics.parseArguments([
                "PhononBar", "--phonon-diagnostic",
            ]))
    }
}
