import AppKit
import CoreGraphics
import Foundation
import ScreenCaptureKit

enum PrivacyPane: String, CaseIterable {
    case microphone = "Privacy_Microphone"
    case accessibility = "Privacy_Accessibility"
    case inputMonitoring = "Privacy_ListenEvent"
    case screenRecording = "Privacy_ScreenCapture"

    var settingsURL: URL {
        URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)")!
    }

    var openTitle: String {
        switch self {
        case .microphone: return "Open Microphone Settings"
        case .accessibility: return "Open Accessibility Settings"
        case .inputMonitoring: return "Open Input Monitoring Settings"
        case .screenRecording: return "Open Screen Recording Settings"
        }
    }

    func open() {
        NSWorkspace.shared.open(settingsURL)
    }
}

enum PermissionGuide: String, Identifiable {
    case inputMonitoring
    case screenRecording

    var id: String { rawValue }

    var title: String {
        switch self {
        case .inputMonitoring: return "Allow Input Monitoring"
        case .screenRecording: return "Allow Screen Recording"
        }
    }

    var detail: String {
        switch self {
        case .inputMonitoring: return "This makes the shortcut work in other apps."
        case .screenRecording: return "Phonon reads visible text to improve spelling."
        }
    }

    var steps: [String] {
        switch self {
        case .inputMonitoring:
            return ["Turn Phonon on.", "Restart Phonon."]
        case .screenRecording:
            return ["Turn Phonon on.", "Choose Screen Recording.", "Restart Phonon."]
        }
    }

    var pane: PrivacyPane {
        switch self {
        case .inputMonitoring: return .inputMonitoring
        case .screenRecording: return .screenRecording
        }
    }
}

enum ScreenRecordingPermission {
    /// Current macOS adds an app to Screen Recording only after it attempts an
    /// actual ScreenCaptureKit capture. A cursor-free one-pixel capture is
    /// enough to register Phonon; its image is never retained.
    static func requestEnrollment() async -> Bool {
        guard !CGPreflightScreenCaptureAccess() else { return true }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
            guard let display = content.displays.first(where: { $0.displayID == CGMainDisplayID() })
            else { return CGPreflightScreenCaptureAccess() }
            let filter = SCContentFilter(
                display: display, excludingApplications: [], exceptingWindows: [])
            let configuration = Self.enrollmentConfiguration()
            _ = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: configuration)
            return CGPreflightScreenCaptureAccess()
        } catch {
            return CGPreflightScreenCaptureAccess()
        }
    }

    static func enrollmentConfiguration() -> SCStreamConfiguration {
        let configuration = SCStreamConfiguration()
        configuration.width = 1
        configuration.height = 1
        configuration.showsCursor = false
        return configuration
    }
}
