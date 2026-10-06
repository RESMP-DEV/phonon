import AppKit
import AVFoundation
import ApplicationServices
import CoreGraphics
import Foundation
import ScreenCaptureKit

struct PermissionDiagnostics: Encodable {
    struct Request: Encodable {
        let performed: Bool
        let grantedAfterRequest: Bool?
    }

    let schemaVersion = 1
    let diagnostic = "permissions"
    let bundleIdentifier: String?
    let isAppBundle: Bool
    let accessibilityGranted: Bool
    let inputMonitoringGranted: Bool
    let screenRecordingGranted: Bool
    let microphoneStatus: String
    let screenRecordingRequest: Request

    enum Mode: Equatable {
        case none
        case permissions
    }

    static func parseArguments(
        _ arguments: [String]
    ) throws -> (mode: Mode, requestScreenRecording: Bool) {
        guard let modeIndex = arguments.firstIndex(of: "--phonon-diagnostic") else {
            return (.none, false)
        }
        guard arguments.indices.contains(modeIndex + 1) else {
            throw DiagnosticError.missingValue
        }
        let mode = arguments[modeIndex + 1]
        guard mode == "permissions" else {
            throw DiagnosticError.unknownMode(mode)
        }
        let request = arguments.contains("--request-screen-recording")
        return (.permissions, request)
    }

    static func capture(requestScreenRecording: Bool) async -> PermissionDiagnostics {
        let grantedAfterRequest: Bool?
        if requestScreenRecording {
            grantedAfterRequest = await ScreenRecordingPermission.requestEnrollment()
        } else {
            grantedAfterRequest = nil
        }
        return PermissionDiagnostics(
            bundleIdentifier: Bundle.main.bundleIdentifier,
            isAppBundle: Bundle.main.bundleURL.pathExtension == "app",
            accessibilityGranted: AXIsProcessTrusted(),
            inputMonitoringGranted: CGPreflightListenEventAccess(),
            screenRecordingGranted: CGPreflightScreenCaptureAccess(),
            microphoneStatus: Self.microphoneStatus(
                AVCaptureDevice.authorizationStatus(for: .audio)),
            screenRecordingRequest: Request(
                performed: requestScreenRecording,
                grantedAfterRequest: grantedAfterRequest))
    }

    static func run(requestScreenRecording: Bool) async -> String {
        let diagnostics = await capture(requestScreenRecording: requestScreenRecording)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(diagnostics),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return #"{"diagnostic":"permissions","error":"encoding failed"}"#
        }
        return encoded
    }

    enum DiagnosticError: Error, CustomStringConvertible {
        case missingValue
        case unknownMode(String)

        var description: String {
            switch self {
            case .missingValue: return "--phonon-diagnostic requires a mode"
            case .unknownMode(let mode): return "unknown diagnostic mode: \(mode)"
            }
        }
    }

    private static func microphoneStatus(_ status: AVAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "not_determined"
        case .restricted: return "restricted"
        case .denied: return "denied"
        case .authorized: return "authorized"
        @unknown default: return "unknown"
        }
    }
}
