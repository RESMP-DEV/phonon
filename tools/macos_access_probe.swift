#!/usr/bin/env swift

import AVFoundation
import ApplicationServices
import CoreGraphics
import CryptoKit
import Foundation

struct PlatformReport: Encodable, Equatable {
    var macos_version: String
    var arch: String
}

struct PermissionReport: Encodable, Equatable {
    var accessibility_granted: Bool
    var screen_recording_preflight_granted: Bool
    var input_monitoring_preflight_granted: Bool
    var microphone_status: String
}

struct AccessibilityReport: Encodable, Equatable {
    var copy_attribute_available: Bool
    var copy_attribute_status_code: Int?
    var set_attribute_query_available: Bool
    var set_execution_attempted: Bool
    var selected_text_range_present: Bool
    var selected_text_range_settable: Bool?
    var selected_text_settable: Bool?
}

struct FocusedApplicationReport: Encodable, Equatable {
    var observed: Bool
    var pid_identity_present: Bool
    var pid_sha256_12: String?
    var is_probe_process: Bool?
}

struct FocusedUIReport: Encodable, Equatable {
    var observed: Bool
    var role: String?
    var subrole: String?
    var subrole_present: Bool
    var actions_count: Int?
    var action_names_reported: Bool
    var children_count: Int?
    var attribute_count: Int?
}

struct ScreenCaptureReport: Encodable, Equatable {
    var preflight_available: Bool
    var pixels_read: Bool
}

struct PrivacyReport: Encodable, Equatable {
    var accessibility_prompt_invoked: Bool
    var screen_capture_request_invoked: Bool
    var input_monitoring_request_invoked: Bool
    var microphone_request_invoked: Bool
    var ax_setter_invoked: Bool
    var selected_text_copied: Bool
    var network_used: Bool
}

struct ProbeReport: Encodable, Equatable {
    var schema_version: Int
    var mode: String
    var platform: PlatformReport
    var permissions: PermissionReport
    var accessibility: AccessibilityReport
    var focused_application: FocusedApplicationReport
    var focused_ui: FocusedUIReport
    var screen_capture: ScreenCaptureReport
    var privacy: PrivacyReport
}

struct ProbeError: Encodable, Equatable {
    var schema_version: Int
    var mode: String
    var error: String
    var usage: String
}

enum MicrophoneProbe {
    static var status: String {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return "granted"
        case .denied: return "denied"
        case .notDetermined: return "not_determined"
        case .restricted: return "restricted"
        @unknown default: return "unknown"
        }
    }
}

func metadataString(_ element: AXUIElement, _ attribute: String) -> (value: String?, status: AXError) {
    var raw: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(element, attribute as CFString, &raw)
    guard status == .success, let raw, CFGetTypeID(raw) == CFStringGetTypeID() else {
        return (nil, status)
    }
    let value = raw as! CFString
    return (value as String, status)
}

func redactCustomSemanticName(_ value: String?) -> String? {
    guard let value else { return nil }
    let appleStandardNames: Set<String> = [
        "AXButton", "AXCheckBox", "AXColorWell", "AXComboBox", "AXDrawer",
        "AXGrid", "AXGroup", "AXGrowArea", "AXImage", "AXList", "AXLink",
        "AXMenu", "AXMenuBar", "AXMenuBarItem", "AXMenuItem", "AXOutline",
        "AXPopUpButton", "AXProgressIndicator", "AXRadioButton", "AXRadioGroup",
        "AXRow", "AXScrollArea", "AXSheet", "AXSlider", "AXSplitGroup",
        "AXSplitter", "AXStaticText", "AXStatusBar", "AXTabGroup", "AXTable",
        "AXTextArea", "AXTextField", "AXToolbar", "AXUnknown", "AXValueIndicator",
        "AXWindow", "AXCloseButton", "AXDialog", "AXFloatingWindow",
        "AXFullScreenButton", "AXMinimizeButton", "AXStandardWindow",
        "AXSystemDialog", "AXZoomButton",
    ]
    return appleStandardNames.contains(value) ? value : "AXUnreportedCustomName"
}

func countOnly(_ names: CFArray?) -> Int? {
    names.map { CFArrayGetCount($0) }
}

func settable(_ element: AXUIElement, _ attribute: String) -> (available: Bool, settable: Bool?) {
    var value: DarwinBoolean = false
    let status = AXUIElementIsAttributeSettable(element, attribute as CFString, &value)
    return (status == .success, status == .success ? value.boolValue : nil)
}

func focusedApplicationIdentity(_ application: AXUIElement) -> FocusedApplicationReport {
    var pid: pid_t = -1
    let status = AXUIElementGetPid(application, &pid)
    guard status == .success, pid >= 0 else {
        return FocusedApplicationReport(
            observed: true, pid_identity_present: false, pid_sha256_12: nil, is_probe_process: nil)
    }

    let digest = SHA256.hash(data: Data(String(pid).utf8))
    let identity = digest.prefix(6).map { String(format: "%02x", $0) }.joined()
    return FocusedApplicationReport(
        observed: true,
        pid_identity_present: true,
        pid_sha256_12: identity,
        is_probe_process: pid == ProcessInfo.processInfo.processIdentifier)
}

func liveReport() -> ProbeReport {
    let accessibilityGranted = AXIsProcessTrusted()
    let systemWide = AXUIElementCreateSystemWide()

    var focusedApplicationRaw: CFTypeRef?
    let focusedApplicationStatus = AXUIElementCopyAttributeValue(
        systemWide, kAXFocusedApplicationAttribute as CFString, &focusedApplicationRaw)
    let focusedApplication = focusedApplicationStatus == .success
        ? unsafeDowncast(focusedApplicationRaw!, to: AXUIElement.self)
        : nil

    var focusedUIRaw: CFTypeRef?
    let focusedUIStatus = focusedApplication.map {
        AXUIElementCopyAttributeValue($0, kAXFocusedUIElementAttribute as CFString, &focusedUIRaw)
    } ?? AXError.cannotComplete
    let focusedUI = focusedUIStatus == .success
        ? unsafeDowncast(focusedUIRaw!, to: AXUIElement.self)
        : nil

    var attributeCount: Int?
    if let focusedUI {
        var childCount: CFIndex = 0
        let childStatus = AXUIElementGetAttributeValueCount(
            focusedUI, kAXChildrenAttribute as CFString, &childCount)
        if childStatus == .success {
            attributeCount = childCount
        }
    }

    var attributeNamesCount: Int?
    if let focusedUI {
        var names: CFArray?
        let namesStatus = AXUIElementCopyAttributeNames(focusedUI, &names)
        if namesStatus == .success {
            attributeNamesCount = countOnly(names)
        }
    }

    var actionNames: CFArray?
    var actionsCount: Int?
    if let focusedUI {
        let status = AXUIElementCopyActionNames(focusedUI, &actionNames)
        if status == .success {
            actionsCount = countOnly(actionNames)
        }
    }

    var selectedTextRangePresent = false
    var selectedTextRangeSettable: Bool?
    var selectedTextSettable: Bool?
    var setQueryAvailable = false
    var anyCopySucceeded = focusedApplicationStatus == .success || focusedUIStatus == .success
    if let focusedUI {
        var rangeRaw: CFTypeRef?
        let copyStatus = AXUIElementCopyAttributeValue(
            focusedUI, kAXSelectedTextRangeAttribute as CFString, &rangeRaw)
        selectedTextRangePresent = copyStatus == .success
        anyCopySucceeded = anyCopySucceeded || selectedTextRangePresent

        let rangeSettable = settable(focusedUI, kAXSelectedTextRangeAttribute)
        selectedTextRangeSettable = rangeSettable.settable
        setQueryAvailable = accessibilityGranted
            && (setQueryAvailable || rangeSettable.available)

        let textSettable = settable(focusedUI, kAXSelectedTextAttribute)
        selectedTextSettable = textSettable.settable
        setQueryAvailable = accessibilityGranted
            && (setQueryAvailable || textSettable.available)
    } else {
        let roleSettable = settable(systemWide, kAXRoleAttribute)
        setQueryAvailable = accessibilityGranted && roleSettable.available
    }

    let role = focusedUI.map { metadataString($0, kAXRoleAttribute) }
    let subrole = focusedUI.map { metadataString($0, kAXSubroleAttribute) }
    let focusedUIObserved = focusedUIStatus == .success && focusedUI != nil
    let applicationReport = focusedApplication.map(focusedApplicationIdentity)
        ?? FocusedApplicationReport(
            observed: false, pid_identity_present: false, pid_sha256_12: nil, is_probe_process: nil)

    return ProbeReport(
        schema_version: 1,
        mode: "live",
        platform: PlatformReport(
            macos_version: ProcessInfo.processInfo.operatingSystemVersionString,
            arch: arch()),
        permissions: PermissionReport(
            accessibility_granted: accessibilityGranted,
            screen_recording_preflight_granted: CGPreflightScreenCaptureAccess(),
            input_monitoring_preflight_granted: CGPreflightListenEventAccess(),
            microphone_status: MicrophoneProbe.status),
        accessibility: AccessibilityReport(
            copy_attribute_available: accessibilityGranted && anyCopySucceeded,
            copy_attribute_status_code: Int(focusedApplicationStatus.rawValue),
            set_attribute_query_available: setQueryAvailable,
            set_execution_attempted: false,
            selected_text_range_present: selectedTextRangePresent,
            selected_text_range_settable: selectedTextRangeSettable,
            selected_text_settable: selectedTextSettable),
        focused_application: applicationReport,
        focused_ui: FocusedUIReport(
            observed: focusedUIObserved,
            role: redactCustomSemanticName(role?.value),
            subrole: redactCustomSemanticName(subrole?.value),
            subrole_present: subrole?.status == .success,
            actions_count: actionsCount,
            action_names_reported: false,
            children_count: attributeCount,
            attribute_count: attributeNamesCount),
        screen_capture: ScreenCaptureReport(
            preflight_available: CGPreflightScreenCaptureAccess(), pixels_read: false),
        privacy: PrivacyReport(
            accessibility_prompt_invoked: false,
            screen_capture_request_invoked: false,
            input_monitoring_request_invoked: false,
            microphone_request_invoked: false,
            ax_setter_invoked: false,
            selected_text_copied: false,
            network_used: false))
}

func arch() -> String {
    #if arch(arm64)
        return "arm64"
    #elseif arch(x86_64)
        return "x86_64"
    #else
        return "unknown"
    #endif
}

func fixtureReport() -> ProbeReport {
    ProbeReport(
        schema_version: 1,
        mode: "fixture",
        platform: PlatformReport(macos_version: "Fixture 1.0", arch: "fixture"),
        permissions: PermissionReport(
            accessibility_granted: false,
            screen_recording_preflight_granted: false,
            input_monitoring_preflight_granted: false,
            microphone_status: "not_determined"),
        accessibility: AccessibilityReport(
            copy_attribute_available: false,
            copy_attribute_status_code: -25211,
            set_attribute_query_available: false,
            set_execution_attempted: false,
            selected_text_range_present: false,
            selected_text_range_settable: nil,
            selected_text_settable: nil),
        focused_application: FocusedApplicationReport(
            observed: false, pid_identity_present: false, pid_sha256_12: nil, is_probe_process: nil),
        focused_ui: FocusedUIReport(
            observed: false,
            role: nil,
            subrole: nil,
            subrole_present: false,
            actions_count: nil,
            action_names_reported: false,
            children_count: nil,
            attribute_count: nil),
        screen_capture: ScreenCaptureReport(preflight_available: false, pixels_read: false),
        privacy: PrivacyReport(
            accessibility_prompt_invoked: false,
            screen_capture_request_invoked: false,
            input_monitoring_request_invoked: false,
            microphone_request_invoked: false,
            ax_setter_invoked: false,
            selected_text_copied: false,
            network_used: false))
}

func emit<T: Encodable>(_ value: T, exitStatus: Int32 = 0) -> Never {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value), let text = String(data: data, encoding: .utf8) else {
        FileHandle.standardError.write(Data("probe encoding failed\n".utf8))
        exit(1)
    }
    print(text)
    exit(exitStatus)
}

let arguments = CommandLine.arguments.dropFirst()
switch arguments.first {
case nil:
    emit(liveReport())
case "--fixture":
    emit(fixtureReport())
case "--help":
    emit(ProbeError(
        schema_version: 1,
        mode: "usage",
        error: "help is represented as JSON",
        usage: "macos_access_probe [--fixture|--help]; no arguments runs a read-only live probe"))
default:
    emit(ProbeError(
        schema_version: 1,
        mode: "error",
        error: "unsupported argument",
        usage: "macos_access_probe [--fixture|--help]; no arguments runs a read-only live probe"),
        exitStatus: 2)
}
