import AppKit
import XCTest

@testable import PhononBar

final class InsertionTests: XCTestCase {
    func testTargetValidationAcceptsSameProcessAndBundle() {
        let expected = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: 123)
        let current = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: 123)

        XCTAssertEqual(
            InsertionTargetValidation.validate(expected: expected, current: current),
            .valid)
    }

    func testTargetValidationRejectsProcessOrBundleReuse() {
        let expected = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: nil)
        let current = InsertionTargetSnapshot(
            processID: 456,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: nil)

        XCTAssertEqual(
            InsertionTargetValidation.validate(expected: expected, current: current),
            .changed(
                expectedProcessID: 123,
                currentProcessID: 456,
                expectedBundleIdentifier: "com.example.editor",
                currentBundleIdentifier: "com.example.editor"))
    }

    func testTargetValidationRejectsFocusedElementMovingProcesses() {
        let expected = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: 123)
        let current = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: "com.example.editor",
            localizedName: "Example",
            focusedElementProcessID: 456)

        XCTAssertFalse(expected.matches(current))
    }

    func testMissingTargetsFailClosed() {
        let target = InsertionTargetSnapshot(
            processID: 123,
            bundleIdentifier: nil,
            localizedName: nil,
            focusedElementProcessID: nil)

        XCTAssertEqual(
            InsertionTargetValidation.validate(expected: nil, current: target),
            .missingExpectedTarget)
        XCTAssertEqual(
            InsertionTargetValidation.validate(expected: target, current: nil),
            .missingCurrentTarget)
    }

    func testPasteboardWriteUsesPrivacyMarkersAndRestoresPreviousContents() throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.clearContents() }
        try write("previous clipboard", on: pasteboard)

        let transaction = try XCTUnwrap(PasteboardPrivacy.prepare("dictated", on: pasteboard))

        XCTAssertEqual(pasteboard.string(forType: .string), "dictated")
        XCTAssertEqual(Set(PasteboardPrivacy.markerTypeNames), Set(transactionMarkerNames(pasteboard)))
        XCTAssertTrue(transaction.restoreIfCurrent(on: pasteboard))
        XCTAssertEqual(pasteboard.string(forType: .string), "previous clipboard")
        XCTAssertTrue(transactionMarkerNames(pasteboard).isEmpty)
    }

    func testPasteboardRestoreDoesNotOverwriteARacingUserCopy() throws {
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.clearContents() }
        try write("previous clipboard", on: pasteboard)

        let transaction = try XCTUnwrap(PasteboardPrivacy.prepare("dictated", on: pasteboard))
        try write("user copied after Phonon", on: pasteboard)

        XCTAssertFalse(transaction.restoreIfCurrent(on: pasteboard))
        XCTAssertEqual(
            pasteboard.string(forType: .string), "user copied after Phonon")
    }

    private func write(_ text: String, on pasteboard: NSPasteboard) throws {
        pasteboard.clearContents()
        let item = NSPasteboardItem()
        item.setString(text, forType: .string)
        XCTAssertTrue(pasteboard.writeObjects([item]))
    }

    private func transactionMarkerNames(_ pasteboard: NSPasteboard) -> [String] {
        pasteboard.types?.map(\.rawValue).filter(PasteboardPrivacy.markerTypeNames.contains) ?? []
    }
}
