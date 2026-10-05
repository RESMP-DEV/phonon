import XCTest

@testable import PhononBar

final class TrainingCaptureTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-training-capture-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testSettingsDecodeKeepTrainingAndScreenImagesOffByDefault() throws {
        let data = Data(
            #"{"schema_version":2,"streaming":true,"local_history":false,"screen_context":false}"#
                .utf8)
        let settings = try JSONDecoder().decode(NativeSettings.self, from: data)

        XCTAssertFalse(settings.trainingCaptureEnabled)
        XCTAssertFalse(settings.includeScreenImages)
        XCTAssertEqual(
            settings.screenshotRetentionSeconds,
            NativeSettings.defaultScreenshotRetentionSeconds)
        XCTAssertNil(settings.trainingCaptureConsentedAtUnixMs)
        XCTAssertFalse(settings.screenImageTrainingAllowed)
    }

    func testSettingsDecodeBothRequiredConsentsAndFiniteRetention() throws {
        let data = Data(
            """
            {"schema_version":2,"training_capture_enabled":true,
             "include_screen_images":true,"screenshot_retention_seconds":3600,
             "training_capture_consented_at_unix_ms":123}
            """.utf8)
        let settings = try JSONDecoder().decode(NativeSettings.self, from: data)

        XCTAssertTrue(settings.trainingCaptureEnabled)
        XCTAssertTrue(settings.includeScreenImages)
        XCTAssertEqual(settings.screenshotRetentionSeconds, 3_600)
        XCTAssertEqual(settings.trainingCaptureConsentedAtUnixMs, 123)
        XCTAssertTrue(settings.screenImageTrainingAllowed)
    }

    @MainActor
    func testConsentIsRecordedOnlyWhenBothSwitchesAreOn() throws {
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertFalse(store.settings.screenImageTrainingAllowed)

        store.setTrainingCapture(enabled: true, includesScreenImages: false)
        XCTAssertFalse(store.settings.screenImageTrainingAllowed)
        XCTAssertNil(store.settings.trainingCaptureConsentedAtUnixMs)

        store.setTrainingCapture(enabled: true, includesScreenImages: true)
        let consentedAt = try XCTUnwrap(store.settings.trainingCaptureConsentedAtUnixMs)
        XCTAssertTrue(store.settings.screenImageTrainingAllowed)
        XCTAssertGreaterThan(consentedAt, 0)

        store.setTrainingCapture(enabled: false, includesScreenImages: true)
        XCTAssertNil(store.settings.trainingCaptureConsentedAtUnixMs)
        XCTAssertFalse(store.settings.screenImageTrainingAllowed)
    }

    func testAttachmentPolicyKeepsSuccessfulCandidateButNotFailedCandidateWithoutHistory() {
        XCTAssertTrue(
            TrainingRetentionPolicy.shouldKeepCandidate(
                localHistoryEnabled: false, attachmentSucceeded: true))
        XCTAssertFalse(
            TrainingRetentionPolicy.shouldKeepCandidate(
                localHistoryEnabled: false, attachmentSucceeded: false))
        XCTAssertTrue(
            TrainingRetentionPolicy.shouldKeepCandidate(
                localHistoryEnabled: true, attachmentSucceeded: false))
    }

    func testCLIArgumentShapeIncludesProvenanceConsentAndTCCPreflight() throws {
        let stagedURL = directory.appendingPathComponent("stage/training-capture.png")
        let attachment = TrainingScreenshotAttacher.Attachment(
            audioPath: "/tmp/fake-corpus/audio.wav",
            candidate: ScreenImageCandidate(
                cgImage: try onePixelImage(),
                displayID: "display-1",
                pixelWidth: 2,
                pixelHeight: 2,
                scaleFactor: 2,
                capturedAtUnixMs: 1_500),
            settings: TrainingCaptureSettings(
                trainingCaptureEnabled: true,
                includeScreenImages: true,
                screenshotRetentionSeconds: 3_600,
                consentedAtUnixMs: 1_000))

        XCTAssertEqual(
            TrainingScreenshotAttacher.arguments(for: attachment, stagedImageURL: stagedURL),
            [
                "corpus", "attach-screenshot",
                "--audio-path", "/tmp/fake-corpus/audio.wav",
                "--image-path", stagedURL.path,
                "--display-id", "display-1",
                "--pixel-width", "2",
                "--pixel-height", "2",
                "--scale-factor", "2.0",
                "--capture-origin", "macos-bar-main-display",
                "--captured-at-ms", "1500",
                "--consented-at-ms", "1000",
                "--permission-granted",
            ])
        XCTAssertEqual(TrainingScreenshotAttacher.expiryArguments(nowUnixMs: 2_000), [
            "corpus", "expire-screenshots", "--now-ms", "2000",
        ])
        XCTAssertEqual(TrainingScreenshotAttacher.expiryArguments(nowUnixMs: nil), [
            "corpus", "expire-screenshots",
        ])
    }

    func testSuccessfulAttachCleansItsUniqueStagedDirectory() async throws {
        let argsFile = directory.appendingPathComponent("arguments.txt")
        let fakePhonon = try fakePhonon(exitStatus: 0, argumentsFile: argsFile)
        let stagingRoot = directory.appendingPathComponent("staging")
        let attacher = TrainingScreenshotAttacher(
            configuration: .init(
                phononBinary: fakePhonon.path,
                stagingRoot: stagingRoot,
                permissionPreflight: { true }))

        let outcome = try await attacher.attach(try attachment())

        XCTAssertEqual(outcome, .attached)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: stagingRoot.path), [])
        let arguments = try String(contentsOf: argsFile, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
        XCTAssertTrue(arguments.contains("--permission-granted"))
        XCTAssertEqual(arguments.first, "corpus")
        XCTAssertTrue(arguments.contains("attach-screenshot"))
        XCTAssertTrue(arguments.contains("--capture-origin"))
        XCTAssertTrue(arguments.contains("--captured-at-ms"))
        XCTAssertTrue(arguments.contains("--consented-at-ms"))
        XCTAssertTrue(arguments.contains("--scale-factor"))
        XCTAssertTrue(arguments.contains("--pixel-width"))
        XCTAssertTrue(arguments.contains("--pixel-height"))
    }

    func testRejectedAttachAlsoCleansItsUniqueStagedDirectory() async throws {
        let fakePhonon = try fakePhonon(exitStatus: 3, argumentsFile: nil)
        let stagingRoot = directory.appendingPathComponent("staging")
        let attacher = TrainingScreenshotAttacher(
            configuration: .init(
                phononBinary: fakePhonon.path,
                stagingRoot: stagingRoot,
                permissionPreflight: { true }))

        let outcome = try await attacher.attach(try attachment())

        XCTAssertEqual(outcome, .rejected(exitCode: 3))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: stagingRoot.path), [])
    }

    func testMissingTCCPreflightThrowsBeforePNGStaging() async throws {
        let stagingRoot = directory.appendingPathComponent("staging")
        let attacher = TrainingScreenshotAttacher(
            configuration: .init(
                phononBinary: "/bin/true",
                stagingRoot: stagingRoot,
                permissionPreflight: { false }))

        do {
            _ = try await attacher.attach(try attachment())
            XCTFail("expected missing screen-capture permission to fail closed")
        } catch {
            XCTAssertFalse(FileManager.default.fileExists(atPath: stagingRoot.path))
        }
    }

    func testExpiryFailureIsSurfacedWithoutBlockingCallerAPI() async throws {
        let fakePhonon = try fakePhonon(exitStatus: 4, argumentsFile: nil)
        let attacher = TrainingScreenshotAttacher(
            configuration: .init(
                phononBinary: fakePhonon.path,
                stagingRoot: directory.appendingPathComponent("unused"),
                permissionPreflight: { true }))

        do {
            try await attacher.expire(nowUnixMs: 123)
            XCTFail("expected expiry failure")
        } catch let error as NSError {
            XCTAssertEqual(error.domain, "PhononTrainingCapture")
            XCTAssertEqual(error.code, 4)
        }
    }

    private func attachment() throws -> TrainingScreenshotAttacher.Attachment {
        TrainingScreenshotAttacher.Attachment(
            audioPath: directory.appendingPathComponent("audio.wav").path,
            candidate: ScreenImageCandidate(
                cgImage: try onePixelImage(),
                displayID: "synthetic-display",
                pixelWidth: 1,
                pixelHeight: 1,
                scaleFactor: 1,
                capturedAtUnixMs: 1_500),
            settings: TrainingCaptureSettings(
                trainingCaptureEnabled: true,
                includeScreenImages: true,
                screenshotRetentionSeconds: 3_600,
                consentedAtUnixMs: 1_000))
    }

    /// The only image fixture is a generated, opaque one-pixel rectangle. No
    /// screen capture, app window, or personal content is used by these tests.
    private func onePixelImage() throws -> CGImage {
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw TrainingCaptureError.imageEncodingFailed
        }
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        return try XCTUnwrap(context.makeImage())
    }

    private func fakePhonon(exitStatus: Int32, argumentsFile: URL?) throws -> URL {
        let url = directory.appendingPathComponent("fake-phonon")
        let arguments = argumentsFile.map { file in
            """
            printf '%s\\n' "$@" > '\(file.path)'
            """
        } ?? "true"
        let script = """
        #!/usr/bin/env bash
        set -u
        \(arguments)
        printf 'synthetic phonon response\\n'
        exit \(exitStatus)
        """
        try Data(script.utf8).write(to: url, options: .atomic)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }
}
