import CoreGraphics
import Foundation
import AppKit

/// A main-display image retained in memory only. No PNG exists until a valid,
/// speech-bearing dictation reaches the explicit attachment boundary.
struct ScreenImageCandidate {
    let cgImage: CGImage
    let displayID: String
    let pixelWidth: Int
    let pixelHeight: Int
    let scaleFactor: Double
    let capturedAtUnixMs: UInt64
    let captureOrigin = "macos-bar-main-display"

    func pngData() throws -> Data {
        let representation = NSBitmapImageRep(cgImage: cgImage)
        guard let data = representation.representation(using: .png, properties: [:]) else {
            throw TrainingCaptureError.imageEncodingFailed
        }
        return data
    }
}

enum TrainingCaptureError: Error {
    case imageEncodingFailed
    case consentMissing
    case screenCapturePermissionMissing
    case stagingFailed
}

private struct TrainingProcessOutcome {
    let terminationStatus: Int32
    let standardError: String
}

struct TrainingCaptureSettings {
    var trainingCaptureEnabled: Bool
    var includeScreenImages: Bool
    var screenshotRetentionSeconds: Int
    var consentedAtUnixMs: UInt64?

    init(settings: NativeSettings) {
        self.init(
            trainingCaptureEnabled: settings.trainingCaptureEnabled,
            includeScreenImages: settings.includeScreenImages,
            screenshotRetentionSeconds: settings.screenshotRetentionSeconds,
            consentedAtUnixMs: settings.trainingCaptureConsentedAtUnixMs)
    }

    init(
        trainingCaptureEnabled: Bool,
        includeScreenImages: Bool,
        screenshotRetentionSeconds: Int,
        consentedAtUnixMs: UInt64?
    ) {
        self.trainingCaptureEnabled = trainingCaptureEnabled
        self.includeScreenImages = includeScreenImages
        self.screenshotRetentionSeconds = screenshotRetentionSeconds
        self.consentedAtUnixMs = consentedAtUnixMs
    }

    var canCapture: Bool {
        trainingCaptureEnabled && includeScreenImages && screenshotRetentionSeconds > 0
            && consentedAtUnixMs != nil
    }
}

enum TrainingAttachmentOutcome: Equatable {
    case attached
    case rejected(exitCode: Int32)
}

enum TrainingRetentionPolicy {
    /// With local history on, the audio candidate was already meant to remain.
    /// With it off, only a successful screen-image attachment promotes this
    /// pass from transient dictation to a training candidate.
    static func shouldKeepCandidate(
        localHistoryEnabled: Bool, attachmentSucceeded: Bool
    ) -> Bool {
        localHistoryEnabled || attachmentSucceeded
    }
}

struct TrainingScreenshotAttacher {
    struct Configuration {
        var phononBinary: String
        var stagingRoot: URL
        var permissionPreflight: () -> Bool

        init(
            phononBinary: String? = nil,
            stagingRoot: URL? = nil,
            permissionPreflight: @escaping () -> Bool = { CGPreflightScreenCaptureAccess() }
        ) {
            self.phononBinary = phononBinary ?? Self.defaultPhononBinary()
            self.stagingRoot = stagingRoot
                ?? FileManager.default.temporaryDirectory.appendingPathComponent(
                    "phonon-screen-training-\(ProcessInfo.processInfo.processIdentifier)",
                    isDirectory: true)
            self.permissionPreflight = permissionPreflight
        }

        static func defaultPhononBinary() -> String {
            if let configured = ProcessInfo.processInfo.environment["PHONON_BIN"],
                !configured.isEmpty
            {
                return configured
            }
            if let bundled = Bundle.main.builtInPlugInsURL?
                .deletingLastPathComponent()
                .appendingPathComponent("Helpers/phonon").path,
                FileManager.default.isExecutableFile(atPath: bundled)
            {
                return bundled
            }
            let homeBinary = NSString(string: "~/.local/bin/phonon").expandingTildeInPath
            if FileManager.default.isExecutableFile(atPath: homeBinary) {
                return homeBinary
            }
            return "phonon"
        }
    }

    struct Attachment {
        let audioPath: String
        let candidate: ScreenImageCandidate
        let settings: TrainingCaptureSettings
    }

    let configuration: Configuration

    init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    static func arguments(for attachment: Attachment, stagedImageURL: URL) -> [String] {
        [
            "corpus", "attach-screenshot",
            "--audio-path", attachment.audioPath,
            "--image-path", stagedImageURL.path,
            "--display-id", attachment.candidate.displayID,
            "--pixel-width", String(attachment.candidate.pixelWidth),
            "--pixel-height", String(attachment.candidate.pixelHeight),
            "--scale-factor", String(attachment.candidate.scaleFactor),
            "--capture-origin", attachment.candidate.captureOrigin,
            "--captured-at-ms", String(attachment.candidate.capturedAtUnixMs),
            "--consented-at-ms", String(attachment.settings.consentedAtUnixMs ?? 0),
            "--permission-granted",
        ]
    }

    static func expiryArguments(nowUnixMs: UInt64?) -> [String] {
        var arguments = ["corpus", "expire-screenshots"]
        if let nowUnixMs {
            arguments += ["--now-ms", String(nowUnixMs)]
        }
        return arguments
    }

    static func stage(_ pngData: Data, in root: URL) throws -> URL {
        let uniqueDirectory = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: uniqueDirectory, withIntermediateDirectories: true)
            let url = uniqueDirectory.appendingPathComponent("training-capture.png")
            try pngData.write(to: url, options: .atomic)
            return url
        } catch {
            try? FileManager.default.removeItem(at: uniqueDirectory)
            throw TrainingCaptureError.stagingFailed
        }
    }

    /// Every successful or failed attach removes the unique staged directory.
    /// The staged file never uses the corpus target name, so an interrupted
    /// old implementation cannot be mistaken for a new candidate.
    func attach(_ attachment: Attachment) async throws -> TrainingAttachmentOutcome {
        guard attachment.settings.canCapture else {
            throw TrainingCaptureError.consentMissing
        }
        guard configuration.permissionPreflight() else {
            throw TrainingCaptureError.screenCapturePermissionMissing
        }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                var stagedDirectory: URL?
                do {
                    let pngData = try attachment.candidate.pngData()
                    let stagedURL = try Self.stage(pngData, in: self.configuration.stagingRoot)
                    stagedDirectory = stagedURL.deletingLastPathComponent()
                    let outcome = try Self.runProcess(
                        executablePath: self.configuration.phononBinary,
                        arguments: Self.arguments(for: attachment, stagedImageURL: stagedURL))
                    stagedDirectory = nil
                    try? FileManager.default.removeItem(at: stagedURL.deletingLastPathComponent())
                    if outcome.terminationStatus == 0 {
                        continuation.resume(returning: .attached)
                    } else {
                        NSLog(
                                "phonon screen-image attach rejected: status \(outcome.terminationStatus): \(outcome.standardError)"
                        )
                        continuation.resume(
                            returning: .rejected(exitCode: outcome.terminationStatus))
                    }
                } catch {
                    if let stagedDirectory {
                        try? FileManager.default.removeItem(at: stagedDirectory)
                    }
                    NSLog("phonon screen-image attach failed: \(error.localizedDescription)")
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Expiry is deliberately asynchronous and independent of final text. A
    /// failed privacy deadline remains an explicit error for the caller.
    func expire(nowUnixMs: UInt64? = nil) async throws {
        let outcome = try await Self.runProcessAsync(
            executablePath: configuration.phononBinary,
            arguments: Self.expiryArguments(nowUnixMs: nowUnixMs))
        guard outcome.terminationStatus == 0 else {
            throw NSError(
                domain: "PhononTrainingCapture",
                code: Int(outcome.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: outcome.standardError])
        }
    }

    private static func runProcessAsync(
        executablePath: String, arguments: [String]
    ) async throws -> TrainingProcessOutcome {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    continuation.resume(
                        returning: try Self.runProcess(
                            executablePath: executablePath, arguments: arguments))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func runProcess(executablePath: String, arguments: [String]) throws
    -> TrainingProcessOutcome {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath)
        process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment
        process.standardOutput = FileHandle.nullDevice
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let errors =
            String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
            ?? ""
        return TrainingProcessOutcome(
            terminationStatus: process.terminationStatus, standardError: errors)
    }

    /// Called at quit as a belt-and-braces cleanup; normal attach removes its
    /// unique directory before resuming the caller.
    static func removeStagingRoot(_ root: URL? = nil) {
        let root = root ?? FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-screen-training-\(ProcessInfo.processInfo.processIdentifier)",
            isDirectory: true)
        try? FileManager.default.removeItem(at: root)
    }
}
