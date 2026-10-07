import XCTest

@testable import PhononBar

@MainActor
final class AppDataTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "phonon-app-data-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testDemoSupportOverrideNeverAppliesOutsideDemoMode() {
        setenv("PHONON_UI_SUPPORT_DIR", directory.path, 1)
        defer { unsetenv("PHONON_UI_SUPPORT_DIR") }

        let normal = PhononDataPaths.supportDirectory()
        XCTAssertNotEqual(normal.path, directory.path)

        setenv("PHONON_UI_DEMO", "main", 1)
        defer { unsetenv("PHONON_UI_DEMO") }
        XCTAssertEqual(PhononDataPaths.supportDirectory().path, directory.path)
    }

    func testSettingsMigrationAndSaveAddsNativeControls() throws {
        try Data(
            #"{"schema_version":1,"streaming":true,"local_history":true,"screen_context":false,"microphone_priority":["USB Microphone","MacBook"]}"#.utf8
        ).write(to: directory.appendingPathComponent("settings.json"))

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(store.settings.instantMic)
        XCTAssertEqual(store.settings.shortcutMode, "fn")

        store.updateSettings {
            $0.instantMic = false
            $0.shortcutMode = "right_option"
        }
        let saved = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(contentsOf: directory.appendingPathComponent("settings.json")))
        XCTAssertFalse(saved.instantMic)
        XCTAssertEqual(saved.shortcutMode, "right_option")
        XCTAssertFalse(saved.screenContext)
    }

    func testFreshInstallKeepsRetainingFeaturesOffUntilAsked() throws {
        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertFalse(store.settings.localHistory)
        XCTAssertFalse(store.settings.screenContext)
        XCTAssertTrue(store.needsPrivacyChoice)
        XCTAssertEqual(
            store.settings.historyRetentionDays, NativeSettings.keepRecordingsForever)

        store.recordPrivacyChoice(localHistory: true, screenContext: false)
        XCTAssertFalse(store.needsPrivacyChoice)
        let saved = try JSONDecoder().decode(
            NativeSettings.self,
            from: Data(contentsOf: directory.appendingPathComponent("settings.json")))
        XCTAssertTrue(saved.localHistory)
        XCTAssertFalse(saved.screenContext)
        XCTAssertEqual(saved.schemaVersion, NativeSettings.currentSchemaVersion)
    }

    func testExistingInstallKeepsItsChoicesAndIsNotReprompted() throws {
        try Data(#"{"schema_version":1,"streaming":true,"screen_context":true}"#.utf8)
            .write(to: directory.appendingPathComponent("settings.json"))

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertTrue(store.settings.screenContext)
        // Absent before schema 2 meant enabled, so it must not silently flip off.
        XCTAssertTrue(store.settings.localHistory)
        XCTAssertFalse(store.needsPrivacyChoice)
    }

    func testRetentionOnlyPrunesPastTheChosenWindow() throws {
        let now = Date()
        for (name, ageDays) in [("old", 40.0), ("recent", 2.0)] {
            let recording = directory.appendingPathComponent("Corpus/\(name)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: recording, withIntermediateDirectories: true)
            let created = UInt64(now.addingTimeInterval(-ageDays * 86_400).timeIntervalSince1970 * 1_000)
            try Data(
                """
                {"id":"\(name)","created_at_unix_ms":\(created),"raw_transcript":"x"}
                """.utf8
            ).write(to: recording.appendingPathComponent("metadata.json"))
        }

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.history.count, 2)

        // Default window keeps everything.
        store.pruneExpiredRecordings(now: now)
        XCTAssertEqual(store.history.count, 2)

        store.updateSettings { $0.historyRetentionDays = 30 }
        store.pruneExpiredRecordings(now: now)
        XCTAssertEqual(store.history.map(\.id), ["recent"])
    }

    func testClearAllHistoryEmptiesTheCorpus() throws {
        let recording = directory.appendingPathComponent("Corpus/one", isDirectory: true)
        try FileManager.default.createDirectory(at: recording, withIntermediateDirectories: true)
        try Data(#"{"id":"one","created_at_unix_ms":1000,"raw_transcript":"x"}"#.utf8)
            .write(to: recording.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        XCTAssertEqual(store.history.count, 1)
        store.clearAllHistory()
        XCTAssertTrue(store.history.isEmpty)
    }

    func testInterruptedRecordingWithoutMetadataIsReapedAtLaunch() throws {
        let interrupted = directory.appendingPathComponent("Corpus/interrupted", isDirectory: true)
        try FileManager.default.createDirectory(at: interrupted, withIntermediateDirectories: true)
        try Data([0x01]).write(to: interrupted.appendingPathComponent("audio.wav"))
        let indexed = directory.appendingPathComponent("Corpus/indexed", isDirectory: true)
        try FileManager.default.createDirectory(at: indexed, withIntermediateDirectories: true)
        try Data(
            #"{"id":"indexed","created_at_unix_ms":1,"raw_transcript":"kept"}"#.utf8
        ).write(to: indexed.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        store.updateSettings { $0.localHistory = true }
        let removed = store.reapInterruptedRecordings()

        XCTAssertEqual(removed, ["interrupted"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: interrupted.path))
        XCTAssertTrue(store.history.map(\.id).contains("indexed"))
    }

    func testHistoryOffReapsOnlyIndexedCandidatesStillPendingRetention() throws {
        try Data(
            #"{"schema_version":2,"local_history":false,"screen_context":false}"#.utf8
        ).write(to: directory.appendingPathComponent("settings.json"))
        let interrupted = directory.appendingPathComponent("Corpus/candidate", isDirectory: true)
        try FileManager.default.createDirectory(at: interrupted, withIntermediateDirectories: true)
        try Data([0x01]).write(to: interrupted.appendingPathComponent("audio.wav"))
        try Data(
            #"{"id":"candidate","created_at_unix_ms":1,"raw_transcript":"synthetic","retention_pending":true}"#
                .utf8
        ).write(to: interrupted.appendingPathComponent("metadata.json"))
        let finalized = directory.appendingPathComponent("Corpus/finalized", isDirectory: true)
        try FileManager.default.createDirectory(at: finalized, withIntermediateDirectories: true)
        try Data(
            #"{"id":"finalized","created_at_unix_ms":1,"raw_transcript":"synthetic","retention_pending":false}"#
                .utf8
        ).write(to: finalized.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        let removed = store.reapInterruptedRecordings()

        XCTAssertEqual(removed, ["candidate"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: interrupted.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: finalized.path))
    }

    func testFreshMissingSettingsStillAuthorizePendingRecordingCleanup() throws {
        let pending = directory.appendingPathComponent("Corpus/pending", isDirectory: true)
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
        try Data(
            #"{"id":"pending","created_at_unix_ms":1,"raw_transcript":"synthetic","retention_pending":true}"#
                .utf8
        ).write(to: pending.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        let removed = store.reapInterruptedRecordings()

        XCTAssertEqual(removed, ["pending"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: pending.path))
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: directory.appendingPathComponent("settings.json").path))
    }

    func testMalformedSettingsDoNotAuthorizePendingRecordingDeletion() throws {
        try Data(#"{"schema_version":"broken"}"#.utf8)
            .write(to: directory.appendingPathComponent("settings.json"))
        let pending = directory.appendingPathComponent("Corpus/pending", isDirectory: true)
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
        try Data(
            #"{"id":"pending","created_at_unix_ms":1,"raw_transcript":"synthetic","retention_pending":true}"#
                .utf8
        ).write(to: pending.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)
        let removed = store.reapInterruptedRecordings()

        XCTAssertTrue(removed.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: pending.path))
    }

    func testReapingIgnoresRegularFilesAtTheCorpusRoot() throws {
        let corpus = directory.appendingPathComponent("Corpus", isDirectory: true)
        try FileManager.default.createDirectory(at: corpus, withIntermediateDirectories: true)
        let regularFile = corpus.appendingPathComponent(".DS_Store")
        try Data([0x01]).write(to: regularFile)

        let store = NativeAppStore(supportDirectory: directory)
        let removed = store.reapInterruptedRecordings()

        XCTAssertTrue(removed.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: regularFile.path))
    }

    func testDictionaryAddEditAndRemoveRoundTrips() throws {
        try Data(#"{"schema_version":1,"updated_at_unix_ms":1,"entries":[]}"#.utf8)
            .write(to: directory.appendingPathComponent("dictionary.json"))
        let store = NativeAppStore(supportDirectory: directory)

        store.upsertDictionary(
            originalID: nil, phrase: "black well", replacement: "Blackwell",
            spokenForms: ["blackwell", "black well"])
        let entry = try XCTUnwrap(store.dictionaryEntries.first)
        XCTAssertEqual(entry.replacement, "Blackwell")
        XCTAssertEqual(entry.spokenForms, ["black well", "blackwell"])

        store.removeDictionary(id: entry.id)
        XCTAssertTrue(store.dictionaryEntries.isEmpty)
    }

    func testSavingIntendedTranscriptPreservesUnmodeledMetadata() throws {
        let recording = directory.appendingPathComponent("Corpus/test", isDirectory: true)
        try FileManager.default.createDirectory(at: recording, withIntermediateDirectories: true)
        let metadata = """
            {
              "schema_version": 1,
              "id": "test",
              "created_at_unix_ms": 1000,
              "source": "bar",
              "audio_file": "audio.wav",
              "raw_transcript": "black well",
              "final_transcript": "Blackwell",
              "dictionary_corrections": [{"from":"black well","to":"Blackwell"}],
              "screen_context_terms": []
            }
            """
        try Data(metadata.utf8).write(to: recording.appendingPathComponent("metadata.json"))
        let store = NativeAppStore(supportDirectory: directory)

        store.saveIntendedTranscript(itemID: "test", text: "Use Blackwell.")

        let data = try Data(contentsOf: recording.appendingPathComponent("metadata.json"))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["intended_transcript"] as? String, "Use Blackwell.")
        XCTAssertEqual((object["dictionary_corrections"] as? [[String: String]])?.count, 1)
    }

    func testNoSpeechCorpusItemsStayOffUserHistory() throws {
        let recording = directory.appendingPathComponent("Corpus/silent", isDirectory: true)
        try FileManager.default.createDirectory(at: recording, withIntermediateDirectories: true)
        try Data(
            #"{"schema_version":1,"id":"silent","created_at_unix_ms":1000,"source":"bar","audio_file":"audio.wav","speech_detected":false,"raw_transcript":"","final_transcript":""}"#.utf8
        ).write(to: recording.appendingPathComponent("metadata.json"))

        let store = NativeAppStore(supportDirectory: directory)

        XCTAssertTrue(store.history.isEmpty)
        XCTAssertEqual(store.usage.recordings, 0)
    }

    func testUsageStatsIncludeSpeedDailyActivityAndDictionaryRepairs() throws {
        let calendar = Calendar.current
        let now = Date()
        let previousDay = try XCTUnwrap(calendar.date(byAdding: .day, value: -2, to: now))
        let fixtures: [(String, Date, Int, UInt64, UInt64)] = [
            ("today", now, 120, 60_000, 2),
            ("older", previousDay, 30, 30_000, 1),
        ]

        for (id, date, wordCount, duration, corrections) in fixtures {
            let recording = directory.appendingPathComponent("Corpus/\(id)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: recording, withIntermediateDirectories: true)
            let words = Array(repeating: "word", count: wordCount).joined(separator: " ")
            let metadata: [String: Any] = [
                "schema_version": 1,
                "id": id,
                "created_at_unix_ms": UInt64(date.timeIntervalSince1970 * 1_000),
                "source": "bar",
                "audio_file": "audio.wav",
                "audio_duration_ms": duration,
                "speech_detected": true,
                "raw_transcript": words,
                "final_transcript": words,
                "dictionary_corrections": [
                    ["from": "raw", "to": "term", "count": corrections]
                ],
            ]
            let data = try JSONSerialization.data(withJSONObject: metadata)
            try data.write(to: recording.appendingPathComponent("metadata.json"))
        }

        let store = NativeAppStore(supportDirectory: directory)

        XCTAssertEqual(store.usage.recordings, 2)
        XCTAssertEqual(store.usage.words, 150)
        XCTAssertEqual(store.usage.speakingMilliseconds, 90_000)
        XCTAssertEqual(store.usage.wordsPerMinute, 100)
        XCTAssertEqual(store.usage.averageWordsPerRecording, 75)
        XCTAssertEqual(store.usage.dictionaryFixes, 3)
        XCTAssertEqual(store.usage.wordsToday, 120)
        XCTAssertEqual(store.usage.recordingsToday, 1)
        XCTAssertEqual(store.usage.activeDays, 2)
    }

    func testMicrophonePermissionPresentationMatchesTCCState() {
        let store = NativeAppStore(supportDirectory: directory)
        switch store.microphoneAuthorizationStatus {
        case .notDetermined:
            XCTAssertEqual(store.microphoneStatusText, "Not requested")
            XCTAssertEqual(store.microphoneActionTitle, "Request Access")
        case .authorized:
            XCTAssertEqual(store.microphoneStatusText, "Granted")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        case .denied:
            XCTAssertEqual(store.microphoneStatusText, "Denied")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        case .restricted:
            XCTAssertEqual(store.microphoneStatusText, "Restricted")
            XCTAssertEqual(store.microphoneActionTitle, "Settings")
        @unknown default:
            XCTAssertEqual(store.microphoneStatusText, "Needs access")
        }
    }

    func testPermissionRefreshRechecksInputMonitoringAndNotifiesServices() {
        let store = NativeAppStore(supportDirectory: directory)
        var callbackCount = 0
        store.onPermissionsRefresh = { callbackCount += 1 }

        store.refreshPermissions()

        XCTAssertEqual(callbackCount, 1)
        XCTAssertEqual(store.inputMonitoringAvailable, CGPreflightListenEventAccess())
    }

    func testRecordingCuesAreOffUnlessChosen() throws {
        // Absent key on a fresh install and on an upgrade both mean silence.
        let fresh = NativeSettings()
        XCTAssertFalse(fresh.soundFeedback)

        for stored in [1, 2] {
            let json = Data("{\"schema_version\": \(stored)}".utf8)
            let decoded = try JSONDecoder().decode(NativeSettings.self, from: json)
            XCTAssertFalse(decoded.soundFeedback, "schema \(stored) opted in by accident")
        }

        let opted = Data("{\"schema_version\": 2, \"sound_feedback\": true}".utf8)
        XCTAssertTrue(try JSONDecoder().decode(NativeSettings.self, from: opted).soundFeedback)
    }
}
