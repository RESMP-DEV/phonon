import SwiftUI

struct PermissionSummary: View {
    @ObservedObject var store: NativeAppStore

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Permissions and health").font(.headline)
                    Spacer()
                    Button("Refresh") { store.refreshPermissions() }
                        .buttonStyle(.borderless)
                }
                PermissionRow(
                    name: "Microphone", granted: store.microphonePermission,
                    statusText: store.microphoneStatusText,
                    actionTitle: store.microphoneActionTitle,
                    action: { store.performMicrophonePermissionAction() })
                PermissionRow(
                    name: "Accessibility", granted: store.accessibilityPermission,
                    action: { PrivacyPane.accessibility.open() })
                PermissionRow(
                    name: "Input Monitoring", granted: store.inputMonitoringAvailable,
                    actionTitle: store.inputMonitoringActionTitle,
                    action: { store.performInputMonitoringPermissionAction() })
                PermissionRow(
                    name: "Screen Recording", granted: store.screenRecordingPermission,
                    actionTitle: store.screenRecordingActionTitle,
                    action: { store.performScreenRecordingPermissionAction() })
            }
        }
    }
}

struct PermissionGuideView: View {
    let guide: PermissionGuide
    let onDone: () -> Void
    private let bundleURL = Bundle.main.bundleURL

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(guide.title).font(.title2.bold())
            Text(guide.detail).foregroundStyle(EmberTheme.muted)

            HStack(spacing: 14) {
                Image(nsImage: NSWorkspace.shared.icon(forFile: bundleURL.path))
                    .resizable()
                    .frame(width: 58, height: 58)
                VStack(alignment: .leading, spacing: 4) {
                    Text(bundleName).font(.headline)
                    Text(bundleURL.path)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(EmberTheme.muted)
                        .textSelection(.enabled)
                }
            }
            .padding(14)
            .background(EmberTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 9) {
                ForEach(Array(guide.steps.enumerated()), id: \.offset) { step in
                    HStack(alignment: .firstTextBaseline, spacing: 9) {
                        Text("\(step.offset + 1)")
                            .font(.caption.bold())
                            .frame(width: 20, height: 20)
                            .background(EmberTheme.surfaceRaised, in: Circle())
                        Text(step.element).font(.callout)
                    }
                }
            }

            HStack {
                Button("Quit Phonon", action: onDone)
                Spacer()
                Button("Show in Finder") { showInFinder() }
            }
            Button(guide.pane.openTitle) { guide.pane.open() }
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
        }
        .padding(24)
        .frame(width: 540)
        .background(EmberTheme.background)
        .foregroundStyle(EmberTheme.text)
        .preferredColorScheme(.dark)
    }

    private var bundleName: String {
        bundleURL.pathExtension == "app" ? bundleURL.lastPathComponent : "Phonon"
    }

    private func showInFinder() {
        if bundleURL.pathExtension == "app" {
            NSWorkspace.shared.activateFileViewerSelecting([bundleURL])
        } else {
            NSWorkspace.shared.open(bundleURL.deletingLastPathComponent())
        }
    }
}
