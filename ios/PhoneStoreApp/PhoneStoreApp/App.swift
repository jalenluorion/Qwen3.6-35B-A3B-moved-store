import SwiftUI
import UIKit

@main
struct PhoneStoreApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var gate = SetupGate.shared
    @StateObject private var launch = LaunchGate()
    @Environment(\.scenePhase) private var scenePhase

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)   // line-buffered stdout for `devicectl ... --console`
    }

    var body: some Scene {
        WindowGroup {
            // First launch without a complete model: the setup (download) screen. The chat, and with it the engine, is
            // only created once the store is complete, so makeEngine() picks the native engine.
            if gate.showSetup {
                SetupScreen()
            } else if launch.deferred {
                Color.clear  // iOS launched the app in the background (download events): no chat, no engine (LaunchGate)
            } else {
                ChatRoot()
            }
        }
        .onChange(of: scenePhase) { _, p in
            if p == .active { launch.opened() }
        }
    }
}

/// A launch in the background with no arguments is iOS relaunching the app for the model download's background
/// URLSession (AppDelegate.handleEventsForBackgroundURLSession): it only services the download events. The chat, and
/// with it the engine load (~3 GB) and the launch warm-ups, waits until the app is opened (scene active), so a
/// relaunch never loads a second engine beside a test launch (2026-11-06: one was seen at 21:02 and 21:16). Test launches
/// (devicectl, always with arguments) and the Mac are never deferred.
@MainActor
final class LaunchGate: ObservableObject {
    @Published private(set) var deferred: Bool

    init() {
        #if targetEnvironment(macCatalyst)
        deferred = false
        #else
        deferred = UIApplication.shared.applicationState == .background && CommandLine.arguments.count <= 1
        #endif
        if deferred { log("LAUNCH in the background without arguments (the download session's events): the chat and the engine load wait until the app is opened") }
    }

    func opened() {
        guard deferred else { return }
        deferred = false
        log("LAUNCH opened: starting the chat")
    }
}

/// One engine per device, across processes: ChatModel.load takes an exclusive flock on Documents/.engine.lock before
/// loading the engine and keeps it for the life of the process (the kernel releases it when the process ends: a test
/// launch with devicectl --terminate-existing ends the old instance). Another live instance holding it means a second
/// engine would double the memory, so load waits for it (and is refused if it does not end).
enum EngineLock {
    private static var fd: Int32 = -1

    /// true when this process holds the lock (now or already); else the pid the holder wrote, if readable.
    static func tryAcquire() -> (ok: Bool, holder: String) {
        if fd >= 0 { return (true, "") }
        let path = docsURL.appendingPathComponent(".engine.lock").path
        let f = open(path, O_RDWR | O_CREAT, 0o644)
        guard f >= 0 else { return (true, "") }  // no lock file possible: do not block the load on it
        if flock(f, LOCK_EX | LOCK_NB) == 0 {
            let me = "\(getpid())\n"
            _ = ftruncate(f, 0)
            _ = me.withCString { pwrite(f, $0, strlen($0), 0) }
            fd = f
            return (true, "")
        }
        var buf = [UInt8](repeating: 0, count: 32)
        let n = pread(f, &buf, 31, 0)
        close(f)
        let holder = n > 0 ? String(decoding: buf[0..<n], as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines) : "?"
        return (false, holder)
    }
}

/// The setup screen as the app's root; opens the chat by itself when the download completes.
struct SetupScreen: View {
    @StateObject private var d = ModelDownloader.shared
    @Environment(\.scenePhase) private var phase
    var body: some View {
        SetupView(mode: .firstRun)
            .environmentObject(d)
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on while it downloads in front
                d.onComplete = { SetupGate.shared.downloadFinished() }
                d.resumeIfWanted()
            }
            .onChange(of: phase) { _, p in
                guard p == .active else { return }
                SetupGate.shared.appBecameActive()   // a download that finished in the background opens the chat now
                d.appBecameActive()                  // no space / no connection / failed: try again by itself
            }
    }
}

/// The chat app (as before 2026-11-06).
struct ChatRoot: View {
    @StateObject private var model = ChatModel()
    @Environment(\.scenePhase) private var phase

    var body: some View {
        RootView()
            .environmentObject(model)
            .onAppear {
                SetupGate.engineCreated = true
                UIApplication.shared.isIdleTimerDisabled = true
                model.startOnce()
            }
            .onChange(of: phase) { _, new in
                model.phaseChanged(new)
            }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    /// iOS relaunched the app (in the background) because the model download session has events: the downloader's
    /// session is recreated by ModelDownloader.shared, and this handler is called when its events are delivered.
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        log("DOWNLOAD background session events for \(identifier)")
        Task { @MainActor in ModelDownloader.shared.backgroundCompletion = completionHandler }
    }
}
