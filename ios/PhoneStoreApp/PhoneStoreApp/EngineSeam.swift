import Foundation

// THE SEAM. This is the only place the app chooses an engine.
//
//   -stub                     the stub (canned text, synthetic stats), whatever the store
//   -autotest / -teacherForce / -requireStore the native engine (PhoneStoreChatEngine) whatever the store: ChatModel.load() refuses it
//                             cleanly when the store is incomplete or the cap is below Budget.minCap
//   otherwise                 the native engine when the store in the model folder is complete, else the stub
//
// The choice and its reason are shown in the settings sheet and written to probe.json (`engine.choice`).

func makeEngine() -> (ChatEngine, String) {
    let real = { PhoneStoreChatEngine(cacheDir: nativeCacheURL) }
    if LaunchArgs.has("-stub") { return (StubEngine(), "stub: -stub launch argument") }
    if LaunchArgs.has("-teacherForce") { return (real(), "native: -teacherForce always runs the native engine") }
    if LaunchArgs.has("-autotest") { return (real(), "native: -autotest always runs the native engine") }
    if LaunchArgs.has("-requireStore") { return (real(), "native: -requireStore launch argument") }
    let st = StoreStatus.check(dir: modelDirURL)
    if st.complete { return (real(), "native: store complete in \(modelDirURL.lastPathComponent)/") }
    return (StubEngine(), "stub: store incomplete (\(st.problems.count) problem(s)); copy the store in and relaunch for the native engine")
}
