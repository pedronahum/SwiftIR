# TensorBoard Profiling Integration for SwiftIR

## Executive Summary

SwiftIR provides TensorBoard/XProf profiling support, enabling users to visualize execution traces, memory usage, and performance bottlenecks. Since SwiftIR uses the same MLIR/XLA/PJRT stack as JAX, the profiling infrastructure leverages XLA's native profiling mechanisms.

**Status: Fully Implemented**

---

## Quick Start

### 1. Run ProfilerDemo

```bash
SWIFTIR_DEPS=/opt/swiftir-deps \
LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH \
swift run ProfilerDemo
```

### 2. View in TensorBoard

```bash
tensorboard --logdir=/tmp/swiftir_profiler_test --port=6006
# Open http://localhost:6006 -> Profile tab
```

---

## Architecture Overview

```
+---------------------------------------------------------------+
|  Swift User Code                                               |
|  - pjrtTrainStep(step) { ... }  // Step markers                |
|  - pjrtTraced("name") { ... }   // Custom traces               |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  SwiftIRProfiler Module                                        |
|  - PJRTProfiler: Start/stop profiling, collect XSpace data     |
|  - TraceMe API: Unified Swift/XLA tracing                      |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  PJRT CPU Plugin (patched with profiler extension)             |
|  - Captures XLA compilation, HLO modules, execution            |
|  - Exports TraceMe C API for unified tracing                   |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  .xplane.pb Files (Protocol Buffers)                           |
|  - TensorBoard-compatible format                               |
|  - Contains trace events, HLO graphs, timing data              |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|  TensorBoard / XProf Visualization                             |
|  - Trace Viewer (timeline)                                     |
|  - Overview Page (step timing)                                 |
|  - Memory Profile                                              |
+---------------------------------------------------------------+
```

---

## Swift API

### PJRTProfiler

The main profiler class for capturing XLA execution traces:

```swift
import SwiftIRProfiler
import SwiftIRXLA
import Foundation

// IMPORTANT: Initialize PJRT client FIRST - this loads the plugin
let client = try PJRTClient(backend: .cpu)

// Check if profiler extension is available
guard PJRTProfiler.isAvailable else {
    print("Profiler not available")
    return
}

// Create and start profiler
let profiler = try PJRTProfiler.create()
try profiler.start()

// ... run XLA computations ...

// Stop and collect data
try profiler.stop()
let xspaceData = try profiler.collectData()

// Export to file
let hostname = ProcessInfo.processInfo.hostName
let logDir = "/tmp/my_profile/plugins/profile/run_1"
try FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true, attributes: nil)
try PJRTProfiler.exportToFile(xspaceData, filepath: "\(logDir)/\(hostname).xplane.pb")
```

### TraceMe API (Unified Tracing)

Swift traces appear in the SAME profile as XLA internal traces:

```swift
// Automatic scope tracing - use pjrtTraced for unified profiling
let result = try pjrtTraced("forward_pass") {
    try executable.execute(arguments: inputs, device: device)
}

// Training step markers (for TensorBoard Overview Page)
// Step 0 is typically warmup
try pjrtTrainStep(0) {
    try pjrtTraced("warmup") {
        // warmup computation
    }
}

// Training loop with step markers
for step in 1...100 {
    try pjrtTrainStep(step) {
        try pjrtTraced("compute_loss") {
            // ...
        }
        try pjrtTraced("compute_gradients") {
            // ...
        }
    }
}
```

### Low-Level TraceMe API

For fine-grained control:

```swift
// Check if TraceMe API is available (requires patched PJRT plugin)
if PJRTProfiler.hasTraceMeApi {
    // Check if tracing is active (avoid expensive string ops when disabled)
    if PJRTProfiler.traceMeActive() {
        let activityId = PJRTProfiler.traceMeStart("expensive_operation")
        // ... do work ...
        PJRTProfiler.traceMeStop(activityId)
    }

    // Instant events (no duration, just a point in time)
    PJRTProfiler.traceMeInstant("checkpoint_reached")
}
```

### Checking API Availability

```swift
// Check profiler extension availability (PJRT must be loaded first)
let hasPJRTProfiler = PJRTProfiler.isAvailable

// Check TraceMe API availability (for unified Swift/XLA tracing)
let hasTraceMeApi = PJRTProfiler.hasTraceMeApi

print("PJRT Profiler: \(hasPJRTProfiler ? "Available" : "Not available")")
print("TraceMe API: \(hasTraceMeApi ? "Available" : "Not available")")
```

---

## Implementation Checklist

### Core Infrastructure
- [x] Create `SwiftIRProfiler` module
- [x] Add C bindings for PJRT profiler extension
- [x] Implement `PJRTProfiler.create()` / `start()` / `stop()` / `collectData()`
- [x] XSpace protobuf export via `PJRTProfiler.exportToFile()`

### Trace Annotations
- [x] Implement TraceMe C API in PJRT plugin (`PJRT_TraceMeStart/Stop/Active/Instant`)
- [x] Add `pjrtTraced()` helper function (auto-selects PJRT or fallback)
- [x] Add `pjrtTrainStep()` for step markers
- [x] Unified tracing (Swift + XLA in same profile via shared TraceMeRecorder)

### XSpace Compatibility
- [x] Task Environment XPlane preservation (required by TensorBoard)
- [x] Hostname field injection (via `repairXSpace()`)
- [x] Trailing null byte removal
- [x] TensorBoard parsing verified

### Build System
- [x] GitHub Actions builds patched PJRT plugin
- [x] Local install script builds patched plugin
- [x] Plugin includes profiler extension + TraceMe API

### Documentation
- [x] TensorBoard Profiling Guide (this document)
- [x] Implementation Status document (`docs/TensorBoard_Profiling_Status.md`)
- [x] ProfilerDemo example

### Future Enhancements
- [ ] On-demand profiling via gRPC server
- [ ] Memory profiling integration
- [ ] GPU backend support (waiting for CUDA plugin build)

---

## Backend Support

| Backend | Profiler Extension | TraceMe API | Notes |
|---------|-------------------|-------------|-------|
| **CPU** | Yes | Yes | Full support via patched `pjrt_c_api_cpu_plugin.so` |
| **GPU** | Planned | Planned | Waiting for CUDA plugin build with profiler patches |
| **TPU** | No | No | `libtpu.so` is provided by Google and cannot be patched |

### TPU Profiling Limitations

The TPU backend uses Google's `libtpu.so` library, which is a pre-built binary that we cannot modify. This means:

1. **No SwiftIR Profiler Extension**: The `PJRT_Profiler_*` functions are not available
2. **No Custom TraceMe API**: The `PJRT_TraceMeStart/Stop` exports don't exist in `libtpu.so`
3. **XLA Internal Profiling Still Works**: XLA's internal profiling is built into libtpu, so basic execution traces are captured

For TPU profiling, users should use standard JAX/TensorFlow profiling tools or Cloud TPU Profiler, which work with libtpu's built-in profiling capabilities.

### Checking Backend Profiling Support

```swift
// After initializing PJRTClient
let hasPJRTProfiler = PJRTProfiler.isAvailable   // Works on CPU, not TPU
let hasTraceMeApi = PJRTProfiler.hasTraceMeApi   // Works on CPU, not TPU

if !hasPJRTProfiler {
    print("Profiler extension not available on this backend")
    print("For TPU, use Cloud TPU Profiler or JAX profiling tools")
}
```

---

## Output Directory Structure

SwiftIR profiles use the TensorBoard-compatible structure:

```
/tmp/swiftir_profiler_test/
+-- plugins/
    +-- profile/
        +-- run_1/
            +-- <hostname>.xplane.pb   # Main trace data
```

The hostname is automatically obtained from `ProcessInfo.processInfo.hostName`.

---

## TensorBoard Tools Available

Once profiles are captured, users can visualize them with:

| Tool | Description |
|------|-------------|
| **Trace Viewer** | Timeline showing XLA ops, custom scopes, device activity |
| **Overview Page** | Step timing breakdown, recommendations |
| **Memory Profile** | Memory usage over time |
| **HLO Op Profile** | Time spent per HLO operation |

### Installing TensorBoard

```bash
pip install tensorboard tensorboard-plugin-profile
```

---

## Complete Example: ProfilerDemo

This is the actual implementation from `Examples/ProfilerDemo.swift`:

```swift
import SwiftIRProfiler
import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders
import SwiftIRStableHLO
import SwiftIRXLA
import Foundation

@main
struct ProfilerDemo {
    static func main() throws {
        // STEP 1: Initialize PJRT first - this loads the plugin with profiler
        let cpuClient = try PJRTClient(backend: .cpu)
        guard let device = cpuClient.addressableDevices.first else {
            print("No devices available")
            return
        }

        // STEP 2: Check profiler availability
        let hasPJRTProfiler = PJRTProfiler.isAvailable
        let hasTraceMeApi = PJRTProfiler.hasTraceMeApi
        print("PJRT Profiler Extension: \(hasPJRTProfiler ? "Available" : "Not available")")
        print("PJRT TraceMe API: \(hasTraceMeApi ? "Available" : "Not available")")

        // STEP 3: Create and start profiler
        guard hasPJRTProfiler else {
            print("Profiler not available")
            return
        }
        let profiler = try PJRTProfiler.create()
        try profiler.start()

        // STEP 4: Run computations with step markers and traces
        // Warmup (step 0)
        try pjrtTrainStep(0) {
            try pjrtTraced("warmup") {
                try runMatMul(client: cpuClient, device: device, size: 64)
            }
        }

        // Training steps
        for stepNum in 1...5 {
            try pjrtTrainStep(stepNum) {
                try pjrtTraced("train_step_\(stepNum)") {
                    try runMatMul(client: cpuClient, device: device, size: 256)
                }
            }
        }

        // STEP 5: Collect and export
        try profiler.stop()
        let data = try profiler.collectData()

        let logDir = "/tmp/swiftir_profiler_test"
        let pluginDir = "\(logDir)/plugins/profile/run_1"
        try FileManager.default.createDirectory(atPath: pluginDir, withIntermediateDirectories: true, attributes: nil)

        let hostname = ProcessInfo.processInfo.hostName
        let filepath = "\(pluginDir)/\(hostname).xplane.pb"
        try PJRTProfiler.exportToFile(data, filepath: filepath)

        print("Exported XSpace to: \(filepath)")
        print("View with: tensorboard --logdir=\(logDir)")
    }
}
```

---

## Troubleshooting

### Profiler Extension Not Available

If `PJRTProfiler.isAvailable` returns `false`:

1. **Ensure PJRT client is initialized first** - The profiler extension is loaded with the PJRT plugin
2. Verify patched plugin is installed:
   ```bash
   nm -D /opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so | grep PJRT_Profiler
   ```

### TraceMe API Not Available

If `PJRTProfiler.hasTraceMeApi` returns `false`:

1. Verify patched plugin has TraceMe exports:
   ```bash
   nm -D /opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so | grep TraceMeStart
   # Should show: PJRT_TraceMeStart
   ```

2. Rebuild plugin from `xla-patches/pjrt_cpu/`

### TensorBoard Shows Empty Profile

1. Check XSpace file exists and has content:
   ```bash
   ls -la /tmp/swiftir_profiler_test/plugins/profile/run_1/*.xplane.pb
   ```

2. Verify hostname matches file pattern:
   - File should be named `<hostname>.xplane.pb`
   - Check with: `hostname` command

3. Check TensorBoard logs for parse errors

### Traces Not Appearing in Profile

1. Ensure profiler is started **before** computations
2. Ensure profiler is stopped **after** computations (before `collectData()`)
3. Use `pjrtTraced()` not `traced()` for unified tracing
4. Check `PJRTProfiler.traceMeActive()` returns `true` during profiling

---

## Key Files

| File | Purpose |
|------|---------|
| `Sources/SwiftIRProfiler/PJRTProfiler.swift` | Main profiler class and TraceMe API |
| `Sources/PJRTCWrappers/PJRTSimpleWrapper.c` | C wrapper for TraceMe dlsym lookup |
| `xla-patches/pjrt_cpu/pjrt_c_api_cpu_internal_profiler.cc` | Patched PJRT plugin with TraceMe exports |
| `Examples/ProfilerDemo.swift` | Complete working example |
| `docs/TensorBoard_Profiling_Status.md` | Implementation status document |

---

## References

- [JAX Profiling Documentation](https://docs.jax.dev/en/latest/profiling.html)
- [XProf GitHub Repository](https://github.com/openxla/xprof)
- [TensorBoard Profiler Guide](https://www.tensorflow.org/guide/profiler)
- [PJRT C API Header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
