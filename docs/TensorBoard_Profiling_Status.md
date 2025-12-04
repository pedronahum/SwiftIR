# SwiftIR TensorBoard Profiling - Implementation Status

**Last Updated:** December 3, 2024

## Executive Summary

SwiftIR has a **fully working** PJRT profiler extension that captures XLA internal profiling data and exports TensorBoard-compatible `.xplane.pb` files. The implementation includes unified Swift/XLA tracing via TraceMe API.

### Current Status: Working

| Component | Status | Notes |
|-----------|--------|-------|
| PJRT Profiler Extension | Working | Captures XLA internal traces |
| XSpace Protobuf Generation | Working | Valid protobuf verified with TensorBoard |
| Swift Profiler API | Working | `PJRTProfiler` class functional |
| TraceMe API (Unified Tracing) | Working | Swift traces appear in same profile as XLA |
| TensorBoard Parsing | Working | Profile tab loads successfully |
| Step Markers | Working | `pjrtTrainStep()` for Overview Page |

---

## Architecture

```
+-----------------------------------------------------------------------------+
|  Swift User Code (ProfilerDemo.swift)                                        |
|  - pjrtTrainStep() for step markers                                          |
|  - pjrtTraced() for custom traces                                            |
|  - PJRTProfiler.start() / stop() / collectData()                             |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|  SwiftIRProfiler Module (Sources/SwiftIRProfiler/)                           |
|  - PJRTProfiler.swift: High-level Swift API + TraceMe wrappers               |
|  - ProfilerBindings.swift: Legacy TSL profiler bindings                      |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|  C Wrapper (Sources/PJRTCWrappers/)                                          |
|  - PJRTSimpleWrapper.c: Loads TraceMe API from plugin via dlsym              |
|  - swiftir_pjrt_profiler.cpp: PJRT profiler extension calls                  |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|  PJRT CPU Plugin (pjrt_c_api_cpu_plugin.so) - PATCHED VERSION                |
|  - pjrt_c_api_cpu_internal_profiler.cc: Custom plugin with profiler ext      |
|  - Exports PJRT_TraceMeStart/Stop/Active/Instant for unified tracing         |
|  - Uses tsl::profiler::TraceMeRecorder (shared instance)                     |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|  Output: /tmp/swiftir_profiler_test/plugins/profile/run_1/*.xplane.pb        |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
|  TensorBoard (tensorboard --logdir=/tmp/swiftir_profiler_test)               |
|  - Profile tab shows Trace Viewer, Overview Page, etc.                       |
+-----------------------------------------------------------------------------+
```

---

## Key Implementation Details

### 1. TraceMe API for Unified Tracing

The critical insight: XLA's `tsl::profiler::TraceMeRecorder` is a **thread-local singleton**. If Swift creates traces via a separate library (`libswiftir_profiler.so`), those traces go to a different recorder instance than XLA's internal traces.

**Solution:** Export TraceMe C API directly from the PJRT plugin:

```cpp
// In pjrt_c_api_cpu_internal_profiler.cc
extern "C" {
    int64_t PJRT_TraceMeStart(const char* name, int32_t level);
    void PJRT_TraceMeStop(int64_t activity_id);
    bool PJRT_TraceMeActive(int32_t level);
    void PJRT_TraceMeInstant(const char* name, int32_t level);
}
```

Swift loads these via `dlsym()` from the already-loaded plugin handle, ensuring all traces use the SAME `TraceMeRecorder`.

### 2. Step Markers for TensorBoard Overview Page

TensorBoard's Overview Page requires step markers with specific format:

```swift
// Creates: "TrainStep#step_num=0,_r=1#"
try pjrtTrainStep(0) {
    // training code
}
```

### 3. Required XSpace Fields

TensorBoard requires:
- **Task Environment XPlane**: Contains `profile_start_time` and `profile_stop_time`
- **Hostname field**: Added during export in `repairXSpace()`

---

## Files in SwiftIR Repository

### xla-patches/pjrt_cpu/
| File | Purpose |
|------|---------|
| `pjrt_c_api_cpu_internal_profiler.cc` | Custom CPU plugin with profiler ext + TraceMe API |
| `BUILD` | Bazel build for patched plugin |

### Sources/SwiftIRProfiler/
| File | Purpose |
|------|---------|
| `PJRTProfiler.swift` | High-level Swift API, TraceMe wrappers, XSpace repair |
| `ProfilerBindings.swift` | Legacy TSL profiler bindings (fallback) |
| `ProfilerErrors.swift` | Error types |

### Sources/PJRTCWrappers/
| File | Purpose |
|------|---------|
| `PJRTSimpleWrapper.c` | dlsym lookup for TraceMe API from plugin |
| `include/PJRTSimpleWrapper.h` | TraceMe function declarations |
| `swiftir_pjrt_profiler.cpp` | PJRT profiler extension C wrapper |

### Examples/
| File | Purpose |
|------|---------|
| `ProfilerDemo.swift` | Full demo with step markers and traces |
| `ProfilerTest.swift` | Basic profiler test |

---

## Usage

### Generate Profile

```bash
rm -rf /tmp/swiftir_profiler_test/*
SWIFTIR_DEPS=/opt/swiftir-deps \
LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH \
swift run ProfilerDemo
```

### View in TensorBoard

```bash
tensorboard --logdir=/tmp/swiftir_profiler_test --port=6006
# Open http://localhost:6006 -> Profile tab
```

### Swift API Example

```swift
import SwiftIRProfiler

// Start profiling
let profiler = try PJRTProfiler.create()
try profiler.start()

// Run computations with traces
for step in 0..<10 {
    try pjrtTrainStep(step) {
        try pjrtTraced("forward_pass") {
            // XLA computation
        }
        try pjrtTraced("backward_pass") {
            // Gradient computation
        }
    }
}

// Collect and export
try profiler.stop()
let data = try profiler.collectData()
try PJRTProfiler.exportToFile(data, filepath: "/tmp/profile/host.xplane.pb")
```

---

## Build System Integration

### GitHub Actions (ubuntu-build-dependencies.yml)

The CI workflow:
1. Clones XLA repository
2. Copies `xla-patches/pjrt_cpu/` to XLA
3. Builds `pjrt_c_api_cpu_plugin_profiler.so`
4. Renames to `pjrt_c_api_cpu_plugin.so` for compatibility
5. Includes in SDK artifact

### Local Installation (install-swiftir-ubuntu.sh)

Same process as CI - builds patched plugin with profiler extension and TraceMe API.

---

## Verification

Check TraceMe API is available:

```bash
nm -D /opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so | grep TraceMeStart
# Should show: PJRT_TraceMeStart
```

Run ProfilerDemo and check output:

```
PJRT Profiler Extension: Available
PJRT TraceMe API: Available (Swift traces in same profile)
```

---

## Resolved Issues

1. **Task Environment XPlane removal** - Fixed by removing truncation logic in `repairXSpace()`
2. **Trailing null bytes** - Fixed by stripping in `repairXSpace()`
3. **Missing hostname field** - Fixed by `addHostnameField()` in `repairXSpace()`
4. **Separate TraceMeRecorder instances** - Fixed by exporting TraceMe API from PJRT plugin

---

## Backend Support

| Backend | Profiler Extension | TraceMe API | Status |
|---------|-------------------|-------------|--------|
| **CPU** | Yes | Yes | Full support |
| **GPU** | Planned | Planned | Waiting for CUDA plugin build |
| **TPU** | TBD | TBD | Needs investigation on TPU VM |

### TPU Profiling Investigation

The TPU backend uses Google's `libtpu.so`, a pre-built binary. Key questions to investigate on a TPU VM:

1. **Does libtpu export the PJRT Profiler Extension?**
   ```bash
   nm -D /lib/libtpu.so | grep -i profiler
   nm -D /lib/libtpu.so | grep PJRT_Extension
   ```

2. **Check PJRT API extension chain for profiler:**
   - The official PJRT C API has `PJRT_Extension_Type_Profiler`
   - libtpu may expose `PLUGIN_Profiler_Api` via this extension

3. **Alternative profiling methods for TPU:**
   - JAX's `jax.profiler.trace()` works via internal libtpu profiling
   - TensorBoard "Capture Profile" connects to TPU VM port 6009
   - XProf CLI: `xprof capture --tpu=localhost:6009`

**Action Required:** Test on TPU VM to determine if standard PJRT profiler extension is available.

---

## Roadmap

### Completed
- [x] PJRT Profiler Extension for CPU
- [x] TraceMe API for unified Swift/XLA tracing
- [x] XSpace protobuf generation with TensorBoard compatibility
- [x] Step markers (`pjrtTrainStep()`)
- [x] ProfilerDemo example
- [x] GitHub Actions CI builds patched plugin
- [x] Local install script builds patched plugin

### In Progress
- [ ] TPU profiler extension investigation (test on TPU VM)
- [ ] SwiftIRJupyter profiling integration

### Future
- [ ] GPU profiler extension (requires CUDA plugin build)
- [ ] On-demand profiling via gRPC server
- [ ] Memory profiling integration

---

## References

- [XLA Profiler Plugin Source](https://github.com/openxla/xla/tree/main/xla/backends/profiler/plugin)
- [TensorBoard Profile Plugin](https://github.com/tensorflow/profiler)
- [XSpace Proto Definition](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto)
- [JAX Profiling Documentation](https://jax.readthedocs.io/en/latest/profiling.html)
- [Profile TPU VMs](https://cloud.google.com/tpu/docs/profile-tpu-vm)
