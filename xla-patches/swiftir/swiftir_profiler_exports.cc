// SwiftIR Profiler C API Exports
// This file provides a C API to the TSL profiler for use from Swift.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

extern "C" {

// ============================================================================
// Version and Capability
// ============================================================================

const char* SwiftIRProfiler_Version(void) {
    return "2.0.0-tsl";
}

bool SwiftIRProfiler_IsTSLAvailable(void) {
    return true;
}

// ============================================================================
// Profiler Session API
// ============================================================================

struct SwiftIRProfilerSession {
    std::unique_ptr<tsl::ProfilerSession> session;
};

SwiftIRProfilerSession* SwiftIRProfiler_CreateSession(void) {
    auto* wrapper = new SwiftIRProfilerSession();
    tensorflow::ProfileOptions options;
    options.set_version(1);  // Protocol version
    options.set_include_dataset_ops(false);
    options.set_host_tracer_level(2);  // Level 2: XLA ops + user events
    options.set_device_tracer_level(1);
    options.set_python_tracer_level(0);  // No Python tracing
    options.set_enable_hlo_proto(true);  // Capture HLO module structure for op-level metrics
    wrapper->session = tsl::ProfilerSession::Create(options);
    return wrapper;
}

SwiftIRProfilerSession* SwiftIRProfiler_CreateSessionWithOptions(
    int32_t host_tracer_level,
    int32_t device_tracer_level,
    bool include_dataset_ops) {
    auto* wrapper = new SwiftIRProfilerSession();
    tensorflow::ProfileOptions options;
    options.set_version(1);
    options.set_include_dataset_ops(include_dataset_ops);
    options.set_host_tracer_level(host_tracer_level);
    options.set_device_tracer_level(device_tracer_level);
    options.set_python_tracer_level(0);
    options.set_enable_hlo_proto(true);  // Always capture HLO for XLA metrics
    wrapper->session = tsl::ProfilerSession::Create(options);
    return wrapper;
}

// Full configuration with all XLA profiling options
SwiftIRProfilerSession* SwiftIRProfiler_CreateSessionFull(
    int32_t host_tracer_level,
    int32_t device_tracer_level,
    int32_t python_tracer_level,
    bool enable_hlo_proto,
    bool include_dataset_ops) {
    auto* wrapper = new SwiftIRProfilerSession();
    tensorflow::ProfileOptions options;
    options.set_version(1);
    options.set_host_tracer_level(host_tracer_level);
    options.set_device_tracer_level(device_tracer_level);
    options.set_python_tracer_level(python_tracer_level);
    options.set_enable_hlo_proto(enable_hlo_proto);
    options.set_include_dataset_ops(include_dataset_ops);
    wrapper->session = tsl::ProfilerSession::Create(options);
    return wrapper;
}

void SwiftIRProfiler_DestroySession(SwiftIRProfilerSession* session) {
    delete session;
}

int32_t SwiftIRProfiler_SessionStatus(SwiftIRProfilerSession* session) {
    if (!session || !session->session) return -1;
    auto status = session->session->Status();
    return status.ok() ? 0 : static_cast<int32_t>(status.code());
}

// Collect data and return serialized XSpace protobuf
// Returns the size of the data, or -1 on error
// The caller must free the buffer using SwiftIRProfiler_FreeBuffer
int64_t SwiftIRProfiler_CollectData(SwiftIRProfilerSession* session,
                                     uint8_t** out_buffer) {
    if (!session || !session->session || !out_buffer) return -1;

    tensorflow::profiler::XSpace xspace;
    auto status = session->session->CollectData(&xspace);
    if (!status.ok()) return -1;

    size_t size = xspace.ByteSizeLong();
    *out_buffer = static_cast<uint8_t*>(malloc(size));
    if (!*out_buffer) return -1;

    if (!xspace.SerializeToArray(*out_buffer, size)) {
        free(*out_buffer);
        *out_buffer = nullptr;
        return -1;
    }

    return static_cast<int64_t>(size);
}

void SwiftIRProfiler_FreeBuffer(uint8_t* buffer) {
    free(buffer);
}

// ============================================================================
// TraceMe API (for marking regions in the trace)
// ============================================================================

// Start a TraceMe region. Returns an opaque handle.
// level: 1 = important, 2 = moderate, 3 = verbose
uint64_t SwiftIRProfiler_TraceMeStart(const char* name, int32_t level) {
    auto* traceme = new tsl::profiler::TraceMe(name, level);
    return reinterpret_cast<uint64_t>(traceme);
}

void SwiftIRProfiler_TraceMeStop(uint64_t handle) {
    auto* traceme = reinterpret_cast<tsl::profiler::TraceMe*>(handle);
    delete traceme;
}

// Simple scoped trace (for use when you can manage lifetimes manually)
void SwiftIRProfiler_TraceMeInstant(const char* name, int32_t level) {
    tsl::profiler::TraceMe traceme(name, level);
}

// ============================================================================
// XSpace File Export
// ============================================================================

// Export XSpace data to a file (TensorBoard format)
// Returns 0 on success, error code otherwise
int32_t SwiftIRProfiler_ExportToFile(const uint8_t* data, int64_t size,
                                      const char* filepath) {
    if (!data || size <= 0 || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -2;

    size_t written = fwrite(data, 1, size, f);
    fclose(f);

    return (written == static_cast<size_t>(size)) ? 0 : -3;
}

}  // extern "C"
