// XLAProfilerWrapper.h - C wrapper for XLA/TSL profiling functionality
//
// This header provides a C API for XLA profiling that can be used both:
// 1. Via Swift C++ interop (standard compilation)
// 2. Via dlopen/dlsym (Jupyter/REPL environments)
//
// The wrapper creates XSpace protobufs compatible with TensorBoard/XProf.

#ifndef SWIFTIR_XLA_PROFILER_WRAPPER_H
#define SWIFTIR_XLA_PROFILER_WRAPPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error Handling
// =============================================================================

typedef enum {
    SWIFTIR_PROFILER_OK = 0,
    SWIFTIR_PROFILER_ERROR_INVALID_ARGUMENT = 1,
    SWIFTIR_PROFILER_ERROR_NOT_INITIALIZED = 2,
    SWIFTIR_PROFILER_ERROR_ALREADY_ACTIVE = 3,
    SWIFTIR_PROFILER_ERROR_NOT_ACTIVE = 4,
    SWIFTIR_PROFILER_ERROR_EXPORT_FAILED = 5,
    SWIFTIR_PROFILER_ERROR_INTERNAL = 6,
    SWIFTIR_PROFILER_ERROR_UNSUPPORTED = 7,
} SwiftIRProfilerErrorCode;

// Get human-readable error message for the last error
const char* SwiftIRProfiler_GetLastError(void);

// =============================================================================
// Profiler Options
// =============================================================================

// Options for configuring the profiler session
typedef struct {
    // Host tracer level (0-3): Controls detail of host-side traces
    // 0: Disabled, 1: Minimal, 2: Moderate (default), 3: Verbose
    int32_t host_tracer_level;

    // Device tracer level (0-2): Controls device-side tracing
    // 0: Disabled, 1: Enabled (default), 2: Verbose
    int32_t device_tracer_level;

    // Python tracer level (0-1): For Python stack traces
    // 0: Disabled, 1: Enabled (default in Python contexts)
    int32_t python_tracer_level;

    // Whether to include HLO proto in the trace
    bool include_hlo_proto;

    // Duration hint in milliseconds (0 = no limit)
    uint64_t duration_ms;

} SwiftIRProfilerOptions;

// Get default profiler options
SwiftIRProfilerOptions SwiftIRProfiler_DefaultOptions(void);

// =============================================================================
// Profiler Session Management
// =============================================================================

// Opaque handle to a profiler session
typedef struct SwiftIRProfilerSession SwiftIRProfilerSession;

// Create a new profiler session with default options
// Returns NULL on failure (check SwiftIRProfiler_GetLastError())
SwiftIRProfilerSession* SwiftIRProfiler_CreateSession(void);

// Create a new profiler session with custom options
// Returns NULL on failure (check SwiftIRProfiler_GetLastError())
SwiftIRProfilerSession* SwiftIRProfiler_CreateSessionWithOptions(
    const SwiftIRProfilerOptions* options
);

// Destroy a profiler session and free resources
void SwiftIRProfiler_DestroySession(SwiftIRProfilerSession* session);

// =============================================================================
// Trace Capture
// =============================================================================

// Start collecting trace data
// Returns error code
SwiftIRProfilerErrorCode SwiftIRProfiler_Start(SwiftIRProfilerSession* session);

// Stop collecting and export trace to a directory
// The directory will be created if it doesn't exist
// Output format is TensorBoard/XProf compatible (.xplane.pb files)
SwiftIRProfilerErrorCode SwiftIRProfiler_StopAndExport(
    SwiftIRProfilerSession* session,
    const char* log_dir
);

// Stop collecting and get raw XSpace data as serialized protobuf
// Caller must free the returned buffer with SwiftIRProfiler_FreeBuffer()
// Returns error code, sets *out_data and *out_size on success
SwiftIRProfilerErrorCode SwiftIRProfiler_StopAndGetData(
    SwiftIRProfilerSession* session,
    uint8_t** out_data,
    size_t* out_size
);

// Free a buffer allocated by SwiftIRProfiler_StopAndGetData
void SwiftIRProfiler_FreeBuffer(uint8_t* buffer);

// Check if a session is currently active (tracing)
bool SwiftIRProfiler_IsActive(SwiftIRProfilerSession* session);

// =============================================================================
// TraceMe Annotations (named scopes in trace)
// =============================================================================

// Opaque handle to a TraceMe scope
typedef struct SwiftIRTraceMe SwiftIRTraceMe;

// Begin a named trace scope
// The name will appear in the TensorBoard Trace Viewer
// level: 1 = important, 2 = moderate detail, 3 = verbose
// Returns NULL if tracing is not active (safe to call unconditionally)
SwiftIRTraceMe* SwiftIRTraceMe_Start(const char* name, int32_t level);

// Begin a trace scope with additional metadata
// metadata_key and metadata_value are optional (can be NULL)
SwiftIRTraceMe* SwiftIRTraceMe_StartWithMetadata(
    const char* name,
    int32_t level,
    const char* metadata_key,
    const char* metadata_value
);

// End a trace scope
// Safe to call with NULL (no-op)
void SwiftIRTraceMe_Stop(SwiftIRTraceMe* trace);

// =============================================================================
// Simple Timing API (always available, even without TSL profiler)
// =============================================================================

// These functions provide basic timing that works everywhere,
// independent of the TSL profiler infrastructure.

// Opaque handle to a timing entry
typedef struct SwiftIRTimingEntry SwiftIRTimingEntry;

// Timing session for collecting custom timing data
typedef struct SwiftIRTimingSession SwiftIRTimingSession;

// Create a timing session
SwiftIRTimingSession* SwiftIRTiming_CreateSession(void);

// Destroy a timing session
void SwiftIRTiming_DestroySession(SwiftIRTimingSession* session);

// Start timing an operation
SwiftIRTimingEntry* SwiftIRTiming_Start(
    SwiftIRTimingSession* session,
    const char* name
);

// Stop timing and record duration
void SwiftIRTiming_Stop(SwiftIRTimingEntry* entry);

// Get timing statistics as JSON string
// Caller must free with SwiftIRProfiler_FreeBuffer()
SwiftIRProfilerErrorCode SwiftIRTiming_GetStatisticsJSON(
    SwiftIRTimingSession* session,
    char** out_json
);

// Export timing data to XSpace format (TensorBoard compatible)
// This creates a simplified XSpace with custom timing events
SwiftIRProfilerErrorCode SwiftIRTiming_ExportToXSpace(
    SwiftIRTimingSession* session,
    const char* log_dir,
    const char* run_name
);

// =============================================================================
// PJRT Profiler Extension Support
// =============================================================================

// Check if the loaded PJRT plugin supports the profiler extension
// This is relevant for TPU and some GPU plugins that expose device-level traces
bool SwiftIRProfiler_HasPJRTExtension(void* pjrt_client);

// Get the PJRT profiler extension from a client (if available)
// Returns NULL if not supported
void* SwiftIRProfiler_GetPJRTExtension(void* pjrt_client);

// =============================================================================
// Utility Functions
// =============================================================================

// Get current timestamp in nanoseconds (monotonic clock)
uint64_t SwiftIRProfiler_GetTimestampNs(void);

// Get profiler library version string
const char* SwiftIRProfiler_GetVersion(void);

// Check if full TSL profiler is available (vs. simple timing only)
bool SwiftIRProfiler_IsTSLAvailable(void);

#ifdef __cplusplus
}
#endif

#endif // SWIFTIR_XLA_PROFILER_WRAPPER_H
