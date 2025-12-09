//===-- PJRTDirect.h - Direct PJRT C++ Wrapper ----------------*- C -*-===//
//
// SwiftIR - Optimized PJRT wrapper using C++ for better performance
//
// This wrapper provides a clean C API for Swift while using C++ internally
// to optimize buffer management, caching, and reduce overhead compared to
// the pure C wrapper (PJRTSimpleWrapper).
//
// Key optimizations:
// 1. Cached API struct - no repeated dlsym lookups
// 2. Pre-allocated argument structs - no memset overhead on hot path
// 3. Execution contexts - reuse buffers and metadata across calls
// 4. Multi-backend support - CPU/GPU/TPU through plugin loading
//
//===------------------------------------------------------------------===//

#ifndef PJRT_DIRECT_H
#define PJRT_DIRECT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//===------------------------------------------------------------------===//
// Opaque Handles
//===------------------------------------------------------------------===//

/// Opaque handle to a PJRT client (manages devices and compilation)
typedef struct PJRTDirect_Client_s* PJRTDirect_Client;

/// Opaque handle to a PJRT device (CPU core, GPU, TPU chip)
typedef struct PJRTDirect_Device_s* PJRTDirect_Device;

/// Opaque handle to a compiled executable
typedef struct PJRTDirect_Executable_s* PJRTDirect_Executable;

/// Opaque handle to an execution context (for repeated execution)
typedef struct PJRTDirect_ExecContext_s* PJRTDirect_ExecContext;

//===------------------------------------------------------------------===//
// Error Handling
//===------------------------------------------------------------------===//

/// Status code for PJRT operations
typedef enum {
    PJRT_DIRECT_OK = 0,
    PJRT_DIRECT_ERROR_INVALID_ARGUMENT = 1,
    PJRT_DIRECT_ERROR_NOT_FOUND = 2,
    PJRT_DIRECT_ERROR_ALREADY_EXISTS = 3,
    PJRT_DIRECT_ERROR_PERMISSION_DENIED = 4,
    PJRT_DIRECT_ERROR_RESOURCE_EXHAUSTED = 5,
    PJRT_DIRECT_ERROR_FAILED_PRECONDITION = 6,
    PJRT_DIRECT_ERROR_ABORTED = 7,
    PJRT_DIRECT_ERROR_OUT_OF_RANGE = 8,
    PJRT_DIRECT_ERROR_UNIMPLEMENTED = 9,
    PJRT_DIRECT_ERROR_INTERNAL = 10,
    PJRT_DIRECT_ERROR_UNAVAILABLE = 11,
    PJRT_DIRECT_ERROR_DATA_LOSS = 12,
    PJRT_DIRECT_ERROR_UNAUTHENTICATED = 13,
    PJRT_DIRECT_ERROR_UNKNOWN = 14,
} PJRTDirect_ErrorCode;

/// Status structure with error code and optional message
typedef struct {
    PJRTDirect_ErrorCode code;
    char* message;  // Heap-allocated, NULL if success. Caller must free.
} PJRTDirect_Status;

/// Free a status message (call after handling error)
void PJRTDirect_StatusFree(PJRTDirect_Status* status);

/// Check if status indicates success
static inline bool PJRTDirect_StatusOk(PJRTDirect_Status status) {
    return status.code == PJRT_DIRECT_OK;
}

//===------------------------------------------------------------------===//
// Backend Selection
//===------------------------------------------------------------------===//

/// Supported compute backends
typedef enum {
    PJRT_BACKEND_CPU = 0,
    PJRT_BACKEND_GPU_CUDA = 1,
    PJRT_BACKEND_GPU_ROCM = 2,
    PJRT_BACKEND_TPU = 3,
} PJRTDirect_Backend;

/// Get the default plugin path for a backend
/// Returns NULL if backend is not supported or path is not set
const char* PJRTDirect_GetDefaultPluginPath(PJRTDirect_Backend backend);

/// Set the default plugin path for a backend (overrides environment)
void PJRTDirect_SetDefaultPluginPath(PJRTDirect_Backend backend, const char* path);

//===------------------------------------------------------------------===//
// Client Management
//===------------------------------------------------------------------===//

/// Create a client for the specified backend
///
/// @param backend      Which backend to use (CPU/GPU/TPU)
/// @param plugin_path  Path to PJRT plugin .so (NULL for default)
/// @param out_client   Output: created client handle
/// @return Status (PJRT_DIRECT_OK on success)
PJRTDirect_Status PJRTDirect_CreateClient(
    PJRTDirect_Backend backend,
    const char* plugin_path,
    PJRTDirect_Client* out_client
);

/// Destroy a client and release all resources
void PJRTDirect_DestroyClient(PJRTDirect_Client client);

/// Get the platform name (e.g., "cpu", "cuda", "tpu")
const char* PJRTDirect_GetPlatformName(PJRTDirect_Client client);

/// Get the platform version string
const char* PJRTDirect_GetPlatformVersion(PJRTDirect_Client client);

/// Get number of devices available
int32_t PJRTDirect_GetDeviceCount(PJRTDirect_Client client);

/// Get a specific device by index
///
/// @param client       Client handle
/// @param device_index Device index (0 to device_count-1)
/// @param out_device   Output: device handle (non-owning, valid while client exists)
/// @return Status
PJRTDirect_Status PJRTDirect_GetDevice(
    PJRTDirect_Client client,
    int32_t device_index,
    PJRTDirect_Device* out_device
);

/// Get the default device (usually device 0)
PJRTDirect_Status PJRTDirect_GetDefaultDevice(
    PJRTDirect_Client client,
    PJRTDirect_Device* out_device
);

//===------------------------------------------------------------------===//
// Compilation
//===------------------------------------------------------------------===//

/// Compilation options
typedef struct {
    int32_t num_replicas;       // Number of replicas (default: 1)
    int32_t num_partitions;     // Number of partitions (default: 1)
    int32_t optimization_level; // XLA opt level: -1=default, 0-3
} PJRTDirect_CompileOptions;

/// Get default compile options
PJRTDirect_CompileOptions PJRTDirect_DefaultCompileOptions(void);

/// Compile an MLIR/StableHLO module to an executable
///
/// @param client       Client to compile for
/// @param mlir_module  MLIR module string (StableHLO dialect)
/// @param mlir_size    Size of MLIR string in bytes
/// @param options      Compilation options
/// @param out_exec     Output: compiled executable handle
/// @return Status
PJRTDirect_Status PJRTDirect_Compile(
    PJRTDirect_Client client,
    const char* mlir_module,
    size_t mlir_size,
    PJRTDirect_CompileOptions options,
    PJRTDirect_Executable* out_exec
);

/// Destroy an executable
void PJRTDirect_DestroyExecutable(PJRTDirect_Executable exec);

/// Get the number of inputs expected by the executable
int32_t PJRTDirect_GetNumInputs(PJRTDirect_Executable exec);

/// Get the number of outputs produced by the executable
int32_t PJRTDirect_GetNumOutputs(PJRTDirect_Executable exec);

//===------------------------------------------------------------------===//
// Data Types
//===------------------------------------------------------------------===//

/// Supported data types for buffers
typedef enum {
    PJRT_DTYPE_F32 = 0,   // 32-bit float
    PJRT_DTYPE_F64 = 1,   // 64-bit float
    PJRT_DTYPE_F16 = 2,   // 16-bit float
    PJRT_DTYPE_BF16 = 3,  // bfloat16
    PJRT_DTYPE_I8 = 4,    // 8-bit signed integer
    PJRT_DTYPE_I16 = 5,   // 16-bit signed integer
    PJRT_DTYPE_I32 = 6,   // 32-bit signed integer
    PJRT_DTYPE_I64 = 7,   // 64-bit signed integer
    PJRT_DTYPE_U8 = 8,    // 8-bit unsigned integer
    PJRT_DTYPE_U16 = 9,   // 16-bit unsigned integer
    PJRT_DTYPE_U32 = 10,  // 32-bit unsigned integer
    PJRT_DTYPE_U64 = 11,  // 64-bit unsigned integer
    PJRT_DTYPE_BOOL = 12, // Boolean
    PJRT_DTYPE_C64 = 13,  // Complex64 (2x float32)
    PJRT_DTYPE_C128 = 14, // Complex128 (2x float64)
} PJRTDirect_DType;

/// Get the size in bytes of a data type element
size_t PJRTDirect_DTypeSize(PJRTDirect_DType dtype);

//===------------------------------------------------------------------===//
// Timing Information
//===------------------------------------------------------------------===//

/// Timing breakdown for execution profiling (all times in nanoseconds)
typedef struct {
    uint64_t h2d_ns;       // Host-to-device transfer time
    uint64_t execute_ns;   // Kernel execution time
    uint64_t d2h_ns;       // Device-to-host transfer time
    uint64_t total_ns;     // Total end-to-end time
} PJRTDirect_Timing;

//===------------------------------------------------------------------===//
// Execution - Simple API
//===------------------------------------------------------------------===//

/// Host buffer semantics for input data
typedef enum {
    /// Data is copied immediately, caller can modify after call returns
    PJRT_HOST_BUFFER_COPY = 0,
    /// Data must remain valid until transfer completes (async)
    PJRT_HOST_BUFFER_IMMUTABLE_UNTIL_TRANSFER = 1,
    /// Zero-copy: data must remain valid until execution completes
    PJRT_HOST_BUFFER_ZERO_COPY = 2,
} PJRTDirect_HostBufferSemantics;

/// Execute with float32 inputs/outputs (most common case)
/// This is a convenience function for the typical ML workload.
///
/// @param exec         Compiled executable
/// @param device       Device to run on
/// @param inputs       Array of input float arrays (host memory)
/// @param input_sizes  Number of float elements in each input
/// @param num_inputs   Number of inputs
/// @param outputs      Pre-allocated output float arrays (host memory)
/// @param output_sizes Number of float elements in each output
/// @param num_outputs  Number of outputs
/// @param semantics    Host buffer semantics for inputs
/// @param out_timing   Optional: timing breakdown (NULL to skip)
/// @return Status
PJRTDirect_Status PJRTDirect_ExecuteF32(
    PJRTDirect_Executable exec,
    PJRTDirect_Device device,
    const float* const* inputs,
    const int64_t* input_sizes,
    size_t num_inputs,
    float** outputs,
    const int64_t* output_sizes,
    size_t num_outputs,
    PJRTDirect_HostBufferSemantics semantics,
    PJRTDirect_Timing* out_timing
);

/// Execute with generic data types
/// For non-float32 cases (int32, float64, etc.)
PJRTDirect_Status PJRTDirect_Execute(
    PJRTDirect_Executable exec,
    PJRTDirect_Device device,
    const void* const* inputs,
    const int64_t* input_sizes,
    const PJRTDirect_DType* input_dtypes,
    size_t num_inputs,
    void** outputs,
    const int64_t* output_sizes,
    const PJRTDirect_DType* output_dtypes,
    size_t num_outputs,
    PJRTDirect_HostBufferSemantics semantics,
    PJRTDirect_Timing* out_timing
);

//===------------------------------------------------------------------===//
// Execution Context - Optimized for Repeated Execution
//===------------------------------------------------------------------===//

/// Create an execution context for repeated execution with the same shapes.
/// This pre-caches metadata and can optionally pre-allocate device buffers.
///
/// Using an execution context provides the fastest execution path for
/// repeated calls with the same input/output shapes.
///
/// @param exec          Compiled executable
/// @param device        Device to run on
/// @param input_sizes   Number of elements in each input
/// @param num_inputs    Number of inputs
/// @param output_sizes  Number of elements in each output
/// @param num_outputs   Number of outputs
/// @param out_ctx       Output: execution context handle
/// @return Status
PJRTDirect_Status PJRTDirect_CreateExecContext(
    PJRTDirect_Executable exec,
    PJRTDirect_Device device,
    const int64_t* input_sizes,
    size_t num_inputs,
    const int64_t* output_sizes,
    size_t num_outputs,
    PJRTDirect_ExecContext* out_ctx
);

/// Execute using a pre-created context (fastest path)
/// Input and output arrays must match the sizes specified when creating the context.
///
/// @param ctx          Execution context
/// @param inputs       Array of input float arrays
/// @param outputs      Pre-allocated output float arrays
/// @param out_timing   Optional: timing breakdown (NULL to skip)
/// @return Status
PJRTDirect_Status PJRTDirect_ExecuteWithContext(
    PJRTDirect_ExecContext ctx,
    const float* const* inputs,
    float** outputs,
    PJRTDirect_Timing* out_timing
);

/// Destroy an execution context
void PJRTDirect_DestroyExecContext(PJRTDirect_ExecContext ctx);

//===------------------------------------------------------------------===//
// Utility Functions
//===------------------------------------------------------------------===//

/// Get library version string
const char* PJRTDirect_GetVersion(void);

/// Check if a backend is available (plugin exists and can be loaded)
bool PJRTDirect_IsBackendAvailable(PJRTDirect_Backend backend);

#ifdef __cplusplus
}
#endif

#endif // PJRT_DIRECT_H
