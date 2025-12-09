//===-- PJRTSimpleWrapper.h - Simplified PJRT C Wrapper ------*- C -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// Simplified C wrapper for PJRT C API to ease Swift interop
//
//===------------------------------------------------------------------===//

#ifndef PJRT_SIMPLE_WRAPPER_H
#define PJRT_SIMPLE_WRAPPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

//===------------------------------------------------------------------===//
// Opaque Type Declarations for Swift interop
//===------------------------------------------------------------------===//

// Use void* for all opaque types to ensure Swift compatibility
// Swift handles void* perfectly and converts it to OpaquePointer

#ifdef __cplusplus
extern "C" {
#endif

//===------------------------------------------------------------------===//
// Simplified Type Aliases
//===------------------------------------------------------------------===//

// We use int32_t for error codes and buffer types to avoid enum conflicts
// Swift will treat these as Int32
typedef enum {
    SW_PJRT_Error_OK = 0,
    SW_PJRT_Error_CANCELLED = 1,
    SW_PJRT_Error_UNKNOWN = 2,
    SW_PJRT_Error_INVALID_ARGUMENT = 3,
    SW_PJRT_Error_DEADLINE_EXCEEDED = 4,
    SW_PJRT_Error_NOT_FOUND = 5,
    SW_PJRT_Error_ALREADY_EXISTS = 6,
    SW_PJRT_Error_PERMISSION_DENIED = 7,
    SW_PJRT_Error_RESOURCE_EXHAUSTED = 8,
    SW_PJRT_Error_FAILED_PRECONDITION = 9,
    SW_PJRT_Error_ABORTED = 10,
    SW_PJRT_Error_OUT_OF_RANGE = 11,
    SW_PJRT_Error_UNIMPLEMENTED = 12,
    SW_PJRT_Error_INTERNAL = 13,
    SW_PJRT_Error_UNAVAILABLE = 14,
    SW_PJRT_Error_DATA_LOSS = 15,
    SW_PJRT_Error_UNAUTHENTICATED = 16,
} SW_PJRT_Error_Code;

typedef enum {
    SW_PJRT_Buffer_Type_PRED = 0,
    SW_PJRT_Buffer_Type_S8 = 1,
    SW_PJRT_Buffer_Type_S16 = 2,
    SW_PJRT_Buffer_Type_S32 = 3,
    SW_PJRT_Buffer_Type_S64 = 4,
    SW_PJRT_Buffer_Type_U8 = 5,
    SW_PJRT_Buffer_Type_U16 = 6,
    SW_PJRT_Buffer_Type_U32 = 7,
    SW_PJRT_Buffer_Type_U64 = 8,
    SW_PJRT_Buffer_Type_F16 = 9,
    SW_PJRT_Buffer_Type_F32 = 10,
    SW_PJRT_Buffer_Type_F64 = 11,
    SW_PJRT_Buffer_Type_BF16 = 12,
    SW_PJRT_Buffer_Type_C64 = 13,
    SW_PJRT_Buffer_Type_C128 = 14,
} SW_PJRT_Buffer_Type;

//===------------------------------------------------------------------===//
// Plugin Loading
//===------------------------------------------------------------------===//

/// Load a PJRT plugin from the specified path
/// Returns SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_LoadPlugin(const char* plugin_path);

/// Unload the currently loaded plugin
void PJRT_UnloadPlugin(void);

/// Get the last dlopen/dlsym error message
const char* PJRT_GetLastError(void);

//===------------------------------------------------------------------===//
// Error Handling
//===------------------------------------------------------------------===//

/// Get error code from PJRT_Error
SW_PJRT_Error_Code PJRT_GetErrorCode(void* error);

/// Get error message from PJRT_Error
const char* PJRT_GetErrorMessage(void* error);

/// Destroy a PJRT_Error
void PJRT_DestroyError(void* error);

//===------------------------------------------------------------------===//
// Client Management
//===------------------------------------------------------------------===//

/// Create a PJRT client
/// Returns SW_PJRT_Error_OK on success, out_client contains the client
SW_PJRT_Error_Code PJRT_CreateClient(void** out_client);

/// Destroy a PJRT client
void PJRT_DestroyClient(void* client);

/// Get platform name from client
SW_PJRT_Error_Code PJRT_GetPlatformName(void* client, const char** out_name);

//===------------------------------------------------------------------===//
// Device Enumeration
//===------------------------------------------------------------------===//

/// Get addressable devices from client
/// Returns SW_PJRT_Error_OK on success
/// out_devices points to an array of void* (device pointers)
/// out_num_devices contains the number of devices
SW_PJRT_Error_Code PJRT_GetAddressableDevices(
    void* client,
    void*** out_devices,
    size_t* out_num_devices
);

/// Get device ID
SW_PJRT_Error_Code PJRT_GetDeviceId(void* device, int32_t* out_id);

/// Get device kind (e.g., "CPU", "GPU")
SW_PJRT_Error_Code PJRT_GetDeviceKind(void* device, const char** out_kind);

//===------------------------------------------------------------------===//
// Buffer Management
//===------------------------------------------------------------------===//

/// Host buffer semantics for buffer creation
typedef enum {
    /// Data is copied during the call, buffer can be modified after return
    SW_PJRT_HostBuffer_ImmutableOnlyDuringCall = 0,
    /// Data must remain valid until transfer completes (use with kZeroCopy)
    SW_PJRT_HostBuffer_ImmutableUntilTransferCompletes = 1,
    /// Zero-copy: host memory used directly, must remain valid for buffer lifetime
    SW_PJRT_HostBuffer_ZeroCopy = 2,
    /// Mutable zero-copy: for buffer donation - memory can be reused for output
    SW_PJRT_HostBuffer_MutableZeroCopy = 3,
} SW_PJRT_HostBufferSemantics;

/// Create a buffer from host data
SW_PJRT_Error_Code PJRT_CreateBuffer(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    void** out_buffer
);

/// Create a buffer from host data with specified semantics
/// Use zero_copy=true when host buffer will remain valid for the lifetime of the PJRT buffer
SW_PJRT_Error_Code PJRT_CreateBufferWithSemantics(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffer
);

/// Fast buffer creation using cached descriptor
/// When is_same_shape is true, reuses cached args structure and only updates data pointer
/// This avoids memset and reduces per-buffer overhead for repeated same-shape inputs
/// cached_slot: 0-15, used to cache args for up to 16 different input slots
SW_PJRT_Error_Code PJRT_CreateBufferFast(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    SW_PJRT_HostBufferSemantics semantics,
    int cached_slot,
    bool is_same_shape,
    void** out_buffer
);

/// Batched buffer creation - creates multiple buffers in a single call
/// This reduces FFI overhead by batching multiple buffer creations together
/// All buffers are created with the same semantics on the same device
/// Returns SW_PJRT_Error_OK on success, error code otherwise
/// On success, out_buffers[0..num_buffers-1] will contain valid buffer handles
SW_PJRT_Error_Code PJRT_CreateBuffersBatched(
    void* client,
    size_t num_buffers,
    void* const* data_ptrs,          // Array of data pointers (mutable for Swift interop)
    const SW_PJRT_Buffer_Type* types, // Array of element types
    const int64_t* const* dims_ptrs, // Array of dimension pointers
    const size_t* num_dims_array,    // Array of dimension counts
    void* device,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffers               // Output: array of buffer handles
);

/// Destroy a buffer
void PJRT_DestroyBuffer(void* buffer);

/// Transfer buffer data from device to host (synchronous - waits for completion)
/// out_data should point to a buffer large enough to hold the data
SW_PJRT_Error_Code PJRT_BufferToHost(
    void* buffer,
    void* out_data,
    size_t data_size
);

/// Transfer buffer data from device to host (asynchronous - returns immediately)
/// out_data should point to a buffer large enough to hold the data
/// out_event receives an event handle that must be awaited before accessing out_data
/// Use PJRT_EventAwaitAndDestroy() to wait for completion
SW_PJRT_Error_Code PJRT_BufferToHostAsync(
    void* buffer,
    void* out_data,
    size_t data_size,
    void** out_event
);

/// Wait for an event to complete
SW_PJRT_Error_Code PJRT_EventAwait(void* event);

/// Destroy an event (must be called after awaiting)
void PJRT_EventDestroy(void* event);

/// Wait for an event to complete and destroy it
/// Convenience function combining PJRT_EventAwait and PJRT_EventDestroy
SW_PJRT_Error_Code PJRT_EventAwaitAndDestroy(void* event);

/// Get buffer dimensions (shape)
/// out_dims will point to an array owned by the buffer (do not free)
/// out_num_dims will contain the number of dimensions
SW_PJRT_Error_Code PJRT_GetBufferDimensions(
    void* buffer,
    const int64_t** out_dims,
    size_t* out_num_dims
);

/// Get buffer size in bytes on device
SW_PJRT_Error_Code PJRT_GetBufferOnDeviceSizeInBytes(
    void* buffer,
    size_t* out_size
);

/// Increase external reference count for a buffer
/// Must be called when taking ownership of a buffer from PJRT
SW_PJRT_Error_Code PJRT_Buffer_IncRefCount(void* buffer);

/// Decrease external reference count for a buffer
/// Must be called before destroying a buffer
SW_PJRT_Error_Code PJRT_Buffer_DecRefCount(void* buffer);

//===------------------------------------------------------------------===//
// Compilation & Execution
//===------------------------------------------------------------------===//

/// Compile an MLIR module to a loaded executable
SW_PJRT_Error_Code PJRT_CompileWrapper(
    void* client,
    const char* mlir_module,
    void** out_executable
);

/// XLA backend optimization level
typedef enum {
    SW_XLA_OPT_DEFAULT = -1,  // Use XLA's default optimization
    SW_XLA_OPT_O0 = 0,        // No optimization (fastest compile)
    SW_XLA_OPT_O1 = 1,        // Basic optimization
    SW_XLA_OPT_O2 = 2,        // Standard optimization (recommended, GPU requires >= 2)
    SW_XLA_OPT_O3 = 3,        // Maximum optimization (slowest compile, best execution)
} SW_XLA_OptLevel;

/// Compile an MLIR module with specified XLA optimization level
SW_PJRT_Error_Code PJRT_CompileWrapperWithOptLevel(
    void* client,
    const char* mlir_module,
    SW_XLA_OptLevel xla_opt_level,
    void** out_executable
);

/// Destroy a loaded executable
void PJRT_DestroyExecutable(void* executable);

/// Execute a loaded executable with input buffers
SW_PJRT_Error_Code PJRT_ExecuteWrapper(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void*** out_outputs,
    size_t* out_num_outputs
);

/// Execute with buffer donation support
/// non_donatable_indices: array of input indices that should NOT be donated (NULL = donate all eligible)
/// num_non_donatable: number of indices in non_donatable_indices array
/// When an input is donated, its buffer memory may be reused for an output with matching shape.
/// After execution, donated input buffers become invalid and must not be used.
SW_PJRT_Error_Code PJRT_ExecuteWithDonation(
    void* executable,
    void** inputs,
    size_t num_inputs,
    const int64_t* non_donatable_indices,
    size_t num_non_donatable,
    void*** out_outputs,
    size_t* out_num_outputs
);

//===------------------------------------------------------------------===//
// Fast/Optimized APIs (reduced FFI overhead)
//===------------------------------------------------------------------===//

/// Fast async D2H transfer using designated initializers (avoids memset)
SW_PJRT_Error_Code PJRT_BufferToHostAsyncFast(
    void* buffer,
    void* out_data,
    size_t data_size,
    void** out_event
);

/// Fast event destroy using designated initializers
void PJRT_EventDestroyFast(void* event);

/// Fast combined await and destroy using designated initializers
SW_PJRT_Error_Code PJRT_EventAwaitAndDestroyFast(void* event);

/// Execute and transfer outputs to host in a single call
/// This reduces FFI overhead by combining execute + async D2H + await into one C call.
/// out_data_ptrs: Pre-allocated host memory for each output
/// out_data_sizes: Size of each output buffer in bytes
/// num_outputs: Number of outputs to transfer (max 4)
/// out_actual_outputs: Actual number of outputs transferred
SW_PJRT_Error_Code PJRT_ExecuteAndTransfer(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void** out_data_ptrs,
    size_t* out_data_sizes,
    size_t num_outputs,
    size_t* out_actual_outputs
);

/// Hot path: Full H2D + Execute + D2H in a single FFI call
/// This is the most optimized path for repeated execution with the same shapes.
/// Combines buffer creation, execution, and transfer all in one C call.
///
/// Parameters:
///   client: PJRT client handle
///   executable: PJRT executable handle
///   device: PJRT device handle
///   input_data: Array of pointers to input data (raw float arrays on host)
///   input_sizes: Number of elements (NOT bytes) in each input
///   num_inputs: Number of inputs
///   output_data: Array of pointers to output buffers (pre-allocated on host)
///   output_sizes: Number of elements (NOT bytes) in each output
///   num_outputs: Number of outputs
///   semantics: Host buffer semantics (zeroCopy recommended for CPU)
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecuteHotPath(
    void* client,
    void* executable,
    void* device,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_HostBufferSemantics semantics
);

//===------------------------------------------------------------------===//
// Profiler Extension API
//===------------------------------------------------------------------===//

/// Check if the loaded plugin has a profiler extension
/// Returns true if profiler extension is available
bool PJRT_HasProfilerExtension(void);

/// Get the profiler API from the extension (if available)
/// Returns NULL if no profiler extension
void* PJRT_GetProfilerApi(void);

/// Create a profiler using the PJRT profiler extension
/// options is a serialized ProfileOptions protobuf (can be NULL for defaults)
/// Returns SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ProfilerCreate(
    const char* options,
    size_t options_size,
    void** out_profiler
);

/// Start profiling
SW_PJRT_Error_Code PJRT_ProfilerStart(void* profiler);

/// Stop profiling
SW_PJRT_Error_Code PJRT_ProfilerStop(void* profiler);

/// Collect profiler data (XSpace protobuf)
/// First call with buffer=NULL to get size, then call again with allocated buffer
SW_PJRT_Error_Code PJRT_ProfilerCollectData(
    void* profiler,
    uint8_t* buffer,
    size_t* buffer_size
);

/// Destroy a profiler
SW_PJRT_Error_Code PJRT_ProfilerDestroy(void* profiler);

//===------------------------------------------------------------------===//
// Execution Timing/Profiling API
//===------------------------------------------------------------------===//

/// Timing breakdown for PJRT execution operations
/// All times are in nanoseconds for maximum precision
typedef struct {
    /// Time spent creating input buffers (H2D transfer initiation)
    uint64_t h2d_create_ns;
    /// Time spent executing the kernel
    uint64_t execute_ns;
    /// Time spent initiating D2H transfers (async start)
    uint64_t d2h_initiate_ns;
    /// Time spent awaiting D2H completion (sync)
    uint64_t d2h_await_ns;
    /// Time spent destroying buffers (cleanup)
    uint64_t buffer_destroy_ns;
    /// Total end-to-end time
    uint64_t total_ns;
    /// Number of input buffers
    size_t num_inputs;
    /// Number of output buffers
    size_t num_outputs;
} SW_PJRT_ExecutionTiming;

/// Execute with detailed timing breakdown
/// This function profiles each phase of execution and returns timing data.
/// Use this to identify bottlenecks in the PJRT execution path.
///
/// Parameters:
///   executable: PJRT executable handle
///   inputs: Array of PJRT buffer handles (already on device)
///   num_inputs: Number of inputs
///   out_outputs: Pointer to receive output buffer array
///   out_num_outputs: Pointer to receive number of outputs
///   out_timing: Pointer to receive timing breakdown
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecuteWithTiming(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void*** out_outputs,
    size_t* out_num_outputs,
    SW_PJRT_ExecutionTiming* out_timing
);

/// Full profiled execution: H2D + Execute + D2H with timing breakdown
/// This profiles the complete execution path including buffer creation.
///
/// Parameters:
///   client: PJRT client handle
///   executable: PJRT executable handle
///   device: PJRT device handle
///   input_data: Array of pointers to input data (raw float arrays on host)
///   input_sizes: Number of elements in each input
///   num_inputs: Number of inputs
///   output_data: Array of pointers to output buffers (pre-allocated on host)
///   output_sizes: Number of elements in each output
///   num_outputs: Number of outputs
///   semantics: Host buffer semantics
///   out_timing: Pointer to receive timing breakdown
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecuteProfiled(
    void* client,
    void* executable,
    void* device,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_HostBufferSemantics semantics,
    SW_PJRT_ExecutionTiming* out_timing
);

//===------------------------------------------------------------------===//
// Async Execution API (callback-based, non-blocking)
//===------------------------------------------------------------------===//

/// Callback type for async execution completion
/// Parameters:
///   error: Error code (SW_PJRT_Error_OK on success)
///   user_data: User-provided context pointer
typedef void (*SW_PJRT_AsyncCallback)(SW_PJRT_Error_Code error, void* user_data);

/// Async execution context - holds state between start and completion
typedef struct {
    void* client;           // PJRT client handle
    void* executable;       // PJRT executable handle
    void* device;           // PJRT device handle
    void** output_buffers;  // Array of output PJRT_Buffer handles
    size_t num_outputs;     // Number of outputs
    void* const* output_data;     // Host destination arrays
    const size_t* output_sizes;   // Sizes of each output
    void** d2h_events;      // Array of D2H transfer events
    size_t pending_count;   // Number of pending D2H transfers
    SW_PJRT_AsyncCallback callback;  // User callback
    void* user_data;        // User context
    SW_PJRT_Error_Code error;  // Accumulated error status
} SW_PJRT_AsyncContext;

/// Execute asynchronously with callback on completion
/// This initiates execution and D2H transfers, then returns immediately.
/// The callback is invoked when all outputs are ready.
///
/// Parameters:
///   client: PJRT client handle
///   executable: PJRT executable handle
///   device: PJRT device handle
///   input_data: Array of pointers to input data (raw float arrays on host)
///   input_sizes: Number of elements in each input
///   num_inputs: Number of inputs
///   output_data: Array of pointers to output buffers (pre-allocated on host)
///   output_sizes: Number of elements in each output
///   num_outputs: Number of outputs
///   semantics: Host buffer semantics
///   callback: Callback function to invoke on completion
///   user_data: User context passed to callback
/// Returns: SW_PJRT_Error_OK if execution was initiated, error code otherwise
SW_PJRT_Error_Code PJRT_ExecuteAsync(
    void* client,
    void* executable,
    void* device,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_HostBufferSemantics semantics,
    SW_PJRT_AsyncCallback callback,
    void* user_data
);

/// Execute with OnReady callbacks instead of blocking await
/// This uses PJRT_Event_OnReady for D2H completion instead of PJRT_Event_Await.
/// Returns timing breakdown for comparison with blocking version.
SW_PJRT_Error_Code PJRT_ExecuteWithCallbacks(
    void* client,
    void* executable,
    void* device,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_HostBufferSemantics semantics,
    SW_PJRT_ExecutionTiming* out_timing
);

//===------------------------------------------------------------------===//
// Buffer Pool API (for reduced allocation overhead)
//===------------------------------------------------------------------===//

/// Maximum number of pooled buffer slots per executable
#define PJRT_BUFFER_POOL_SIZE 8

/// Buffer pool entry - caches a PJRT buffer for reuse
typedef struct {
    void* buffer;           // PJRT_Buffer handle
    size_t size_elements;   // Number of elements
    SW_PJRT_Buffer_Type type;  // Buffer element type
    bool in_use;            // Whether buffer is currently in use
} SW_PJRT_PooledBuffer;

/// Buffer pool for an executable - caches input and output buffers
typedef struct {
    void* client;           // PJRT client handle
    void* device;           // PJRT device handle
    void* executable;       // PJRT executable handle
    SW_PJRT_PooledBuffer inputs[PJRT_BUFFER_POOL_SIZE];
    size_t num_inputs;
    bool initialized;
} SW_PJRT_BufferPool;

/// Create a buffer pool for an executable
/// This pre-allocates buffers for the given input sizes.
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_BufferPoolCreate(
    void* client,
    void* device,
    void* executable,
    const size_t* input_sizes,
    size_t num_inputs,
    SW_PJRT_BufferPool* out_pool
);

/// Execute using pooled buffers
/// This reuses pre-allocated buffers, avoiding allocation overhead.
/// Input data is copied to pooled buffers before execution.
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecutePooled(
    SW_PJRT_BufferPool* pool,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_ExecutionTiming* out_timing
);

/// Destroy a buffer pool and release all pooled buffers
void PJRT_BufferPoolDestroy(SW_PJRT_BufferPool* pool);

//===------------------------------------------------------------------===//
// Enhanced Buffer Pool V2 API (persistent device buffers)
//===------------------------------------------------------------------===//

/// Maximum slots for V2 buffer pool
#define PJRT_BUFFER_POOL_V2_MAX_INPUTS 8
#define PJRT_BUFFER_POOL_V2_MAX_OUTPUTS 4

/// Enhanced buffer pool with persistent device buffers
/// Unlike SW_PJRT_BufferPool, this keeps output buffers alive between executions
/// and uses a double-buffering scheme for inputs to enable overlapped transfers.
typedef struct {
    void* client;           // PJRT client handle
    void* device;           // PJRT device handle
    void* executable;       // PJRT executable handle

    // Input configuration
    size_t input_sizes[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
    size_t num_inputs;

    // Output configuration
    size_t output_sizes[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS];
    size_t num_outputs;

    // Cached H2D args structures (avoid memset per call)
    // These are initialized once and only data pointer is updated
    void* cached_h2d_args[PJRT_BUFFER_POOL_V2_MAX_INPUTS];

    // Cached dimension arrays
    int64_t input_dims[PJRT_BUFFER_POOL_V2_MAX_INPUTS];

    // Execution counter for statistics
    uint64_t execution_count;

    bool initialized;
} SW_PJRT_BufferPoolV2;

/// Create an enhanced buffer pool V2
/// This caches metadata for faster repeated execution.
///
/// Parameters:
///   client: PJRT client handle
///   device: PJRT device handle
///   executable: PJRT executable handle
///   input_sizes: Array of input sizes (number of F32 elements)
///   num_inputs: Number of inputs (max 8)
///   output_sizes: Array of output sizes (number of F32 elements)
///   num_outputs: Number of outputs (max 4)
///   out_pool: Output pool structure
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_BufferPoolV2Create(
    void* client,
    void* device,
    void* executable,
    const size_t* input_sizes,
    size_t num_inputs,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_BufferPoolV2* out_pool
);

/// Execute using V2 buffer pool with optimized path
/// This minimizes per-call overhead by reusing cached structures.
///
/// Parameters:
///   pool: Buffer pool handle
///   input_data: Array of pointers to input data (F32 arrays)
///   output_data: Array of pointers to output buffers (pre-allocated)
///   out_timing: Optional timing breakdown (can be NULL)
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecutePooledV2(
    SW_PJRT_BufferPoolV2* pool,
    const void* const* input_data,
    void* const* output_data,
    SW_PJRT_ExecutionTiming* out_timing
);

/// Destroy V2 buffer pool
void PJRT_BufferPoolV2Destroy(SW_PJRT_BufferPoolV2* pool);

//===------------------------------------------------------------------===//
// Batched Transfer API (reduced FFI overhead)
//===------------------------------------------------------------------===//

/// Maximum number of buffers in a batched transfer
#define PJRT_BATCH_TRANSFER_MAX 8

/// Batched H2D transfer - creates multiple buffers in parallel
/// This reduces per-buffer overhead by batching the PJRT calls.
///
/// Parameters:
///   client: PJRT client handle
///   device: PJRT device handle
///   data_ptrs: Array of host data pointers
///   sizes: Array of sizes (number of F32 elements)
///   num_buffers: Number of buffers to create (max 8)
///   semantics: Host buffer semantics to use
///   out_buffers: Output array of buffer handles
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_BatchedH2DTransfer(
    void* client,
    void* device,
    const void* const* data_ptrs,
    const size_t* sizes,
    size_t num_buffers,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffers
);

/// Batched D2H transfer - transfers multiple buffers to host
/// This initiates all transfers before awaiting any, enabling overlap.
///
/// Parameters:
///   buffers: Array of PJRT buffer handles
///   dest_ptrs: Array of destination host pointers
///   sizes: Array of sizes in bytes
///   num_buffers: Number of buffers to transfer (max 8)
///   destroy_buffers: If true, destroy source buffers after transfer
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_BatchedD2HTransfer(
    void** buffers,
    void** dest_ptrs,
    const size_t* sizes,
    size_t num_buffers,
    bool destroy_buffers
);

/// Ultra-fast execution path combining all optimizations
/// This is the fastest path for repeated same-shape executions.
///
/// Features:
/// - Cached H2D args (only update data pointers)
/// - Batched H2D buffer creation
/// - Batched D2H transfers with overlap
/// - Minimal FFI boundary crossings
///
/// Parameters:
///   pool: V2 buffer pool with cached configuration
///   input_data: Array of input data pointers
///   output_data: Array of output buffer pointers (pre-allocated)
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_ExecuteUltraFast(
    SW_PJRT_BufferPoolV2* pool,
    const void* const* input_data,
    void* const* output_data
);

//===------------------------------------------------------------------===//
// Pipelined Execution API (overlapped compute and transfer)
//===------------------------------------------------------------------===//

/// Maximum pipeline depth (number of in-flight executions)
#define PJRT_PIPELINE_MAX_DEPTH 4

/// Pipeline slot state
typedef enum {
    SW_PJRT_PipelineSlot_Empty = 0,      // Slot is available
    SW_PJRT_PipelineSlot_Executing = 1,  // H2D + Execute done, D2H in progress
    SW_PJRT_PipelineSlot_Ready = 2       // D2H complete, data ready to consume
} SW_PJRT_PipelineSlotState;

/// Pipeline slot - holds state for one in-flight execution
typedef struct {
    SW_PJRT_PipelineSlotState state;
    void* output_buffers[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS];  // PJRT output buffers
    void* d2h_events[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS];      // D2H events
    void* user_output_ptrs[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS]; // User's output arrays
    uint64_t submit_time_ns;  // For latency tracking
} SW_PJRT_PipelineSlot;

/// Execution pipeline - enables overlapped execution
///
/// The pipeline allows multiple executions to be "in flight":
/// - Slot N: D2H transfer in progress
/// - Slot N+1: Currently executing
/// - Slot N+2: Ready for next submission
///
/// This hides D2H latency by overlapping it with the next execution.
typedef struct {
    SW_PJRT_BufferPoolV2* pool;  // Underlying buffer pool

    SW_PJRT_PipelineSlot slots[PJRT_PIPELINE_MAX_DEPTH];
    size_t depth;           // Active pipeline depth (1-4)
    size_t head;            // Next slot to submit to
    size_t tail;            // Next slot to complete from
    size_t in_flight;       // Number of slots currently in use

    // Statistics
    uint64_t total_submissions;
    uint64_t total_completions;
    uint64_t total_wait_ns;     // Time spent waiting for slots

    bool initialized;
} SW_PJRT_ExecutionPipeline;

/// Create an execution pipeline
///
/// Parameters:
///   pool: Buffer pool V2 to use (must remain valid for pipeline lifetime)
///   depth: Pipeline depth (1-4, higher = more overlap but more memory)
///   out_pipeline: Output pipeline structure
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_PipelineCreate(
    SW_PJRT_BufferPoolV2* pool,
    size_t depth,
    SW_PJRT_ExecutionPipeline* out_pipeline
);

/// Submit an execution to the pipeline (non-blocking if slots available)
///
/// This performs H2D + Execute + initiates D2H, then returns immediately.
/// The D2H transfer completes asynchronously.
///
/// If all pipeline slots are full, this blocks until one completes.
///
/// Parameters:
///   pipeline: Pipeline handle
///   input_data: Array of input data pointers
///   output_data: Array of output buffer pointers (will be filled when complete)
/// Returns: SW_PJRT_Error_OK on success, or error code
SW_PJRT_Error_Code PJRT_PipelineSubmit(
    SW_PJRT_ExecutionPipeline* pipeline,
    const void* const* input_data,
    void* const* output_data
);

/// Wait for the oldest in-flight execution to complete
///
/// Returns the output_data pointer that was passed to the corresponding submit.
///
/// Parameters:
///   pipeline: Pipeline handle
///   out_output_data: Receives pointer to output data array (as passed to submit)
/// Returns: SW_PJRT_Error_OK on success, SW_PJRT_Error_NOT_FOUND if pipeline empty
SW_PJRT_Error_Code PJRT_PipelineAwaitOne(
    SW_PJRT_ExecutionPipeline* pipeline,
    void*** out_output_data
);

/// Flush the pipeline - wait for all in-flight executions to complete
///
/// Parameters:
///   pipeline: Pipeline handle
/// Returns: SW_PJRT_Error_OK on success
SW_PJRT_Error_Code PJRT_PipelineFlush(
    SW_PJRT_ExecutionPipeline* pipeline
);

/// Get pipeline statistics
///
/// Parameters:
///   pipeline: Pipeline handle
///   out_submissions: Total submissions
///   out_completions: Total completions
///   out_avg_wait_us: Average wait time in microseconds
void PJRT_PipelineGetStats(
    SW_PJRT_ExecutionPipeline* pipeline,
    uint64_t* out_submissions,
    uint64_t* out_completions,
    double* out_avg_wait_us
);

/// Destroy the pipeline (flushes first)
void PJRT_PipelineDestroy(SW_PJRT_ExecutionPipeline* pipeline);

//===------------------------------------------------------------------===//
// TraceMe API (for unified Swift/XLA tracing)
//===------------------------------------------------------------------===//

/// Check if the plugin has TraceMe API available
/// Returns true if TraceMe functions are available
bool PJRT_HasTraceMeApi(void);

/// Start a TraceMe activity. Returns activity_id (or 0 if tracing disabled).
/// The activity_id is used to pair start/end events in the trace.
/// level: 1 = critical (always shown), 2 = info, 3 = verbose
int64_t PJRT_TraceMeStart(const char* name, int32_t level);

/// End a TraceMe activity started by PJRT_TraceMeStart.
void PJRT_TraceMeStop(int64_t activity_id);

/// Check if tracing is active at the given level.
/// Useful to avoid expensive string operations when tracing is disabled.
bool PJRT_TraceMeActive(int32_t level);

/// Record an instant event (no duration, just a point in time).
void PJRT_TraceMeInstant(const char* name, int32_t level);

#ifdef __cplusplus
}
#endif

#endif // PJRT_SIMPLE_WRAPPER_H
