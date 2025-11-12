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

/// Destroy a buffer
void PJRT_DestroyBuffer(void* buffer);

/// Transfer buffer data from device to host
/// out_data should point to a buffer large enough to hold the data
SW_PJRT_Error_Code PJRT_BufferToHost(
    void* buffer,
    void* out_data,
    size_t data_size
);

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

#ifdef __cplusplus
}
#endif

#endif // PJRT_SIMPLE_WRAPPER_H
