//===-- PJRTSimpleWrapper.h - Simplified PJRT C Wrapper ------*- C -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// Simplified C wrapper for PJRT C API to ease Swift interop
//
//===------------------------------------------------------------------===//

#ifndef PJRT_SIMPLE_WRAPPER_H
#define PJRT_SIMPLE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Include the full PJRT C API (for internal use)
#include "pjrt_c_api.h"

//===------------------------------------------------------------------===//
// Simplified Error Codes
//===------------------------------------------------------------------===//

typedef enum {
    PJRT_Error_Code_OK = 0,
    PJRT_Error_Code_CANCELLED = 1,
    PJRT_Error_Code_UNKNOWN = 2,
    PJRT_Error_Code_INVALID_ARGUMENT = 3,
    PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
    PJRT_Error_Code_NOT_FOUND = 5,
    PJRT_Error_Code_ALREADY_EXISTS = 6,
    PJRT_Error_Code_PERMISSION_DENIED = 7,
    PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
    PJRT_Error_Code_FAILED_PRECONDITION = 9,
    PJRT_Error_Code_ABORTED = 10,
    PJRT_Error_Code_OUT_OF_RANGE = 11,
    PJRT_Error_Code_UNIMPLEMENTED = 12,
    PJRT_Error_Code_INTERNAL = 13,
    PJRT_Error_Code_UNAVAILABLE = 14,
    PJRT_Error_Code_DATA_LOSS = 15,
    PJRT_Error_Code_UNAUTHENTICATED = 16,
} PJRT_Error_Code;

//===------------------------------------------------------------------===//
// Simplified Buffer Types
//===------------------------------------------------------------------===//

typedef enum {
    PJRT_Buffer_Type_PRED = 0,
    PJRT_Buffer_Type_S8 = 1,
    PJRT_Buffer_Type_S16 = 2,
    PJRT_Buffer_Type_S32 = 3,
    PJRT_Buffer_Type_S64 = 4,
    PJRT_Buffer_Type_U8 = 5,
    PJRT_Buffer_Type_U16 = 6,
    PJRT_Buffer_Type_U32 = 7,
    PJRT_Buffer_Type_U64 = 8,
    PJRT_Buffer_Type_F16 = 9,
    PJRT_Buffer_Type_F32 = 10,
    PJRT_Buffer_Type_F64 = 11,
    PJRT_Buffer_Type_BF16 = 12,
    PJRT_Buffer_Type_C64 = 13,
    PJRT_Buffer_Type_C128 = 14,
} PJRT_Buffer_Type;

//===------------------------------------------------------------------===//
// Plugin Loading
//===------------------------------------------------------------------===//

/// Load a PJRT plugin from the specified path
/// Returns PJRT_Error_Code_OK on success
PJRT_Error_Code PJRT_LoadPlugin(const char* plugin_path);

/// Unload the currently loaded plugin
void PJRT_UnloadPlugin(void);

/// Get the last dlopen/dlsym error message
const char* PJRT_GetLastError(void);

//===------------------------------------------------------------------===//
// Error Handling
//===------------------------------------------------------------------===//

/// Get error code from PJRT_Error
PJRT_Error_Code PJRT_GetErrorCode(PJRT_Error* error);

/// Get error message from PJRT_Error
const char* PJRT_GetErrorMessage(PJRT_Error* error);

/// Destroy a PJRT_Error
void PJRT_DestroyError(PJRT_Error* error);

//===------------------------------------------------------------------===//
// Client Management
//===------------------------------------------------------------------===//

/// Create a PJRT client
/// Returns PJRT_Error_Code_OK on success, out_client contains the client
PJRT_Error_Code PJRT_CreateClient(PJRT_Client** out_client);

/// Destroy a PJRT client
void PJRT_DestroyClient(PJRT_Client* client);

/// Get platform name from client
PJRT_Error_Code PJRT_GetPlatformName(PJRT_Client* client, const char** out_name);

//===------------------------------------------------------------------===//
// Device Enumeration
//===------------------------------------------------------------------===//

/// Get addressable devices from client
/// Returns PJRT_Error_Code_OK on success
/// out_devices points to an array of PJRT_Device*
/// out_num_devices contains the number of devices
PJRT_Error_Code PJRT_GetAddressableDevices(
    PJRT_Client* client,
    PJRT_Device*** out_devices,
    size_t* out_num_devices
);

/// Get device ID
PJRT_Error_Code PJRT_GetDeviceId(PJRT_Device* device, int32_t* out_id);

/// Get device kind (e.g., "CPU", "GPU")
PJRT_Error_Code PJRT_GetDeviceKind(PJRT_Device* device, const char** out_kind);

//===------------------------------------------------------------------===//
// Buffer Management
//===------------------------------------------------------------------===//

/// Create a buffer from host data
PJRT_Error_Code PJRT_CreateBuffer(
    PJRT_Client* client,
    const void* data,
    PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    PJRT_Device* device,
    PJRT_Buffer** out_buffer
);

/// Destroy a buffer
void PJRT_DestroyBuffer(PJRT_Buffer* buffer);

//===------------------------------------------------------------------===//
// Compilation & Execution
//===------------------------------------------------------------------===//

/// Compile an MLIR module to a loaded executable
PJRT_Error_Code PJRT_Compile(
    PJRT_Client* client,
    const char* mlir_module,
    PJRT_LoadedExecutable** out_executable
);

/// Destroy a loaded executable
void PJRT_DestroyExecutable(PJRT_LoadedExecutable* executable);

/// Execute a loaded executable with input buffers
PJRT_Error_Code PJRT_Execute(
    PJRT_LoadedExecutable* executable,
    PJRT_Buffer** inputs,
    size_t num_inputs,
    PJRT_Buffer*** out_outputs,
    size_t* out_num_outputs
);

#ifdef __cplusplus
}
#endif

#endif // PJRT_SIMPLE_WRAPPER_H
