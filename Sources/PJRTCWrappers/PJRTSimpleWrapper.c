//===-- PJRTSimpleWrapper.c - Simplified PJRT C Wrapper ------*- C -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// Simplified C wrapper for PJRT C API to ease Swift interop
//
//===------------------------------------------------------------------===//

#include "pjrt_c_api.h"
#include "PJRTSimpleWrapper.h"
#include "PJRTProtoHelper.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global state
static void* g_plugin_handle = NULL;
static const PJRT_Api* g_api = NULL;

//===------------------------------------------------------------------===//
// Plugin Loading
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_LoadPlugin(const char* plugin_path) {
    if (g_plugin_handle != NULL) {
        return SW_PJRT_Error_OK; // Already loaded
    }

    // Use RTLD_DEEPBIND on Linux to isolate plugin's LLVM symbols from ours
    // This prevents duplicate LLVM CommandLine registration errors
    #ifdef __linux__
    g_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    #else
    g_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    #endif
    if (g_plugin_handle == NULL) {
        fprintf(stderr, "Failed to load PJRT plugin: %s\n", dlerror());
        return SW_PJRT_Error_INTERNAL;
    }

    // Get the GetPjrtApi function
    typedef const PJRT_Api* (*GetPjrtApiFunc)(void);
    GetPjrtApiFunc get_api = (GetPjrtApiFunc)dlsym(g_plugin_handle, "GetPjrtApi");
    if (get_api == NULL) {
        fprintf(stderr, "Failed to find GetPjrtApi: %s\n", dlerror());
        dlclose(g_plugin_handle);
        g_plugin_handle = NULL;
        return SW_PJRT_Error_INTERNAL;
    }

    g_api = get_api();
    if (g_api == NULL) {
        fprintf(stderr, "GetPjrtApi returned NULL\n");
        dlclose(g_plugin_handle);
        g_plugin_handle = NULL;
        return SW_PJRT_Error_INTERNAL;
    }

    return SW_PJRT_Error_OK;
}

void PJRT_UnloadPlugin(void) {
    if (g_plugin_handle != NULL) {
        dlclose(g_plugin_handle);
        g_plugin_handle = NULL;
        g_api = NULL;
    }
}

const char* PJRT_GetLastError(void) {
    return dlerror();
}

//===------------------------------------------------------------------===//
// Error Handling
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_GetErrorCode(void* error) {
    if (error == NULL) {
        return SW_PJRT_Error_OK;
    }

    PJRT_Error_GetCode_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.error = (PJRT_Error*)error;

    if (g_api && g_api->PJRT_Error_GetCode) {
        g_api->PJRT_Error_GetCode(&args);
        return args.code;
    }

    return SW_PJRT_Error_INTERNAL;
}

const char* PJRT_GetErrorMessage(void* error) {
    if (error == NULL) {
        return "No error";
    }

    PJRT_Error_Message_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.error = (PJRT_Error*)error;

    if (g_api && g_api->PJRT_Error_Message) {
        g_api->PJRT_Error_Message(&args);
        return args.message;
    }

    return "Unknown error";
}

void PJRT_DestroyError(void* error) {
    if (error == NULL || g_api == NULL) {
        return;
    }

    PJRT_Error_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.error = (PJRT_Error*)error;

    if (g_api->PJRT_Error_Destroy) {
        g_api->PJRT_Error_Destroy(&args);
    }
}

//===------------------------------------------------------------------===//
// Client Management
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_CreateClient(void** out_client) {
    if (g_api == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    PJRT_Client_Create_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);

    PJRT_Error* error = g_api->PJRT_Client_Create(&args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_client = (void*)args.client;
    return SW_PJRT_Error_OK;
}

void PJRT_DestroyClient(void* client) {
    if (client == NULL || g_api == NULL) {
        return;
    }

    PJRT_Client_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;

    if (g_api->PJRT_Client_Destroy) {
        PJRT_Error* error = g_api->PJRT_Client_Destroy(&args);
        if (error != NULL) {
            fprintf(stderr, "Warning: PJRT_Client_Destroy failed: %s\n",
                    PJRT_GetErrorMessage(error));
            PJRT_DestroyError(error);
        }
    }
}

SW_PJRT_Error_Code PJRT_GetPlatformName(void* client, const char** out_name) {
    if (g_api == NULL || client == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    PJRT_Client_PlatformName_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;

    PJRT_Error* error = g_api->PJRT_Client_PlatformName(&args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_name = args.platform_name;
    return SW_PJRT_Error_OK;
}

//===------------------------------------------------------------------===//
// Device Enumeration
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_GetAddressableDevices(
    void* client,
    void*** out_devices,
    size_t* out_num_devices
) {
    if (g_api == NULL || client == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    PJRT_Client_AddressableDevices_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;

    PJRT_Error* error = g_api->PJRT_Client_AddressableDevices(&args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_devices = (void**)args.addressable_devices;
    *out_num_devices = args.num_addressable_devices;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_GetDeviceId(void* device, int32_t* out_id) {
    if (g_api == NULL || device == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    // First get device description
    PJRT_Device_GetDescription_Args desc_args;
    memset(&desc_args, 0, sizeof(desc_args));
    desc_args.struct_size = sizeof(desc_args);
    desc_args.device = (PJRT_Device*)device;

    PJRT_Error* error = g_api->PJRT_Device_GetDescription(&desc_args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get ID from description
    PJRT_DeviceDescription_Id_Args id_args;
    memset(&id_args, 0, sizeof(id_args));
    id_args.struct_size = sizeof(id_args);
    id_args.device_description = desc_args.device_description;

    error = g_api->PJRT_DeviceDescription_Id(&id_args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_id = id_args.id;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_GetDeviceKind(void* device, const char** out_kind) {
    if (g_api == NULL || device == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    // First get device description
    PJRT_Device_GetDescription_Args desc_args;
    memset(&desc_args, 0, sizeof(desc_args));
    desc_args.struct_size = sizeof(desc_args);
    desc_args.device = (PJRT_Device*)device;

    PJRT_Error* error = g_api->PJRT_Device_GetDescription(&desc_args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get kind from description
    PJRT_DeviceDescription_Kind_Args kind_args;
    memset(&kind_args, 0, sizeof(kind_args));
    kind_args.struct_size = sizeof(kind_args);
    kind_args.device_description = desc_args.device_description;

    error = g_api->PJRT_DeviceDescription_Kind(&kind_args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_kind = kind_args.device_kind;
    return SW_PJRT_Error_OK;
}

//===------------------------------------------------------------------===//
// Buffer Management
//===------------------------------------------------------------------===//

// Map simplified buffer types to PJRT element types
static PJRT_Buffer_Type MapToPJRTElementType(SW_PJRT_Buffer_Type type) {
    switch (type) {
        case SW_PJRT_Buffer_Type_PRED: return PJRT_Buffer_Type_PRED;
        case SW_PJRT_Buffer_Type_S8:   return PJRT_Buffer_Type_S8;
        case SW_PJRT_Buffer_Type_S16:  return PJRT_Buffer_Type_S16;
        case SW_PJRT_Buffer_Type_S32:  return PJRT_Buffer_Type_S32;
        case SW_PJRT_Buffer_Type_S64:  return PJRT_Buffer_Type_S64;
        case SW_PJRT_Buffer_Type_U8:   return PJRT_Buffer_Type_U8;
        case SW_PJRT_Buffer_Type_U16:  return PJRT_Buffer_Type_U16;
        case SW_PJRT_Buffer_Type_U32:  return PJRT_Buffer_Type_U32;
        case SW_PJRT_Buffer_Type_U64:  return PJRT_Buffer_Type_U64;
        case SW_PJRT_Buffer_Type_F16:  return PJRT_Buffer_Type_F16;
        case SW_PJRT_Buffer_Type_F32:  return PJRT_Buffer_Type_F32;
        case SW_PJRT_Buffer_Type_F64:  return PJRT_Buffer_Type_F64;
        case SW_PJRT_Buffer_Type_BF16: return PJRT_Buffer_Type_BF16;
        case SW_PJRT_Buffer_Type_C64:  return PJRT_Buffer_Type_C64;
        case SW_PJRT_Buffer_Type_C128: return PJRT_Buffer_Type_C128;
        default: return PJRT_Buffer_Type_F32; // Default to F32
    }
}

// Map simplified semantics to PJRT host buffer semantics
static PJRT_HostBufferSemantics MapToHostBufferSemantics(SW_PJRT_HostBufferSemantics semantics) {
    switch (semantics) {
        case SW_PJRT_HostBuffer_ImmutableOnlyDuringCall:
            return PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        case SW_PJRT_HostBuffer_ImmutableUntilTransferCompletes:
            return PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        case SW_PJRT_HostBuffer_ZeroCopy:
            return PJRT_HostBufferSemantics_kImmutableZeroCopy;
        case SW_PJRT_HostBuffer_MutableZeroCopy:
            return PJRT_HostBufferSemantics_kMutableZeroCopy;
        default:
            return PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    }
}

SW_PJRT_Error_Code PJRT_CreateBufferWithSemantics(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffer
) {
    if (g_api == NULL || client == NULL || data == NULL || dims == NULL || device == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Create buffer from host memory with specified semantics
    PJRT_Client_BufferFromHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;
    args.data = data;
    args.type = MapToPJRTElementType(type);
    args.dims = dims;
    args.num_dims = num_dims;
    args.byte_strides = NULL;  // Use default row-major layout
    args.host_buffer_semantics = MapToHostBufferSemantics(semantics);
    args.device = (PJRT_Device*)device;

    PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_CreateBufferWithSemantics failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    *out_buffer = (void*)args.buffer;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_CreateBuffer(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    void** out_buffer
) {
    // Default to ImmutableOnlyDuringCall for backward compatibility
    return PJRT_CreateBufferWithSemantics(
        client, data, type, dims, num_dims, device,
        SW_PJRT_HostBuffer_ImmutableOnlyDuringCall, out_buffer
    );
}

// Thread-local cached args for fast buffer creation
// Supports up to 16 input slots per thread
#define PJRT_MAX_CACHED_BUFFER_SLOTS 16
#define PJRT_MAX_DIMS_PER_BUFFER 8
static __thread PJRT_Client_BufferFromHostBuffer_Args g_cached_buffer_args[PJRT_MAX_CACHED_BUFFER_SLOTS];
static __thread int64_t g_cached_buffer_dims[PJRT_MAX_CACHED_BUFFER_SLOTS][PJRT_MAX_DIMS_PER_BUFFER];
static __thread bool g_cached_buffer_initialized[PJRT_MAX_CACHED_BUFFER_SLOTS] = {false};

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
) {
    if (g_api == NULL || client == NULL || data == NULL || dims == NULL || device == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (cached_slot < 0 || cached_slot >= PJRT_MAX_CACHED_BUFFER_SLOTS ||
        num_dims > PJRT_MAX_DIMS_PER_BUFFER) {
        // Fall back to regular creation for invalid slot or too many dims
        return PJRT_CreateBufferWithSemantics(client, data, type, dims, num_dims, device, semantics, out_buffer);
    }

    PJRT_Client_BufferFromHostBuffer_Args* args = &g_cached_buffer_args[cached_slot];

    if (is_same_shape && g_cached_buffer_initialized[cached_slot]) {
        // Fast path: only update data pointer, dims are already cached
        args->data = data;
    } else {
        // Copy dims to cached storage (they may be from a temporary Swift buffer)
        for (size_t i = 0; i < num_dims; i++) {
            g_cached_buffer_dims[cached_slot][i] = dims[i];
        }

        // Initialize or reinitialize the cached args
        args->struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
        args->extension_start = NULL;
        args->client = (PJRT_Client*)client;
        args->data = data;
        args->type = MapToPJRTElementType(type);
        args->dims = g_cached_buffer_dims[cached_slot];  // Use cached dims
        args->num_dims = num_dims;
        args->byte_strides = NULL;
        args->num_byte_strides = 0;
        args->host_buffer_semantics = MapToHostBufferSemantics(semantics);
        args->device = (PJRT_Device*)device;
        args->memory = NULL;
        args->device_layout = NULL;
        // Output fields will be set by PJRT
        args->buffer = NULL;
        args->done_with_host_buffer = NULL;
        g_cached_buffer_initialized[cached_slot] = true;
    }

    PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_CreateBufferFast failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    *out_buffer = (void*)args->buffer;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_CreateBuffersBatched(
    void* client,
    size_t num_buffers,
    void* const* data_ptrs,
    const SW_PJRT_Buffer_Type* types,
    const int64_t* const* dims_ptrs,
    const size_t* num_dims_array,
    void* device,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffers
) {
    if (g_api == NULL || client == NULL || num_buffers == 0) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Pre-map semantics once for all buffers
    PJRT_HostBufferSemantics pjrt_semantics = MapToHostBufferSemantics(semantics);

    // Create all buffers in a tight loop - minimizes per-buffer overhead
    for (size_t i = 0; i < num_buffers; i++) {
        PJRT_Client_BufferFromHostBuffer_Args args;
        memset(&args, 0, sizeof(args));
        args.struct_size = sizeof(args);
        args.client = (PJRT_Client*)client;
        args.data = data_ptrs[i];
        args.type = MapToPJRTElementType(types[i]);
        args.dims = dims_ptrs[i];
        args.num_dims = num_dims_array[i];
        args.host_buffer_semantics = pjrt_semantics;
        args.device = (PJRT_Device*)device;

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&args);
        if (error != NULL) {
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            fprintf(stderr, "PJRT_CreateBuffersBatched failed at buffer %zu: %s\n",
                    i, PJRT_GetErrorMessage(error));
            PJRT_DestroyError(error);

            // Clean up already-created buffers on error
            for (size_t j = 0; j < i; j++) {
                if (out_buffers[j] != NULL) {
                    PJRT_DestroyBuffer(out_buffers[j]);
                    out_buffers[j] = NULL;
                }
            }
            return code;
        }

        out_buffers[i] = (void*)args.buffer;
    }

    return SW_PJRT_Error_OK;
}

void PJRT_DestroyBuffer(void* buffer) {
    if (buffer == NULL || g_api == NULL) {
        return;
    }

    PJRT_Buffer_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = buffer;

    if (g_api->PJRT_Buffer_Destroy) {
        PJRT_Error* error = g_api->PJRT_Buffer_Destroy(&args);
        if (error != NULL) {
            fprintf(stderr, "Warning: PJRT_Buffer_Destroy failed: %s\n",
                    PJRT_GetErrorMessage(error));
            PJRT_DestroyError(error);
        }
    }
}

SW_PJRT_Error_Code PJRT_BufferToHost(
    void* buffer,
    void* out_data,
    size_t data_size
) {
    if (g_api == NULL || buffer == NULL || out_data == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Transfer buffer from device to host (asynchronous)
    PJRT_Buffer_ToHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.src = (PJRT_Buffer*)buffer;
    args.dst = out_data;
    args.dst_size = data_size;
    args.host_layout = NULL;  // Use default layout

    PJRT_Error* error = g_api->PJRT_Buffer_ToHostBuffer(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Wait for the transfer to complete
    if (args.event != NULL) {
        PJRT_Event_Await_Args await_args;
        memset(&await_args, 0, sizeof(await_args));
        await_args.struct_size = sizeof(await_args);
        await_args.event = args.event;

        error = g_api->PJRT_Event_Await(&await_args);

        // Destroy the event after awaiting
        PJRT_Event_Destroy_Args destroy_args;
        memset(&destroy_args, 0, sizeof(destroy_args));
        destroy_args.struct_size = sizeof(destroy_args);
        destroy_args.event = args.event;
        g_api->PJRT_Event_Destroy(&destroy_args);

        if (error != NULL) {
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }
    }

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_BufferToHostAsync(
    void* buffer,
    void* out_data,
    size_t data_size,
    void** out_event
) {
    if (g_api == NULL || buffer == NULL || out_data == NULL || out_event == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Transfer buffer from device to host (asynchronous - returns immediately)
    PJRT_Buffer_ToHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.src = (PJRT_Buffer*)buffer;
    args.dst = out_data;
    args.dst_size = data_size;
    args.host_layout = NULL;  // Use default layout

    PJRT_Error* error = g_api->PJRT_Buffer_ToHostBuffer(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        *out_event = NULL;
        return code;
    }

    // Return the event for later awaiting
    *out_event = (void*)args.event;
    return SW_PJRT_Error_OK;
}

// Fast version using designated initializers (avoids memset overhead)
SW_PJRT_Error_Code PJRT_BufferToHostAsyncFast(
    void* buffer,
    void* out_data,
    size_t data_size,
    void** out_event
) {
    if (g_api == NULL || buffer == NULL || out_data == NULL || out_event == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Transfer buffer from device to host using designated initializer
    PJRT_Buffer_ToHostBuffer_Args args = {
        .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
        .src = (PJRT_Buffer*)buffer,
        .dst = out_data,
        .dst_size = data_size,
        .host_layout = NULL
    };

    PJRT_Error* error = g_api->PJRT_Buffer_ToHostBuffer(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        *out_event = NULL;
        return code;
    }

    *out_event = (void*)args.event;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_EventAwait(void* event) {
    if (g_api == NULL || event == NULL) {
        return SW_PJRT_Error_OK;  // Nothing to wait for
    }

    PJRT_Event_Await_Args await_args;
    memset(&await_args, 0, sizeof(await_args));
    await_args.struct_size = sizeof(await_args);
    await_args.event = (PJRT_Event*)event;

    PJRT_Error* error = g_api->PJRT_Event_Await(&await_args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    return SW_PJRT_Error_OK;
}

void PJRT_EventDestroy(void* event) {
    if (g_api == NULL || event == NULL) {
        return;
    }

    PJRT_Event_Destroy_Args destroy_args;
    memset(&destroy_args, 0, sizeof(destroy_args));
    destroy_args.struct_size = sizeof(destroy_args);
    destroy_args.event = (PJRT_Event*)event;
    g_api->PJRT_Event_Destroy(&destroy_args);
}

// Fast version using designated initializers
void PJRT_EventDestroyFast(void* event) {
    if (g_api == NULL || event == NULL) {
        return;
    }

    PJRT_Event_Destroy_Args destroy_args = {
        .struct_size = sizeof(PJRT_Event_Destroy_Args),
        .event = (PJRT_Event*)event
    };
    g_api->PJRT_Event_Destroy(&destroy_args);
}

SW_PJRT_Error_Code PJRT_EventAwaitAndDestroy(void* event) {
    if (event == NULL) {
        return SW_PJRT_Error_OK;
    }

    SW_PJRT_Error_Code code = PJRT_EventAwait(event);
    PJRT_EventDestroy(event);
    return code;
}

// Fast version using designated initializers - inlined await and destroy
SW_PJRT_Error_Code PJRT_EventAwaitAndDestroyFast(void* event) {
    if (g_api == NULL || event == NULL) {
        return SW_PJRT_Error_OK;
    }

    // Await using designated initializer
    PJRT_Event_Await_Args await_args = {
        .struct_size = sizeof(PJRT_Event_Await_Args),
        .event = (PJRT_Event*)event
    };

    PJRT_Error* error = g_api->PJRT_Event_Await(&await_args);
    SW_PJRT_Error_Code code = SW_PJRT_Error_OK;

    if (error != NULL) {
        code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
    }

    // Destroy using designated initializer
    PJRT_Event_Destroy_Args destroy_args = {
        .struct_size = sizeof(PJRT_Event_Destroy_Args),
        .event = (PJRT_Event*)event
    };
    g_api->PJRT_Event_Destroy(&destroy_args);

    return code;
}

SW_PJRT_Error_Code PJRT_GetBufferDimensions(
    void* buffer,
    const int64_t** out_dims,
    size_t* out_num_dims
) {
    if (g_api == NULL || buffer == NULL || out_dims == NULL || out_num_dims == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_Dimensions_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_Dimensions(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_dims = args.dims;
    *out_num_dims = args.num_dims;

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_GetBufferOnDeviceSizeInBytes(
    void* buffer,
    size_t* out_size
) {
    if (g_api == NULL || buffer == NULL || out_size == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_OnDeviceSizeInBytes_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_OnDeviceSizeInBytes(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_size = args.on_device_size_in_bytes;

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_Buffer_IncRefCount(void* buffer) {
    if (g_api == NULL || buffer == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_IncreaseExternalReferenceCount_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_IncreaseExternalReferenceCount(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_Buffer_DecRefCount(void* buffer) {
    if (g_api == NULL || buffer == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_DecreaseExternalReferenceCount_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_DecreaseExternalReferenceCount(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    return SW_PJRT_Error_OK;
}

//===------------------------------------------------------------------===//
// Compilation & Execution
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_CompileWrapper(
    void* client,
    const char* mlir_module,
    void** out_executable
) {
    // Use default XLA optimization level
    return PJRT_CompileWrapperWithOptLevel(client, mlir_module, SW_XLA_OPT_DEFAULT, out_executable);
}

SW_PJRT_Error_Code PJRT_CompileWrapperWithOptLevel(
    void* client,
    const char* mlir_module,
    SW_XLA_OptLevel xla_opt_level,
    void** out_executable
) {
    if (g_api == NULL || client == NULL || mlir_module == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Set up compilation options
    PJRT_Program program;
    memset(&program, 0, sizeof(program));
    program.struct_size = sizeof(program);  // Required for API versioning
    program.code = (char*)mlir_module;  // Cast away const for PJRT API
    program.code_size = strlen(mlir_module);
    program.format = (char*)"mlir";  // or "hlo" for HLO text format
    program.format_size = strlen("mlir");

    // Create CompileOptionsProto using C++ helper that properly constructs
    // the protobuf message with XLA's protobuf library
    size_t compile_opts_size = 0;
    char* compile_opts = PJRT_CreateCompileOptionsWithOptLevel(1, 1, (int32_t)xla_opt_level, &compile_opts_size);
    if (compile_opts == NULL) {
        return SW_PJRT_Error_INTERNAL;
    }

    // Prepare compilation args
    PJRT_Client_Compile_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;
    args.program = &program;
    args.compile_options = compile_opts;
    args.compile_options_size = compile_opts_size;

    // Compile the program
    PJRT_Error* error = g_api->PJRT_Client_Compile(&args);

    // Free the compile options buffer (no longer needed after compile call)
    PJRT_FreeCompileOptions(compile_opts);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    *out_executable = (void*)args.executable;
    return SW_PJRT_Error_OK;
}

void PJRT_DestroyExecutable(void* executable) {
    if (executable == NULL || g_api == NULL) {
        return;
    }

    PJRT_LoadedExecutable_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.executable = executable;

    if (g_api->PJRT_LoadedExecutable_Destroy) {
        PJRT_Error* error = g_api->PJRT_LoadedExecutable_Destroy(&args);
        if (error != NULL) {
            PJRT_DestroyError(error);
        }
    }
}

// Thread-local static storage for execute wrapper to avoid malloc/free per call
// Max 16 outputs should be sufficient for most use cases
#define PJRT_MAX_OUTPUTS 16
static __thread PJRT_Buffer* g_output_buffer_array[PJRT_MAX_OUTPUTS];
static __thread PJRT_Buffer** g_output_lists_array[1];
static __thread bool g_execute_storage_initialized = false;

// Cache for number of outputs per executable (avoids repeated queries)
// Simple hash map with executable pointer as key
#define PJRT_NUM_OUTPUTS_CACHE_SIZE 32
static struct {
    void* executable;
    size_t num_outputs;
} g_num_outputs_cache[PJRT_NUM_OUTPUTS_CACHE_SIZE];
static size_t g_num_outputs_cache_index = 0;

static size_t GetCachedNumOutputs(void* executable) {
    for (size_t i = 0; i < PJRT_NUM_OUTPUTS_CACHE_SIZE; i++) {
        if (g_num_outputs_cache[i].executable == executable) {
            return g_num_outputs_cache[i].num_outputs;
        }
    }
    return (size_t)-1;  // Not found
}

static void SetCachedNumOutputs(void* executable, size_t num_outputs) {
    // Simple round-robin replacement
    g_num_outputs_cache[g_num_outputs_cache_index].executable = executable;
    g_num_outputs_cache[g_num_outputs_cache_index].num_outputs = num_outputs;
    g_num_outputs_cache_index = (g_num_outputs_cache_index + 1) % PJRT_NUM_OUTPUTS_CACHE_SIZE;
}

SW_PJRT_Error_Code PJRT_ExecuteWrapper(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void*** out_outputs,
    size_t* out_num_outputs
) {
    if (g_api == NULL || executable == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = (num_inputs > 0 && inputs != NULL)
        ? (PJRT_Buffer**)inputs : NULL;

    // Prepare execution options - use designated initializer to avoid memset
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    // Prepare execute args - use designated initializer
    PJRT_LoadedExecutable_Execute_Args args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .output_lists = g_output_lists_array
    };

    // Workaround for const qualifier warnings
    PJRT_Buffer*const* input_list_const = (PJRT_Buffer*const*)input_buffers;
    PJRT_Buffer*const*const* argument_lists_const = (PJRT_Buffer*const*const*)&input_list_const;
    args.argument_lists = (PJRT_Buffer****)argument_lists_const;

    // Execute
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Check cache for number of outputs first
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1) {
        *out_num_outputs = cached_num_outputs;
    } else {
        // Query number of outputs (only on first call for this executable)
        PJRT_LoadedExecutable_GetExecutable_Args get_exec_args = {
            .struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
            .loaded_executable = (PJRT_LoadedExecutable*)executable
        };

        error = g_api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);

        if (error != NULL) {
            PJRT_DestroyError(error);
            *out_num_outputs = 0;
        } else {
            PJRT_Executable_NumOutputs_Args num_outputs_args = {
                .struct_size = sizeof(PJRT_Executable_NumOutputs_Args),
                .executable = get_exec_args.executable
            };

            error = g_api->PJRT_Executable_NumOutputs(&num_outputs_args);

            if (error != NULL) {
                PJRT_DestroyError(error);
                *out_num_outputs = 0;
            } else {
                *out_num_outputs = num_outputs_args.num_outputs;
                SetCachedNumOutputs(executable, num_outputs_args.num_outputs);
            }
        }
    }

    // Return pointer to static output array (caller should copy if needed)
    // NOTE: This is safe because we're single-threaded per executable
    *out_outputs = (void**)g_output_buffer_array;

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ExecuteWithDonation(
    void* executable,
    void** inputs,
    size_t num_inputs,
    const int64_t* non_donatable_indices,
    size_t num_non_donatable,
    void*** out_outputs,
    size_t* out_num_outputs
) {
    if (g_api == NULL || executable == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = (num_inputs > 0 && inputs != NULL)
        ? (PJRT_Buffer**)inputs : NULL;

    // Prepare execution options with donation settings
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0,
        .non_donatable_input_indices = non_donatable_indices,
        .num_non_donatable_input_indices = num_non_donatable,
    };

    // Prepare execute args
    PJRT_LoadedExecutable_Execute_Args args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .output_lists = g_output_lists_array
    };

    // Workaround for const qualifier warnings
    PJRT_Buffer*const* input_list_const = (PJRT_Buffer*const*)input_buffers;
    PJRT_Buffer*const*const* argument_lists_const = (PJRT_Buffer*const*const*)&input_list_const;
    args.argument_lists = (PJRT_Buffer****)argument_lists_const;

    // Execute
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Check cache for number of outputs first
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1) {
        *out_num_outputs = cached_num_outputs;
    } else {
        // Query number of outputs (only on first call for this executable)
        PJRT_LoadedExecutable_GetExecutable_Args get_exec_args = {
            .struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
            .loaded_executable = (PJRT_LoadedExecutable*)executable
        };

        error = g_api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);

        if (error != NULL) {
            PJRT_DestroyError(error);
            *out_num_outputs = 0;
        } else {
            PJRT_Executable_NumOutputs_Args num_outputs_args = {
                .struct_size = sizeof(PJRT_Executable_NumOutputs_Args),
                .executable = get_exec_args.executable
            };

            error = g_api->PJRT_Executable_NumOutputs(&num_outputs_args);

            if (error != NULL) {
                PJRT_DestroyError(error);
                *out_num_outputs = 0;
            } else {
                *out_num_outputs = num_outputs_args.num_outputs;
                SetCachedNumOutputs(executable, num_outputs_args.num_outputs);
            }
        }
    }

    // Return pointer to static output array
    *out_outputs = (void**)g_output_buffer_array;

    return SW_PJRT_Error_OK;
}

//===------------------------------------------------------------------===//
// Fast Combined Execute + Transfer API
//===------------------------------------------------------------------===//

// Maximum number of outputs for combined execute+transfer
#define PJRT_MAX_COMBINED_OUTPUTS 4

// Thread-local storage for D2H events
static __thread PJRT_Event* g_d2h_events[PJRT_MAX_COMBINED_OUTPUTS];

/// Execute and transfer outputs to host in a single call
/// This reduces FFI overhead by combining execute + async D2H + await into one C call
SW_PJRT_Error_Code PJRT_ExecuteAndTransfer(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void** out_data_ptrs,      // Pre-allocated host memory for each output
    size_t* out_data_sizes,    // Size of each output buffer
    size_t num_outputs,        // Number of outputs (must match executable)
    size_t* out_actual_outputs // Actual number of outputs written
) {
    if (g_api == NULL || executable == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_outputs > PJRT_MAX_COMBINED_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = (num_inputs > 0 && inputs != NULL)
        ? (PJRT_Buffer**)inputs : NULL;

    // Prepare execution options using designated initializer
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    // Prepare execute args using designated initializer
    PJRT_LoadedExecutable_Execute_Args args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .output_lists = g_output_lists_array
    };

    // Workaround for const qualifier warnings
    PJRT_Buffer*const* input_list_const = (PJRT_Buffer*const*)input_buffers;
    PJRT_Buffer*const*const* argument_lists_const = (PJRT_Buffer*const*const*)&input_list_const;
    args.argument_lists = (PJRT_Buffer****)argument_lists_const;

    // Execute
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get actual number of outputs
    size_t actual_outputs = num_outputs;  // Assume caller knows
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1) {
        actual_outputs = cached_num_outputs < num_outputs ? cached_num_outputs : num_outputs;
    }

    *out_actual_outputs = actual_outputs;

    // Initiate async D2H transfers for all outputs
    for (size_t i = 0; i < actual_outputs; i++) {
        if (out_data_ptrs[i] == NULL || g_output_buffer_array[i] == NULL) {
            g_d2h_events[i] = NULL;
            continue;
        }

        // Start async transfer using designated initializer
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_output_buffer_array[i],
            .dst = out_data_ptrs[i],
            .dst_size = out_data_sizes[i],
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Clean up any events we've created so far
            for (size_t j = 0; j < i; j++) {
                if (g_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_d2h_events[i] = d2h_args.event;
    }

    // Await all transfers
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_d2h_events[i] == NULL) {
            continue;
        }

        PJRT_Event_Await_Args await_args = {
            .struct_size = sizeof(PJRT_Event_Await_Args),
            .event = g_d2h_events[i]
        };

        error = g_api->PJRT_Event_Await(&await_args);
        SW_PJRT_Error_Code await_code = SW_PJRT_Error_OK;

        if (error != NULL) {
            await_code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
        }

        // Destroy event
        PJRT_Event_Destroy_Args destroy_args = {
            .struct_size = sizeof(PJRT_Event_Destroy_Args),
            .event = g_d2h_events[i]
        };
        g_api->PJRT_Event_Destroy(&destroy_args);

        if (await_code != SW_PJRT_Error_OK) {
            // Continue destroying remaining events but return error
            for (size_t j = i + 1; j < actual_outputs; j++) {
                if (g_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args cleanup_args = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&cleanup_args);
                }
            }
            return await_code;
        }
    }

    // Destroy output buffers (they're no longer needed after D2H)
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_output_buffer_array[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_output_buffer_array[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_output_buffer_array[i] = NULL;
        }
    }

    return SW_PJRT_Error_OK;
}

/// Hot path: Full H2D + Execute + D2H in a single FFI call
/// This is the most optimized path for repeated execution with the same shapes.
/// All operations happen in C, avoiding multiple FFI boundary crossings.
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
) {
    if (g_api == NULL || client == NULL || executable == NULL || device == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs > 8 || num_outputs > PJRT_MAX_COMBINED_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Thread-local storage for input buffers
    static __thread PJRT_Buffer* hot_path_input_buffers[8];
    static __thread int64_t hot_path_input_dims[8];

    // Map semantics
    PJRT_HostBufferSemantics pjrt_semantics = MapToHostBufferSemantics(semantics);

    // Step 1: Create input buffers (H2D)
    for (size_t i = 0; i < num_inputs; i++) {
        hot_path_input_dims[i] = (int64_t)input_sizes[i];

        PJRT_Client_BufferFromHostBuffer_Args h2d_args = {
            .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
            .client = (PJRT_Client*)client,
            .data = input_data[i],
            .type = PJRT_Buffer_Type_F32,
            .dims = &hot_path_input_dims[i],
            .num_dims = 1,
            .host_buffer_semantics = pjrt_semantics,
            .device = (PJRT_Device*)device
        };

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&h2d_args);
        if (error != NULL) {
            // Cleanup any buffers created so far
            for (size_t j = 0; j < i; j++) {
                if (hot_path_input_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = hot_path_input_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        hot_path_input_buffers[i] = h2d_args.buffer;
    }

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // Step 2: Execute
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[8];
    for (size_t i = 0; i < num_inputs; i++) {
        input_list[i] = hot_path_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_output_lists_array
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    // Destroy input buffers immediately (we're done with them)
    for (size_t i = 0; i < num_inputs; i++) {
        if (hot_path_input_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = hot_path_input_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            hot_path_input_buffers[i] = NULL;
        }
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Step 3: D2H transfers
    size_t actual_outputs = num_outputs;
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1 && cached_num_outputs < num_outputs) {
        actual_outputs = cached_num_outputs;
    }

    // Start all async transfers
    for (size_t i = 0; i < actual_outputs; i++) {
        if (output_data[i] == NULL || g_output_buffer_array[i] == NULL) {
            g_d2h_events[i] = NULL;
            continue;
        }

        size_t byte_size = output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_output_buffer_array[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Cleanup events and buffers
            for (size_t j = 0; j < i; j++) {
                if (g_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < actual_outputs; j++) {
                if (g_output_buffer_array[j] != NULL) {
                    PJRT_Buffer_Destroy_Args buf_destroy = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = g_output_buffer_array[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&buf_destroy);
                    g_output_buffer_array[j] = NULL;
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_d2h_events[i] = d2h_args.event;
    }

    // Await all transfers
    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_d2h_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = g_d2h_events[i]
            };

            error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_d2h_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    // Destroy output buffers
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_output_buffer_array[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_output_buffer_array[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_output_buffer_array[i] = NULL;
        }
    }

    return result;
}

//===------------------------------------------------------------------===//
// Execution Timing/Profiling API Implementation
//===------------------------------------------------------------------===//

#include <time.h>

// High-precision timing using clock_gettime
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

SW_PJRT_Error_Code PJRT_ExecuteWithTiming(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void*** out_outputs,
    size_t* out_num_outputs,
    SW_PJRT_ExecutionTiming* out_timing
) {
    if (g_api == NULL || executable == NULL || out_timing == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    uint64_t total_start = get_time_ns();

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = (num_inputs > 0 && inputs != NULL)
        ? (PJRT_Buffer**)inputs : NULL;

    // Prepare execution options
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    // Prepare execute args
    PJRT_LoadedExecutable_Execute_Args args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .output_lists = g_output_lists_array
    };

    PJRT_Buffer*const* input_list_const = (PJRT_Buffer*const*)input_buffers;
    PJRT_Buffer*const*const* argument_lists_const = (PJRT_Buffer*const*const*)&input_list_const;
    args.argument_lists = (PJRT_Buffer****)argument_lists_const;

    // Time the execution
    uint64_t exec_start = get_time_ns();
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);
    uint64_t exec_end = get_time_ns();

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get number of outputs
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1) {
        *out_num_outputs = cached_num_outputs;
    } else {
        PJRT_LoadedExecutable_GetExecutable_Args get_exec_args = {
            .struct_size = sizeof(PJRT_LoadedExecutable_GetExecutable_Args),
            .loaded_executable = (PJRT_LoadedExecutable*)executable
        };
        error = g_api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
        if (error == NULL) {
            PJRT_Executable_NumOutputs_Args num_outputs_args = {
                .struct_size = sizeof(PJRT_Executable_NumOutputs_Args),
                .executable = get_exec_args.executable
            };
            error = g_api->PJRT_Executable_NumOutputs(&num_outputs_args);
            if (error == NULL) {
                *out_num_outputs = num_outputs_args.num_outputs;
                SetCachedNumOutputs(executable, num_outputs_args.num_outputs);
            } else {
                PJRT_DestroyError(error);
                *out_num_outputs = 0;
            }
        } else {
            PJRT_DestroyError(error);
            *out_num_outputs = 0;
        }
    }

    *out_outputs = (void**)g_output_buffer_array;

    uint64_t total_end = get_time_ns();

    // Fill timing structure
    out_timing->h2d_create_ns = 0;  // Buffers already on device
    out_timing->execute_ns = exec_end - exec_start;
    out_timing->d2h_initiate_ns = 0;  // No D2H in this function
    out_timing->d2h_await_ns = 0;
    out_timing->buffer_destroy_ns = 0;
    out_timing->total_ns = total_end - total_start;
    out_timing->num_inputs = num_inputs;
    out_timing->num_outputs = *out_num_outputs;

    return SW_PJRT_Error_OK;
}

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
) {
    if (g_api == NULL || client == NULL || executable == NULL || device == NULL || out_timing == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs > 8 || num_outputs > PJRT_MAX_COMBINED_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    uint64_t total_start = get_time_ns();

    // Thread-local storage for input buffers
    static __thread PJRT_Buffer* profiled_input_buffers[8];
    static __thread int64_t profiled_input_dims[8];

    PJRT_HostBufferSemantics pjrt_semantics = MapToHostBufferSemantics(semantics);

    // ===== PHASE 1: H2D Buffer Creation =====
    uint64_t h2d_start = get_time_ns();

    for (size_t i = 0; i < num_inputs; i++) {
        profiled_input_dims[i] = (int64_t)input_sizes[i];

        PJRT_Client_BufferFromHostBuffer_Args h2d_args = {
            .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
            .client = (PJRT_Client*)client,
            .data = input_data[i],
            .type = PJRT_Buffer_Type_F32,
            .dims = &profiled_input_dims[i],
            .num_dims = 1,
            .host_buffer_semantics = pjrt_semantics,
            .device = (PJRT_Device*)device
        };

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&h2d_args);
        if (error != NULL) {
            for (size_t j = 0; j < i; j++) {
                if (profiled_input_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = profiled_input_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        profiled_input_buffers[i] = h2d_args.buffer;
    }

    uint64_t h2d_end = get_time_ns();

    // Initialize thread-local storage on first use
    if (!g_execute_storage_initialized) {
        g_output_lists_array[0] = g_output_buffer_array;
        g_execute_storage_initialized = true;
    }

    // ===== PHASE 2: Execute =====
    uint64_t exec_start = get_time_ns();

    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[8];
    for (size_t i = 0; i < num_inputs; i++) {
        input_list[i] = profiled_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_output_lists_array
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    uint64_t exec_end = get_time_ns();

    // Destroy input buffers (they're consumed)
    uint64_t input_destroy_start = get_time_ns();
    for (size_t i = 0; i < num_inputs; i++) {
        if (profiled_input_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = profiled_input_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            profiled_input_buffers[i] = NULL;
        }
    }
    uint64_t input_destroy_end = get_time_ns();

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get actual number of outputs
    size_t actual_outputs = num_outputs;
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1 && cached_num_outputs < num_outputs) {
        actual_outputs = cached_num_outputs;
    }

    // ===== PHASE 3: D2H Initiate =====
    uint64_t d2h_init_start = get_time_ns();

    for (size_t i = 0; i < actual_outputs; i++) {
        if (output_data[i] == NULL || g_output_buffer_array[i] == NULL) {
            g_d2h_events[i] = NULL;
            continue;
        }

        size_t byte_size = output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_output_buffer_array[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            for (size_t j = 0; j < i; j++) {
                if (g_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < actual_outputs; j++) {
                if (g_output_buffer_array[j] != NULL) {
                    PJRT_Buffer_Destroy_Args buf_destroy = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = g_output_buffer_array[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&buf_destroy);
                    g_output_buffer_array[j] = NULL;
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_d2h_events[i] = d2h_args.event;
    }

    uint64_t d2h_init_end = get_time_ns();

    // ===== PHASE 4: D2H Await =====
    uint64_t d2h_await_start = get_time_ns();

    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_d2h_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = g_d2h_events[i]
            };

            error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_d2h_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    uint64_t d2h_await_end = get_time_ns();

    // ===== PHASE 5: Output Buffer Cleanup =====
    uint64_t output_destroy_start = get_time_ns();

    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_output_buffer_array[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_output_buffer_array[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_output_buffer_array[i] = NULL;
        }
    }

    uint64_t output_destroy_end = get_time_ns();
    uint64_t total_end = get_time_ns();

    // Fill timing structure
    out_timing->h2d_create_ns = h2d_end - h2d_start;
    out_timing->execute_ns = exec_end - exec_start;
    out_timing->d2h_initiate_ns = d2h_init_end - d2h_init_start;
    out_timing->d2h_await_ns = d2h_await_end - d2h_await_start;
    out_timing->buffer_destroy_ns = (input_destroy_end - input_destroy_start) +
                                    (output_destroy_end - output_destroy_start);
    out_timing->total_ns = total_end - total_start;
    out_timing->num_inputs = num_inputs;
    out_timing->num_outputs = actual_outputs;

    return result;
}

//===------------------------------------------------------------------===//
// Async Execution API Implementation
//===------------------------------------------------------------------===//

// Thread-local storage for async execution context
static __thread SW_PJRT_AsyncContext g_async_context;
static __thread PJRT_Buffer* g_async_output_buffers[PJRT_MAX_COMBINED_OUTPUTS];
static __thread PJRT_Event* g_async_d2h_events[PJRT_MAX_COMBINED_OUTPUTS];
static __thread volatile int g_async_pending_count;
static __thread volatile SW_PJRT_Error_Code g_async_error;

// Callback context for OnReady
typedef struct {
    size_t output_index;
    SW_PJRT_AsyncContext* context;
} OnReadyContext;

static __thread OnReadyContext g_on_ready_contexts[PJRT_MAX_COMBINED_OUTPUTS];

// Internal callback invoked when a single D2H transfer completes
static void on_d2h_ready_callback(PJRT_Error* error, void* user_arg) {
    OnReadyContext* ctx = (OnReadyContext*)user_arg;
    SW_PJRT_AsyncContext* async_ctx = ctx->context;

    // Handle any error from this transfer
    if (error != NULL) {
        if (async_ctx->error == SW_PJRT_Error_OK) {
            async_ctx->error = PJRT_GetErrorCode(error);
        }
        // Callback owns the error - destroy it
        PJRT_DestroyError(error);
    }

    // Decrement pending count atomically
    int remaining = __sync_sub_and_fetch((int*)&async_ctx->pending_count, 1);

    // If all transfers complete, invoke user callback and cleanup
    if (remaining == 0) {
        // Cleanup output buffers
        for (size_t i = 0; i < async_ctx->num_outputs; i++) {
            if (async_ctx->output_buffers[i] != NULL) {
                PJRT_Buffer_Destroy_Args destroy_args = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = (PJRT_Buffer*)async_ctx->output_buffers[i]
                };
                g_api->PJRT_Buffer_Destroy(&destroy_args);
            }
        }

        // Destroy events
        for (size_t i = 0; i < async_ctx->num_outputs; i++) {
            if (async_ctx->d2h_events[i] != NULL) {
                PJRT_Event_Destroy_Args ev_destroy = {
                    .struct_size = sizeof(PJRT_Event_Destroy_Args),
                    .event = (PJRT_Event*)async_ctx->d2h_events[i]
                };
                g_api->PJRT_Event_Destroy(&ev_destroy);
            }
        }

        // Invoke user callback
        if (async_ctx->callback != NULL) {
            async_ctx->callback(async_ctx->error, async_ctx->user_data);
        }
    }
}

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
) {
    if (g_api == NULL || client == NULL || executable == NULL || device == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs > 8 || num_outputs > PJRT_MAX_COMBINED_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize async context
    g_async_context.client = client;
    g_async_context.executable = executable;
    g_async_context.device = device;
    g_async_context.output_buffers = (void**)g_async_output_buffers;
    g_async_context.num_outputs = num_outputs;
    g_async_context.output_data = output_data;
    g_async_context.output_sizes = output_sizes;
    g_async_context.d2h_events = (void**)g_async_d2h_events;
    g_async_context.callback = callback;
    g_async_context.user_data = user_data;
    g_async_context.error = SW_PJRT_Error_OK;
    g_async_context.pending_count = 0;

    PJRT_HostBufferSemantics pjrt_semantics = MapToHostBufferSemantics(semantics);

    // Thread-local storage for input buffers
    static __thread PJRT_Buffer* async_input_buffers[8];
    static __thread int64_t async_input_dims[8];

    // Create input buffers (H2D)
    for (size_t i = 0; i < num_inputs; i++) {
        async_input_dims[i] = (int64_t)input_sizes[i];

        PJRT_Client_BufferFromHostBuffer_Args h2d_args = {
            .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
            .client = (PJRT_Client*)client,
            .data = input_data[i],
            .type = PJRT_Buffer_Type_F32,
            .dims = &async_input_dims[i],
            .num_dims = 1,
            .host_buffer_semantics = pjrt_semantics,
            .device = (PJRT_Device*)device
        };

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&h2d_args);
        if (error != NULL) {
            // Cleanup previously created buffers
            for (size_t j = 0; j < i; j++) {
                if (async_input_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = async_input_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        async_input_buffers[i] = h2d_args.buffer;
    }

    // Execute
    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[8];
    for (size_t i = 0; i < num_inputs; i++) {
        input_list[i] = async_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    // Use thread-local output array
    static __thread PJRT_Buffer* async_exec_output_array[PJRT_MAX_COMBINED_OUTPUTS];
    static __thread PJRT_Buffer** async_output_lists_array[1];
    async_output_lists_array[0] = async_exec_output_array;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = async_output_lists_array
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    // Destroy input buffers (consumed)
    for (size_t i = 0; i < num_inputs; i++) {
        if (async_input_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = async_input_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            async_input_buffers[i] = NULL;
        }
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Copy output buffer handles to context
    size_t actual_outputs = num_outputs;
    size_t cached_num_outputs = GetCachedNumOutputs(executable);
    if (cached_num_outputs != (size_t)-1 && cached_num_outputs < num_outputs) {
        actual_outputs = cached_num_outputs;
    }
    g_async_context.num_outputs = actual_outputs;

    for (size_t i = 0; i < actual_outputs; i++) {
        g_async_output_buffers[i] = async_exec_output_array[i];
    }

    // Initiate D2H transfers with OnReady callbacks
    g_async_context.pending_count = actual_outputs;

    for (size_t i = 0; i < actual_outputs; i++) {
        if (output_data[i] == NULL || async_exec_output_array[i] == NULL) {
            g_async_d2h_events[i] = NULL;
            // Decrement pending count for skipped outputs
            __sync_sub_and_fetch((int*)&g_async_context.pending_count, 1);
            continue;
        }

        size_t byte_size = output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = async_exec_output_array[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Store error but continue - we'll report in callback
            if (g_async_context.error == SW_PJRT_Error_OK) {
                g_async_context.error = PJRT_GetErrorCode(error);
            }
            PJRT_DestroyError(error);
            g_async_d2h_events[i] = NULL;
            __sync_sub_and_fetch((int*)&g_async_context.pending_count, 1);
            continue;
        }

        g_async_d2h_events[i] = d2h_args.event;

        // Setup OnReady callback for this event
        g_on_ready_contexts[i].output_index = i;
        g_on_ready_contexts[i].context = &g_async_context;

        PJRT_Event_OnReady_Args on_ready_args = {
            .struct_size = sizeof(PJRT_Event_OnReady_Args),
            .event = d2h_args.event,
            .callback = on_d2h_ready_callback,
            .user_arg = &g_on_ready_contexts[i]
        };

        error = g_api->PJRT_Event_OnReady(&on_ready_args);
        if (error != NULL) {
            // OnReady registration failed - fall back to await
            if (g_async_context.error == SW_PJRT_Error_OK) {
                g_async_context.error = PJRT_GetErrorCode(error);
            }
            PJRT_DestroyError(error);

            // Await this one synchronously
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = d2h_args.event
            };
            g_api->PJRT_Event_Await(&await_args);
            __sync_sub_and_fetch((int*)&g_async_context.pending_count, 1);
        }
    }

    // If no outputs were pending, invoke callback immediately
    if (g_async_context.pending_count == 0) {
        // Cleanup output buffers
        for (size_t i = 0; i < actual_outputs; i++) {
            if (g_async_output_buffers[i] != NULL) {
                PJRT_Buffer_Destroy_Args destroy_args = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = g_async_output_buffers[i]
                };
                g_api->PJRT_Buffer_Destroy(&destroy_args);
            }
            if (g_async_d2h_events[i] != NULL) {
                PJRT_Event_Destroy_Args ev_destroy = {
                    .struct_size = sizeof(PJRT_Event_Destroy_Args),
                    .event = g_async_d2h_events[i]
                };
                g_api->PJRT_Event_Destroy(&ev_destroy);
            }
        }

        if (callback != NULL) {
            callback(g_async_context.error, user_data);
        }
    }

    return SW_PJRT_Error_OK;
}

// Synchronization context for blocking version of callback-based execution
typedef struct {
    volatile int completed;
    volatile SW_PJRT_Error_Code result;
} SyncContext;

static void sync_completion_callback(SW_PJRT_Error_Code error, void* user_data) {
    SyncContext* ctx = (SyncContext*)user_data;
    ctx->result = error;
    __sync_synchronize();  // Memory barrier
    ctx->completed = 1;
}

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
) {
    if (out_timing == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    uint64_t total_start = get_time_ns();

    SyncContext sync_ctx = { .completed = 0, .result = SW_PJRT_Error_OK };

    uint64_t exec_start = get_time_ns();

    SW_PJRT_Error_Code result = PJRT_ExecuteAsync(
        client, executable, device,
        input_data, input_sizes, num_inputs,
        output_data, output_sizes, num_outputs,
        semantics,
        sync_completion_callback,
        &sync_ctx
    );

    uint64_t exec_end = get_time_ns();

    if (result != SW_PJRT_Error_OK) {
        return result;
    }

    // Wait for completion (spin-wait with yield)
    uint64_t wait_start = get_time_ns();
    while (!sync_ctx.completed) {
        // CPU pause/yield to reduce contention
        #if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause" ::: "memory");
        #elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
        #endif
    }
    uint64_t wait_end = get_time_ns();

    uint64_t total_end = get_time_ns();

    // Fill timing (approximate breakdown)
    out_timing->h2d_create_ns = 0;  // Included in execute time
    out_timing->execute_ns = exec_end - exec_start;
    out_timing->d2h_initiate_ns = 0;  // Included in execute time
    out_timing->d2h_await_ns = wait_end - wait_start;
    out_timing->buffer_destroy_ns = 0;  // Included in callback
    out_timing->total_ns = total_end - total_start;
    out_timing->num_inputs = num_inputs;
    out_timing->num_outputs = num_outputs;

    return sync_ctx.result;
}

//===------------------------------------------------------------------===//
// Profiler Extension API Implementation
//===------------------------------------------------------------------===//

// PJRT_Extension_Type is already defined in pjrt_c_api.h
// We use PJRT_Extension_Type_Profiler directly from there

// Profiler C API structures (from profiler_c_api.h)
typedef struct PLUGIN_Profiler PLUGIN_Profiler;
typedef struct PLUGIN_Profiler_Error PLUGIN_Profiler_Error;

typedef struct {
    size_t struct_size;
    void* priv;
    PLUGIN_Profiler_Error* error;
} PLUGIN_Profiler_Error_Destroy_Args;

typedef void (*PLUGIN_Profiler_Error_Destroy)(PLUGIN_Profiler_Error_Destroy_Args* args);

typedef struct {
    size_t struct_size;
    void* priv;
    const PLUGIN_Profiler_Error* error;
    const char* message;
    size_t message_size;
} PLUGIN_Profiler_Error_Message_Args;

typedef void (*PLUGIN_Profiler_Error_Message)(PLUGIN_Profiler_Error_Message_Args* args);

typedef struct {
    size_t struct_size;
    void* priv;
    const PLUGIN_Profiler_Error* error;
    int code;
} PLUGIN_Profiler_Error_GetCode_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_Error_GetCode)(PLUGIN_Profiler_Error_GetCode_Args* args);

typedef struct {
    size_t struct_size;
    const char* options;
    size_t options_size;
    PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Create_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_Create)(PLUGIN_Profiler_Create_Args* args);

typedef struct {
    size_t struct_size;
    PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Destroy_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_Destroy)(PLUGIN_Profiler_Destroy_Args* args);

typedef struct {
    size_t struct_size;
    PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Start_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_Start)(PLUGIN_Profiler_Start_Args* args);

typedef struct {
    size_t struct_size;
    PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Stop_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_Stop)(PLUGIN_Profiler_Stop_Args* args);

typedef struct {
    size_t struct_size;
    PLUGIN_Profiler* profiler;
    uint8_t* buffer;
    size_t buffer_size_in_bytes;
} PLUGIN_Profiler_CollectData_Args;

typedef PLUGIN_Profiler_Error* (*PLUGIN_Profiler_CollectData)(PLUGIN_Profiler_CollectData_Args* args);

typedef struct {
    size_t struct_size;
    void* priv;
    PLUGIN_Profiler_Error_Destroy error_destroy;
    PLUGIN_Profiler_Error_Message error_message;
    PLUGIN_Profiler_Error_GetCode error_get_code;
    PLUGIN_Profiler_Create create;
    PLUGIN_Profiler_Destroy destroy;
    PLUGIN_Profiler_Start start;
    PLUGIN_Profiler_Stop stop;
    PLUGIN_Profiler_CollectData collect_data;
} PLUGIN_Profiler_Api;

// Cached profiler API pointer
static PLUGIN_Profiler_Api* g_profiler_api = NULL;
static bool g_profiler_api_searched = false;

// Helper to find profiler extension in the extension chain
static PLUGIN_Profiler_Api* FindProfilerExtension(void) {
    if (g_profiler_api_searched) {
        return g_profiler_api;
    }
    g_profiler_api_searched = true;

    if (g_api == NULL) {
        return NULL;
    }

    // Walk the extension chain
    PJRT_Extension_Base* ext = g_api->extension_start;
    while (ext != NULL) {
        if (ext->type == (PJRT_Extension_Type)PJRT_Extension_Type_Profiler) {
            // Found profiler extension, cast to profiler extension struct
            // PJRT_Profiler_Extension has: base, profiler_api, traceme_context_id
            // The profiler_api is at offset sizeof(PJRT_Extension_Base)
            void** ext_ptr = (void**)ext;
            // Skip PJRT_Extension_Base (struct_size, type, next = 3 pointers)
            g_profiler_api = (PLUGIN_Profiler_Api*)ext_ptr[3];  // profiler_api is 4th field
            return g_profiler_api;
        }
        ext = ext->next;
    }

    return NULL;
}

bool PJRT_HasProfilerExtension(void) {
    return FindProfilerExtension() != NULL;
}

void* PJRT_GetProfilerApi(void) {
    return FindProfilerExtension();
}

SW_PJRT_Error_Code PJRT_ProfilerCreate(
    const char* options,
    size_t options_size,
    void** out_profiler
) {
    PLUGIN_Profiler_Api* api = FindProfilerExtension();
    if (api == NULL || api->create == NULL) {
        return SW_PJRT_Error_UNIMPLEMENTED;
    }

    PLUGIN_Profiler_Create_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.options = options;
    args.options_size = options_size;

    PLUGIN_Profiler_Error* error = api->create(&args);
    if (error != NULL) {
        if (api->error_destroy) {
            PLUGIN_Profiler_Error_Destroy_Args destroy_args;
            destroy_args.struct_size = sizeof(destroy_args);
            destroy_args.error = error;
            api->error_destroy(&destroy_args);
        }
        return SW_PJRT_Error_INTERNAL;
    }

    *out_profiler = args.profiler;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ProfilerStart(void* profiler) {
    PLUGIN_Profiler_Api* api = FindProfilerExtension();
    if (api == NULL || api->start == NULL || profiler == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PLUGIN_Profiler_Start_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = (PLUGIN_Profiler*)profiler;

    PLUGIN_Profiler_Error* error = api->start(&args);
    if (error != NULL) {
        if (api->error_destroy) {
            PLUGIN_Profiler_Error_Destroy_Args destroy_args;
            destroy_args.struct_size = sizeof(destroy_args);
            destroy_args.error = error;
            api->error_destroy(&destroy_args);
        }
        return SW_PJRT_Error_INTERNAL;
    }

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ProfilerStop(void* profiler) {
    PLUGIN_Profiler_Api* api = FindProfilerExtension();
    if (api == NULL || api->stop == NULL || profiler == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PLUGIN_Profiler_Stop_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = (PLUGIN_Profiler*)profiler;

    PLUGIN_Profiler_Error* error = api->stop(&args);
    if (error != NULL) {
        if (api->error_destroy) {
            PLUGIN_Profiler_Error_Destroy_Args destroy_args;
            destroy_args.struct_size = sizeof(destroy_args);
            destroy_args.error = error;
            api->error_destroy(&destroy_args);
        }
        return SW_PJRT_Error_INTERNAL;
    }

    return SW_PJRT_Error_OK;
}

// Cache for profiler data - the XLA API fills internal buffer on first call only
typedef struct {
    uint8_t* data;
    size_t size;
} ProfilerDataCache;

// Simple cache - one per profiler (in practice, one profiler at a time)
static ProfilerDataCache g_profiler_data_cache = {NULL, 0};

SW_PJRT_Error_Code PJRT_ProfilerCollectData(
    void* profiler,
    uint8_t* buffer,
    size_t* buffer_size
) {
    PLUGIN_Profiler_Api* api = FindProfilerExtension();
    if (api == NULL || api->collect_data == NULL || profiler == NULL || buffer_size == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // If we already have cached data, use it
    if (g_profiler_data_cache.data != NULL && g_profiler_data_cache.size > 0) {
        if (buffer != NULL) {
            size_t copy_size = g_profiler_data_cache.size;
            if (copy_size > *buffer_size) {
                copy_size = *buffer_size;
            }
            memcpy(buffer, g_profiler_data_cache.data, copy_size);
        }
        *buffer_size = g_profiler_data_cache.size;
        return SW_PJRT_Error_OK;
    }

    // The XLA profiler API has a peculiar design:
    // First call with buffer=NULL: allocates internal buffer, serializes data, returns size and buffer pointer
    // Second call: does nothing - expects you to use the buffer pointer from first call
    //
    // We call with buffer=NULL to get the data and cache it.

    PLUGIN_Profiler_CollectData_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = (PLUGIN_Profiler*)profiler;
    args.buffer = NULL;  // NULL to trigger serialization and get internal buffer
    args.buffer_size_in_bytes = 0;

    PLUGIN_Profiler_Error* error = api->collect_data(&args);
    if (error != NULL) {
        if (api->error_destroy) {
            PLUGIN_Profiler_Error_Destroy_Args destroy_args;
            destroy_args.struct_size = sizeof(destroy_args);
            destroy_args.error = error;
            api->error_destroy(&destroy_args);
        }
        return SW_PJRT_Error_INTERNAL;
    }

    // Cache the data pointer for subsequent calls
    if (args.buffer != NULL && args.buffer_size_in_bytes > 0) {
        g_profiler_data_cache.data = args.buffer;
        g_profiler_data_cache.size = args.buffer_size_in_bytes;
    }

    // If user provided a buffer, copy the data
    if (buffer != NULL && args.buffer != NULL && args.buffer_size_in_bytes > 0) {
        size_t copy_size = args.buffer_size_in_bytes;
        if (copy_size > *buffer_size) {
            copy_size = *buffer_size;
        }
        memcpy(buffer, args.buffer, copy_size);
    }

    *buffer_size = args.buffer_size_in_bytes;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ProfilerDestroy(void* profiler) {
    // Clear the profiler data cache
    g_profiler_data_cache.data = NULL;
    g_profiler_data_cache.size = 0;

    PLUGIN_Profiler_Api* api = FindProfilerExtension();
    if (api == NULL || api->destroy == NULL || profiler == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PLUGIN_Profiler_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = (PLUGIN_Profiler*)profiler;

    PLUGIN_Profiler_Error* error = api->destroy(&args);
    if (error != NULL) {
        if (api->error_destroy) {
            PLUGIN_Profiler_Error_Destroy_Args destroy_args;
            destroy_args.struct_size = sizeof(destroy_args);
            destroy_args.error = error;
            api->error_destroy(&destroy_args);
        }
        return SW_PJRT_Error_INTERNAL;
    }

    return SW_PJRT_Error_OK;
}

//===------------------------------------------------------------------===//
// TraceMe API Implementation
// These functions call into the PJRT plugin's TraceMe exports so Swift
// traces appear in the same profile as XLA internal traces.
//===------------------------------------------------------------------===//

// Cached function pointers for TraceMe API
static int64_t (*g_traceme_start)(const char*, int32_t) = NULL;
static void (*g_traceme_stop)(int64_t) = NULL;
static bool (*g_traceme_active)(int32_t) = NULL;
static void (*g_traceme_instant)(const char*, int32_t) = NULL;
static bool g_traceme_api_searched = false;

// Helper to find TraceMe API in the loaded plugin
static void FindTraceMeApi(void) {
    if (g_traceme_api_searched) {
        return;
    }
    g_traceme_api_searched = true;

    if (g_plugin_handle == NULL) {
        return;
    }

    // Look up TraceMe functions from the plugin
    g_traceme_start = (int64_t (*)(const char*, int32_t))dlsym(g_plugin_handle, "PJRT_TraceMeStart");
    g_traceme_stop = (void (*)(int64_t))dlsym(g_plugin_handle, "PJRT_TraceMeStop");
    g_traceme_active = (bool (*)(int32_t))dlsym(g_plugin_handle, "PJRT_TraceMeActive");
    g_traceme_instant = (void (*)(const char*, int32_t))dlsym(g_plugin_handle, "PJRT_TraceMeInstant");

    if (g_traceme_start != NULL) {
        // fprintf(stderr, "PJRT: TraceMe API found in plugin\n");
    }
}

bool PJRT_HasTraceMeApi(void) {
    FindTraceMeApi();
    return g_traceme_start != NULL && g_traceme_stop != NULL;
}

int64_t PJRT_TraceMeStart(const char* name, int32_t level) {
    FindTraceMeApi();
    if (g_traceme_start == NULL) {
        return 0;  // Return 0 (no-op activity ID) if TraceMe not available
    }
    return g_traceme_start(name, level);
}

void PJRT_TraceMeStop(int64_t activity_id) {
    FindTraceMeApi();
    if (g_traceme_stop == NULL || activity_id == 0) {
        return;  // No-op if TraceMe not available or invalid activity ID
    }
    g_traceme_stop(activity_id);
}

bool PJRT_TraceMeActive(int32_t level) {
    FindTraceMeApi();
    if (g_traceme_active == NULL) {
        return false;  // Assume tracing is disabled if API not available
    }
    return g_traceme_active(level);
}

void PJRT_TraceMeInstant(const char* name, int32_t level) {
    FindTraceMeApi();
    if (g_traceme_instant == NULL) {
        return;  // No-op if TraceMe not available
    }
    g_traceme_instant(name, level);
}

//===------------------------------------------------------------------===//
// Buffer Pool Implementation
//===------------------------------------------------------------------===//

// Thread-local storage for pooled execution
static __thread PJRT_Buffer* g_pooled_output_buffers[PJRT_MAX_COMBINED_OUTPUTS];
static __thread PJRT_Buffer** g_pooled_output_lists[1];
static __thread PJRT_Event* g_pooled_d2h_events[PJRT_MAX_COMBINED_OUTPUTS];
static __thread bool g_pooled_storage_initialized = false;

SW_PJRT_Error_Code PJRT_BufferPoolCreate(
    void* client,
    void* device,
    void* executable,
    const size_t* input_sizes,
    size_t num_inputs,
    SW_PJRT_BufferPool* out_pool
) {
    if (g_api == NULL || client == NULL || device == NULL || out_pool == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs > PJRT_BUFFER_POOL_SIZE) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize pool structure
    memset(out_pool, 0, sizeof(SW_PJRT_BufferPool));
    out_pool->client = client;
    out_pool->device = device;
    out_pool->executable = executable;
    out_pool->num_inputs = num_inputs;

    // Store input sizes for later validation
    for (size_t i = 0; i < num_inputs; i++) {
        out_pool->inputs[i].size_elements = input_sizes[i];
        out_pool->inputs[i].type = SW_PJRT_Buffer_Type_F32;
        out_pool->inputs[i].buffer = NULL;  // Buffers created on first use
        out_pool->inputs[i].in_use = false;
    }

    out_pool->initialized = true;
    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ExecutePooled(
    SW_PJRT_BufferPool* pool,
    const void* const* input_data,
    const size_t* input_sizes,
    size_t num_inputs,
    void* const* output_data,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_ExecutionTiming* out_timing
) {
    if (pool == NULL || !pool->initialized || g_api == NULL || out_timing == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs != pool->num_inputs || num_inputs > PJRT_BUFFER_POOL_SIZE ||
        num_outputs > PJRT_MAX_COMBINED_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Validate input sizes match pool configuration
    for (size_t i = 0; i < num_inputs; i++) {
        if (input_sizes[i] != pool->inputs[i].size_elements) {
            return SW_PJRT_Error_INVALID_ARGUMENT;
        }
    }

    uint64_t total_start = get_time_ns();

    // Thread-local input buffer storage
    static __thread PJRT_Buffer* pooled_input_buffers[PJRT_BUFFER_POOL_SIZE];
    static __thread int64_t pooled_input_dims[PJRT_BUFFER_POOL_SIZE];

    // ===== PHASE 1: H2D Buffer Creation with Zero-Copy =====
    // Use zero-copy semantics to minimize transfer overhead
    uint64_t h2d_start = get_time_ns();

    for (size_t i = 0; i < num_inputs; i++) {
        pooled_input_dims[i] = (int64_t)input_sizes[i];

        PJRT_Client_BufferFromHostBuffer_Args h2d_args = {
            .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
            .client = (PJRT_Client*)pool->client,
            .data = input_data[i],
            .type = PJRT_Buffer_Type_F32,
            .dims = &pooled_input_dims[i],
            .num_dims = 1,
            // Use zero-copy for fastest transfer
            .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy,
            .device = (PJRT_Device*)pool->device
        };

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&h2d_args);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (pooled_input_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = pooled_input_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        pooled_input_buffers[i] = h2d_args.buffer;

        // For zero-copy, we don't need to await the done_with_host_buffer event
        // The event signals when the runtime is done with our memory, but for
        // kImmutableZeroCopy the runtime uses our memory directly during execution.
        // We just need to destroy the event to prevent leaks.
        if (h2d_args.done_with_host_buffer != NULL) {
            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = h2d_args.done_with_host_buffer
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    uint64_t h2d_end = get_time_ns();

    // Initialize thread-local storage
    if (!g_pooled_storage_initialized) {
        g_pooled_output_lists[0] = g_pooled_output_buffers;
        g_pooled_storage_initialized = true;
    }

    // ===== PHASE 2: Execute =====
    uint64_t exec_start = get_time_ns();

    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[PJRT_BUFFER_POOL_SIZE];
    for (size_t i = 0; i < num_inputs; i++) {
        input_list[i] = pooled_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)pool->executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_pooled_output_lists
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    uint64_t exec_end = get_time_ns();

    // Destroy input buffers (consumed by execution)
    for (size_t i = 0; i < num_inputs; i++) {
        if (pooled_input_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = pooled_input_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            pooled_input_buffers[i] = NULL;
        }
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get actual number of outputs
    size_t actual_outputs = num_outputs;
    size_t cached_num_outputs = GetCachedNumOutputs(pool->executable);
    if (cached_num_outputs != (size_t)-1 && cached_num_outputs < num_outputs) {
        actual_outputs = cached_num_outputs;
    }

    // ===== PHASE 3: D2H Initiate =====
    uint64_t d2h_init_start = get_time_ns();

    for (size_t i = 0; i < actual_outputs; i++) {
        if (output_data[i] == NULL || g_pooled_output_buffers[i] == NULL) {
            g_pooled_d2h_events[i] = NULL;
            continue;
        }

        size_t byte_size = output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_pooled_output_buffers[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (g_pooled_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_pooled_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < actual_outputs; j++) {
                if (g_pooled_output_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args buf_destroy = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = g_pooled_output_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&buf_destroy);
                    g_pooled_output_buffers[j] = NULL;
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_pooled_d2h_events[i] = d2h_args.event;
    }

    uint64_t d2h_init_end = get_time_ns();

    // ===== PHASE 4: D2H Await =====
    uint64_t d2h_await_start = get_time_ns();

    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_pooled_d2h_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = g_pooled_d2h_events[i]
            };

            error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_pooled_d2h_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    uint64_t d2h_await_end = get_time_ns();

    // ===== PHASE 5: Output Buffer Cleanup =====
    uint64_t output_destroy_start = get_time_ns();

    for (size_t i = 0; i < actual_outputs; i++) {
        if (g_pooled_output_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_pooled_output_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_pooled_output_buffers[i] = NULL;
        }
    }

    uint64_t output_destroy_end = get_time_ns();

    // ===== Fill timing structure =====
    out_timing->h2d_create_ns = h2d_end - h2d_start;
    out_timing->execute_ns = exec_end - exec_start;
    out_timing->d2h_initiate_ns = d2h_init_end - d2h_init_start;
    out_timing->d2h_await_ns = d2h_await_end - d2h_await_start;
    out_timing->buffer_destroy_ns = output_destroy_end - output_destroy_start;
    out_timing->total_ns = output_destroy_end - total_start;
    out_timing->num_inputs = num_inputs;
    out_timing->num_outputs = actual_outputs;

    return result;
}

void PJRT_BufferPoolDestroy(SW_PJRT_BufferPool* pool) {
    if (pool == NULL || !pool->initialized) {
        return;
    }

    // Nothing to destroy since we don't keep buffers alive
    // (PJRT input buffers are consumed after execution)

    memset(pool, 0, sizeof(SW_PJRT_BufferPool));
}

//===------------------------------------------------------------------===//
// Enhanced Buffer Pool V2 Implementation
//===------------------------------------------------------------------===//

// Thread-local storage for V2 pooled execution
static __thread PJRT_Buffer* g_v2_input_buffers[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
static __thread PJRT_Buffer* g_v2_output_buffers[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS];
static __thread PJRT_Buffer** g_v2_output_lists[1];
static __thread PJRT_Event* g_v2_d2h_events[PJRT_BUFFER_POOL_V2_MAX_OUTPUTS];
static __thread bool g_v2_storage_initialized = false;

// Pre-allocated H2D args structures to avoid per-call initialization
static __thread PJRT_Client_BufferFromHostBuffer_Args g_v2_h2d_args[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
static __thread int64_t g_v2_input_dims[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
static __thread bool g_v2_h2d_args_initialized = false;

SW_PJRT_Error_Code PJRT_BufferPoolV2Create(
    void* client,
    void* device,
    void* executable,
    const size_t* input_sizes,
    size_t num_inputs,
    const size_t* output_sizes,
    size_t num_outputs,
    SW_PJRT_BufferPoolV2* out_pool
) {
    if (g_api == NULL || client == NULL || device == NULL || executable == NULL || out_pool == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_inputs > PJRT_BUFFER_POOL_V2_MAX_INPUTS || num_outputs > PJRT_BUFFER_POOL_V2_MAX_OUTPUTS) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize pool structure
    memset(out_pool, 0, sizeof(SW_PJRT_BufferPoolV2));
    out_pool->client = client;
    out_pool->device = device;
    out_pool->executable = executable;
    out_pool->num_inputs = num_inputs;
    out_pool->num_outputs = num_outputs;

    // Store sizes
    for (size_t i = 0; i < num_inputs; i++) {
        out_pool->input_sizes[i] = input_sizes[i];
        out_pool->input_dims[i] = (int64_t)input_sizes[i];
    }
    for (size_t i = 0; i < num_outputs; i++) {
        out_pool->output_sizes[i] = output_sizes[i];
    }

    out_pool->execution_count = 0;
    out_pool->initialized = true;

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_ExecutePooledV2(
    SW_PJRT_BufferPoolV2* pool,
    const void* const* input_data,
    void* const* output_data,
    SW_PJRT_ExecutionTiming* out_timing
) {
    if (pool == NULL || !pool->initialized || g_api == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    uint64_t total_start = out_timing ? get_time_ns() : 0;

    // Initialize thread-local storage on first use
    if (!g_v2_storage_initialized) {
        g_v2_output_lists[0] = g_v2_output_buffers;
        g_v2_storage_initialized = true;
    }

    // Initialize H2D args structures once per thread
    bool is_first_call = !g_v2_h2d_args_initialized;
    if (is_first_call) {
        for (size_t i = 0; i < PJRT_BUFFER_POOL_V2_MAX_INPUTS; i++) {
            g_v2_h2d_args[i].struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
            g_v2_h2d_args[i].extension_start = NULL;
            g_v2_h2d_args[i].type = PJRT_Buffer_Type_F32;
            g_v2_h2d_args[i].num_dims = 1;
            g_v2_h2d_args[i].byte_strides = NULL;
            g_v2_h2d_args[i].num_byte_strides = 0;
            g_v2_h2d_args[i].host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
            g_v2_h2d_args[i].memory = NULL;
            g_v2_h2d_args[i].device_layout = NULL;
        }
        g_v2_h2d_args_initialized = true;
    }

    // ===== PHASE 1: H2D Buffer Creation (optimized) =====
    uint64_t h2d_start = out_timing ? get_time_ns() : 0;

    for (size_t i = 0; i < pool->num_inputs; i++) {
        // Update only the fields that change per call
        g_v2_h2d_args[i].client = (PJRT_Client*)pool->client;
        g_v2_h2d_args[i].data = input_data[i];
        g_v2_h2d_args[i].dims = &pool->input_dims[i];
        g_v2_h2d_args[i].device = (PJRT_Device*)pool->device;
        g_v2_h2d_args[i].buffer = NULL;
        g_v2_h2d_args[i].done_with_host_buffer = NULL;

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&g_v2_h2d_args[i]);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (g_v2_input_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = g_v2_input_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_v2_input_buffers[i] = g_v2_h2d_args[i].buffer;

        // Destroy done_with_host_buffer event if present (we use zero-copy)
        if (g_v2_h2d_args[i].done_with_host_buffer != NULL) {
            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_v2_h2d_args[i].done_with_host_buffer
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    uint64_t h2d_end = out_timing ? get_time_ns() : 0;

    // ===== PHASE 2: Execute =====
    uint64_t exec_start = out_timing ? get_time_ns() : 0;

    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
    for (size_t i = 0; i < pool->num_inputs; i++) {
        input_list[i] = g_v2_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)pool->executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = pool->num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_v2_output_lists
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    uint64_t exec_end = out_timing ? get_time_ns() : 0;

    // Destroy input buffers immediately (consumed)
    for (size_t i = 0; i < pool->num_inputs; i++) {
        if (g_v2_input_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_v2_input_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_v2_input_buffers[i] = NULL;
        }
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // ===== PHASE 3: Batched D2H Initiate =====
    uint64_t d2h_init_start = out_timing ? get_time_ns() : 0;

    // Initiate ALL D2H transfers before awaiting any (enables overlap)
    for (size_t i = 0; i < pool->num_outputs; i++) {
        if (output_data[i] == NULL || g_v2_output_buffers[i] == NULL) {
            g_v2_d2h_events[i] = NULL;
            continue;
        }

        size_t byte_size = pool->output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_v2_output_buffers[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (g_v2_d2h_events[j] != NULL) {
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_v2_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < pool->num_outputs; j++) {
                if (g_v2_output_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args buf_destroy = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = g_v2_output_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&buf_destroy);
                    g_v2_output_buffers[j] = NULL;
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_v2_d2h_events[i] = d2h_args.event;
    }

    uint64_t d2h_init_end = out_timing ? get_time_ns() : 0;

    // ===== PHASE 4: Batched D2H Await =====
    uint64_t d2h_await_start = out_timing ? get_time_ns() : 0;

    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < pool->num_outputs; i++) {
        if (g_v2_d2h_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = g_v2_d2h_events[i]
            };

            error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_v2_d2h_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    uint64_t d2h_await_end = out_timing ? get_time_ns() : 0;

    // ===== PHASE 5: Output Buffer Cleanup =====
    uint64_t output_destroy_start = out_timing ? get_time_ns() : 0;

    for (size_t i = 0; i < pool->num_outputs; i++) {
        if (g_v2_output_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = g_v2_output_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            g_v2_output_buffers[i] = NULL;
        }
    }

    uint64_t output_destroy_end = out_timing ? get_time_ns() : 0;

    pool->execution_count++;

    // Fill timing structure if provided
    if (out_timing != NULL) {
        out_timing->h2d_create_ns = h2d_end - h2d_start;
        out_timing->execute_ns = exec_end - exec_start;
        out_timing->d2h_initiate_ns = d2h_init_end - d2h_init_start;
        out_timing->d2h_await_ns = d2h_await_end - d2h_await_start;
        out_timing->buffer_destroy_ns = output_destroy_end - output_destroy_start;
        out_timing->total_ns = output_destroy_end - total_start;
        out_timing->num_inputs = pool->num_inputs;
        out_timing->num_outputs = pool->num_outputs;
    }

    return result;
}

void PJRT_BufferPoolV2Destroy(SW_PJRT_BufferPoolV2* pool) {
    if (pool == NULL || !pool->initialized) {
        return;
    }
    memset(pool, 0, sizeof(SW_PJRT_BufferPoolV2));
}

//===------------------------------------------------------------------===//
// Batched Transfer API Implementation
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_BatchedH2DTransfer(
    void* client,
    void* device,
    const void* const* data_ptrs,
    const size_t* sizes,
    size_t num_buffers,
    SW_PJRT_HostBufferSemantics semantics,
    void** out_buffers
) {
    if (g_api == NULL || client == NULL || device == NULL ||
        data_ptrs == NULL || sizes == NULL || out_buffers == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_buffers > PJRT_BATCH_TRANSFER_MAX) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_HostBufferSemantics pjrt_semantics = MapToHostBufferSemantics(semantics);

    // Thread-local dims storage
    static __thread int64_t batch_dims[PJRT_BATCH_TRANSFER_MAX];

    // Create all buffers in a tight loop
    for (size_t i = 0; i < num_buffers; i++) {
        batch_dims[i] = (int64_t)sizes[i];

        PJRT_Client_BufferFromHostBuffer_Args args = {
            .struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args),
            .client = (PJRT_Client*)client,
            .data = data_ptrs[i],
            .type = PJRT_Buffer_Type_F32,
            .dims = &batch_dims[i],
            .num_dims = 1,
            .host_buffer_semantics = pjrt_semantics,
            .device = (PJRT_Device*)device
        };

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&args);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (out_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args destroy_args = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = (PJRT_Buffer*)out_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&destroy_args);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        out_buffers[i] = (void*)args.buffer;

        // Destroy done_with_host_buffer event if present
        if (args.done_with_host_buffer != NULL) {
            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = args.done_with_host_buffer
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_BatchedD2HTransfer(
    void** buffers,
    void** dest_ptrs,
    const size_t* sizes,
    size_t num_buffers,
    bool destroy_buffers
) {
    if (g_api == NULL || buffers == NULL || dest_ptrs == NULL || sizes == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (num_buffers > PJRT_BATCH_TRANSFER_MAX) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Thread-local event storage
    static __thread PJRT_Event* batch_events[PJRT_BATCH_TRANSFER_MAX];

    // Step 1: Initiate ALL transfers (enables overlap)
    for (size_t i = 0; i < num_buffers; i++) {
        if (buffers[i] == NULL || dest_ptrs[i] == NULL) {
            batch_events[i] = NULL;
            continue;
        }

        PJRT_Buffer_ToHostBuffer_Args args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = (PJRT_Buffer*)buffers[i],
            .dst = dest_ptrs[i],
            .dst_size = sizes[i],
            .host_layout = NULL
        };

        PJRT_Error* error = g_api->PJRT_Buffer_ToHostBuffer(&args);
        if (error != NULL) {
            // Cleanup events created so far
            for (size_t j = 0; j < i; j++) {
                if (batch_events[j] != NULL) {
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = batch_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        batch_events[i] = args.event;
    }

    // Step 2: Await ALL transfers
    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < num_buffers; i++) {
        if (batch_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = batch_events[i]
            };

            PJRT_Error* error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = batch_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    // Step 3: Optionally destroy source buffers
    if (destroy_buffers) {
        for (size_t i = 0; i < num_buffers; i++) {
            if (buffers[i] != NULL) {
                PJRT_Buffer_Destroy_Args destroy_args = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = (PJRT_Buffer*)buffers[i]
                };
                g_api->PJRT_Buffer_Destroy(&destroy_args);
            }
        }
    }

    return result;
}

SW_PJRT_Error_Code PJRT_ExecuteUltraFast(
    SW_PJRT_BufferPoolV2* pool,
    const void* const* input_data,
    void* const* output_data
) {
    if (pool == NULL || !pool->initialized || g_api == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Initialize thread-local storage
    if (!g_v2_storage_initialized) {
        g_v2_output_lists[0] = g_v2_output_buffers;
        g_v2_storage_initialized = true;
    }

    // Initialize H2D args structures once per thread
    if (!g_v2_h2d_args_initialized) {
        for (size_t i = 0; i < PJRT_BUFFER_POOL_V2_MAX_INPUTS; i++) {
            g_v2_h2d_args[i].struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
            g_v2_h2d_args[i].extension_start = NULL;
            g_v2_h2d_args[i].type = PJRT_Buffer_Type_F32;
            g_v2_h2d_args[i].num_dims = 1;
            g_v2_h2d_args[i].byte_strides = NULL;
            g_v2_h2d_args[i].num_byte_strides = 0;
            g_v2_h2d_args[i].host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
            g_v2_h2d_args[i].memory = NULL;
            g_v2_h2d_args[i].device_layout = NULL;
        }
        g_v2_h2d_args_initialized = true;
    }

    // ===== PHASE 1: Ultra-fast H2D (minimal per-call overhead) =====
    for (size_t i = 0; i < pool->num_inputs; i++) {
        // Only update the 4 fields that change per call
        g_v2_h2d_args[i].client = (PJRT_Client*)pool->client;
        g_v2_h2d_args[i].data = input_data[i];
        g_v2_h2d_args[i].dims = &pool->input_dims[i];
        g_v2_h2d_args[i].device = (PJRT_Device*)pool->device;

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&g_v2_h2d_args[i]);
        if (error != NULL) {
            for (size_t j = 0; j < i; j++) {
                PJRT_Buffer_Destroy_Args destroy_args = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = g_v2_input_buffers[j]
                };
                g_api->PJRT_Buffer_Destroy(&destroy_args);
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_v2_input_buffers[i] = g_v2_h2d_args[i].buffer;

        // Destroy done_with_host_buffer event
        if (g_v2_h2d_args[i].done_with_host_buffer != NULL) {
            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_v2_h2d_args[i].done_with_host_buffer
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    // ===== PHASE 2: Execute =====
    static __thread PJRT_ExecuteOptions exec_options;
    static __thread bool exec_options_initialized = false;
    if (!exec_options_initialized) {
        exec_options.struct_size = sizeof(PJRT_ExecuteOptions);
        exec_options.num_send_ops = 0;
        exec_options.num_recv_ops = 0;
        exec_options.launch_id = 0;
        exec_options_initialized = true;
    }

    PJRT_Buffer* input_list[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
    for (size_t i = 0; i < pool->num_inputs; i++) {
        input_list[i] = g_v2_input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)pool->executable,
        .options = &exec_options,
        .num_devices = 1,
        .num_args = pool->num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_v2_output_lists
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    // Destroy input buffers
    for (size_t i = 0; i < pool->num_inputs; i++) {
        PJRT_Buffer_Destroy_Args destroy_args = {
            .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
            .buffer = g_v2_input_buffers[i]
        };
        g_api->PJRT_Buffer_Destroy(&destroy_args);
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // ===== PHASE 3: Batched D2H (initiate all, then await all) =====
    // Initiate all transfers
    for (size_t i = 0; i < pool->num_outputs; i++) {
        size_t byte_size = pool->output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = g_v2_output_buffers[i],
            .dst = output_data[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // On error, cleanup and return
            for (size_t j = 0; j < i; j++) {
                if (g_v2_d2h_events[j]) {
                    PJRT_Event_Await_Args await = {
                        .struct_size = sizeof(PJRT_Event_Await_Args),
                        .event = g_v2_d2h_events[j]
                    };
                    g_api->PJRT_Event_Await(&await);
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = g_v2_d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < pool->num_outputs; j++) {
                PJRT_Buffer_Destroy_Args buf_destroy = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = g_v2_output_buffers[j]
                };
                g_api->PJRT_Buffer_Destroy(&buf_destroy);
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        g_v2_d2h_events[i] = d2h_args.event;
    }

    // Await all transfers and destroy events
    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;
    for (size_t i = 0; i < pool->num_outputs; i++) {
        PJRT_Event_Await_Args await_args = {
            .struct_size = sizeof(PJRT_Event_Await_Args),
            .event = g_v2_d2h_events[i]
        };

        error = g_api->PJRT_Event_Await(&await_args);
        if (error != NULL && result == SW_PJRT_Error_OK) {
            result = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
        }

        PJRT_Event_Destroy_Args ev_destroy = {
            .struct_size = sizeof(PJRT_Event_Destroy_Args),
            .event = g_v2_d2h_events[i]
        };
        g_api->PJRT_Event_Destroy(&ev_destroy);
    }

    // Destroy output buffers
    for (size_t i = 0; i < pool->num_outputs; i++) {
        PJRT_Buffer_Destroy_Args destroy_args = {
            .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
            .buffer = g_v2_output_buffers[i]
        };
        g_api->PJRT_Buffer_Destroy(&destroy_args);
    }

    pool->execution_count++;

    return result;
}

//===------------------------------------------------------------------===//
// Pipelined Execution Implementation
//===------------------------------------------------------------------===//

SW_PJRT_Error_Code PJRT_PipelineCreate(
    SW_PJRT_BufferPoolV2* pool,
    size_t depth,
    SW_PJRT_ExecutionPipeline* out_pipeline
) {
    if (pool == NULL || !pool->initialized || out_pipeline == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (depth < 1 || depth > PJRT_PIPELINE_MAX_DEPTH) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    memset(out_pipeline, 0, sizeof(SW_PJRT_ExecutionPipeline));
    out_pipeline->pool = pool;
    out_pipeline->depth = depth;
    out_pipeline->head = 0;
    out_pipeline->tail = 0;
    out_pipeline->in_flight = 0;
    out_pipeline->total_submissions = 0;
    out_pipeline->total_completions = 0;
    out_pipeline->total_wait_ns = 0;
    out_pipeline->initialized = true;

    // Initialize all slots as empty
    for (size_t i = 0; i < PJRT_PIPELINE_MAX_DEPTH; i++) {
        out_pipeline->slots[i].state = SW_PJRT_PipelineSlot_Empty;
    }

    return SW_PJRT_Error_OK;
}

// Helper: Complete the oldest slot (await D2H and cleanup)
static SW_PJRT_Error_Code pipeline_complete_slot(
    SW_PJRT_ExecutionPipeline* pipeline,
    SW_PJRT_PipelineSlot* slot
) {
    if (slot->state != SW_PJRT_PipelineSlot_Executing) {
        return SW_PJRT_Error_INTERNAL;
    }

    SW_PJRT_BufferPoolV2* pool = pipeline->pool;
    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;

    // Await all D2H events
    for (size_t i = 0; i < pool->num_outputs; i++) {
        if (slot->d2h_events[i] != NULL) {
            PJRT_Event_Await_Args await_args = {
                .struct_size = sizeof(PJRT_Event_Await_Args),
                .event = (PJRT_Event*)slot->d2h_events[i]
            };

            PJRT_Error* error = g_api->PJRT_Event_Await(&await_args);
            if (error != NULL && result == SW_PJRT_Error_OK) {
                result = PJRT_GetErrorCode(error);
                PJRT_DestroyError(error);
            }

            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = (PJRT_Event*)slot->d2h_events[i]
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
            slot->d2h_events[i] = NULL;
        }
    }

    // Destroy output buffers
    for (size_t i = 0; i < pool->num_outputs; i++) {
        if (slot->output_buffers[i] != NULL) {
            PJRT_Buffer_Destroy_Args destroy_args = {
                .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                .buffer = (PJRT_Buffer*)slot->output_buffers[i]
            };
            g_api->PJRT_Buffer_Destroy(&destroy_args);
            slot->output_buffers[i] = NULL;
        }
    }

    slot->state = SW_PJRT_PipelineSlot_Empty;
    return result;
}

SW_PJRT_Error_Code PJRT_PipelineSubmit(
    SW_PJRT_ExecutionPipeline* pipeline,
    const void* const* input_data,
    void* const* output_data
) {
    if (pipeline == NULL || !pipeline->initialized || g_api == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    SW_PJRT_BufferPoolV2* pool = pipeline->pool;

    // If pipeline is full, wait for the oldest slot
    if (pipeline->in_flight >= pipeline->depth) {
        uint64_t wait_start = get_time_ns();

        SW_PJRT_PipelineSlot* oldest_slot = &pipeline->slots[pipeline->tail];
        SW_PJRT_Error_Code err = pipeline_complete_slot(pipeline, oldest_slot);
        if (err != SW_PJRT_Error_OK) {
            return err;
        }

        pipeline->tail = (pipeline->tail + 1) % pipeline->depth;
        pipeline->in_flight--;
        pipeline->total_completions++;
        pipeline->total_wait_ns += get_time_ns() - wait_start;
    }

    // Get the next available slot
    SW_PJRT_PipelineSlot* slot = &pipeline->slots[pipeline->head];
    slot->submit_time_ns = get_time_ns();

    // Store user's output pointers
    for (size_t i = 0; i < pool->num_outputs; i++) {
        slot->user_output_ptrs[i] = output_data[i];
    }

    // Initialize thread-local H2D args if needed
    if (!g_v2_h2d_args_initialized) {
        for (size_t i = 0; i < PJRT_BUFFER_POOL_V2_MAX_INPUTS; i++) {
            g_v2_h2d_args[i].struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
            g_v2_h2d_args[i].extension_start = NULL;
            g_v2_h2d_args[i].type = PJRT_Buffer_Type_F32;
            g_v2_h2d_args[i].num_dims = 1;
            g_v2_h2d_args[i].byte_strides = NULL;
            g_v2_h2d_args[i].num_byte_strides = 0;
            g_v2_h2d_args[i].host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
            g_v2_h2d_args[i].memory = NULL;
            g_v2_h2d_args[i].device_layout = NULL;
        }
        g_v2_h2d_args_initialized = true;
    }

    // ===== PHASE 1: H2D Buffer Creation =====
    PJRT_Buffer* input_buffers[PJRT_BUFFER_POOL_V2_MAX_INPUTS];

    for (size_t i = 0; i < pool->num_inputs; i++) {
        g_v2_h2d_args[i].client = (PJRT_Client*)pool->client;
        g_v2_h2d_args[i].data = input_data[i];
        g_v2_h2d_args[i].dims = &pool->input_dims[i];
        g_v2_h2d_args[i].device = (PJRT_Device*)pool->device;

        PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&g_v2_h2d_args[i]);
        if (error != NULL) {
            for (size_t j = 0; j < i; j++) {
                PJRT_Buffer_Destroy_Args destroy_args = {
                    .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                    .buffer = input_buffers[j]
                };
                g_api->PJRT_Buffer_Destroy(&destroy_args);
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        input_buffers[i] = g_v2_h2d_args[i].buffer;

        if (g_v2_h2d_args[i].done_with_host_buffer != NULL) {
            PJRT_Event_Destroy_Args ev_destroy = {
                .struct_size = sizeof(PJRT_Event_Destroy_Args),
                .event = g_v2_h2d_args[i].done_with_host_buffer
            };
            g_api->PJRT_Event_Destroy(&ev_destroy);
        }
    }

    // ===== PHASE 2: Execute =====
    if (!g_v2_storage_initialized) {
        g_v2_output_lists[0] = g_v2_output_buffers;
        g_v2_storage_initialized = true;
    }

    PJRT_ExecuteOptions execute_options = {
        .struct_size = sizeof(PJRT_ExecuteOptions),
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0
    };

    PJRT_Buffer* input_list[PJRT_BUFFER_POOL_V2_MAX_INPUTS];
    for (size_t i = 0; i < pool->num_inputs; i++) {
        input_list[i] = input_buffers[i];
    }
    PJRT_Buffer*const* input_list_ptr = input_list;
    PJRT_Buffer*const*const* argument_lists_ptr = &input_list_ptr;

    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = sizeof(PJRT_LoadedExecutable_Execute_Args),
        .executable = (PJRT_LoadedExecutable*)pool->executable,
        .options = &execute_options,
        .num_devices = 1,
        .num_args = pool->num_inputs,
        .argument_lists = (PJRT_Buffer****)argument_lists_ptr,
        .output_lists = g_v2_output_lists
    };

    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&exec_args);

    // Destroy input buffers (consumed)
    for (size_t i = 0; i < pool->num_inputs; i++) {
        PJRT_Buffer_Destroy_Args destroy_args = {
            .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
            .buffer = input_buffers[i]
        };
        g_api->PJRT_Buffer_Destroy(&destroy_args);
    }

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Save output buffer handles to slot
    for (size_t i = 0; i < pool->num_outputs; i++) {
        slot->output_buffers[i] = g_v2_output_buffers[i];
        g_v2_output_buffers[i] = NULL;  // Transfer ownership to slot
    }

    // ===== PHASE 3: Initiate D2H (non-blocking) =====
    for (size_t i = 0; i < pool->num_outputs; i++) {
        size_t byte_size = pool->output_sizes[i] * sizeof(float);
        PJRT_Buffer_ToHostBuffer_Args d2h_args = {
            .struct_size = sizeof(PJRT_Buffer_ToHostBuffer_Args),
            .src = (PJRT_Buffer*)slot->output_buffers[i],
            .dst = slot->user_output_ptrs[i],
            .dst_size = byte_size,
            .host_layout = NULL
        };

        error = g_api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (error != NULL) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                if (slot->d2h_events[j] != NULL) {
                    PJRT_Event_Await_Args await = {
                        .struct_size = sizeof(PJRT_Event_Await_Args),
                        .event = (PJRT_Event*)slot->d2h_events[j]
                    };
                    g_api->PJRT_Event_Await(&await);
                    PJRT_Event_Destroy_Args ev_destroy = {
                        .struct_size = sizeof(PJRT_Event_Destroy_Args),
                        .event = (PJRT_Event*)slot->d2h_events[j]
                    };
                    g_api->PJRT_Event_Destroy(&ev_destroy);
                }
            }
            for (size_t j = 0; j < pool->num_outputs; j++) {
                if (slot->output_buffers[j] != NULL) {
                    PJRT_Buffer_Destroy_Args buf_destroy = {
                        .struct_size = sizeof(PJRT_Buffer_Destroy_Args),
                        .buffer = (PJRT_Buffer*)slot->output_buffers[j]
                    };
                    g_api->PJRT_Buffer_Destroy(&buf_destroy);
                    slot->output_buffers[j] = NULL;
                }
            }
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }

        slot->d2h_events[i] = d2h_args.event;
    }

    slot->state = SW_PJRT_PipelineSlot_Executing;
    pipeline->head = (pipeline->head + 1) % pipeline->depth;
    pipeline->in_flight++;
    pipeline->total_submissions++;
    pool->execution_count++;

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_PipelineAwaitOne(
    SW_PJRT_ExecutionPipeline* pipeline,
    void*** out_output_data
) {
    if (pipeline == NULL || !pipeline->initialized) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    if (pipeline->in_flight == 0) {
        return SW_PJRT_Error_NOT_FOUND;
    }

    SW_PJRT_PipelineSlot* slot = &pipeline->slots[pipeline->tail];

    // Complete the slot
    SW_PJRT_Error_Code err = pipeline_complete_slot(pipeline, slot);

    // Return user's output pointers if requested
    if (out_output_data != NULL) {
        *out_output_data = slot->user_output_ptrs;
    }

    pipeline->tail = (pipeline->tail + 1) % pipeline->depth;
    pipeline->in_flight--;
    pipeline->total_completions++;

    return err;
}

SW_PJRT_Error_Code PJRT_PipelineFlush(
    SW_PJRT_ExecutionPipeline* pipeline
) {
    if (pipeline == NULL || !pipeline->initialized) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    SW_PJRT_Error_Code result = SW_PJRT_Error_OK;

    while (pipeline->in_flight > 0) {
        SW_PJRT_Error_Code err = PJRT_PipelineAwaitOne(pipeline, NULL);
        if (err != SW_PJRT_Error_OK && result == SW_PJRT_Error_OK) {
            result = err;
        }
    }

    return result;
}

void PJRT_PipelineGetStats(
    SW_PJRT_ExecutionPipeline* pipeline,
    uint64_t* out_submissions,
    uint64_t* out_completions,
    double* out_avg_wait_us
) {
    if (pipeline == NULL || !pipeline->initialized) {
        return;
    }

    if (out_submissions) *out_submissions = pipeline->total_submissions;
    if (out_completions) *out_completions = pipeline->total_completions;
    if (out_avg_wait_us) {
        if (pipeline->total_completions > 0) {
            *out_avg_wait_us = (double)pipeline->total_wait_ns / (double)pipeline->total_completions / 1000.0;
        } else {
            *out_avg_wait_us = 0.0;
        }
    }
}

void PJRT_PipelineDestroy(SW_PJRT_ExecutionPipeline* pipeline) {
    if (pipeline == NULL || !pipeline->initialized) {
        return;
    }

    // Flush any pending executions
    PJRT_PipelineFlush(pipeline);

    memset(pipeline, 0, sizeof(SW_PJRT_ExecutionPipeline));
}
