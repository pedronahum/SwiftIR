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

    g_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
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

SW_PJRT_Error_Code PJRT_CreateBuffer(
    void* client,
    const void* data,
    SW_PJRT_Buffer_Type type,
    const int64_t* dims,
    size_t num_dims,
    void* device,
    void** out_buffer
) {
    if (g_api == NULL || client == NULL || data == NULL || dims == NULL || device == NULL) {
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Calculate total number of elements
    int64_t num_elements = 1;
    for (size_t i = 0; i < num_dims; i++) {
        num_elements *= dims[i];
    }

    // Create buffer from host memory
    PJRT_Client_BufferFromHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = (PJRT_Client*)client;
    args.data = data;
    args.type = MapToPJRTElementType(type);
    args.dims = dims;
    args.num_dims = num_dims;
    args.byte_strides = NULL;  // Use default row-major layout
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = (PJRT_Device*)device;

    PJRT_Error* error = g_api->PJRT_Client_BufferFromHostBuffer(&args);
    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_CreateBuffer failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    *out_buffer = (void*)args.buffer;
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
    fprintf(stderr, "DEBUG: PJRT_BufferToHost called, buffer=%p, out_data=%p, size=%zu\n", buffer, out_data, data_size);
    fflush(stderr);

    if (g_api == NULL || buffer == NULL || out_data == NULL) {
        fprintf(stderr, "DEBUG: PJRT_BufferToHost validation failed\n");
        fflush(stderr);
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    fprintf(stderr, "DEBUG: Preparing ToHostBuffer args\n");
    fflush(stderr);

    // Transfer buffer from device to host (asynchronous)
    PJRT_Buffer_ToHostBuffer_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.src = (PJRT_Buffer*)buffer;
    args.dst = out_data;
    args.dst_size = data_size;
    args.host_layout = NULL;  // Use default layout

    fprintf(stderr, "DEBUG: About to call PJRT_Buffer_ToHostBuffer\n");
    fflush(stderr);

    PJRT_Error* error = g_api->PJRT_Buffer_ToHostBuffer(&args);

    fprintf(stderr, "DEBUG: PJRT_Buffer_ToHostBuffer returned, error=%p\n", (void*)error);
    fflush(stderr);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_BufferToHost failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    fprintf(stderr, "DEBUG: Checking if event needs await, event=%p\n", (void*)args.event);
    fflush(stderr);

    // Wait for the transfer to complete
    if (args.event != NULL) {
        fprintf(stderr, "DEBUG: About to call PJRT_Event_Await\n");
        fflush(stderr);

        PJRT_Event_Await_Args await_args;
        memset(&await_args, 0, sizeof(await_args));
        await_args.struct_size = sizeof(await_args);
        await_args.event = args.event;

        error = g_api->PJRT_Event_Await(&await_args);

        fprintf(stderr, "DEBUG: PJRT_Event_Await returned, error=%p\n", (void*)error);
        fflush(stderr);

        if (error != NULL) {
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            fprintf(stderr, "PJRT_Event_Await failed: %s\n", PJRT_GetErrorMessage(error));
            PJRT_DestroyError(error);
            return code;
        }
    }

    fprintf(stderr, "DEBUG: PJRT_BufferToHost completed successfully\n");
    fflush(stderr);

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_GetBufferDimensions(
    void* buffer,
    const int64_t** out_dims,
    size_t* out_num_dims
) {
    fprintf(stderr, "DEBUG: PJRT_GetBufferDimensions called, buffer=%p\n", buffer);
    fflush(stderr);

    if (g_api == NULL || buffer == NULL || out_dims == NULL || out_num_dims == NULL) {
        fprintf(stderr, "DEBUG: PJRT_GetBufferDimensions validation failed\n");
        fflush(stderr);
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    fprintf(stderr, "DEBUG: About to call g_api->PJRT_Buffer_Dimensions\n");
    fflush(stderr);

    PJRT_Buffer_Dimensions_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_Dimensions(&args);

    fprintf(stderr, "DEBUG: PJRT_Buffer_Dimensions returned, error=%p\n", (void*)error);
    fflush(stderr);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_GetBufferDimensions failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    *out_dims = args.dims;
    *out_num_dims = args.num_dims;

    fprintf(stderr, "DEBUG: PJRT_GetBufferDimensions completed successfully, num_dims=%zu\n", *out_num_dims);
    fflush(stderr);

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_GetBufferOnDeviceSizeInBytes(
    void* buffer,
    size_t* out_size
) {
    fprintf(stderr, "DEBUG: PJRT_GetBufferOnDeviceSizeInBytes called, buffer=%p\n", buffer);
    fflush(stderr);

    if (g_api == NULL || buffer == NULL || out_size == NULL) {
        fprintf(stderr, "DEBUG: PJRT_GetBufferOnDeviceSizeInBytes validation failed\n");
        fflush(stderr);
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    fprintf(stderr, "DEBUG: About to call g_api->PJRT_Buffer_OnDeviceSizeInBytes\n");
    fflush(stderr);

    PJRT_Buffer_OnDeviceSizeInBytes_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_OnDeviceSizeInBytes(&args);

    fprintf(stderr, "DEBUG: PJRT_Buffer_OnDeviceSizeInBytes returned, error=%p\n", (void*)error);
    fflush(stderr);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_GetBufferOnDeviceSizeInBytes failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    *out_size = args.on_device_size_in_bytes;

    fprintf(stderr, "DEBUG: PJRT_GetBufferOnDeviceSizeInBytes completed successfully, size=%zu\n", *out_size);
    fflush(stderr);

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_Buffer_IncRefCount(void* buffer) {
    if (g_api == NULL || buffer == NULL) {
        fprintf(stderr, "ERROR: PJRT_Buffer_IncRefCount - invalid arguments\n");
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_IncreaseExternalReferenceCount_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_IncreaseExternalReferenceCount(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_Buffer_IncRefCount failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    return SW_PJRT_Error_OK;
}

SW_PJRT_Error_Code PJRT_Buffer_DecRefCount(void* buffer) {
    if (g_api == NULL || buffer == NULL) {
        fprintf(stderr, "ERROR: PJRT_Buffer_DecRefCount - invalid arguments\n");
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    PJRT_Buffer_DecreaseExternalReferenceCount_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = (PJRT_Buffer*)buffer;

    PJRT_Error* error = g_api->PJRT_Buffer_DecreaseExternalReferenceCount(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_Buffer_DecRefCount failed: %s\n", PJRT_GetErrorMessage(error));
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

    // Debug: print struct_size
    fprintf(stderr, "DEBUG: PJRT_Program struct_size = %zu (expected 48)\n", program.struct_size);

    // Create CompileOptionsProto using C++ helper that properly constructs
    // the protobuf message with XLA's protobuf library
    size_t compile_opts_size = 0;
    char* compile_opts = PJRT_CreateCompileOptions(1, 1, &compile_opts_size);
    if (compile_opts == NULL) {
        fprintf(stderr, "Failed to create compile options protobuf\n");
        return SW_PJRT_Error_INTERNAL;
    }

    fprintf(stderr, "DEBUG: CompileOptions protobuf size = %zu bytes\n", compile_opts_size);

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
        fprintf(stderr, "PJRT_Compile failed: %s\n", PJRT_GetErrorMessage(error));
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
            fprintf(stderr, "Warning: PJRT_LoadedExecutable_Destroy failed: %s\n",
                    PJRT_GetErrorMessage(error));
            PJRT_DestroyError(error);
        }
    }
}

SW_PJRT_Error_Code PJRT_ExecuteWrapper(
    void* executable,
    void** inputs,
    size_t num_inputs,
    void*** out_outputs,
    size_t* out_num_outputs
) {
    fprintf(stderr, "DEBUG: PJRT_ExecuteWrapper called with %zu inputs\n", num_inputs);

    if (g_api == NULL || executable == NULL) {
        fprintf(stderr, "DEBUG: Invalid arguments - g_api=%p, executable=%p\n", g_api, executable);
        return SW_PJRT_Error_INVALID_ARGUMENT;
    }

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = NULL;
    if (num_inputs > 0 && inputs != NULL) {
        input_buffers = (PJRT_Buffer**)inputs;
        fprintf(stderr, "DEBUG: Input buffers prepared, first buffer=%p\n", input_buffers[0]);
    } else {
        fprintf(stderr, "DEBUG: No input buffers\n");
    }

    // Prepare execution arguments
    PJRT_ExecuteOptions execute_options;
    memset(&execute_options, 0, sizeof(execute_options));
    execute_options.struct_size = sizeof(execute_options);
    execute_options.num_send_ops = 0;
    execute_options.num_recv_ops = 0;
    execute_options.launch_id = 0;
    fprintf(stderr, "DEBUG: Execute options prepared\n");

    // Allocate output buffer array (caller must allocate for PJRT API)
    // We need to allocate space for up to 16 output buffers per device
    const size_t max_outputs = 16;
    PJRT_Buffer** output_buffer_array = (PJRT_Buffer**)malloc(max_outputs * sizeof(PJRT_Buffer*));
    PJRT_Buffer*** output_lists_array = (PJRT_Buffer***)malloc(sizeof(PJRT_Buffer**));
    output_lists_array[0] = output_buffer_array;

    PJRT_LoadedExecutable_Execute_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.executable = (PJRT_LoadedExecutable*)executable;
    args.options = &execute_options;
    args.num_devices = 1;  // Single device execution
    args.num_args = num_inputs;
    args.output_lists = output_lists_array;  // Pre-allocated output array

    // Workaround for const qualifier warnings - PJRT API shouldn't modify input lists
    PJRT_Buffer*const* input_list_const = (PJRT_Buffer*const*)input_buffers;
    PJRT_Buffer*const*const* argument_lists_const = (PJRT_Buffer*const*const*)&input_list_const;
    args.argument_lists = (PJRT_Buffer****)argument_lists_const;

    fprintf(stderr, "DEBUG: About to call PJRT_LoadedExecutable_Execute\n");

    // Execute
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);

    fprintf(stderr, "DEBUG: PJRT_LoadedExecutable_Execute returned, error=%p\n", error);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        fprintf(stderr, "PJRT_Execute failed: %s\n", PJRT_GetErrorMessage(error));
        PJRT_DestroyError(error);
        return code;
    }

    fprintf(stderr, "DEBUG: Processing execution outputs\n");

    // Get the executable from loaded executable to query outputs
    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    memset(&get_exec_args, 0, sizeof(get_exec_args));
    get_exec_args.struct_size = sizeof(get_exec_args);
    get_exec_args.loaded_executable = (PJRT_LoadedExecutable*)executable;

    fprintf(stderr, "DEBUG: About to call PJRT_LoadedExecutable_GetExecutable\n");
    error = g_api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
    fprintf(stderr, "DEBUG: PJRT_LoadedExecutable_GetExecutable returned, error=%p\n", error);

    if (error != NULL) {
        fprintf(stderr, "DEBUG: Failed to get executable, setting num_outputs=0\n");
        PJRT_DestroyError(error);
        *out_num_outputs = 0;
    } else {
        // Get number of outputs
        PJRT_Executable_NumOutputs_Args num_outputs_args;
        memset(&num_outputs_args, 0, sizeof(num_outputs_args));
        num_outputs_args.struct_size = sizeof(num_outputs_args);
        num_outputs_args.executable = get_exec_args.executable;

        fprintf(stderr, "DEBUG: About to call PJRT_Executable_NumOutputs\n");
        error = g_api->PJRT_Executable_NumOutputs(&num_outputs_args);
        fprintf(stderr, "DEBUG: PJRT_Executable_NumOutputs returned, error=%p\n", error);

        if (error != NULL) {
            fprintf(stderr, "DEBUG: Failed to get num outputs, setting num_outputs=0\n");
            PJRT_DestroyError(error);
            *out_num_outputs = 0;
        } else {
            *out_num_outputs = num_outputs_args.num_outputs;
            fprintf(stderr, "DEBUG: Number of outputs = %zu\n", *out_num_outputs);
        }
    }

    fprintf(stderr, "DEBUG: Checking output_lists: output_lists=%p\n", args.output_lists);
    if (args.output_lists) {
        fprintf(stderr, "DEBUG: output_lists[0]=%p\n", args.output_lists[0]);
    }

    // Return output buffers (caller takes ownership of the arrays)
    if (args.output_lists && args.output_lists[0]) {
        *out_outputs = (void**)args.output_lists[0];
        fprintf(stderr, "DEBUG: Set out_outputs to %p\n", *out_outputs);
    } else {
        *out_outputs = NULL;
        *out_num_outputs = 0;
        fprintf(stderr, "DEBUG: No output buffers available\n");
        // Free our allocations if execution failed
        free(output_buffer_array);
        free(output_lists_array);
    }

    // Don't free output_buffer_array or output_lists_array here -
    // they're returned to the caller via out_outputs

    fprintf(stderr, "DEBUG: PJRT_ExecuteWrapper completed successfully\n");
    return SW_PJRT_Error_OK;
}
