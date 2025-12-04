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

        if (error != NULL) {
            SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
            PJRT_DestroyError(error);
            return code;
        }
    }

    return SW_PJRT_Error_OK;
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
    char* compile_opts = PJRT_CreateCompileOptions(1, 1, &compile_opts_size);
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

    // Cast input buffers to PJRT_Buffer** array
    PJRT_Buffer** input_buffers = NULL;
    if (num_inputs > 0 && inputs != NULL) {
        input_buffers = (PJRT_Buffer**)inputs;
    }

    // Prepare execution arguments
    PJRT_ExecuteOptions execute_options;
    memset(&execute_options, 0, sizeof(execute_options));
    execute_options.struct_size = sizeof(execute_options);
    execute_options.num_send_ops = 0;
    execute_options.num_recv_ops = 0;
    execute_options.launch_id = 0;

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

    // Execute
    PJRT_Error* error = g_api->PJRT_LoadedExecutable_Execute(&args);

    if (error != NULL) {
        SW_PJRT_Error_Code code = PJRT_GetErrorCode(error);
        PJRT_DestroyError(error);
        return code;
    }

    // Get the executable from loaded executable to query outputs
    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    memset(&get_exec_args, 0, sizeof(get_exec_args));
    get_exec_args.struct_size = sizeof(get_exec_args);
    get_exec_args.loaded_executable = (PJRT_LoadedExecutable*)executable;

    error = g_api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args);

    if (error != NULL) {
        PJRT_DestroyError(error);
        *out_num_outputs = 0;
    } else {
        // Get number of outputs
        PJRT_Executable_NumOutputs_Args num_outputs_args;
        memset(&num_outputs_args, 0, sizeof(num_outputs_args));
        num_outputs_args.struct_size = sizeof(num_outputs_args);
        num_outputs_args.executable = get_exec_args.executable;

        error = g_api->PJRT_Executable_NumOutputs(&num_outputs_args);

        if (error != NULL) {
            PJRT_DestroyError(error);
            *out_num_outputs = 0;
        } else {
            *out_num_outputs = num_outputs_args.num_outputs;
        }
    }

    // Return output buffers (caller takes ownership of the arrays)
    if (args.output_lists && args.output_lists[0]) {
        *out_outputs = (void**)args.output_lists[0];
    } else {
        *out_outputs = NULL;
        *out_num_outputs = 0;
        // Free our allocations if execution failed
        free(output_buffer_array);
        free(output_lists_array);
    }

    // Don't free output_buffer_array or output_lists_array here -
    // they're returned to the caller via out_outputs

    return SW_PJRT_Error_OK;
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
