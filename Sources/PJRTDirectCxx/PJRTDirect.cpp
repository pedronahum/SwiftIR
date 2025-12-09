//===-- PJRTDirect.cpp - Direct PJRT C++ Wrapper ---------*- C++ -*-===//
//
// SwiftIR - Optimized PJRT wrapper using C++ for better performance
//
// This implementation provides a clean C API for Swift while using C++
// internally to optimize buffer management and reduce overhead.
//
//===----------------------------------------------------------------===//

#include "include/PJRTDirect.h"
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <atomic>

// Include the PJRT C API header
extern "C" {
#include "../../PJRTCWrappers/include/pjrt_c_api.h"
}

//===----------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------===//

static const char* kVersion = "1.0.0";

//===----------------------------------------------------------------===//
// Internal Structures
//===----------------------------------------------------------------===//

/// Internal client structure wrapping PJRT client and API
struct PJRTDirect_Client_s {
    void* plugin_handle;           // dlopen handle
    const PJRT_Api* api;           // Cached API struct
    PJRT_Client* client;           // PJRT client
    PJRTDirect_Backend backend;    // Backend type
    std::vector<PJRT_Device*> devices;  // Cached device list
    std::string platform_name;     // Cached platform name
    std::string platform_version;  // Cached platform version
};

/// Internal device structure (thin wrapper, non-owning)
struct PJRTDirect_Device_s {
    PJRT_Device* device;
    PJRTDirect_Client_s* client;
    int32_t index;
};

/// Internal executable structure
struct PJRTDirect_Executable_s {
    PJRT_LoadedExecutable* executable;
    PJRTDirect_Client_s* client;
    int32_t num_inputs;
    int32_t num_outputs;
};

/// Internal execution context for repeated execution
struct PJRTDirect_ExecContext_s {
    PJRTDirect_Executable_s* exec;
    PJRTDirect_Device_s* device;

    // Cached sizes
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;

    // Pre-allocated argument structs (to avoid memset on hot path)
    PJRT_Client_BufferFromHostBuffer_Args* h2d_args;
    PJRT_LoadedExecutable_Execute_Args* exec_args;
    PJRT_Buffer_ToHostBuffer_Args* d2h_args;
};

//===----------------------------------------------------------------===//
// Global State
//===----------------------------------------------------------------===//

// Default plugin paths per backend
static std::unordered_map<int, std::string> g_default_paths;
static std::mutex g_paths_mutex;

// Environment variable names for plugin paths
static const char* kEnvCpuPlugin = "PJRT_CPU_PLUGIN_PATH";
static const char* kEnvGpuCudaPlugin = "PJRT_GPU_CUDA_PLUGIN_PATH";
static const char* kEnvGpuRocmPlugin = "PJRT_GPU_ROCM_PLUGIN_PATH";
static const char* kEnvTpuPlugin = "PJRT_TPU_PLUGIN_PATH";

//===----------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------===//

static inline uint64_t now_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

static PJRTDirect_Status make_status(PJRTDirect_ErrorCode code, const char* msg = nullptr) {
    PJRTDirect_Status status;
    status.code = code;
    if (msg) {
        status.message = strdup(msg);
    } else {
        status.message = nullptr;
    }
    return status;
}

static PJRTDirect_Status ok_status() {
    return make_status(PJRT_DIRECT_OK, nullptr);
}

static PJRTDirect_Status from_pjrt_error(const PJRT_Api* api, PJRT_Error* error) {
    if (!error) {
        return ok_status();
    }

    // Get error code
    PJRT_Error_GetCode_Args code_args;
    code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.extension_start = nullptr;
    code_args.error = error;
    api->PJRT_Error_GetCode(&code_args);

    // Get error message
    PJRT_Error_Message_Args msg_args;
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.extension_start = nullptr;
    msg_args.error = error;
    api->PJRT_Error_Message(&msg_args);

    // Create message copy
    char* msg_copy = (char*)malloc(msg_args.message_size + 1);
    memcpy(msg_copy, msg_args.message, msg_args.message_size);
    msg_copy[msg_args.message_size] = '\0';

    // Map PJRT error code to our error code
    PJRTDirect_ErrorCode our_code;
    switch (code_args.code) {
        case PJRT_Error_Code_INVALID_ARGUMENT: our_code = PJRT_DIRECT_ERROR_INVALID_ARGUMENT; break;
        case PJRT_Error_Code_NOT_FOUND: our_code = PJRT_DIRECT_ERROR_NOT_FOUND; break;
        case PJRT_Error_Code_ALREADY_EXISTS: our_code = PJRT_DIRECT_ERROR_ALREADY_EXISTS; break;
        case PJRT_Error_Code_PERMISSION_DENIED: our_code = PJRT_DIRECT_ERROR_PERMISSION_DENIED; break;
        case PJRT_Error_Code_RESOURCE_EXHAUSTED: our_code = PJRT_DIRECT_ERROR_RESOURCE_EXHAUSTED; break;
        case PJRT_Error_Code_FAILED_PRECONDITION: our_code = PJRT_DIRECT_ERROR_FAILED_PRECONDITION; break;
        case PJRT_Error_Code_ABORTED: our_code = PJRT_DIRECT_ERROR_ABORTED; break;
        case PJRT_Error_Code_OUT_OF_RANGE: our_code = PJRT_DIRECT_ERROR_OUT_OF_RANGE; break;
        case PJRT_Error_Code_UNIMPLEMENTED: our_code = PJRT_DIRECT_ERROR_UNIMPLEMENTED; break;
        case PJRT_Error_Code_INTERNAL: our_code = PJRT_DIRECT_ERROR_INTERNAL; break;
        case PJRT_Error_Code_UNAVAILABLE: our_code = PJRT_DIRECT_ERROR_UNAVAILABLE; break;
        case PJRT_Error_Code_DATA_LOSS: our_code = PJRT_DIRECT_ERROR_DATA_LOSS; break;
        case PJRT_Error_Code_UNAUTHENTICATED: our_code = PJRT_DIRECT_ERROR_UNAUTHENTICATED; break;
        default: our_code = PJRT_DIRECT_ERROR_UNKNOWN; break;
    }

    // Destroy PJRT error
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);

    PJRTDirect_Status status;
    status.code = our_code;
    status.message = msg_copy;
    return status;
}

static PJRT_Buffer_Type dtype_to_pjrt(PJRTDirect_DType dtype) {
    switch (dtype) {
        case PJRT_DTYPE_F32: return PJRT_Buffer_Type_F32;
        case PJRT_DTYPE_F64: return PJRT_Buffer_Type_F64;
        case PJRT_DTYPE_F16: return PJRT_Buffer_Type_F16;
        case PJRT_DTYPE_BF16: return PJRT_Buffer_Type_BF16;
        case PJRT_DTYPE_I8: return PJRT_Buffer_Type_S8;
        case PJRT_DTYPE_I16: return PJRT_Buffer_Type_S16;
        case PJRT_DTYPE_I32: return PJRT_Buffer_Type_S32;
        case PJRT_DTYPE_I64: return PJRT_Buffer_Type_S64;
        case PJRT_DTYPE_U8: return PJRT_Buffer_Type_U8;
        case PJRT_DTYPE_U16: return PJRT_Buffer_Type_U16;
        case PJRT_DTYPE_U32: return PJRT_Buffer_Type_U32;
        case PJRT_DTYPE_U64: return PJRT_Buffer_Type_U64;
        case PJRT_DTYPE_BOOL: return PJRT_Buffer_Type_PRED;
        case PJRT_DTYPE_C64: return PJRT_Buffer_Type_C64;
        case PJRT_DTYPE_C128: return PJRT_Buffer_Type_C128;
        default: return PJRT_Buffer_Type_F32;
    }
}

static PJRT_HostBufferSemantics semantics_to_pjrt(PJRTDirect_HostBufferSemantics sem) {
    switch (sem) {
        case PJRT_HOST_BUFFER_COPY:
            return PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        case PJRT_HOST_BUFFER_IMMUTABLE_UNTIL_TRANSFER:
            return PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        case PJRT_HOST_BUFFER_ZERO_COPY:
            return PJRT_HostBufferSemantics_kImmutableZeroCopy;
        default:
            return PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    }
}

//===----------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------===//

extern "C" {

void PJRTDirect_StatusFree(PJRTDirect_Status* status) {
    if (status && status->message) {
        free(status->message);
        status->message = nullptr;
    }
}

const char* PJRTDirect_GetVersion(void) {
    return kVersion;
}

const char* PJRTDirect_GetDefaultPluginPath(PJRTDirect_Backend backend) {
    std::lock_guard<std::mutex> lock(g_paths_mutex);

    // Check if we have a cached path
    auto it = g_default_paths.find((int)backend);
    if (it != g_default_paths.end()) {
        return it->second.c_str();
    }

    // Check environment variable
    const char* env_var = nullptr;
    const char* default_path = nullptr;

    switch (backend) {
        case PJRT_BACKEND_CPU:
            env_var = kEnvCpuPlugin;
            default_path = "/opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so";
            break;
        case PJRT_BACKEND_GPU_CUDA:
            env_var = kEnvGpuCudaPlugin;
            default_path = "/opt/swiftir-deps/lib/pjrt_c_api_cuda_plugin.so";
            break;
        case PJRT_BACKEND_GPU_ROCM:
            env_var = kEnvGpuRocmPlugin;
            default_path = "/opt/swiftir-deps/lib/pjrt_c_api_rocm_plugin.so";
            break;
        case PJRT_BACKEND_TPU:
            env_var = kEnvTpuPlugin;
            default_path = nullptr; // TPU path varies
            break;
    }

    const char* env_path = env_var ? getenv(env_var) : nullptr;
    const char* result = env_path ? env_path : default_path;

    if (result) {
        g_default_paths[(int)backend] = result;
        return g_default_paths[(int)backend].c_str();
    }

    return nullptr;
}

void PJRTDirect_SetDefaultPluginPath(PJRTDirect_Backend backend, const char* path) {
    std::lock_guard<std::mutex> lock(g_paths_mutex);
    if (path) {
        g_default_paths[(int)backend] = path;
    } else {
        g_default_paths.erase((int)backend);
    }
}

bool PJRTDirect_IsBackendAvailable(PJRTDirect_Backend backend) {
    const char* path = PJRTDirect_GetDefaultPluginPath(backend);
    if (!path) return false;

    // Try to open the plugin
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return false;

    // Check for GetPjrtApi symbol
    void* sym = dlsym(handle, "GetPjrtApi");
    dlclose(handle);

    return sym != nullptr;
}

PJRTDirect_Status PJRTDirect_CreateClient(
    PJRTDirect_Backend backend,
    const char* plugin_path,
    PJRTDirect_Client* out_client
) {
    if (!out_client) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "out_client is null");
    }

    // Get plugin path
    const char* path = plugin_path ? plugin_path : PJRTDirect_GetDefaultPluginPath(backend);
    if (!path) {
        return make_status(PJRT_DIRECT_ERROR_NOT_FOUND, "No plugin path for backend");
    }

    // Load plugin
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        return make_status(PJRT_DIRECT_ERROR_NOT_FOUND, dlerror());
    }

    // Get API function
    typedef const PJRT_Api* (*GetPjrtApiFn)();
    GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        dlclose(handle);
        return make_status(PJRT_DIRECT_ERROR_NOT_FOUND, "GetPjrtApi not found");
    }

    // Get API struct
    const PJRT_Api* api = get_api();
    if (!api) {
        dlclose(handle);
        return make_status(PJRT_DIRECT_ERROR_INTERNAL, "GetPjrtApi returned null");
    }

    // Initialize plugin
    PJRT_Plugin_Initialize_Args init_args;
    init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    init_args.extension_start = nullptr;
    PJRT_Error* init_error = api->PJRT_Plugin_Initialize(&init_args);
    if (init_error) {
        PJRTDirect_Status status = from_pjrt_error(api, init_error);
        dlclose(handle);
        return status;
    }

    // Create client
    PJRT_Client_Create_Args create_args;
    memset(&create_args, 0, sizeof(create_args));
    create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.extension_start = nullptr;

    PJRT_Error* create_error = api->PJRT_Client_Create(&create_args);
    if (create_error) {
        PJRTDirect_Status status = from_pjrt_error(api, create_error);
        dlclose(handle);
        return status;
    }

    // Allocate our client struct
    PJRTDirect_Client_s* client = new PJRTDirect_Client_s();
    client->plugin_handle = handle;
    client->api = api;
    client->client = create_args.client;
    client->backend = backend;

    // Cache platform info
    PJRT_Client_PlatformName_Args name_args;
    name_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
    name_args.extension_start = nullptr;
    name_args.client = client->client;
    api->PJRT_Client_PlatformName(&name_args);
    client->platform_name = std::string(name_args.platform_name, name_args.platform_name_size);

    PJRT_Client_PlatformVersion_Args ver_args;
    ver_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
    ver_args.extension_start = nullptr;
    ver_args.client = client->client;
    api->PJRT_Client_PlatformVersion(&ver_args);
    client->platform_version = std::string(ver_args.platform_version, ver_args.platform_version_size);

    // Cache devices
    PJRT_Client_Devices_Args dev_args;
    dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    dev_args.extension_start = nullptr;
    dev_args.client = client->client;
    api->PJRT_Client_Devices(&dev_args);

    client->devices.reserve(dev_args.num_devices);
    for (size_t i = 0; i < dev_args.num_devices; i++) {
        client->devices.push_back(dev_args.devices[i]);
    }

    *out_client = client;
    return ok_status();
}

void PJRTDirect_DestroyClient(PJRTDirect_Client client) {
    if (!client) return;

    // Destroy PJRT client
    PJRT_Client_Destroy_Args args;
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client->client;
    client->api->PJRT_Client_Destroy(&args);

    // Close plugin
    if (client->plugin_handle) {
        dlclose(client->plugin_handle);
    }

    delete client;
}

const char* PJRTDirect_GetPlatformName(PJRTDirect_Client client) {
    return client ? client->platform_name.c_str() : nullptr;
}

const char* PJRTDirect_GetPlatformVersion(PJRTDirect_Client client) {
    return client ? client->platform_version.c_str() : nullptr;
}

int32_t PJRTDirect_GetDeviceCount(PJRTDirect_Client client) {
    return client ? (int32_t)client->devices.size() : 0;
}

PJRTDirect_Status PJRTDirect_GetDevice(
    PJRTDirect_Client client,
    int32_t device_index,
    PJRTDirect_Device* out_device
) {
    if (!client || !out_device) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "null argument");
    }

    if (device_index < 0 || device_index >= (int32_t)client->devices.size()) {
        return make_status(PJRT_DIRECT_ERROR_OUT_OF_RANGE, "device index out of range");
    }

    PJRTDirect_Device_s* device = new PJRTDirect_Device_s();
    device->device = client->devices[device_index];
    device->client = client;
    device->index = device_index;

    *out_device = device;
    return ok_status();
}

PJRTDirect_Status PJRTDirect_GetDefaultDevice(
    PJRTDirect_Client client,
    PJRTDirect_Device* out_device
) {
    return PJRTDirect_GetDevice(client, 0, out_device);
}

PJRTDirect_CompileOptions PJRTDirect_DefaultCompileOptions(void) {
    PJRTDirect_CompileOptions opts;
    opts.num_replicas = 1;
    opts.num_partitions = 1;
    opts.optimization_level = -1; // Default
    return opts;
}

PJRTDirect_Status PJRTDirect_Compile(
    PJRTDirect_Client client,
    const char* mlir_module,
    size_t mlir_size,
    PJRTDirect_CompileOptions options,
    PJRTDirect_Executable* out_exec
) {
    if (!client || !mlir_module || !out_exec) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "null argument");
    }

    const PJRT_Api* api = client->api;

    // Suppress unused parameter warning
    (void)options;

    // Create program
    PJRT_Program program;
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.extension_start = nullptr;
    program.code = const_cast<char*>(mlir_module);
    program.code_size = mlir_size;
    program.format = "mlir";
    program.format_size = 4;

    // Compile
    PJRT_Client_Compile_Args compile_args;
    memset(&compile_args, 0, sizeof(compile_args));
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.extension_start = nullptr;
    compile_args.client = client->client;
    compile_args.program = &program;

    PJRT_Error* error = api->PJRT_Client_Compile(&compile_args);
    if (error) {
        return from_pjrt_error(api, error);
    }

    // Create our executable struct
    PJRTDirect_Executable_s* exec = new PJRTDirect_Executable_s();
    exec->executable = compile_args.executable;
    exec->client = client;

    // Get num inputs/outputs (these aren't directly available in PJRT C API,
    // so we'd need to parse the MLIR or track them separately)
    // For now, set to -1 to indicate unknown
    exec->num_inputs = -1;
    exec->num_outputs = -1;

    *out_exec = exec;
    return ok_status();
}

void PJRTDirect_DestroyExecutable(PJRTDirect_Executable exec) {
    if (!exec) return;

    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = exec->executable;
    exec->client->api->PJRT_LoadedExecutable_Destroy(&args);

    delete exec;
}

int32_t PJRTDirect_GetNumInputs(PJRTDirect_Executable exec) {
    return exec ? exec->num_inputs : -1;
}

int32_t PJRTDirect_GetNumOutputs(PJRTDirect_Executable exec) {
    return exec ? exec->num_outputs : -1;
}

size_t PJRTDirect_DTypeSize(PJRTDirect_DType dtype) {
    switch (dtype) {
        case PJRT_DTYPE_F32: return 4;
        case PJRT_DTYPE_F64: return 8;
        case PJRT_DTYPE_F16: return 2;
        case PJRT_DTYPE_BF16: return 2;
        case PJRT_DTYPE_I8: return 1;
        case PJRT_DTYPE_I16: return 2;
        case PJRT_DTYPE_I32: return 4;
        case PJRT_DTYPE_I64: return 8;
        case PJRT_DTYPE_U8: return 1;
        case PJRT_DTYPE_U16: return 2;
        case PJRT_DTYPE_U32: return 4;
        case PJRT_DTYPE_U64: return 8;
        case PJRT_DTYPE_BOOL: return 1;
        case PJRT_DTYPE_C64: return 8;
        case PJRT_DTYPE_C128: return 16;
        default: return 4;
    }
}

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
) {
    // Forward to generic execute with F32 types
    std::vector<PJRTDirect_DType> input_dtypes(num_inputs, PJRT_DTYPE_F32);
    std::vector<PJRTDirect_DType> output_dtypes(num_outputs, PJRT_DTYPE_F32);

    return PJRTDirect_Execute(
        exec, device,
        (const void* const*)inputs, input_sizes, input_dtypes.data(), num_inputs,
        (void**)outputs, output_sizes, output_dtypes.data(), num_outputs,
        semantics, out_timing
    );
}

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
) {
    if (!exec || !device || !inputs || !outputs) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "null argument");
    }

    const PJRT_Api* api = exec->client->api;
    uint64_t start_ns = out_timing ? now_ns() : 0;
    uint64_t h2d_end_ns = 0;
    uint64_t exec_end_ns = 0;
    uint64_t d2h_end_ns = 0;

    // ======== H2D: Create input buffers ========
    std::vector<PJRT_Buffer*> input_buffers(num_inputs);
    std::vector<PJRT_Event*> h2d_events(num_inputs);

    PJRT_HostBufferSemantics pjrt_semantics = semantics_to_pjrt(semantics);

    for (size_t i = 0; i < num_inputs; i++) {
        PJRT_Client_BufferFromHostBuffer_Args h2d_args;
        memset(&h2d_args, 0, sizeof(h2d_args));
        h2d_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        h2d_args.extension_start = nullptr;
        h2d_args.client = exec->client->client;
        h2d_args.data = inputs[i];
        h2d_args.type = dtype_to_pjrt(input_dtypes[i]);

        // 1D tensor with input_sizes[i] elements
        int64_t dim = input_sizes[i];
        h2d_args.dims = &dim;
        h2d_args.num_dims = 1;
        h2d_args.host_buffer_semantics = pjrt_semantics;
        h2d_args.device = device->device;

        PJRT_Error* error = api->PJRT_Client_BufferFromHostBuffer(&h2d_args);
        if (error) {
            // Cleanup already created buffers
            for (size_t j = 0; j < i; j++) {
                PJRT_Buffer_Destroy_Args destroy_args;
                destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
                destroy_args.extension_start = nullptr;
                destroy_args.buffer = input_buffers[j];
                api->PJRT_Buffer_Destroy(&destroy_args);
            }
            return from_pjrt_error(api, error);
        }

        input_buffers[i] = h2d_args.buffer;
        h2d_events[i] = h2d_args.done_with_host_buffer;
    }

    // Wait for H2D transfers
    for (size_t i = 0; i < num_inputs; i++) {
        if (h2d_events[i]) {
            PJRT_Event_Await_Args await_args;
            await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
            await_args.extension_start = nullptr;
            await_args.event = h2d_events[i];
            api->PJRT_Event_Await(&await_args);

            PJRT_Event_Destroy_Args destroy_args;
            destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
            destroy_args.extension_start = nullptr;
            destroy_args.event = h2d_events[i];
            api->PJRT_Event_Destroy(&destroy_args);
        }
    }

    if (out_timing) h2d_end_ns = now_ns();

    // ======== Execute ========
    PJRT_LoadedExecutable_Execute_Args exec_args;
    memset(&exec_args, 0, sizeof(exec_args));
    exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    exec_args.extension_start = nullptr;
    exec_args.executable = exec->executable;
    exec_args.num_devices = 1;
    exec_args.num_args = num_inputs;

    // Execution inputs: pointer to array of PJRT_Buffer* arrays (one per device)
    PJRT_Buffer** input_buffer_ptrs = input_buffers.data();
    exec_args.argument_lists = &input_buffer_ptrs;

    // Execute options
    PJRT_ExecuteOptions execute_options;
    memset(&execute_options, 0, sizeof(execute_options));
    execute_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    exec_args.options = &execute_options;

    PJRT_Error* exec_error = api->PJRT_LoadedExecutable_Execute(&exec_args);
    if (exec_error) {
        // Cleanup input buffers
        for (size_t i = 0; i < num_inputs; i++) {
            PJRT_Buffer_Destroy_Args destroy_args;
            destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            destroy_args.extension_start = nullptr;
            destroy_args.buffer = input_buffers[i];
            api->PJRT_Buffer_Destroy(&destroy_args);
        }
        return from_pjrt_error(api, exec_error);
    }

    // Wait for execution to complete
    if (exec_args.device_complete_events && exec_args.device_complete_events[0]) {
        PJRT_Event_Await_Args await_args;
        await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        await_args.extension_start = nullptr;
        await_args.event = exec_args.device_complete_events[0];
        api->PJRT_Event_Await(&await_args);

        PJRT_Event_Destroy_Args destroy_args;
        destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.event = exec_args.device_complete_events[0];
        api->PJRT_Event_Destroy(&destroy_args);
    }

    if (out_timing) exec_end_ns = now_ns();

    // ======== D2H: Copy outputs to host ========
    // Output buffers are in exec_args.output_lists[0]
    PJRT_Buffer** output_buffers = exec_args.output_lists[0];

    for (size_t i = 0; i < num_outputs; i++) {
        PJRT_Buffer_ToHostBuffer_Args d2h_args;
        memset(&d2h_args, 0, sizeof(d2h_args));
        d2h_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        d2h_args.extension_start = nullptr;
        d2h_args.src = output_buffers[i];
        d2h_args.dst = outputs[i];
        d2h_args.dst_size = output_sizes[i] * PJRTDirect_DTypeSize(output_dtypes[i]);

        PJRT_Error* d2h_error = api->PJRT_Buffer_ToHostBuffer(&d2h_args);
        if (d2h_error) {
            // Cleanup
            for (size_t j = 0; j < num_inputs; j++) {
                PJRT_Buffer_Destroy_Args destroy_args;
                destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
                destroy_args.extension_start = nullptr;
                destroy_args.buffer = input_buffers[j];
                api->PJRT_Buffer_Destroy(&destroy_args);
            }
            for (size_t j = 0; j < num_outputs; j++) {
                PJRT_Buffer_Destroy_Args destroy_args;
                destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
                destroy_args.extension_start = nullptr;
                destroy_args.buffer = output_buffers[j];
                api->PJRT_Buffer_Destroy(&destroy_args);
            }
            return from_pjrt_error(api, d2h_error);
        }

        // Wait for D2H transfer
        if (d2h_args.event) {
            PJRT_Event_Await_Args await_args;
            await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
            await_args.extension_start = nullptr;
            await_args.event = d2h_args.event;
            api->PJRT_Event_Await(&await_args);

            PJRT_Event_Destroy_Args destroy_args;
            destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
            destroy_args.extension_start = nullptr;
            destroy_args.event = d2h_args.event;
            api->PJRT_Event_Destroy(&destroy_args);
        }
    }

    if (out_timing) d2h_end_ns = now_ns();

    // ======== Cleanup buffers ========
    for (size_t i = 0; i < num_inputs; i++) {
        PJRT_Buffer_Destroy_Args destroy_args;
        destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.buffer = input_buffers[i];
        api->PJRT_Buffer_Destroy(&destroy_args);
    }

    for (size_t i = 0; i < num_outputs; i++) {
        PJRT_Buffer_Destroy_Args destroy_args;
        destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.buffer = output_buffers[i];
        api->PJRT_Buffer_Destroy(&destroy_args);
    }

    // Fill timing if requested
    if (out_timing) {
        uint64_t end_ns = now_ns();
        out_timing->h2d_ns = h2d_end_ns - start_ns;
        out_timing->execute_ns = exec_end_ns - h2d_end_ns;
        out_timing->d2h_ns = d2h_end_ns - exec_end_ns;
        out_timing->total_ns = end_ns - start_ns;
    }

    return ok_status();
}

// ======== Execution Context API ========

PJRTDirect_Status PJRTDirect_CreateExecContext(
    PJRTDirect_Executable exec,
    PJRTDirect_Device device,
    const int64_t* input_sizes,
    size_t num_inputs,
    const int64_t* output_sizes,
    size_t num_outputs,
    PJRTDirect_ExecContext* out_ctx
) {
    if (!exec || !device || !out_ctx) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "null argument");
    }

    PJRTDirect_ExecContext_s* ctx = new PJRTDirect_ExecContext_s();
    ctx->exec = exec;
    ctx->device = device;

    // Cache sizes
    ctx->input_sizes.assign(input_sizes, input_sizes + num_inputs);
    ctx->output_sizes.assign(output_sizes, output_sizes + num_outputs);

    // Pre-allocate argument structs
    ctx->h2d_args = new PJRT_Client_BufferFromHostBuffer_Args[num_inputs];
    ctx->exec_args = new PJRT_LoadedExecutable_Execute_Args();
    ctx->d2h_args = new PJRT_Buffer_ToHostBuffer_Args[num_outputs];

    // Initialize h2d_args
    for (size_t i = 0; i < num_inputs; i++) {
        memset(&ctx->h2d_args[i], 0, sizeof(PJRT_Client_BufferFromHostBuffer_Args));
        ctx->h2d_args[i].struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        ctx->h2d_args[i].client = exec->client->client;
        ctx->h2d_args[i].type = PJRT_Buffer_Type_F32;
        ctx->h2d_args[i].num_dims = 1;
        ctx->h2d_args[i].host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        ctx->h2d_args[i].device = device->device;
    }

    // Initialize exec_args
    memset(ctx->exec_args, 0, sizeof(PJRT_LoadedExecutable_Execute_Args));
    ctx->exec_args->struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ctx->exec_args->executable = exec->executable;
    ctx->exec_args->num_devices = 1;
    ctx->exec_args->num_args = num_inputs;

    // Initialize d2h_args
    for (size_t i = 0; i < num_outputs; i++) {
        memset(&ctx->d2h_args[i], 0, sizeof(PJRT_Buffer_ToHostBuffer_Args));
        ctx->d2h_args[i].struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        ctx->d2h_args[i].dst_size = output_sizes[i] * sizeof(float);
    }

    *out_ctx = ctx;
    return ok_status();
}

PJRTDirect_Status PJRTDirect_ExecuteWithContext(
    PJRTDirect_ExecContext ctx,
    const float* const* inputs,
    float** outputs,
    PJRTDirect_Timing* out_timing
) {
    if (!ctx || !inputs || !outputs) {
        return make_status(PJRT_DIRECT_ERROR_INVALID_ARGUMENT, "null argument");
    }

    // Use the generic Execute function with cached sizes
    return PJRTDirect_ExecuteF32(
        ctx->exec, ctx->device,
        inputs, ctx->input_sizes.data(), ctx->input_sizes.size(),
        outputs, ctx->output_sizes.data(), ctx->output_sizes.size(),
        PJRT_HOST_BUFFER_COPY, out_timing
    );
}

void PJRTDirect_DestroyExecContext(PJRTDirect_ExecContext ctx) {
    if (!ctx) return;

    delete[] ctx->h2d_args;
    delete ctx->exec_args;
    delete[] ctx->d2h_args;
    delete ctx;
}

} // extern "C"
