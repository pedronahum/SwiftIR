//===-- PJRTWrapper.h - PJRT C API Wrapper --------------------*- C++ -*-===//
//
// SwiftIR - Phase 11: PJRT Integration
// C++ wrapper for PJRT C API
//
//===------------------------------------------------------------------===//

#ifndef PJRT_WRAPPER_H
#define PJRT_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <dlfcn.h>

// Include the actual PJRT C API header
// This provides the full PJRT C API definitions
#include "pjrt_c_api.h"
#include "pjrt_c_api_cpu.h"

// Helper functions for dynamic loading of PJRT plugins

// Global handles for loaded plugins
static void* g_pjrt_cpu_plugin_handle = NULL;
static void* g_pjrt_gpu_plugin_handle = NULL;

// Load PJRT CPU plugin dynamically
// Returns NULL if plugin cannot be loaded
static inline const PJRT_Api* PJRT_LoadCpuPluginWrapper(const char* plugin_path) {
    if (g_pjrt_cpu_plugin_handle != NULL) {
        // Already loaded
        PJRT_GetCpuApi_t get_api = (PJRT_GetCpuApi_t)dlsym(g_pjrt_cpu_plugin_handle, "GetPjrtApi");
        if (get_api != NULL) {
            return get_api();
        }
        return NULL;
    }

    // Try to load the plugin
    g_pjrt_cpu_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (g_pjrt_cpu_plugin_handle == NULL) {
        return NULL;
    }

    // Get the API function
    PJRT_GetCpuApi_t get_api = (PJRT_GetCpuApi_t)dlsym(g_pjrt_cpu_plugin_handle, "GetPjrtApi");
    if (get_api == NULL) {
        dlclose(g_pjrt_cpu_plugin_handle);
        g_pjrt_cpu_plugin_handle = NULL;
        return NULL;
    }

    return get_api();
}

// Load PJRT GPU plugin dynamically
// Returns NULL if plugin cannot be loaded
static inline const PJRT_Api* PJRT_LoadGpuPluginWrapper(const char* plugin_path) {
    if (g_pjrt_gpu_plugin_handle != NULL) {
        // Already loaded
        PJRT_GetGpuApi_t get_api = (PJRT_GetGpuApi_t)dlsym(g_pjrt_gpu_plugin_handle, "GetPjrtApi");
        if (get_api != NULL) {
            return get_api();
        }
        return NULL;
    }

    // Try to load the plugin
    g_pjrt_gpu_plugin_handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (g_pjrt_gpu_plugin_handle == NULL) {
        return NULL;
    }

    // Get the API function
    PJRT_GetGpuApi_t get_api = (PJRT_GetGpuApi_t)dlsym(g_pjrt_gpu_plugin_handle, "GetPjrtApi");
    if (get_api == NULL) {
        dlclose(g_pjrt_gpu_plugin_handle);
        g_pjrt_gpu_plugin_handle = NULL;
        return NULL;
    }

    return get_api();
}

// Unload PJRT plugins
static inline void PJRT_UnloadPluginsWrapper(void) {
    if (g_pjrt_cpu_plugin_handle != NULL) {
        dlclose(g_pjrt_cpu_plugin_handle);
        g_pjrt_cpu_plugin_handle = NULL;
    }
    if (g_pjrt_gpu_plugin_handle != NULL) {
        dlclose(g_pjrt_gpu_plugin_handle);
        g_pjrt_gpu_plugin_handle = NULL;
    }
}

// Get last dlopen/dlsym error
static inline const char* PJRT_GetLoadErrorWrapper(void) {
    return dlerror();
}

#ifdef __cplusplus
}
#endif

#endif // PJRT_WRAPPER_H
