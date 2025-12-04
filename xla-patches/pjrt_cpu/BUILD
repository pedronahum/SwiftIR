# BUILD file for SwiftIR CPU PJRT plugin with profiler extension
# This creates a modified CPU plugin that includes profiler support.
#
# Copy this directory to XLA repo and build with:
#   bazelisk build //pjrt_cpu:pjrt_c_api_cpu_plugin_profiler.so
#
# The plugin includes:
# - All standard CPU PJRT functionality
# - Profiler extension (like GPU plugin) for TensorBoard profiling
# - XLA internal metrics via metadata collector
# - TraceMe C API for unified Swift/XLA tracing

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")

cc_library(
    name = "pjrt_c_api_cpu_internal_profiler",
    srcs = ["pjrt_c_api_cpu_internal_profiler.cc"],
    visibility = ["//visibility:public"],
    deps = [
        # Core PJRT CPU dependencies (from xla/pjrt/c:pjrt_c_api_cpu_internal)
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_ffi_internal",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_memory_descriptions_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_phase_compile_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_phase_compile_internal",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",

        # Profiler dependencies (from xla/pjrt/c:pjrt_c_api_gpu_internal)
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:host_tracer",
        "//xla/backends/profiler/cpu:metadata_collector",
        "//xla/backends/profiler/plugin:plugin_tracer_impl",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/backends/profiler/plugin:profiler_error",

        # XLA debug info for HLO collection
        "//xla/service:xla_debug_info_manager",

        # TraceMe for unified Swift/XLA tracing
        "@tsl//tsl/profiler/lib:traceme",

        # Abseil dependencies
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

xla_cc_binary(
    name = "pjrt_c_api_cpu_plugin_profiler.so",
    linkshared = True,
    deps = [":pjrt_c_api_cpu_internal_profiler"],
    visibility = ["//visibility:public"],
)
