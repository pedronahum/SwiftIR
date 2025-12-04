# BUILD file for SwiftIR profiler shared library
# This creates a standalone .so that can be used from Swift
#
# Copy this directory to XLA repo and build with:
#   bazelisk build //swiftir:libswiftir_profiler.so
#
# The profiler captures:
# - User TraceMe events (via traced() in Swift)
# - XLA internal operations (host_tracer_level=2)
# - HLO module structure (enable_hlo_proto=true)

cc_binary(
    name = "libswiftir_profiler.so",
    linkshared = True,
    srcs = ["swiftir_profiler_exports.cc"],
    deps = [
        # Core profiler infrastructure
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",

        # XLA profiler backends (registers profiler factories)
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:host_tracer",
        "//xla/backends/profiler/cpu:metadata_collector",

        # XLA debug info for HLO collection
        "//xla/service:xla_debug_info_manager",
    ],
    visibility = ["//visibility:public"],
)
