//===-- PJRTProtoHelper.h - PJRT Protobuf Helper ------------*- C++ -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// C helper functions for creating XLA protobuf messages
//
//===------------------------------------------------------------------===//

#ifndef PJRT_PROTO_HELPER_H
#define PJRT_PROTO_HELPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a CompileOptionsProto with the specified parameters
/// Returns a malloc'd buffer containing the serialized protobuf
/// Caller must free the returned buffer with PJRT_FreeCompileOptions
char* PJRT_CreateCompileOptions(int64_t num_replicas, int64_t num_partitions, size_t* out_size);

/// Create a CompileOptionsProto with optimization level
/// xla_opt_level: XLA backend optimization level (0-3)
///   0 = No optimization (fastest compile, slowest execution)
///   1 = Basic optimization
///   2 = Standard optimization (recommended, GPU requires >= 2)
///   3 = Maximum optimization (slowest compile, best execution)
/// Returns a malloc'd buffer containing the serialized protobuf
/// Caller must free the returned buffer with PJRT_FreeCompileOptions
char* PJRT_CreateCompileOptionsWithOptLevel(
    int64_t num_replicas,
    int64_t num_partitions,
    int32_t xla_opt_level,
    size_t* out_size
);

/// Free a buffer allocated by PJRT_CreateCompileOptions
void PJRT_FreeCompileOptions(char* buffer);

#ifdef __cplusplus
}
#endif

#endif // PJRT_PROTO_HELPER_H
