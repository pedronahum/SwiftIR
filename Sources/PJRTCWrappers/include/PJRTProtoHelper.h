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

/// Free a buffer allocated by PJRT_CreateCompileOptions
void PJRT_FreeCompileOptions(char* buffer);

#ifdef __cplusplus
}
#endif

#endif // PJRT_PROTO_HELPER_H
