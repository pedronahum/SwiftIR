//===-- PJRTProtoHelper.cpp - PJRT Protobuf Helper ----------*- C++ -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// C++ helper to construct XLA CompileOptionsProto using protobuf library
//
//===------------------------------------------------------------------===//

#include "PJRTProtoHelper.h"
#include "compile_options.pb.h"  // Located via -I flag pointing to proto directory
#include <cstring>
#include <memory>

extern "C" {

// Create a minimal CompileOptionsProto with num_replicas and num_partitions
// Returns a malloc'd buffer containing the serialized protobuf
// Caller must free the returned buffer with PJRT_FreeCompileOptions
char* PJRT_CreateCompileOptions(int64_t num_replicas, int64_t num_partitions, size_t* out_size) {
    // Create the CompileOptionsProto message
    xla::CompileOptionsProto options;

    // Set the executable build options
    auto* build_options = options.mutable_executable_build_options();
    build_options->set_num_replicas(num_replicas);
    build_options->set_num_partitions(num_partitions);

    // Serialize to string
    std::string serialized = options.SerializeAsString();

    // Allocate buffer and copy
    char* buffer = (char*)malloc(serialized.size());
    if (buffer) {
        memcpy(buffer, serialized.data(), serialized.size());
        *out_size = serialized.size();
    } else {
        *out_size = 0;
    }

    return buffer;
}

// Free a buffer allocated by PJRT_CreateCompileOptions
void PJRT_FreeCompileOptions(char* buffer) {
    free(buffer);
}

} // extern "C"
