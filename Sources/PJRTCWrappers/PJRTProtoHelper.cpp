//===-- PJRTProtoHelper.cpp - PJRT Protobuf Helper ----------*- C++ -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// C++ helper to construct XLA CompileOptionsProto using manual protobuf encoding
// This version avoids XLA proto library dependencies by encoding directly
//
//===------------------------------------------------------------------===//

#include "PJRTProtoHelper.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdint>

// Manual protobuf encoding helpers
static void writeVarint(std::vector<uint8_t>& buf, uint64_t value) {
    while (value > 0x7F) {
        buf.push_back((value & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push_back(value & 0x7F);
}

static void writeTag(std::vector<uint8_t>& buf, int fieldNumber, int wireType) {
    writeVarint(buf, (fieldNumber << 3) | wireType);
}

extern "C" {

// Create a minimal CompileOptionsProto with num_replicas and num_partitions
// Returns a malloc'd buffer containing the serialized protobuf
// Caller must free the returned buffer with PJRT_FreeCompileOptions
//
// CompileOptionsProto structure (from xla/pjrt/proto/compile_options.proto):
//   message CompileOptionsProto {
//     ExecutableBuildOptionsProto executable_build_options = 3;
//   }
//   message ExecutableBuildOptionsProto {
//     int64 num_replicas = 4;
//     int64 num_partitions = 5;
//   }
char* PJRT_CreateCompileOptions(int64_t num_replicas, int64_t num_partitions, size_t* out_size) {
    std::vector<uint8_t> buf;

    // First build ExecutableBuildOptionsProto content
    std::vector<uint8_t> execBuildOpts;

    // Field 4: num_replicas (varint)
    if (num_replicas != 0) {
        writeTag(execBuildOpts, 4, 0);  // field 4, wire type 0 (varint)
        writeVarint(execBuildOpts, num_replicas);
    }

    // Field 5: num_partitions (varint)
    if (num_partitions != 0) {
        writeTag(execBuildOpts, 5, 0);  // field 5, wire type 0 (varint)
        writeVarint(execBuildOpts, num_partitions);
    }

    // Now build CompileOptionsProto
    // Field 3: executable_build_options (length-delimited)
    if (!execBuildOpts.empty()) {
        writeTag(buf, 3, 2);  // field 3, wire type 2 (length-delimited)
        writeVarint(buf, execBuildOpts.size());
        buf.insert(buf.end(), execBuildOpts.begin(), execBuildOpts.end());
    }

    // Allocate buffer and copy
    char* buffer = (char*)malloc(buf.size());
    if (buffer) {
        memcpy(buffer, buf.data(), buf.size());
        *out_size = buf.size();
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
