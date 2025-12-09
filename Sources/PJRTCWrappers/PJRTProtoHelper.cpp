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

static void writeString(std::vector<uint8_t>& buf, const char* str) {
    size_t len = strlen(str);
    writeVarint(buf, len);
    for (size_t i = 0; i < len; i++) {
        buf.push_back((uint8_t)str[i]);
    }
}

extern "C" {

// Create a minimal CompileOptionsProto with num_replicas and num_partitions
// Returns a malloc'd buffer containing the serialized protobuf
// Caller must free the returned buffer with PJRT_FreeCompileOptions
//
// CompileOptionsProto structure (from xla/pjrt/proto/compile_options.proto):
//   message CompileOptionsProto {
//     ExecutableBuildOptionsProto executable_build_options = 3;
//     map<string, OptionOverrideProto> env_option_overrides = 7;
//   }
//   message ExecutableBuildOptionsProto {
//     int64 num_replicas = 4;
//     int64 num_partitions = 5;
//   }
//   message OptionOverrideProto {
//     oneof value {
//       string string_field = 1;
//       bool bool_field = 2;
//       int64 int_field = 3;
//       double double_field = 4;
//     }
//   }
char* PJRT_CreateCompileOptions(int64_t num_replicas, int64_t num_partitions, size_t* out_size) {
    // Delegate to the version with default optimization level (-1 means use XLA default)
    return PJRT_CreateCompileOptionsWithOptLevel(num_replicas, num_partitions, -1, out_size);
}

char* PJRT_CreateCompileOptionsWithOptLevel(
    int64_t num_replicas,
    int64_t num_partitions,
    int32_t xla_opt_level,
    size_t* out_size
) {
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

    // Field 7: env_option_overrides (map<string, OptionOverrideProto>)
    // Map entries are encoded as repeated messages with key=1, value=2
    // Only add if we have a specific optimization level
    if (xla_opt_level >= 0) {
        // Build OptionOverrideProto with int_field = xla_opt_level
        std::vector<uint8_t> optionOverride;
        writeTag(optionOverride, 3, 0);  // field 3 (int_field), wire type 0 (varint)
        writeVarint(optionOverride, xla_opt_level);

        // Build map entry: key="xla_backend_optimization_level", value=optionOverride
        std::vector<uint8_t> mapEntry;
        // Field 1: key (string)
        writeTag(mapEntry, 1, 2);  // field 1, wire type 2 (length-delimited)
        writeString(mapEntry, "xla_backend_optimization_level");
        // Field 2: value (OptionOverrideProto)
        writeTag(mapEntry, 2, 2);  // field 2, wire type 2 (length-delimited)
        writeVarint(mapEntry, optionOverride.size());
        mapEntry.insert(mapEntry.end(), optionOverride.begin(), optionOverride.end());

        // Write map entry as field 7
        writeTag(buf, 7, 2);  // field 7, wire type 2 (length-delimited)
        writeVarint(buf, mapEntry.size());
        buf.insert(buf.end(), mapEntry.begin(), mapEntry.end());
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
