#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

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

// Signed varint encoding (zig-zag for negative values)
static void writeSignedVarint(std::vector<uint8_t>& buf, int32_t value) {
    // For int32 in protobuf, positive values are encoded directly
    // but the wire format can handle 10 bytes for 64-bit values
    // XLA's int32 fields use standard varint encoding
    if (value >= 0) {
        writeVarint(buf, value);
    } else {
        // Negative int32 is sign-extended to 64 bits
        writeVarint(buf, (uint64_t)(int64_t)value);
    }
}

void printBytes(const char* label, const std::vector<uint8_t>& buf) {
    printf("%s (%zu bytes):\n", label, buf.size());
    for (size_t i = 0; i < buf.size(); i++) {
        printf("%02x ", buf[i]);
    }
    printf("\n\n");
}

int main() {
    // Test encoding just DebugOptions with xla_backend_optimization_level = 3
    // Field 31, wire type 0 (varint)
    // Tag = (31 << 3) | 0 = 248
    // 248 = 0xF8, which needs 2 bytes: F8 01
    // Value 3 is just 03
    
    printf("Testing protobuf encoding for xla_backend_optimization_level\n\n");
    
    // Just the DebugOptions message
    std::vector<uint8_t> debugOpts;
    writeTag(debugOpts, 31, 0);  // field 31, wire type 0
    writeSignedVarint(debugOpts, 3);
    printBytes("DebugOptions (field 31 = 3)", debugOpts);
    
    // ExecutableBuildOptionsProto with debug_options
    std::vector<uint8_t> execBuildOpts;
    writeTag(execBuildOpts, 3, 2);  // field 3 (debug_options), wire type 2 (length-delimited)
    writeVarint(execBuildOpts, debugOpts.size());
    execBuildOpts.insert(execBuildOpts.end(), debugOpts.begin(), debugOpts.end());
    // Add num_replicas = 1
    writeTag(execBuildOpts, 4, 0);  // field 4, wire type 0
    writeVarint(execBuildOpts, 1);
    // Add num_partitions = 1
    writeTag(execBuildOpts, 5, 0);  // field 5, wire type 0
    writeVarint(execBuildOpts, 1);
    printBytes("ExecutableBuildOptionsProto", execBuildOpts);
    
    // CompileOptionsProto
    std::vector<uint8_t> compileOpts;
    writeTag(compileOpts, 3, 2);  // field 3 (executable_build_options), wire type 2
    writeVarint(compileOpts, execBuildOpts.size());
    compileOpts.insert(compileOpts.end(), execBuildOpts.begin(), execBuildOpts.end());
    printBytes("CompileOptionsProto (final)", compileOpts);
    
    // Now test with different optimization levels
    for (int level = 0; level <= 3; level++) {
        std::vector<uint8_t> dbg;
        writeTag(dbg, 31, 0);
        writeSignedVarint(dbg, level);
        
        std::vector<uint8_t> exec;
        writeTag(exec, 3, 2);
        writeVarint(exec, dbg.size());
        exec.insert(exec.end(), dbg.begin(), dbg.end());
        writeTag(exec, 4, 0);
        writeVarint(exec, 1);
        writeTag(exec, 5, 0);
        writeVarint(exec, 1);
        
        std::vector<uint8_t> compile;
        writeTag(compile, 3, 2);
        writeVarint(compile, exec.size());
        compile.insert(compile.end(), exec.begin(), exec.end());
        
        char label[64];
        snprintf(label, sizeof(label), "xla_backend_optimization_level = %d", level);
        printBytes(label, compile);
    }
    
    // Test without optimization level (default behavior)
    std::vector<uint8_t> execDefault;
    writeTag(execDefault, 4, 0);
    writeVarint(execDefault, 1);
    writeTag(execDefault, 5, 0);
    writeVarint(execDefault, 1);
    
    std::vector<uint8_t> compileDefault;
    writeTag(compileDefault, 3, 2);
    writeVarint(compileDefault, execDefault.size());
    compileDefault.insert(compileDefault.end(), execDefault.begin(), execDefault.end());
    printBytes("Default (no xla_backend_optimization_level)", compileDefault);
    
    return 0;
}
