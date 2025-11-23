// SwiftIRMLIR.cpp - Unified MLIR/LLVM Library for SwiftIR
// This file creates a shared library that embeds all MLIR/LLVM dependencies
// Following the pattern used by StableHLO, TensorFlow, and JAX

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir-c/Dialect/Linalg.h"
#include <stablehlo/integrations/c/StablehloDialect.h>
#include <stablehlo/integrations/c/ChloDialect.h>
// Note: StableHLO v1.0.0 doesn't have the passes C API header
// We provide stub implementations below

// This is a minimal wrapper that ensures all MLIR/LLVM symbols are included
// in the shared library. The actual functionality is provided by the C API
// libraries that are linked in via CMake.

// Version information
extern "C" {

const char* swiftir_mlir_get_version() {
    return "1.0.0-stablehlo";
}

// Initialization function that ensures all dialect handles are available
void swiftir_mlir_init() {
    // Initialization code can go here if needed
    // For now, this is just a placeholder to ensure the library has symbols
}

// StableHLO dialect registration wrapper
void swiftir_register_stablehlo_dialect(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__stablehlo__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

// CHLO dialect registration wrapper
void swiftir_register_chlo_dialect(MlirContext context) {
    MlirDialectHandle handle = mlirGetDialectHandle__chlo__();
    mlirDialectHandleRegisterDialect(handle, context);
    mlirDialectHandleLoadDialect(handle, context);
}

// StableHLO pass registration wrapper
// Note: StableHLO v1.0.0 doesn't expose pass registration via C API
// This is a stub that does nothing - passes must be registered differently
void swiftir_register_all_stablehlo_passes() {
    // No-op for StableHLO v1.0.0
    // Passes are available but not through C API
}

} // extern "C"

// Note: All MLIR/LLVM symbols are included in this library via CMake's
// target_link_libraries. The libraries are built as OBJECT libraries
// and get embedded into this shared library automatically by CMake.
