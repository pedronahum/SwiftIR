# SwiftIRCore

**Foundation layer providing Swift bindings to MLIR C API**

## Purpose

SwiftIRCore is the foundational module of SwiftIR that provides direct Swift bindings to the MLIR (Multi-Level Intermediate Representation) C API. It handles memory management, type safety, and provides the core abstractions needed to work with MLIR from Swift.

## What's Inside

### Swift Code
- `MLIRCore.swift` - Core MLIR types and operations
  - `MLIRContext` - Manages MLIR context lifetime and state
  - `MLIRModule` - Represents MLIR modules (top-level IR containers)
  - `MLIRError` - Error handling for MLIR operations
  - Module parsing and verification
  - Memory-safe wrappers around C API

### C Headers (`include/`)
- `MLIRCoreWrapper.h` - C wrapper functions for MLIR C API
  - Context creation/destruction
  - Module operations
  - Type system integration
  - Location handling
- `pjrt_c_api.h` - PJRT (Plugin JAX Runtime) C API headers
- `pjrt_c_api_cpu.h` - CPU-specific PJRT API
- `PJRTWrapper.h` - C wrappers for PJRT runtime
- `PJRTSimpleWrapper.h` - Simplified PJRT C wrapper for Swift interop
- `module.modulemap` - Clang module map for Swift/C++ interoperability

## Architecture

```
Swift Code (MLIRCore.swift)
         ↓
C Wrapper (MLIRCoreWrapper.h)
         ↓
MLIR C API (from StableHLO build)
         ↓
MLIR C++ Core
```

## Key Design Decisions

### Memory Management
- Uses Swift's ARC for automatic memory management
- `deinit` methods ensure MLIR resources are properly destroyed
- Safe wrappers prevent use-after-free errors

### Type Safety
- Swift classes wrap C handles (MlirContext, MlirModule)
- Null checks prevent invalid handle usage
- Strong typing enforces correct API usage

### C Interoperability
All C headers use Swift-friendly patterns:
- Opaque pointers for complex types
- Clear error codes
- Manual memory management with explicit create/destroy pairs

## Dependencies

### External
- MLIR C API (from `/Users/pedro/programming/swift/stablehlo/llvm-build/`)
- PJRT C API (for runtime execution)

### Internal
- None (this is the foundation layer)

## Usage Example

```swift
import SwiftIRCore

// Create MLIR context
let context = MLIRContext()

// Parse MLIR module from text
let module = try MLIRModule.parse("""
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
""", context: context)

// Verify module
guard module.verify() else {
    throw MLIRError.verificationFailed("Module is invalid")
}

// Dump IR
print(module.dump())
```

## Build Configuration

This module requires:
- Swift 6.0+ with C++ interop enabled
- `-interoperabilityMode(.Cxx)` flag
- Include paths to MLIR headers
- Clang module map for header bridging

## Related Modules

- **SwiftIRTypes** - Builds on top of this to provide type-safe MLIR types
- **SwiftIRDialects** - Uses this for dialect operations
- **SwiftIRXLA** - Uses this for XLA/PJRT runtime integration

## Technical Notes

### Why C Wrappers?

MLIR's C API is designed for stable ABI across language boundaries. We add thin C wrappers (`MLIRCoreWrapper.h`) to:
1. Simplify Swift interop patterns
2. Handle ownership semantics explicitly
3. Provide Swift-friendly null checks

### PJRT Integration

PJRT (Plugin JAX Runtime) headers are included here because they represent the runtime layer that executes compiled MLIR. The separation is:
- **Compile-time**: MLIR C API (IR manipulation)
- **Run-time**: PJRT C API (execution on CPU/GPU/TPU)

---

**Next Steps**: See [SwiftIRTypes](../SwiftIRTypes/) for type-safe MLIR type construction.
