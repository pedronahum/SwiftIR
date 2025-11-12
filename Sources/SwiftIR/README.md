# SwiftIR

**Top-level API and S4MLIR (Swift for MLIR) entry point**

## Purpose

SwiftIR is the main user-facing module that provides the high-level API for the entire SwiftIR project. It re-exports all other modules and defines the `@MLIRFunction` macro for compile-time MLIR generation.

## What's Inside

### Core File
- `S4MLIR.swift` - S4MLIR system definition and macro export
  - Version information
  - Main entry point
  - `@MLIRFunction` macro declaration
  - Re-exports of all submodules

## What is S4MLIR?

**S4MLIR = Swift for MLIR**

A modern metaprogramming system for MLIR that uses Swift as:
1. **Host language** - Write programs in Swift
2. **Metalanguage** - Generate MLIR at compile time
3. **DSL** - Declarative IR construction

### Vision

```swift
import SwiftIR

// Write Swift code
@MLIRFunction
func neuralNetLayer(_ input: Tensor<Float>, _ weights: Tensor<Float>) -> Tensor<Float> {
    let matmul = input.matmul(weights)
    let relu = matmul.relu()
    return relu
}

// Automatically generates MLIR
let mlirIR = neuralNetLayer_mlir()

// Execute via PJRT
let output = execute(mlirIR, inputs: [inputTensor, weightsTensor])
```

## Module Structure

```swift
// S4MLIR.swift
@_exported import SwiftIRCore
@_exported import SwiftIRTypes
@_exported import SwiftIRDialects
@_exported import SwiftIRBuilders
@_exported import SwiftIRXLA
@_exported import SwiftIRStableHLO

public struct S4MLIR {
    public static let version = "0.1.0"
}

@attached(peer, names: arbitrary)
public macro MLIRFunction() = #externalMacro(
    module: "SwiftIRMacros",
    type: "MLIRFunctionMacro"
)
```

## Usage

### Single Import

```swift
import SwiftIR

// Now have access to:
// - SwiftIRCore (MLIRContext, MLIRModule)
// - SwiftIRTypes (IntegerType, TensorType, etc.)
// - SwiftIRDialects (Func, Arith, etc.)
// - SwiftIRBuilders (Function, If, For, etc.)
// - SwiftIRXLA (PJRT, XLA, tensor ops)
// - SwiftIRStableHLO (StableHLO operations)
```

### Macro Usage

```swift
import SwiftIR

@MLIRFunction
func add(_ x: Int32, _ y: Int32) -> Int32 {
    return x + y
}

// Generated companion function
let irString = add_mlir()
print(irString)
// Output:
// func.func @add(%arg0: i32, %arg1: i32) -> i32 {
//   %0 = arith.addi %arg0, %arg1 : i32
//   return %0 : i32
// }
```

## Re-Exported Modules

### SwiftIRCore
Foundation MLIR types:
```swift
let context = MLIRContext()
let module = MLIRModule(context: context)
```

### SwiftIRTypes
Type-safe MLIR types:
```swift
let i32 = IntegerType(bitWidth: 32, isSigned: true, context: context)
let tensor = TensorType(shape: [2, 3], elementType: i32, context: context)
```

### SwiftIRDialects
Dialect operations:
```swift
let sum = Arith.addi(x, y, context: context)
```

### SwiftIRBuilders
High-level DSL:
```swift
let func = Function("compute", inputs: [i32], results: [i32]) { builder, args in
    // Function body
}
```

### SwiftIRXLA
XLA and PJRT integration:
```swift
let client = try PJRTClient.createCPU()
let executable = try client.compile(module)
```

### SwiftIRStableHLO
StableHLO operations:
```swift
let result = stablehlo.add(a, b, context: context)
```

## Design Philosophy

### Progressive Disclosure

SwiftIR supports multiple levels of abstraction:

#### Level 1: Macros (Highest)
```swift
@MLIRFunction
func compute(_ x: Int) -> Int {
    return x * x + 1
}
```

#### Level 2: DSL
```swift
let func = Function("compute", inputs: [i32], results: [i32]) { builder, args in
    let x = args[0]
    let squared = Arith.muli(x, x, context: context)
    let one = Arith.constant(1, type: i32, context: context)
    let result = Arith.addi(squared, one, context: context)
    Func.return([result], context: context)
}
```

#### Level 3: Builders
```swift
let builder = IRBuilder(context: context)
builder.setInsertionPoint(block)
let op = builder.create(Arith.addi, [x, y])
```

#### Level 4: Raw MLIR
```swift
let mlirString = """
func.func @compute(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %0, %c1 : i32
    return %1 : i32
}
"""
let module = try MLIRModule.parse(mlirString, context: context)
```

### Unified Experience

All approaches integrate seamlessly:
```swift
// Mix macro and builder code
@MLIRFunction
func part1() -> Int { ... }

let part2 = Function("part2") { ... }

// Combine in single module
module.add(part1_mlir())
module.add(part2)
```

## Version Information

```swift
print("SwiftIR version: \(S4MLIR.version)")
```

## Project Structure

```
SwiftIR (Top-level API)
├── SwiftIRCore (Foundation)
├── SwiftIRTypes (Type system)
├── SwiftIRDialects (Operations)
├── SwiftIRBuilders (DSL)
├── SwiftIRXLA (XLA/PJRT)
├── SwiftIRStableHLO (StableHLO)
├── SwiftIRGPU (GPU - future)
├── PJRTCWrappers (C bindings)
└── SwiftIRMacros (Macro implementation)
```

## Entry Points

### For Application Developers
```swift
import SwiftIR

// Use macros and high-level API
@MLIRFunction
func myKernel() { ... }
```

### For Library Developers
```swift
import SwiftIRCore
import SwiftIRBuilders

// Use builders for more control
let customOp = ...
```

### For Framework Developers
```swift
import SwiftIRCore

// Direct MLIR access
let context = MLIRContext()
let module = MLIRModule.parse(...)
```

## Dependencies

### Internal
- All other SwiftIR modules (re-exported)

### External
- None (all external deps are in submodules)

## Build Configuration

To use SwiftIR:

```swift
// Package.swift
.product(name: "SwiftIR", package: "SwiftIR")
```

This transitively includes all submodules.

## Related Documentation

- See [Sources/README.md](../README.md) for module overview
- See individual module READMEs for details
- See `/Examples` for usage examples
- See [GPU_LOWERING_ROADMAP.md](../../GPU_LOWERING_ROADMAP.md) for GPU plans

## Current Status

### Production Ready
- Core MLIR bindings
- Type system
- Basic dialects (func, arith)
- StableHLO integration
- PJRT runtime
- CPU execution

### In Development
- `@MLIRFunction` macro (basic implementation)
- GPU lowering pipelines
- Advanced DSL features
- Full tensor operations

### Planned
- Complete macro system
- Automatic differentiation
- Graph optimization DSL
- Multi-device execution

---

**This is the main entry point - import this module to use SwiftIR!**
