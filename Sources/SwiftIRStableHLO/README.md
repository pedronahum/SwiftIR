# SwiftIRStableHLO

**Swift bindings for StableHLO dialect - portable ML operations**

## Purpose

SwiftIRStableHLO provides Swift interfaces to the StableHLO dialect, a versioned set of operations for machine learning that serves as a portability layer between ML frameworks (JAX, PyTorch, TensorFlow) and ML compilers (XLA).

## What's Inside

### Core Files
- `StablehloDialect.swift` - Dialect registration and loading
  - Dialect and pass registration functions
  - CHLO (Client HLO) dialect support
  - Pass management utilities

- `StablehloOps.swift` - StableHLO operation builders
  - Element-wise operations (add, multiply, etc.)
  - Reduction operations (reduce, reduce_window)
  - Shape operations (broadcast, reshape, transpose)
  - Convolution and dot operations

- `StableHLODSL.swift` - High-level DSL for StableHLO
  - Declarative operation construction
  - Tensor manipulation utilities
  - Composite operations

## What is StableHLO?

StableHLO is a **portability layer** for machine learning:

### Problem
Different ML frameworks have incompatible IR:
- JAX → JAX IR
- PyTorch → TorchScript
- TensorFlow → TensorFlow GraphDef

### Solution
StableHLO provides:
- **Versioned** operations with backward compatibility
- **Framework-agnostic** representation
- **Compiler-friendly** lowering to XLA HLO

### Flow
```
JAX / PyTorch / TensorFlow
           ↓
      StableHLO (portable)
           ↓
        XLA HLO
           ↓
    CPU / GPU / TPU code
```

## StableHLO vs CHLO

### StableHLO
- **Stable** operations with version guarantees
- Lower-level, explicit broadcasts
- Direct XLA mapping

### CHLO (Client HLO)
- **Higher-level** operations for framework convenience
- Implicit broadcasting
- Later lowered to StableHLO

Example:
```swift
// CHLO - implicit broadcast
let result = chlo.broadcast_add(scalar, tensor)

// StableHLO - explicit broadcast
let broadcasted = stablehlo.broadcast_in_dim(scalar, shape, dims)
let result = stablehlo.add(broadcasted, tensor)
```

## Common Operations

### Element-wise Operations
```swift
// Addition
let sum = stablehlo.add(lhs, rhs, context: context)

// Multiplication
let product = stablehlo.multiply(a, b, context: context)

// Exponential
let exp = stablehlo.exponential(x, context: context)
```

### Shape Operations
```swift
// Reshape
let reshaped = stablehlo.reshape(tensor, newShape, context: context)

// Transpose
let transposed = stablehlo.transpose(tensor, permutation: [1, 0], context: context)

// Broadcast
let broadcasted = stablehlo.broadcast_in_dim(
    scalar,
    shape: [10, 20],
    broadcastDims: [],
    context: context
)
```

### Reductions
```swift
// Reduce sum along axis
let sum = stablehlo.reduce(
    inputs: [tensor],
    initValues: [zero],
    dimensions: [1],  // Reduce along axis 1
    context: context
) { builder, args in
    // Reduction body: add elements
    let sum = stablehlo.add(args[0], args[1], context: context)
    stablehlo.return([sum], context: context)
}
```

### Matrix Operations
```swift
// Matrix multiplication
let result = stablehlo.dot_general(
    lhs, rhs,
    lhsBatchDims: [],
    rhsBatchDims: [],
    lhsContractingDims: [1],
    rhsContractingDims: [0],
    context: context
)

// Convolution
let output = stablehlo.convolution(
    input, kernel,
    strides: [1, 1],
    padding: [[0, 0], [0, 0]],
    context: context
)
```

## Pass Registration

StableHLO includes many transformation passes:

```swift
// Register all StableHLO passes
registerAllStablehloPasses()

// Use passes in pipeline
let pm = PassManager(context: context)
pm.parsePipeline("""
builtin.module(
    stablehlo-canonicalize-dynamism,
    stablehlo-refine-shapes,
    stablehlo-legalize-to-linalg
)
""")
try pm.run(on: module)
```

### Common Passes
- `stablehlo-canonicalize-dynamism` - Convert dynamic ops to static
- `stablehlo-refine-shapes` - Propagate shape information
- `stablehlo-legalize-to-linalg` - Lower to Linalg operations
- `shape-legalize-to-stablehlo` - Convert shape dialect to StableHLO

## Usage Example

```swift
import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO

// Create context and load dialect
let context = MLIRContext()
loadStablehloDialect(context)
loadChloDialect(context)

// Create tensor types
let f32 = FloatType(bitWidth: 32, context: context)
let tensorType = TensorType(shape: [2, 3], elementType: f32, context: context)

// Create StableHLO operations
let a = ... // Input tensor
let b = ... // Input tensor

// Element-wise addition
let sum = stablehlo.add(a, b, context: context)

// Apply activation
let activated = stablehlo.exponential(sum, context: context)

// Reduce to scalar
let total = stablehlo.reduce(
    inputs: [activated],
    initValues: [zero],
    dimensions: [0, 1],
    context: context
) { builder, args in
    let sum = stablehlo.add(args[0], args[1], context: context)
    stablehlo.return([sum], context: context)
}
```

## Lowering to XLA

StableHLO can be lowered to XLA HLO for execution:

```swift
// Create StableHLO module
let module = createStableHLOModule()

// Lower to XLA HLO via PJRT
let client = try PJRTClient.createCPU()
let executable = try client.compile(module)

// Execute
let outputs = try executable.execute(inputs)
```

The lowering happens automatically inside XLA.

## Dependencies

### Internal
- **SwiftIRCore** - MLIR foundation
- **SwiftIRTypes** - Type system

### External
- StableHLO MLIR dialect (from StableHLO project)
- CHLO dialect (from StableHLO project)

## Design Decisions

### Direct MLIR Bindings
Operations map 1:1 to StableHLO MLIR operations for:
- Predictable behavior
- Easy debugging (compare MLIR output)
- Maximum compatibility

### Type Safety
Swift type system ensures:
- Correct operand types
- Valid shape transformations
- Proper attribute specification

### DSL Layer
High-level DSL provides:
- Composite operations
- Swifty syntax
- Common patterns

## StableHLO Specification

StableHLO has a formal specification:
- Version numbering
- Backward compatibility guarantees
- Operation semantics

See: https://github.com/openxla/stablehlo

## Related Modules

- **SwiftIRXLA** - Uses StableHLO for XLA compilation
- **SwiftIRCore** - Foundation
- **SwiftIRTypes** - Type system

## Technical Notes

### Versioning
StableHLO uses semantic versioning:
- Major version: Breaking changes
- Minor version: New operations
- Patch version: Bug fixes

Current SwiftIR supports StableHLO v1.x

### XLA Compatibility
StableHLO is designed to map directly to XLA HLO:
- Nearly 1:1 operation correspondence
- Same semantics
- Efficient lowering

### Why Not Use XLA HLO Directly?
XLA HLO is:
- Unstable (changes frequently)
- XLA-specific (not portable)
- No versioning guarantees

StableHLO provides the stability needed for framework integration.

---

**Next Steps**: See [SwiftIRXLA](../SwiftIRXLA/) for XLA compilation and execution.
