# SwiftIRTypes

**Type-safe Swift abstractions for MLIR type system**

## Purpose

SwiftIRTypes provides a type-safe, Swift-idiomatic layer over MLIR's type system. It uses Swift protocols and generics to ensure type correctness at compile time, preventing common type-related errors in IR construction.

## What's Inside

### Core Files
- `MLIRType.swift` - Core type protocols and concrete types
  - `MLIRType` protocol - Base protocol for all MLIR types
  - `BitWidthType` protocol - For types with known bit width (integers, floats)
  - `ShapedType` protocol - For tensors, memrefs, and vectors
  - Concrete implementations: `IntegerType`, `FloatType`, `TensorType`, etc.

- `TypeValidation.swift` - Runtime type validation and checking
  - Type compatibility checking
  - Shape validation
  - Bitwidth validation

## Architecture

```
User Code (type-safe Swift)
         ↓
Swift Protocols (MLIRType, ShapedType, etc.)
         ↓
Type Implementations (IntegerType, TensorType, etc.)
         ↓
SwiftIRCore (MlirType handles)
         ↓
MLIR C API
```

## Key Design Decisions

### Protocol-Oriented Design
Uses Swift protocols to model MLIR's type hierarchy:
```swift
protocol MLIRType {
    var typeHandle: MlirType { get }
    var context: MLIRContext { get }
}

protocol BitWidthType: MLIRType {
    var bitWidth: UInt { get }
}

protocol ShapedType: MLIRType {
    var shape: [Int64] { get }
    var elementType: any MLIRType { get }
}
```

### Type Safety
- Compile-time checking via Swift generics
- Runtime validation for dynamic scenarios
- Clear error messages for type mismatches

### Performance
- Thin wrappers around C handles (zero overhead)
- Lazy validation where possible
- Efficient shape and type queries

## Common Types

### Scalar Types
- `IntegerType` - Signed/unsigned integers (i1, i8, i32, i64, etc.)
- `FloatType` - Floating point (f16, f32, f64, bf16)
- `IndexType` - Machine-dependent index type

### Shaped Types
- `TensorType` - Multi-dimensional tensors with static or dynamic shapes
- `MemRefType` - Memory references (for bufferization)
- `VectorType` - Fixed-size vectors

### Function Types
- `FunctionType` - Function signatures (inputs → results)

## Usage Example

```swift
import SwiftIRCore
import SwiftIRTypes

let context = MLIRContext()

// Create integer types
let i32 = IntegerType(bitWidth: 32, isSigned: true, context: context)
let i64 = IntegerType(bitWidth: 64, isSigned: false, context: context)

// Create tensor types
let shape = [1, 224, 224, 3]
let f32 = FloatType(bitWidth: 32, context: context)
let imageTensor = TensorType(shape: shape, elementType: f32, context: context)

// Type checking
if imageTensor.hasStaticShape {
    print("Tensor has static shape: \(imageTensor.shape)")
    print("Element type bit width: \(f32.bitWidth)")
}

// Create function type
let funcType = FunctionType(
    inputs: [i32.typeHandle, i32.typeHandle],
    results: [i64.typeHandle],
    context: context
)
```

## Dependencies

### Internal
- **SwiftIRCore** - For `MLIRContext` and core MLIR types

### External
- None (pure Swift abstractions)

## Type Hierarchy

```
MLIRType (protocol)
├── BitWidthType (protocol)
│   ├── IntegerType
│   ├── FloatType
│   └── IndexType
├── ShapedType (protocol)
│   ├── TensorType
│   ├── MemRefType
│   └── VectorType
└── FunctionType
```

## Validation Features

### Static Shape Checking
```swift
let tensor = TensorType(shape: [2, 3], elementType: f32, context: context)
assert(tensor.hasStaticShape == true)
assert(tensor.rank == 2)
```

### Dynamic Shapes
```swift
// Use -1 for dynamic dimensions
let dynamicTensor = TensorType(
    shape: [-1, 224, 224, 3],  // Batch size dynamic
    elementType: f32,
    context: context
)
assert(dynamicTensor.hasStaticShape == false)
```

### Bit Width Queries
```swift
let type = IntegerType(bitWidth: 32, isSigned: true, context: context)
print("Storage size: \(type.bitWidth) bits")
```

## Related Modules

- **SwiftIRCore** - Foundation layer (used by this module)
- **SwiftIRDialects** - Uses these types for dialect operations
- **SwiftIRBuilders** - Uses these types in DSL construction

## Design Philosophy

### Type Safety First
Every MLIR type is wrapped in a Swift type that encodes semantic information at compile time. This prevents errors like:
- Mixing integer and float types accidentally
- Creating tensors with invalid shapes
- Mismatched function signatures

### Zero Overhead Abstraction
All type wrappers are thin structs that wrap MLIR C API handles directly. There's no runtime cost compared to using the C API directly.

### Swifty API
The API follows Swift conventions:
- Properties instead of getter functions
- Protocol conformance for capabilities
- Computed properties for derived information

---

**Next Steps**: See [SwiftIRDialects](../SwiftIRDialects/) for operations that use these types.
