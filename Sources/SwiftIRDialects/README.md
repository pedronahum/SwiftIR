# SwiftIRDialects

**Swift bindings for MLIR dialect operations**

## Purpose

SwiftIRDialects provides Swift-friendly APIs for creating operations from various MLIR dialects. Each dialect represents a specific domain of operations (functions, arithmetic, tensor ops, etc.), and this module wraps them in type-safe Swift interfaces.

## What's Inside

### Core Files
- `FuncDialect.swift` - Function definition and call operations
  - `func.func` - Function definitions
  - `func.call` - Function calls
  - `func.return` - Return statements
  - Function type creation

- `ArithDialect.swift` - Arithmetic and logical operations
  - Integer arithmetic (`addi`, `subi`, `muli`, `divsi`, `divui`)
  - Floating-point arithmetic (`addf`, `subf`, `mulf`, `divf`)
  - Comparison operations (`cmpi`, `cmpf`)
  - Logical operations (`andi`, `ori`, `xori`)
  - Type conversions and casts

## Dialect Organization

MLIR organizes operations into **dialects**:
- Each dialect has a namespace (e.g., `func`, `arith`, `tensor`)
- Operations are named `dialect.operation` (e.g., `func.func`, `arith.addi`)
- Dialects must be registered before use

## Architecture

```
User Code
         ↓
Dialect-specific Swift APIs (Func, Arith, etc.)
         ↓
SwiftIRTypes (for type safety)
         ↓
SwiftIRCore (MLIR operations)
         ↓
MLIR C API
```

## Supported Dialects

### Func Dialect
Purpose: Function definitions and calls

```swift
// Create a function
let funcType = FunctionType(
    inputs: [i32.typeHandle, i32.typeHandle],
    results: [i32.typeHandle],
    context: context
)

let addFunc = Func.function(
    name: "add",
    type: funcType,
    location: loc,
    context: context
)
```

### Arith Dialect
Purpose: Arithmetic and logical operations on integers and floats

```swift
// Integer addition
let sum = Arith.addi(lhs, rhs, context: context)

// Floating-point multiplication
let product = Arith.mulf(a, b, context: context)

// Comparison
let isGreater = Arith.cmpi(.sgt, x, y, context: context)
```

## Usage Example

```swift
import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects

let context = MLIRContext()
let module = MLIRModule(context: context)

// Register dialects
Func.load(in: context)
Arith.load(in: context)

// Create function type
let i32 = IntegerType(bitWidth: 32, isSigned: true, context: context)
let funcType = FunctionType(
    inputs: [i32, i32],
    results: [i32],
    context: context
)

// Build function body
let builder = IRBuilder(context: context)
let addFunc = Func.function(name: "add", type: funcType, context: context)

// Add operations to function body
builder.setInsertionPoint(addFunc.body)
let arg0 = addFunc.getArgument(0)
let arg1 = addFunc.getArgument(1)
let result = Arith.addi(arg0, arg1, context: context)
Func.return([result], context: context)
```

## Dialect Registration

Before using a dialect, it must be registered with the context:

```swift
// Option 1: Load specific dialect
Func.load(in: context)
Arith.load(in: context)

// Option 2: Load all available dialects
context.loadAllDialects()

// Option 3: Use extension methods
context.registerFuncDialect()
context.registerArithDialect()
```

## Operation Builders

Each dialect operation has a builder that:
1. Validates operand types
2. Infers result types
3. Creates MLIR operation with proper attributes
4. Returns typed results

Example:
```swift
// Arith.addi builder
public static func addi(
    _ lhs: MLIRValue,
    _ rhs: MLIRValue,
    context: MLIRContext,
    location: MLIRLocation? = nil
) -> MLIRValue {
    // Validate both operands are integers
    // Create arith.addi operation
    // Return result value
}
```

## Type Safety

The dialect APIs enforce type correctness:
```swift
// This works - both operands are integers
let sum = Arith.addi(intVal1, intVal2, context: context)

// This would fail at compile time if we had full type checking
// let invalid = Arith.addi(floatVal, intVal, context: context)
```

## Dependencies

### Internal
- **SwiftIRCore** - For `MLIRContext`, `MLIRModule`, `MLIROperation`
- **SwiftIRTypes** - For type-safe type construction

### External
- MLIR dialect implementations (from MLIR C API)

## Common Dialects (Defined Elsewhere)

While this module contains `Func` and `Arith`, other dialects are defined in specialized modules:
- **tensor** dialect → SwiftIRXLA module
- **linalg** dialect → SwiftIRXLA module
- **stablehlo** dialect → SwiftIRStableHLO module
- **gpu** dialect → SwiftIRGPU module (planned)

## Dialect Pattern

Each dialect follows this pattern:

```swift
public enum DialectName {
    public static let dialectName = "dialect"

    public static func load(in context: MLIRContext) {
        // Register dialect
    }

    public static func operationName(
        _ operands: MLIRValue...,
        context: MLIRContext,
        location: MLIRLocation? = nil
    ) -> MLIRValue {
        // Build operation
    }
}
```

## Related Modules

- **SwiftIRCore** - Foundation (used by this module)
- **SwiftIRTypes** - Type system (used by this module)
- **SwiftIRBuilders** - Uses these dialects in higher-level DSL
- **SwiftIRXLA** - Additional dialects (tensor, linalg)
- **SwiftIRStableHLO** - StableHLO dialect

## Design Principles

### Namespace Organization
Each dialect is an enum (not instantiable) that acts as a namespace for related operations.

### Type Safety
Operations validate operand types and return appropriately typed results.

### Swifty Naming
- Enum for namespaces
- Static methods for operations
- Default parameters for optional arguments

---

**Next Steps**: See [SwiftIRBuilders](../SwiftIRBuilders/) for high-level DSL constructs using these dialects.
