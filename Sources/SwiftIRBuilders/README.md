# SwiftIRBuilders

**High-level DSL for constructing MLIR IR using Swift result builders**

## Purpose

SwiftIRBuilders provides a declarative, Swift-native DSL for building MLIR IR. Instead of manually creating operations and managing control flow, you write Swift-like code that generates MLIR automatically.

## What's Inside

### Core Files
- `ResultBuilders.swift` - Swift result builders for declarative IR construction
  - `@IRBuilder` - Result builder for operation sequences
  - Block and region construction
  - Automatic SSA value threading

- `ControlFlowDSL.swift` - Control flow constructs (if/while/for)
  - `If` statements with else branches
  - `While` loops
  - `For` loops with induction variables
  - Region and block management

- `HighLevelDSL.swift` - High-level abstractions
  - `Function` - Declarative function definitions
  - Tensor operations
  - ML-specific patterns

## Result Builders: The Magic

Swift's result builders (formerly function builders) allow you to write declarative code:

```swift
@IRBuilder
func buildAddFunction() {
    let sum = arith.addi(arg0, arg1)
    func.return(sum)
}
```

Instead of:
```swift
let builder = MLIRBuilder()
builder.setInsertionPoint(...)
let sumOp = builder.create(.addi, [arg0, arg1])
let sum = sumOp.getResult(0)
builder.create(.return, [sum])
```

## Usage Example

### Basic Function with DSL

```swift
import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders

let context = MLIRContext()
let module = MLIRModule(context: context)

let i32 = IntegerType(bitWidth: 32, isSigned: true, context: context)

// Declarative function definition
let addFunc = Function("add", inputs: [i32, i32], results: [i32]) { builder, args in
    let sum = Arith.addi(args[0], args[1], context: context)
    Func.return([sum], context: context)
}

module.add(addFunc)
```

### Control Flow Example

```swift
// If-else construct
let comparison = Arith.cmpi(.slt, x, y, context: context)

If(comparison) {
    // True branch
    let result = Arith.addi(x, constant, context: context)
    yield(result)
} else: {
    // False branch
    let result = Arith.subi(y, constant, context: context)
    yield(result)
}
```

### Loop Example

```swift
// For loop with induction variable
For(start: 0, end: 10, step: 1) { i in
    let value = load(array, i)
    let doubled = Arith.muli(value, two, context: context)
    store(doubled, array, i)
}
```

## Architecture

```
User DSL Code (Swift syntax)
         ↓
Result Builders (transform to builder calls)
         ↓
Control Flow DSL (manage blocks/regions)
         ↓
SwiftIRDialects (create operations)
         ↓
SwiftIRCore (MLIR module/operations)
```

## Key Components

### Result Builders
Swift's `@resultBuilder` attribute enables:
- Declarative syntax for operation sequences
- Automatic value passing
- Compile-time validation

### Control Flow DSL
Manages MLIR's structured control flow:
- **Regions**: Isolated sub-graphs of operations
- **Blocks**: Basic blocks with arguments
- **Branches**: Explicit control flow edges

### IRBuilder Context
Tracks:
- Current insertion point
- Active region/block
- Symbol table for named values
- Type inference state

## Pattern: Function Builder

```swift
public struct Function {
    let name: String
    let inputs: [MlirType]
    let results: [MlirType]
    let bodyBuilder: (IRBuilder, [MLIRValue]) -> Void

    public func build(in module: MLIRModule) -> MLIROperation {
        // Create function operation
        // Create entry block with arguments
        // Call bodyBuilder
        // Return completed function
    }
}
```

## Pattern: Control Flow

```swift
public struct If {
    let condition: MLIRValue
    let thenBuilder: (IRBuilder) -> Void
    let elseBuilder: ((IRBuilder) -> Void)?

    public func build(in builder: IRBuilder) -> MLIRValue? {
        // Create scf.if operation
        // Build then region
        // Build else region (if present)
        // Return merged result
    }
}
```

## Dependencies

### Internal
- **SwiftIRCore** - For module and operation primitives
- **SwiftIRTypes** - For type-safe type construction
- **SwiftIRDialects** - For operation creation

### External
- None (pure Swift abstractions)

## Design Principles

### Swifty Syntax
The DSL should feel like writing Swift code:
```swift
// Looks like Swift
if condition {
    doSomething()
} else {
    doSomethingElse()
}

// Generates MLIR
scf.if %condition {
    ...
} else {
    ...
}
```

### Type Safety
Result builders validate operation sequences at compile time where possible.

### Progressive Disclosure
- Simple cases use simple syntax
- Complex cases drop down to lower-level APIs
- Escape hatches for raw MLIR operations

## Common Patterns

### Tensor Operations
```swift
let result = tensor.add(lhs, rhs)
    .then(tensor.relu(_))
    .then(tensor.matmul(_, weights))
```

### Loop Nest
```swift
For(0..<M) { i in
    For(0..<N) { j in
        // Compute C[i,j]
        let c = computeElement(A, B, i, j)
        store(c, C, [i, j])
    }
}
```

### Conditional Operations
```swift
let result = Select(condition, trueValue, falseValue)
```

## Related Modules

- **SwiftIRDialects** - Provides operations used by builders
- **SwiftIRTypes** - Provides types for builders
- **SwiftIRXLA** - Uses builders for tensor operations
- **SwiftIR** - Top-level API uses builders for macros

## Implementation Notes

### SSA Form
MLIR requires Static Single Assignment (SSA) form:
- Every value defined exactly once
- Values dominated by definitions
- Phi nodes at block merges

The builders manage this automatically.

### Region Management
MLIR operations can contain regions (sub-graphs). Builders track:
- Current region
- Entry/exit blocks
- Value capture from parent regions

---

**Next Steps**: See [SwiftIRXLA](../SwiftIRXLA/) for ML-specific operations using these builders.
