# SwiftIRMacros

**Swift macro implementations for MLIR code generation**

## Purpose

SwiftIRMacros provides Swift macro implementations that enable compile-time transformation of Swift functions into MLIR representations. This is part of the "S4MLIR" (Swift for MLIR) vision - using Swift as a metalanguage for MLIR.

## What's Inside

### Core Files
- `MLIRFunctionMacro.swift` - Implementation of `@MLIRFunction` macro
  - Peer macro that generates companion MLIR representation
  - Analyzes Swift function AST
  - Generates corresponding MLIR IR
  - Maintains source location information

## What are Swift Macros?

Swift macros are compile-time metaprogramming features (Swift 5.9+) that transform code:

```swift
@SomeMacro
func myFunction() { }

// Expands to:
func myFunction() { }
// ... plus generated code
```

### Macro Types
- **Attached macros** - Attached to declarations
  - `@attached(peer)` - Generate peer declarations
  - `@attached(member)` - Add members to types
  - `@attached(accessor)` - Add property accessors
- **Freestanding macros** - Standalone expressions

SwiftIRMacros uses **peer macros** to generate MLIR companions to Swift functions.

## The @MLIRFunction Macro

### Purpose
Transform Swift functions into MLIR representations automatically:

```swift
import SwiftIR

@MLIRFunction
func add(_ x: Int32, _ y: Int32) -> Int32 {
    return x + y
}

// Generates:
func add_mlir() -> String {
    return """
    func.func @add(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
    """
}
```

### How It Works

1. **AST Analysis**
   - Parse function declaration
   - Extract parameter types
   - Extract return type
   - Analyze function body

2. **Type Mapping**
   ```swift
   Swift Type → MLIR Type
   Int32      → i32
   Float      → f32
   Double     → f64
   [Int32]    → tensor<?xi32>
   ```

3. **Operation Mapping**
   ```swift
   Swift Operation → MLIR Operation
   x + y          → arith.addi %x, %y
   x * y          → arith.muli %x, %y
   if condition   → scf.if %condition
   ```

4. **IR Generation**
   - Generate MLIR function signature
   - Convert body to MLIR operations
   - Maintain SSA form
   - Add source location attributes

## Implementation Details

### Macro Structure

```swift
public struct MLIRFunctionMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        // 1. Validate it's a function
        // 2. Extract function information
        // 3. Generate MLIR IR
        // 4. Create peer function returning IR
    }
}
```

### AST Traversal

Uses SwiftSyntax to analyze code:
```swift
// Extract function name
let funcName = funcDecl.name.text

// Get parameters
for param in funcDecl.signature.parameterClause.parameters {
    let type = mapSwiftTypeToMLIR(param.type)
    // ...
}

// Analyze body
let body = funcDecl.body
// Recursively convert statements to MLIR ops
```

### Error Diagnostics

Provides helpful compile-time errors:
```swift
context.diagnose(
    Diagnostic(
        node: node,
        message: MLIRDiagnostic.notAFunction
    )
)
```

## Usage Example

### Basic Example

```swift
import SwiftIR

@MLIRFunction
func square(_ x: Float) -> Float {
    return x * x
}

// Use generated MLIR
let mlirIR = square_mlir()
let module = try MLIRModule.parse(mlirIR, context: context)
```

### Control Flow Example

```swift
@MLIRFunction
func abs(_ x: Int32) -> Int32 {
    if x < 0 {
        return -x
    } else {
        return x
    }
}

// Generates scf.if operation in MLIR
```

### Tensor Operations Example

```swift
@MLIRFunction
func matmul(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    // Matrix multiplication logic
}

// Generates linalg.matmul or stablehlo.dot operation
```

## Limitations (Current)

### What Works
- Basic arithmetic operations
- Simple control flow (if/else)
- Scalar types (Int, Float, etc.)
- Function signatures

### What Doesn't Work Yet
- Complex data structures
- Closures and higher-order functions
- Generic functions
- Advanced Swift features (async, actors, etc.)
- Type inference (types must be explicit)

### Why?
Macro implementation is complex:
- Need full Swift → MLIR mapping
- SSA form management
- Type system differences
- Swift's rich feature set

## Macro Expansion Plugin

This module compiles to a **macro plugin** that runs during compilation:

```swift
// Package.swift
.macro(
    name: "SwiftIRMacros",
    dependencies: [
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
    ]
)
```

The plugin is automatically invoked when the compiler encounters `@MLIRFunction`.

## Dependencies

### External
- **swift-syntax** - Swift AST manipulation
  - `SwiftSyntax` - AST types
  - `SwiftSyntaxBuilder` - AST construction
  - `SwiftSyntaxMacros` - Macro protocols
  - `SwiftCompilerPlugin` - Plugin infrastructure
  - `SwiftDiagnostics` - Error reporting

### Internal
- None (macros are compile-time only)

## Integration with SwiftIR

The macro is defined in `SwiftIR` module:

```swift
// SwiftIR/S4MLIR.swift
@attached(peer, names: arbitrary)
public macro MLIRFunction() = #externalMacro(
    module: "SwiftIRMacros",
    type: "MLIRFunctionMacro"
)
```

This separates:
- **Definition** (SwiftIR) - What users import
- **Implementation** (SwiftIRMacros) - How it works

## Future Enhancements

### Planned Features
1. **Full Swift → MLIR mapping**
   - All scalar types
   - Arrays → tensors
   - Structs → MLIR types

2. **Advanced operations**
   - Loops → scf.for
   - Pattern matching → scf.switch
   - Closures → higher-order ops

3. **Type inference**
   - Infer MLIR types from Swift context
   - Reduce annotation burden

4. **Optimization hints**
   ```swift
   @MLIRFunction(optimize: .aggressive)
   @MLIRFunction(target: .gpu)
   ```

5. **Dialect selection**
   ```swift
   @MLIRFunction(dialect: .stablehlo)
   @MLIRFunction(dialect: .tosa)
   ```

## Related Modules

- **SwiftIR** - Exports the macro for users
- **SwiftIRCore** - MLIR foundation (runtime)
- **SwiftIRBuilders** - Alternative DSL approach

## Design Philosophy

### Compile-Time Safety
Macros catch errors at compile time:
- Type mismatches
- Invalid operations
- Malformed syntax

### Transparency
Macro expansion is visible:
- `swift build -Xswiftc -Xfrontend -Xswiftc -dump-macro-expansions`
- Shows generated MLIR

### Gradual Adoption
You can mix macro and non-macro code:
- Use macros for simple functions
- Use builders for complex cases
- Use raw MLIR for full control

---

**Next Steps**: See [SwiftIR](../SwiftIR/) for the macro definition and usage examples.
