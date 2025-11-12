# SwiftIR Library Architecture

**A comprehensive guide to SwiftIR's modular architecture**

## Overview

SwiftIR is built as a layered architecture where each module has a specific responsibility. The design follows principles of separation of concerns, type safety, and progressive disclosure.

## Quick Start

```swift
import SwiftIR  // Single import gives you everything

let context = MLIRContext()
let module = MLIRModule(context: context)

// Write MLIR using Swift!
```

## Module Hierarchy

```
SwiftIR (Top-level API)
├── SwiftIRCore (Foundation - MLIR C API bindings)
├── SwiftIRTypes (Type system - type-safe MLIR types)
├── SwiftIRDialects (Dialect operations - func, arith, etc.)
├── SwiftIRBuilders (DSL - declarative IR construction)
├── SwiftIRXLA (XLA/PJRT - compilation and execution)
│   ├── Runtime (Tensor execution)
│   ├── Backend (XLA compiler)
│   ├── PJRT (Plugin JAX Runtime)
│   ├── Lowering (Pass management, pipelines)
│   ├── SPIRV (GPU SPIR-V pipeline)
│   └── LLVM (GPU LLVM/PTX pipeline)
├── SwiftIRStableHLO (StableHLO dialect - portable ML ops)
├── SwiftIRGPU (GPU dialect - planned)
├── PJRTCWrappers (C wrappers for PJRT API)
└── SwiftIRMacros (Macro implementations)
```

## Modules by Purpose

### Foundation Layer

#### [SwiftIRCore](SwiftIRCore/)
**Purpose**: MLIR C API bindings and memory management

- Core types: `MLIRContext`, `MLIRModule`, `MLIROperation`
- PJRT C API headers
- Safe wrappers around C handles
- **Import when**: You need low-level MLIR access

#### [PJRTCWrappers](PJRTCWrappers/)
**Purpose**: C wrappers for PJRT runtime

- Simplified PJRT C API wrappers
- Swift-friendly types and error handling
- Plugin interface definitions
- **Import when**: You're implementing runtime backends

### Type System Layer

#### [SwiftIRTypes](SwiftIRTypes/)
**Purpose**: Type-safe MLIR type construction

- Protocols: `MLIRType`, `BitWidthType`, `ShapedType`
- Concrete types: `IntegerType`, `FloatType`, `TensorType`, `MemRefType`
- Type validation and checking
- **Import when**: You need type-safe IR construction

### Operation Layer

#### [SwiftIRDialects](SwiftIRDialects/)
**Purpose**: Basic MLIR dialect operations

- Func dialect (functions, calls, returns)
- Arith dialect (arithmetic, comparisons)
- Dialect registration
- **Import when**: You need basic operations

#### [SwiftIRStableHLO](SwiftIRStableHLO/)
**Purpose**: StableHLO portable ML operations

- StableHLO and CHLO dialects
- Element-wise, reduction, shape operations
- XLA-compatible IR
- **Import when**: You need portable ML operations

#### [SwiftIRGPU](SwiftIRGPU/) (Planned)
**Purpose**: GPU dialect operations

- GPU kernel definitions
- Launch operations
- Thread/block management
- **Status**: Reserved for future implementation

### DSL and Metaprogramming Layer

#### [SwiftIRBuilders](SwiftIRBuilders/)
**Purpose**: Declarative IR construction DSL

- Result builders for operation sequences
- Control flow (`If`, `While`, `For`)
- High-level abstractions
- **Import when**: You want declarative IR construction

#### [SwiftIRMacros](SwiftIRMacros/)
**Purpose**: Swift macro implementations

- `@MLIRFunction` macro implementation
- Compile-time Swift → MLIR transformation
- AST analysis and code generation
- **Import when**: You're developing macros (users don't import directly)

### Execution and Compilation Layer

#### [SwiftIRXLA](SwiftIRXLA/)
**Purpose**: XLA compilation and PJRT execution

Largest module with multiple subsystems:
- **Tensor operations**: High-level tensor and ML ops
- **PJRT runtime**: Execute on CPU/GPU/TPU
- **XLA backend**: Google's XLA compiler integration
- **Lowering pipelines**: Transform high-level → executable code
- **GPU lowering**: SPIR-V and LLVM/PTX pipelines

**Import when**: You need compilation and execution

### Top-Level API

#### [SwiftIR](SwiftIR/)
**Purpose**: Main entry point and API

- Re-exports all modules
- `@MLIRFunction` macro definition
- Version information
- **Import when**: Always! (This is the main API)

## Architecture Principles

### 1. Layered Design

Each layer builds on the previous:
```
Application Code
       ↓
SwiftIR (Top-level API)
       ↓
Builders/Macros (DSL)
       ↓
Dialects/Types (Operations)
       ↓
Core (Foundation)
       ↓
MLIR C API
```

### 2. Separation of Concerns

Each module has a single responsibility:
- **Core**: C API bindings and memory
- **Types**: Type system
- **Dialects**: Operations
- **Builders**: DSL
- **XLA**: Execution

### 3. Progressive Disclosure

Users can choose their level of abstraction:

```swift
// Level 1: Macros (highest)
@MLIRFunction
func add(x: Int, y: Int) -> Int { x + y }

// Level 2: DSL
Function("add", inputs: [i32, i32], results: [i32]) { ... }

// Level 3: Builders
builder.create(Arith.addi, [x, y])

// Level 4: Raw MLIR (lowest)
MLIRModule.parse("func.func @add...")
```

### 4. Type Safety

Swift's type system prevents errors:
- Protocol-based type hierarchy
- Compile-time checking where possible
- Runtime validation when needed

## Module Dependencies

### Import Graph

```
SwiftIR
├── SwiftIRCore (no internal deps)
├── SwiftIRTypes
│   └── SwiftIRCore
├── SwiftIRDialects
│   ├── SwiftIRCore
│   └── SwiftIRTypes
├── SwiftIRBuilders
│   ├── SwiftIRCore
│   ├── SwiftIRTypes
│   └── SwiftIRDialects
├── SwiftIRStableHLO
│   ├── SwiftIRCore
│   └── SwiftIRTypes
├── SwiftIRXLA
│   ├── SwiftIRCore
│   ├── SwiftIRTypes
│   ├── SwiftIRDialects
│   ├── SwiftIRStableHLO
│   └── PJRTCWrappers
├── PJRTCWrappers (no internal deps)
└── SwiftIRMacros (no runtime deps - compile-time only)
```

### Dependency Rules

1. **Core has no dependencies** - It's the foundation
2. **Types only depend on Core** - Keep type system simple
3. **Higher layers depend on lower layers** - Respect the hierarchy
4. **No circular dependencies** - Clean module graph

## Build System

### Module Types

- **Libraries**: Most modules (SwiftIRCore, SwiftIRTypes, etc.)
- **Macro plugin**: SwiftIRMacros (compile-time execution)
- **C library**: PJRTCWrappers (C headers and wrappers)

### Linking

SwiftIR uses:
- **Dynamic linking**: MLIR libraries
- **Static linking**: Swift modules
- **Plugin loading**: PJRT runtime plugins
- **Macro expansion**: Compile-time code generation

## Common Usage Patterns

### Pattern 1: High-Level ML (Recommended)

```swift
import SwiftIR

@MLIRFunction
func neuralNetLayer(input: Tensor<Float>, weights: Tensor<Float>) -> Tensor<Float> {
    input.matmul(weights).relu()
}

let client = try PJRTClient.createCPU()
let output = try execute(neuralNetLayer_mlir(), inputs: [input, weights])
```

### Pattern 2: Custom Lowering

```swift
import SwiftIRCore
import SwiftIRXLA

let module = createHighLevelModule()
let pipeline = StableHLOPipeline(context: context)
try pipeline.run(on: module)
let executable = try client.compile(module)
```

### Pattern 3: Direct MLIR Construction

```swift
import SwiftIRCore
import SwiftIRDialects

let context = MLIRContext()
let module = MLIRModule(context: context)
let func = Func.function("compute", ...)
module.add(func)
```

### Pattern 4: GPU Code Generation

```swift
import SwiftIRXLA

let module = createLinalgModule()
let llvmPipeline = LLVMPipeline(context: context)
try llvmPipeline.run(on: module)
// Result: GPU kernel in LLVM IR/PTX
```

## Testing Strategy

Each module has its own tests:
- **Unit tests**: Test individual components
- **Integration tests**: Test module interactions
- **Examples**: End-to-end usage demonstrations

See `/Tests` and `/Examples` directories.

## Documentation Organization

Each module has:
- `README.md` - Module overview and usage
- Inline documentation - API documentation
- Examples - Working code samples

## Current Status

### Production Ready ✅
- SwiftIRCore
- SwiftIRTypes
- SwiftIRDialects (basic)
- SwiftIRStableHLO
- SwiftIRXLA (PJRT runtime)
- PJRTCWrappers

### In Development 🚧
- SwiftIRBuilders (advanced features)
- SwiftIRMacros (full implementation)
- SwiftIRXLA (GPU pipelines - see [GPU_LOWERING_ROADMAP.md](../GPU_LOWERING_ROADMAP.md))

### Planned 📋
- SwiftIRGPU (dedicated GPU module)
- Additional dialects (TOSA, MHLO, etc.)
- Advanced DSL features
- Automatic differentiation

## Getting Help

- **Module README**: See each module's README for details
- **Examples**: Check `/Examples` directory
- **API docs**: Inline documentation in source files
- **Roadmap**: See `/GPU_LOWERING_ROADMAP.md` and `/ROADMAP.md`

## Module Selection Guide

**I want to...**

- **Just use SwiftIR** → Import `SwiftIR` only
- **Build IR manually** → Import `SwiftIRCore`, `SwiftIRDialects`
- **Use type-safe types** → Import `SwiftIRTypes`
- **Write declarative IR** → Import `SwiftIRBuilders`
- **Execute on hardware** → Import `SwiftIRXLA`
- **Use ML operations** → Import `SwiftIRStableHLO`, `SwiftIRXLA`
- **Generate GPU code** → Import `SwiftIRXLA` (SPIRV or LLVM pipelines)
- **Develop macros** → Study `SwiftIRMacros`
- **Implement runtime** → Study `PJRTCWrappers`

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           SwiftIR (Main API)                │
│  @MLIRFunction macro, re-exports all        │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────┐        ┌──────────────┐
│   Builders   │        │    Macros    │
│ DSL Layer    │        │ Compile-time │
└──────────────┘        └──────────────┘
        │
        ├─────────┬─────────┬─────────┐
        ↓         ↓         ↓         ↓
   ┌────────┐ ┌──────┐ ┌─────────┐ ┌──────┐
   │Dialects│ │Types │ │StableHLO│ │ XLA  │
   │        │ │      │ │         │ │      │
   └────────┘ └──────┘ └─────────┘ └──────┘
        │         │         │         │
        └─────────┴─────────┴─────────┘
                    │
            ┌───────┴───────┐
            ↓               ↓
        ┌──────┐      ┌──────────┐
        │ Core │      │  PJRT    │
        │      │      │ Wrappers │
        └──────┘      └──────────┘
            │               │
            └───────┬───────┘
                    ↓
              MLIR C API
              PJRT C API
```

---

**Start exploring**: Import `SwiftIR` and see the examples in `/Examples`!
