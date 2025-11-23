# SwiftIR

**Swift as a Metalanguage for MLIR - Building the Future of ML Compilers in Swift**

[![Swift Version](https://img.shields.io/badge/swift-6.0-orange.svg)](https://swift.org)
[![MLIR](https://img.shields.io/badge/MLIR-StableHLO-blue.svg)](https://github.com/openxla/stablehlo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-pedronahum-blue.svg?logo=github)](https://github.com/pedronahum)

SwiftIR is a comprehensive Swift framework for building, manipulating, and executing [MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) code. It provides multiple levels of abstraction - from direct MLIR string manipulation to declarative DSLs, with a vision for Swift macros that generate MLIR at compile-time - all with type safety and Swift's modern language features.

> **🚀 Working Examples:** SwiftIR includes 20+ working examples executing real ML operations
> (CNNs, ResNets, matrix ops) on CPU via PJRT/XLA - the same runtime powering TensorFlow and JAX.
> This is a research project exploring modern Swift for ML compiler construction.

## Why SwiftIR?

SwiftIR brings modern Swift features to ML compiler construction with concrete benefits:

- **Write once, run anywhere** - Compile Swift to CPUs, GPUs, and TPUs using industry-standard PJRT
- **Catch errors at compile-time, not runtime** - Type-safe compiler construction with Swift's type system
- **From prototype to production in one framework** - Same code scales from experiments to deployment
- **Industry-standard runtime** - Uses the same PJRT backend as JAX, PyTorch/XLA, and TensorFlow
- **20+ working examples** - Real neural networks (CNNs, ResNets), not toy demos

## Status at a Glance

| Feature | Status | What It Means |
|---------|--------|---------------|
| **MLIR Bindings** | ✅ Working | Swift bindings to MLIR C API - parse, build, manipulate IR |
| **String-based MLIR** | ✅ Working | Write MLIR as strings, parse and execute via PJRT/XLA |
| **CPU Execution** | ✅ Working | Run compiled MLIR on CPU (20+ examples work) |
| **StableHLO Ops** | ✅ Working | Use StableHLO operations for ML workloads |
| **Type System** | 🚧 Partial | Type-safe IR construction (basic types work) |
| **DSL Builders** | 🚧 Experimental | Declarative IR construction (limited coverage) |
| **Neural Networks** | 🚧 Examples | CNNs, ResNets work but API still evolving |
| **GPU Execution** | 🚧 Blocked | Code ready, needs MLIR rebuild (see GPU_LOWERING_ROADMAP.md) |
| **Swift Macros** | 🔮 Concept | Vision demonstrated, implementation in early stages |

## Quick Win Example

Two ways to get started - pick your style:

### Approach 1: String-based MLIR (10 lines, works today)

```swift
import SwiftIR

let context = MLIRContext()
context.loadAllDialects()

let module = try MLIRModule.parse("""
func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}
""", context: context)

let client = try PJRTClient.createCPU()
let result = try client.compile(module).execute(
    inputs: [Tensor([1,2,3,4]), Tensor([5,6,7,8])]
)
print(result) // [6, 8, 10, 12]
```

### Approach 2: Type-safe DSL (experimental)

```swift
import SwiftIR

let tensorType = TensorType(shape: [4])

let module = StableHLOModule(name: "add_module") {
    StableHLOFunction(
        name: "add",
        parameters: [
            Parameter(name: "a", type: tensorType),
            Parameter(name: "b", type: tensorType)
        ],
        returnType: tensorType
    ) {
        Add("a", "b", type: tensorType)
    }
}

let mlir = module.build()  // Generate MLIR from DSL
let client = try PJRTClient.createCPU()
let result = try client.compile(mlir).execute(
    inputs: [Tensor([1,2,3,4]), Tensor([5,6,7,8])]
)
print(result) // [6, 8, 10, 12]
```

**Both approaches** compile and execute on CPU using the same XLA backend as TensorFlow. Choose string-based for simplicity, or DSL for type safety and composability.

## Who Is This For?

SwiftIR is designed for developers who want type safety and modern tooling in ML compiler development:

- **ML Engineers** tired of Python's runtime errors in production deployments
- **iOS/macOS Developers** wanting to add on-device ML to their apps without leaving Swift
- **Compiler Enthusiasts** seeking a modern, safe approach to MLIR development
- **Students & Researchers** learning how ML compilers work with real, executable examples
- **Systems Programmers** who want C-level performance with Swift-level safety

## How SwiftIR Compares

Understanding where SwiftIR fits in the ML compiler ecosystem:

| Aspect | SwiftIR | PyTorch | TensorFlow | IREE |
|--------|---------|---------|------------|------|
| **Language** | Swift | Python | Python | C++ |
| **Type Safety** | Compile-time | Runtime | Runtime | Compile-time |
| **iOS/macOS Native** | ✅ First-class | ❌ Wrapper | ❌ Wrapper | ⚠️ Limited |
| **Learning Curve** | Moderate | Easy | Hard | Very Hard |
| **Runtime** | PJRT/XLA | ATen | XLA | IREE |
| **Production Ready** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **MLIR Access** | ✅ Direct | ❌ No | ⚠️ Limited | ✅ Direct |
| **Use Case** | Compilers & iOS ML | Training & Research | Production ML | Edge Deployment |

**Key Differentiator:** SwiftIR is the only framework offering native Swift access to MLIR with full compile-time type safety and first-class Apple platform support.

## How It Works

```
Swift Code → SwiftIR → MLIR → XLA → CPU/GPU/TPU
    ↑           ↑        ↑       ↑        ↑
Type-Safe   Portable  Optimized Same as  Your Hardware
            Format              JAX/TF
```

SwiftIR provides the type-safe Swift layer that generates portable MLIR, which is then optimized by industry-standard XLA and executed on any hardware - the same battle-tested pipeline powering TensorFlow and JAX in production.

## What Can You Build?

Real-world applications enabled by SwiftIR:

- **On-device ML for iOS/macOS** - Ship ML models without Python runtime, fully native Swift
- **Custom ML Compilers** - Build domain-specific optimizers with type safety and modern tooling
- **Research Prototypes** - Test new ML compilation techniques with readable, maintainable code
- **Educational Tools** - Teach compiler concepts with clear examples and immediate feedback
- **Cross-platform ML** - Write once in Swift, deploy to CPU/GPU/TPU via standard PJRT
- **High-performance Computing** - Leverage XLA's optimization for numerical computing beyond ML

## Origin Story

This project is inspired by discussions in the Swift community about ["Swift as Syntactic Sugar for MLIR"](https://forums.swift.org/t/swift-as-syntactic-sugar-for-mlir/27672) on the Swift forums. The original idea proposed using Swift's syntax to represent MLIR operations directly.

**We take this vision further** by leveraging modern Swift features that didn't exist when the original discussion took place:
- **Swift Macros** (Swift 5.9+) - Compile-time code generation for automatic MLIR IR creation
- **Result Builders** - Declarative DSL construction with natural Swift syntax
- **C++ Interoperability** - Direct, efficient bindings to MLIR C API
- **Advanced Type System** - Protocol-oriented design for type-safe IR construction

SwiftIR proves that Swift isn't just syntactic sugar for MLIR - it's a powerful metalanguage that makes compiler construction accessible, safe, and enjoyable.

## What SwiftIR Achieves

### 1. Multiple Levels of Abstraction

Write MLIR in the style that fits your needs:

#### Direct MLIR Construction (Production Ready ✅)
```swift
let context = MLIRContext()
let module = MLIRModule.parse("""
func.func @compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
""", context: context)
```

#### Declarative DSL (Partially Available 🚧)
```swift
let addFunc = Function("add", inputs: [i32, i32], results: [i32]) { builder, args in
    let sum = Arith.addi(args[0], args[1], context: context)
    Func.return([sum], context: context)
}
```

#### Macro-Based (Vision/In Development 🔮)
```swift
// Future vision - macro implementation in progress
@MLIRFunction
func neuralNetLayer(_ input: Tensor<Float>, _ weights: Tensor<Float>) -> Tensor<Float> {
    let result = input.matmul(weights)
    return result.relu()
}

// Would automatically generate MLIR at compile time
let mlir = neuralNetLayer_mlir()
```

### 2. Full Compilation and Execution Pipeline

SwiftIR isn't just about building IR - it compiles and executes it:

```swift
// Compile with XLA
let client = try PJRTClient.createCPU()
let executable = try client.compile(module)

// Execute on hardware
let outputs = try executable.execute(inputs: [inputTensor])
```

**Supported backends:**
- CPU via PJRT
- GPU via PJRT (CUDA, ROCm)
- TPU via PJRT (Google Cloud)

### 3. Industry-Standard Integrations

- **PJRT Runtime** - Same runtime used by JAX, PyTorch/XLA, and TensorFlow
- **StableHLO** - Portable ML operations with version guarantees
- **XLA Compiler** - Google's production ML compiler
- **MLIR Dialects** - Full access to MLIR's modular dialect system

### 4. Type Safety Throughout

```swift
// Type-safe type construction
let tensorType = TensorType(shape: [2, 3], elementType: f32, context: context)

// Compile-time checking
let sum = Arith.addi(int1, int2, context: context)  // ✓ Type safe
// let invalid = Arith.addi(float1, int2, context: context)  // ✗ Won't compile
```

### 5. GPU Code Generation (In Progress)

Transform high-level operations into GPU kernels:

```swift
// High-level Linalg operations
let module = createLinalgModule()

// Lower to GPU code
let pipeline = LLVMPipeline(context: context)
try pipeline.run(on: module)

// Result: PTX for NVIDIA GPUs or SPIR-V for Vulkan/OpenCL
```

## Key Benefits

### For ML Researchers
- **Rapid Prototyping** - Write models in Swift, execute on accelerators
- **Portability** - StableHLO ensures code works across frameworks
- **Performance** - XLA compilation with production-grade optimizations

### For Compiler Developers
- **Type Safety** - Catch errors at compile time, not runtime
- **Modularity** - Clean separation between dialects and transformations
- **Debugging** - Swift's error handling and debugging tools

### For Systems Programmers
- **Low-Level Control** - Direct access to MLIR when needed
- **Zero Overhead** - Thin wrappers around C API
- **Memory Safety** - Swift's ARC manages MLIR resources

### For Language Designers
- **Metaprogramming** - Macros for compile-time code generation
- **DSL Creation** - Result builders for embedded languages
- **Extensibility** - Add new dialects and operations easily

## Architecture

SwiftIR is built as a layered architecture:

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   String MLIR, DSL, Macros (future) │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   High-Level Layer                  │
│   SwiftIRBuilders, SwiftIRMacros*   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Operation Layer                   │
│   SwiftIRDialects, SwiftIRStableHLO │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Type Layer                        │
│   SwiftIRTypes                      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Foundation Layer                  │
│   SwiftIRCore, PJRTCWrappers        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Runtime Layer                     │
│   SwiftIRXLA (PJRT, XLA, Pipelines) │
└─────────────────────────────────────┘
```

*SwiftIRMacros module exists but full implementation is in development.

See [Sources/README.md](Sources/README.md) for detailed architecture documentation.

## Modules

SwiftIR consists of 10 modules, each with a specific purpose:

| Module | Purpose | Documentation |
|--------|---------|---------------|
| **SwiftIR** | Main API and macro definitions | [README](Sources/SwiftIR/README.md) |
| **SwiftIRCore** | MLIR C API bindings | [README](Sources/SwiftIRCore/README.md) |
| **SwiftIRTypes** | Type-safe MLIR types | [README](Sources/SwiftIRTypes/README.md) |
| **SwiftIRDialects** | Basic dialects (func, arith) | [README](Sources/SwiftIRDialects/README.md) |
| **SwiftIRBuilders** | Declarative DSL | [README](Sources/SwiftIRBuilders/README.md) |
| **SwiftIRXLA** | XLA/PJRT integration + GPU pipelines | [README](Sources/SwiftIRXLA/README.md) |
| **SwiftIRStableHLO** | StableHLO dialect | [README](Sources/SwiftIRStableHLO/README.md) |
| **SwiftIRGPU** | GPU dialect (planned) | [README](Sources/SwiftIRGPU/README.md) |
| **PJRTCWrappers** | PJRT C API wrappers | [README](Sources/PJRTCWrappers/README.md) |
| **SwiftIRMacros** | Macro implementations | [README](Sources/SwiftIRMacros/README.md) |

## Quick Start

### Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/pedronahum/SwiftIR.git", from: "0.1.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["SwiftIR"]
    )
]
```

### Basic Example

```swift
import SwiftIR

// Create MLIR context
let context = MLIRContext()
context.loadAllDialects()

// Define computation using StableHLO
let module = try MLIRModule.parse("""
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
""", context: context)

// Compile and execute via PJRT
let client = try PJRTClient.createCPU()
let executable = try client.compile(module)

// Run with actual data
let a = Tensor<Float>([1.0, 2.0, 3.0, 4.0])
let b = Tensor<Float>([5.0, 6.0, 7.0, 8.0])
let result = try executable.execute(inputs: [a, b])

print(result)  // [6.0, 8.0, 10.0, 12.0]
```

### More Examples

The [Examples/](Examples/) directory contains 20+ working examples demonstrating SwiftIR's capabilities:

#### **String-Based MLIR Construction**
- `PJRT_SimpleString_Example.swift` - Basic tensor addition with string MLIR
- `PJRT_Example.swift` - Core PJRT execution workflow
- `PJRT_Add_Example.swift` - Vector addition on CPU
- `PJRT_MatMul_Example.swift` - Matrix multiplication

#### **Declarative DSL**
- `PJRT_PowerfulDSL_Example.swift` - Type-safe tensor operations with DSL
- `PJRT_SequentialDSL_Example.swift` - Building sequential computations
- `PJRT_MultiPath_Example.swift` - Multiple execution paths

#### **Neural Networks & ML**
- `PJRT_NeuralNet_Example.swift` - 2-layer neural network (4→3→2)
- `StableHLO_CNN.swift` - Complete CNN with StableHLO operations
- `StableHLO_ResNet.swift` - ResNet-style architecture
- `CNN.swift` - Convolutional neural network example
- `SimpleNN.swift` - Basic neural network

#### **GPU Code Generation (Experimental)**
- `SPIRV_Example.swift` - SPIR-V pipeline for Vulkan/OpenCL
- `SPIRV_VectorAdd_Example.swift` - GPU vector addition via SPIR-V
- `LLVM_VectorAdd_Example.swift` - LLVM/PTX pipeline for NVIDIA GPUs

#### **Advanced Features**
- `XLA_Execution.swift` - Direct XLA compiler integration
- `PJRT_ThreeExecutables_Test.swift` - Multiple executables management
- `PJRT_Approaches1and2_Test.swift` - Comparing different approaches

#### **Future Vision**
- `Macro_Example.swift` - Conceptual @MLIRFunction macro demonstration

## Current Status

### Production Ready ✅

- **Core MLIR Bindings** - Full access to MLIR C API
- **Type System** - Type-safe IR construction
- **Basic Dialects** - func, arith, tensor operations
- **StableHLO Integration** - Portable ML operations
- **PJRT Runtime** - CPU execution
- **XLA Compilation** - Industry-standard optimizer

### Working Examples ✅

SwiftIR includes 20+ fully functional examples demonstrating real-world usage:

- **Basic Operations** - Vector addition, matrix multiplication on CPU via PJRT
- **Neural Networks** - 2-layer networks, CNNs, ResNet-style architectures
- **StableHLO Integration** - Portable ML operations for XLA compilation
- **Declarative DSL** - Type-safe tensor operations with builder pattern
- **Multiple Approaches** - String MLIR, DSL, and low-level IR construction
- **GPU Pipelines** - SPIR-V (Vulkan/OpenCL) and LLVM/PTX (NVIDIA) code generation
- **XLA Execution** - Direct XLA compiler integration for optimization

See the detailed list in the "More Examples" section above.

### In Development 🚧

- **GPU Lowering Pipelines** - Transform Linalg → GPU → SPIR-V/PTX
  - Current blocker: Need MLIR build with GPU passes
  - See [GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md)
- **@MLIRFunction Macro** - Basic implementation exists, needs expansion
- **Advanced DSL Features** - More control flow constructs

## Roadmap and Next Steps

### High Priority

1. **Fix GPU Lowering** (see [GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md))
   - Rebuild MLIR with full GPU support (NVPTX, AMDGPU targets)
   - Enable GPU conversion passes (Linalg → GPU → SPIR-V/LLVM)
   - Test LLVM/PTX pipeline for NVIDIA GPUs
   - Test SPIR-V pipeline for Vulkan/OpenCL
   - **Estimated effort**: 4-8 hours

2. **Automatic Differentiation**
   - Implement differentiation protocol/trait system
   - Add reverse-mode autodiff (backpropagation)
   - Forward-mode autodiff for Jacobians
   - Integration with existing tensor operations
   - **Estimated effort**: 2-4 weeks

3. **Complete Macro System**
   - Full Swift → MLIR type mapping
   - Support for complex control flow
   - Generic function support
   - Type inference improvements
   - **Estimated effort**: 3-4 weeks

### Medium Priority

4. **Expanded Dialect Support**
   - TOSA dialect (Tensor Operator Set Architecture)
   - MHLO dialect (additional XLA operations)
   - Linalg dialect (structured linear algebra)
   - **Estimated effort**: 2-3 weeks

5. **Optimization DSL**
   - Pattern matching and rewriting
   - Graph-level optimizations
   - Fusion passes
   - Memory optimization hints
   - **Estimated effort**: 3-4 weeks

6. **Multi-Device Execution**
   - Sharding and partitioning
   - Device placement strategies
   - Cross-device communication
   - **Estimated effort**: 4-6 weeks

### Long-Term Goals

7. **Training Framework**
   - Loss functions
   - Optimizers (SGD, Adam, etc.)
   - Data loading and batching
   - Distributed training

8. **Model Zoo**
   - Pre-built neural network architectures
   - Vision models (ResNet, VGG, etc.)
   - NLP models (Transformer, BERT, etc.)
   - Import from other frameworks

9. **Debugger and Profiler**
   - IR visualization
   - Step-through debugging
   - Performance profiling
   - Memory analysis

10. **Language Server Protocol (LSP)**
    - Code completion for MLIR
    - Jump to definition
    - Hover documentation
    - Inline error checking

## Technical Highlights

### Inspired by Swift Forums Discussion

The original [Swift forums discussion](https://forums.swift.org/t/swift-as-syntactic-sugar-for-mlir/27672) proposed Swift as a frontend for MLIR. SwiftIR builds on this with:

**Modern Swift Features:**
- Swift Macros (Swift 5.9+) - Enabling compile-time code generation (in development for SwiftIR)
- Result builders for declarative syntax (working)
- Improved C++ interop for direct MLIR access (working)
- Protocol-oriented design for extensibility (working)

**Additional Flexibility:**
- Multiple abstraction levels (string MLIR, DSL, future macros)
- Full compilation pipeline (not just IR generation)
- Production runtime integration (PJRT)
- Type safety throughout the stack

### Why Swift for MLIR?

1. **Natural Fit** - Swift's syntax maps well to MLIR's structure
2. **Type Safety** - Catch IR errors at compile time
3. **Metaprogramming** - Macros and result builders enable powerful DSLs
4. **Modern Language** - Clean, readable code for complex compilers
5. **Performance** - Zero-overhead abstractions over C API
6. **Ecosystem** - Leverage Swift's tooling and libraries

## Building from Source

### Prerequisites

- Swift 6.0+ with development toolchain
- MLIR from StableHLO project (or full LLVM build)
- XLA/PJRT libraries (for runtime execution)
- Bazel (for XLA components)

### Build Instructions

```bash
# Clone repository
git clone https://github.com/pedronahum/SwiftIR.git
cd SwiftIR

# Build MLIR dependencies (see GPU_LOWERING_ROADMAP.md for details)
cd /path/to/stablehlo
./build_tools/build_mlir.sh

# Build SwiftIR
swift build

# Run tests
swift test

# Run examples
swift run PJRT_Example
```

### Important: Environment Setup for Building and Running

**Before building or running examples**, you must set up library paths:

```bash
# Option 1: Source the environment file (recommended)
source /etc/profile.d/swiftir.sh

# Option 2: Set paths manually
# LIBRARY_PATH - needed at link time (for swift build)
export LIBRARY_PATH=/opt/swiftir-deps/lib:$LIBRARY_PATH

# LD_LIBRARY_PATH - needed at runtime (for swift run) [Linux]
export LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH

# DYLD_LIBRARY_PATH - needed at runtime [macOS]
export DYLD_LIBRARY_PATH=/opt/swiftir-deps/lib:$DYLD_LIBRARY_PATH
```

**Common errors without proper setup:**

```bash
# Link error (missing LIBRARY_PATH):
error: link command failed with exit code 1
/usr/bin/ld: cannot find -lPJRTProtoHelper

# Runtime error (missing LD_LIBRARY_PATH):
error while loading shared libraries: libPJRTProtoHelper.so: cannot open shared object file
```

For permanent setup, add `source /etc/profile.d/swiftir.sh` to your `~/.bashrc` or `~/.zshrc`.

### Build Configuration

The project uses Swift Package Manager with custom build settings:
- C++ interoperability enabled
- Custom include paths for MLIR headers
- Linked libraries for PJRT runtime

See [Package.swift](Package.swift) for details.

## Contributing

Contributions are welcome! Areas where help is needed:

1. **GPU Lowering** - Test and debug GPU pipelines
2. **Dialect Wrappers** - Add more MLIR dialects
3. **Documentation** - Improve examples and guides
4. **Testing** - Expand test coverage
5. **Automatic Differentiation** - Design and implement autodiff system

Please open an issue to discuss major changes before submitting PRs.

## Documentation

- **[Sources/README.md](Sources/README.md)** - Complete architecture guide
- **[GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md)** - GPU implementation plan
- **Module READMEs** - Detailed documentation for each module
- **Examples/** - Working code samples
- **Inline Documentation** - API docs in source files

## Research and Learning

This project is ideal for:
- **Compiler Research** - Experiment with MLIR transformations
- **ML Systems** - Understand how ML compilers work
- **Language Design** - Study metaprogramming techniques
- **Systems Programming** - Learn low-level IR manipulation

## Acknowledgments

- **MLIR Team** for creating MLIR
- **Swift Community** for discussions on Swift + MLIR integration
- **OpenXLA Team** for StableHLO and PJRT
- **Google** for XLA compiler infrastructure

## Related Projects

- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
- [StableHLO](https://github.com/openxla/stablehlo) - Portable ML operations
- [OpenXLA](https://github.com/openxla/xla) - XLA compiler and PJRT runtime
- [Swift](https://swift.org/) - The Swift programming language

## License

Apache 2.0 License. See [LICENSE](LICENSE) file for details.

## Author

**Pedro Nahum** ([@pedronahum](https://github.com/pedronahum))

SwiftIR is developed and maintained by Pedro Nahum. Contributions, feedback, and discussions are welcome!

## Contact

- **GitHub**: [@pedronahum](https://github.com/pedronahum)
- **Issues**: [GitHub Issues](https://github.com/pedronahum/SwiftIR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedronahum/SwiftIR/discussions)
- **Swift Forums**: [Swift.org Forums](https://forums.swift.org/)

---

**SwiftIR: Making compiler construction accessible, type-safe, and enjoyable in Swift.**

*Created by [Pedro Nahum (@pedronahum)](https://github.com/pedronahum) | Inspired by Swift community discussions | Powered by modern Swift features | Built for the future of ML compilation*
