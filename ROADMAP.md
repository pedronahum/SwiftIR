# SwiftIR Roadmap

**Vision**: Type-safe ML compiler construction in Swift, leveraging MLIR and modern Swift features for accessible, maintainable compiler development.

**Current Reality**: Working examples with string-based MLIR and experimental DSL. Research project exploring Swift as a metalanguage for MLIR.

---

## 🎯 Project Vision

### Core Value Propositions

1. **Type-Safe MLIR Access** - Swift bindings to MLIR with compile-time safety
2. **Multiple Abstractions** - String MLIR, declarative DSL, and future macro-based generation
3. **Industry-Standard Runtime** - Execute via PJRT/XLA (same as JAX, TensorFlow)
4. **Apple Platform Integration** - First-class Swift on iOS/macOS

### Target Audiences

- **ML Engineers**: Type-safe alternative to Python for ML compilation
- **iOS/macOS Developers**: Native Swift ML without Python dependency
- **Compiler Researchers**: Modern, safe approach to MLIR development
- **Students & Educators**: Learn compiler construction with working examples

---

## ✅ What Works Today (Completed)

### Core Infrastructure
- ✅ **MLIR Bindings** - Complete Swift bindings to MLIR C API
  - Context, Module, Operation, Type, Attribute management
  - Dialect registration (Func, Arith, Tensor, StableHLO, etc.)
  - IR parsing and manipulation
  - Source location tracking

- ✅ **PJRT/XLA Runtime** - Full execution pipeline
  - CPU client creation and device management
  - Buffer allocation and data marshalling
  - Compilation via XLA
  - Execution with input/output handling
  - 20+ working examples

- ✅ **StableHLO Integration** - Portable ML operations
  - Core operations: Add, Multiply, Dot, Convolution
  - Activation functions: ReLU, Tanh, Sigmoid, Exp, Log
  - Sequential composition with let-bindings
  - Integration with PJRT execution

### Working Examples (20+)
- ✅ **Basic Operations** - Vector addition, matrix multiplication
- ✅ **Neural Networks** - 2-layer networks, CNNs, ResNet architectures
- ✅ **String-based MLIR** - Parse and execute MLIR from strings
- ✅ **Declarative DSL** - Type-safe tensor operations (experimental)
- ✅ **GPU Code Generation** - SPIR-V and LLVM/PTX pipeline prototypes

### Developer Experience
- ✅ **Comprehensive Documentation** - 11 module READMEs + main README
- ✅ **Module Organization** - 10 well-structured modules
- ✅ **Examples Directory** - Categorized, documented examples
- ✅ **Build System** - Swift Package Manager integration

---

## 🚧 In Progress

### DSL Expansion
**Goal**: Expand declarative DSL coverage and ergonomics

**Current Focus**:
- 🚧 Sequential composition patterns - Partially working (let-bindings implemented)
- 🚧 More StableHLO operations - Core ops done, advanced ops needed
- 🚧 Type inference improvements - Basic types work, need shape inference
- 🚧 Error messages and diagnostics - Basic errors, need better context

**What's Needed**:
- [ ] Broadcast operations for bias addition
- [ ] Convolution with full configuration (padding, stride, dilation)
- [ ] Pooling operations (max pool, avg pool)
- [ ] Reduction operations (sum, max, min along axes)
- [ ] Reshape and transpose operations
- [ ] Control flow (if/while in StableHLO)

### GPU Execution Pipeline
**Goal**: Lower MLIR to GPU targets (SPIR-V, PTX)

**Current Status**: Code generation prototypes exist, blocked on MLIR build

**Blocker**: MLIR build needs GPU dialect support
- Need to rebuild MLIR with GPU, SPIR-V, NVVM dialects enabled
- Current build only includes StableHLO, Func, Arith, Tensor, Linalg
- See [GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md) for details

**What Works**:
- ✅ SPIR-V pipeline prototype (generates SPIR-V module structure)
- ✅ LLVM/PTX pipeline prototype (generates LLVM with nvvm attributes)
- ✅ Examples showing intended API

**What's Blocked**:
- ❌ Actual GPU dialect operations (need MLIR rebuild)
- ❌ GPU kernel launch configuration
- ❌ Memory space annotations
- ❌ SPIR-V binary serialization

### Macro System
**Goal**: Compile-time Swift → MLIR generation via macros

**Current Status**: Vision defined, basic structure exists, needs implementation

**What Exists**:
- ✅ Macro module structure (SwiftIRMacros)
- ✅ Macro definition and export (@MLIRFunction)
- ✅ Conceptual example (Macro_Example.swift)
- ✅ Documentation of intended behavior

**What's Needed**:
- [ ] AST traversal and analysis
- [ ] Swift type → MLIR type mapping
- [ ] Swift expression → MLIR operation mapping
- [ ] SSA form generation
- [ ] Control flow handling (if/for/while → scf dialect)
- [ ] Error diagnostics during macro expansion

---

## 🔮 Future Vision

### Long-term Goals (No Timeline)

These represent aspirational directions for the project, not committed features:

#### 1. SPIR-V Shader Development
- Direct SPIR-V dialect access from Swift
- Shader macro system (@SPIRVShader, @SPIRVCompute)
- Vulkan/Metal/OpenCL integration
- Graphics and compute shader examples

**Status**: Exploratory. Prototypes exist but full implementation requires significant effort.

#### 2. Training & Autodiff
- Automatic differentiation for StableHLO operations
- Backpropagation support
- Optimizer implementations (SGD, Adam)
- Training loop DSL

**Status**: Conceptual. Would require significant autodiff infrastructure.

#### 3. High-Level ML API
- Layer abstractions (Dense, Conv2D, BatchNorm)
- Sequential/Functional model APIs
- Pre-trained model loading (ONNX, SavedModel)
- Transfer learning support

**Status**: Aspirational. Current focus is on compiler infrastructure, not high-level APIs.

#### 4. Production Features
- Model quantization (INT8, FP16)
- Dynamic shapes support
- Streaming and batching APIs
- Profiling and monitoring tools
- Performance optimization

**Status**: Future work. Many examples work today, but production hardening is ongoing.

---

## 📊 Technical Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SwiftIR (Swift)                      │
│                                                         │
│  ┌────────────────┐  ┌────────────────┐               │
│  │  SwiftIRXLA    │  │ SwiftIRStableHLO│              │
│  │  (PJRT/Runtime)│  │ (StableHLO ops) │              │
│  └────────────────┘  └────────────────┘               │
│                                                         │
│  ┌────────────────────────────────────────────┐        │
│  │  SwiftIRCore (MLIR Bindings)               │        │
│  │  - Types, Dialects, Builders, IR           │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                         │
                         │ links against
                         ▼
┌─────────────────────────────────────────────────────────┐
│        MLIR/StableHLO Libraries                         │
│                                                         │
│  • StableHLO v1.13.0                                   │
│  • MLIR Dialects (Func, Arith, SCF, Tensor, Linalg)   │
│  • LLVM Core                                           │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│        XLA/PJRT Runtime (CPU execution)                 │
└─────────────────────────────────────────────────────────┘
```

**Key Points**:
- 10 Swift modules providing different abstraction levels
- Direct C API bindings to MLIR (SwiftIRCore)
- StableHLO operations for portable ML (SwiftIRStableHLO)
- PJRT/XLA runtime integration for execution (SwiftIRXLA)
- String-based MLIR and experimental DSL both supported

---

## 🎯 Real-World Use Cases Today

### Use Case 1: On-Device ML for iOS/macOS

```swift
import SwiftIR

// Define your model in MLIR (string or DSL)
let model = try MLIRModule.parse("""
func.func @inference(%input: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> {
  // Your CNN architecture here
  %0 = stablehlo.convolution(%input, %weights) : tensor<1x224x224x3xf32>
  return %0 : tensor<1x1000xf32>
}
""", context: context)

// Compile and execute on device
let client = try PJRTClient.createCPU()
let executable = try client.compile(model)
let predictions = try executable.execute(inputs: [imageData])
```

**Benefits**: No Python runtime, fully native Swift, runs on all Apple platforms

### Use Case 2: ML Compiler Research

```swift
// Experiment with MLIR transformations
let module = try MLIRModule.parse(mlirCode, context: context)

// Apply custom passes
let pm = PassManager(context: context)
pm.addPass("your-custom-pass")
try pm.run(on: module)

// Inspect transformed IR
print(module.description)
```

**Benefits**: Type-safe IR manipulation, rapid prototyping, clear examples

### Use Case 3: Educational Tool

```swift
// Students can see MLIR generation in action
let dsl = StableHLOModule(name: "simple") {
    StableHLOFunction(name: "add", ...) {
        Add("a", "b", type: tensorType)
    }
}

print(dsl.buildMLIR())  // See the generated MLIR
```

**Benefits**: Immediate feedback, working examples, gradual complexity

---

## 🚀 Getting Started

### For Users

```bash
# Clone the repository
git clone https://github.com/pedronahum/SwiftIR.git
cd SwiftIR

# Build the project
swift build

# Run an example
swift run PJRT_Add_Example
```

### For Contributors

Current focus areas where contributions would be valuable:

1. **DSL Expansion** - Add more StableHLO operations to the declarative DSL
2. **Documentation** - Improve examples, add tutorials
3. **Testing** - Add more test coverage, especially for edge cases
4. **Bug Fixes** - Help resolve issues in the GitHub tracker

---

## 📚 Resources

### Documentation
- [README.md](README.md) - Main project documentation
- [Sources/README.md](Sources/README.md) - Architecture overview
- [GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md) - GPU execution roadmap
- [Examples/](Examples/) - 20+ working examples

### Module Documentation
Each module has detailed documentation:
- [SwiftIRCore](Sources/SwiftIRCore/README.md) - MLIR bindings
- [SwiftIRStableHLO](Sources/SwiftIRStableHLO/README.md) - StableHLO operations
- [SwiftIRXLA](Sources/SwiftIRXLA/README.md) - PJRT/XLA runtime
- [SwiftIRMacros](Sources/SwiftIRMacros/README.md) - Macro system

### External References
- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [MLIR Documentation](https://mlir.llvm.org/)
- [XLA Documentation](https://www.tensorflow.org/xla)
- [Swift Forums Discussion](https://forums.swift.org/t/swift-as-syntactic-sugar-for-mlir/27672)

---

## 🤝 Community & Support

- **GitHub**: [github.com/pedronahum/SwiftIR](https://github.com/pedronahum/SwiftIR)
- **Issues**: Report bugs and request features
- **Discussions**: Design discussions and questions
- **Swift Forums**: Engage with the broader Swift community

---

**Last Updated**: November 11, 2025
**Current Focus**: DSL expansion, GPU execution (blocked), macro system (exploratory)
**Project Status**: Research project with working examples
