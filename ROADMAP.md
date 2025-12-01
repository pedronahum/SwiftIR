# SwiftIR Roadmap

**Vision**: Flexible ML compiler infrastructure in Swift that leverages the Swift compiler to build computation graphs with automatic differentiation, supporting multiple programming paradigms (functional transformations, tensor operations, native Swift AD) and executing via XLA/PJRT on CPU, GPU, and TPU.

**Current Status**: Production-ready with comprehensive automatic differentiation, functional transformations (vmap, scan, cond, PRNG, DiffTree), tensor-style operations, and native Swift `@differentiable` support. Executes on CPU and TPU.

---

## What's Complete

### Core Infrastructure
- **MLIR Bindings** - Complete Swift bindings to MLIR C API
- **StableHLO Generation** - 80+ operations with full gradient support
- **XLA Compilation** - Industry-standard optimizer
- **PJRT Execution** - CPU and TPU support

### Automatic Differentiation
- **Reverse-Mode AD** - Full VJP implementation using Swift's `@differentiable`
- **DifferentiableTracer / JTracer** - Symbolic tracing with gradient capture
- **300+ Tests** - Comprehensive gradient verification
- **~1.0x Gradient Overhead** - XLA fusion eliminates typical 2-4x overhead

### Programming Paradigms

SwiftIR's compiler-based tracing enables multiple programming styles:

| Paradigm | Description | Examples |
|----------|-------------|----------|
| **Functional Transformations** | JAX-style higher-order functions | `jVmap`, `jScan`, `jCond`, `JPRNGKey`, `JTree` |
| **Tensor Operations** | PyTorch/TensorFlow-style method chaining | `x.matmul(w).relu().softmax()` |
| **Native Swift AD** | Swift's `@differentiable` attribute | `gradient(at: x) { f($0) }` |

### Functional Transformations (Complete)

| Feature | Status | Description |
|---------|--------|-------------|
| **vmap** | ✅ Complete | Automatic vectorization/batching (`jVmap`, `diffVmap`) |
| **scan** | ✅ Complete | Sequential operations (`jScan`, `jCumsum`, `jCumprod`) |
| **cond** | ✅ Complete | Differentiable conditionals (`jCond`, `jSelect`, `jWhere`) |
| **PRNG** | ✅ Complete | Functional random (`JPRNGKey`, `jRandomNormal`, `jDropout`) |
| **DiffTree** | ✅ Complete | Tree-structured parameters (`JTree`, `jTreeMap`, `jTreeZipWith`) |
| **while** | ✅ Complete | Native `stablehlo.while` with O(1) compile time |

### Runtime & Hardware
- **CPU Execution** - Full support via PJRT CPU plugin
- **TPU Execution** - Full support via libtpu.so
- **Runtime Detection** - Automatic TPU → GPU → CPU priority
- **Unified API** - `PJRTClientFactory.create()` auto-selects hardware

### SwiftIRJupyter (Pure Swift)
Complete parallel implementation for Jupyter/Colab without C++ dependencies:
- All functional transformations (vmap, scan, cond, PRNG, DiffTree)
- Tensor-style operations
- 177 feature parity tests passing
- Ready for Colab deployment

---

## Priority Roadmap

### High Priority

#### 1. Higher-Order Differentiation
**Status**: Not started
**Complexity**: High
**Value**: Enables Hessians, JVPs, HVPs for advanced optimization (Newton's method, natural gradient)

```swift
// Target API
let hessian = diffHessian(at: x) { f($0) }
let hvp = diffHessianVectorProduct(at: x, vector: v) { loss($0) }
let jvp = diffJVP(at: x, tangent: dx) { f($0) }
```

**Implementation**:
- `Sources/SwiftIR/SymbolicAD/HigherOrderAD.swift`
- `Sources/SwiftIRJupyter/JupyterHigherOrderAD.swift`

#### 2. Gradient Checkpointing
**Status**: Not started
**Complexity**: Medium
**Value**: Memory efficiency for large models (50%+ reduction by trading compute for memory)

```swift
// Target API
let output = checkpoint(at: x) { expensiveForward($0) }
// Recomputes forward during backward instead of storing activations
```

**Implementation**:
- `Sources/SwiftIR/SymbolicAD/Checkpointing.swift`
- `Sources/SwiftIRJupyter/JupyterCheckpointing.swift`

#### 3. Shape-Typed Tensors
**Status**: Not started
**Complexity**: High
**Value**: Compile-time shape checking - a unique Swift advantage over Python frameworks

```swift
// Target API
let x: Tensor<D784> = input
let w: Tensor<D784, D256> = weights
let y: Tensor<D256> = x.matmul(w)  // Shape verified at compile time!
```

**Implementation**:
- `Sources/SwiftIR/TypedTensors/` (new module)

### Medium Priority

#### 4. GPU Execution
**Status**: Ready (needs MLIR rebuild with GPU dialects)
**Complexity**: Medium
**Value**: CUDA/ROCm acceleration

**Blocker**: MLIR needs rebuild with GPU, SPIR-V, NVVM dialects

#### 5. Distributed Training
**Status**: API exists, needs testing
**Complexity**: High
**Value**: Multi-device/multi-node training

```swift
// Existing API (needs validation)
let mesh = DeviceMesh(shape: [2, 4], deviceIds: [...])
let sharded = shard(params, mesh: mesh, spec: .replicated)
```

### Lower Priority

#### 6. Model Serialization
- Save/load compiled models
- ONNX import/export
- Checkpoint format

#### 7. Pre-built Layers
- `Linear`, `Conv2D`, `BatchNorm`, `LayerNorm`
- Sequential/Functional model API
- Built on DiffTree for parameter management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Your Swift Code (any paradigm)                                   │
│ • Functional: jVmap, jScan, jCond, JPRNGKey, JTree              │
│ • Tensor-style: x.matmul(w).relu().softmax()                    │
│ • Native Swift: @differentiable functions                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Swift Compiler + Symbolic Tracing                                │
│ • Swift's type system validates your code                        │
│ • Swift's @differentiable generates gradient rules               │
│ • DifferentiableTracer / JTracer captures computation graph      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ MLIR StableHLO Generation                                        │
│ • Portable, versioned IR                                         │
│ • Compatible with TensorFlow, JAX, PyTorch/XLA                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ XLA Compilation + PJRT Execution                                 │
│ • Industry-standard optimizations                                │
│ • CPU, GPU, TPU backends                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
Sources/
├── SwiftIR/                     # C++ interop version
│   └── SymbolicAD/
│       ├── ADIntegration.swift      # 80+ differentiable ops
│       ├── DifferentiableWhile.swift # While loop support
│       ├── Vmap.swift               # ✅ Automatic vectorization
│       ├── Scan.swift               # ✅ Sequential operations
│       ├── Cond.swift               # ✅ Conditionals
│       ├── PRNG.swift               # ✅ Functional random
│       ├── DiffTree.swift           # ✅ Tree operations
│       ├── HigherOrderAD.swift      # 🔮 Future: Hessians, JVPs
│       └── Checkpointing.swift      # 🔮 Future: Memory efficiency
│
├── SwiftIRJupyter/              # Pure Swift version (no C++)
│   ├── JupyterSymbolicAD.swift      # Core tracing
│   ├── JupyterCompiler.swift        # MLIR generation
│   ├── JupyterVmap.swift            # ✅ Vmap
│   ├── JupyterScan.swift            # ✅ Scan
│   ├── JupyterCond.swift            # ✅ Cond
│   ├── JupyterPRNG.swift            # ✅ PRNG
│   └── JupyterTree.swift            # ✅ DiffTree
│
├── SwiftIRRuntime/              # Hardware detection
│   ├── RuntimeDetector.swift        # TPU/GPU/CPU detection
│   └── PJRTClientFactory.swift      # Unified client creation
│
└── SwiftIRXLA/                  # XLA/PJRT integration
```

---

## Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Basic Operations | 40+ | ✅ Passing |
| Unary Operations | 30+ | ✅ Passing |
| Matrix Operations | 20+ | ✅ Passing |
| Loss Functions | 15+ | ✅ Passing |
| Control Flow | 25+ | ✅ Passing |
| Vmap | 20+ | ✅ Passing |
| Scan | 15+ | ✅ Passing |
| Cond | 15+ | ✅ Passing |
| PRNG | 24+ | ✅ Passing |
| DiffTree | 34+ | ✅ Passing |
| **Total** | **300+** | **✅ All Passing** |

Feature Parity Tests (SwiftIRJupyter): **177 tests passing**

---

## Resources

### Documentation
- [README.md](README.md) - Main project documentation
- [Examples/BENCHMARK_RESULTS.md](Examples/BENCHMARK_RESULTS.md) - Performance data
- [GPU_LOWERING_ROADMAP.md](GPU_LOWERING_ROADMAP.md) - GPU implementation plan

### Specifications
- [Jax/01-VMAP-SPEC.md](Jax/01-VMAP-SPEC.md) - Vmap specification
- [Jax/02-SCAN-SPEC.md](Jax/02-SCAN-SPEC.md) - Scan specification
- [Jax/03-COND-SPEC.md](Jax/03-COND-SPEC.md) - Cond specification
- [Jax/06-PRNG-SPEC.md](Jax/06-PRNG-SPEC.md) - PRNG specification
- [Jax/08-PYTREES-SPEC.md](Jax/08-PYTREES-SPEC.md) - DiffTree specification

### External References
- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [MLIR Documentation](https://mlir.llvm.org/)
- [XLA Documentation](https://www.tensorflow.org/xla)

---

**Last Updated**: December 1, 2025
**Current Focus**: Higher-Order AD, Checkpointing, Shape-Typed Tensors
**Project Status**: Production-ready with multiple programming paradigms
