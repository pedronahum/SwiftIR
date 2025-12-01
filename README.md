# SwiftIR

**Swift ML Compiler Infrastructure - Flexible Automatic Differentiation with Multiple Programming Paradigms**

[![Swift Version](https://img.shields.io/badge/swift-6.0-orange.svg)](https://swift.org)
[![MLIR](https://img.shields.io/badge/MLIR-StableHLO-blue.svg)](https://github.com/openxla/stablehlo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)]()

---

## What is SwiftIR?

SwiftIR is a flexible ML compiler infrastructure that uses Swift's compiler to build computation graphs with automatic differentiation. Because it leverages Swift's type system and `@differentiable` attribute, SwiftIR can naturally express **multiple programming paradigms**:

```swift
import SwiftIRJupyter

// ═══════════════════════════════════════════════════════════════
// PARADIGM 1: Functional Transformations
// ═══════════════════════════════════════════════════════════════

// Automatic batching with vmap
let batchedForward = jVmap { x in model(x) }

// Sequential operations with scan
let (finalState, outputs) = jScan(rnnCell, initialState, sequence)

// Functional PRNG
let key = JPRNGKey(seed: 42)
let weights = jRandomNormal(key, shape: [784, 256])

// Tree-structured parameters
let updated = jTreeZipWith(params, grads) { p, g in p - lr * g }

// ═══════════════════════════════════════════════════════════════
// PARADIGM 2: PyTorch/TensorFlow-Style Eager Operations
// ═══════════════════════════════════════════════════════════════

// Method chaining on tensors
let output = input
    .matmul(weights)
    .relu()
    .dropout(rate: 0.5)
    .softmax()

// Familiar tensor operations
let loss = (predictions - targets).pow(2).mean()

// ═══════════════════════════════════════════════════════════════
// PARADIGM 3: Native Swift with @differentiable
// ═══════════════════════════════════════════════════════════════

@differentiable(reverse)
func neuralNetwork(_ x: DifferentiableTracer) -> DifferentiableTracer {
    var h = diffReLU(diffMatmul(x, w1) + b1)
    h = diffReLU(diffMatmul(h, w2) + b2)
    return diffSoftmax(diffMatmul(h, w3) + b3)
}

// Swift's native gradient computation
let grad = gradient(at: params) { p in loss(model(p, x), y) }
```

**The key insight**: SwiftIR uses Swift's compiler to trace operations into a computation graph. This means any Swift code that uses `DifferentiableTracer` or `JTracer` automatically becomes a traceable, differentiable program - regardless of the programming style you prefer.

---

## Why SwiftIR is Different

### Compiler-Based Tracing

Unlike frameworks that define their own DSL, SwiftIR leverages Swift's existing infrastructure:

```
┌─────────────────────────────────────────────────────────────────┐
│ Your Swift Code (any paradigm)                                   │
│ • JAX-style: jVmap, jScan, jCond                                │
│ • Tensor-style: x.matmul(w).relu()                              │
│ • Native Swift: @differentiable functions                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Swift Compiler + DifferentiableTracer                            │
│ • Swift's type system validates your code                        │
│ • Swift's @differentiable generates gradient rules               │
│ • Tracers capture operations into computation graph              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ MLIR StableHLO                                                   │
│ • Portable, versioned IR                                         │
│ • Same format as JAX, TensorFlow, PyTorch/XLA                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ XLA Compilation + PJRT Execution                                 │
│ • Industry-standard optimizations                                │
│ • CPU, GPU (CUDA/ROCm), TPU                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Flexibility Through Composition

Because SwiftIR is built on Swift's type system, you can freely mix paradigms:

```swift
// Mix JAX-style vmap with tensor-style operations
let batchedModel = jVmap { x in
    x.matmul(weights).relu().softmax()  // Tensor-style inside vmap
}

// Use scan with @differentiable functions
@differentiable(reverse)
func rnnCell(_ h: JTracer, _ x: JTracer) -> (JTracer, JTracer) {
    let newH = (h.matmul(Wh) + x.matmul(Wx)).tanh()
    return (newH, newH)
}
let (final, outputs) = jScan(rnnCell, h0, inputs)

// Tree operations work with any nested structure
let model = JTuple2(first: encoder, second: decoder)
let grads = jTreeMap(model) { param in computeGrad(param) }
```

---

## Supported Paradigms

### Functional Transformations

| Transformation | API | Description |
|----------------|-----|-------------|
| **vmap** | `jVmap`, `jVmap2`, `jVmap3` | Automatic vectorization/batching |
| **scan** | `jScan`, `jCumsum`, `jCumprod` | Sequential operations (RNNs, time series) |
| **cond** | `jCond`, `jSelect`, `jWhere` | Differentiable conditionals |
| **PRNG** | `JPRNGKey`, `jRandomNormal` | Functional random number generation |
| **DiffTree** | `JTree`, `jTreeMap`, `jTreeZipWith` | Tree-structured parameters |
| **grad** | `jGrad`, `jValueAndGrad` | Reverse-mode automatic differentiation |

### Tensor-Style Operations

```swift
// Arithmetic with broadcasting
let z = x + y * 2.0

// Method chaining
let out = input.matmul(w).add(b).relu().dropout(rate: 0.1)

// Reductions
let mean = tensor.mean()
let sum = tensor.sum(axis: 1)

// Reshaping
let reshaped = tensor.reshape([batch, -1])
let transposed = tensor.transpose([0, 2, 1])
```

### Native Swift Differentiation

```swift
import _Differentiation

@differentiable(reverse)
func myModel(_ x: DifferentiableTracer) -> DifferentiableTracer {
    // Write normal Swift code - it's automatically differentiable
    let h1 = diffReLU(diffMatmul(x, w1))
    let h2 = diffReLU(diffMatmul(h1, w2))
    return diffSoftmax(diffMatmul(h2, w3))
}

// Use Swift's native gradient API
let (value, grad) = valueWithGradient(at: params) { p in
    loss(myModel(p, input), target)
}
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/pedronahum/SwiftIR.git
cd SwiftIR
swift build
swift test  # 300+ tests
```

### Google Colab / Jupyter

SwiftIR works with [swift-jupyter](https://github.com/pedronahum/swift-jupyter) for interactive notebook development:

```python
# In a Colab notebook
!curl -L https://github.com/pedronahum/SwiftIR/releases/latest/download/SwiftIR-linux-x86_64.tar.gz | tar xz -C /content
%env LD_LIBRARY_PATH=/content/SwiftIR/lib
```

```swift
// Then in a Swift cell
import SwiftIRJupyter

let x = JTracer(value: 1.0, shape: [2, 3])
let y = x.matmul(weights).relu().softmax()
```

---

## Examples by Paradigm

### JAX-Style: Batched Neural Network

```swift
// Define model
func mlp(_ x: JTracer) -> JTracer {
    var h = x.matmul(w1).relu()
    h = h.matmul(w2).relu()
    return h.matmul(w3).softmax()
}

// Batch it with vmap
let batchedMLP = jVmap(mlp)
let predictions = batchedMLP(batchedInputs)

// Get gradients with tree operations
let params = JTuple3(first: w1, second: w2, third: w3)
let grads = jTreeMap(params) { p in computeGradient(p) }
let updated = jTreeZipWith(params, grads) { p, g in p - 0.01 * g }
```

### Tensor-Style: Convolutional Network

```swift
func cnn(_ x: JTracer) -> JTracer {
    x.conv2d(filters, stride: 2)
     .batchNorm()
     .relu()
     .maxPool(size: 2)
     .flatten()
     .matmul(fc)
     .softmax()
}

let loss = crossEntropy(cnn(images), labels)
```

### Native Swift: Physics Simulation

```swift
@differentiable(reverse)
func simulate(_ params: DifferentiableTracer) -> DifferentiableTracer {
    var state = initialState

    // Use native while loop - compiles to stablehlo.while
    let (_, finalState, _, _) = diffWhileLoop(
        initial: (iteration, state, velocity, acceleration),
        condition: { $0.0 < maxSteps },
        body: { s in
            let (newState, newVel) = physicsStep(s.1, s.2, params)
            return (s.0 + 1, newState, newVel, s.3)
        }
    )

    return finalState
}

// Gradient of simulation w.r.t. parameters
let grad = gradient(at: params) { p in
    let final = simulate(p)
    return (final - target).pow(2).sum()
}
```

---

## Performance

Performance benchmarks use a **thermal building simulation** from [PassiveLogic's Differentiable Swift benchmarks](https://passivelogic.com/blog/?post=benchmarking-differentiable-swift) - a real-world physics simulation with heat transfer equations running over many timesteps. This represents a practical ML/scientific computing workload rather than synthetic benchmarks.

### O(1) Compilation for Loops

| Iterations | While Loop | Unrolled | Speedup |
|------------|------------|----------|---------|
| 1,000 | 43ms | 24s | **566x** |
| 10,000 | 42ms | 4.2min | **6,002x** |
| 100,000 | 43ms | ~42min | **59,523x** |

SwiftIR's native `stablehlo.while` maintains constant compilation time regardless of iteration count.

### Low Gradient Overhead

| Framework | Gradient Overhead |
|-----------|-------------------|
| SwiftIR (XLA) | ~1.0x |
| Standard Swift | 2.5-4.3x |

XLA's operation fusion eliminates the typical overhead of reverse-mode AD.

### Comparison with JAX/TensorFlow

Direct performance comparisons with JAX and TensorFlow are planned but not yet completed. Both frameworks use the same XLA backend, so execution performance should be similar. The key differences are:

- **Compilation**: SwiftIR benefits from Swift's type system catching errors at compile time
- **Interoperability**: SwiftIR integrates natively with Swift/iOS/macOS ecosystems
- **Python overhead**: SwiftIR avoids Python interpreter overhead for graph construction

See [Examples/BENCHMARK_RESULTS.md](Examples/BENCHMARK_RESULTS.md) for detailed benchmark methodology and results.

---

## Hardware Support

```swift
import SwiftIRRuntime

// Auto-detect best accelerator (TPU → GPU → CPU)
let client = try PJRTClientFactory.create()
RuntimeDetector.printInfo()
```

| Hardware | Status | Plugin |
|----------|--------|--------|
| **CPU** | ✅ Full support | `pjrt_c_api_cpu_plugin.so` |
| **TPU** | ✅ Full support | `libtpu.so` |
| **GPU** | Ready (needs MLIR rebuild) | `xla_cuda_plugin.so` |

---

## Two Implementations

SwiftIR provides two parallel implementations to support different deployment scenarios:

### SwiftIR (C++ Interop)
- **Full MLIR toolchain** - Direct bindings to MLIR C API
- **Swift's native `@differentiable`** - Leverages Swift's built-in AD infrastructure
- **Best for**: Local development, production deployment, full compiler integration

### SwiftIRJupyter (Pure Swift)
- **No C++ dependencies** - Pure Swift implementation generates MLIR as strings
- **PythonKit integration** - Smooth interaction with Jupyter kernel via [swift-jupyter](https://github.com/pedronahum/swift-jupyter)
- **Colab-ready** - Run SwiftIR notebooks in Google Colab without complex setup
- **Best for**: Interactive development, education, rapid prototyping

```swift
// Same API, different backends
// SwiftIR (C++ interop)
import SwiftIR
let result = diffVmap { x in diffReLU(x) }

// SwiftIRJupyter (pure Swift, Jupyter/Colab)
import SwiftIRJupyter
let result = jVmap { x in x.relu() }
```

**Why two implementations?**

1. **Deployment flexibility**: SwiftIRJupyter can run anywhere Swift runs, without needing MLIR/XLA libraries installed locally. The MLIR is generated as text and can be compiled remotely.

2. **Jupyter/Colab support**: SwiftIRJupyter integrates with [swift-jupyter](https://github.com/pedronahum/swift-jupyter) for interactive notebook development, making it easy to experiment with ML models in Google Colab.

3. **Feature parity**: Both implementations produce identical MLIR output and support all paradigms. 177 feature parity tests ensure they stay in sync.

4. **Incremental adoption**: Start with SwiftIRJupyter for prototyping, then switch to SwiftIR for production when you need full MLIR toolchain access.

---

## Complete Operation Coverage

### 80+ Differentiable Operations

**Arithmetic**: add, subtract, multiply, divide, power, sqrt, rsqrt
**Activations**: relu, leakyRelu, elu, selu, gelu, silu, mish, sigmoid, tanh, softplus, softmax
**Matrix**: matmul, transpose, reshape, flatten, concat, slice
**Reductions**: sum, mean, max, min, prod
**Convolutions**: conv2d, maxPool, avgPool
**Normalization**: batchNorm, layerNorm
**Loss**: mse, crossEntropy, binaryCrossEntropy, huber, klDivergence
**Control Flow**: while, cond, select, where

All operations have complete gradient implementations validated by 300+ tests.

---

## Project Structure

```
Sources/
├── SwiftIRJupyter/          # Pure Swift implementation
│   ├── JupyterSymbolicAD.swift  # Core tracer
│   ├── JupyterVmap.swift        # vmap transformation
│   ├── JupyterScan.swift        # scan transformation
│   ├── JupyterCond.swift        # conditionals
│   ├── JupyterPRNG.swift        # functional PRNG
│   ├── JupyterTree.swift        # tree operations
│   └── JupyterCompiler.swift    # MLIR generation
│
├── SwiftIR/                 # C++ interop version
│   └── SymbolicAD/              # Same features, different backend
│
└── SwiftIRRuntime/          # Hardware detection
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for details.

### Completed
- ✅ Automatic differentiation (300+ tests)
- ✅ Functional transformations (vmap, scan, cond, PRNG, DiffTree)
- ✅ Tensor-style operations
- ✅ Native Swift @differentiable support
- ✅ CPU and TPU execution
- ✅ SwiftIRJupyter (177 feature parity tests)

### Next Priorities
- Higher-order differentiation (Hessians, JVPs)
- Gradient checkpointing
- Shape-typed tensors (compile-time shape checking)

---

## Contributing

Contributions welcome! Priority areas:
1. **Testing**: More test coverage
2. **Operations**: New differentiable ops
3. **Documentation**: Tutorials and examples
4. **GPU**: Testing when MLIR is rebuilt

---

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Author

**Pedro Nahum** ([@pedronahum](https://github.com/pedronahum))

---

**SwiftIR: Flexible ML compiler infrastructure in Swift - Multiple paradigms, one powerful foundation.**

*Built with Swift 6.0 | Powered by MLIR, StableHLO, and XLA*
