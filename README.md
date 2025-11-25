# SwiftIR

**Modern ML Compiler Infrastructure in Swift - Type-Safe MLIR with Production-Ready Automatic Differentiation**

[![Swift Version](https://img.shields.io/badge/swift-6.0-orange.svg)](https://swift.org)
[![MLIR](https://img.shields.io/badge/MLIR-StableHLO-blue.svg)](https://github.com/openxla/stablehlo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-pedronahum-blue.svg?logo=github)](https://github.com/pedronahum)

---

## 🚀 Performance Highlights

SwiftIR provides **constant O(1) compilation time** regardless of iteration count, and **~1.0x gradient overhead** (vs 2.5-4.3x for Standard Swift). The tradeoff: Standard Swift is faster for small workloads, but SwiftIR wins at scale.

### Execution Time Comparison (Forward Pass)

| Iterations | Standard Swift | SwiftIR While Loop | Winner |
|------------|---------------|-------------------|--------|
| 500 | 18μs | 268μs | Standard Swift **14.8x faster** |
| 1,000 | 36μs | 390μs | Standard Swift **10.7x faster** |
| 10,000 | 365μs | 577μs | Standard Swift **1.6x faster** |
| **100,000** | **3,612μs** | **2,622μs** | **SwiftIR 1.38x faster** |

### Gradient Computation Comparison

| Iterations | Standard Swift | SwiftIR While Loop | Winner |
|------------|---------------|-------------------|--------|
| 1,000 | 92μs | 389μs | Standard Swift **4.2x faster** |
| 10,000 | 1,244μs | 569μs | **SwiftIR 2.2x faster** |
| **100,000** | **10,616μs** | **2,622μs** | **SwiftIR 4.0x faster** |

### Compilation Time (SwiftIR While Loop vs Unrolled)

| Iterations | While Loop | Unrolled | Speedup |
|------------|------------|----------|---------|
| 1,000 | 43ms | 24s | **566x** |
| 10,000 | 42ms | 4.2min | **6,002x** |
| 100,000 | 43ms | ~42min | **59,523x** |

**Key insight:** SwiftIR compilation stays constant (~43ms) while unrolled scales linearly. At 100k iterations, SwiftIR becomes faster than Standard Swift for both forward and gradient computation.

📊 **[See Full Benchmark Results →](Examples/BENCHMARK_RESULTS.md)**

---

## What Makes SwiftIR Unique

SwiftIR achieves something no other framework does: **native Swift automatic differentiation that compiles to portable MLIR and executes via industry-standard XLA/PJRT**.

```swift
import SwiftIR
import _Differentiation

// Write differentiable Swift code - same syntax as Standard Swift!
@differentiable(reverse)
func neuralNetwork(_ x: DifferentiableTracer, _ weights: DifferentiableTracer) -> DifferentiableTracer {
    let hidden = diffReLU(diffMatmul(x, weights))
    return diffSoftmax(hidden)
}

// Compile to MLIR + XLA (once, ~42ms)
let gradFunc = try compileGradientForPJRT(
    input: TensorSpec(shape: [32, 784], dtype: .float32),
    weights: TensorSpec(shape: [784, 10], dtype: .float32)
) { x, w in
    neuralNetwork(x, w)
}

// Execute with gradients - runs on same hardware as TensorFlow/JAX
let (loss, gradients) = try gradFunc.forwardWithGradient(inputData, weightsData)
```

This isn't a wrapper or bindings library - it's a **complete compiler stack** from Swift source to optimized machine code.

## How Gradient Computation Works

Both Standard Swift and SwiftIR use Swift's `@differentiable` attribute and the `_Differentiation` module. The key difference is **how gradients are executed**.

### Standard Swift: Eager Execution

```
Swift Source (@differentiable)
         ↓
Swift Compiler generates pullback functions
         ↓
Native Swift execution (function calls)
         ↓
Gradients via pullback call stack
```

Each operation's pullback is a separate function call → **2.5-4.3x gradient overhead**.

### SwiftIR: Graph Compilation

```
Swift Source (@differentiable)
         ↓
DifferentiableTracer ("Trojan Horse")
         ↓
Swift's AD generates pullbacks (same!)
         ↓
Tracers emit MLIR operations
         ↓
Complete forward+backward graph
         ↓
XLA compilation + fusion
         ↓
Single optimized kernel via PJRT
```

XLA sees the entire computation → **~1.0x gradient overhead** (forward ≈ backward time).

### The "Trojan Horse" Mechanism

```swift
// DifferentiableTracer looks like a number to Swift's type system
public struct DifferentiableTracer: Differentiable, AdditiveArithmetic {
    public let irValue: String  // MLIR SSA value (e.g., "%v42")
    public let shape: [Int]
    public let dtype: DType

    // Swift's AD calls this during backprop - we emit MLIR instead of computing!
    @derivative(of: *)
    static func vjpMultiply(lhs: Self, rhs: Self) -> (value: Self, pullback: (Self) -> (Self, Self)) {
        // Forward: emit "stablehlo.multiply"
        // Pullback: emit gradient operations
    }
}
```

This allows SwiftIR to:
- Use **unmodified Swift `@differentiable` syntax**
- Leverage **Swift's battle-tested AD compiler**
- Capture **complete gradient graphs** for XLA optimization

📊 **[See detailed explanation with benchmarks →](Examples/BENCHMARK_RESULTS.md#how-gradient-computation-works)**

## Current Status: Production Ready

| Component | Status | Description |
|-----------|--------|-------------|
| **MLIR Integration** | ✅ Complete | Native Swift bindings to MLIR C API |
| **StableHLO Generation** | ✅ Complete | Portable ML IR (80+ operations) |
| **Automatic Differentiation** | ✅ Complete | Full reverse-mode AD with 300+ tests |
| **While Loop Support** | ✅ Complete | Native `stablehlo.while` with constant compile time |
| **XLA Compilation** | ✅ Complete | Industry-standard optimizer |
| **PJRT Execution** | ✅ Complete | CPU execution (GPU ready, needs MLIR rebuild) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Swift Code with @differentiable                             │
│ • Write normal Swift functions                              │
│ • Use @differentiable(reverse) for autodiff                 │
│ • Native while loops with diffWhileLoop()                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Symbolic Tracing (DifferentiableTracer)                     │
│ • Intercepts operations during forward pass                 │
│ • Builds computation graph symbolically                     │
│ • Swift's @derivative handles gradient rules                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ MLIR StableHLO Generation                                   │
│ • Generates portable, versioned IR                          │
│ • Native stablehlo.while for loops (O(1) compile time)      │
│ • Compatible with TensorFlow, JAX, PyTorch/XLA              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ XLA Compilation                                             │
│ • Industry-standard optimizations                           │
│ • Operation fusion (forward + backward together)            │
│ • Target-specific code generation                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PJRT Execution                                              │
│ • Same runtime as JAX and TensorFlow                        │
│ • CPU, GPU (CUDA/ROCm), TPU support                         │
│ • Forward pass + gradient in single fused kernel            │
└─────────────────────────────────────────────────────────────┘
```

## While Loop Support

SwiftIR now supports **native while loops** that compile to `stablehlo.while`, enabling:
- **Constant O(1) compilation time** regardless of iteration count
- **Efficient execution** for iterative algorithms
- **Physics simulations**, RNNs, optimization loops, and more

```swift
// Thermal simulation with 100,000 timesteps - compiles in 43ms!
let (_, finalTemp, _, _) = diffWhileLoop(
    initial: (iteration, slabTemp, fluidTemp, tankTemp),
    condition: { state in
        let maxIter = createConstant(100000.0, shape: [], dtype: .float32)
        return state.0 < maxIter
    },
    body: { state in
        let (newSlab, newFluid, newTank) = simulateTimestep(state.1, state.2, state.3)
        let one = createConstant(1.0, shape: [], dtype: .float32)
        return (state.0 + one, newSlab, newFluid, newTank)
    }
)
```

Without while loop support, 100,000 iterations would require ~42 minutes to compile (loop unrolling). With `stablehlo.while`, compilation stays at ~43ms.

## Complete Operation Coverage

SwiftIR supports all operations needed for modern neural networks:

### Neural Network Layers
- **Dense layers**: Matrix multiplication, bias addition
- **Activations**: ReLU, Leaky ReLU, ELU, SELU, Sigmoid, Tanh, Softplus, Swish, GELU, Mish
- **Normalizations**: Batch normalization, Layer normalization
- **Regularization**: Dropout

### Convolutional Networks
- **Conv2D**: Strided, dilated, with flexible padding
- **Pooling**: Max pooling, Average pooling
- **Shape operations**: Reshape, Flatten, Permute

### Loss Functions
- **Regression**: MSE, Huber loss
- **Classification**: Cross-entropy, Softmax
- **Distribution**: KL divergence

### Mathematical Operations
- **Arithmetic**: Add, subtract, multiply, divide (with broadcasting)
- **Transcendental**: Exp, log, sqrt, pow, rsqrt
- **Trigonometric**: Sin, cos, tan, asin, acos, atan
- **Reductions**: Sum, mean, max, min
- **Control Flow**: While loops (stablehlo.while)

All operations have **complete gradient implementations** validated by 300+ tests.

## Real-World Example: Building Thermal Simulation

This example demonstrates SwiftIR's while loop support for physics simulation:

```swift
import SwiftIR
import _Differentiation

// Physics simulation step (differentiable)
@differentiable(reverse)
func simulateTimestep(
    _ slabTemp: DifferentiableTracer,
    _ fluidTemp: DifferentiableTracer,
    _ tankTemp: DifferentiableTracer
) -> (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) {
    // Heat transfer physics...
    let heatFlow = conductance * (fluidTemp - slabTemp)
    let newSlabTemp = slabTemp + heatFlow * dt / slabMass
    // ... more physics
    return (newSlabTemp, newFluidTemp, newTankTemp)
}

// Run 100,000 timesteps with automatic differentiation
let gradFunc = try compileGradientForPJRT(
    input: TensorSpec(shape: [], dtype: .float32)
) { _ in
    let (_, finalSlab, _, _) = diffWhileLoop(
        initial: (iter, slabInit, fluidInit, tankInit),
        condition: { $0.0 < maxIterations },
        body: { state in
            let (s, f, t) = simulateTimestep(state.1, state.2, state.3)
            return (state.0 + 1, s, f, t)
        }
    )
    let loss = (finalSlab - targetTemp) * (finalSlab - targetTemp)
    return loss
}

// Get gradients for optimization
let (loss, gradient) = try gradFunc.forwardWithGradient([0.0], seed: [1.0])
```

**Results at 100,000 iterations:**
- Compilation: 43.2ms
- Forward pass: 2,622μs
- Gradient: 2,622μs
- Physics results identical to Standard Swift ✅

## Installation & Setup

### Step 1: Install Dependencies

SwiftIR requires Swift 6.0+, LLVM/MLIR, StableHLO, and XLA/PJRT libraries. Installation scripts are provided in the `scripts/` folder for Ubuntu.

```bash
# Clone repository first
git clone https://github.com/pedronahum/SwiftIR.git
cd SwiftIR

# Check if your system meets prerequisites
./scripts/check-prerequisites.sh

# Install all dependencies (Ubuntu 24.04)
# This installs Swift, LLVM/MLIR, StableHLO, and XLA/PJRT to /opt/swiftir-deps
sudo ./scripts/install-swiftir-ubuntu.sh

# Verify installation
./scripts/verify-installation.sh
```

**Note:** The installation script can take 1-2 hours as it builds LLVM, StableHLO, and XLA from source. You can skip components you already have:

```bash
# Skip Swift if already installed
sudo ./scripts/install-swiftir-ubuntu.sh --skip-swift

# Skip LLVM if already have compatible version
sudo ./scripts/install-swiftir-ubuntu.sh --skip-llvm
```

### Step 2: Set Up Environment

After installing dependencies, set up the environment variables:

```bash
# Add to your ~/.bashrc or ~/.zshrc for permanent setup
source /etc/profile.d/swiftir.sh

# Or manually set paths:
export LIBRARY_PATH=/opt/swiftir-deps/lib:$LIBRARY_PATH        # Link time
export LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH  # Runtime (Linux)
```

**Important:** Environment setup is required both at build time and runtime.

### Step 3: Build and Test

```bash
# Build SwiftIR
swift build

# Run tests (300+ tests)
swift test

# Run examples
swift run BuildingSimulation_SwiftIR
swift run PJRT_NeuralNet_Example
```

### Troubleshooting

```bash
# Missing LIBRARY_PATH (link error)
error: link command failed with exit code 1
/usr/bin/ld: cannot find -lPJRTProtoHelper
→ Fix: export LIBRARY_PATH=/opt/swiftir-deps/lib:$LIBRARY_PATH

# Missing LD_LIBRARY_PATH (runtime error)
error while loading shared libraries: libPJRTProtoHelper.so
→ Fix: export LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH
```

## Comparison with Other Frameworks

| Feature | SwiftIR | JAX | PyTorch | Standard Swift |
|---------|---------|-----|---------|----------------|
| **Language** | Swift | Python | Python | Swift |
| **Type Safety** | Compile-time | Runtime | Runtime | Compile-time |
| **AD Mechanism** | Swift @differentiable | Python/XLA | Python/C++ | Swift @differentiable |
| **Execution** | XLA/PJRT | XLA/PJRT | ATen | Native Swift |
| **Gradient Overhead** | ~1.0x | ~1.0x | Variable | 2.5-4.3x |
| **While Loops** | O(1) compile | O(1) compile | Eager | O(n) compile* |
| **GPU Support** | Ready* | Yes | Yes | No |

*Standard Swift with loop unrolling; SwiftIR GPU needs MLIR rebuild.

## Project Structure

```
Sources/
├── SwiftIR/                 # Main API
│   └── SymbolicAD/          # Automatic differentiation ⭐
│       ├── ADIntegration.swift      # 80+ differentiable ops
│       ├── DifferentiableWhile.swift # While loop support
│       ├── BackendCompilation.swift  # MLIR generation
│       └── PJRTExecution.swift       # Runtime execution
│
├── SwiftIRXLA/              # XLA/PJRT integration
└── PJRTCWrappers/           # PJRT C API wrappers

Examples/
├── BuildingSimulation_SwiftIR.swift   # Physics simulation benchmark
├── BuildingSimulation_StandardSwift.swift  # Comparison baseline
├── BENCHMARK_RESULTS.md               # Full performance data
└── PJRT_NeuralNet_Example.swift       # Neural network example

Tests/SwiftIRTests/
└── Phase9-Phase38*.swift    # 300+ autodiff tests
```

## Documentation

- **[Benchmark Results](Examples/BENCHMARK_RESULTS.md)** - Performance comparison with Standard Swift
- **[SymbolicAD README](Sources/SwiftIR/SymbolicAD/README.md)** - AD system documentation
- **[GPU Roadmap](GPU_LOWERING_ROADMAP.md)** - GPU implementation plan

## Who Should Use SwiftIR?

### ✅ Use SwiftIR if you:
- Want **type-safe ML compilation** with Swift
- Need **fast compilation** for iterative algorithms (while loops)
- Are building **iOS/macOS ML applications** natively
- Want **production runtime** (same as JAX/TensorFlow)
- Need **low gradient overhead** (~1.0x vs 2.5-4.3x)

### ❌ Don't use SwiftIR if you:
- Need Python ecosystem (use JAX/PyTorch)
- Require pre-trained model zoo (use HuggingFace)
- Need dynamic computation graphs

## Contributing

Contributions welcome! Priority areas:
1. **GPU Support** - Help test when MLIR is rebuilt
2. **Operations** - New differentiable operations
3. **Testing** - More test coverage
4. **Documentation** - Tutorials and examples

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Author

**Pedro Nahum** ([@pedronahum](https://github.com/pedronahum))

---

**SwiftIR: Modern ML compilation in Swift - Type-safe, portable, and production-ready.**

*Built with Swift 6.0 | Powered by MLIR, StableHLO, and XLA*
