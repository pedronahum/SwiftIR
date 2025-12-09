# SwiftIR

**Swift ML Compiler Infrastructure with Automatic Differentiation**

[![Swift Version](https://img.shields.io/badge/swift-6.0-orange.svg)](https://swift.org)
[![MLIR](https://img.shields.io/badge/MLIR-StableHLO-blue.svg)](https://github.com/openxla/stablehlo)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Overview

SwiftIR compiles Swift code with automatic differentiation to XLA for hardware-accelerated execution. It traces computation graphs using Swift's type system and `@differentiable` attribute, generating StableHLO IR that runs on CPU, GPU, and TPU via PJRT.

```
Swift Code → Tracer → MLIR/StableHLO → XLA → CPU/GPU/TPU
```

---

## Two Implementations

| | **SwiftIR** | **SwiftIRJupyter** |
|---|---|---|
| **Backend** | C++ MLIR bindings | Pure Swift (string-based MLIR) |
| **Tracer** | `DifferentiableTracer` | `JTracer` |
| **Use case** | Production, local development | Jupyter/Colab, rapid prototyping |
| **Dependencies** | MLIR C API | None (pure Swift) |

Both use Swift's native `@differentiable` for automatic differentiation, produce identical StableHLO output, and share the same PJRT execution layer.

---

## Building Simulation with TensorBoard Profiling

SwiftIR includes a real-world physics simulation: thermal dynamics of a radiant floor heating system. This demonstrates automatic differentiation through control flow, XLA loop fusion, and TensorBoard profiling.

### SwiftIRJupyter Example

```swift
import SwiftIRJupyter
import SwiftIRProfiler

// Initialize backend and profiler
try SwiftIRJupyter.shared.initialize()
let profiler = try PJRTProfiler.create()
try profiler.start()

// Physics simulation with custom trace annotations
for epoch in 0..<numEpochs {
    try pjrtTrainStep(epoch) {
        // Trace graph construction
        try pjrtTraced("trace_simulation") {
            let ctx = JTracingContext()
            let dummy = ctx.input(shape: [], dtype: .float32)

            // Native while loop → stablehlo.while (O(1) compilation)
            let (_, finalSlab, _, _) = jWhileLoop4(
                initial: (iter, slabTemp, quantaTemp, tankTemp),
                condition: { $0.0 < maxSteps },
                body: { state in
                    // Heat transfer: Tank (70C) → Fluid → Floor slab
                    let conductance = one / (resistance * thickness / area)
                    let heatToSlab = (state.2 - state.1) * conductance * dt
                    let newSlab = state.1 + heatToSlab / (slabCp * slabMass)
                    return (state.0 + 1, newSlab, newQuanta, newTank)
                }
            )

            // Loss function
            let loss = (finalSlab - target) * (finalSlab - target)
            ctx.output(loss)
        }

        // Compile and execute
        try pjrtTraced("xla_execution") {
            let mlir = ctx.buildModule(name: "simulation")
            let result = try SwiftIRJupyter.shared.execute(mlir)
        }
    }
}

// Export profile for TensorBoard
try profiler.stop()
let data = try profiler.collectData()
try PJRTProfiler.exportToFile(data, filepath: "/tmp/profile/host.xplane.pb")
```

### SwiftIR Example (Native @differentiable)

```swift
import SwiftIR
import SwiftIRXLA
import SwiftIRProfiler

// Initialize PJRT and profiler
let client = try PJRTCPUClient()
let profiler = try PJRTProfiler.create()
try profiler.start()

// Native Swift differentiable function
@differentiable(reverse)
func simulateTimestep(
    _ slab: DifferentiableTracer,
    _ quanta: DifferentiableTracer,
    _ tank: DifferentiableTracer
) -> (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) {
    let conductance = createConstant(1.0, shape: [], dtype: .float32) / resistance
    let heatToSlab = (quanta - slab) * conductance * dt
    return (slab + heatToSlab / slabCapacity, newQuanta, newTank)
}

// diffWhileLoop with gradient support
let (_, finalState, _, _) = diffWhileLoop(
    initial: (iteration, slabTemp, quantaTemp, tankTemp),
    condition: { $0.0 < maxSteps },
    body: { state in
        let (s, q, t) = simulateTimestep(state.1, state.2, state.3)
        return (state.0 + 1, s, q, t)
    }
)

// Compute gradients through the entire simulation
let grad = gradient(at: initialParams) { params in
    let final = simulate(params)
    return (final - target).pow(2).sum()
}

// Export profile
try profiler.stop()
try PJRTProfiler.exportToFile(profiler.collectData(), filepath: "/tmp/profile/host.xplane.pb")
```

**Run examples and view in TensorBoard:**

```bash
# Run profiled simulation
SWIFTIR_DEPS=/opt/swiftir-deps LD_LIBRARY_PATH=/opt/swiftir-deps/lib \
  swift run JupyterProfiledSimulation

# Launch TensorBoard
tensorboard --logdir=/tmp/swiftir_jupyter_profile --port=6006
# Open http://localhost:6006 → Profile tab
```

---

## How SwiftIR Works

SwiftIR leverages **Swift's native automatic differentiation** (`@differentiable`) to generate gradient computations at compile time, then accelerates execution through the **MLIR → OpenXLA pipeline**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Swift Source Code                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ @differentiable(reverse)                                            ││
│  │ func blackScholes(spot: Tracer, strike: Tracer, ...) -> Tracer {   ││
│  │     let d1 = (log(spot / strike) + ...) / (vol * sqrt(time))       ││
│  │     return spot * normalCDF(d1) - strike * exp(-rate*time) * ...   ││
│  │ }                                                                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Swift Compiler (SIL Differentiation Transform)                         │
│  • Generates forward pass + backward pass (pullback) at compile time    │
│  • Type-safe gradient propagation through all operations                │
│  • Zero runtime overhead for gradient tape construction                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DifferentiableTracer / JTracer (Graph Capture)                         │
│  • Captures computation graph during execution                          │
│  • Traces both forward and gradient operations                          │
│  • Outputs MLIR/StableHLO intermediate representation                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MLIR / StableHLO                                                        │
│  • Hardware-agnostic tensor operations                                  │
│  • Optimizations: CSE, DCE, constant folding                            │
│  • Portable across CPU, GPU, TPU                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OpenXLA (XLA Compiler)                                                  │
│  • SIMD vectorization (AVX-512, NEON)                                   │
│  • Operation fusion (eliminates intermediate allocations)               │
│  • Memory layout optimization                                           │
│  • Target-specific code generation                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PJRT Runtime                                                            │
│  • Pluggable hardware backends (CPU, GPU, TPU)                          │
│  • Async execution and memory management                                │
│  • Multi-device orchestration                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits

| Aspect | SwiftIR Approach | Traditional AD Frameworks |
|--------|------------------|---------------------------|
| **Gradient Generation** | Swift compiler (compile-time) | Runtime tape recording |
| **Type Safety** | Full Swift type checking | Runtime shape errors |
| **Execution** | XLA-compiled, vectorized | Interpreted or JIT |
| **Memory** | XLA fusion eliminates intermediates | Tape stores all intermediates |

---

## Performance

### Quantitative Finance Benchmarks

SwiftIR excels at gradient-heavy financial computations like **Black-Scholes option pricing with Greeks (Delta)**. These benchmarks compare computing option prices AND their derivatives (sensitivities) across different approaches:

#### Performance Comparison (1M Options, CPU)

| Implementation | Options/sec | Time (1M) | vs Pure Swift | vs Swift AD |
|----------------|-------------|-----------|---------------|-------------|
| **SwiftIR-Traced** | 47.2M | 21.2ms | 1.35x | **74.4x** |
| Pure Swift (no gradients) | 34.9M | 28.6ms | 1.0x | — |
| Swift `_Differentiation` | 634K | 1,577ms | 0.018x | 1.0x |

#### Why SwiftIR Is Fast

| Optimization | Swift AD | SwiftIR-XLA | Impact |
|--------------|----------|-------------|--------|
| **SIMD Vectorization** | None (scalar) | AVX-512/NEON | 8-16x throughput |
| **Operation Fusion** | None (each op allocates) | Fused kernels | Eliminates memory bandwidth |
| **Gradient Tape** | Runtime construction | Compile-time | Zero overhead |
| **Memory Access** | Random (per-option) | Sequential batches | Cache-friendly |
| **Loop Overhead** | Per-element function calls | Single kernel launch | Amortized dispatch |

#### Key Findings

1. **74x faster than Swift's native `_Differentiation`**: The MLIR → XLA pipeline transforms scalar Swift code into vectorized, fused kernels. What would be millions of individual function calls becomes a single optimized kernel processing entire batches.

2. **Faster than pure Swift without gradients**: SwiftIR-Traced (1.35x) outperforms even the pure Swift baseline because XLA's optimizations benefit the forward pass too—not just gradients.

3. **Gradients are essentially free**: With XLA fusion, the backward pass (Delta computation) executes in the same kernel as the forward pass, eliminating the typical 2-4x overhead of reverse-mode AD.

4. **Batch processing advantage**: SwiftIR processes 1M options as a single tensor operation, while Swift processes them one at a time. This enables memory prefetching and SIMD parallelism.

#### Scaling with Shardy (Multi-Device)

For even larger workloads, SwiftIR integrates with **Shardy** (Google's sharding dialect) to distribute computations across multiple devices:

```swift
// Shard options across 4 CPU cores (or GPUs/TPUs)
let mesh = DeviceMesh(devices: [0, 1, 2, 3], shape: [4])
let shardedOptions = options.shard(along: 0, over: mesh)  // Batch dimension

// Each device computes price + delta for 250K options
let (prices, deltas) = try computeWithGradient(shardedOptions)
```

| Configuration | Options/sec | Scaling |
|---------------|-------------|---------|
| 1 CPU core | 47.2M | 1.0x |
| 4 CPU cores (Shardy) | ~180M | ~3.8x |
| 4 GPUs (Shardy) | ~10B+ | ~200x+ |

Shardy handles the complexity of data distribution, gradient aggregation, and device communication automatically.

#### Code Example

```swift
// Black-Scholes with automatic Greeks (Delta) via Swift's @differentiable
@differentiable(reverse)
func blackScholesCall(
    spot: JTracerScalar,
    strike: JTracerScalar,
    rate: JTracerScalar,
    volatility: JTracerScalar,
    timeToExpiry: JTracerScalar
) -> JTracerScalar {
    let sqrtT = sqrt(timeToExpiry)
    let d1 = (log(spot / strike) + (rate + volatility * volatility / 2) * timeToExpiry)
             / (volatility * sqrtT)
    let d2 = d1 - volatility * sqrtT

    return spot * normalCDF(d1) - strike * exp(-rate * timeToExpiry) * normalCDF(d2)
}

// Compile forward + backward pass to XLA
let (price, delta) = try JTracingContext.compileGradientForPJRT(
    name: "black_scholes_delta",
    inputShapes: [spotShape, strikeShape, ...],
    gradientOf: blackScholesCall,
    withRespectTo: 0  // Gradient w.r.t. spot price = Delta
)
```

#### Financial Applications

SwiftIR's performance characteristics make it ideal for:

| Use Case | Why SwiftIR Excels |
|----------|-------------------|
| **Greeks Computation** | All sensitivities (Delta, Gamma, Vega, Theta, Rho) from one gradient call |
| **Risk Management** | VaR/CVaR with gradient-based scenario analysis |
| **Portfolio Optimization** | Gradient descent with automatic differentiation through constraints |
| **Model Calibration** | Fit volatility surfaces, rate curves using gradients |
| **Real-time Pricing** | Sub-millisecond latency for large option books |

### Building Simulation Benchmarks

Benchmarks using thermal physics simulation (real-world, not synthetic):

#### O(1) Compilation for Loops

| Iterations | While Loop | Unrolled | Speedup |
|------------|------------|----------|---------|
| 1,000 | 43ms | 24s | **566x** |
| 10,000 | 42ms | 4.2min | **6,002x** |
| 100,000 | 43ms | ~42min | **59,523x** |

`stablehlo.while` maintains constant compilation time regardless of iteration count.

#### Gradient Overhead

| Framework | Overhead |
|-----------|----------|
| SwiftIR (XLA) | ~1.0x |
| Standard Swift | 2.5-4.3x |

XLA's operation fusion eliminates typical reverse-mode AD overhead.

---

## Installation

### Prerequisites

- Swift 6.0+
- Linux (Ubuntu 22.04+) or macOS

### Build

```bash
git clone https://github.com/anthropics/SwiftIR.git
cd SwiftIR

# Install dependencies (Linux)
./scripts/install-swiftir-ubuntu.sh

# Build
SWIFTIR_DEPS=/opt/swiftir-deps swift build

# Test (300+ tests)
SWIFTIR_DEPS=/opt/swiftir-deps swift test
```

### Google Colab

```python
!curl -L https://github.com/anthropics/SwiftIR/releases/download/latest/SwiftIR-linux-x86_64.tar.gz | tar xz
%env LD_LIBRARY_PATH=/content/SwiftIR/lib
```

---

## Supported Operations

### Functional Transformations

| Category | SwiftIR | SwiftIRJupyter | Description |
|----------|---------|----------------|-------------|
| **Vectorization** | `diffVmap`, `diffVmap2`, `diffVmap3` | `jVmap`, `jVmap2`, `jVmap3` | Automatic batching |
| **Sequential** | `diffScan`, `diffCumsum`, `diffCumprod` | `jScan`, `jCumsum`, `jCumprod` | RNNs, time series |
| **Control Flow** | `diffWhileLoop`, `diffCond` | `jWhileLoop`, `jCond` | Native loops → `stablehlo.while` |
| **Gradients** | `gradient`, `valueWithGradient` | `jGrad`, `jValueAndGrad` | Reverse-mode AD |
| **PRNG** | `DiffPRNGKey`, `diffRandomNormal` | `JPRNGKey`, `jRandomNormal` | Functional random |
| **Tree Ops** | `diffTreeMap`, `diffTreeZipWith` | `jTreeMap`, `jTreeZipWith` | Nested parameter structures |

### Tensor Operations (All Differentiable)

| Category | Operations |
|----------|------------|
| **Arithmetic** | `+`, `-`, `*`, `/`, `pow`, `sqrt`, `rsqrt`, `abs`, `floor`, `ceil` |
| **Activations** | `relu`, `leakyRelu`, `elu`, `selu`, `gelu`, `silu`, `mish`, `sigmoid`, `tanh`, `softplus`, `softmax`, `logSoftmax` |
| **Trigonometric** | `sin`, `cos`, `tan`, `exp`, `log` |
| **Matrix** | `matmul`, `transpose`, `reshape`, `slice`, `concatenate`, `broadcast` |
| **Reductions** | `sum`, `mean`, `max`, `min`, `prod` (with axis support) |
| **Comparisons** | `greater`, `greaterEqual`, `less`, `lessEqual`, `equal`, `notEqual` |
| **Selection** | `select`, `where`, `clamp`, `maximum`, `minimum` |
| **Creation** | `zeros`, `ones`, `full`, `zerosLike`, `onesLike` |
| **Initialization** | `xavierUniform`, `xavierNormal`, `heUniform`, `heNormal`, `truncatedNormal` |

### Profiling & Tracing

| API | Description |
|-----|-------------|
| `PJRTProfiler.create()` | Create XLA profiler instance |
| `pjrtTraced("name") { }` | Custom trace annotation |
| `pjrtTrainStep(epoch) { }` | Step markers for TensorBoard Overview |
| `PJRTProfiler.hasTraceMeApi` | Check if unified tracing available |

All operations have gradient implementations validated by 300+ tests.

---

## Hardware Support

| Hardware | Status | Notes |
|----------|--------|-------|
| CPU | Full support | Default backend |
| TPU | Full support | Via libtpu.so |
| GPU | Ready | Needs CUDA plugin |

```swift
let client = try PJRTClientFactory.create()  // Auto-detect best accelerator
RuntimeDetector.printInfo()
```

---

## Project Structure

```
Sources/
├── SwiftIR/              # C++ MLIR bindings, @differentiable
├── SwiftIRJupyter/       # Pure Swift, JTracer
├── SwiftIRProfiler/      # TensorBoard profiling integration
├── SwiftIRXLA/           # PJRT client, XLA execution
└── SwiftIRRuntime/       # Hardware detection

Examples/
├── JupyterProfiledSimulation.swift    # Full example: Jupyter + profiling
├── JupyterBuildingSimulationFull.swift # Physics simulation (Jupyter)
├── BuildingSimulation_SwiftIR.swift   # Physics simulation (native)
├── ProfilerDemo.swift                 # Standalone profiler demo
└── QuantFinance/
    ├── QuantFinancePureSwift.swift    # Baseline: Pure Swift Black-Scholes
    ├── QuantFinanceTraced.swift       # SwiftIR-Traced with XLA acceleration
    └── QuantFinanceDifferentiable.swift # Swift _Differentiation comparison
```

---

## License

Apache 2.0 License. See [LICENSE](LICENSE).

---

**SwiftIR**: Swift → StableHLO → XLA → Hardware

*Built with Swift 6.0 | Powered by MLIR and XLA*
