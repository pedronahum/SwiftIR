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

## Performance

Benchmarks use the thermal building simulation (real physics, not synthetic):

### O(1) Compilation for Loops

| Iterations | While Loop | Unrolled | Speedup |
|------------|------------|----------|---------|
| 1,000 | 43ms | 24s | **566x** |
| 10,000 | 42ms | 4.2min | **6,002x** |
| 100,000 | 43ms | ~42min | **59,523x** |

`stablehlo.while` maintains constant compilation time regardless of iteration count.

### Gradient Overhead

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
└── ProfilerDemo.swift                 # Standalone profiler demo
```

---

## License

Apache 2.0 License. See [LICENSE](LICENSE).

---

**SwiftIR**: Swift → StableHLO → XLA → Hardware

*Built with Swift 6.0 | Powered by MLIR and XLA*
