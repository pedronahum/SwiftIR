# SwiftIRXLA

**XLA compiler integration, PJRT runtime, and GPU lowering pipelines**

## Purpose

SwiftIRXLA is the largest and most complex module in SwiftIR. It provides:
1. **XLA Integration** - Compile and execute MLIR via Google's XLA compiler
2. **PJRT Runtime** - Execute compiled code on CPU/GPU/TPU
3. **Tensor Operations** - High-level tensor and ML operations
4. **GPU Lowering** - SPIR-V and LLVM/PTX pipelines for GPU code generation
5. **StableHLO Integration** - Lowering to portable ML representation

## Directory Structure

```
SwiftIRXLA/
├── SwiftIRXLA.swift         # Main entry point and top-level API
├── TensorOps.swift           # Tensor operations (add, mul, conv, etc.)
├── MLOps.swift               # ML-specific operations (matmul, softmax, etc.)
├── ValueExtensions.swift     # Extensions for MLIRValue
├── TensorDialect.swift       # Tensor dialect operations
├── StableHLOMLOps.swift      # StableHLO ML operations
│
├── Runtime/
│   ├── Backend.swift         # Abstract backend interface
│   ├── Tensor.swift          # Runtime tensor type
│   └── MLIRCPUBackend.swift  # CPU execution backend
│
├── Backend/
│   ├── XLADevice.swift       # XLA device management
│   ├── XLACompiler.swift     # XLA compilation
│   └── XLAExecutable.swift   # Compiled executable wrapper
│
├── PJRT/
│   ├── PJRTCAPI.swift        # PJRT C API bindings
│   └── PJRTClient.swift      # High-level PJRT client
│
├── Lowering/
│   ├── PassManager.swift     # MLIR pass management
│   ├── LoweringPipeline.swift # Generic lowering infrastructure
│   ├── StableHLOPipeline.swift # Linalg → StableHLO → XLA
│   ├── ExecutionEngine.swift  # JIT execution engine
│
├── SPIRV/
│   ├── SPIRVPipeline.swift   # Linalg → GPU → SPIR-V
│   ├── VulkanRuntime.swift   # Vulkan runtime (planned)
│   └── MetalRuntime.swift    # Metal runtime (planned)
│
└── LLVM/
    └── LLVMPipeline.swift    # Linalg → GPU → LLVM/PTX

include/
└── TensorAPI.h               # C API for tensor operations
```

## Core Components

### 1. Tensor Operations (`TensorOps.swift`, `MLOps.swift`)

High-level operations on tensors:
```swift
// Element-wise operations
let result = tensor.add(a, b)
let scaled = tensor.mul(x, constant)

// ML operations
let output = tensor.matmul(input, weights)
let activated = tensor.relu(output)
let normalized = tensor.softmax(activated)
```

### 2. PJRT Runtime (`PJRT/`)

Execute compiled code via Plugin JAX Runtime:
```swift
// Create PJRT client (CPU/GPU/TPU)
let client = try PJRTClient.createCPU()

// Compile MLIR module
let executable = try client.compile(module)

// Execute with inputs
let outputs = try executable.execute(inputs)
```

**PJRT** is the standard runtime interface for:
- JAX (Google)
- PyTorch/XLA
- TensorFlow/XLA

### 3. XLA Backend (`Backend/`)

Integrate with Google's XLA (Accelerated Linear Algebra) compiler:
```swift
// Compile via XLA
let device = try XLADevice.default()
let compiler = XLACompiler(device: device)
let executable = try compiler.compile(module)

// Execute
let results = try executable.run(arguments: inputs)
```

XLA provides:
- Advanced optimizations (fusion, layout, memory planning)
- Multi-backend support (CPU, GPU, TPU)
- Production-grade performance

### 4. Lowering Pipelines (`Lowering/`)

Transform high-level operations to executable code:

#### StableHLO Pipeline
```
Linalg Operations
       ↓
StableHLO (portable ML IR)
       ↓
XLA HLO
       ↓
Backend-specific code (CPU/GPU/TPU)
```

#### GPU Pipelines

**SPIR-V** (Vulkan/OpenCL):
```
Linalg → SCF parallel loops → GPU dialect → SPIR-V
```

**LLVM/PTX** (NVIDIA CUDA):
```
Linalg → SCF parallel loops → GPU dialect → LLVM IR → PTX
```

### 5. Pass Management (`PassManager.swift`)

Orchestrate MLIR transformation passes:
```swift
let pm = PassManager(context: context)
pm.enableVerifier(true)
pm.parsePipeline("builtin.module(convert-linalg-to-parallel-loops)")
try pm.run(on: module)
```

## Usage Examples

### Example 1: Simple Tensor Computation via PJRT

```swift
import SwiftIRCore
import SwiftIRXLA

// Create context and module
let context = MLIRContext()
context.loadAllDialects()

// Define computation in Linalg
let module = try MLIRModule.parse("""
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic ... {
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
""", context: context)

// Lower to StableHLO
let pipeline = StableHLOPipeline(context: context)
try pipeline.run(on: module)

// Execute via PJRT
let client = try PJRTClient.createCPU()
let executable = try client.compile(module)
let outputs = try executable.execute(inputs: [a, b])
```

### Example 2: GPU Code Generation (LLVM)

```swift
let module = createLinalgModule()  // High-level tensor ops

// Lower to GPU code via LLVM
let llvmPipeline = LLVMPipeline(
    context: context,
    options: LLVMPipelineOptions(
        indexBitwidth: 64,
        computeCapability: "sm_80"
    )
)
try llvmPipeline.run(on: module)

// Result: LLVM IR with NVVM intrinsics
// Can be compiled to PTX via llc
```

## Key Design Decisions

### Separation of Concerns
- **Frontend**: Tensor operations and DSL (`TensorOps`, `MLOps`)
- **Middle**: Lowering and optimization (`Lowering/`, `PassManager`)
- **Backend**: Execution (`Runtime/`, `PJRT/`, `Backend/`)

### Multiple Execution Paths
SwiftIRXLA supports multiple ways to execute MLIR:
1. **PJRT** - Production runtime (preferred)
2. **XLA direct** - Direct XLA compilation
3. **ExecutionEngine** - JIT via MLIR's execution engine
4. **GPU pipelines** - Manual GPU code generation

### Extensibility
New backends and lowering pipelines can be added by:
1. Implementing backend interface
2. Creating lowering pipeline
3. Registering with runtime

## Current Status

### Production Ready
- PJRT runtime integration
- StableHLO lowering pipeline
- CPU execution
- Basic tensor operations

### In Development
- GPU lowering pipelines (see [GPU_LOWERING_ROADMAP.md](../../GPU_LOWERING_ROADMAP.md))
  - SPIR-V pipeline (blocked by MLIR build)
  - LLVM/PTX pipeline (blocked by MLIR build)
- Vulkan/Metal runtimes
- Advanced ML operations

## Dependencies

### Internal
- **SwiftIRCore** - MLIR foundation
- **SwiftIRTypes** - Type system
- **SwiftIRDialects** - Basic operations
- **SwiftIRStableHLO** - StableHLO dialect
- **PJRTCWrappers** - PJRT C API bindings

### External
- PJRT C API (from XLA)
- StableHLO MLIR build
- XLA compiler (optional)

## Related Modules

- **SwiftIRStableHLO** - StableHLO dialect operations
- **PJRTCWrappers** - C wrappers for PJRT
- **SwiftIRCore** - Foundation

## Technical Notes

### Why PJRT?
PJRT (Plugin JAX Runtime) is the standard interface for ML runtimes:
- Used by JAX, PyTorch/XLA, TensorFlow
- Vendor-neutral (works with any backend)
- Production-tested at Google scale

### Why StableHLO?
StableHLO provides:
- Portable ML operations (framework-agnostic)
- Version stability (backward compatibility)
- Direct path to XLA compiler

### GPU Lowering Challenge
See [GPU_LOWERING_ROADMAP.md](../../GPU_LOWERING_ROADMAP.md) for details on GPU code generation challenges and solutions.

The StableHLO MLIR build is optimized for TPU/XLA and lacks GPU conversion passes. Solutions:
1. Rebuild MLIR with full GPU support
2. Use PJRT with GPU plugin (recommended short-term)

---

**Next Steps**: See [SwiftIRStableHLO](../SwiftIRStableHLO/) for StableHLO operations.
