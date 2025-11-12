# SPIR-V Integration Guide

## Overview

SPIR-V (Standard Portable Intermediate Representation - V) is a binary intermediate language for graphics and compute kernels. SwiftIR's SPIR-V integration enables GPU execution via:
- **Vulkan** (cross-platform: Windows, Linux, macOS via MoltenVK)
- **Metal** (macOS/iOS - native or via SPIR-V cross-compilation)
- **OpenCL** (compute-focused)

## Why SPIR-V?

### vs. PJRT (XLA)
| Feature | SPIR-V + Vulkan | PJRT + XLA |
|---------|----------------|------------|
| Platform Support | Windows/Linux/macOS/Android/iOS | Linux/macOS (GPU), Google Cloud (TPU) |
| Graphics Integration | ✅ Full graphics pipeline | ❌ Compute only |
| Mobile Support | ✅ iOS/Android | ❌ Limited |
| Game Engine Integration | ✅ Native | ❌ Not designed for it |
| ML Optimizations | Good | Excellent |
| Setup Complexity | Moderate | Moderate |

### Use Cases
- **Mobile ML**: iOS/Android inference
- **Cross-platform**: Single codebase for all OSs
- **Graphics + ML**: Real-time rendering with ML
- **Edge Devices**: Embedded GPU acceleration

## Architecture

```
┌─────────────────────────────────────────┐
│   SwiftIR Application                    │
│   (Linalg/Tensor Operations)             │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   SPIRVPipeline.swift                    │
│   - Linalg → GPU dialect                 │
│   - GPU → SPIR-V dialect                 │
│   - SPIR-V serialization                 │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   VulkanRuntime.swift                    │
│   - Device enumeration                   │
│   - Buffer management                    │
│   - Shader module loading                │
│   - Compute pipeline execution           │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   Vulkan API (MoltenVK on macOS)        │
│   - VkDevice, VkQueue                    │
│   - VkBuffer (GPU memory)                │
│   - VkShaderModule (SPIR-V binary)      │
│   - VkPipeline (compute shader)          │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   GPU Hardware                            │
│   (AMD, NVIDIA, Intel, Apple Silicon)     │
└───────────────────────────────────────────┘
```

## Lowering Pipeline

### Stage 1: High-Level → Linalg
```
StableHLO/Tensor → Linalg (already implemented)
```

### Stage 2: Linalg → GPU Dialect
```mlir
// Before: Linalg operation
linalg.matmul ins(%A, %B : tensor<MxK>, tensor<KxN>)
              outs(%C : tensor<MxN>)

// After: GPU dialect with kernel launch
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z)
           threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z) {
  // Compute tile of matrix multiplication
  gpu.terminator
}
```

**MLIR Passes:**
- `convert-linalg-to-parallel-loops` - Parallelize loops
- `gpu-map-parallel-loops` - Map to GPU grid/blocks
- `convert-parallel-loops-to-gpu` - Generate GPU dialect

### Stage 3: GPU → SPIR-V Dialect
```mlir
// Before: GPU dialect
gpu.func @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  %idx = gpu.thread_id x
  %val = memref.load %arg0[%idx]
  memref.store %val, %arg1[%idx]
  gpu.return
}

// After: SPIR-V dialect
spirv.module Logical GLSL450 {
  spirv.func @kernel(%arg0: !spirv.ptr<!spirv.array<f32>>,
                      %arg1: !spirv.ptr<!spirv.array<f32>>) {
    %idx = spirv.GlobalInvocationId x
    %val = spirv.Load %arg0[%idx]
    spirv.Store %arg1[%idx], %val
    spirv.Return
  }
}
```

**MLIR Passes (LLVM 19+):**
- `convert-gpu-to-spirv` - Convert GPU ops to SPIR-V
- `convert-func-to-spirv` - Convert functions
- `convert-arith-to-spirv` - Convert arithmetic ops
- `convert-scf-to-spirv` - Convert control flow
- `convert-vector-to-spirv` - Convert vector ops

### Stage 4: SPIR-V → Binary
```swift
// Serialize SPIR-V module to binary
let spirvBinary: [UInt32] = spirvModule.toBinary()

// Load into Vulkan
let shaderModule = VkShaderModule(device: device, spirvCode: spirvBinary)
```

## Components

### 1. SPIR-V Lowering Pipeline

`Sources/SwiftIRXLA/SPIRV/SPIRVPipeline.swift`:
```swift
public class SPIRVPipeline {
    private let context: MLIRContext

    /// Lower Linalg to SPIR-V
    public func lowerToSPIRV(module: MLIRModule) throws

    /// Serialize SPIR-V module to binary
    public func serializeSPIRV(module: MLIRModule) throws -> [UInt32]
}
```

### 2. Vulkan Runtime

`Sources/SwiftIRXLA/SPIRV/VulkanRuntime.swift`:
```swift
public class VulkanDevice {
    /// Physical device (GPU)
    public let physicalDevice: VkPhysicalDevice

    /// Logical device handle
    public let device: VkDevice

    /// Compute queue
    public let computeQueue: VkQueue

    /// Device properties
    public let properties: VkPhysicalDeviceProperties
}

public class VulkanBuffer {
    /// Buffer handle
    public let buffer: VkBuffer

    /// Device memory
    public let memory: VkDeviceMemory

    /// Size in bytes
    public let size: Int

    /// Copy from host to device
    public func upload(data: UnsafeRawPointer) throws

    /// Copy from device to host
    public func download(data: UnsafeMutableRawPointer) throws
}

public class VulkanExecutable {
    /// Shader module (SPIR-V binary)
    public let shaderModule: VkShaderModule

    /// Compute pipeline
    public let pipeline: VkPipeline

    /// Execute compute shader
    public func execute(buffers: [VulkanBuffer],
                       workgroupCount: (UInt32, UInt32, UInt32)) throws
}
```

### 3. Metal Backend (macOS/iOS)

Option A: Via MoltenVK (Vulkan → Metal translation)
Option B: Direct Metal Shading Language generation

`Sources/SwiftIRXLA/SPIRV/MetalRuntime.swift`:
```swift
#if os(macOS) || os(iOS)
import Metal

public class MetalDevice {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
}

public class MetalExecutable {
    public let library: MTLLibrary
    public let function: MTLFunction
    public let pipeline: MTLComputePipelineState

    public func execute(buffers: [MTLBuffer],
                       threadgroups: MTLSize,
                       threadsPerThreadgroup: MTLSize) throws
}
#endif
```

## Installation

### Prerequisites

#### Vulkan SDK (All Platforms)
```bash
# macOS
brew install vulkan-sdk molten-vk

# Linux
sudo apt-get install vulkan-tools libvulkan-dev vulkan-validationlayers

# Windows
# Download from https://vulkan.lunarg.com/
```

#### Metal (macOS/iOS only)
Metal is included with Xcode - no additional installation needed.

### Building SwiftIR with SPIR-V Support

The SPIR-V dialect is part of MLIR, so it's already available in our LLVM build:

```bash
# SPIR-V dialect is automatically included with MLIR
# Just need to add Vulkan linkage to Package.swift

.target(
    name: "SwiftIRXLA",
    dependencies: ["SwiftIRCore"],
    linkerSettings: [
        .linkedLibrary("vulkan", .when(platforms: [.linux, .windows])),
        .linkedFramework("Metal", .when(platforms: [.macOS, .iOS]))
    ]
)
```

## Usage Examples

### Basic Compute Kernel

```swift
import SwiftIRXLA

// Create Vulkan device
let devices = try VulkanRuntime.enumerateDevices()
let device = devices.first!

// Build SPIR-V module
let context = MLIRContext()
let module = MLIRModule(context: context)

// ... build Linalg IR ...

// Lower to SPIR-V
let pipeline = SPIRVPipeline(context: context)
try pipeline.lowerToSPIRV(module: module)

// Serialize to binary
let spirvBinary = try pipeline.serializeSPIRV(module: module)

// Create executable
let executable = try VulkanExecutable(
    device: device,
    spirvCode: spirvBinary
)

// Allocate buffers
let inputBuffer = try device.createBuffer(size: 1024)
let outputBuffer = try device.createBuffer(size: 1024)

// Upload data
var inputData: [Float] = Array(repeating: 1.0, count: 256)
try inputBuffer.upload(data: &inputData)

// Execute
try executable.execute(
    buffers: [inputBuffer, outputBuffer],
    workgroupCount: (16, 1, 1)
)

// Download result
var outputData = [Float](repeating: 0, count: 256)
try outputBuffer.download(data: &outputData)
```

### Matrix Multiplication

```swift
// Create 2x2 matrix multiplication kernel
let A = [[1.0, 2.0], [3.0, 4.0]]
let B = [[5.0, 6.0], [7.0, 8.0]]

// Build Linalg matmul IR
let matmul = LinalgMLOps.matmul(lhs: A, rhs: B, ...)

// Lower to SPIR-V
let pipeline = SPIRVPipeline(context: context)
try pipeline.lowerToSPIRV(module: module)

// Execute on GPU
let result = try executable.execute(...)
// Result: [[19, 22], [43, 50]]
```

## Performance Characteristics

### Vulkan (Cross-Platform)
- **Throughput**: 1-10 TFLOPS (consumer GPUs)
- **Latency**: 0.5-2ms kernel launch overhead
- **Memory**: Explicit management, DMA transfers
- **Best for**: Compute-heavy workloads, cross-platform

### Metal (macOS/iOS)
- **Throughput**:
  - Apple Silicon (M1/M2/M3): 2-15 TFLOPS
  - AMD GPUs: 5-10 TFLOPS
- **Latency**: <0.5ms (unified memory on Apple Silicon)
- **Memory**: Unified memory on Apple Silicon (zero-copy)
- **Best for**: Apple ecosystem, graphics + ML

## Optimization Tips

### 1. Workgroup Size
```swift
// Good: Multiple of 32/64 (warp/wavefront size)
workgroupSize = (64, 1, 1)

// Bad: Not a multiple of hardware size
workgroupSize = (37, 1, 1)
```

### 2. Memory Access Patterns
```swift
// Good: Coalesced access
for i in 0..<N {
    output[threadId * N + i] = input[threadId * N + i]
}

// Bad: Strided access
for i in 0..<N {
    output[i * gridSize + threadId] = input[i * gridSize + threadId]
}
```

### 3. Buffer Reuse
```swift
// Reuse buffers across kernel launches
let buffer = device.createBuffer(size: maxSize)
for iteration in 0..<100 {
    executable.execute(buffers: [buffer, ...])
}
```

## Debugging

### SPIR-V Validation
```bash
# Validate SPIR-V binary
spirv-val shader.spv

# Disassemble to readable format
spirv-dis shader.spv -o shader.spvasm
```

### Vulkan Validation Layers
```swift
// Enable validation in debug builds
#if DEBUG
let createInfo = VkInstanceCreateInfo(
    enabledLayerCount: 1,
    ppEnabledLayerNames: ["VK_LAYER_KHRONOS_validation"]
)
#endif
```

## Roadmap

### Phase 1: Foundation (Current)
- [ ] SPIR-V dialect integration
- [ ] Basic lowering pipeline (Linalg → GPU → SPIR-V)
- [ ] SPIR-V serialization

### Phase 2: Vulkan Backend
- [ ] Vulkan device enumeration
- [ ] Buffer management
- [ ] Compute pipeline creation
- [ ] Kernel execution

### Phase 3: Metal Backend
- [ ] Metal device wrapper
- [ ] SPIR-V → MSL cross-compilation (via spirv-cross)
- [ ] Metal compute pipeline
- [ ] Unified memory optimization

### Phase 4: Optimization
- [ ] Kernel fusion
- [ ] Memory pooling
- [ ] Asynchronous execution
- [ ] Multi-GPU support

## References

- [MLIR SPIR-V Dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Compute](https://www.khronos.org/blog/vulkan-subgroup-tutorial)
- [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [SPIR-V Specification](https://www.khronos.org/registry/SPIR-V/)
- [MoltenVK (Vulkan on Metal)](https://github.com/KhronosGroup/MoltenVK)

## Support

For SPIR-V/Vulkan issues:
1. Check Vulkan SDK installation
2. Verify GPU driver supports Vulkan 1.2+
3. Enable validation layers for detailed errors
4. Review SPIR-V assembly output
