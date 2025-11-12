# SwiftIRGPU

**GPU dialect operations and utilities (Planned)**

## Purpose

SwiftIRGPU will provide Swift bindings for MLIR's GPU dialect, enabling explicit GPU kernel definition and launch operations.

## Current Status

**This module is currently empty and reserved for future development.**

GPU lowering functionality is currently implemented in:
- `SwiftIRXLA/SPIRV/` - SPIR-V pipeline for Vulkan/OpenCL
- `SwiftIRXLA/LLVM/` - LLVM/PTX pipeline for NVIDIA CUDA

See [GPU_LOWERING_ROADMAP.md](../../GPU_LOWERING_ROADMAP.md) for implementation plan.

## Planned Features

### GPU Dialect Operations
```swift
// GPU kernel definition
gpu.module @kernels {
    gpu.func @vector_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>)
        kernel attributes {workgroup_size = [256, 1, 1]} {
        %tid = gpu.thread_id x
        %val0 = memref.load %arg0[%tid]
        %val1 = memref.load %arg1[%tid]
        %sum = arith.addf %val0, %val1
        memref.store %sum, %arg0[%tid]
        gpu.return
    }
}

// GPU kernel launch
gpu.launch_func @kernels::@vector_add
    blocks in (%num_blocks, %c1, %c1)
    threads in (%threads_per_block, %c1, %c1)
    args(%buffer0, %buffer1)
```

### Swift API (Planned)
```swift
// Define GPU kernel
let kernel = GPU.kernel("vector_add", workgroupSize: [256, 1, 1]) { builder in
    let tid = GPU.threadId(.x, context: context)
    // Kernel body
}

// Launch kernel
GPU.launch(
    kernel,
    gridSize: [numBlocks, 1, 1],
    blockSize: [256, 1, 1],
    arguments: [buffer0, buffer1]
)
```

## Why Separate Module?

GPU operations are complex enough to warrant their own module:
- Distinct dialect with unique semantics
- Thread/block hierarchy management
- Memory space management (global, shared, local)
- Barrier synchronization
- Specialized lowering pipelines

## Integration Points

### With SwiftIRXLA
SwiftIRXLA's GPU lowering pipelines will use operations from this module:
```
Linalg ops (SwiftIRXLA)
       ↓
GPU ops (SwiftIRGPU) - kernel outlining
       ↓
SPIR-V or LLVM (SwiftIRXLA/SPIRV or SwiftIRXLA/LLVM)
```

### With SwiftIRDialects
GPU dialect will integrate with other dialects:
- `scf` - Structured control flow → GPU parallel loops
- `memref` - Memory operations in GPU address spaces
- `arith` - Arithmetic operations in kernels

## Target Platforms

Once implemented, SwiftIRGPU will support:
- **NVIDIA CUDA** - via LLVM/NVVM → PTX
- **AMD ROCm** - via LLVM/AMDGPU → GCN/RDNA
- **Vulkan** - via SPIR-V
- **OpenCL** - via SPIR-V
- **Metal** - via Metal Shading Language (planned)

## Implementation Roadmap

See [GPU_LOWERING_ROADMAP.md](../../GPU_LOWERING_ROADMAP.md) for detailed implementation plan.

### Phase 1: Basic Operations (Future)
- GPU kernel definition operations
- Launch operations
- Thread/block ID queries
- Barriers

### Phase 2: Memory Management (Future)
- Shared memory allocation
- Memory space annotations
- Async copies

### Phase 3: Advanced Features (Future)
- Tensor cores / matrix operations
- Cooperative groups
- Dynamic parallelism

## Dependencies

### Internal (Future)
- **SwiftIRCore** - MLIR foundation
- **SwiftIRTypes** - Type system
- **SwiftIRDialects** - Basic operations

### External (Future)
- MLIR GPU dialect
- GPU runtime libraries (CUDA, ROCm, Vulkan, etc.)

## Design Principles (When Implemented)

### Explicit Control
Unlike automatic GPU offloading, this module provides explicit control over:
- Kernel definition
- Thread/block dimensions
- Memory allocation
- Synchronization

### Portability
Operations should be portable across GPU backends where possible, with backend-specific lowering handling differences.

### Safety
Swift's type system will help prevent common GPU programming errors:
- Memory space violations
- Race conditions
- Barrier misuse

## Related Modules

- **SwiftIRXLA** - Contains GPU lowering pipelines
- **SwiftIRCore** - Foundation
- **SwiftIRDialects** - Related dialects (scf, memref)

---

**Status**: Module reserved for future development. GPU functionality currently in SwiftIRXLA.
