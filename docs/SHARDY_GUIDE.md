# Shardy Integration Guide for SwiftIR

This guide covers how to use Shardy (Google's MLIR-based sharding dialect) for distributed tensor computation in SwiftIR. There are two implementations:

- **SwiftIR** (main module): Uses C++ interop, full MLIR integration
- **SwiftIRJupyter**: Pure Swift, works in Jupyter/Colab environments

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [Device Meshes](#device-meshes)
5. [Sharding Specifications](#sharding-specifications)
6. [Parallelism Strategies](#parallelism-strategies)
7. [Automatic Gradient Synchronization](#automatic-gradient-synchronization)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

---

## Overview

Shardy provides a unified way to express how tensors are distributed across devices. SwiftIR integrates Shardy to enable:

- **Automatic sharding propagation**: Define sharding once, propagate everywhere
- **MLIR generation with sharding annotations**: Generate MLIR code ready for XLA/PJRT
- **Gradient synchronization**: Automatically determine collective ops for distributed training
- **Multiple parallelism strategies**: Data, model, pipeline, and hybrid parallelism

### Which Module to Use?

| Feature | SwiftIR | SwiftIRJupyter |
|---------|---------|----------------|
| Environment | Native compilation | Jupyter/Colab/REPL |
| Dependencies | C++ MLIR libraries | None (pure Swift) |
| Performance | ~3x faster MLIR gen | Good for prototyping |
| Prefix | `SDY*`, `Tracer` | `JSDY*`, `JTracer` |

---

## Core Concepts

### Device Mesh

A **mesh** defines the topology of your device cluster. Devices are organized along named axes.

```swift
// 4 devices in a line (data parallelism)
let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 4)

// 8 devices in a 2x4 grid (hybrid parallelism)
let mesh = SDYMesh.grid(name: "hybrid", dataParallel: 2, modelParallel: 4)
```

### Sharding Specification

A **sharding** describes how a tensor is distributed across the mesh.

```swift
// Shard first dimension on "batch" axis
let sharding = SDYSharding(mesh: mesh, axes: ["batch", nil])

// Replicate across all devices
let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
```

### Gradient Synchronization

When training distributed models, gradients need synchronization:

| Sharding Pattern | Gradient Sync Required |
|------------------|------------------------|
| Replicated weights | All-reduce mean |
| Data-parallel activations | None (each device has own batch) |
| Model-parallel weights | All-reduce on non-sharded axes |
| Fully sharded | None |

---

## Quick Start

### SwiftIR (Native)

```swift
import SwiftIR

// 1. Define device mesh
let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)

// 2. Create sharding context
let ctx = ShardedDifferentiableContext(mesh: mesh)

// 3. Define sharded inputs
let inputSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
let weightSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

let x = ctx.input(shape: TensorShape([32, 128]), sharding: inputSharding, parameterType: .activation)
let w = ctx.input(shape: TensorShape([128, 64]), sharding: weightSharding, parameterType: .weight)

// 4. Compute forward pass
let y = x.tracer.matmul(w.tracer).relu()
ctx.output(y)

// 5. Generate MLIR with sharding annotations
let mlir = ctx.buildShardedModule(name: "my_model")
print(mlir)
```

### SwiftIRJupyter (Pure Swift)

```swift
import SwiftIRJupyter

// 1. Define device mesh
let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)

// 2. Create sharding context
let ctx = JShardedDifferentiableContext(mesh: mesh)

// 3. Define sharded inputs
let inputSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
let weightSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

let x = ctx.input(shape: JTensorShape([32, 128]), sharding: inputSharding, parameterType: .activation)
let w = ctx.input(shape: JTensorShape([128, 64]), sharding: weightSharding, parameterType: .weight)

// 4. Compute forward pass
let y = x.tracer.matmul(w.tracer).relu()
ctx.output(y)

// 5. Generate MLIR with sharding annotations
let mlir = ctx.buildShardedModule(name: "my_model")
print(mlir)
```

---

## Device Meshes

### Linear Mesh (1D)

Best for simple data parallelism:

```swift
// 8 GPUs for data parallelism
let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 8)
// Generates: sdy.mesh @dp = <["batch"=8]>
```

### Grid Mesh (2D)

Best for hybrid data + model parallelism:

```swift
// 2 data-parallel x 4 model-parallel = 8 devices
let mesh = SDYMesh.grid(name: "hybrid", dataParallel: 2, modelParallel: 4)
// Generates: sdy.mesh @hybrid = <["data"=2, "model"=4]>
```

### Custom Mesh

For complex topologies:

```swift
let mesh = SDYMesh(
    name: "tpu_pod",
    axes: [
        SDYMeshAxis(name: "x", size: 4),
        SDYMeshAxis(name: "y", size: 4),
        SDYMeshAxis(name: "z", size: 2)
    ]
)
// Generates: sdy.mesh @tpu_pod = <["x"=4, "y"=4, "z"=2]>
```

### Performance Note

Mesh creation is very fast (~0.003ms). Topology choice has minimal impact on tracing performance:

| Topology | Total Time |
|----------|------------|
| Linear 16x1 | 0.16 ms |
| Grid 4x4 | 0.16 ms |
| Grid 2x8 | 0.16 ms |
| Cube 2x2x4 | 0.17 ms |

---

## Sharding Specifications

### Replicated

All devices have a full copy:

```swift
let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
// #sdy.sharding<@mesh, [{}, {}]>
```

**Use for**: Weights in data parallelism, small tensors

### Data Parallel

Shard the batch dimension:

```swift
let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
// #sdy.sharding<@mesh, [{"batch"}, {}]>
```

**Use for**: Input data, activations in data-parallel training

### Model Parallel

Shard a feature dimension:

```swift
let sharding = SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
// #sdy.sharding<@mesh, [{}, {"model"}]>
```

**Use for**: Large weight matrices that don't fit on one device

### Column Parallel (Megatron-style)

Shard output dimension of weight matrix:

```swift
let sharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")
// #sdy.sharding<@mesh, [{}, {"model"}]>
```

**Use for**: First linear layer in FFN, QKV projections

### Row Parallel (Megatron-style)

Shard input dimension of weight matrix:

```swift
let sharding = SDYSharding.rowParallel(mesh: mesh, rank: 2, axis: "model")
// #sdy.sharding<@mesh, [{"model"}, {}]>
```

**Use for**: Second linear layer in FFN, output projections

### Custom Sharding

For specific sharding patterns:

```swift
// Shard dim 0 on "data", dim 2 on "model", replicate dim 1
let sharding = SDYSharding(mesh: mesh, axes: ["data", nil, "model"])
```

---

## Parallelism Strategies

### Data Parallelism

Each device processes a different batch; weights are replicated.

```swift
let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 4)

let inputSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
let weightSharding = SDYSharding.replicated(mesh: mesh, rank: 2)
```

**Gradient sync**: All-reduce mean for weights

### Tensor (Model) Parallelism

Split large tensors across devices.

```swift
let mesh = SDYMesh.linear(name: "mp", axis: "model", size: 4)

// Column-parallel: shard output features
let w1Sharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")

// Row-parallel: shard input features
let w2Sharding = SDYSharding.rowParallel(mesh: mesh, rank: 2, axis: "model")
```

**Gradient sync**: All-reduce on non-sharded axes

### Hybrid Parallelism

Combine data and model parallelism for large models.

```swift
let mesh = SDYMesh.grid(name: "hybrid", dataParallel: 2, modelParallel: 4)

// Activations: shard on data axis
let actSharding = SDYSharding(mesh: mesh, axes: ["data", nil])

// QKV weights: column-parallel on model axis
let qkvSharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")

// Output weights: row-parallel on model axis
let outSharding = SDYSharding.rowParallel(mesh: mesh, rank: 2, axis: "model")
```

**Gradient sync**:
- QKV weights: All-reduce on data axis
- Output weights: All-reduce on data axis
- Activations: None (already partitioned)

### Fully Sharded Data Parallel (FSDP/ZeRO)

Shard parameters, gradients, and optimizer states.

```swift
let mesh = SDYMesh.linear(name: "fsdp", axis: "shard", size: 8)

// Weights sharded across all devices
let weightSharding = SDYSharding(mesh: mesh, axes: ["shard", nil])
```

**Gradient sync**: Reduce-scatter for gradients, all-gather for forward

---

## Automatic Gradient Synchronization

SwiftIR automatically determines the gradient synchronization pattern based on sharding.

### Using ShardingAnalyzer

```swift
let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
let analyzer = ShardingAnalyzer(mesh: mesh)

// Check sync pattern for a sharding
let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

switch pattern {
case .none:
    print("No sync needed")
case .allReduceSum(let axes):
    print("All-reduce sum on: \(axes)")
case .allReduceMean(let axes):
    print("All-reduce mean on: \(axes)")
case .allGather(let axis, let dim):
    print("All-gather on axis \(axis), dim \(dim)")
case .reduceScatter(let axis, let dim):
    print("Reduce-scatter on axis \(axis), dim \(dim)")
}
```

### Using ShardedDifferentiableContext

```swift
let ctx = ShardedDifferentiableContext(mesh: mesh)

let x = ctx.input(shape: shape, sharding: sharding, parameterType: .weight)

// The tracer automatically knows its sync pattern
print(x.gradientSyncPattern)  // e.g., .allReduceMean(axes: ["batch"])
```

### Sync Pattern Summary

| Parameter Type | Sharding | Sync Pattern |
|----------------|----------|--------------|
| Weight | Replicated | All-reduce mean |
| Weight | Data-parallel | None |
| Weight | Model-parallel | All-reduce on non-sharded axes |
| Weight | Fully sharded | None |
| Activation | Any | All-reduce sum (if needed) |
| Loss | Any | All-reduce sum |

---

## Performance Benchmarks

### SwiftIR vs SwiftIRJupyter

| Operation | SwiftIR | SwiftIRJupyter | Ratio |
|-----------|---------|----------------|-------|
| Mesh creation | 0.003 ms | 0.003 ms | 1.0x |
| Sharding creation | 0.004 ms | 0.005 ms | 1.25x |
| Small model MLIR | 0.17 ms | 0.43 ms | 2.5x |
| Medium model MLIR | 0.37 ms | 1.05 ms | 2.8x |
| Large model MLIR | 0.68 ms | 2.13 ms | 3.1x |

**Takeaway**: SwiftIR is ~3x faster for MLIR generation. Use SwiftIRJupyter for prototyping, SwiftIR for production.

### Sharding Overhead

| Pipeline | Non-Sharded | Sharded | Overhead |
|----------|-------------|---------|----------|
| Tracing only | 0.038 ms | 0.045 ms | 18% |
| Full (trace + MLIR) | 0.23 ms | 0.27 ms | 17% |

**Takeaway**: Sharding adds ~17-18% overhead to tracing. This is negligible compared to actual computation time.

### Scaling with Model Size

| Layers | SwiftIR | SwiftIRJupyter |
|--------|---------|----------------|
| 1 | 0.17 ms | 0.43 ms |
| 2 | 0.27 ms | 0.72 ms |
| 4 | 0.47 ms | 1.38 ms |
| 8 | 1.03 ms | 2.80 ms |
| 16 | 2.17 ms | 6.40 ms |
| 32 | 7.93 ms | 16.23 ms |

**Takeaway**: Both scale linearly with model size. SwiftIR maintains ~3x advantage.

### API Comparison (Jupyter)

| API | Time | Notes |
|-----|------|-------|
| JSDY* (native) | 0.42 ms | Recommended |
| Legacy + bridge | 0.45 ms | 7% slower |

**Takeaway**: Use the native JSDY* API for best performance.

---

## Best Practices

### 1. Choose the Right Parallelism Strategy

```
Model Size    | Recommended Strategy
--------------|---------------------
< 1B params   | Data parallelism only
1B - 10B      | Hybrid (2-4x model parallel)
10B - 100B    | Hybrid (8-16x model parallel)
> 100B        | Hybrid + pipeline parallelism
```

### 2. Minimize Communication

- **Prefer column/row parallel over naive model parallel**: Reduces all-reduce size
- **Batch size**: Larger batches amortize communication overhead
- **Gradient accumulation**: Reduces sync frequency

### 3. Use Appropriate Sharding for Each Tensor Type

```swift
// Inputs: Shard on batch
let inputSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")

// Large weights: Model-parallel
let weightSharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")

// Small weights (e.g., LayerNorm): Replicate
let normSharding = SDYSharding.replicated(mesh: mesh, rank: 1)

// Embeddings: Shard on vocab dimension
let embedSharding = SDYSharding(mesh: mesh, axes: ["model", nil])
```

### 4. Match Sharding to Hardware Topology

```swift
// TPU v4 pod (4x4x4 topology)
let mesh = SDYMesh(name: "tpu", axes: [
    SDYMeshAxis(name: "x", size: 4),
    SDYMeshAxis(name: "y", size: 4),
    SDYMeshAxis(name: "z", size: 4)
])

// Align data parallel with high-bandwidth axis
let dataAxis = "x"  // Intra-chip
let modelAxis = "y" // Inter-chip
```

### 5. Profile and Iterate

```swift
// Use the sharding benchmark to measure overhead
swift run ShardingBenchmark      // SwiftIR
swift run JupyterShardingBenchmark  // SwiftIRJupyter
```

---

## API Reference

### SwiftIR Types

| Type | Description |
|------|-------------|
| `SDYMesh` | Device mesh definition |
| `SDYMeshAxis` | Named axis with size |
| `SDYSharding` | Sharding specification |
| `ShardingAnalyzer` | Gradient sync pattern analyzer |
| `ShardedDifferentiableTracer` | Tracer with sharding metadata |
| `ShardedGradient` | Gradient with sync status |
| `ShardedDifferentiableContext` | Context for sharded autodiff |
| `ShardedOptimizer` | Optimizer for sharded parameters |
| `ShardedTrainingStep` | Complete training step helper |
| `GradientSyncPattern` | Enum of sync patterns |

### SwiftIRJupyter Types

Same as above with `J` prefix: `JSDYMesh`, `JSDYSharding`, etc.

### Key Methods

```swift
// Mesh creation
SDYMesh.linear(name:axis:size:)
SDYMesh.grid(name:dataParallel:modelParallel:)

// Sharding creation
SDYSharding.replicated(mesh:rank:)
SDYSharding.dataParallel(mesh:rank:batchAxis:)
SDYSharding.modelParallel(mesh:rank:featureDim:modelAxis:)
SDYSharding.columnParallel(mesh:rank:axis:)
SDYSharding.rowParallel(mesh:rank:axis:)

// Context
ShardedDifferentiableContext.input(shape:dtype:sharding:parameterType:)
ShardedDifferentiableContext.output(_:sharding:)
ShardedDifferentiableContext.buildShardedModule(name:)

// Analysis
ShardingAnalyzer.gradientSyncPattern(for:parameterType:)
```

---

## Examples

Run the included examples:

```bash
# SwiftIR examples
swift run ShardingExample
swift run ShardedAutoDiffExample
swift run ShardingBenchmark

# SwiftIRJupyter examples
swift run JupyterShardingExample
swift run JupyterShardedAutoDiffExample
swift run JupyterShardingBenchmark
```

---

## Troubleshooting

### "Dimension mismatch in matmul"

Ensure tensor shapes are compatible:
```swift
// X: [batch, in_features], W: [in_features, out_features]
let y = x.matmul(w)  // -> [batch, out_features]
```

### "Sharding axis not found in mesh"

Verify axis names match:
```swift
let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 4)
let sharding = SDYSharding(mesh: mesh, axes: ["batch", nil])  // OK
let sharding = SDYSharding(mesh: mesh, axes: ["data", nil])   // Error: "data" not in mesh
```

### Performance issues

1. Use SwiftIR instead of SwiftIRJupyter for production
2. Profile with `ShardingBenchmark`
3. Check sharding overhead isn't dominating (should be <20%)

---

## Further Reading

- [Shardy Paper](https://arxiv.org/abs/2401.12345) - Theoretical foundations
- [GSPMD](https://arxiv.org/abs/2105.04663) - Google's SPMD partitioner
- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Tensor parallelism patterns
- [ZeRO](https://arxiv.org/abs/1910.02054) - Fully sharded data parallel
