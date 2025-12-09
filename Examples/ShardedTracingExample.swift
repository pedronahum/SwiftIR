/// ShardedTracingExample.swift - SDY Sharding with SwiftIR Tracer
///
/// Demonstrates how to use SDY sharding annotations with the main SwiftIR
/// module for distributed tensor computation.

import SwiftIR

// MARK: - Example 1: Basic Data Parallelism

/// Demonstrates data-parallel sharding for a simple matmul.
func dataParallelMatmulExample() {
    print("=== Example 1: Data Parallel Matmul ===\n")

    // Create a 4-device linear mesh for data parallelism
    let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
    mesh.printInfo()
    print()

    // Create sharded tracing context
    let ctx = ShardedTracingContext(mesh: mesh)

    // Define sharded inputs:
    // - Input X[16, 64]: shard batch dimension across "batch" axis
    // - Weight W[64, 32]: fully replicated
    let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
    let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

    print("Input shardings:")
    xSharding.printInfo(shape: TensorShape([16, 64]))
    wSharding.printInfo(shape: TensorShape([64, 32]))
    print()

    // Create symbolic inputs with shardings
    let x = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([64, 32]), dtype: .float32, sharding: wSharding)

    // Trace the computation
    let y = x.tracer.matmul(w.tracer)

    // Set output with data-parallel sharding
    ctx.output(y, sharding: xSharding)

    // Build sharded MLIR module
    let mlir = ctx.buildShardedModule(name: "data_parallel_matmul")

    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 2: Hybrid 2D Parallelism

/// Demonstrates 2D parallelism combining data and model sharding.
func hybridParallelExample() {
    print("=== Example 2: Hybrid 2D Parallelism ===\n")

    // Create a 2x4 mesh: 2 devices for data, 4 for model
    let mesh = SDYMesh.grid(
        name: "hybrid_mesh",
        dataParallel: 2,
        modelParallel: 4,
        dataAxis: "data",
        modelAxis: "model"
    )
    mesh.printInfo()
    print()

    // For a matmul Y = X @ W:
    // - X[batch, in_features]: shard batch on "data"
    // - W[in_features, out_features]: shard out_features on "model"
    // - Y[batch, out_features]: shard batch on "data", out on "model"

    let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
    let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])
    let ySharding = SDYSharding(mesh: mesh, axes: ["data", "model"])

    print("Tensor shardings:")
    print("  X: \(xSharding)")
    print("  W: \(wSharding)")
    print("  Y: \(ySharding)")
    print()

    // Create the context
    let ctx = ShardedTracingContext(mesh: mesh)

    // Create sharded inputs
    let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([128, 256]), dtype: .float32, sharding: wSharding)

    // Trace
    let y = x.tracer.matmul(w.tracer)

    ctx.output(y, sharding: ySharding)

    // Build module
    let mlir = ctx.buildShardedModule(name: "hybrid_parallel_matmul")
    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 3: Column-Parallel Dense Layer

/// Demonstrates column-parallel sharding for tensor parallelism.
func columnParallelExample() {
    print("=== Example 3: Column-Parallel Dense Layer ===\n")

    // 8 devices in a 2x4 TPU-style grid
    let mesh = SDYMesh.tpuMesh(name: "tpu_mesh", x: 2, y: 4)
    mesh.printInfo()
    print()

    // Column-parallel: shard the output dimension of weights
    let batchSize = 64
    let inputDim = 512
    let hiddenDim = 2048

    // Shardings for column-parallel dense layer
    let xSharding = SDYSharding(mesh: mesh, axes: ["x", nil])        // batch sharded
    let wSharding = SDYSharding(mesh: mesh, axes: [nil, "y"])        // column-parallel
    let outSharding = SDYSharding(mesh: mesh, axes: ["x", "y"])      // 2D sharded output

    print("Layer shardings:")
    print("  Input X [\(batchSize)x\(inputDim)]:   \(xSharding)")
    print("  Weight W [\(inputDim)x\(hiddenDim)]: \(wSharding)")
    print("  Output [\(batchSize)x\(hiddenDim)]:   \(outSharding)")
    print()

    // Create context
    let ctx = ShardedTracingContext(mesh: mesh)

    // Create inputs
    let x = ctx.input(shape: TensorShape([batchSize, inputDim]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([inputDim, hiddenDim]), dtype: .float32, sharding: wSharding)

    // Trace the layer: Y = relu(X @ W)
    let matmul = x.tracer.matmul(w.tracer)
    let output = matmul.relu()

    ctx.output(output, sharding: outSharding)

    // Build module
    let mlir = ctx.buildShardedModule(name: "column_parallel_dense")
    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 4: 3D Parallelism (Data + Tensor + Pipeline)

/// Demonstrates 3D parallelism for large model training.
func parallelism3DExample() {
    print("=== Example 4: 3D Parallelism ===\n")

    // Create a cube mesh: data x tensor x pipeline
    let mesh = SDYMesh.cube(
        name: "3d_mesh",
        data: 2,
        tensor: 4,
        pipeline: 2
    )
    mesh.printInfo()
    print()

    // For 3D parallelism:
    // - Data parallel: shard batch across "data" axis
    // - Tensor parallel: shard hidden dimension across "tensor" axis
    // - Pipeline parallel: handled separately (different stages)

    let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
    let wSharding = SDYSharding(mesh: mesh, axes: [nil, "tensor"])
    let ySharding = SDYSharding(mesh: mesh, axes: ["data", "tensor"])

    print("3D Parallelism shardings:")
    print("  Input X: \(xSharding)")
    print("  Weight W: \(wSharding)")
    print("  Output Y: \(ySharding)")
    print()
}

// MARK: - Example 5: Bridge from DeviceMesh API

/// Demonstrates bridging from existing DeviceMesh/ShardingSpec APIs.
func bridgeExample() {
    print("=== Example 5: Bridge from DeviceMesh ===\n")

    // Create using the existing DeviceMesh API
    let deviceMesh = DeviceMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
    print("DeviceMesh: \(deviceMesh)")

    // Convert to SDY mesh
    let sdyMesh = deviceMesh.toSDYMesh()
    print("SDYMesh MLIR: \(sdyMesh.mlirText)")
    print()

    // Create sharding spec using existing API
    let shardingSpec = ShardingSpec.dataParallel(mesh: deviceMesh, rank: 2)
    print("ShardingSpec: \(shardingSpec)")

    // Convert to SDY sharding
    let sdySharding = shardingSpec.toSDYSharding()
    print("SDYSharding: \(sdySharding)")
    print()
}

// MARK: - Example 6: Using Convenience Functions

/// Demonstrates the traceDataParallel helper function.
func convenienceFunctionExample() {
    print("=== Example 6: Convenience Functions ===\n")

    // Use traceDataParallel helper for simple data-parallel setup
    let mlir = traceDataParallel(
        numDevices: 4,
        inputShapes: [TensorShape([16, 64]), TensorShape([64, 32])],
        dtype: .float32
    ) { inputs in
        // Trace matmul
        inputs[0].matmul(inputs[1])
    }

    print("Data-parallel MLIR from helper:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Main

print("SwiftIR Sharded Tracing Examples")
print("=================================\n")

dataParallelMatmulExample()
hybridParallelExample()
columnParallelExample()
parallelism3DExample()
bridgeExample()
convenienceFunctionExample()

print("All examples completed!")
print()
print("Summary:")
print("  - SDYMesh: Define device mesh topologies (1D, 2D, 3D)")
print("  - SDYSharding: Specify tensor partitioning across mesh axes")
print("  - ShardedTracingContext: Trace with sharding annotations")
print("  - traceSharded: Compile sharded functions")
print("  - traceDataParallel: Quick data-parallel setup")
print()
print("Next steps:")
print("  1. Run SDY propagation via sdy_opt to infer missing shardings")
print("  2. Execute on multiple devices with PJRT")
print("  3. Combine with gradient synchronization for distributed training")
