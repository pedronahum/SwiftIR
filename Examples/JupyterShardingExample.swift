/// Jupyter Sharding Example for SwiftIR
///
/// This example demonstrates how to use SDY sharding annotations with the
/// SwiftIRJupyter tracing API for distributed tensor computation.

import SwiftIRJupyter

// MARK: - Example 1: Basic Data Parallelism

/// Demonstrates data-parallel sharding for a simple matrix multiply.
func dataParallelExample() {
    print("=== Example 1: Data Parallel Matmul ===\n")

    // Create a 4-device linear mesh for data parallelism
    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
    mesh.printInfo()
    print()

    // Create sharded tracing context
    let ctx = JShardedTracingContext(mesh: mesh)

    // Define sharded inputs:
    // - Input X[16, 64]: shard batch dimension across "batch" axis
    // - Weight W[64, 32]: fully replicated
    let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
    let wSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

    print("Input shardings:")
    xSharding.printInfo(shape: [16, 64])
    wSharding.printInfo(shape: [64, 32])
    print()

    // Create symbolic inputs with shardings
    let x = ctx.input(shape: JTensorShape([16, 64]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: JTensorShape([64, 32]), dtype: .float32, sharding: wSharding)

    // Trace the computation
    let y = x.matmul(w)

    // Set output
    ctx.output(y, sharding: xSharding)

    // Build sharded MLIR module
    let mlir = ctx.buildShardedModule(name: "data_parallel_matmul")

    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 2: Hybrid Data + Model Parallelism

/// Demonstrates 2D parallelism combining data and model sharding.
func hybridParallelExample() {
    print("=== Example 2: Hybrid 2D Parallelism ===\n")

    // Create a 2x4 mesh: 2 devices for data, 4 for model
    let mesh = JSDYMesh.grid(
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

    let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
    let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])
    let ySharding = JSDYSharding(mesh: mesh, axes: ["data", "model"])

    print("Tensor shardings:")
    print("  X: \(xSharding)")
    print("  W: \(wSharding)")
    print("  Y: \(ySharding)")
    print()

    // Create the context
    let ctx = JShardedTracingContext(mesh: mesh)

    // Create sharded inputs
    let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32, sharding: wSharding)

    // Trace
    let y = x.matmul(w)

    ctx.output(y, sharding: ySharding)

    // Build module
    let mlir = ctx.buildShardedModule(name: "hybrid_parallel_matmul")
    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 3: Neural Network Layer with Sharding

/// Demonstrates sharding for a column-parallel dense layer: Y = relu(X @ W)
func shardedNNLayerExample() {
    print("=== Example 3: Sharded Neural Network Layer ===\n")

    // 8 TPU-style devices in a 2x4 grid
    let mesh = JSDYMesh.tpuMesh(name: "tpu_mesh", x: 2, y: 4)
    mesh.printInfo()
    print()

    // Define the layer: Y = relu(X @ W)
    let batchSize = 64
    let inputDim = 512
    let hiddenDim = 2048

    // Shardings for column-parallel dense layer:
    // - Input X: shard batch on x-axis
    // - Weight W: shard hidden dimension on y-axis (column parallel)
    // - Output: shard both dimensions

    let xSharding = JSDYSharding(mesh: mesh, axes: ["x", nil])
    let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "y"])
    let outSharding = JSDYSharding(mesh: mesh, axes: ["x", "y"])

    print("Layer shardings:")
    print("  Input X [\(batchSize)x\(inputDim)]:   \(xSharding)")
    print("  Weight W [\(inputDim)x\(hiddenDim)]: \(wSharding)")
    print("  Output [\(batchSize)x\(hiddenDim)]:   \(outSharding)")
    print()

    // Create context
    let ctx = JShardedTracingContext(mesh: mesh)

    // Create inputs
    let x = ctx.input(shape: JTensorShape([batchSize, inputDim]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: JTensorShape([inputDim, hiddenDim]), dtype: .float32, sharding: wSharding)

    // Trace the layer: Y = relu(X @ W)
    let matmul = x.matmul(w)
    let output = matmul.relu()

    ctx.output(output, sharding: outSharding)

    // Build module
    let mlir = ctx.buildShardedModule(name: "dense_relu_layer")
    print("Generated sharded MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 4: Data Parallel Compilation Helper

/// Demonstrates the convenience function for data-parallel compilation.
func dataParallelHelperExample() {
    print("=== Example 4: Data Parallel Compilation Helper ===\n")

    // This example uses the jitCompileDataParallel helper
    // which automatically sets up data-parallel sharding
    print("Using jitCompileDataParallel helper...")
    print("  numDevices: 4")
    print("  inputShapes: [[16, 64], [64, 32]]")
    print()

    // Note: This would create a compiled function if PJRT is available
    // For now, we just demonstrate the sharded context setup

    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
    let inputShapes = [[16, 64], [64, 32]]

    let inputSpecs = inputShapes.map { shape in
        (
            shape: shape,
            dtype: JDType.float32,
            sharding: JSDYSharding.dataParallel(mesh: mesh, rank: shape.count, batchAxis: "batch")
        )
    }

    print("Generated input specs:")
    for (i, spec) in inputSpecs.enumerated() {
        print("  Input \(i): shape=\(spec.shape), sharding=\(spec.sharding)")
    }
    print()
}

// MARK: - Example 5: Bridge to JDeviceMesh

/// Demonstrates bridging from the existing JDeviceMesh API.
func bridgeExample() {
    print("=== Example 5: Bridge from JDeviceMesh ===\n")

    // Create using the existing JDeviceMesh API
    let jMesh = JDeviceMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
    print("JDeviceMesh: \(jMesh)")

    // Convert to SDY mesh
    let sdyMesh = jMesh.toSDYMesh()
    print("JSDYMesh MLIR: \(sdyMesh.mlirText)")
    print()

    // Create sharding spec using existing API
    let jSharding = JShardingSpec.dataParallel(mesh: jMesh, rank: 2)
    print("JShardingSpec: \(jSharding)")

    // Convert to SDY sharding
    let sdySharding = jSharding.toSDYSharding()
    print("JSDYSharding: \(sdySharding)")
    print()
}

// MARK: - Main

print("SwiftIR Jupyter Sharding Examples")
print("==================================\n")

dataParallelExample()
hybridParallelExample()
shardedNNLayerExample()
dataParallelHelperExample()
bridgeExample()

print("All examples completed!")
print()
print("Summary:")
print("  - JSDYMesh: Define device mesh topologies")
print("  - JSDYSharding: Specify tensor partitioning")
print("  - JShardedTracingContext: Trace with sharding annotations")
print("  - jitCompileSharded: Compile sharded functions")
print("  - jitCompileDataParallel: Quick data-parallel setup")
print()
print("Next steps:")
print("  1. Run SDY propagation via sdy_opt")
print("  2. Execute on multiple devices with PJRT")
print("  3. Combine with gradient synchronization for distributed training")
