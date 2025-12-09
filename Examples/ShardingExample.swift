/// Sharding Example for SwiftIR
///
/// This example demonstrates how to use the SwiftIRSharding module to:
/// 1. Define device meshes
/// 2. Create tensor shardings
/// 3. Generate sharded StableHLO modules
/// 4. Prepare modules for SDY propagation

import SwiftIRShardingLite

// MARK: - Example 1: Basic Data Parallelism

/// Demonstrates data-parallel sharding for a simple matrix multiply.
func dataParallelMatmul() {
    print("=== Example 1: Data Parallel Matrix Multiply ===\n")

    // Create a sharding pipeline for text-based MLIR generation
    // For actual pass execution, use ShardingPipeline(context: mlirContext)
    let pipeline = ShardingPipeline()

    // Define a 4-device linear mesh for data parallelism
    let mesh = DeviceMesh.linear(name: "data_parallel", axisName: "batch", size: 4)
    pipeline.addMesh(mesh)

    print("Device mesh:")
    print("  \(mesh.mlirText)")
    print()

    // Define tensor shapes
    let batchSize = 16  // Will be split across 4 devices = 4 per device
    let inputDim = 64
    let outputDim = 32

    let inputType = "tensor<\(batchSize)x\(inputDim)xf32>"
    let weightType = "tensor<\(inputDim)x\(outputDim)xf32>"
    let outputType = "tensor<\(batchSize)x\(outputDim)xf32>"

    // Create shardings
    // Input: shard batch dimension on "batch" axis, replicate features
    let inputSharding = pipeline.dataParallelSharding(
        mesh: "data_parallel",
        batchAxis: "batch",
        rank: 2
    )

    // Weights: fully replicated (each device has a copy)
    let weightSharding = TensorSharding.replicated(meshName: "data_parallel", rank: 2)

    // Output: same sharding as input
    let outputSharding = inputSharding

    print("Tensor shardings:")
    print("  Input:  \(inputSharding.mlirAttributeText)")
    print("  Weight: \(weightSharding.mlirAttributeText)")
    print("  Output: \(outputSharding.mlirAttributeText)")
    print()

    // Generate the sharded module
    let body = """
        %result = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (\(inputType), \(weightType)) -> \(outputType)
        return %result : \(outputType)
        """

    let module = pipeline.generateShardedModule(
        funcName: "data_parallel_matmul",
        args: [
            ("%arg0", inputType, inputSharding),
            ("%arg1", weightType, weightSharding)
        ],
        results: [outputType],
        body: body
    )

    print("Generated MLIR module:")
    print("---")
    print(module)
    print("---\n")
}

// MARK: - Example 2: 2D Parallelism (Data + Model)

/// Demonstrates 2D parallelism combining data and model sharding.
func hybridParallelMatmul() {
    print("=== Example 2: Hybrid 2D Parallelism ===\n")

    let pipeline = ShardingPipeline()

    // Define a 2x4 mesh: 2 devices for data, 4 for model
    let mesh = DeviceMesh.grid(
        name: "hybrid_mesh",
        rows: 2, cols: 4,
        rowAxis: "data", colAxis: "model"
    )
    pipeline.addMesh(mesh)

    print("Device mesh (8 devices total):")
    print("  \(mesh.mlirText)")
    print("  Data axis: 2 devices")
    print("  Model axis: 4 devices")
    print()

    // For a matmul Y = X @ W:
    // - X[batch, in_features]: shard batch on "data"
    // - W[in_features, out_features]: shard out_features on "model"
    // - Y[batch, out_features]: shard batch on "data", out on "model"

    let batchSize = 32
    let inFeatures = 128
    let outFeatures = 256

    let xType = "tensor<\(batchSize)x\(inFeatures)xf32>"
    let wType = "tensor<\(inFeatures)x\(outFeatures)xf32>"
    let yType = "tensor<\(batchSize)x\(outFeatures)xf32>"

    // X: shard batch dimension on "data", replicate in_features
    let xSharding = TensorSharding(meshName: "hybrid_mesh", axisNames: ["data", nil])

    // W: replicate in_features, shard out_features on "model"
    let wSharding = TensorSharding(meshName: "hybrid_mesh", axisNames: [nil, "model"])

    // Y: shard both dimensions
    let ySharding = pipeline.matrixSharding(mesh: "hybrid_mesh", rowAxis: "data", colAxis: "model")

    print("Tensor shardings:")
    print("  X (input):  \(xSharding)")
    print("  W (weight): \(wSharding)")
    print("  Y (output): \(ySharding)")
    print()

    // For demonstration, show the sharded operation text
    let matmulOp = StableHLOSharding.dotGeneral(
        result: "%0",
        lhs: "%arg0",
        rhs: "%arg1",
        lhsType: xType,
        rhsType: wType,
        resultType: yType,
        contractingDims: (lhs: [1], rhs: [0]),
        sharding: ySharding
    )

    print("Sharded matmul operation:")
    print("  \(matmulOp)")
    print()
}

// MARK: - Example 3: Neural Network Layer with Sharding

/// Demonstrates sharding for a complete neural network layer.
func shardedNNLayer() {
    print("=== Example 3: Sharded Neural Network Layer ===\n")

    let pipeline = ShardingPipeline()

    // 8 TPU-style devices in a 2x4 grid
    let mesh = DeviceMesh.grid(name: "tpu_mesh", rows: 2, cols: 4)
    pipeline.addMesh(mesh)

    print("TPU-style mesh:")
    print("  \(mesh.mlirText)")
    print()

    // Define a dense layer: Y = relu(X @ W + b)
    let batchSize = 64
    let inputDim = 512
    let hiddenDim = 2048

    let xType = "tensor<\(batchSize)x\(inputDim)xf32>"
    let wType = "tensor<\(inputDim)x\(hiddenDim)xf32>"
    let bType = "tensor<\(hiddenDim)xf32>"
    let outType = "tensor<\(batchSize)x\(hiddenDim)xf32>"

    // Shardings for this layer:
    // - Input X: shard batch on x-axis
    // - Weight W: shard hidden dimension on y-axis (column parallel)
    // - Bias b: shard on y-axis (same as W columns)
    // - Output: shard both dimensions

    let xSharding = TensorSharding(meshName: "tpu_mesh", axisNames: ["x", nil])
    let wSharding = TensorSharding(meshName: "tpu_mesh", axisNames: [nil, "y"])
    let bSharding = TensorSharding(meshName: "tpu_mesh", axisNames: ["y"])
    _ = TensorSharding(meshName: "tpu_mesh", axisNames: ["x", "y"])  // outSharding for reference

    print("Layer shardings:")
    print("  Input X [\(batchSize)x\(inputDim)]:   batch on x")
    print("  Weight W [\(inputDim)x\(hiddenDim)]: hidden on y")
    print("  Bias b [\(hiddenDim)]:             on y")
    print("  Output [\(batchSize)x\(hiddenDim)]:   batch on x, hidden on y")
    print()

    // Generate the full layer as MLIR text
    let body = """
        // Matrix multiply: X @ W
        %matmul = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (\(xType), \(wType)) -> \(outType)
        // Broadcast bias to match output shape
        %bias_broadcast = stablehlo.broadcast_in_dim %arg2, dims = [1] : (\(bType)) -> \(outType)
        // Add bias
        %biased = stablehlo.add %matmul, %bias_broadcast : \(outType)
        // ReLU activation
        %zero = stablehlo.constant dense<0.0> : \(outType)
        %output = stablehlo.maximum %biased, %zero : \(outType)
        return %output : \(outType)
        """

    let module = pipeline.generateShardedModule(
        funcName: "dense_relu_layer",
        args: [
            ("%arg0", xType, xSharding),
            ("%arg1", wType, wSharding),
            ("%arg2", bType, bSharding)
        ],
        results: [outType],
        body: body
    )

    print("Generated sharded layer module:")
    print("---")
    print(module)
    print("---\n")
}

// MARK: - Example 4: Using Sharding Constraints

/// Demonstrates explicit sharding constraints for propagation hints.
func shardingConstraintsExample() {
    print("=== Example 4: Sharding Constraints ===\n")

    let pipeline = ShardingPipeline()
    pipeline.addMesh(DeviceMesh.linear(name: "mesh", axisName: "x", size: 8))

    let tensorType = "tensor<32x32xf32>"
    let sharding = TensorSharding(meshName: "mesh", axisNames: ["x", nil])

    // Create a sharding constraint operation
    let constraint = StableHLOSharding.shardingConstraint(
        result: "%1",
        input: "%0",
        type: tensorType,
        sharding: sharding
    )

    print("Sharding constraint:")
    print("  Input: unsharded tensor")
    print("  Constraint: shard dim 0 on 'x' axis")
    print()
    print("  MLIR: \(constraint)")
    print()

    // Show how this would be used in a function
    print("Usage in a function:")
    print("  func.func @apply_sharding(%arg0: \(tensorType)) -> \(tensorType) {")
    print("    \(constraint)")
    print("    return %1 : \(tensorType)")
    print("  }")
    print()
}

// MARK: - Example 5: Using SdyOptRunner

/// Demonstrates running actual sharding propagation via sdy_opt.
func sdyOptPropagationExample() {
    print("=== Example 5: SDY Opt Propagation ===\n")

    // Check if sdy_opt is available
    if ShardingPipeline.isSdyOptAvailable() {
        print("sdy_opt is available for propagation!")

        let pipeline = ShardingPipeline()
        pipeline.addMesh(DeviceMesh.linear(name: "mesh", axisName: "x", size: 4))

        let module = """
            func.func @matmul(%arg0: tensor<8x4xf32>, %arg1: tensor<4x8xf32>) -> tensor<8x8xf32> {
              %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
              return %0 : tensor<8x8xf32>
            }
            """

        print("Input module (with mesh definitions):")
        if let prepared = pipeline.runImportPipeline(moduleText: module) {
            print("---")
            print(prepared)
            print("---")
        }

        print()
        print("To run propagation, use:")
        print("  let result = pipeline.runPropagationWithSdyOpt(moduleText: module)")
        print()
    } else {
        print("sdy_opt not found - propagation requires sdy_opt binary")
        print("Build Shardy with: bazelisk build //shardy/tools:sdy_opt")
        print()
    }
}

// MARK: - Main

print("SwiftIR Sharding Examples")
print("=========================\n")

dataParallelMatmul()
hybridParallelMatmul()
shardedNNLayer()
shardingConstraintsExample()
sdyOptPropagationExample()

print("All examples completed successfully!")
print()
print("Summary:")
print("  - ShardingPipeline: Pure Swift MLIR text generation (no dependencies)")
print("  - SdyOptRunner: External process-based propagation (requires sdy_opt)")
print("  - C API: Direct SDY integration (requires libsdy_capi linking)")
print()
print("Next steps:")
print("  1. Use sdy_opt to run propagation on these modules")
print("  2. Integrate with XLA/PJRT for distributed execution")
print("  3. Connect to SwiftIRJupyter for interactive sharding design")
