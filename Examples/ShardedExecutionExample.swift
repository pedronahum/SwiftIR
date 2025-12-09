/// Sharded Execution Example for SwiftIR
///
/// This example demonstrates PJRT distributed execution with SDY sharding annotations.
/// It shows how to:
/// 1. Create a sharded execution context
/// 2. Compile functions with sharding specifications
/// 3. Execute on distributed buffers
/// 4. Use collective operations (all-reduce, all-gather)

import SwiftIR

// MARK: - Example 1: Basic Sharded Execution

/// Demonstrates basic sharded execution with data parallelism.
func basicShardedExecution() {
    print("=== Example 1: Basic Sharded Execution ===\n")

    // Create a 4-device linear mesh for data parallelism
    let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print()

    // Create sharded tracing context
    let ctx = ShardedTracingContext(mesh: mesh)

    // Define sharded inputs for matmul Y = X @ W:
    // - X[16, 64]: batch dimension sharded across "batch" axis
    // - W[64, 32]: fully replicated (same weights on all devices)
    let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
    let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

    print("Sharding configuration:")
    print("  X sharding: \(xSharding.mlirAttributeText)")
    print("  W sharding: \(wSharding.mlirAttributeText)")
    print()

    // Create symbolic inputs with shardings
    let x = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([64, 32]), dtype: .float32, sharding: wSharding)

    // Trace the computation - use .tracer to access underlying Tracer for operations
    let y = x.tracer.matmul(w.tracer)

    // Set output with data-parallel sharding
    ctx.output(y, sharding: xSharding)

    // Build sharded MLIR module
    let mlir = ctx.buildShardedModule(name: "data_parallel_matmul")

    print("Generated MLIR with SDY sharding:")
    print("---")
    print(mlir)
    print("---")

    // Show expected data distribution:
    print("\nExpected data distribution:")
    print("  Device 0: X[0:4, :] @ W[:, :] -> Y[0:4, :]")
    print("  Device 1: X[4:8, :] @ W[:, :] -> Y[4:8, :]")
    print("  Device 2: X[8:12, :] @ W[:, :] -> Y[8:12, :]")
    print("  Device 3: X[12:16, :] @ W[:, :] -> Y[12:16, :]")
    print()
}

// MARK: - Example 2: Hybrid 2D Parallelism

/// Demonstrates hybrid data + model parallelism with a 2D mesh.
func hybridParallelExecution() {
    print("=== Example 2: Hybrid 2D Parallelism ===\n")

    // Create a 2x4 mesh: 2 devices for data parallelism, 4 for model parallelism
    // Total: 8 devices
    let mesh = SDYMesh.grid(
        name: "hybrid_mesh",
        dataParallel: 2,
        modelParallel: 4
    )
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print()

    // For a matmul Y = X @ W:
    // - X[32, 128]: shard batch on "data" axis (split into 2 parts)
    // - W[128, 256]: shard features on "model" axis (split into 4 parts)
    // - Y[32, 256]: shard both dimensions

    let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
    let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])
    let ySharding = SDYSharding(mesh: mesh, axes: ["data", "model"])

    print("2D Sharding strategy:")
    print("  X[32, 128]: batch on 'data' axis -> [16, 128] per data-parallel group")
    print("  W[128, 256]: features on 'model' axis -> [128, 64] per model-parallel group")
    print("  Y[32, 256]: both dims sharded -> [16, 64] per device")
    print()

    // Create context and trace
    let ctx = ShardedTracingContext(mesh: mesh)

    let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([128, 256]), dtype: .float32, sharding: wSharding)

    // Use .tracer to access the underlying Tracer for operations
    let y = x.tracer.matmul(w.tracer)
    ctx.output(y, sharding: ySharding)

    let mlir = ctx.buildShardedModule(name: "hybrid_parallel_matmul")

    print("Generated MLIR:")
    print("---")
    print(mlir)
    print("---\n")
}

// MARK: - Example 3: Gradient Synchronization Pattern

/// Demonstrates the gradient all-reduce pattern for distributed training.
func gradientSynchronizationPattern() {
    print("=== Example 3: Gradient Synchronization Pattern ===\n")

    let mesh = SDYMesh.linear(name: "data_parallel", axis: "dp", size: 4)
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print()

    print("Distributed training pattern:")
    print("1. Forward pass: Each device computes on its batch shard")
    print("2. Backward pass: Each device computes local gradients")
    print("3. All-reduce: Gradients are averaged across all devices")
    print("4. Update: All devices apply the same averaged gradient")
    print()

    // Trace a forward pass with all-reduce for gradient sync
    let ctx = ShardedTracingContext(mesh: mesh)

    // Data-parallel inputs
    let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "dp")
    let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

    let x = ctx.input(shape: TensorShape([64, 128]), dtype: .float32, sharding: xSharding)
    let w = ctx.input(shape: TensorShape([128, 64]), dtype: .float32, sharding: wSharding)

    // Forward: Y = X @ W
    let y = x.tracer.matmul(w.tracer)

    // Simulate gradient computation (in real training, this would be autodiff)
    // For demonstration, we show the pattern
    // Note: allReduce is on Tracer, uses ReductionType enum
    let grad = y.allReduce(reduction: .mean)  // Average gradients across replicas

    ctx.output(grad, sharding: wSharding)  // Gradients are replicated after all-reduce

    let mlir = ctx.buildShardedModule(name: "training_step")

    print("Training step MLIR (forward + gradient sync):")
    print("---")
    print(mlir)
    print("---")

    print("\nCollective operations used:")
    print("  all_reduce(mean): Averages gradients across all 4 devices")
    print("  Result: Same gradient value on all devices for synchronized update")
    print()
}

// MARK: - Example 4: Tensor Parallel MLP

/// Demonstrates tensor parallelism for a multi-layer perceptron.
func tensorParallelMLP() {
    print("=== Example 4: Tensor Parallel MLP ===\n")

    // For tensor parallelism, we shard weight matrices across the model axis
    let mesh = SDYMesh.linear(name: "tensor_parallel", axis: "tp", size: 4)
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print()

    let batchSize = 32
    let inputDim = 512
    let hiddenDim = 2048  // Will be split across 4 devices -> 512 per device
    let outputDim = 512

    print("MLP architecture with tensor parallelism:")
    print("  Input: [\(batchSize), \(inputDim)]")
    print("  Hidden: [\(batchSize), \(hiddenDim)] (split: [\(batchSize), \(hiddenDim/4)] per device)")
    print("  Output: [\(batchSize), \(outputDim)]")
    print()

    let ctx = ShardedTracingContext(mesh: mesh)

    // Shardings:
    // - Input X: replicated (all devices see full input)
    // - W1 (column parallel): shard output dimension
    // - W2 (row parallel): shard input dimension
    // - Output: replicated (all-reduce at end)

    let xSharding = SDYSharding.replicated(mesh: mesh, rank: 2)
    let w1Sharding = SDYSharding(mesh: mesh, axes: [nil, "tp"])  // Column parallel
    let w2Sharding = SDYSharding(mesh: mesh, axes: ["tp", nil])  // Row parallel
    let outSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

    print("Sharding strategy (Megatron-style):")
    print("  X[32, 512]: replicated")
    print("  W1[512, 2048]: column-parallel (shard dim 1 on 'tp')")
    print("  W2[2048, 512]: row-parallel (shard dim 0 on 'tp')")
    print("  Output: replicated (requires all-reduce)")
    print()

    // Create inputs
    let x = ctx.input(shape: TensorShape([batchSize, inputDim]), dtype: .float32, sharding: xSharding)
    let w1 = ctx.input(shape: TensorShape([inputDim, hiddenDim]), dtype: .float32, sharding: w1Sharding)
    let w2 = ctx.input(shape: TensorShape([hiddenDim, outputDim]), dtype: .float32, sharding: w2Sharding)

    // Layer 1: H = relu(X @ W1) - column parallel
    let h1 = x.tracer.matmul(w1.tracer)
    let h1_activated = h1.relu()

    // Layer 2: Y = H @ W2 - row parallel (needs all-reduce)
    let y = h1_activated.matmul(w2.tracer)

    // All-reduce to combine partial sums from row-parallel matmul
    let output = y.allReduce(reduction: .sum)

    ctx.output(output, sharding: outSharding)

    let mlir = ctx.buildShardedModule(name: "tensor_parallel_mlp")

    print("Generated MLIR:")
    print("---")
    print(mlir)
    print("---")

    print("\nCommunication pattern:")
    print("  No communication for X @ W1 (column parallel)")
    print("  All-reduce after H @ W2 (row parallel combines partial sums)")
    print()
}

// MARK: - Example 5: Sharding Analysis

/// Demonstrates the ShardingAnalyzer for gradient sync pattern inference.
func shardingAnalysisExample() {
    print("=== Example 5: Sharding Analysis ===\n")

    let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
    let analyzer = ShardingAnalyzer(mesh: mesh)

    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print()

    print("Gradient sync pattern analysis:")
    print()

    // Analyze different sharding patterns
    let patterns: [(String, SDYSharding, ShardingAnalyzer.ParameterType)] = [
        ("Replicated weight", SDYSharding.replicated(mesh: mesh, rank: 2), .weight),
        ("Data-parallel activation", SDYSharding.dataParallel(mesh: mesh, rank: 2), .activation),
        ("Model-parallel weight", SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model"), .weight),
        ("Fully sharded", SDYSharding(mesh: mesh, axes: ["data", "model"]), .weight)
    ]

    for (name, sharding, paramType) in patterns {
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: paramType)
        print("  \(name):")
        print("    Sharding: \(sharding.mlirAttributeText)")
        print("    Sync: \(describePattern(pattern))")
        print()
    }
}

/// Helper to describe sync pattern
func describePattern(_ pattern: GradientSyncPattern) -> String {
    switch pattern {
    case .none:
        return "None (already synchronized by sharding)"
    case .allReduceSum(let axes):
        return "All-Reduce Sum on axes: [\(axes.joined(separator: ", "))]"
    case .allReduceMean(let axes):
        return "All-Reduce Mean on axes: [\(axes.joined(separator: ", "))]"
    case .allGather(let axis, let dim):
        return "All-Gather on axis '\(axis)', dimension \(dim)"
    case .reduceScatter(let axis, let dim):
        return "Reduce-Scatter on axis '\(axis)', dimension \(dim)"
    }
}

// MARK: - Main

print("SwiftIR Sharded Execution Examples")
print("===================================\n")

basicShardedExecution()
hybridParallelExecution()
gradientSynchronizationPattern()
tensorParallelMLP()
shardingAnalysisExample()

print("All examples completed!")
print()
print("Summary of Sharded Execution Components:")
print("  - SDYMesh: Device mesh topology (linear, grid, TPU)")
print("  - SDYSharding: Tensor partitioning specification")
print("  - ShardedTracingContext: Trace with sharding annotations")
print("  - ShardingAnalyzer: Gradient sync pattern inference")
print("  - ShardedDifferentiableContext: Autodiff with automatic sync")
print("  - ShardedOptimizer: Optimizer for sharded parameters")
print()
print("Parallelism patterns demonstrated:")
print("  1. Data parallelism: Shard batch dimension")
print("  2. Model parallelism: Shard feature dimension")
print("  3. Hybrid 2D: Combine data and model parallelism")
print("  4. Tensor parallelism: Column-parallel and row-parallel layers")
print()
print("Next steps:")
print("  1. Run sdy_opt to propagate sharding annotations")
print("  2. Lower to XLA with GSPMD partitioner")
print("  3. Execute on actual multi-device hardware (TPU, multi-GPU)")
