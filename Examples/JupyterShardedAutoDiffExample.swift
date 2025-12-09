/// JupyterShardedAutoDiffExample - Demonstrates Sharding-Aware Automatic Differentiation
/// For SwiftIRJupyter module (pure Swift, works in Colab)
///
/// This example shows how to:
/// 1. Set up sharding for distributed training
/// 2. Use automatic gradient synchronization
/// 3. Understand when all-reduce/all-gather operations are needed

import SwiftIRJupyter

// MARK: - Data Parallel Training Example

/// Simple data-parallel MLP training with automatic gradient sync
func dataParallelMLPExample() {
    print("=== Data Parallel MLP Example ===\n")

    // 1. Define device mesh (4 devices for data parallelism)
    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)\n")

    // 2. Create sharding specifications
    // - Input: shard batch dimension across devices
    let inputSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
    print("Input sharding: \(inputSharding.mlirAttributeText)")

    // - Weights: replicated across all devices
    let weightSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)
    print("Weight sharding: \(weightSharding.mlirAttributeText)\n")

    // 3. Set up sharded differentiable context
    let ctx = JShardedDifferentiableContext(mesh: mesh)

    // Create sharded inputs
    let x = ctx.input(
        shape: JTensorShape([32, 128]),  // Batch of 32 per device = 128 total
        dtype: .float32,
        sharding: inputSharding,
        parameterType: .activation
    )

    let w1 = ctx.input(
        shape: JTensorShape([128, 256]),
        dtype: .float32,
        sharding: weightSharding,
        parameterType: .weight
    )

    let w2 = ctx.input(
        shape: JTensorShape([256, 64]),
        dtype: .float32,
        sharding: weightSharding,
        parameterType: .weight
    )

    // 4. Forward pass
    let h = x.tracer.matmul(w1.tracer).relu()
    let y = h.matmul(w2.tracer)
    let loss = y.sum()

    ctx.output(loss)

    // 5. Generate MLIR with sharding and gradient sync
    let mlir = ctx.buildShardedModule(name: "data_parallel_mlp")

    print("Generated MLIR with gradient synchronization:")
    print("----------------------------------------")
    print(mlir)
    print("----------------------------------------\n")

    // 6. Explain gradient sync patterns
    print("Gradient Synchronization Analysis:")
    print("- Input gradients (batch-sharded): No sync needed - each device has its own batch")
    print("- Weight gradients (replicated): All-reduce mean across devices")
    print("  * Each device computes gradients on its batch shard")
    print("  * Gradients are averaged across all 4 devices")
    print("  * This gives the same result as if we computed on the full batch")
}

// MARK: - Hybrid Parallelism Example

/// Hybrid data + model parallelism for transformer layers
func hybridParallelTransformerExample() {
    print("\n=== Hybrid Parallel Transformer Example ===\n")

    // 1. Define 2D mesh: 2 data-parallel x 4 model-parallel = 8 devices
    let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
    print("Device Mesh: \(mesh.mlirText)")
    print("Total devices: \(mesh.deviceCount)")
    print("Data parallel axis: 2 devices")
    print("Model parallel axis: 4 devices\n")

    // 2. Sharding strategies for different tensors
    // - Input: shard on data axis only
    let inputSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])

    // - QKV weights: column-parallel (shard output dim)
    let columnParallel = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")

    // - Output projection: row-parallel (shard input dim)
    let rowParallel = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 0, modelAxis: "model")

    print("Sharding Strategies:")
    print("- Input: \(inputSharding.mlirAttributeText)")
    print("- Column-parallel (QKV): \(columnParallel.mlirAttributeText)")
    print("- Row-parallel (Output): \(rowParallel.mlirAttributeText)\n")

    // 3. Set up context
    let ctx = JShardedDifferentiableContext(mesh: mesh)

    let x = ctx.input(
        shape: JTensorShape([32, 512]),
        dtype: .float32,
        sharding: inputSharding,
        parameterType: .activation
    )

    let wQKV = ctx.input(
        shape: JTensorShape([512, 1536]),  // Q, K, V concatenated
        dtype: .float32,
        sharding: columnParallel,
        parameterType: .weight
    )

    let wO = ctx.input(
        shape: JTensorShape([512, 512]),
        dtype: .float32,
        sharding: rowParallel,
        parameterType: .weight
    )

    // 4. Transformer attention layer (simplified)
    let qkv = x.tracer.matmul(wQKV.tracer)
    let attnOut = qkv.relu()  // Simplified attention
    let y = attnOut.matmul(wO.tracer)

    ctx.output(y)

    // 5. Generate MLIR
    let mlir = ctx.buildShardedModule(name: "hybrid_transformer")

    print("Generated MLIR:")
    print("----------------------------------------")
    print(mlir)
    print("----------------------------------------\n")

    // 6. Explain gradient sync patterns
    print("Gradient Synchronization Analysis:")
    print("\n1. QKV weights (column-parallel, sharded on model axis):")
    print("   - Forward: each device has 1/4 of output columns")
    print("   - Backward: gradients need all-reduce on DATA axis only")
    print("   - No sync needed on model axis (already partitioned)")

    print("\n2. Output weights (row-parallel, sharded on model axis):")
    print("   - Forward: each device has 1/4 of input rows")
    print("   - Backward: gradients need all-reduce on DATA axis")

    print("\n3. Activation gradients (data-parallel):")
    print("   - Forward: each device has 1/2 of batch")
    print("   - Backward: no sync needed - each device backprops its batch")
}

// MARK: - Training Loop Example

/// Complete training loop with sharded gradients
func shardedTrainingLoopExample() {
    print("\n=== Sharded Training Loop Example ===\n")

    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)

    // Sharding specs
    let inputSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
    let weightSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

    // Create optimizer
    let optimizer = JShardedOptimizer(mesh: mesh, learningRate: 0.01)
    print("Optimizer learning rate: \(optimizer.learningRate)")

    // Simulate training step
    print("\nSimulated Training Step:")
    print("1. Forward pass: compute loss")
    print("2. Backward pass: compute gradients (local to each device)")
    print("3. Gradient sync: all-reduce mean (for replicated weights)")
    print("4. Optimizer step: apply synchronized gradients")

    // Create training step helper
    let step = JShardedTrainingStep(mesh: mesh, parameterShardings: [inputSharding, weightSharding])
    print("\nTraining step configured for \(step.mesh.deviceCount) devices")
}

// MARK: - Gradient Sync Pattern Demonstration

/// Shows different gradient sync patterns based on sharding
func gradientSyncPatternsDemo() {
    print("\n=== Gradient Sync Patterns Demo ===\n")

    let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
    let analyzer = JShardingAnalyzer(mesh: mesh)

    // Case 1: Fully replicated
    let replicated = JSDYSharding.replicated(mesh: mesh, rank: 2)
    let replicatedPattern = analyzer.gradientSyncPattern(for: replicated, parameterType: .weight)
    print("1. Fully Replicated Weights:")
    print("   Sharding: \(replicated.mlirAttributeText)")
    print("   Gradient Sync: \(patternDescription(replicatedPattern))")

    // Case 2: Data parallel only
    let dataParallel = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
    let dataPattern = analyzer.gradientSyncPattern(for: dataParallel, parameterType: .weight)
    print("\n2. Data Parallel (batch sharded):")
    print("   Sharding: \(dataParallel.mlirAttributeText)")
    print("   Gradient Sync: \(patternDescription(dataPattern))")

    // Case 3: Model parallel only
    let modelParallel = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
    let modelPattern = analyzer.gradientSyncPattern(for: modelParallel, parameterType: .weight)
    print("\n3. Model Parallel (feature sharded):")
    print("   Sharding: \(modelParallel.mlirAttributeText)")
    print("   Gradient Sync: \(patternDescription(modelPattern))")

    // Case 4: Hybrid parallel
    let hybrid = JSDYSharding(mesh: mesh, axes: ["data", "model"])
    let hybridPattern = analyzer.gradientSyncPattern(for: hybrid, parameterType: .weight)
    print("\n4. Hybrid Parallel (both axes sharded):")
    print("   Sharding: \(hybrid.mlirAttributeText)")
    print("   Gradient Sync: \(patternDescription(hybridPattern))")
}

/// Helper to describe sync pattern
func patternDescription(_ pattern: JGradientSyncPattern) -> String {
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

// MARK: - VJP Rules Example

/// Demonstrates sharding-aware VJP rules
func vjpRulesExample() {
    print("\n=== Sharding-Aware VJP Rules Example ===\n")

    let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)

    // Create sharded tracers for matmul
    let x = JTracer(shape: JTensorShape([32, 128]), dtype: .float32)
    let w = JTracer(shape: JTensorShape([128, 512]), dtype: .float32)
    let upstream = JTracer(shape: JTensorShape([32, 512]), dtype: .float32)

    // X is data-parallel sharded
    let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
    let shardedX = JShardedDifferentiableTracer(
        tracer: x,
        sharding: xSharding,
        mesh: mesh,
        parameterType: .activation
    )

    // W is model-parallel (column) sharded
    let wSharding = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
    let shardedW = JShardedDifferentiableTracer(
        tracer: w,
        sharding: wSharding,
        mesh: mesh,
        parameterType: .weight
    )

    // Compute sharding-aware VJP
    let (gradX, gradW) = JShardedVJPRules.matmulVJP(
        x: shardedX,
        w: shardedW,
        upstream: upstream,
        mesh: mesh
    )

    print("MatMul VJP Analysis:")
    print("- Y = X @ W")
    print("- X shape: \(x.shape.dimensions) (data-parallel on batch)")
    print("- W shape: \(w.shape.dimensions) (model-parallel on output dim)")
    print("")
    print("Gradient computation:")
    print("- dX = dY @ W^T (synchronized: \(gradX.isSynchronized))")
    print("- dW = X^T @ dY (synchronized: \(gradW.isSynchronized))")
    print("")
    print("The gradients are automatically synchronized:")
    print("- dW needs all-reduce on DATA axis (partial sums from each batch)")
    print("- dX needs all-gather if W was sharded (to get full W for transpose)")
}

// MARK: - Main

print("╔══════════════════════════════════════════════════════════════╗")
print("║   SwiftIRJupyter Sharded Automatic Differentiation Demo      ║")
print("║                                                              ║")
print("║   Demonstrates automatic gradient synchronization based on  ║")
print("║   tensor sharding patterns for distributed training.        ║")
print("║                                                              ║")
print("║   Pure Swift - works in Jupyter/Colab without C++ deps.     ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

dataParallelMLPExample()
hybridParallelTransformerExample()
shardedTrainingLoopExample()
gradientSyncPatternsDemo()
vjpRulesExample()

print("\n=== Summary ===")
print("SwiftIRJupyter automatically determines gradient synchronization based on sharding:")
print("• Replicated weights → All-reduce mean across all devices")
print("• Data-parallel sharded → No sync (each device has independent batch)")
print("• Model-parallel sharded → All-reduce on non-sharded axes only")
print("• Hybrid parallel → Sync on axes where gradients need aggregation")
print("")
print("Key Features:")
print("• Pure Swift implementation (no C++ dependencies)")
print("• Works in Jupyter/Colab environments")
print("• Automatic sync pattern inference from sharding specs")
print("• Support for data, model, and hybrid parallelism")
