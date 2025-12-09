/// JupyterShardedAutoDiff Tests for SwiftIRJupyter Module
/// Tests sharding-aware automatic differentiation in pure Swift

import XCTest
@testable import SwiftIRJupyter

// MARK: - JShardingAnalyzer Tests

final class JShardingAnalyzerTests: XCTestCase {

    func testReplicatedParameterNeedsAllReduce() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = JShardingAnalyzer(mesh: mesh)

        let sharding = JSDYSharding.replicated(mesh: mesh, rank: 2)
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Replicated weights need all-reduce mean
        if case .allReduceMean(let axes) = pattern {
            XCTAssertEqual(axes, ["batch"])
        } else {
            XCTFail("Expected allReduceMean pattern")
        }
    }

    func testDataParallelNeedsNoSync() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = JShardingAnalyzer(mesh: mesh)

        let sharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Data parallel sharding has no non-sharded axes, so no sync needed
        XCTAssertEqual(pattern, .none)
    }

    func testHybridParallelismPattern() {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let analyzer = JShardingAnalyzer(mesh: mesh)

        // Weight sharded on model axis only
        let sharding = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Model parallel weights need all-reduce on data axis
        if case .allReduceSum(let axes) = pattern {
            XCTAssertEqual(axes, ["data"])
        } else {
            XCTFail("Expected allReduceSum pattern for model parallel weights")
        }
    }

    func testActivationSyncPattern() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = JShardingAnalyzer(mesh: mesh)

        let sharding = JSDYSharding.replicated(mesh: mesh, rank: 2)
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .activation)

        // Activations use sum reduction
        if case .allReduceSum = pattern {
            // Expected
        } else {
            XCTFail("Expected allReduceSum pattern for activations")
        }
    }
}

// MARK: - JShardedGradient Tests

final class JShardedGradientTests: XCTestCase {

    func testUnsynchronizedGradient() {
        let tracer = JTracer(shape: JTensorShape([4, 8]), dtype: .float32)
        let sharding = JSDYSharding(mesh: "mesh", axes: ["x", nil])

        let gradient = JShardedGradient(gradient: tracer, sharding: sharding, isSynchronized: false)

        XCTAssertFalse(gradient.isSynchronized)
        XCTAssertNotNil(gradient.sharding)
    }

    func testSynchronizedGradient() {
        let tracer = JTracer(shape: JTensorShape([4, 8]), dtype: .float32)
        let sharding = JSDYSharding(mesh: "mesh", axes: ["x", nil])

        let gradient = JShardedGradient(gradient: tracer, sharding: sharding, isSynchronized: true)

        XCTAssertTrue(gradient.isSynchronized)
    }
}

// MARK: - JShardedDifferentiableTracer Tests

final class JShardedDifferentiableTracerTests: XCTestCase {

    func testShardedTracerCreation() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let tracer = JTracer(shape: JTensorShape([16, 64]), dtype: .float32)
        let sharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")

        let shardedTracer = JShardedDifferentiableTracer(
            tracer: tracer,
            sharding: sharding,
            mesh: mesh,
            parameterType: .weight
        )

        XCTAssertEqual(shardedTracer.shape.dimensions, [16, 64])
        XCTAssertEqual(shardedTracer.dtype, .float32)
        XCTAssertNotNil(shardedTracer.sharding)
    }

    func testAutomaticPatternInference() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let tracer = JTracer(shape: JTensorShape([16, 64]), dtype: .float32)
        let sharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

        let shardedTracer = JShardedDifferentiableTracer(
            tracer: tracer,
            sharding: sharding,
            mesh: mesh,
            parameterType: .weight
        )

        // Replicated weights should get allReduceMean pattern
        if case .allReduceMean = shardedTracer.gradientSyncPattern {
            // Expected
        } else {
            XCTFail("Expected allReduceMean pattern for replicated weights")
        }
    }
}

// MARK: - JGradientSyncPattern Tests

final class JGradientSyncPatternTests: XCTestCase {

    func testPatternEquality() {
        let pattern1 = JGradientSyncPattern.allReduceSum(axes: ["x", "y"])
        let pattern2 = JGradientSyncPattern.allReduceSum(axes: ["x", "y"])
        let pattern3 = JGradientSyncPattern.allReduceMean(axes: ["x", "y"])

        XCTAssertEqual(pattern1, pattern2)
        XCTAssertNotEqual(pattern1, pattern3)
    }

    func testNonePattern() {
        let pattern = JGradientSyncPattern.none
        XCTAssertEqual(pattern, .none)
    }
}

// MARK: - JShardedDifferentiableContext Tests

final class JShardedDifferentiableContextTests: XCTestCase {

    func testContextCreation() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        XCTAssertEqual(ctx.mesh.name, "mesh")
        XCTAssertEqual(ctx.mesh.deviceCount, 4)
    }

    func testInputCreation() {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
        let input = ctx.input(
            shape: JTensorShape([16, 64]),
            dtype: .float32,
            sharding: xSharding,
            parameterType: .activation
        )

        XCTAssertEqual(input.shape.dimensions, [16, 64])
        XCTAssertNotNil(input.sharding)
    }

    func testMLIRGeneration() {
        let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        let sharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let x = ctx.input(shape: JTensorShape([16, 64]), dtype: .float32, sharding: sharding)
        let w = ctx.input(shape: JTensorShape([64, 32]), dtype: .float32, sharding: nil)

        let y = x.tracer.matmul(w.tracer)
        ctx.output(y, sharding: sharding)

        let mlir = ctx.buildShardedModule(name: "test_module")

        // Check module structure
        XCTAssertTrue(mlir.contains("module @test_module"))
        XCTAssertTrue(mlir.contains("sdy.mesh @data_parallel"))

        // Check forward function
        XCTAssertTrue(mlir.contains("func.func @forward"))

        // Check backward function
        XCTAssertTrue(mlir.contains("func.func @backward"))
        XCTAssertTrue(mlir.contains("Gradient synchronization operations"))
    }
}

// MARK: - JShardedOptimizer Tests

final class JShardedOptimizerTests: XCTestCase {

    func testOptimizerCreation() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let optimizer = JShardedOptimizer(mesh: mesh, learningRate: 0.01)

        XCTAssertEqual(optimizer.learningRate, 0.01)
        XCTAssertEqual(optimizer.mesh.name, "mesh")
    }

    func testLearningRateUpdate() {
        let mesh = JSDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let optimizer = JShardedOptimizer(mesh: mesh, learningRate: 0.01)

        optimizer.learningRate = 0.001
        XCTAssertEqual(optimizer.learningRate, 0.001)
    }
}

// MARK: - JShardedTrainingStep Tests

final class JShardedTrainingStepTests: XCTestCase {

    func testTrainingStepCreation() {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let shardings = [
            JSDYSharding.dataParallel(mesh: mesh, rank: 2),
            JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1)
        ]

        let step = JShardedTrainingStep(mesh: mesh, parameterShardings: shardings)

        XCTAssertEqual(step.mesh.deviceCount, 8)
        XCTAssertEqual(step.parameterShardings.count, 2)
    }
}

// MARK: - JShardedVJPRules Tests

final class JShardedVJPRulesTests: XCTestCase {

    func testMatmulVJPPattern() {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)

        // X is data-parallel (batch sharded)
        let x = JTracer(shape: JTensorShape([32, 128]), dtype: .float32)
        let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
        let shardedX = JShardedDifferentiableTracer(tracer: x, sharding: xSharding, mesh: mesh, parameterType: .activation)

        // W is model-parallel (output sharded)
        let w = JTracer(shape: JTensorShape([128, 512]), dtype: .float32)
        let wSharding = JSDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")
        let shardedW = JShardedDifferentiableTracer(tracer: w, sharding: wSharding, mesh: mesh, parameterType: .weight)

        // Upstream gradient
        let upstream = JTracer(shape: JTensorShape([32, 512]), dtype: .float32)

        // Compute VJP
        let (gradX, gradW) = JShardedVJPRules.matmulVJP(
            x: shardedX,
            w: shardedW,
            upstream: upstream,
            mesh: mesh
        )

        // Both gradients should be synchronized
        XCTAssertTrue(gradX.isSynchronized)
        XCTAssertTrue(gradW.isSynchronized)
    }
}

// MARK: - Integration Tests

final class JupyterShardedAutoDiffIntegrationTests: XCTestCase {

    func testDataParallelMLP() {
        let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        // Data-parallel input sharding
        let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        // Replicated weights
        let wSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32,
                          sharding: xSharding, parameterType: .activation)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32,
                           sharding: wSharding, parameterType: .weight)
        let w2 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32,
                           sharding: wSharding, parameterType: .weight)

        let h = x.tracer.matmul(w1.tracer).relu()
        let y = h.matmul(w2.tracer)
        let loss = y.sum()

        ctx.output(loss)

        let mlir = ctx.buildShardedModule(name: "data_parallel_mlp")

        XCTAssertTrue(mlir.contains("sdy.mesh @data_parallel"))
        XCTAssertTrue(mlir.contains("func.func @forward"))
        XCTAssertTrue(mlir.contains("func.func @backward"))
    }

    func testHybridParallelTransformer() {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        // Data-parallel on batch dimension
        let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])

        // Column-parallel for QKV projection (shard output)
        let qkvSharding = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")

        // Row-parallel for output projection (shard input)
        let outSharding = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 0, modelAxis: "model")

        let x = ctx.input(shape: JTensorShape([32, 512]), dtype: .float32,
                          sharding: xSharding, parameterType: .activation)
        let wQKV = ctx.input(shape: JTensorShape([512, 1536]), dtype: .float32,
                             sharding: qkvSharding, parameterType: .weight)
        let wOut = ctx.input(shape: JTensorShape([512, 512]), dtype: .float32,
                             sharding: outSharding, parameterType: .weight)

        let qkv = x.tracer.matmul(wQKV.tracer)
        let attnOut = qkv.relu()  // Simplified attention
        let y = attnOut.matmul(wOut.tracer)

        ctx.output(y)

        let mlir = ctx.buildShardedModule(name: "hybrid_transformer")

        XCTAssertTrue(mlir.contains("sdy.mesh @mesh"))
        XCTAssertTrue(mlir.contains("\"data\"=2"))
        XCTAssertTrue(mlir.contains("\"model\"=4"))
    }

    func testEndToEndTrainingLoop() {
        let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let optimizer = JShardedOptimizer(mesh: mesh, learningRate: 0.01)

        // Simple linear layer
        let x = JTracer(shape: JTensorShape([32, 64]), dtype: .float32)
        let w = JTracer(shape: JTensorShape([64, 32]), dtype: .float32)

        let sharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let wSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

        let ctx = JShardedDifferentiableContext(mesh: mesh)
        let shardedX = ctx.input(shape: x.shape, dtype: x.dtype, sharding: sharding, parameterType: .activation)
        let shardedW = ctx.input(shape: w.shape, dtype: w.dtype, sharding: wSharding, parameterType: .weight)

        // Forward
        let y = shardedX.tracer.matmul(shardedW.tracer)
        let loss = y.sum()

        ctx.output(loss)

        // Generate MLIR with gradient sync
        let mlir = ctx.buildShardedModule(name: "training_step")

        XCTAssertTrue(mlir.contains("module @training_step"))
        XCTAssertTrue(mlir.contains("func.func @backward"))
    }
}
