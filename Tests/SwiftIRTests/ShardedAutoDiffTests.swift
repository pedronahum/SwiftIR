/// ShardedAutoDiff Tests for SwiftIR Main Module
/// Tests sharding-aware automatic differentiation

import XCTest
@testable import SwiftIR

// MARK: - ShardingAnalyzer Tests

final class ShardingAnalyzerTests: XCTestCase {

    func testReplicatedParameterNeedsAllReduce() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = ShardingAnalyzer(mesh: mesh)

        let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Replicated weights need all-reduce mean
        if case .allReduceMean(let axes) = pattern {
            XCTAssertEqual(axes, ["batch"])
        } else {
            XCTFail("Expected allReduceMean pattern")
        }
    }

    func testDataParallelNeedsAllReduce() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = ShardingAnalyzer(mesh: mesh)

        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Data parallel sharding has no non-sharded axes, so no sync needed
        XCTAssertEqual(pattern, .none)
    }

    func testHybridParallelismPattern() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let analyzer = ShardingAnalyzer(mesh: mesh)

        // Weight sharded on model axis only
        let sharding = SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .weight)

        // Model parallel weights need all-reduce on data axis
        if case .allReduceSum(let axes) = pattern {
            XCTAssertEqual(axes, ["data"])
        } else {
            XCTFail("Expected allReduceSum pattern for model parallel weights")
        }
    }

    func testActivationSyncPattern() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let analyzer = ShardingAnalyzer(mesh: mesh)

        let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
        let pattern = analyzer.gradientSyncPattern(for: sharding, parameterType: .activation)

        // Activations use sum reduction
        if case .allReduceSum = pattern {
            // Expected
        } else {
            XCTFail("Expected allReduceSum pattern for activations")
        }
    }
}

// MARK: - ShardedGradient Tests

final class ShardedGradientTests: XCTestCase {

    func testUnsynchronizedGradient() {
        let tracer = Tracer(shape: TensorShape([4, 8]), dtype: .float32)
        let sharding = SDYSharding(mesh: "mesh", axes: ["x", nil])

        let gradient = ShardedGradient(gradient: tracer, sharding: sharding, isSynchronized: false)

        XCTAssertFalse(gradient.isSynchronized)
        XCTAssertNotNil(gradient.sharding)
    }

    func testSynchronizedGradient() {
        let tracer = Tracer(shape: TensorShape([4, 8]), dtype: .float32)
        let sharding = SDYSharding(mesh: "mesh", axes: ["x", nil])

        let gradient = ShardedGradient(gradient: tracer, sharding: sharding, isSynchronized: true)

        XCTAssertTrue(gradient.isSynchronized)
    }
}

// MARK: - ShardedDifferentiableTracer Tests

final class ShardedDifferentiableTracerTests: XCTestCase {

    func testShardedTracerCreation() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let tracer = Tracer(shape: TensorShape([16, 64]), dtype: .float32)
        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")

        let shardedTracer = ShardedDifferentiableTracer(
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
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let tracer = Tracer(shape: TensorShape([16, 64]), dtype: .float32)
        let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)

        let shardedTracer = ShardedDifferentiableTracer(
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

// MARK: - GradientSyncPattern Tests

final class GradientSyncPatternTests: XCTestCase {

    func testPatternEquality() {
        let pattern1 = GradientSyncPattern.allReduceSum(axes: ["x", "y"])
        let pattern2 = GradientSyncPattern.allReduceSum(axes: ["x", "y"])
        let pattern3 = GradientSyncPattern.allReduceMean(axes: ["x", "y"])

        XCTAssertEqual(pattern1, pattern2)
        XCTAssertNotEqual(pattern1, pattern3)
    }

    func testNonePattern() {
        let pattern = GradientSyncPattern.none
        XCTAssertEqual(pattern, .none)
    }
}

// MARK: - ShardedDifferentiableContext Tests

final class ShardedDifferentiableContextTests: XCTestCase {

    func testContextCreation() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        XCTAssertEqual(ctx.mesh.name, "mesh")
        XCTAssertEqual(ctx.mesh.deviceCount, 4)
    }

    func testInputCreation() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2)
        let input = ctx.input(
            shape: TensorShape([16, 64]),
            dtype: .float32,
            sharding: xSharding,
            parameterType: .activation
        )

        XCTAssertEqual(input.shape.dimensions, [16, 64])
        XCTAssertNotNil(input.sharding)
    }

    func testMLIRGeneration() {
        let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let x = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: sharding)
        let w = ctx.input(shape: TensorShape([64, 32]), dtype: .float32, sharding: nil)

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

// MARK: - ShardedOptimizer Tests

final class ShardedOptimizerTests: XCTestCase {

    func testOptimizerCreation() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let optimizer = ShardedOptimizer(mesh: mesh, learningRate: 0.01)

        XCTAssertEqual(optimizer.learningRate, 0.01)
        XCTAssertEqual(optimizer.mesh.name, "mesh")
    }

    func testLearningRateUpdate() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let optimizer = ShardedOptimizer(mesh: mesh, learningRate: 0.01)

        optimizer.learningRate = 0.001
        XCTAssertEqual(optimizer.learningRate, 0.001)
    }
}

// MARK: - ShardedTrainingStep Tests

final class ShardedTrainingStepTests: XCTestCase {

    func testTrainingStepCreation() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let shardings = [
            SDYSharding.dataParallel(mesh: mesh, rank: 2),
            SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1)
        ]

        let step = ShardedTrainingStep(mesh: mesh, parameterShardings: shardings)

        XCTAssertEqual(step.mesh.deviceCount, 8)
        XCTAssertEqual(step.parameterShardings.count, 2)
    }
}

// MARK: - Convenience Function Tests

final class ShardedAutoDiffConvenienceFunctionTests: XCTestCase {

    func testShardedGradientConstruction() {
        // Test that we can construct the gradient sync infrastructure
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let input = Tracer(shape: TensorShape([16, 64]), dtype: .float32)

        // Create sharded tracer directly
        let shardedTracer = ShardedDifferentiableTracer(
            tracer: input,
            sharding: sharding,
            mesh: mesh,
            parameterType: .weight
        )

        XCTAssertNotNil(shardedTracer.sharding)
        // Data-parallel sharding should have no sync needed (already partitioned)
        XCTAssertEqual(shardedTracer.gradientSyncPattern, .none)
    }

    func testReplicatedParameterGradientPattern() {
        // Test that replicated parameters get the correct sync pattern
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)
        let input = Tracer(shape: TensorShape([16, 64]), dtype: .float32)

        let shardedTracer = ShardedDifferentiableTracer(
            tracer: input,
            sharding: sharding,
            mesh: mesh,
            parameterType: .weight
        )

        // Replicated weights should get allReduceMean pattern
        if case .allReduceMean(let axes) = shardedTracer.gradientSyncPattern {
            XCTAssertEqual(axes, ["batch"])
        } else {
            XCTFail("Expected allReduceMean pattern for replicated weights")
        }
    }
}

// MARK: - Integration Tests

final class ShardedAutoDiffIntegrationTests: XCTestCase {

    func testDataParallelMLP() {
        let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        // Data-parallel input sharding
        let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        // Replicated weights
        let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32,
                          sharding: xSharding, parameterType: .activation)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32,
                           sharding: wSharding, parameterType: .weight)
        let w2 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32,
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
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        // Data-parallel on batch dimension
        let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])

        // Column-parallel for QKV projection (shard output)
        let qkvSharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")

        // Row-parallel for output projection (shard input)
        let outSharding = SDYSharding.rowParallel(mesh: mesh, rank: 2, axis: "model")

        let x = ctx.input(shape: TensorShape([32, 512]), dtype: .float32,
                          sharding: xSharding, parameterType: .activation)
        let wQKV = ctx.input(shape: TensorShape([512, 1536]), dtype: .float32,
                             sharding: qkvSharding, parameterType: .weight)
        // Output projection: 1536 -> 512 to match QKV output
        let wOut = ctx.input(shape: TensorShape([1536, 512]), dtype: .float32,
                             sharding: outSharding, parameterType: .weight)

        let qkv = x.tracer.matmul(wQKV.tracer)  // [32, 512] x [512, 1536] = [32, 1536]
        let attnOut = qkv.relu()  // Simplified attention [32, 1536]
        let y = attnOut.matmul(wOut.tracer)  // [32, 1536] x [1536, 512] = [32, 512]

        ctx.output(y)

        let mlir = ctx.buildShardedModule(name: "hybrid_transformer")

        XCTAssertTrue(mlir.contains("sdy.mesh @mesh"))
        XCTAssertTrue(mlir.contains("\"data\"=2"))
        XCTAssertTrue(mlir.contains("\"model\"=4"))
    }
}
