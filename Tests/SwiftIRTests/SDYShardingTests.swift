/// SDY Sharding Tests for SwiftIR Main Module
/// Tests the SDYMesh, SDYSharding, and ShardedTracingContext types

import XCTest
@testable import SwiftIR

// MARK: - SDYMesh Tests

final class SDYMeshTests: XCTestCase {

    func testLinearMeshCreation() {
        let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)

        XCTAssertEqual(mesh.name, "data_parallel")
        XCTAssertEqual(mesh.axes.count, 1)
        XCTAssertEqual(mesh.axes[0].name, "batch")
        XCTAssertEqual(mesh.axes[0].size, 4)
        XCTAssertEqual(mesh.deviceCount, 4)
    }

    func testGridMeshCreation() {
        let mesh = SDYMesh.grid(
            name: "hybrid",
            dataParallel: 2,
            modelParallel: 4,
            dataAxis: "data",
            modelAxis: "model"
        )

        XCTAssertEqual(mesh.name, "hybrid")
        XCTAssertEqual(mesh.axes.count, 2)
        XCTAssertEqual(mesh.axes[0].name, "data")
        XCTAssertEqual(mesh.axes[0].size, 2)
        XCTAssertEqual(mesh.axes[1].name, "model")
        XCTAssertEqual(mesh.axes[1].size, 4)
        XCTAssertEqual(mesh.deviceCount, 8)
    }

    func testTPUMeshCreation() {
        let mesh = SDYMesh.tpuMesh(name: "tpu_grid", x: 2, y: 4)

        XCTAssertEqual(mesh.name, "tpu_grid")
        XCTAssertEqual(mesh.axes.count, 2)
        XCTAssertEqual(mesh.axes[0].name, "x")
        XCTAssertEqual(mesh.axes[1].name, "y")
        XCTAssertEqual(mesh.deviceCount, 8)
    }

    func testCubeMeshCreation() {
        let mesh = SDYMesh.cube(name: "3d", data: 2, tensor: 4, pipeline: 2)

        XCTAssertEqual(mesh.name, "3d")
        XCTAssertEqual(mesh.axes.count, 3)
        XCTAssertEqual(mesh.deviceCount, 16)
    }

    func testMeshMLIRText() {
        let mesh = SDYMesh.linear(name: "test", axis: "x", size: 4)
        let mlir = mesh.mlirText

        XCTAssertTrue(mlir.contains("sdy.mesh @test"))
        XCTAssertTrue(mlir.contains("\"x\"=4"))
    }

    func testMeshEquality() {
        let mesh1 = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let mesh2 = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let mesh3 = SDYMesh.linear(name: "other", axis: "x", size: 4)

        XCTAssertEqual(mesh1, mesh2)
        XCTAssertNotEqual(mesh1, mesh3)
    }

    func testAxisSizeLookup() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)

        XCTAssertEqual(mesh.axisSize("data"), 2)
        XCTAssertEqual(mesh.axisSize("model"), 4)
        XCTAssertNil(mesh.axisSize("nonexistent"))
    }
}

// MARK: - SDYSharding Tests

final class SDYShardingTests: XCTestCase {

    func testReplicatedSharding() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let sharding = SDYSharding.replicated(mesh: mesh, rank: 2)

        XCTAssertEqual(sharding.meshName, "mesh")
        XCTAssertEqual(sharding.dimAxes.count, 2)
        XCTAssertTrue(sharding.isReplicated)
        XCTAssertTrue(sharding.dimAxes.allSatisfy { $0 == nil })
    }

    func testDataParallelSharding() {
        let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 4)
        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")

        XCTAssertEqual(sharding.dimAxes[0], "batch")
        XCTAssertNil(sharding.dimAxes[1])
        XCTAssertFalse(sharding.isReplicated)
    }

    func testModelParallelSharding() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let sharding = SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")

        XCTAssertNil(sharding.dimAxes[0])
        XCTAssertEqual(sharding.dimAxes[1], "model")
    }

    func testColumnParallelSharding() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "tp", size: 4)
        let sharding = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "tp")

        XCTAssertNil(sharding.dimAxes[0])  // Input dim replicated
        XCTAssertEqual(sharding.dimAxes[1], "tp")  // Output dim sharded
    }

    func testRowParallelSharding() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "tp", size: 4)
        let sharding = SDYSharding.rowParallel(mesh: mesh, rank: 2, axis: "tp")

        XCTAssertEqual(sharding.dimAxes[0], "tp")  // Input dim sharded
        XCTAssertNil(sharding.dimAxes[1])  // Output dim replicated
    }

    func testShardingMLIRAttributeText() {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let sharding = SDYSharding(mesh: mesh, axes: ["data", nil, "model"])

        let mlir = sharding.mlirAttributeText
        XCTAssertTrue(mlir.contains("#sdy.sharding<@mesh"))
        XCTAssertTrue(mlir.contains("{\"data\"}"))
        XCTAssertTrue(mlir.contains("{}"))
        XCTAssertTrue(mlir.contains("{\"model\"}"))
    }

    func testShardingForDim() {
        let sharding = SDYSharding(mesh: "mesh", axes: ["x", nil, "y"])

        XCTAssertEqual(sharding.shardingForDim(0), "x")
        XCTAssertNil(sharding.shardingForDim(1))
        XCTAssertEqual(sharding.shardingForDim(2), "y")
        XCTAssertNil(sharding.shardingForDim(10))  // Out of bounds
    }
}

// MARK: - ShardedTracer Tests

final class ShardedTracerTests: XCTestCase {

    func testShardedTracerCreation() {
        let tracer = Tracer(shape: TensorShape([4, 8]), dtype: .float32)
        let sharding = SDYSharding(mesh: "mesh", axes: ["x", nil])
        let shardedTracer = ShardedTracer(tracer: tracer, sharding: sharding)

        XCTAssertEqual(shardedTracer.shape.dimensions, [4, 8])
        XCTAssertEqual(shardedTracer.dtype, .float32)
        XCTAssertNotNil(shardedTracer.sharding)
    }

    func testShardedTracerWithoutSharding() {
        let tracer = Tracer(shape: TensorShape([4, 8]), dtype: .float32)
        let shardedTracer = ShardedTracer(tracer: tracer)

        XCTAssertNil(shardedTracer.sharding)
    }
}

// MARK: - ShardedTracingContext Tests

final class ShardedTracingContextTests: XCTestCase {

    func testContextCreation() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        XCTAssertEqual(ctx.mesh.name, "mesh")
    }

    func testInputWithSharding() {
        let mesh = SDYMesh.linear(name: "mesh", axis: "batch", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let sharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let input = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: sharding)

        XCTAssertEqual(input.shape.dimensions, [16, 64])
        XCTAssertNotNil(input.sharding)
    }

    func testMLIRGeneration() {
        let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

        let x = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: xSharding)
        let w = ctx.input(shape: TensorShape([64, 32]), dtype: .float32, sharding: wSharding)

        let y = x.tracer.matmul(w.tracer)
        ctx.output(y, sharding: xSharding)

        let mlir = ctx.buildShardedModule(name: "test_module")

        // Check module structure
        XCTAssertTrue(mlir.contains("module @test_module"))
        XCTAssertTrue(mlir.contains("sdy.mesh @data_parallel"))

        // Check sharding annotations
        XCTAssertTrue(mlir.contains("{sdy.sharding = #sdy.sharding<@data_parallel"))
        XCTAssertTrue(mlir.contains("{\"batch\"}"))

        // Check operations
        XCTAssertTrue(mlir.contains("stablehlo.dot"))
        XCTAssertTrue(mlir.contains("return"))
    }
}

// MARK: - Bridge Tests

final class SDYBridgeTests: XCTestCase {

    func testDeviceMeshToSDYMesh() {
        let deviceMesh = DeviceMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let sdyMesh = deviceMesh.toSDYMesh()

        XCTAssertEqual(sdyMesh.name, "mesh")
        XCTAssertEqual(sdyMesh.axes.count, 2)
        XCTAssertEqual(sdyMesh.axes[0].name, "data")
        XCTAssertEqual(sdyMesh.axes[0].size, 2)
    }

    func testShardingSpecToSDYSharding() {
        let deviceMesh = DeviceMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let spec = ShardingSpec.dataParallel(mesh: deviceMesh, rank: 2)
        let sdySharding = spec.toSDYSharding()

        XCTAssertEqual(sdySharding.meshName, "mesh")
        XCTAssertEqual(sdySharding.dimAxes[0], "data")
        XCTAssertNil(sdySharding.dimAxes[1])
    }
}

// MARK: - Convenience Function Tests

final class SDYConvenienceFunctionTests: XCTestCase {

    func testTraceDataParallel() {
        let mlir = traceDataParallel(
            numDevices: 4,
            inputShapes: [TensorShape([16, 64]), TensorShape([64, 32])],
            dtype: .float32
        ) { inputs in
            inputs[0].matmul(inputs[1])
        }

        XCTAssertTrue(mlir.contains("sdy.mesh @data_parallel"))
        XCTAssertTrue(mlir.contains("stablehlo.dot"))
    }
}
