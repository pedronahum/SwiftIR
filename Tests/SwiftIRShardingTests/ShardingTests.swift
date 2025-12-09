/// Unit Tests for SwiftIRSharding Module
/// Tests DeviceMesh, TensorSharding, and SDY integration

import XCTest
@testable import SwiftIRSharding

final class MeshAxisTests: XCTestCase {
    func testMeshAxisCreation() {
        let axis = MeshAxis(name: "x", size: 4)
        XCTAssertEqual(axis.name, "x")
        XCTAssertEqual(axis.size, 4)
    }

    func testMeshAxisDescription() {
        let axis = MeshAxis(name: "batch", size: 8)
        XCTAssertEqual(axis.description, "\"batch\"=8")
    }

    func testMeshAxisEquality() {
        let axis1 = MeshAxis(name: "x", size: 4)
        let axis2 = MeshAxis(name: "x", size: 4)
        let axis3 = MeshAxis(name: "y", size: 4)
        let axis4 = MeshAxis(name: "x", size: 8)

        XCTAssertEqual(axis1, axis2)
        XCTAssertNotEqual(axis1, axis3)
        XCTAssertNotEqual(axis1, axis4)
    }
}

final class DeviceMeshTests: XCTestCase {
    func testLinearMeshCreation() {
        let mesh = DeviceMesh.linear(name: "data_parallel", axisName: "batch", size: 8)

        XCTAssertEqual(mesh.name, "data_parallel")
        XCTAssertEqual(mesh.axes.count, 1)
        XCTAssertEqual(mesh.axes[0].name, "batch")
        XCTAssertEqual(mesh.axes[0].size, 8)
        XCTAssertNil(mesh.deviceIds)
    }

    func testGridMeshCreation() {
        let mesh = DeviceMesh.grid(name: "mesh_2x4", rows: 2, cols: 4)

        XCTAssertEqual(mesh.name, "mesh_2x4")
        XCTAssertEqual(mesh.axes.count, 2)
        XCTAssertEqual(mesh.axes[0].name, "x")
        XCTAssertEqual(mesh.axes[0].size, 2)
        XCTAssertEqual(mesh.axes[1].name, "y")
        XCTAssertEqual(mesh.axes[1].size, 4)
    }

    func testGridMeshWithCustomAxisNames() {
        let mesh = DeviceMesh.grid(name: "tpu_mesh", rows: 2, cols: 4, rowAxis: "data", colAxis: "model")

        XCTAssertEqual(mesh.axes[0].name, "data")
        XCTAssertEqual(mesh.axes[1].name, "model")
    }

    func testMeshMLIRText() {
        let mesh = DeviceMesh.grid(name: "mesh_2x2", rows: 2, cols: 2)
        let mlirText = mesh.mlirText

        XCTAssertTrue(mlirText.contains("sdy.mesh @mesh_2x2"))
        XCTAssertTrue(mlirText.contains("\"x\"=2"))
        XCTAssertTrue(mlirText.contains("\"y\"=2"))
    }

    func testMeshWithDeviceIds() {
        let mesh = DeviceMesh(
            name: "custom_mesh",
            axes: [MeshAxis(name: "a", size: 2)],
            deviceIds: [0, 1]
        )

        XCTAssertEqual(mesh.deviceIds, [0, 1])
        XCTAssertTrue(mesh.mlirText.contains("device_ids=[0, 1]"))
    }

    func testMeshEquality() {
        let mesh1 = DeviceMesh.grid(name: "mesh", rows: 2, cols: 2)
        let mesh2 = DeviceMesh.grid(name: "mesh", rows: 2, cols: 2)
        let mesh3 = DeviceMesh.grid(name: "other", rows: 2, cols: 2)

        XCTAssertEqual(mesh1, mesh2)
        XCTAssertNotEqual(mesh1, mesh3)
    }
}

final class AxisRefTests: XCTestCase {
    func testSimpleAxisRef() {
        let ref = AxisRef("x")
        XCTAssertEqual(ref.name, "x")
        XCTAssertNil(ref.subAxisInfo)
        XCTAssertEqual(ref.description, "\"x\"")
    }

    func testAxisRefWithSubAxis() {
        let subAxis = SubAxisInfo(preSize: 2, size: 4)
        let ref = AxisRef(name: "batch", subAxisInfo: subAxis)

        XCTAssertEqual(ref.name, "batch")
        XCTAssertNotNil(ref.subAxisInfo)
        XCTAssertEqual(ref.subAxisInfo?.preSize, 2)
        XCTAssertEqual(ref.subAxisInfo?.size, 4)
        XCTAssertEqual(ref.description, "\"batch\":(2)4")
    }
}

final class SubAxisInfoTests: XCTestCase {
    func testSubAxisInfoCreation() {
        let info = SubAxisInfo(preSize: 1, size: 4)
        XCTAssertEqual(info.preSize, 1)
        XCTAssertEqual(info.size, 4)
    }

    func testSubAxisInfoDescription() {
        let info = SubAxisInfo(preSize: 2, size: 8)
        XCTAssertEqual(info.description, "(2)8")
    }
}

final class DimensionShardingTests: XCTestCase {
    func testSimpleDimensionSharding() {
        let sharding = DimensionSharding("x")

        XCTAssertEqual(sharding.axes.count, 1)
        XCTAssertEqual(sharding.axes[0].name, "x")
        XCTAssertFalse(sharding.isClosed)
        XCTAssertNil(sharding.priority)
    }

    func testReplicatedDimension() {
        let replicated = DimensionSharding.replicated

        XCTAssertTrue(replicated.axes.isEmpty)
        XCTAssertTrue(replicated.isClosed)
        XCTAssertEqual(replicated.description, "{}")
    }

    func testOpenDimension() {
        let open = DimensionSharding.open

        XCTAssertTrue(open.axes.isEmpty)
        XCTAssertFalse(open.isClosed)
        XCTAssertEqual(open.description, "{?}")
    }

    func testMultiAxisDimensionSharding() {
        let sharding = DimensionSharding(
            axes: [AxisRef("x"), AxisRef("y")],
            isClosed: true
        )

        XCTAssertEqual(sharding.axes.count, 2)
        XCTAssertTrue(sharding.isClosed)
        XCTAssertTrue(sharding.description.contains("\"x\""))
        XCTAssertTrue(sharding.description.contains("\"y\""))
    }

    func testDimensionShardingWithPriority() {
        let sharding = DimensionSharding(
            axes: [AxisRef("x")],
            isClosed: false,
            priority: 1
        )

        XCTAssertEqual(sharding.priority, 1)
        XCTAssertTrue(sharding.description.contains("p1"))
    }
}

final class TensorShardingTests: XCTestCase {
    func testSimpleTensorSharding() {
        let sharding = TensorSharding(meshName: "mesh", axisNames: ["x", "y"])

        XCTAssertEqual(sharding.meshName, "mesh")
        XCTAssertEqual(sharding.dimShardings.count, 2)
        XCTAssertEqual(sharding.dimShardings[0].axes[0].name, "x")
        XCTAssertEqual(sharding.dimShardings[1].axes[0].name, "y")
    }

    func testMixedTensorSharding() {
        let sharding = TensorSharding(meshName: "mesh", axisNames: ["x", nil])

        XCTAssertEqual(sharding.dimShardings.count, 2)
        XCTAssertFalse(sharding.dimShardings[0].axes.isEmpty)
        XCTAssertTrue(sharding.dimShardings[1].axes.isEmpty) // Replicated
    }

    func testReplicatedTensorSharding() {
        let sharding = TensorSharding.replicated(meshName: "mesh", rank: 3)

        XCTAssertEqual(sharding.dimShardings.count, 3)
        for dimSharding in sharding.dimShardings {
            XCTAssertTrue(dimSharding.axes.isEmpty)
            XCTAssertTrue(dimSharding.isClosed)
        }
    }

    func testOpenTensorSharding() {
        let sharding = TensorSharding.open(meshName: "mesh", rank: 2)

        XCTAssertEqual(sharding.dimShardings.count, 2)
        for dimSharding in sharding.dimShardings {
            XCTAssertTrue(dimSharding.axes.isEmpty)
            XCTAssertFalse(dimSharding.isClosed)
        }
    }

    func testTensorShardingMLIRText() {
        let sharding = TensorSharding(meshName: "mesh_2x2", axisNames: ["x", "y"])
        let mlirText = sharding.mlirAttributeText

        XCTAssertTrue(mlirText.contains("#sdy.sharding"))
        XCTAssertTrue(mlirText.contains("@mesh_2x2"))
        XCTAssertTrue(mlirText.contains("\"x\""))
        XCTAssertTrue(mlirText.contains("\"y\""))
    }

    func testTensorShardingWithReplicatedAxes() {
        let sharding = TensorSharding(
            meshName: "mesh",
            dimShardings: [DimensionSharding("x")],
            replicatedAxes: [AxisRef("y")]
        )

        XCTAssertEqual(sharding.replicatedAxes.count, 1)
        XCTAssertTrue(sharding.description.contains("replicated"))
    }
}

final class ShardingContextTests: XCTestCase {
    func testShardingContextCreation() {
        let ctx = ShardingContext()
        XCTAssertTrue(ctx.meshes.isEmpty)
    }

    func testAddMesh() {
        let ctx = ShardingContext()
        let mesh = DeviceMesh.grid(name: "test_mesh", rows: 2, cols: 2)

        ctx.addMesh(mesh)

        XCTAssertEqual(ctx.meshes.count, 1)
        XCTAssertNotNil(ctx.mesh(named: "test_mesh"))
    }

    func testMeshLookup() {
        let ctx = ShardingContext()
        let mesh1 = DeviceMesh.linear(name: "mesh1", axisName: "x", size: 4)
        let mesh2 = DeviceMesh.grid(name: "mesh2", rows: 2, cols: 2)

        ctx.addMesh(mesh1)
        ctx.addMesh(mesh2)

        XCTAssertNotNil(ctx.mesh(named: "mesh1"))
        XCTAssertNotNil(ctx.mesh(named: "mesh2"))
        XCTAssertNil(ctx.mesh(named: "nonexistent"))
    }

    func testMeshDefinitions() {
        let ctx = ShardingContext()
        ctx.addMesh(DeviceMesh.grid(name: "mesh", rows: 2, cols: 2))

        let defs = ctx.meshDefinitions()

        XCTAssertTrue(defs.contains("sdy.mesh @mesh"))
    }

    func testCreateSharding() {
        let ctx = ShardingContext()
        ctx.addMesh(DeviceMesh.grid(name: "mesh", rows: 2, cols: 2))

        let sharding = ctx.sharding(mesh: "mesh", axes: ["x", "y"])

        XCTAssertNotNil(sharding)
        XCTAssertEqual(sharding?.meshName, "mesh")
    }

    func testCreateShardingForNonexistentMesh() {
        let ctx = ShardingContext()

        let sharding = ctx.sharding(mesh: "nonexistent", axes: ["x"])

        XCTAssertNil(sharding)
    }
}

final class SdyPassPipelineTests: XCTestCase {
    func testPassPipelineNames() {
        XCTAssertEqual(SdyPassPipeline.propagation, "sdy-propagation-pipeline")
        XCTAssertEqual(SdyPassPipeline.import, "sdy-import-pipeline")
        XCTAssertEqual(SdyPassPipeline.export, "sdy-export-pipeline")
        XCTAssertEqual(SdyPassPipeline.basicPropagate, "sdy-basic-propagate")
        XCTAssertEqual(SdyPassPipeline.aggressivePropagate, "sdy-aggressive-propagate")
        XCTAssertEqual(SdyPassPipeline.userPriorityPropagate, "sdy-user-priority-propagate")
        XCTAssertEqual(SdyPassPipeline.opPriorityPropagate, "sdy-op-priority-propagate")
    }
}

// MARK: - Integration Tests

final class ShardingIntegrationTests: XCTestCase {
    func testFullShardingWorkflow() {
        // Create a sharding context
        let ctx = ShardingContext()

        // Add a 2x2 mesh
        let mesh = DeviceMesh.grid(name: "mesh_2x2", rows: 2, cols: 2)
        ctx.addMesh(mesh)

        // Create sharding for a 2D tensor (batch x features)
        // - Dimension 0 (batch) sharded on x axis
        // - Dimension 1 (features) sharded on y axis
        let sharding = ctx.sharding(mesh: "mesh_2x2", axes: ["x", "y"])

        XCTAssertNotNil(sharding)

        // Verify the MLIR output looks correct
        let meshDef = ctx.meshDefinitions()
        XCTAssertTrue(meshDef.contains("sdy.mesh @mesh_2x2"))

        if let s = sharding {
            let attr = s.mlirAttributeText
            XCTAssertTrue(attr.contains("#sdy.sharding"))
            XCTAssertTrue(attr.contains("@mesh_2x2"))
        }
    }

    func testDataParallelSharding() {
        // Common pattern: shard only the batch dimension
        let ctx = ShardingContext()
        ctx.addMesh(DeviceMesh.linear(name: "data", axisName: "batch", size: 8))

        // 4D tensor: [batch, height, width, channels]
        // Only batch is sharded, rest replicated
        let sharding = ctx.sharding(mesh: "data", axes: ["batch", nil, nil, nil])

        XCTAssertNotNil(sharding)
        XCTAssertEqual(sharding?.dimShardings.count, 4)

        // First dimension should be sharded on "batch"
        XCTAssertEqual(sharding?.dimShardings[0].axes.first?.name, "batch")

        // Rest should be replicated (closed, no axes)
        for i in 1..<4 {
            XCTAssertTrue(sharding!.dimShardings[i].axes.isEmpty)
            XCTAssertTrue(sharding!.dimShardings[i].isClosed)
        }
    }

    func testModelParallelSharding() {
        // Pattern: shard the hidden dimension
        let ctx = ShardingContext()
        ctx.addMesh(DeviceMesh.linear(name: "model", axisName: "hidden", size: 4))

        // Weight matrix: [in_features, out_features]
        // Shard out_features dimension
        let sharding = TensorSharding(
            meshName: "model",
            dimShardings: [
                .replicated,  // in_features
                DimensionSharding("hidden")  // out_features sharded
            ]
        )

        XCTAssertEqual(sharding.dimShardings[0].description, "{}")
        XCTAssertTrue(sharding.dimShardings[1].description.contains("\"hidden\""))
    }
}
