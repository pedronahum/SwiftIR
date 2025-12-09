/// Pure Swift Unit Tests for SwiftIRSharding Module
/// These tests don't require linking against the SDY C API

import XCTest
@testable import SwiftIRSharding

// MARK: - Pure Swift Tests (No C API linking required)

final class PureSwiftMeshAxisTests: XCTestCase {
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

        XCTAssertEqual(axis1, axis2)
        XCTAssertNotEqual(axis1, axis3)
    }
}

final class PureSwiftDeviceMeshTests: XCTestCase {
    func testLinearMeshCreation() {
        let mesh = DeviceMesh.linear(name: "data_parallel", axisName: "batch", size: 8)

        XCTAssertEqual(mesh.name, "data_parallel")
        XCTAssertEqual(mesh.axes.count, 1)
        XCTAssertEqual(mesh.axes[0].name, "batch")
        XCTAssertEqual(mesh.axes[0].size, 8)
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

    func testMeshEquality() {
        let mesh1 = DeviceMesh.grid(name: "mesh", rows: 2, cols: 2)
        let mesh2 = DeviceMesh.grid(name: "mesh", rows: 2, cols: 2)
        let mesh3 = DeviceMesh.grid(name: "other", rows: 2, cols: 2)

        XCTAssertEqual(mesh1, mesh2)
        XCTAssertNotEqual(mesh1, mesh3)
    }
}

final class PureSwiftAxisRefTests: XCTestCase {
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

final class PureSwiftDimensionShardingTests: XCTestCase {
    func testSimpleDimensionSharding() {
        let sharding = DimensionSharding("x")

        XCTAssertEqual(sharding.axes.count, 1)
        XCTAssertEqual(sharding.axes[0].name, "x")
        XCTAssertFalse(sharding.isClosed)
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
}

final class PureSwiftTensorShardingTests: XCTestCase {
    func testSimpleTensorSharding() {
        let sharding = TensorSharding(meshName: "mesh", axisNames: ["x", "y"])

        XCTAssertEqual(sharding.meshName, "mesh")
        XCTAssertEqual(sharding.dimShardings.count, 2)
    }

    func testReplicatedTensorSharding() {
        let sharding = TensorSharding.replicated(meshName: "mesh", rank: 3)

        XCTAssertEqual(sharding.dimShardings.count, 3)
        for dimSharding in sharding.dimShardings {
            XCTAssertTrue(dimSharding.isClosed)
        }
    }

    func testOpenTensorSharding() {
        let sharding = TensorSharding.open(meshName: "mesh", rank: 2)

        XCTAssertEqual(sharding.dimShardings.count, 2)
        for dimSharding in sharding.dimShardings {
            XCTAssertFalse(dimSharding.isClosed)
        }
    }

    func testTensorShardingMLIRText() {
        let sharding = TensorSharding(meshName: "mesh_2x2", axisNames: ["x", "y"])
        let mlirText = sharding.mlirAttributeText

        XCTAssertTrue(mlirText.contains("#sdy.sharding"))
        XCTAssertTrue(mlirText.contains("@mesh_2x2"))
    }
}

final class PureSwiftShardingContextTests: XCTestCase {
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
}

final class PureSwiftSdyPassPipelineTests: XCTestCase {
    func testPassPipelineNames() {
        XCTAssertEqual(SdyPassPipeline.propagation, "sdy-propagation-pipeline")
        XCTAssertEqual(SdyPassPipeline.import, "sdy-import-pipeline")
        XCTAssertEqual(SdyPassPipeline.export, "sdy-export-pipeline")
        XCTAssertEqual(SdyPassPipeline.basicPropagate, "sdy-basic-propagate")
    }
}
