/// DeviceMesh for SwiftIRShardingLite
/// Pure Swift implementation - no C dependencies.
///
/// A device mesh represents the topology of compute devices for distributed execution.

import Foundation

// MARK: - Mesh Axis

/// Represents a single axis of a device mesh.
public struct MeshAxis: Equatable, Hashable {
    /// The name of this axis (e.g., "x", "y", "batch", "model")
    public let name: String

    /// The size of this axis (number of devices along this dimension)
    public let size: Int

    /// Creates a mesh axis.
    public init(name: String, size: Int) {
        precondition(size > 0, "Mesh axis size must be positive")
        self.name = name
        self.size = size
    }

    /// MLIR text representation
    public var mlirText: String {
        "\"\(name)\"=\(size)"
    }
}

// MARK: - Device Mesh

/// Represents a multi-dimensional mesh of compute devices.
///
/// Device meshes define the physical layout of accelerators for distributed
/// computation. Shardy uses meshes to determine how tensors are partitioned
/// across devices.
public struct DeviceMesh: Equatable, Hashable {
    /// The name of this mesh (used as an MLIR symbol)
    public let name: String

    /// The axes of the mesh
    public let axes: [MeshAxis]

    /// Optional explicit device IDs (for non-standard topologies)
    public let deviceIds: [Int]?

    /// Creates a device mesh with the given axes.
    public init(name: String, axes: [MeshAxis], deviceIds: [Int]? = nil) {
        self.name = name
        self.axes = axes
        self.deviceIds = deviceIds
    }

    /// Total number of devices in the mesh
    public var totalDevices: Int {
        axes.reduce(1) { $0 * $1.size }
    }

    /// Creates a 1D linear mesh.
    public static func linear(name: String, axisName: String, size: Int) -> DeviceMesh {
        DeviceMesh(name: name, axes: [MeshAxis(name: axisName, size: size)])
    }

    /// Creates a 2D grid mesh.
    public static func grid(
        name: String,
        rows: Int, cols: Int,
        rowAxis: String = "x", colAxis: String = "y"
    ) -> DeviceMesh {
        DeviceMesh(name: name, axes: [
            MeshAxis(name: rowAxis, size: rows),
            MeshAxis(name: colAxis, size: cols)
        ])
    }

    /// Creates a 3D mesh.
    public static func cube(
        name: String,
        x: Int, y: Int, z: Int,
        xAxis: String = "x", yAxis: String = "y", zAxis: String = "z"
    ) -> DeviceMesh {
        DeviceMesh(name: name, axes: [
            MeshAxis(name: xAxis, size: x),
            MeshAxis(name: yAxis, size: y),
            MeshAxis(name: zAxis, size: z)
        ])
    }

    /// MLIR text representation for mesh definition
    public var mlirText: String {
        let axesStr = axes.map(\.mlirText).joined(separator: ", ")
        var result = "sdy.mesh @\(name) = <[\(axesStr)]"
        if let ids = deviceIds {
            let idsStr = ids.map(String.init).joined(separator: ", ")
            result += ", device_ids=[\(idsStr)]"
        }
        result += ">"
        return result
    }
}

// MARK: - Sharding Context

/// Manages meshes and shardings for a compilation.
public class ShardingContext {
    /// Registered meshes by name
    private var meshes: [String: DeviceMesh] = [:]

    public init() {}

    /// Adds a mesh to the context.
    public func addMesh(_ mesh: DeviceMesh) {
        meshes[mesh.name] = mesh
    }

    /// Gets a mesh by name.
    public func mesh(named name: String) -> DeviceMesh? {
        meshes[name]
    }

    /// Gets all registered meshes.
    public var allMeshes: [DeviceMesh] {
        Array(meshes.values)
    }

    /// Generates MLIR text for all mesh definitions.
    public func meshDefinitions() -> String {
        meshes.values.map(\.mlirText).joined(separator: "\n")
    }

    /// Creates a sharding specification for a tensor.
    public func sharding(mesh meshName: String, axes axisNames: [String?]) -> TensorSharding? {
        guard meshes[meshName] != nil else { return nil }
        return TensorSharding(meshName: meshName, axisNames: axisNames)
    }

    /// Generates a sharding attribute string.
    public func shardingAttribute(
        mesh meshName: String,
        dimShardings: [DimensionSharding]
    ) -> String {
        let dimStr = dimShardings.map(\.mlirText).joined(separator: ", ")
        return "#sdy.sharding<@\(meshName), [\(dimStr)]>"
    }
}
