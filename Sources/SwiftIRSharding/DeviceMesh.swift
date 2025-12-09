/// Device Mesh types for tensor sharding in SwiftIR
/// These types provide Swift-friendly wrappers around Shardy's mesh concepts.

import SdyCAPIWrapper
import MLIRCoreWrapper

// MARK: - MeshAxis

/// Represents a single axis of a device mesh.
///
/// A mesh axis has a name (e.g., "x", "y", "batch") and a size representing
/// the number of devices along that dimension.
///
/// Example:
/// ```swift
/// let xAxis = MeshAxis(name: "x", size: 4)  // 4 devices along x axis
/// ```
public struct MeshAxis: Equatable, CustomStringConvertible {
    /// The name of this axis (e.g., "x", "y", "data", "model")
    public let name: String

    /// The number of devices along this axis
    public let size: Int64

    /// Creates a mesh axis with the given name and size.
    /// - Parameters:
    ///   - name: The axis name
    ///   - size: The number of devices along this axis
    public init(name: String, size: Int64) {
        self.name = name
        self.size = size
    }

    public var description: String {
        "\"\(name)\"=\(size)"
    }

    /// Creates an MLIR attribute for this mesh axis.
    /// - Parameter context: The MLIR context
    /// - Returns: An MlirAttribute representing this mesh axis
    public func toAttribute(context: MlirContext) -> MlirAttribute {
        name.withCString { namePtr in
            let nameRef = sdyStringRefCreate(namePtr, name.utf8.count)
            return sdyMeshAxisAttrGet(context, nameRef, size)
        }
    }

    /// Creates a MeshAxis from an MLIR attribute.
    /// - Parameter attr: The MLIR attribute (must be a MeshAxisAttr)
    /// - Returns: A MeshAxis, or nil if the attribute is not a MeshAxisAttr
    public static func fromAttribute(_ attr: MlirAttribute) -> MeshAxis? {
        guard sdyAttributeIsAMeshAxisAttr(attr) else { return nil }

        let nameRef = sdyMeshAxisAttrGetName(attr)
        let name: String
        if let data = nameRef.data {
            name = String(
                decoding: UnsafeBufferPointer(
                    start: UnsafeRawPointer(data).assumingMemoryBound(to: UInt8.self),
                    count: Int(nameRef.length)
                ),
                as: UTF8.self
            )
        } else {
            name = ""
        }
        let size = sdyMeshAxisAttrGetSize(attr)

        return MeshAxis(name: name, size: size)
    }
}

// MARK: - DeviceMesh

/// Represents a device mesh topology for tensor sharding.
///
/// A device mesh describes how physical devices are organized into a logical
/// multi-dimensional grid. Each dimension of the mesh is named and can be used
/// to specify how tensor dimensions are partitioned across devices.
///
/// Example:
/// ```swift
/// // Create a 2x4 mesh (8 devices total)
/// let mesh = DeviceMesh(
///     name: "tpu_mesh",
///     axes: [
///         MeshAxis(name: "x", size: 2),
///         MeshAxis(name: "y", size: 4)
///     ]
/// )
/// ```
public struct DeviceMesh: Equatable, CustomStringConvertible {
    /// The name of this mesh (used for referencing in MLIR)
    public let name: String

    /// The axes of this mesh
    public let axes: [MeshAxis]

    /// Optional explicit device IDs for non-uniform meshes
    public let deviceIds: [Int64]?

    /// Creates a device mesh with the given configuration.
    /// - Parameters:
    ///   - name: A name for this mesh (will be used as @name in MLIR)
    ///   - axes: The axes defining the mesh topology
    ///   - deviceIds: Optional explicit device IDs
    public init(name: String, axes: [MeshAxis], deviceIds: [Int64]? = nil) {
        self.name = name
        self.axes = axes
        self.deviceIds = deviceIds
    }

    /// Creates a simple 1D mesh.
    /// - Parameters:
    ///   - name: The mesh name
    ///   - axisName: The name of the single axis
    ///   - size: The number of devices
    public static func linear(name: String, axisName: String = "x", size: Int64) -> DeviceMesh {
        DeviceMesh(name: name, axes: [MeshAxis(name: axisName, size: size)])
    }

    /// Creates a 2D mesh.
    /// - Parameters:
    ///   - name: The mesh name
    ///   - rows: Number of rows (axis "x")
    ///   - cols: Number of columns (axis "y")
    public static func grid(name: String, rows: Int64, cols: Int64) -> DeviceMesh {
        grid(name: name, rows: rows, cols: cols, rowAxis: "x", colAxis: "y")
    }

    /// Creates a 2D mesh with custom axis names.
    /// - Parameters:
    ///   - name: The mesh name
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - rowAxis: Name for the row axis
    ///   - colAxis: Name for the column axis
    public static func grid(
        name: String,
        rows: Int64,
        cols: Int64,
        rowAxis: String,
        colAxis: String
    ) -> DeviceMesh {
        DeviceMesh(name: name, axes: [
            MeshAxis(name: rowAxis, size: rows),
            MeshAxis(name: colAxis, size: cols)
        ])
    }

    /// The total number of devices in this mesh.
    public var deviceCount: Int64 {
        axes.reduce(1) { $0 * $1.size }
    }

    /// Gets an axis by name.
    /// - Parameter name: The axis name
    /// - Returns: The axis, or nil if not found
    public func axis(named name: String) -> MeshAxis? {
        axes.first { $0.name == name }
    }

    public var description: String {
        let axesStr = axes.map { $0.description }.joined(separator: ", ")
        if let ids = deviceIds {
            return "@\(name) = <[\(axesStr)], device_ids=[\(ids.map(String.init).joined(separator: ", "))]>"
        }
        return "@\(name) = <[\(axesStr)]>"
    }

    /// Converts this mesh to MLIR textual form suitable for sdy.mesh op.
    public var mlirText: String {
        let axesStr = axes.map { $0.description }.joined(separator: ", ")
        if let ids = deviceIds {
            return "sdy.mesh @\(name) = <[\(axesStr)], device_ids=[\(ids.map(String.init).joined(separator: ", "))]>"
        }
        return "sdy.mesh @\(name) = <[\(axesStr)]>"
    }

    /// Creates an MLIR MeshAttr for this mesh.
    /// - Parameter context: The MLIR context
    /// - Returns: An MlirAttribute representing this mesh
    public func toAttribute(context: MlirContext) -> MlirAttribute {
        var axisAttrs = axes.map { $0.toAttribute(context: context) }

        return axisAttrs.withUnsafeMutableBufferPointer { axesPtr in
            if let deviceIds = deviceIds {
                var ids = deviceIds
                return ids.withUnsafeMutableBufferPointer { idsPtr in
                    sdyMeshAttrGet(
                        context,
                        axesPtr.count,
                        axesPtr.baseAddress,
                        idsPtr.count,
                        idsPtr.baseAddress
                    )
                }
            } else {
                return sdyMeshAttrGet(
                    context,
                    axesPtr.count,
                    axesPtr.baseAddress,
                    0,
                    nil
                )
            }
        }
    }

    /// Creates a DeviceMesh from an MLIR MeshAttr.
    /// - Parameters:
    ///   - attr: The MLIR attribute (must be a MeshAttr)
    ///   - name: The name to assign to the mesh
    /// - Returns: A DeviceMesh, or nil if the attribute is not a MeshAttr
    public static func fromAttribute(_ attr: MlirAttribute, name: String) -> DeviceMesh? {
        guard sdyAttributeIsAMeshAttr(attr) else { return nil }

        // Extract axes
        let axesCount = sdyMeshAttrGetAxesSize(attr)
        var axes: [MeshAxis] = []
        for i in 0..<axesCount {
            let axisAttr = sdyMeshAttrGetAxesElem(attr, i)
            if let axis = MeshAxis.fromAttribute(axisAttr) {
                axes.append(axis)
            }
        }

        // Extract device IDs
        let idsCount = sdyMeshAttrGetDeviceIdsSize(attr)
        var deviceIds: [Int64]? = nil
        if idsCount > 0 {
            deviceIds = []
            for i in 0..<idsCount {
                deviceIds?.append(sdyMeshAttrGetDeviceIdsElem(attr, i))
            }
        }

        return DeviceMesh(name: name, axes: axes, deviceIds: deviceIds)
    }
}
