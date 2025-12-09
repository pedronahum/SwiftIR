/// Tensor Sharding types for SwiftIR
/// These types specify how tensors are partitioned across a device mesh.

import SdyCAPIWrapper
import MLIRCoreWrapper

// MARK: - AxisRef

/// A reference to a mesh axis, optionally with sub-axis information.
///
/// AxisRef is used to specify which axis (or part of an axis) a tensor
/// dimension should be sharded along.
public struct AxisRef: Equatable, CustomStringConvertible {
    /// The name of the referenced axis
    public let name: String

    /// Optional sub-axis information for hierarchical partitioning
    public let subAxisInfo: SubAxisInfo?

    /// Creates an axis reference.
    /// - Parameters:
    ///   - name: The axis name
    ///   - subAxisInfo: Optional sub-axis information
    public init(name: String, subAxisInfo: SubAxisInfo? = nil) {
        self.name = name
        self.subAxisInfo = subAxisInfo
    }

    /// Creates a simple axis reference without sub-axis info.
    /// - Parameter name: The axis name
    public init(_ name: String) {
        self.name = name
        self.subAxisInfo = nil
    }

    public var description: String {
        if let sub = subAxisInfo {
            return "\"\(name)\":(\(sub.preSize))\(sub.size)"
        }
        return "\"\(name)\""
    }

    /// Creates an MLIR attribute for this axis reference.
    public func toAttribute(context: MlirContext) -> MlirAttribute {
        name.withCString { namePtr in
            let nameRef = sdyStringRefCreate(namePtr, name.utf8.count)
            let subAttr = subAxisInfo?.toAttribute(context: context) ?? sdyNullAttribute()
            return sdyAxisRefAttrGet(context, nameRef, subAttr)
        }
    }

    /// Creates an AxisRef from an MLIR attribute.
    public static func fromAttribute(_ attr: MlirAttribute) -> AxisRef? {
        guard sdyAttributeIsAnAxisRefAttr(attr) else { return nil }

        let nameRef = sdyAxisRefAttrGetName(attr)
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

        let subAttr = sdyAxisRefAttrGetSubAxisInfo(attr)
        let subAxisInfo = sdyAttributeIsNull(subAttr) ? nil : SubAxisInfo.fromAttribute(subAttr)

        return AxisRef(name: name, subAxisInfo: subAxisInfo)
    }
}

// MARK: - SubAxisInfo

/// Information about a sub-axis for hierarchical mesh partitioning.
///
/// Sub-axes allow splitting a mesh axis into multiple parts. For example,
/// an axis of size 8 could be split into two sub-axes of sizes 2 and 4.
public struct SubAxisInfo: Equatable, CustomStringConvertible {
    /// Product of sizes of all sub-axes before this one
    public let preSize: Int64

    /// Size of this sub-axis
    public let size: Int64

    public init(preSize: Int64, size: Int64) {
        self.preSize = preSize
        self.size = size
    }

    public var description: String {
        "(\(preSize))\(size)"
    }

    /// Creates an MLIR attribute for this sub-axis info.
    public func toAttribute(context: MlirContext) -> MlirAttribute {
        sdySubAxisInfoAttrGet(context, preSize, size)
    }

    /// Creates a SubAxisInfo from an MLIR attribute.
    public static func fromAttribute(_ attr: MlirAttribute) -> SubAxisInfo? {
        guard sdyAttributeIsASubAxisInfoAttr(attr) else { return nil }
        return SubAxisInfo(
            preSize: sdySubAxisInfoAttrGetPreSize(attr),
            size: sdySubAxisInfoAttrGetSize(attr)
        )
    }
}

// MARK: - DimensionSharding

/// Specifies how a single tensor dimension is sharded across mesh axes.
///
/// A dimension can be sharded along one or more mesh axes. The `isClosed`
/// flag indicates whether additional sharding is allowed during propagation.
public struct DimensionSharding: Equatable, CustomStringConvertible {
    /// The axes this dimension is sharded along (in order)
    public let axes: [AxisRef]

    /// Whether this dimension is closed to further sharding
    public let isClosed: Bool

    /// Optional priority for propagation ordering
    public let priority: Int64?

    /// Creates a dimension sharding specification.
    /// - Parameters:
    ///   - axes: The axes to shard along
    ///   - isClosed: Whether further sharding is disallowed
    ///   - priority: Optional priority for propagation
    public init(axes: [AxisRef], isClosed: Bool = false, priority: Int64? = nil) {
        self.axes = axes
        self.isClosed = isClosed
        self.priority = priority
    }

    /// Creates a dimension sharding along a single axis.
    /// - Parameter axisName: The axis name
    public init(_ axisName: String) {
        self.axes = [AxisRef(axisName)]
        self.isClosed = false
        self.priority = nil
    }

    /// Creates an unsharded (replicated) dimension.
    public static var replicated: DimensionSharding {
        DimensionSharding(axes: [], isClosed: true)
    }

    /// Creates an open dimension that can be sharded during propagation.
    public static var open: DimensionSharding {
        DimensionSharding(axes: [], isClosed: false)
    }

    public var description: String {
        if axes.isEmpty {
            return isClosed ? "{}" : "{?}"
        }
        let axesStr = axes.map { $0.description }.joined(separator: ", ")
        let closedMarker = isClosed ? "" : ", ?"
        if let p = priority {
            return "{\(axesStr)\(closedMarker)}p\(p)"
        }
        return "{\(axesStr)\(closedMarker)}"
    }

    /// Creates an MLIR attribute for this dimension sharding.
    public func toAttribute(context: MlirContext) -> MlirAttribute {
        var axisAttrs = axes.map { $0.toAttribute(context: context) }
        return axisAttrs.withUnsafeMutableBufferPointer { ptr in
            sdyDimensionShardingAttrGet(
                context,
                ptr.count,
                ptr.baseAddress,
                isClosed,
                priority ?? -1
            )
        }
    }

    /// Creates a DimensionSharding from an MLIR attribute.
    public static func fromAttribute(_ attr: MlirAttribute) -> DimensionSharding? {
        guard sdyAttributeIsADimensionShardingAttr(attr) else { return nil }

        let axesCount = sdyDimensionShardingAttrGetAxesSize(attr)
        var axes: [AxisRef] = []
        for i in 0..<axesCount {
            let axisAttr = sdyDimensionShardingAttrGetAxesElem(attr, i)
            if let axis = AxisRef.fromAttribute(axisAttr) {
                axes.append(axis)
            }
        }

        let isClosed = sdyDimensionShardingAttrGetIsClosed(attr)
        let priorityVal = sdyDimensionShardingAttrGetPriority(attr)
        let priority = priorityVal == -1 ? nil : priorityVal

        return DimensionSharding(axes: axes, isClosed: isClosed, priority: priority)
    }
}

// MARK: - TensorSharding

/// Complete sharding specification for a tensor.
///
/// Combines a mesh reference with per-dimension sharding specifications,
/// plus optional replicated and unreduced axes.
///
/// Example:
/// ```swift
/// // Shard a 2D tensor: dim 0 on "x", dim 1 on "y"
/// let sharding = TensorSharding(
///     meshName: "my_mesh",
///     dimShardings: [
///         DimensionSharding("x"),
///         DimensionSharding("y")
///     ]
/// )
/// ```
public struct TensorSharding: Equatable, CustomStringConvertible {
    /// The name of the mesh this sharding refers to
    public let meshName: String

    /// Sharding specification for each tensor dimension
    public let dimShardings: [DimensionSharding]

    /// Axes along which this tensor is replicated
    public let replicatedAxes: [AxisRef]

    /// Axes that have not been reduced (for reduce operations)
    public let unreducedAxes: [AxisRef]

    /// Creates a tensor sharding specification.
    /// - Parameters:
    ///   - meshName: The mesh name
    ///   - dimShardings: Per-dimension sharding specs
    ///   - replicatedAxes: Replicated axes
    ///   - unreducedAxes: Unreduced axes
    public init(
        meshName: String,
        dimShardings: [DimensionSharding],
        replicatedAxes: [AxisRef] = [],
        unreducedAxes: [AxisRef] = []
    ) {
        self.meshName = meshName
        self.dimShardings = dimShardings
        self.replicatedAxes = replicatedAxes
        self.unreducedAxes = unreducedAxes
    }

    /// Creates a simple sharding from axis names.
    /// - Parameters:
    ///   - meshName: The mesh name
    ///   - axisNames: Axis name for each dimension (nil for replicated)
    public init(meshName: String, axisNames: [String?]) {
        self.meshName = meshName
        self.dimShardings = axisNames.map { name in
            if let name = name {
                return DimensionSharding(name)
            } else {
                return .replicated
            }
        }
        self.replicatedAxes = []
        self.unreducedAxes = []
    }

    public var description: String {
        let dims = dimShardings.map { $0.description }.joined(separator: ", ")
        var result = "#sdy.sharding<@\(meshName), [\(dims)]"

        if !replicatedAxes.isEmpty {
            let replicated = replicatedAxes.map { $0.description }.joined(separator: ", ")
            result += ", replicated={\(replicated)}"
        }

        if !unreducedAxes.isEmpty {
            let unreduced = unreducedAxes.map { $0.description }.joined(separator: ", ")
            result += ", unreduced={\(unreduced)}"
        }

        result += ">"
        return result
    }

    /// Generates MLIR attribute text for this sharding.
    public var mlirAttributeText: String {
        description
    }

    /// Creates an MLIR TensorShardingAttr.
    /// Note: Requires the mesh attribute to be provided separately.
    public func toAttribute(context: MlirContext, meshAttr: MlirAttribute) -> MlirAttribute {
        var dimAttrs = dimShardings.map { $0.toAttribute(context: context) }
        var repAttrs = replicatedAxes.map { $0.toAttribute(context: context) }
        var unredAttrs = unreducedAxes.map { $0.toAttribute(context: context) }

        return dimAttrs.withUnsafeMutableBufferPointer { dimsPtr in
            repAttrs.withUnsafeMutableBufferPointer { repPtr in
                unredAttrs.withUnsafeMutableBufferPointer { unredPtr in
                    sdyTensorShardingAttrGet(
                        context,
                        meshAttr,
                        dimsPtr.count,
                        dimsPtr.baseAddress,
                        repPtr.count,
                        repPtr.baseAddress,
                        unredPtr.count,
                        unredPtr.baseAddress
                    )
                }
            }
        }
    }
}

// MARK: - Convenience Extensions

extension TensorSharding {
    /// Creates a fully replicated tensor sharding.
    /// - Parameters:
    ///   - meshName: The mesh name
    ///   - rank: Number of tensor dimensions
    public static func replicated(meshName: String, rank: Int) -> TensorSharding {
        TensorSharding(
            meshName: meshName,
            dimShardings: Array(repeating: .replicated, count: rank)
        )
    }

    /// Creates an open sharding that allows propagation to determine sharding.
    /// - Parameters:
    ///   - meshName: The mesh name
    ///   - rank: Number of tensor dimensions
    public static func open(meshName: String, rank: Int) -> TensorSharding {
        TensorSharding(
            meshName: meshName,
            dimShardings: Array(repeating: .open, count: rank)
        )
    }
}
