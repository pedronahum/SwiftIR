/// TensorSharding for SwiftIRShardingLite
/// Pure Swift implementation - no C dependencies.
///
/// Defines how tensor dimensions are partitioned across mesh axes.

import Foundation

// MARK: - Dimension Sharding

/// Specifies how a single tensor dimension is sharded.
public struct DimensionSharding: Equatable, Hashable {
    /// Axes that this dimension is sharded across
    public let axes: [String]

    /// Whether this dimension is "closed" (no further sharding allowed)
    public let isClosed: Bool

    /// Optional priority for propagation
    public let priority: Int?

    /// Creates a dimension sharding.
    public init(axes: [String] = [], isClosed: Bool = false, priority: Int? = nil) {
        self.axes = axes
        self.isClosed = isClosed
        self.priority = priority
    }

    /// Creates a replicated (unsharded) dimension.
    public static var replicated: DimensionSharding {
        DimensionSharding()
    }

    /// Creates a sharded dimension on the given axis.
    public static func sharded(on axis: String) -> DimensionSharding {
        DimensionSharding(axes: [axis])
    }

    /// MLIR text representation
    public var mlirText: String {
        if axes.isEmpty {
            return "{}"
        }
        let axesStr = axes.map { "\"\($0)\"" }.joined(separator: ", ")
        return "{\(axesStr)}"
    }
}

// MARK: - Tensor Sharding

/// Complete sharding specification for a tensor.
///
/// A TensorSharding describes how each dimension of a tensor maps to
/// mesh axes. Dimensions can be sharded (split across devices) or
/// replicated (each device has a full copy).
public struct TensorSharding: Equatable, Hashable, CustomStringConvertible {
    /// The mesh this sharding refers to
    public let meshName: String

    /// Sharding for each dimension
    public let dimShardings: [DimensionSharding]

    /// Axes that are replicated across all dimensions
    public let replicatedAxes: [String]

    /// Creates a tensor sharding from dimension specifications.
    public init(
        meshName: String,
        dimShardings: [DimensionSharding],
        replicatedAxes: [String] = []
    ) {
        self.meshName = meshName
        self.dimShardings = dimShardings
        self.replicatedAxes = replicatedAxes
    }

    /// Creates a tensor sharding from axis names.
    ///
    /// Each axis name corresponds to a dimension. Use nil for replicated dimensions.
    public init(meshName: String, axisNames: [String?]) {
        self.meshName = meshName
        self.dimShardings = axisNames.map { axis in
            if let axis = axis {
                return DimensionSharding.sharded(on: axis)
            } else {
                return DimensionSharding.replicated
            }
        }
        self.replicatedAxes = []
    }

    /// Creates a fully replicated sharding.
    public static func replicated(meshName: String, rank: Int) -> TensorSharding {
        TensorSharding(
            meshName: meshName,
            dimShardings: Array(repeating: .replicated, count: rank)
        )
    }

    /// The tensor rank (number of dimensions)
    public var rank: Int {
        dimShardings.count
    }

    /// MLIR attribute text representation
    public var mlirAttributeText: String {
        let dimStr = dimShardings.map(\.mlirText).joined(separator: ", ")
        return "#sdy.sharding<@\(meshName), [\(dimStr)]>"
    }

    public var description: String {
        mlirAttributeText
    }
}
