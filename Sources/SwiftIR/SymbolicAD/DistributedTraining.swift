/// Distributed Training - Collective operations, device mesh, and sharding
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides distributed training support through:
/// - Collective operations (all-reduce, all-gather, broadcast, reduce-scatter)
/// - Device mesh topology management
/// - Tensor sharding specifications
/// - Gradient synchronization

import Foundation

// MARK: - Device Mesh

/// Represents a logical mesh of devices for distributed computation
public struct DeviceMesh: Sendable, Hashable {
    /// Name of this mesh
    public let name: String

    /// Shape of the device mesh (e.g., [2, 4] for 8 devices in 2x4 grid)
    public let shape: [Int]

    /// Axis names for the mesh dimensions
    public let axisNames: [String]

    /// Total number of devices
    public var deviceCount: Int {
        shape.reduce(1, *)
    }

    /// Initialize a device mesh
    public init(name: String, shape: [Int], axisNames: [String]) {
        precondition(shape.count == axisNames.count, "Shape and axis names must match")
        self.name = name
        self.shape = shape
        self.axisNames = axisNames
    }

    /// Create a 1D mesh
    public static func linear(name: String = "devices", size: Int) -> DeviceMesh {
        DeviceMesh(name: name, shape: [size], axisNames: ["x"])
    }

    /// Create a 2D mesh for data and model parallelism
    public static func grid(name: String = "mesh", dataParallel: Int, modelParallel: Int) -> DeviceMesh {
        DeviceMesh(name: name, shape: [dataParallel, modelParallel], axisNames: ["data", "model"])
    }

    /// Get the size of a named axis
    public func axisSize(_ name: String) -> Int? {
        guard let index = axisNames.firstIndex(of: name) else { return nil }
        return shape[index]
    }
}

// MARK: - Sharding Specification

/// Specifies how a tensor is sharded across a device mesh
public struct ShardingSpec: Sendable, Hashable {
    /// The device mesh to shard across
    public let mesh: DeviceMesh

    /// Mapping from tensor dimensions to mesh axes
    /// nil means the dimension is replicated
    public let dimMapping: [String?]

    /// Initialize a sharding spec
    public init(mesh: DeviceMesh, dimMapping: [String?]) {
        self.mesh = mesh
        self.dimMapping = dimMapping
    }

    /// Create a fully replicated sharding (no sharding)
    public static func replicated(mesh: DeviceMesh, rank: Int) -> ShardingSpec {
        ShardingSpec(mesh: mesh, dimMapping: Array(repeating: nil, count: rank))
    }

    /// Create a sharding that shards dimension 0 across data parallel axis
    public static func dataParallel(mesh: DeviceMesh, rank: Int) -> ShardingSpec {
        var mapping: [String?] = Array(repeating: nil, count: rank)
        if rank > 0 {
            mapping[0] = "data"
        }
        return ShardingSpec(mesh: mesh, dimMapping: mapping)
    }

    /// Check if this sharding is fully replicated
    public var isReplicated: Bool {
        dimMapping.allSatisfy { $0 == nil }
    }
}

// MARK: - Collective Operations

/// Types of collective operations
public enum CollectiveOperation: String, Sendable {
    case allReduce = "all_reduce"
    case allGather = "all_gather"
    case reduceScatter = "reduce_scatter"
    case broadcast = "broadcast"
    case allToAll = "all_to_all"
}

/// Reduction operations for collectives
public enum ReductionType: String, Sendable {
    case sum
    case mean
    case max
    case min
    case product
}

/// Records a collective operation in the trace
internal struct CollectiveOperationRecord: Sendable {
    let id: UInt64
    let operation: CollectiveOperation
    let inputId: UInt64
    let replicaGroups: [[Int]]
    let reduction: ReductionType?
    let shape: TensorShape
    let dtype: DType
}

// MARK: - Distributed Operations on Tracer

extension Tracer {
    /// All-reduce operation - reduces values across all replicas
    public func allReduce(
        reduction: ReductionType = .sum,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> Tracer {
        let groups = replicaGroups ?? [[]]  // Empty means all devices

        let id = DistributedOperationBuilder.shared.createAllReduce(
            input: self.value,
            reduction: reduction,
            replicaGroups: groups,
            channelId: channelId,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "all_reduce")
        )
    }

    /// All-gather operation - gathers values from all replicas
    public func allGather(
        gatherDim: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> Tracer {
        let groups = replicaGroups ?? [[]]

        // Calculate output shape (gathered dimension grows)
        var outputDims = shape.dimensions
        let replicaCount = groups.first?.count ?? 1
        if gatherDim < outputDims.count {
            if let dim = outputDims[gatherDim] {
                outputDims[gatherDim] = dim * replicaCount
            }
        }
        let outputShape = TensorShape(dimensions: outputDims)

        let id = DistributedOperationBuilder.shared.createAllGather(
            input: self.value,
            gatherDim: gatherDim,
            replicaGroups: groups,
            channelId: channelId,
            shape: outputShape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: outputShape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "all_gather")
        )
    }

    /// Reduce-scatter operation - reduces and scatters result
    public func reduceScatter(
        reduction: ReductionType = .sum,
        scatterDim: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> Tracer {
        let groups = replicaGroups ?? [[]]

        // Calculate output shape (scattered dimension shrinks)
        var outputDims = shape.dimensions
        let replicaCount = groups.first?.count ?? 1
        if scatterDim < outputDims.count, replicaCount > 0 {
            if let dim = outputDims[scatterDim] {
                outputDims[scatterDim] = dim / replicaCount
            }
        }
        let outputShape = TensorShape(dimensions: outputDims)

        let id = DistributedOperationBuilder.shared.createReduceScatter(
            input: self.value,
            reduction: reduction,
            scatterDim: scatterDim,
            replicaGroups: groups,
            channelId: channelId,
            shape: outputShape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: outputShape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "reduce_scatter")
        )
    }

    /// Broadcast from one replica to all others
    public func broadcast(
        rootReplica: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> Tracer {
        let groups = replicaGroups ?? [[]]

        let id = DistributedOperationBuilder.shared.createBroadcast(
            input: self.value,
            rootReplica: rootReplica,
            replicaGroups: groups,
            channelId: channelId,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "broadcast")
        )
    }
}

// MARK: - Distributed Operation Builder

/// Builds distributed operations for the trace
internal final class DistributedOperationBuilder: @unchecked Sendable {
    static let shared = DistributedOperationBuilder()

    private var nextId: UInt64 = 1
    private let lock = NSLock()

    /// Recorded collective operations
    private var operations: [CollectiveOperationRecord] = []

    private init() {}

    func createAllReduce(
        input: MLIRValue,
        reduction: ReductionType,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: TensorShape,
        dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(CollectiveOperationRecord(
            id: id,
            operation: .allReduce,
            inputId: input.id,
            replicaGroups: replicaGroups,
            reduction: reduction,
            shape: shape,
            dtype: dtype
        ))

        return id
    }

    func createAllGather(
        input: MLIRValue,
        gatherDim: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: TensorShape,
        dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(CollectiveOperationRecord(
            id: id,
            operation: .allGather,
            inputId: input.id,
            replicaGroups: replicaGroups,
            reduction: nil,
            shape: shape,
            dtype: dtype
        ))

        return id
    }

    func createReduceScatter(
        input: MLIRValue,
        reduction: ReductionType,
        scatterDim: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: TensorShape,
        dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(CollectiveOperationRecord(
            id: id,
            operation: .reduceScatter,
            inputId: input.id,
            replicaGroups: replicaGroups,
            reduction: reduction,
            shape: shape,
            dtype: dtype
        ))

        return id
    }

    func createBroadcast(
        input: MLIRValue,
        rootReplica: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: TensorShape,
        dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(CollectiveOperationRecord(
            id: id,
            operation: .broadcast,
            inputId: input.id,
            replicaGroups: replicaGroups,
            reduction: nil,
            shape: shape,
            dtype: dtype
        ))

        return id
    }

    /// Get all recorded operations
    func getOperations() -> [CollectiveOperationRecord] {
        lock.lock()
        defer { lock.unlock() }
        return operations
    }

    /// Reset the builder
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 1
        operations.removeAll()
    }
}

// MARK: - Gradient Synchronization

/// Synchronizes gradients across devices
public struct GradientSynchronizer: Sendable {
    /// The device mesh to synchronize across
    public let mesh: DeviceMesh

    /// Reduction type for gradient aggregation
    public let reduction: ReductionType

    /// Initialize a synchronizer
    public init(mesh: DeviceMesh, reduction: ReductionType = .mean) {
        self.mesh = mesh
        self.reduction = reduction
    }

    /// Synchronize a single gradient tensor
    public func synchronize(_ gradient: Tracer) -> Tracer {
        return gradient.allReduce(reduction: reduction)
    }

    /// Synchronize multiple gradients
    public func synchronize(_ gradients: [Tracer]) -> [Tracer] {
        return gradients.map { synchronize($0) }
    }
}

// MARK: - Distributed Training Context

/// Context for distributed training setup
public final class DistributedContext: @unchecked Sendable {
    /// Shared instance
    public static let shared = DistributedContext()

    private let lock = NSLock()

    /// Current device mesh
    public private(set) var mesh: DeviceMesh?

    /// Current replica ID
    public private(set) var replicaId: Int = 0

    /// Total number of replicas
    public private(set) var worldSize: Int = 1

    private init() {}

    /// Initialize distributed context
    public func initialize(mesh: DeviceMesh, replicaId: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.mesh = mesh
        self.replicaId = replicaId
        self.worldSize = mesh.deviceCount
    }

    /// Check if running in distributed mode
    public var isDistributed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return worldSize > 1
    }

    /// Reset distributed context
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        mesh = nil
        replicaId = 0
        worldSize = 1
    }
}
