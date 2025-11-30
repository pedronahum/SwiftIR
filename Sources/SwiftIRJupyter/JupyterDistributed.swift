// JupyterDistributed.swift - Distributed Training for Jupyter/Colab
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// Provides distributed training support through:
// - Collective operations (all-reduce, all-gather, broadcast, reduce-scatter)
// - Device mesh topology management
// - Tensor sharding specifications
// - Gradient synchronization

import Foundation

// MARK: - Device Mesh

/// Represents a logical mesh of devices for distributed computation
public struct JDeviceMesh: Sendable, Hashable {
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
    public static func linear(name: String = "devices", size: Int) -> JDeviceMesh {
        JDeviceMesh(name: name, shape: [size], axisNames: ["x"])
    }

    /// Create a 2D mesh for data and model parallelism
    public static func grid(name: String = "mesh", dataParallel: Int, modelParallel: Int) -> JDeviceMesh {
        JDeviceMesh(name: name, shape: [dataParallel, modelParallel], axisNames: ["data", "model"])
    }

    /// Get the size of a named axis
    public func axisSize(_ name: String) -> Int? {
        guard let index = axisNames.firstIndex(of: name) else { return nil }
        return shape[index]
    }
}

// MARK: - Sharding Specification

/// Specifies how a tensor is sharded across a device mesh
public struct JShardingSpec: Sendable, Hashable {
    /// The device mesh to shard across
    public let mesh: JDeviceMesh

    /// Mapping from tensor dimensions to mesh axes
    /// nil means the dimension is replicated
    public let dimMapping: [String?]

    /// Initialize a sharding spec
    public init(mesh: JDeviceMesh, dimMapping: [String?]) {
        self.mesh = mesh
        self.dimMapping = dimMapping
    }

    /// Create a fully replicated sharding (no sharding)
    public static func replicated(mesh: JDeviceMesh, rank: Int) -> JShardingSpec {
        JShardingSpec(mesh: mesh, dimMapping: Array(repeating: nil, count: rank))
    }

    /// Create a sharding that shards dimension 0 across data parallel axis
    public static func dataParallel(mesh: JDeviceMesh, rank: Int) -> JShardingSpec {
        var mapping: [String?] = Array(repeating: nil, count: rank)
        if rank > 0 {
            mapping[0] = "data"
        }
        return JShardingSpec(mesh: mesh, dimMapping: mapping)
    }

    /// Check if this sharding is fully replicated
    public var isReplicated: Bool {
        dimMapping.allSatisfy { $0 == nil }
    }
}

// MARK: - Collective Operations

/// Types of collective operations
public enum JCollectiveOperation: String, Sendable {
    case allReduce = "all_reduce"
    case allGather = "all_gather"
    case reduceScatter = "reduce_scatter"
    case broadcast = "broadcast"
    case allToAll = "all_to_all"
}

/// Reduction operations for collectives
public enum JReductionType: String, Sendable {
    case sum
    case mean
    case max
    case min
    case product
}

// MARK: - Distributed Operations on JTracer

extension JTracer {
    /// All-reduce operation - reduces values across all replicas
    public func allReduce(
        reduction: JReductionType = .sum,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> JTracer {
        let groups = replicaGroups ?? [[]]

        let id = JDistributedOperationBuilder.shared.createAllReduce(
            input: self.value,
            reduction: reduction,
            replicaGroups: groups,
            channelId: channelId,
            shape: self.shape,
            dtype: self.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion()
        )
    }

    /// All-gather operation - gathers values from all replicas
    public func allGather(
        gatherDim: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> JTracer {
        let groups = replicaGroups ?? [[]]

        // Calculate output shape (gathered dimension grows)
        var outputDims = shape.dimensions
        let replicaCount = groups.first?.count ?? 1
        if gatherDim < outputDims.count {
            if let dim = outputDims[gatherDim] {
                outputDims[gatherDim] = dim * replicaCount
            }
        }
        let outputShape = JTensorShape(dimensions: outputDims)

        let id = JDistributedOperationBuilder.shared.createAllGather(
            input: self.value,
            gatherDim: gatherDim,
            replicaGroups: groups,
            channelId: channelId,
            shape: outputShape,
            dtype: self.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: outputShape,
            dtype: self.dtype,
            version: Self.incrementVersion()
        )
    }

    /// Reduce-scatter operation - reduces and scatters result
    public func reduceScatter(
        reduction: JReductionType = .sum,
        scatterDim: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> JTracer {
        let groups = replicaGroups ?? [[]]

        // Calculate output shape (scattered dimension shrinks)
        var outputDims = shape.dimensions
        let replicaCount = groups.first?.count ?? 1
        if scatterDim < outputDims.count, replicaCount > 0 {
            if let dim = outputDims[scatterDim] {
                outputDims[scatterDim] = dim / replicaCount
            }
        }
        let outputShape = JTensorShape(dimensions: outputDims)

        let id = JDistributedOperationBuilder.shared.createReduceScatter(
            input: self.value,
            reduction: reduction,
            scatterDim: scatterDim,
            replicaGroups: groups,
            channelId: channelId,
            shape: outputShape,
            dtype: self.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: outputShape,
            dtype: self.dtype,
            version: Self.incrementVersion()
        )
    }

    /// Broadcast from one replica to all others
    public func broadcast(
        rootReplica: Int = 0,
        replicaGroups: [[Int]]? = nil,
        channelId: Int? = nil
    ) -> JTracer {
        let groups = replicaGroups ?? [[]]

        let id = JDistributedOperationBuilder.shared.createBroadcast(
            input: self.value,
            rootReplica: rootReplica,
            replicaGroups: groups,
            channelId: channelId,
            shape: self.shape,
            dtype: self.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion()
        )
    }
}

// MARK: - Distributed Operation Builder

/// Records a collective operation in the trace
internal struct JCollectiveOperationRecord: Sendable {
    let id: UInt64
    let operation: JCollectiveOperation
    let inputId: UInt64
    let replicaGroups: [[Int]]
    let reduction: JReductionType?
    let shape: JTensorShape
    let dtype: JDType
}

/// Builds distributed operations for the trace
internal final class JDistributedOperationBuilder: @unchecked Sendable {
    nonisolated(unsafe) static let shared = JDistributedOperationBuilder()

    private var nextId: UInt64 = 20000  // Start high to avoid collisions
    private let lock = NSLock()

    /// Recorded collective operations
    private var operations: [JCollectiveOperationRecord] = []

    private init() {}

    func createAllReduce(
        input: JMLIRValue,
        reduction: JReductionType,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(JCollectiveOperationRecord(
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
        input: JMLIRValue,
        gatherDim: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(JCollectiveOperationRecord(
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
        input: JMLIRValue,
        reduction: JReductionType,
        scatterDim: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(JCollectiveOperationRecord(
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
        input: JMLIRValue,
        rootReplica: Int,
        replicaGroups: [[Int]],
        channelId: Int?,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(JCollectiveOperationRecord(
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
    func getOperations() -> [JCollectiveOperationRecord] {
        lock.lock()
        defer { lock.unlock() }
        return operations
    }

    /// Reset the builder
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 20000
        operations.removeAll()
    }
}

// MARK: - Gradient Synchronization

/// Synchronizes gradients across devices
public struct JGradientSynchronizer: Sendable {
    /// The device mesh to synchronize across
    public let mesh: JDeviceMesh

    /// Reduction type for gradient aggregation
    public let reduction: JReductionType

    /// Initialize a synchronizer
    public init(mesh: JDeviceMesh, reduction: JReductionType = .mean) {
        self.mesh = mesh
        self.reduction = reduction
    }

    /// Synchronize a single gradient tensor
    public func synchronize(_ gradient: JTracer) -> JTracer {
        return gradient.allReduce(reduction: reduction)
    }

    /// Synchronize multiple gradients
    public func synchronize(_ gradients: [JTracer]) -> [JTracer] {
        return gradients.map { synchronize($0) }
    }
}

// MARK: - Distributed Training Context

/// Context for distributed training setup
public final class JDistributedContext: @unchecked Sendable {
    /// Shared instance
    public nonisolated(unsafe) static let shared = JDistributedContext()

    private let lock = NSLock()

    /// Current device mesh
    public private(set) var mesh: JDeviceMesh?

    /// Current replica ID
    public private(set) var replicaId: Int = 0

    /// Total number of replicas
    public private(set) var worldSize: Int = 1

    private init() {}

    /// Initialize distributed context
    public func initialize(mesh: JDeviceMesh, replicaId: Int) {
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
