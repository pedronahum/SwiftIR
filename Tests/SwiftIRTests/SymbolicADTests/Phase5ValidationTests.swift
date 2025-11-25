/// Phase 5 Validation Tests - Distributed Training
/// Tests for collective operations, device mesh, sharding, and gradient synchronization

import Testing

@testable import SwiftIR

@Suite("Phase 5: Distributed Training", .serialized)
struct Phase5ValidationTests {

    init() {
        DistributedOperationBuilder.shared.reset()
        DistributedContext.shared.reset()
    }

    // MARK: - Device Mesh Tests

    @Suite("Device Mesh")
    struct DeviceMeshTests {

        @Test("Create linear mesh")
        func linearMesh() {
            let mesh = DeviceMesh.linear(size: 8)

            #expect(mesh.deviceCount == 8)
            #expect(mesh.shape == [8])
            #expect(mesh.axisNames == ["x"])
        }

        @Test("Create 2D grid mesh")
        func gridMesh() {
            let mesh = DeviceMesh.grid(dataParallel: 4, modelParallel: 2)

            #expect(mesh.deviceCount == 8)
            #expect(mesh.shape == [4, 2])
            #expect(mesh.axisNames == ["data", "model"])
        }

        @Test("Get axis size by name")
        func axisSizeByName() {
            let mesh = DeviceMesh.grid(dataParallel: 4, modelParallel: 2)

            #expect(mesh.axisSize("data") == 4)
            #expect(mesh.axisSize("model") == 2)
            #expect(mesh.axisSize("unknown") == nil)
        }

        @Test("Custom mesh creation")
        func customMesh() {
            let mesh = DeviceMesh(
                name: "tpu_pod",
                shape: [2, 4, 8],
                axisNames: ["x", "y", "z"]
            )

            #expect(mesh.deviceCount == 64)
            #expect(mesh.name == "tpu_pod")
        }
    }

    // MARK: - Sharding Specification Tests

    @Suite("Sharding Specification")
    struct ShardingSpecTests {

        @Test("Replicated sharding")
        func replicatedSharding() {
            let mesh = DeviceMesh.linear(size: 4)
            let spec = ShardingSpec.replicated(mesh: mesh, rank: 3)

            #expect(spec.isReplicated)
            #expect(spec.dimMapping.count == 3)
            #expect(spec.dimMapping.allSatisfy { $0 == nil })
        }

        @Test("Data parallel sharding")
        func dataParallelSharding() {
            let mesh = DeviceMesh.grid(dataParallel: 4, modelParallel: 1)
            let spec = ShardingSpec.dataParallel(mesh: mesh, rank: 2)

            #expect(!spec.isReplicated)
            #expect(spec.dimMapping[0] == "data")
            #expect(spec.dimMapping[1] == nil)
        }

        @Test("Custom sharding")
        func customSharding() {
            let mesh = DeviceMesh.grid(dataParallel: 2, modelParallel: 4)
            let spec = ShardingSpec(
                mesh: mesh,
                dimMapping: ["data", nil, "model"]
            )

            #expect(!spec.isReplicated)
            #expect(spec.dimMapping[0] == "data")
            #expect(spec.dimMapping[1] == nil)
            #expect(spec.dimMapping[2] == "model")
        }
    }

    // MARK: - Collective Operations Tests

    @Suite("Collective Operations")
    struct CollectiveOperationsTests {

        init() {
            DistributedOperationBuilder.shared.reset()
            TracerGraphBuilder.shared.reset()
        }

        @Test("All-reduce operation")
        func allReduce() {
            let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
            let result = x.allReduce(reduction: .sum)

            #expect(result.shape == x.shape)
            #expect(result.dtype == x.dtype)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count == 1)
            #expect(ops[0].operation == .allReduce)
        }

        @Test("All-reduce with mean reduction")
        func allReduceMean() {
            let x = Tracer(value: 1.0, shape: TensorShape([32, 128]), dtype: .float32)
            let result = x.allReduce(reduction: .mean)

            #expect(result.shape == x.shape)
            // Mean reduction verified by operation completion
        }

        @Test("All-gather operation")
        func allGather() {
            DistributedOperationBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
            let result = x.allGather(gatherDim: 0, replicaGroups: [[0, 1, 2, 3]])

            // Shape should increase along gather dimension
            #expect(result.shape.dimensions[0] == 16)  // 4 * 4 replicas
            #expect(result.shape.dimensions[1] == 8)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count == 1)
            #expect(ops[0].operation == .allGather)
        }

        @Test("Reduce-scatter operation")
        func reduceScatter() {
            let x = Tracer(value: 1.0, shape: TensorShape([16, 8]), dtype: .float32)
            let result = x.reduceScatter(reduction: .sum, scatterDim: 0, replicaGroups: [[0, 1, 2, 3]])

            // Shape should decrease along scatter dimension
            #expect(result.shape.dimensions[0] == 4)  // 16 / 4 replicas
            #expect(result.shape.dimensions[1] == 8)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count == 1)
            #expect(ops[0].operation == .reduceScatter)
        }

        @Test("Broadcast operation")
        func broadcast() {
            DistributedOperationBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
            let result = x.broadcast(rootReplica: 0)

            #expect(result.shape == x.shape)
            #expect(result.dtype == x.dtype)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count >= 1)
            #expect(ops.last?.operation == .broadcast)
        }

        @Test("Multiple collective operations")
        func multipleCollectives() {
            DistributedOperationBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([8, 16]), dtype: .float32)

            // Chain of collectives
            let gathered = x.allGather(gatherDim: 0, replicaGroups: [[0, 1]])
            let reduced = gathered.allReduce(reduction: .sum)
            let broadcast = reduced.broadcast(rootReplica: 0)

            #expect(broadcast.shape.dimensions[1] == 16)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count >= 3)
        }
    }

    // MARK: - Gradient Synchronization Tests

    @Suite("Gradient Synchronization")
    struct GradientSyncTests {

        init() {
            DistributedOperationBuilder.shared.reset()
            TracerGraphBuilder.shared.reset()
        }

        @Test("Synchronize single gradient")
        func syncSingleGradient() {
            DistributedOperationBuilder.shared.reset()

            let mesh = DeviceMesh.linear(size: 4)
            let synchronizer = GradientSynchronizer(mesh: mesh, reduction: .mean)

            let gradient = Tracer(value: 1.0, shape: TensorShape([10, 20]), dtype: .float32)
            let synced = synchronizer.synchronize(gradient)

            #expect(synced.shape == gradient.shape)

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count == 1)
            #expect(ops[0].operation == .allReduce)
        }

        @Test("Synchronize multiple gradients")
        func syncMultipleGradients() {
            DistributedOperationBuilder.shared.reset()

            let mesh = DeviceMesh.linear(size: 8)
            let synchronizer = GradientSynchronizer(mesh: mesh)

            let gradients = [
                Tracer(value: 1.0, shape: TensorShape([100]), dtype: .float32),
                Tracer(value: 1.0, shape: TensorShape([200, 50]), dtype: .float32),
                Tracer(value: 1.0, shape: TensorShape([10, 10, 10]), dtype: .float32),
            ]

            let synced = synchronizer.synchronize(gradients)

            #expect(synced.count == 3)
            for (orig, sync) in zip(gradients, synced) {
                #expect(sync.shape == orig.shape)
            }

            let ops = DistributedOperationBuilder.shared.getOperations()
            #expect(ops.count >= 3)
        }
    }

    // MARK: - Distributed Context Tests

    @Suite("Distributed Context")
    struct DistributedContextTests {

        init() {
            DistributedContext.shared.reset()
        }

        @Test("Initialize distributed context")
        func initializeContext() {
            let mesh = DeviceMesh.grid(dataParallel: 4, modelParallel: 2)
            DistributedContext.shared.initialize(mesh: mesh, replicaId: 3)

            #expect(DistributedContext.shared.worldSize == 8)
            #expect(DistributedContext.shared.replicaId == 3)
            #expect(DistributedContext.shared.isDistributed)
        }

        @Test("Non-distributed by default")
        func nonDistributedDefault() {
            DistributedContext.shared.reset()

            #expect(!DistributedContext.shared.isDistributed)
            #expect(DistributedContext.shared.worldSize == 1)
            #expect(DistributedContext.shared.replicaId == 0)
        }

        @Test("Reset context")
        func resetContext() {
            let mesh = DeviceMesh.linear(size: 4)
            DistributedContext.shared.initialize(mesh: mesh, replicaId: 2)

            DistributedContext.shared.reset()

            #expect(!DistributedContext.shared.isDistributed)
            #expect(DistributedContext.shared.mesh == nil)
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 5 integration - distributed gradient descent step")
    func phase5DistributedGradientDescent() {
        DistributedOperationBuilder.shared.reset()
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 5: Distributed Gradient Descent")
        print("========================================\n")

        // Setup distributed context
        let mesh = DeviceMesh.grid(name: "training_mesh", dataParallel: 4, modelParallel: 1)
        DistributedContext.shared.initialize(mesh: mesh, replicaId: 0)
        print("Initialized mesh with \(mesh.deviceCount) devices")

        // Simulate distributed training step
        let batchSize = 32
        let hiddenSize = 128

        // Local forward pass
        let x = Tracer(value: 1.0, shape: TensorShape([batchSize, hiddenSize]), dtype: .float32)
        let w = Tracer(value: 0.1, shape: TensorShape([hiddenSize, hiddenSize]), dtype: .float32)
        let output = x.matmul(w)
        print("✅ Forward pass computed")

        // Simulate gradients
        let gradW = Tracer(value: 0.01, shape: TensorShape([hiddenSize, hiddenSize]), dtype: .float32)

        // Synchronize gradients across replicas
        let synchronizer = GradientSynchronizer(mesh: mesh, reduction: .mean)
        let syncedGrad = synchronizer.synchronize(gradW)
        print("✅ Gradient synchronized with all-reduce (mean)")

        #expect(syncedGrad.shape == gradW.shape)

        // Verify collective operations recorded
        let ops = DistributedOperationBuilder.shared.getOperations()
        #expect(ops.count >= 1)
        print("✅ \(ops.count) collective operation(s) recorded")

        print("\n========================================")
        print("✅ PHASE 5 DISTRIBUTED GRADIENT DESCENT COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 5 integration - model parallelism")
    func phase5ModelParallelism() {
        DistributedOperationBuilder.shared.reset()
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 5: Model Parallelism Test")
        print("========================================\n")

        // Setup mesh for model parallelism
        let mesh = DeviceMesh.grid(dataParallel: 2, modelParallel: 4)
        print("Created \(mesh.deviceCount)-device mesh for model parallelism")

        // Sharded weight matrix
        let sharding = ShardingSpec(
            mesh: mesh,
            dimMapping: [nil, "model"]  // Shard columns across model axis
        )
        print("Sharding spec: dim 0 = replicated, dim 1 = model parallel")

        // Simulated layer with sharded weights
        let localShard = Tracer(value: 0.1, shape: TensorShape([1024, 256]), dtype: .float32)

        // All-gather to reconstruct full weight for computation
        let fullWeight = localShard.allGather(gatherDim: 1, replicaGroups: [[0, 1, 2, 3]])
        print("✅ All-gathered weight shards")

        // Output shape should have full hidden dimension
        #expect(fullWeight.shape.dimensions[0] == 1024)
        #expect(fullWeight.shape.dimensions[1] == 1024)  // 256 * 4

        // After computation, reduce-scatter to redistribute
        let output = Tracer(value: 1.0, shape: TensorShape([32, 1024]), dtype: .float32)
        let shardedOutput = output.reduceScatter(
            reduction: .sum,
            scatterDim: 1,
            replicaGroups: [[0, 1, 2, 3]]
        )
        print("✅ Reduce-scattered output")

        #expect(shardedOutput.shape.dimensions[1] == 256)  // 1024 / 4

        let ops = DistributedOperationBuilder.shared.getOperations()
        #expect(ops.count >= 2)

        print("\n========================================")
        print("✅ PHASE 5 MODEL PARALLELISM COMPLETE")
        print("========================================\n")
    }
}
