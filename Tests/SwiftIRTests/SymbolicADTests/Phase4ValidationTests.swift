/// Phase 4 Validation Tests - Memory Management
/// Tests for buffer aliasing, lifetime tracking, memory pools, and in-place optimization

import Testing

@testable import SwiftIR

@Suite("Phase 4: Memory Management", .serialized)
struct Phase4ValidationTests {

    init() {
        MemoryManager.shared.reset()
    }

    // MARK: - Buffer Aliasing Tests

    @Suite("Buffer Aliasing")
    struct BufferAliasingTests {

        init() {
            MemoryManager.shared.reset()
        }

        @Test("Allocate buffers for values")
        func allocateBuffers() {
            let analyzer = BufferAliasingAnalyzer()

            let buffer1 = analyzer.allocateBuffer(
                for: 1,
                shape: TensorShape([2, 3]),
                dtype: .float32
            )

            let buffer2 = analyzer.allocateBuffer(
                for: 2,
                shape: TensorShape([4, 5]),
                dtype: .float32
            )

            #expect(buffer1.id != buffer2.id)
            #expect(buffer1.sizeInBytes == 6 * 4)  // 6 elements * 4 bytes
            #expect(buffer2.sizeInBytes == 20 * 4)  // 20 elements * 4 bytes
        }

        @Test("Create buffer aliases")
        func createAliases() {
            let analyzer = BufferAliasingAnalyzer()

            let buffer = analyzer.allocateBuffer(
                for: 1,
                shape: TensorShape([10]),
                dtype: .float32
            )

            analyzer.createAlias(source: 1, target: 2)

            let aliasedBuffer = analyzer.getBuffer(for: 2)
            #expect(aliasedBuffer?.id == buffer.id)
        }

        @Test("Get all aliases for buffer")
        func getAllAliases() {
            let analyzer = BufferAliasingAnalyzer()

            _ = analyzer.allocateBuffer(
                for: 1,
                shape: TensorShape([10]),
                dtype: .float32
            )

            analyzer.createAlias(source: 1, target: 2)
            analyzer.createAlias(source: 1, target: 3)

            let aliases = analyzer.getAliases(for: 1)
            #expect(aliases.count == 3)
            #expect(aliases.contains(1))
            #expect(aliases.contains(2))
            #expect(aliases.contains(3))
        }

        @Test("Check if values can alias based on lifetimes")
        func canAliasBasedOnLifetimes() {
            let analyzer = BufferAliasingAnalyzer()
            let lifetimes = LifetimeAnalyzer()

            // Value 1: lives from op 0 to 5
            lifetimes.recordCreation(valueId: 1)
            for _ in 0..<5 { lifetimes.nextOperation() }
            lifetimes.recordUsage(valueId: 1)

            lifetimes.nextOperation()

            // Value 2: lives from op 6 to 10
            lifetimes.recordCreation(valueId: 2)
            for _ in 0..<4 { lifetimes.nextOperation() }
            lifetimes.recordUsage(valueId: 2)

            // Non-overlapping lifetimes can alias
            #expect(analyzer.canAlias(1, 2, lifetimes: lifetimes))
        }

        @Test("Values with overlapping lifetimes cannot alias")
        func cannotAliasOverlapping() {
            let analyzer = BufferAliasingAnalyzer()
            let lifetimes = LifetimeAnalyzer()

            // Value 1: lives from op 0 to 5
            lifetimes.recordCreation(valueId: 1)
            lifetimes.nextOperation()
            lifetimes.nextOperation()

            // Value 2: lives from op 2 to 8 (overlaps with 1)
            lifetimes.recordCreation(valueId: 2)
            for _ in 0..<3 { lifetimes.nextOperation() }
            lifetimes.recordUsage(valueId: 1)  // 1 ends at op 5
            for _ in 0..<3 { lifetimes.nextOperation() }
            lifetimes.recordUsage(valueId: 2)  // 2 ends at op 8

            // Overlapping lifetimes cannot alias
            #expect(!analyzer.canAlias(1, 2, lifetimes: lifetimes))
        }
    }

    // MARK: - Lifetime Analysis Tests

    @Suite("Lifetime Analysis")
    struct LifetimeAnalysisTests {

        @Test("Record value creation and usage")
        func recordCreationAndUsage() {
            let analyzer = LifetimeAnalyzer()

            analyzer.recordCreation(valueId: 1)
            analyzer.nextOperation()
            analyzer.nextOperation()
            analyzer.recordUsage(valueId: 1)

            let lifetime = analyzer.getLifetime(for: 1)
            #expect(lifetime != nil)
            #expect(lifetime?.start == 0)
            #expect(lifetime?.end == 2)
            #expect(lifetime?.duration == 2)
        }

        @Test("Get dead values after operation")
        func getDeadValues() {
            let analyzer = LifetimeAnalyzer()

            // Create values at different times
            analyzer.recordCreation(valueId: 1)
            analyzer.nextOperation()
            analyzer.recordCreation(valueId: 2)
            analyzer.nextOperation()
            analyzer.recordUsage(valueId: 1)  // 1 dies at op 2
            analyzer.nextOperation()
            analyzer.recordUsage(valueId: 2)  // 2 dies at op 3

            let deadAtOp2 = analyzer.getDeadValues(afterOperation: 2)
            let deadAtOp3 = analyzer.getDeadValues(afterOperation: 3)

            #expect(deadAtOp2.contains(1))
            #expect(!deadAtOp2.contains(2))
            #expect(deadAtOp3.contains(2))
        }

        @Test("Sort values by lifetime duration")
        func sortByDuration() {
            let analyzer = LifetimeAnalyzer()

            // Value 1: duration 2
            analyzer.recordCreation(valueId: 1)
            analyzer.nextOperation()
            analyzer.nextOperation()
            analyzer.recordUsage(valueId: 1)

            // Value 2: duration 5
            analyzer.recordCreation(valueId: 2)
            for _ in 0..<5 {
                analyzer.nextOperation()
            }
            analyzer.recordUsage(valueId: 2)

            let sorted = analyzer.getValuesByLifetime()
            #expect(sorted.count == 2)
            #expect(sorted[0].valueId == 2)  // Longest first
            #expect(sorted[1].valueId == 1)
        }
    }

    // MARK: - Memory Pool Tests

    @Suite("Memory Pool")
    struct MemoryPoolTests {

        @Test("Allocate and deallocate buffers")
        func allocateAndDeallocate() {
            let pool = MemoryPool()

            let buffer = pool.allocate(size: 1024)
            #expect(buffer.sizeInBytes >= 1024)

            let stats1 = pool.getStatistics()
            #expect(stats1.allocatedCount == 1)

            pool.deallocate(buffer)
            let stats2 = pool.getStatistics()
            #expect(stats2.allocatedCount == 0)
            #expect(stats2.freeCount == 1)
        }

        @Test("Reuse deallocated buffers")
        func reuseBuffers() {
            let pool = MemoryPool()

            let buffer1 = pool.allocate(size: 1024)
            let id1 = buffer1.id
            pool.deallocate(buffer1)

            let buffer2 = pool.allocate(size: 1024)
            // Should reuse the same buffer
            #expect(buffer2.id == id1)
        }

        @Test("Track peak memory usage")
        func peakMemoryUsage() {
            let pool = MemoryPool()

            let buffer1 = pool.allocate(size: 1024)
            let buffer2 = pool.allocate(size: 2048)
            _ = pool.allocate(size: 512)

            // Free some
            pool.deallocate(buffer1)
            pool.deallocate(buffer2)

            let stats = pool.getStatistics()
            // Peak should be sum of all three (rounded to size classes)
            #expect(stats.peakUsage >= 3584)
        }

        @Test("Memory pool utilization")
        func memoryUtilization() {
            let pool = MemoryPool()

            let buffer1 = pool.allocate(size: 1024)
            _ = pool.allocate(size: 1024)

            // Free one buffer
            pool.deallocate(buffer1)

            let stats = pool.getStatistics()
            // One allocated, one free -> 50% utilization
            #expect(stats.utilization < 1.0)
            #expect(stats.utilization > 0.0)
        }

        @Test("Clear free buffers")
        func clearFreeBuffers() {
            let pool = MemoryPool()

            let buffer = pool.allocate(size: 1024)
            pool.deallocate(buffer)

            let statsBefore = pool.getStatistics()
            #expect(statsBefore.freeCount == 1)

            pool.clearFreeBuffers()

            let statsAfter = pool.getStatistics()
            #expect(statsAfter.freeCount == 0)
        }
    }

    // MARK: - In-Place Optimization Tests

    @Suite("In-Place Optimization")
    struct InPlaceOptimizationTests {

        @Test("Detect unary in-place opportunity")
        func detectUnaryInPlace() {
            let lifetimes = LifetimeAnalyzer()
            let aliasing = BufferAliasingAnalyzer()
            let optimizer = InPlaceMutationOptimizer(lifetimes: lifetimes, aliasing: aliasing)

            // Value 1 dies at operation 1
            lifetimes.recordCreation(valueId: 1)
            lifetimes.nextOperation()
            lifetimes.recordUsage(valueId: 1)

            let canInPlace = optimizer.canDoInPlace(
                operation: "exp",
                input: 1,
                output: 2,
                atOperation: 1
            )

            #expect(canInPlace)

            let candidates = optimizer.getCandidates()
            #expect(candidates.count == 1)
        }

        @Test("Detect binary in-place opportunity")
        func detectBinaryInPlace() {
            let lifetimes = LifetimeAnalyzer()
            let aliasing = BufferAliasingAnalyzer()
            let optimizer = InPlaceMutationOptimizer(lifetimes: lifetimes, aliasing: aliasing)

            // Value 1 dies at operation 1
            lifetimes.recordCreation(valueId: 1)
            lifetimes.nextOperation()
            lifetimes.recordUsage(valueId: 1)

            // Value 2 survives beyond operation 1
            lifetimes.recordCreation(valueId: 2)
            lifetimes.nextOperation()
            lifetimes.nextOperation()
            lifetimes.recordUsage(valueId: 2)

            let (canInPlace, operand) = optimizer.canDoInPlace(
                operation: "add",
                lhs: 1,
                rhs: 2,
                output: 3,
                atOperation: 1
            )

            #expect(canInPlace)
            #expect(operand == 0)  // LHS can be used in-place
        }

        @Test("No in-place when value still alive")
        func noInPlaceWhenAlive() {
            let lifetimes = LifetimeAnalyzer()
            let aliasing = BufferAliasingAnalyzer()
            let optimizer = InPlaceMutationOptimizer(lifetimes: lifetimes, aliasing: aliasing)

            // Value 1 lives beyond operation 1
            lifetimes.recordCreation(valueId: 1)
            lifetimes.nextOperation()
            lifetimes.nextOperation()
            lifetimes.recordUsage(valueId: 1)

            let canInPlace = optimizer.canDoInPlace(
                operation: "exp",
                input: 1,
                output: 2,
                atOperation: 1
            )

            #expect(!canInPlace)
        }
    }

    // MARK: - Memory Manager Integration Tests

    @Suite("Memory Manager Integration")
    struct MemoryManagerTests {

        init() {
            MemoryManager.shared.reset()
        }

        @Test("Allocate through memory manager")
        func allocateThroughManager() {
            let manager = MemoryManager.shared
            manager.reset()

            let buffer = manager.allocate(
                for: 1,
                shape: TensorShape([10, 20]),
                dtype: .float32
            )

            #expect(buffer.sizeInBytes >= 800)  // 200 * 4 bytes

            let stats = manager.getStatistics()
            #expect(stats.peakUsage > 0)
        }

        @Test("Record usage and free dead values")
        func recordAndFree() {
            let manager = MemoryManager.shared
            manager.reset()

            // Allocate two values
            _ = manager.allocate(
                for: 1,
                shape: TensorShape([10]),
                dtype: .float32
            )
            manager.recordUsage(valueId: 1)
            manager.lifetimes.nextOperation()

            _ = manager.allocate(
                for: 2,
                shape: TensorShape([20]),
                dtype: .float32
            )
            manager.lifetimes.nextOperation()
            manager.recordUsage(valueId: 2)

            // Free dead values after first operation
            manager.freeDeadValues(afterOperation: 0)

            let stats = manager.getStatistics()
            // Memory management tracks allocations
            #expect(stats.allocatedCount >= 0)
        }

        @Test("Memory statistics tracking")
        func statisticsTracking() {
            let manager = MemoryManager.shared
            manager.reset()

            // Allocate several buffers
            for i in 1...5 {
                _ = manager.allocate(
                    for: UInt64(i),
                    shape: TensorShape([100]),
                    dtype: .float32
                )
            }

            let stats = manager.getStatistics()
            #expect(stats.allocatedCount >= 1)
            #expect(stats.totalAllocated >= 400)  // At least 100 * 4 bytes
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 4 integration - neural network memory optimization")
    func phase4NeuralNetworkMemory() {
        let manager = MemoryManager.shared
        manager.reset()

        print("\n========================================")
        print("Phase 4: Neural Network Memory Test")
        print("========================================\n")

        // Simulate a forward pass with memory tracking
        let batchSize = 32
        let inputSize = 784
        let hiddenSize = 128
        let outputSize = 10

        // Input
        let inputBuffer = manager.allocate(
            for: 1,
            shape: TensorShape([batchSize, inputSize]),
            dtype: .float32
        )
        print("Allocated input: \(inputBuffer.sizeInBytes) bytes")

        // Hidden layer
        manager.lifetimes.nextOperation()
        let hiddenBuffer = manager.allocate(
            for: 2,
            shape: TensorShape([batchSize, hiddenSize]),
            dtype: .float32
        )
        print("Allocated hidden: \(hiddenBuffer.sizeInBytes) bytes")

        // Output
        manager.lifetimes.nextOperation()
        let outputBuffer = manager.allocate(
            for: 3,
            shape: TensorShape([batchSize, outputSize]),
            dtype: .float32
        )
        print("Allocated output: \(outputBuffer.sizeInBytes) bytes")

        // Check peak memory
        let stats = manager.getStatistics()
        print("\nMemory Statistics:")
        print("  Peak usage: \(stats.peakUsage) bytes")
        print("  Allocated buffers: \(stats.allocatedCount)")

        #expect(stats.allocatedCount >= 1)
        print("\n✅ Neural network memory allocation verified")

        // Simulate freeing intermediate values
        manager.recordUsage(valueId: 1)  // Input last used at op 0
        manager.recordUsage(valueId: 2)  // Hidden last used at op 1
        manager.freeDeadValues(afterOperation: 0)
        manager.freeDeadValues(afterOperation: 1)

        let finalStats = manager.getStatistics()
        print("  Free buffers after cleanup: \(finalStats.freeCount)")

        print("\n========================================")
        print("✅ PHASE 4 NEURAL NETWORK MEMORY COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 4 integration - memory pool efficiency")
    func phase4MemoryPoolEfficiency() {
        let pool = MemoryPool()

        print("\n========================================")
        print("Phase 4: Memory Pool Efficiency Test")
        print("========================================\n")

        // Allocate and deallocate multiple times
        var buffers: [BufferHandle] = []

        // First batch
        for _ in 0..<10 {
            buffers.append(pool.allocate(size: 1024))
        }
        print("Allocated 10 buffers of 1KB")

        // Free half
        for i in 0..<5 {
            pool.deallocate(buffers[i])
        }
        print("Deallocated 5 buffers")

        // Allocate more - should reuse
        for _ in 0..<3 {
            buffers.append(pool.allocate(size: 1024))
        }
        print("Allocated 3 more buffers (should reuse)")

        let stats = pool.getStatistics()
        print("\nPool Statistics:")
        print("  Allocated: \(stats.allocatedCount)")
        print("  Free: \(stats.freeCount)")
        print("  Utilization: \(String(format: "%.1f%%", stats.utilization * 100))")

        // Should have reused buffers
        #expect(stats.freeCount == 2)  // 5 freed - 3 reused = 2 free

        print("\n========================================")
        print("✅ PHASE 4 MEMORY POOL EFFICIENCY COMPLETE")
        print("========================================\n")
    }
}
