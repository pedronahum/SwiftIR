/// Memory Management - Buffer aliasing, lifetime tracking, and optimization
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides memory optimization through:
/// - Buffer aliasing analysis to reuse memory
/// - Tensor lifetime tracking for early deallocation
/// - Memory pool management for efficient allocation
/// - In-place mutation detection and optimization

import Foundation

// MARK: - Buffer Aliasing Analysis

/// Represents a memory buffer that can be shared between tensors
public struct BufferHandle: Hashable, Sendable {
    /// Unique identifier for this buffer
    public let id: UInt64
    /// Size in bytes
    public let sizeInBytes: Int
    /// Alignment requirement
    public let alignment: Int

    public init(id: UInt64, sizeInBytes: Int, alignment: Int = 64) {
        self.id = id
        self.sizeInBytes = sizeInBytes
        self.alignment = alignment
    }
}

/// Tracks buffer aliasing relationships for memory optimization
public final class BufferAliasingAnalyzer: @unchecked Sendable {
    private var nextBufferId: UInt64 = 1
    private let lock = NSLock()

    /// Map from value ID to buffer handle
    private var valueToBuffer: [UInt64: BufferHandle] = [:]

    /// Map from buffer to all values sharing it
    private var bufferToValues: [UInt64: Set<UInt64>] = [:]

    /// Initialize a new analyzer
    public init() {}

    /// Allocate a new buffer for a value
    public func allocateBuffer(
        for valueId: UInt64,
        shape: TensorShape,
        dtype: DType
    ) -> BufferHandle {
        lock.lock()
        defer { lock.unlock() }

        let sizeInBytes = (shape.elementCount ?? 1) * dtype.sizeInBytes
        let buffer = BufferHandle(
            id: nextBufferId,
            sizeInBytes: sizeInBytes,
            alignment: 64  // Standard cache line alignment
        )
        nextBufferId += 1

        valueToBuffer[valueId] = buffer
        bufferToValues[buffer.id, default: []].insert(valueId)

        return buffer
    }

    /// Create an alias - two values share the same buffer
    public func createAlias(source: UInt64, target: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        guard let buffer = valueToBuffer[source] else { return }

        valueToBuffer[target] = buffer
        bufferToValues[buffer.id]?.insert(target)
    }

    /// Check if two values can share a buffer (no overlapping lifetimes)
    public func canAlias(_ a: UInt64, _ b: UInt64, lifetimes: LifetimeAnalyzer) -> Bool {
        guard let lifetimeA = lifetimes.getLifetime(for: a),
              let lifetimeB = lifetimes.getLifetime(for: b) else {
            return false
        }

        // Can alias if lifetimes don't overlap
        return lifetimeA.end <= lifetimeB.start || lifetimeB.end <= lifetimeA.start
    }

    /// Get the buffer for a value
    public func getBuffer(for valueId: UInt64) -> BufferHandle? {
        lock.lock()
        defer { lock.unlock() }
        return valueToBuffer[valueId]
    }

    /// Get all values sharing a buffer
    public func getAliases(for bufferId: UInt64) -> Set<UInt64> {
        lock.lock()
        defer { lock.unlock() }
        return bufferToValues[bufferId] ?? []
    }

    /// Reset the analyzer
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextBufferId = 1
        valueToBuffer.removeAll()
        bufferToValues.removeAll()
    }
}

// MARK: - Tensor Lifetime Analysis

/// Represents the lifetime of a value in the computation graph
public struct ValueLifetime: Sendable {
    /// Operation index where value is created
    public let start: Int
    /// Operation index where value is last used
    public let end: Int
    /// The value ID
    public let valueId: UInt64

    public init(valueId: UInt64, start: Int, end: Int) {
        self.valueId = valueId
        self.start = start
        self.end = end
    }

    /// Duration of the lifetime
    public var duration: Int { end - start }
}

/// Analyzes tensor lifetimes for memory optimization
public final class LifetimeAnalyzer: @unchecked Sendable {
    private let lock = NSLock()

    /// Map from value ID to its lifetime
    private var lifetimes: [UInt64: ValueLifetime] = [:]

    /// Current operation index
    private var currentOpIndex: Int = 0

    /// Initialize a new analyzer
    public init() {}

    /// Record a value creation
    public func recordCreation(valueId: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        lifetimes[valueId] = ValueLifetime(
            valueId: valueId,
            start: currentOpIndex,
            end: currentOpIndex  // Will be updated on use
        )
    }

    /// Record a value usage
    public func recordUsage(valueId: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        guard let existing = lifetimes[valueId] else { return }
        lifetimes[valueId] = ValueLifetime(
            valueId: valueId,
            start: existing.start,
            end: currentOpIndex
        )
    }

    /// Advance to next operation
    public func nextOperation() {
        lock.lock()
        defer { lock.unlock() }
        currentOpIndex += 1
    }

    /// Get lifetime for a value
    public func getLifetime(for valueId: UInt64) -> ValueLifetime? {
        lock.lock()
        defer { lock.unlock() }
        return lifetimes[valueId]
    }

    /// Get all values that are dead after a given operation
    public func getDeadValues(afterOperation opIndex: Int) -> [UInt64] {
        lock.lock()
        defer { lock.unlock() }

        return lifetimes.values
            .filter { $0.end == opIndex }
            .map { $0.valueId }
    }

    /// Get values sorted by lifetime duration (longest first)
    public func getValuesByLifetime() -> [ValueLifetime] {
        lock.lock()
        defer { lock.unlock() }

        return lifetimes.values.sorted { $0.duration > $1.duration }
    }

    /// Reset the analyzer
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        lifetimes.removeAll()
        currentOpIndex = 0
    }
}

// MARK: - Memory Pool

/// A memory pool for efficient tensor allocation
public final class MemoryPool: @unchecked Sendable {
    private let lock = NSLock()

    /// Available buffers by size class
    private var freeBuffers: [Int: [BufferHandle]] = [:]

    /// All allocated buffers
    private var allocatedBuffers: Set<UInt64> = []

    /// Size classes for pooling (powers of 2)
    private let sizeClasses: [Int]

    /// Total memory allocated
    private var totalAllocated: Int = 0

    /// Peak memory usage
    public private(set) var peakMemoryUsage: Int = 0

    /// Current memory usage
    public var currentMemoryUsage: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalAllocated
    }

    private var nextBufferId: UInt64 = 1

    /// Initialize with size classes
    public init(maxSizeClass: Int = 1 << 30) {  // Default 1GB max
        // Create size classes as powers of 2
        var classes: [Int] = []
        var size = 64  // Start at 64 bytes
        while size <= maxSizeClass {
            classes.append(size)
            size *= 2
        }
        self.sizeClasses = classes
    }

    /// Get the size class for a given size
    private func sizeClass(for size: Int) -> Int {
        for sizeClass in sizeClasses {
            if size <= sizeClass {
                return sizeClass
            }
        }
        return sizeClasses.last ?? size
    }

    /// Allocate a buffer from the pool
    public func allocate(size: Int, alignment: Int = 64) -> BufferHandle {
        lock.lock()
        defer { lock.unlock() }

        let sizeClass = self.sizeClass(for: size)

        // Try to reuse a free buffer
        if var buffers = freeBuffers[sizeClass], !buffers.isEmpty {
            let buffer = buffers.removeLast()
            freeBuffers[sizeClass] = buffers
            allocatedBuffers.insert(buffer.id)
            return buffer
        }

        // Allocate new buffer
        let buffer = BufferHandle(
            id: nextBufferId,
            sizeInBytes: sizeClass,
            alignment: alignment
        )
        nextBufferId += 1

        allocatedBuffers.insert(buffer.id)
        totalAllocated += sizeClass
        peakMemoryUsage = max(peakMemoryUsage, totalAllocated)

        return buffer
    }

    /// Return a buffer to the pool
    public func deallocate(_ buffer: BufferHandle) {
        lock.lock()
        defer { lock.unlock() }

        guard allocatedBuffers.contains(buffer.id) else { return }

        allocatedBuffers.remove(buffer.id)
        freeBuffers[buffer.sizeInBytes, default: []].append(buffer)
    }

    /// Clear all free buffers
    public func clearFreeBuffers() {
        lock.lock()
        defer { lock.unlock() }

        for (sizeClass, buffers) in freeBuffers {
            totalAllocated -= sizeClass * buffers.count
        }
        freeBuffers.removeAll()
    }

    /// Reset the pool
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        freeBuffers.removeAll()
        allocatedBuffers.removeAll()
        totalAllocated = 0
        peakMemoryUsage = 0
        nextBufferId = 1
    }

    /// Get statistics about the pool
    public func getStatistics() -> MemoryPoolStatistics {
        lock.lock()
        defer { lock.unlock() }

        var freeCount = 0
        var freeSize = 0
        for (sizeClass, buffers) in freeBuffers {
            freeCount += buffers.count
            freeSize += sizeClass * buffers.count
        }

        return MemoryPoolStatistics(
            allocatedCount: allocatedBuffers.count,
            freeCount: freeCount,
            totalAllocated: totalAllocated,
            freeSize: freeSize,
            peakUsage: peakMemoryUsage
        )
    }
}

/// Statistics about memory pool usage
public struct MemoryPoolStatistics: Sendable {
    public let allocatedCount: Int
    public let freeCount: Int
    public let totalAllocated: Int
    public let freeSize: Int
    public let peakUsage: Int

    /// Utilization ratio
    public var utilization: Double {
        guard totalAllocated > 0 else { return 0 }
        return Double(totalAllocated - freeSize) / Double(totalAllocated)
    }
}

// MARK: - In-Place Mutation Optimizer

/// Detects opportunities for in-place operations
public final class InPlaceMutationOptimizer: @unchecked Sendable {
    private let lifetimes: LifetimeAnalyzer
    private let aliasing: BufferAliasingAnalyzer

    /// Operations that can be done in-place
    public enum InPlaceCandidate: Sendable {
        case unary(operation: String, input: UInt64, output: UInt64)
        case binary(operation: String, lhs: UInt64, rhs: UInt64, output: UInt64, inPlaceOperand: Int)
    }

    /// Detected in-place opportunities
    private var candidates: [InPlaceCandidate] = []
    private let lock = NSLock()

    public init(lifetimes: LifetimeAnalyzer, aliasing: BufferAliasingAnalyzer) {
        self.lifetimes = lifetimes
        self.aliasing = aliasing
    }

    /// Check if a unary operation can be done in-place
    public func canDoInPlace(
        operation: String,
        input: UInt64,
        output: UInt64,
        atOperation opIndex: Int
    ) -> Bool {
        // Check if input dies at this operation
        guard let lifetime = lifetimes.getLifetime(for: input) else { return false }
        if lifetime.end != opIndex { return false }

        // Check if shapes match (required for in-place)
        // In a full implementation, we'd check this against the graph

        // Record the candidate
        lock.lock()
        candidates.append(.unary(operation: operation, input: input, output: output))
        lock.unlock()

        return true
    }

    /// Check if a binary operation can be done in-place
    public func canDoInPlace(
        operation: String,
        lhs: UInt64,
        rhs: UInt64,
        output: UInt64,
        atOperation opIndex: Int
    ) -> (canInPlace: Bool, operand: Int) {
        // Check if either operand dies at this operation
        if let lhsLifetime = lifetimes.getLifetime(for: lhs),
           lhsLifetime.end == opIndex {
            lock.lock()
            candidates.append(.binary(
                operation: operation,
                lhs: lhs, rhs: rhs, output: output,
                inPlaceOperand: 0
            ))
            lock.unlock()
            return (true, 0)
        }

        if let rhsLifetime = lifetimes.getLifetime(for: rhs),
           rhsLifetime.end == opIndex {
            lock.lock()
            candidates.append(.binary(
                operation: operation,
                lhs: lhs, rhs: rhs, output: output,
                inPlaceOperand: 1
            ))
            lock.unlock()
            return (true, 1)
        }

        return (false, -1)
    }

    /// Get all detected in-place candidates
    public func getCandidates() -> [InPlaceCandidate] {
        lock.lock()
        defer { lock.unlock() }
        return candidates
    }

    /// Reset the optimizer
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        candidates.removeAll()
    }
}

// MARK: - Memory Manager

/// Central memory management coordinator
public final class MemoryManager: @unchecked Sendable {
    /// Shared instance
    public static let shared = MemoryManager()

    /// Memory pool for allocation
    public let pool: MemoryPool

    /// Lifetime analyzer
    public let lifetimes: LifetimeAnalyzer

    /// Buffer aliasing analyzer
    public let aliasing: BufferAliasingAnalyzer

    /// In-place mutation optimizer
    public let inPlaceOptimizer: InPlaceMutationOptimizer

    private init() {
        self.pool = MemoryPool()
        self.lifetimes = LifetimeAnalyzer()
        self.aliasing = BufferAliasingAnalyzer()
        self.inPlaceOptimizer = InPlaceMutationOptimizer(
            lifetimes: lifetimes,
            aliasing: aliasing
        )
    }

    /// Allocate memory for a tensor
    public func allocate(
        for valueId: UInt64,
        shape: TensorShape,
        dtype: DType
    ) -> BufferHandle {
        // Record creation
        lifetimes.recordCreation(valueId: valueId)

        // Try to find an aliasable buffer
        // (In full implementation, we'd search dead values with matching size)

        // Allocate from pool
        let size = (shape.elementCount ?? 1) * dtype.sizeInBytes
        let buffer = pool.allocate(size: size)

        // Record aliasing
        _ = aliasing.allocateBuffer(for: valueId, shape: shape, dtype: dtype)

        return buffer
    }

    /// Record usage of a value
    public func recordUsage(valueId: UInt64) {
        lifetimes.recordUsage(valueId: valueId)
    }

    /// Free memory that's no longer needed
    public func freeDeadValues(afterOperation opIndex: Int) {
        let deadValues = lifetimes.getDeadValues(afterOperation: opIndex)
        for valueId in deadValues {
            if let buffer = aliasing.getBuffer(for: valueId) {
                pool.deallocate(buffer)
            }
        }
    }

    /// Get memory statistics
    public func getStatistics() -> MemoryPoolStatistics {
        return pool.getStatistics()
    }

    /// Reset all memory management state
    public func reset() {
        pool.reset()
        lifetimes.reset()
        aliasing.reset()
        inPlaceOptimizer.reset()
    }
}
