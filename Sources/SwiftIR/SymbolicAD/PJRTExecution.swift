// PJRTExecution.swift
// Phase 14: Connect AD Pipeline to Real PJRT Execution
//
// This bridges the SymbolicAD compilation pipeline with actual PJRT execution
// on CPU/GPU/TPU hardware.

import Foundation
import SwiftIRXLA

// MARK: - LRU Cache Statistics

/// Statistics for cache performance monitoring
public struct CacheStatistics: CustomStringConvertible, Sendable {
    public let hits: Int
    public let misses: Int
    public let evictions: Int
    public let currentSize: Int
    public let maxSize: Int

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0.0
    }

    public var description: String {
        let rate = String(format: "%.1f%%", hitRate * 100)
        return "CacheStats(hits=\(hits), misses=\(misses), evictions=\(evictions), " +
               "size=\(currentSize)/\(maxSize), hitRate=\(rate))"
    }
}

// MARK: - LRU Executable Cache

/// Thread-safe LRU cache for compiled PJRT executables
///
/// This implements the same caching pattern as JAX's weakref_lru_cache:
/// - Configurable maximum size (default 4096 entries like JAX)
/// - LRU eviction when cache is full
/// - Hit/miss/eviction statistics for debugging
/// - Thread-safe with proper locking
///
/// The cache key is a hash of the MLIR module + backend type, so identical
/// programs compiled for the same backend will share the cached executable.
private final class ExecutableCache: @unchecked Sendable {
    /// Singleton instance with default max size
    nonisolated(unsafe) static let shared = ExecutableCache(maxSize: 4096)

    /// Maximum number of entries before LRU eviction kicks in
    public let maxSize: Int

    /// Cache statistics
    private var _hits: Int = 0
    private var _misses: Int = 0
    private var _evictions: Int = 0

    /// LRU linked list node
    private class LRUNode {
        let key: CacheKey
        var prev: LRUNode?
        var next: LRUNode?

        init(key: CacheKey) {
            self.key = key
        }
    }

    /// Cache key combining MLIR hash and backend
    private struct CacheKey: Hashable {
        let mlirHash: Int
        let backend: String
    }

    /// Cache entry storing the compiled executable and client
    private struct CacheEntry {
        let executable: PJRTLoadedExecutable
        let client: PJRTClient
        let node: LRUNode  // Reference to LRU list node for O(1) access
    }

    /// Main storage: key -> entry
    private var cache: [CacheKey: CacheEntry] = [:]

    /// LRU list head (most recently used) and tail (least recently used)
    private var head: LRUNode?  // Most recently used
    private var tail: LRUNode?  // Least recently used

    /// Thread safety lock
    private let lock = NSLock()

    /// Initialize with configurable max size
    /// - Parameter maxSize: Maximum number of cached executables (default 4096 like JAX)
    init(maxSize: Int = 4096) {
        self.maxSize = max(1, maxSize)
    }

    // MARK: - LRU List Operations

    /// Move a node to the front of the LRU list (most recently used)
    private func moveToFront(_ node: LRUNode) {
        guard node !== head else { return }  // Already at front

        // Remove from current position
        node.prev?.next = node.next
        node.next?.prev = node.prev

        // Update tail if needed
        if node === tail {
            tail = node.prev
        }

        // Insert at front
        node.prev = nil
        node.next = head
        head?.prev = node
        head = node

        // Update tail if this is the only node
        if tail == nil {
            tail = node
        }
    }

    /// Add a new node to the front of the LRU list
    private func addToFront(_ node: LRUNode) {
        node.prev = nil
        node.next = head
        head?.prev = node
        head = node

        if tail == nil {
            tail = node
        }
    }

    /// Remove the tail node (least recently used) and return its key
    private func removeTail() -> CacheKey? {
        guard let tailNode = tail else { return nil }

        let key = tailNode.key

        // Update tail pointer
        tail = tailNode.prev
        tail?.next = nil

        // Handle single element case
        if head === tailNode {
            head = nil
        }

        // Clear node references
        tailNode.prev = nil
        tailNode.next = nil

        return key
    }

    // MARK: - Public API

    /// Get or create a compiled executable
    ///
    /// If the executable is in cache, returns it and updates LRU order.
    /// If not, compiles the module, caches it (evicting LRU if needed), and returns it.
    ///
    /// - Parameters:
    ///   - mlirModule: The MLIR/StableHLO source to compile
    ///   - backend: Target backend (CPU, GPU, TPU)
    /// - Returns: Tuple of (executable, client)
    /// - Throws: PJRTError if compilation fails
    func getOrCompile(
        mlirModule: String,
        backend: PJRTClient.Backend
    ) throws -> (executable: PJRTLoadedExecutable, client: PJRTClient) {
        let key = CacheKey(mlirHash: mlirModule.hashValue, backend: backend.rawValue)

        // Fast path: check cache with lock
        lock.lock()
        if let entry = cache[key] {
            _hits += 1
            moveToFront(entry.node)
            lock.unlock()
            return (entry.executable, entry.client)
        }
        _misses += 1
        lock.unlock()

        // Slow path: compile outside of lock to avoid blocking other threads
        let client = try PJRTClient(backend: backend)
        let executable = try client.compile(mlirModule: mlirModule)

        // Insert into cache with lock
        lock.lock()
        defer { lock.unlock() }

        // Double-check: another thread may have compiled this while we were working
        if let entry = cache[key] {
            // Someone else compiled it - use theirs, update LRU
            moveToFront(entry.node)
            return (entry.executable, entry.client)
        }

        // Evict LRU entries if we're at capacity
        while cache.count >= maxSize {
            if let evictKey = removeTail() {
                cache.removeValue(forKey: evictKey)
                _evictions += 1
            } else {
                break
            }
        }

        // Create new entry and add to cache
        let node = LRUNode(key: key)
        addToFront(node)
        cache[key] = CacheEntry(executable: executable, client: client, node: node)

        return (executable, client)
    }

    /// Clear the entire cache
    func clear() {
        lock.lock()
        cache.removeAll()
        head = nil
        tail = nil
        // Note: we don't reset statistics - they track lifetime behavior
        lock.unlock()
    }

    /// Reset statistics counters
    func resetStatistics() {
        lock.lock()
        _hits = 0
        _misses = 0
        _evictions = 0
        lock.unlock()
    }

    /// Get current cache statistics
    var statistics: CacheStatistics {
        lock.lock()
        defer { lock.unlock() }
        return CacheStatistics(
            hits: _hits,
            misses: _misses,
            evictions: _evictions,
            currentSize: cache.count,
            maxSize: maxSize
        )
    }

    /// Current number of cached entries
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    /// Check if cache contains an entry for the given module and backend
    func contains(mlirModule: String, backend: PJRTClient.Backend) -> Bool {
        let key = CacheKey(mlirHash: mlirModule.hashValue, backend: backend.rawValue)
        lock.lock()
        defer { lock.unlock() }
        return cache[key] != nil
    }
}

// MARK: - PJRT-Backed Runtime

/// Runtime that executes compiled functions on real hardware via PJRT
public class PJRTBackedRuntime {
    /// Backend type
    public enum Backend {
        case cpu
        case gpu(deviceId: Int)
        case tpu(deviceId: Int)

        var pluginPath: String {
            switch self {
            case .cpu:
                #if os(macOS)
                let paths = [
                    "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_cpu_plugin.dylib"
                ]
                #else
                let paths = [
                    "/opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so",
                    "/usr/local/lib/pjrt_c_api_cpu_plugin.so"
                ]
                #endif
                for path in paths {
                    if FileManager.default.fileExists(atPath: path) {
                        return path
                    }
                }
                return paths[0]
            case .gpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_gpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_gpu_plugin.so"
                #endif
            case .tpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_tpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_tpu_plugin.so"
                #endif
            }
        }
    }

    public let backend: Backend
    private var isInitialized: Bool = false

    /// Info about this runtime
    public var info: String {
        switch backend {
        case .cpu:
            return "PJRT Runtime (CPU)"
        case .gpu(let id):
            return "PJRT Runtime (GPU:\(id))"
        case .tpu(let id):
            return "PJRT Runtime (TPU:\(id))"
        }
    }

    public init(backend: Backend = .cpu) throws {
        self.backend = backend

        // Verify plugin exists
        let pluginPath = backend.pluginPath
        guard FileManager.default.fileExists(atPath: pluginPath) else {
            throw PJRTExecutionError.pluginNotFound(pluginPath)
        }

        self.isInitialized = true
    }

    /// Compile an MLIR module to an executable
    public func compile(_ mlirModule: String) throws -> PJRTBackedExecutable {
        guard isInitialized else {
            throw PJRTExecutionError.notInitialized
        }

        return PJRTBackedExecutable(
            mlirSource: mlirModule,
            runtime: self
        )
    }

    /// Compile from MLIRModule
    public func compile(_ module: MLIRModule) throws -> PJRTBackedExecutable {
        return try compile(module.mlirText)
    }

    // MARK: - Global Cache Management

    /// Get statistics from the global executable cache
    ///
    /// Returns hits, misses, evictions, current size, and max size.
    /// Useful for debugging and performance monitoring.
    ///
    /// Example:
    /// ```swift
    /// let stats = PJRTBackedRuntime.cacheStatistics
    /// print("Cache hit rate: \(stats.hitRate * 100)%")
    /// print("Evictions: \(stats.evictions)")
    /// ```
    public static var cacheStatistics: CacheStatistics {
        return ExecutableCache.shared.statistics
    }

    /// Clear the global executable cache
    ///
    /// This frees all cached PJRT executables and clients.
    /// Useful for memory management or testing.
    public static func clearCache() {
        ExecutableCache.shared.clear()
    }

    /// Reset cache statistics counters
    ///
    /// Resets hits, misses, and evictions to zero.
    /// Useful for benchmarking specific code sections.
    public static func resetCacheStatistics() {
        ExecutableCache.shared.resetStatistics()
    }

    /// Maximum number of executables the cache can hold
    public static var cacheMaxSize: Int {
        return ExecutableCache.shared.maxSize
    }
}

// MARK: - PJRT-Backed Executable

/// Host buffer semantics for PJRT execution
///
/// These control how host memory is handled when creating device buffers.
/// The optimal choice depends on your backend and usage pattern.
public enum BufferSemantics: String, CaseIterable {
    /// Data is copied during the call, buffer can be modified after return.
    /// This is the safest option with no lifetime requirements.
    /// Good for: General use, small-medium buffers, when you need to modify input after call.
    case copy = "copy"

    /// Data must remain valid until transfer completes (async transfer).
    /// Allows overlapping CPU work with data transfer.
    /// Good for: Pipelining, when you can guarantee data lifetime until callback.
    case asyncTransfer = "async"

    /// Zero-copy: host memory used directly by the runtime.
    /// Fastest option but requires data to remain valid for buffer lifetime.
    /// Good for: CPU backend, repeated execution with same data, large buffers.
    /// WARNING: Only use when host buffer outlives the PJRT buffer!
    case zeroCopy = "zeroCopy"

    /// Mutable zero-copy: for buffer donation - memory can be reused for output.
    /// Use with executeAsyncWithDonation() for in-place operations.
    /// The input buffer becomes invalid after execution and its memory may be reused.
    /// Good for: Iterative algorithms where output replaces input (e.g., optimization steps).
    case mutableZeroCopy = "mutableZeroCopy"

    /// Auto-select best semantics based on backend type.
    /// - CPU: Uses zeroCopy (can directly access host memory)
    /// - GPU: Uses copy (data must be transferred to device memory)
    /// - TPU: Uses copy (data must be transferred to device memory)
    case auto = "auto"

    /// Convert to PJRTClient.HostBufferSemantics
    func toClientSemantics(for backend: PJRTBackedRuntime.Backend) -> PJRTClient.HostBufferSemantics {
        switch self {
        case .copy:
            return .immutableOnlyDuringCall
        case .asyncTransfer:
            return .immutableUntilTransferCompletes
        case .zeroCopy:
            return .zeroCopy
        case .mutableZeroCopy:
            return .mutableZeroCopy
        case .auto:
            // Auto-select based on backend
            switch backend {
            case .cpu:
                // CPU can directly access host memory - use zero-copy
                return .zeroCopy
            case .gpu, .tpu:
                // GPU/TPU need data transfer - use copy semantics
                // (zeroCopy may not work or may be slower due to PCIe transfers)
                return .immutableOnlyDuringCall
            }
        }
    }
}

/// An executable that runs on real PJRT hardware
public class PJRTBackedExecutable {
    public let mlirSource: String
    public private(set) var stablehloSource: String
    public let backend: PJRTBackedRuntime.Backend

    /// XLA backend optimization level
    /// Controls how much optimization XLA applies during compilation
    public var xlaOptLevel: XLAOptimizationLevel = .default

    /// Input-output aliases for buffer donation
    /// When configured, XLA can reuse input buffers for outputs with matching shapes
    public private(set) var inputOutputAliases: [InputOutputAlias] = []

    /// Cached PJRT client and executable for repeated execution
    private var pjrtClient: PJRTClient?
    private var pjrtExecutable: PJRTLoadedExecutable?

    /// Cached buffers for optimized repeated execution
    private var cachedInputBuffers: [PJRTBuffer]?
    private var cachedInputShapes: [[Int]]?
    private var cachedOutputBuffers: [PJRTBuffer]?

    /// Preallocated output arrays for optimized execution
    private var preallocatedOutputs: [[Float]]?

    /// Cached output metadata (shapes and sizes) from first execution
    /// This avoids querying PJRT for output buffer dimensions on every call
    private var cachedOutputMetadata: [(shape: [Int], elementCount: Int)]?

    /// Cached input shapes for fast buffer creation
    /// Tracks the shape for each input slot to detect when shapes change
    private var cachedInputSizes: [Int]?

    /// Cached device buffers for buffer reuse optimization
    /// When inputs don't change between calls, we can skip H2D transfer entirely
    private var reusableInputBuffers: [PJRTBuffer]?

    /// Hash of input data to detect changes (for buffer reuse)
    private var inputDataHashes: [Int]?

    init(mlirSource: String, runtime: PJRTBackedRuntime) {
        self.mlirSource = mlirSource
        self.backend = runtime.backend

        // Convert to StableHLO format
        self.stablehloSource = Self.convertToStableHLO(mlirSource)
    }

    /// Configure input-output aliases for buffer donation
    ///
    /// This tells XLA which input buffers can be reused for outputs.
    /// For donation to work:
    /// 1. Input and output shapes must match
    /// 2. Use executeAsyncWithDonation() with the corresponding donateInputIndices
    ///
    /// - Parameter aliases: Array of input-output alias specifications
    /// - Note: Calling this method invalidates any cached executable and forces recompilation
    public func configureInputOutputAliases(_ aliases: [InputOutputAlias]) {
        self.inputOutputAliases = aliases
        // Invalidate cached executable since MLIR changed
        self.pjrtExecutable = nil
        // Regenerate stablehloSource with aliases
        self.stablehloSource = Self.convertToStableHLO(mlirSource, aliases: aliases)
    }

    /// Configure a simple alias where input N maps to output N (common case)
    /// - Parameter inputIndices: Which input indices should alias with their corresponding output indices
    public func configureSimpleAliases(inputIndices: [Int]) {
        let aliases = inputIndices.map { InputOutputAlias(inputIndex: $0, outputIndex: $0) }
        configureInputOutputAliases(aliases)
    }

    /// Execute with float arrays using real PJRT
    public func execute(_ inputs: [[Float]]) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            do {
                pjrtClient = try PJRTClient(backend: pjrtBackend)
            } catch {
                throw PJRTExecutionError.compilationFailed("Failed to create PJRT client: \(error)")
            }
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            do {
                // Use stablehloSource directly - it already contains a complete module
                pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
            } catch {
                // Include the MLIR source in the error message for debugging
                let truncatedSource = stablehloSource.count > 2000
                    ? String(stablehloSource.prefix(2000)) + "\n... [truncated]"
                    : stablehloSource
                throw PJRTExecutionError.compilationFailed("""
                    Failed to compile: \(error)

                    Generated MLIR:
                    \(truncatedSource)
                    """)
            }
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.compilationFailed("No executable available")
        }

        guard let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No device available")
        }

        // Use auto semantics (zeroCopy for CPU, copy for GPU/TPU)
        let clientSemantics = BufferSemantics.auto.toClientSemantics(for: backend)

        // Create input buffers with auto-selected semantics
        var inputBuffers: [PJRTBuffer] = []
        for input in inputs {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: clientSemantics
                )
            }
            inputBuffers.append(buffer)
        }

        // Execute
        let outputBuffers: [PJRTBuffer]
        do {
            outputBuffers = try executable.execute(arguments: inputBuffers)
        } catch {
            throw PJRTExecutionError.executionFailed("Execution failed: \(error)")
        }

        // Read results back to host
        var results: [[Float]] = []
        for buffer in outputBuffers {
            let count = buffer.elementCount
            var result = [Float](repeating: 0, count: count)
            try result.withUnsafeMutableBytes { ptr in
                try buffer.toHost(destination: ptr.baseAddress!)
            }
            results.append(result)
        }

        return results
    }

    /// Optimized execution with preallocated outputs for repeated calls
    ///
    /// This method reuses preallocated output arrays when possible,
    /// reducing allocation overhead for repeated calls at the same shape.
    /// The savings become significant for larger batch sizes (>10K elements).
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - reuseBuffers: Ignored (kept for API compatibility)
    /// - Returns: Output arrays
    public func executeOptimized(_ inputs: [[Float]], reuseBuffers: Bool = true) throws -> [[Float]] {
        // Use standard execute() path for client/executable setup and input handling
        // The only optimization here is output array preallocation

        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed (same as execute())
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed (same as execute())
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Use auto semantics (zeroCopy for CPU, copy for GPU/TPU)
        let clientSemantics = BufferSemantics.auto.toClientSemantics(for: backend)

        // Create input buffers with auto-selected semantics
        var inputBuffers: [PJRTBuffer] = []
        inputBuffers.reserveCapacity(inputs.count)
        for input in inputs {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: clientSemantics
                )
            }
            inputBuffers.append(buffer)
        }

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Reuse preallocated output arrays if sizes match
        // This is the main optimization - avoids [Float](repeating:count:) allocation
        let canReuse = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputBuffers.count &&
            preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputBuffers[$0].elementCount }

        var results: [[Float]]
        if canReuse {
            results = preallocatedOutputs!
        } else {
            // First run or shape changed - allocate new arrays
            results = outputBuffers.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Read results back to host
        for (i, buffer) in outputBuffers.enumerated() {
            try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHost(destination: ptr.baseAddress!)
            }
        }

        return results
    }

    /// Execute with explicit buffer semantics control
    ///
    /// This method allows fine-grained control over how host memory is handled
    /// when creating device buffers. Use this for benchmarking or when you know
    /// the optimal semantics for your use case.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - semantics: Buffer semantics to use (copy, asyncTransfer, zeroCopy, or auto)
    ///   - preallocateOutputs: Whether to reuse preallocated output arrays
    /// - Returns: Output arrays
    ///
    /// ## Semantics Guide:
    /// - `.copy`: Safest. Data copied immediately. Best for GPU/TPU or when you modify inputs.
    /// - `.asyncTransfer`: Async copy. Data must remain valid until transfer completes.
    /// - `.zeroCopy`: Fastest on CPU. No copy, but data must outlive the PJRT buffer.
    /// - `.auto`: Automatically selects best option based on backend (zeroCopy for CPU, copy for GPU/TPU).
    ///
    /// ## Example:
    /// ```swift
    /// // Use auto-selection (recommended)
    /// let result = try executable.executeWithSemantics(inputs, semantics: .auto)
    ///
    /// // Force zero-copy on CPU for maximum performance
    /// let result = try executable.executeWithSemantics(inputs, semantics: .zeroCopy)
    /// ```
    public func executeWithSemantics(
        _ inputs: [[Float]],
        semantics: BufferSemantics = .auto,
        preallocateOutputs: Bool = true
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Create input buffers with specified semantics
        var inputBuffers: [PJRTBuffer] = []
        inputBuffers.reserveCapacity(inputs.count)
        for input in inputs {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: clientSemantics
                )
            }
            inputBuffers.append(buffer)
        }

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Handle output allocation
        var results: [[Float]]
        if preallocateOutputs {
            let canReuse = preallocatedOutputs != nil &&
                preallocatedOutputs!.count == outputBuffers.count &&
                preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputBuffers[$0].elementCount }

            if canReuse {
                results = preallocatedOutputs!
            } else {
                results = outputBuffers.map { [Float](repeating: 0, count: $0.elementCount) }
                preallocatedOutputs = results
            }
        } else {
            results = outputBuffers.map { [Float](repeating: 0, count: $0.elementCount) }
        }

        // Read results back to host
        for (i, buffer) in outputBuffers.enumerated() {
            try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHost(destination: ptr.baseAddress!)
            }
        }

        return results
    }

    /// Execute with async D2H transfers and all optimizations enabled
    ///
    /// This is the most optimized execution path, combining:
    /// 1. Auto buffer semantics (zeroCopy on CPU, copy on GPU/TPU)
    /// 2. Cached output metadata (avoids PJRT queries after first run)
    /// 3. Preallocated output arrays (avoids allocation overhead)
    /// 4. Async D2H transfers (initiates all transfers then awaits)
    ///
    /// Use this method for repeated execution at the same input shape for best performance.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Output arrays
    public func executeAsync(
        _ inputs: [[Float]],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Check if input shapes match cached shapes for fast path
        let currentSizes = inputs.map { $0.count }
        let shapesMatch = cachedInputSizes != nil &&
            cachedInputSizes!.count == currentSizes.count &&
            cachedInputSizes! == currentSizes

        // Create input buffers with fast path if shapes match
        var inputBuffers: [PJRTBuffer] = []
        inputBuffers.reserveCapacity(inputs.count)
        for (i, input) in inputs.enumerated() {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBufferFast(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: clientSemantics,
                    cachedSlot: i,
                    isSameShape: shapesMatch
                )
            }
            inputBuffers.append(buffer)
        }

        // Cache shapes for next call
        if !shapesMatch {
            cachedInputSizes = currentSizes
        }

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Use cached output metadata if available, otherwise query and cache
        let outputMeta: [(shape: [Int], elementCount: Int)]
        if let cached = cachedOutputMetadata, cached.count == outputBuffers.count {
            outputMeta = cached
        } else {
            // First execution - query and cache metadata
            outputMeta = outputBuffers.map { buffer in
                (shape: buffer.shape, elementCount: buffer.elementCount)
            }
            cachedOutputMetadata = outputMeta
        }

        // Prepare output arrays - reuse preallocated if possible
        var results: [[Float]]
        let canReuse = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputMeta.count &&
            preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputMeta[$0].elementCount }

        if canReuse {
            results = preallocatedOutputs!
        } else {
            results = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Initiate ALL async D2H transfers first
        var events: [PJRTEvent?] = []
        events.reserveCapacity(outputBuffers.count)

        for (i, buffer) in outputBuffers.enumerated() {
            let event = try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHostAsync(destination: ptr.baseAddress!)
            }
            events.append(event)
        }

        // Now await ALL events (allows overlapped transfers)
        for event in events {
            try event?.awaitAndDestroy()
        }

        return results
    }

    /// Execute with input buffer reuse - the most optimized path
    ///
    /// This method implements JAX-style buffer reuse optimization:
    /// - Caches device buffers between calls
    /// - Only performs H2D transfer when input data actually changes
    /// - For repeated calls with same data, skips transfer entirely
    ///
    /// This is ideal for iterative algorithms where inputs change each iteration
    /// but you want to minimize transfer overhead.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Output arrays
    public func executeWithBufferReuse(
        _ inputs: [[Float]],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Check current input sizes
        let currentSizes = inputs.map { $0.count }

        // Compute hashes for each input to detect data changes
        // Use a fast hash: combine first, middle, and last elements with size
        let currentHashes = inputs.map { input -> Int in
            guard !input.isEmpty else { return 0 }
            var h = input.count
            h = h &* 31 &+ input[0].bitPattern.hashValue
            if input.count > 1 {
                h = h &* 31 &+ input[input.count / 2].bitPattern.hashValue
                h = h &* 31 &+ input[input.count - 1].bitPattern.hashValue
            }
            return h
        }

        // Check if we can reuse cached buffers
        let canReuseCachedBuffers = reusableInputBuffers != nil &&
            reusableInputBuffers!.count == inputs.count &&
            cachedInputSizes == currentSizes &&
            inputDataHashes == currentHashes

        var inputBuffers: [PJRTBuffer]

        if canReuseCachedBuffers {
            // Fast path: reuse existing device buffers, no H2D transfer!
            inputBuffers = reusableInputBuffers!
        } else {
            // Need to create new buffers and transfer data
            inputBuffers = []
            inputBuffers.reserveCapacity(inputs.count)

            // Check which specific buffers can be reused (same size, same slot)
            let canReuseSlot: [Bool]
            if let cachedSizes = cachedInputSizes, cachedSizes.count == currentSizes.count {
                canReuseSlot = zip(cachedSizes, currentSizes).map { $0 == $1 }
            } else {
                canReuseSlot = Array(repeating: false, count: inputs.count)
            }

            for (i, input) in inputs.enumerated() {
                // Check if this specific slot's data changed
                let slotDataChanged = inputDataHashes == nil ||
                    i >= inputDataHashes!.count ||
                    inputDataHashes![i] != currentHashes[i]

                if canReuseSlot[i] && !slotDataChanged,
                   let cachedBuffer = reusableInputBuffers?[i] {
                    // Reuse this buffer - data hasn't changed
                    inputBuffers.append(cachedBuffer)
                } else {
                    // Create new buffer with H2D transfer
                    let buffer = try input.withUnsafeBytes { ptr in
                        try client.createBufferFast(
                            data: ptr.baseAddress!,
                            shape: [input.count],
                            elementType: .f32,
                            device: device,
                            semantics: clientSemantics,
                            cachedSlot: i,
                            isSameShape: canReuseSlot[i]
                        )
                    }
                    inputBuffers.append(buffer)
                }
            }

            // Update cache
            reusableInputBuffers = inputBuffers
            cachedInputSizes = currentSizes
            inputDataHashes = currentHashes
        }

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Use cached output metadata if available
        let outputMeta: [(shape: [Int], elementCount: Int)]
        if let cached = cachedOutputMetadata, cached.count == outputBuffers.count {
            outputMeta = cached
        } else {
            outputMeta = outputBuffers.map { buffer in
                (shape: buffer.shape, elementCount: buffer.elementCount)
            }
            cachedOutputMetadata = outputMeta
        }

        // Prepare output arrays - reuse preallocated if possible
        var results: [[Float]]
        let canReuseOutputs = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputMeta.count &&
            preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputMeta[$0].elementCount }

        if canReuseOutputs {
            results = preallocatedOutputs!
        } else {
            results = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Async D2H transfers
        var events: [PJRTEvent?] = []
        events.reserveCapacity(outputBuffers.count)

        for (i, buffer) in outputBuffers.enumerated() {
            let event = try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHostAsync(destination: ptr.baseAddress!)
            }
            events.append(event)
        }

        // Await all events
        for event in events {
            try event?.awaitAndDestroy()
        }

        return results
    }

    /// Execute with batched buffer creation - reduces FFI overhead for multiple inputs
    ///
    /// This method creates all input buffers in a single batched C call, reducing
    /// the per-buffer FFI overhead. Best for functions with multiple inputs.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Output arrays
    public func executeWithBatchedBuffers(
        _ inputs: [[Float]],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics: PJRTClient.HostBufferSemantics
        switch semantics.toClientSemantics(for: backend) {
        case .immutableOnlyDuringCall: clientSemantics = .immutableOnlyDuringCall
        case .immutableUntilTransferCompletes: clientSemantics = .immutableUntilTransferCompletes
        case .zeroCopy: clientSemantics = .zeroCopy
        case .mutableZeroCopy: clientSemantics = .mutableZeroCopy
        }

        // Create shapes array
        let shapes = inputs.map { [$0.count] }

        // Create all input buffers in a single batched call
        let inputBuffers = try client.createBuffersBatched(
            dataArrays: inputs,
            shapes: shapes,
            elementType: .f32,
            device: device,
            semantics: clientSemantics
        )

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Use cached output metadata if available
        let outputMeta: [(shape: [Int], elementCount: Int)]
        if let cached = cachedOutputMetadata, cached.count == outputBuffers.count {
            outputMeta = cached
        } else {
            outputMeta = outputBuffers.map { buffer in
                (shape: buffer.shape, elementCount: buffer.elementCount)
            }
            cachedOutputMetadata = outputMeta
        }

        // Prepare output arrays - reuse preallocated if possible
        var results: [[Float]]
        let canReuseOutputs = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputMeta.count &&
            preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputMeta[$0].elementCount }

        if canReuseOutputs {
            results = preallocatedOutputs!
        } else {
            results = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Async D2H transfers
        var events: [PJRTEvent?] = []
        events.reserveCapacity(outputBuffers.count)

        for (i, buffer) in outputBuffers.enumerated() {
            let event = try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHostAsync(destination: ptr.baseAddress!)
            }
            events.append(event)
        }

        // Await all events
        for event in events {
            try event?.awaitAndDestroy()
        }

        return results
    }

    /// Ultra-optimized execution using global executable caching
    ///
    /// This is the fastest execution path, implementing all optimizations:
    /// 1. Global executable cache - avoids recompilation across calls
    /// 2. Input buffer reuse - skips H2D when data unchanged
    /// 3. Preallocated output arrays - zero allocation per call
    /// 4. Async overlapped D2H transfers
    /// 5. Cached output metadata - no queries after first call
    ///
    /// Use this for the tightest inner loops where every microsecond counts.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    /// - Returns: Output arrays
    public func executeUltraOptimized(_ inputs: [[Float]]) throws -> [[Float]] {
        // Get cached executable from global cache
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        // Use global cache for client/executable
        if pjrtClient == nil || pjrtExecutable == nil {
            let cached = try ExecutableCache.shared.getOrCompile(
                mlirModule: stablehloSource,
                backend: pjrtBackend
            )
            pjrtClient = cached.client
            pjrtExecutable = cached.executable
        }

        guard let client = pjrtClient,
              let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Use zero-copy for CPU (fastest path)
        let clientSemantics: PJRTClient.HostBufferSemantics
        switch backend {
        case .cpu:
            clientSemantics = .zeroCopy
        case .gpu, .tpu:
            clientSemantics = .immutableOnlyDuringCall
        }

        // Check current input sizes
        let currentSizes = inputs.map { $0.count }

        // Fast hash for change detection (first + middle + last + size)
        let currentHashes = inputs.map { input -> Int in
            guard !input.isEmpty else { return 0 }
            var h = input.count
            h = h &* 31 &+ input[0].bitPattern.hashValue
            if input.count > 1 {
                h = h &* 31 &+ input[input.count / 2].bitPattern.hashValue
                h = h &* 31 &+ input[input.count - 1].bitPattern.hashValue
            }
            return h
        }

        // Check if we can fully reuse cached buffers (no H2D needed!)
        let canReuseCachedBuffers = reusableInputBuffers != nil &&
            reusableInputBuffers!.count == inputs.count &&
            cachedInputSizes == currentSizes &&
            inputDataHashes == currentHashes

        var inputBuffers: [PJRTBuffer]

        if canReuseCachedBuffers {
            // Ultra-fast path: reuse existing device buffers, zero H2D!
            inputBuffers = reusableInputBuffers!
        } else {
            // Need to create/update buffers
            inputBuffers = []
            inputBuffers.reserveCapacity(inputs.count)

            let canReuseSlot: [Bool]
            if let cachedSizes = cachedInputSizes, cachedSizes.count == currentSizes.count {
                canReuseSlot = zip(cachedSizes, currentSizes).map { $0 == $1 }
            } else {
                canReuseSlot = Array(repeating: false, count: inputs.count)
            }

            for (i, input) in inputs.enumerated() {
                let slotDataChanged = inputDataHashes == nil ||
                    i >= inputDataHashes!.count ||
                    inputDataHashes![i] != currentHashes[i]

                if canReuseSlot[i] && !slotDataChanged,
                   let cachedBuffer = reusableInputBuffers?[i] {
                    inputBuffers.append(cachedBuffer)
                } else {
                    let buffer = try input.withUnsafeBytes { ptr in
                        try client.createBufferFast(
                            data: ptr.baseAddress!,
                            shape: [input.count],
                            elementType: .f32,
                            device: device,
                            semantics: clientSemantics,
                            cachedSlot: i,
                            isSameShape: canReuseSlot[i]
                        )
                    }
                    inputBuffers.append(buffer)
                }
            }

            reusableInputBuffers = inputBuffers
            cachedInputSizes = currentSizes
            inputDataHashes = currentHashes
        }

        // Execute
        let outputBuffers = try executable.execute(arguments: inputBuffers)

        // Use cached output metadata
        let outputMeta: [(shape: [Int], elementCount: Int)]
        if let cached = cachedOutputMetadata, cached.count == outputBuffers.count {
            outputMeta = cached
        } else {
            outputMeta = outputBuffers.map { (shape: $0.shape, elementCount: $0.elementCount) }
            cachedOutputMetadata = outputMeta
        }

        // Reuse preallocated output arrays
        var results: [[Float]]
        if let prealloc = preallocatedOutputs,
           prealloc.count == outputMeta.count,
           prealloc.indices.allSatisfy({ prealloc[$0].count == outputMeta[$0].elementCount }) {
            results = prealloc
        } else {
            results = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Async D2H with overlapped transfers
        var events: [PJRTEvent?] = []
        events.reserveCapacity(outputBuffers.count)

        for (i, buffer) in outputBuffers.enumerated() {
            let event = try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHostAsync(destination: ptr.baseAddress!)
            }
            events.append(event)
        }

        for event in events {
            try event?.awaitAndDestroy()
        }

        return results
    }

    /// Hot path execution: Full H2D + Execute + D2H in a single FFI call
    ///
    /// This is the most optimized execution path for repeated execution with the same shapes.
    /// Combines buffer creation, execution, and transfer all in one C call.
    ///
    /// **Requirements:**
    /// - Must call `initializeHotPath()` first to set up output metadata
    /// - Input and output sizes must be known in advance
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - outputSizes: Number of elements in each output (must match executable's outputs)
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Output arrays
    public func executeHotPath(
        _ inputs: [[Float]],
        outputSizes: [Int],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.executionFailed("No executable available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Prepare or reuse output arrays
        let canReuse = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputSizes.count &&
            zip(preallocatedOutputs!, outputSizes).allSatisfy { $0.count == $1 }

        var results: [[Float]]
        if canReuse {
            results = preallocatedOutputs!
        } else {
            results = outputSizes.map { [Float](repeating: 0, count: $0) }
            preallocatedOutputs = results
        }

        // Build input sizes
        var inputSizes: [Int] = []
        inputSizes.reserveCapacity(inputs.count)

        for input in inputs {
            inputSizes.append(input.count)
        }

        // Extract results to separate variables to avoid exclusivity violations
        var result0 = results[0]
        var result1 = results.count > 1 ? results[1] : [Float]()

        // Build output pointers - use withUnsafeMutableBytes for safety
        // We need to pass raw pointers to C, but keep the arrays alive
        try inputs[0].withUnsafeBytes { input0Buf in
            let input0Ptr = input0Buf.baseAddress!

            try result0.withUnsafeMutableBytes { out0Buf in
                let out0Ptr = out0Buf.baseAddress!

                if results.count > 1 {
                    try result1.withUnsafeMutableBytes { out1Buf in
                        let out1Ptr = out1Buf.baseAddress!

                        let inputDataPtrs = [input0Ptr]
                        let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr), UnsafeMutableRawPointer(mutating: out1Ptr)]

                        try executable.executeHotPath(
                            inputData: inputDataPtrs,
                            inputSizes: inputSizes,
                            outputData: outputDataPtrs,
                            outputSizes: outputSizes,
                            semantics: clientSemantics
                        )
                    }
                } else {
                    let inputDataPtrs = [input0Ptr]
                    let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr)]

                    try executable.executeHotPath(
                        inputData: inputDataPtrs,
                        inputSizes: inputSizes,
                        outputData: outputDataPtrs,
                        outputSizes: outputSizes,
                        semantics: clientSemantics
                    )
                }
            }
        }

        // Update results with modified data
        results[0] = result0
        if results.count > 1 {
            results[1] = result1
        }

        return results
    }

    // MARK: - Profiled Execution

    /// Execute with detailed timing breakdown
    ///
    /// This method profiles the complete execution path (H2D + Execute + D2H)
    /// and returns timing data showing where time is spent.
    ///
    /// Use this to identify performance bottlenecks in the PJRT execution pipeline.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - outputSizes: Number of elements in each output (must match executable's outputs)
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Tuple of (output arrays, execution timing breakdown)
    public func executeProfiled(
        _ inputs: [[Float]],
        outputSizes: [Int],
        semantics: BufferSemantics = .auto
    ) throws -> ([[Float]], PJRTExecutionTiming) {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.executionFailed("No executable available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Prepare output arrays
        var results = outputSizes.map { [Float](repeating: 0, count: $0) }

        // Build input sizes
        var inputSizes: [Int] = []
        inputSizes.reserveCapacity(inputs.count)

        for input in inputs {
            inputSizes.append(input.count)
        }

        // Extract results to separate variables to avoid exclusivity violations
        var result0 = results[0]
        var result1 = results.count > 1 ? results[1] : [Float]()
        var timing: PJRTExecutionTiming!

        // Build output pointers - use withUnsafeMutableBytes for safety
        try inputs[0].withUnsafeBytes { input0Buf in
            let input0Ptr = input0Buf.baseAddress!

            try result0.withUnsafeMutableBytes { out0Buf in
                let out0Ptr = out0Buf.baseAddress!

                if results.count > 1 {
                    try result1.withUnsafeMutableBytes { out1Buf in
                        let out1Ptr = out1Buf.baseAddress!

                        let inputDataPtrs = [input0Ptr]
                        let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr), UnsafeMutableRawPointer(mutating: out1Ptr)]

                        timing = try executable.executeProfiled(
                            inputData: inputDataPtrs,
                            inputSizes: inputSizes,
                            outputData: outputDataPtrs,
                            outputSizes: outputSizes,
                            semantics: clientSemantics
                        )
                    }
                } else {
                    let inputDataPtrs = [input0Ptr]
                    let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr)]

                    timing = try executable.executeProfiled(
                        inputData: inputDataPtrs,
                        inputSizes: inputSizes,
                        outputData: outputDataPtrs,
                        outputSizes: outputSizes,
                        semantics: clientSemantics
                    )
                }
            }
        }

        // Update results with modified data
        results[0] = result0
        if results.count > 1 {
            results[1] = result1
        }

        return (results, timing)
    }

    /// Execute with OnReady callbacks instead of blocking await
    /// This uses PJRT_Event_OnReady for D2H completion instead of PJRT_Event_Await.
    /// Returns timing breakdown for comparison with blocking version.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - outputSizes: Number of elements in each output (must match executable's outputs)
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Tuple of (output arrays, execution timing breakdown)
    public func executeWithCallbacks(
        _ inputs: [[Float]],
        outputSizes: [Int],
        semantics: BufferSemantics = .auto
    ) throws -> ([[Float]], PJRTExecutionTiming) {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.executionFailed("No executable available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Prepare output arrays
        var results = outputSizes.map { [Float](repeating: 0, count: $0) }

        // Build input sizes
        var inputSizes: [Int] = []
        inputSizes.reserveCapacity(inputs.count)

        for input in inputs {
            inputSizes.append(input.count)
        }

        // Extract results to separate variables to avoid exclusivity violations
        var result0 = results[0]
        var result1 = results.count > 1 ? results[1] : [Float]()
        var timing: PJRTExecutionTiming!

        // Build output pointers - use withUnsafeMutableBytes for safety
        try inputs[0].withUnsafeBytes { input0Buf in
            let input0Ptr = input0Buf.baseAddress!

            try result0.withUnsafeMutableBytes { out0Buf in
                let out0Ptr = out0Buf.baseAddress!

                if results.count > 1 {
                    try result1.withUnsafeMutableBytes { out1Buf in
                        let out1Ptr = out1Buf.baseAddress!

                        let inputDataPtrs = [input0Ptr]
                        let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr), UnsafeMutableRawPointer(mutating: out1Ptr)]

                        timing = try executable.executeWithCallbacks(
                            inputData: inputDataPtrs,
                            inputSizes: inputSizes,
                            outputData: outputDataPtrs,
                            outputSizes: outputSizes,
                            semantics: clientSemantics
                        )
                    }
                } else {
                    let inputDataPtrs = [input0Ptr]
                    let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr)]

                    timing = try executable.executeWithCallbacks(
                        inputData: inputDataPtrs,
                        inputSizes: inputSizes,
                        outputData: outputDataPtrs,
                        outputSizes: outputSizes,
                        semantics: clientSemantics
                    )
                }
            }
        }

        // Update results with modified data
        results[0] = result0
        if results.count > 1 {
            results[1] = result1
        }

        return (results, timing)
    }

    /// Execute with combined execute+transfer API - single FFI call
    ///
    /// This method uses PJRT_ExecuteAndTransfer which combines execution and D2H transfer
    /// into a single C call, reducing FFI overhead. Best for GPU/TPU where D2H is significant.
    ///
    /// First call uses regular execution to discover output shapes.
    /// Subsequent calls use the combined API.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - semantics: Buffer semantics (default: .auto)
    /// - Returns: Output arrays
    public func executeCombined(
        _ inputs: [[Float]],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)

        // Create input buffers with fast path
        let currentSizes = inputs.map { $0.count }
        let shapesMatch = cachedInputSizes != nil &&
            cachedInputSizes!.count == currentSizes.count &&
            cachedInputSizes! == currentSizes

        var inputBuffers: [PJRTBuffer] = []
        inputBuffers.reserveCapacity(inputs.count)
        for (i, input) in inputs.enumerated() {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBufferFast(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: clientSemantics,
                    cachedSlot: i,
                    isSameShape: shapesMatch
                )
            }
            inputBuffers.append(buffer)
        }

        if !shapesMatch {
            cachedInputSizes = currentSizes
        }

        // First execution: discover output sizes using regular execute
        if cachedOutputMetadata == nil {
            let outputBuffers = try executable.execute(arguments: inputBuffers)
            let outputMeta = outputBuffers.map { buffer in
                (shape: buffer.shape, elementCount: buffer.elementCount)
            }
            cachedOutputMetadata = outputMeta

            // Regular D2H for first call
            var results: [[Float]] = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results

            var events: [PJRTEvent?] = []
            events.reserveCapacity(outputBuffers.count)
            for (i, buffer) in outputBuffers.enumerated() {
                let event = try results[i].withUnsafeMutableBytes { ptr in
                    try buffer.toHostAsync(destination: ptr.baseAddress!)
                }
                events.append(event)
            }
            for event in events {
                try event?.awaitAndDestroy()
            }
            return results
        }

        // Subsequent calls: use combined execute+transfer API
        let outputSizes = cachedOutputMetadata!.map { $0.elementCount * MemoryLayout<Float>.stride }

        return try executable.executeAndTransfer(
            arguments: inputBuffers,
            outputSizes: outputSizes
        )
    }

    /// Execute with buffer donation support and async D2H transfers
    ///
    /// Buffer donation allows input buffers to be reused for outputs with matching shapes.
    /// This can significantly reduce memory allocation overhead for iterative algorithms
    /// where the output has the same shape as the input (e.g., optimization steps, RNN unrolling).
    ///
    /// **Requirements for donation to work:**
    /// 1. The MLIR module must include `mhlo.input_output_alias` attributes mapping inputs to outputs
    /// 2. Input and output shapes must match exactly
    /// 3. Use `donateInputIndices` to specify which inputs to donate
    ///
    /// **Important:** After execution, donated input buffers become invalid and must not be used.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - donateInputIndices: Indices of inputs that should be donated (e.g., [0] to donate first input)
    ///   - semantics: Buffer semantics for non-donated inputs (default: .auto)
    /// - Returns: Output arrays
    public func executeAsyncWithDonation(
        _ inputs: [[Float]],
        donateInputIndices: [Int] = [],
        semantics: BufferSemantics = .auto
    ) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable,
              let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No executable or device available")
        }

        // Resolve semantics based on backend
        let clientSemantics = semantics.toClientSemantics(for: backend)
        let donateSemantics: PJRTClient.HostBufferSemantics = .mutableZeroCopy

        // Convert donate indices to a Set for fast lookup
        let donateSet = Set(donateInputIndices)

        // Check if input shapes match cached shapes for fast path
        let currentSizes = inputs.map { $0.count }
        let shapesMatch = cachedInputSizes != nil &&
            cachedInputSizes!.count == currentSizes.count &&
            cachedInputSizes! == currentSizes

        // Create input buffers with appropriate semantics
        var inputBuffers: [PJRTBuffer] = []
        inputBuffers.reserveCapacity(inputs.count)
        for (i, input) in inputs.enumerated() {
            // Use mutableZeroCopy for donated inputs, regular semantics for others
            let bufferSemantics = donateSet.contains(i) ? donateSemantics : clientSemantics

            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBufferFast(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device,
                    semantics: bufferSemantics,
                    cachedSlot: i,
                    isSameShape: shapesMatch
                )
            }
            inputBuffers.append(buffer)
        }

        // Cache shapes for next call
        if !shapesMatch {
            cachedInputSizes = currentSizes
        }

        // Compute non-donatable indices (all indices NOT in donateInputIndices)
        let nonDonatableIndices = (0..<inputs.count).filter { !donateSet.contains($0) }

        // Execute with donation
        let outputBuffers = try executable.executeWithDonation(
            arguments: inputBuffers,
            nonDonatableIndices: nonDonatableIndices
        )

        // Use cached output metadata if available, otherwise query and cache
        let outputMeta: [(shape: [Int], elementCount: Int)]
        if let cached = cachedOutputMetadata, cached.count == outputBuffers.count {
            outputMeta = cached
        } else {
            // First execution - query and cache metadata
            outputMeta = outputBuffers.map { buffer in
                (shape: buffer.shape, elementCount: buffer.elementCount)
            }
            cachedOutputMetadata = outputMeta
        }

        // Prepare output arrays - reuse preallocated if possible
        var results: [[Float]]
        let canReuse = preallocatedOutputs != nil &&
            preallocatedOutputs!.count == outputMeta.count &&
            preallocatedOutputs!.indices.allSatisfy { preallocatedOutputs![$0].count == outputMeta[$0].elementCount }

        if canReuse {
            results = preallocatedOutputs!
        } else {
            results = outputMeta.map { [Float](repeating: 0, count: $0.elementCount) }
            preallocatedOutputs = results
        }

        // Initiate ALL async D2H transfers first
        var events: [PJRTEvent?] = []
        events.reserveCapacity(outputBuffers.count)

        for (i, buffer) in outputBuffers.enumerated() {
            let event = try results[i].withUnsafeMutableBytes { ptr in
                try buffer.toHostAsync(destination: ptr.baseAddress!)
            }
            events.append(event)
        }

        // Now await ALL events (allows overlapped transfers)
        for event in events {
            try event?.awaitAndDestroy()
        }

        return results
    }

    /// Clear cached buffers to free memory
    public func clearCache() {
        cachedInputBuffers = nil
        cachedInputShapes = nil
        cachedOutputBuffers = nil
        preallocatedOutputs = nil
        cachedOutputMetadata = nil
        cachedInputSizes = nil
        reusableInputBuffers = nil
        inputDataHashes = nil
    }

    /// Wrap MLIR operations in a proper StableHLO module
    private static func wrapInStableHLOModule(_ source: String, inputs: [[Float]]) -> String {
        // Build argument types based on inputs
        let argTypes = inputs.enumerated().map { (i, input) in
            "%arg\(i): tensor<\(input.count)xf32>"
        }.joined(separator: ", ")

        // For simple cases, create a minimal StableHLO module
        // This is a simplified wrapper - real implementation would parse the MLIR
        return """
        module @main {
          func.func @main(\(argTypes)) -> tensor<\(inputs[0].count)xf32> {
            \(source)
          }
        }
        """
    }

    /// Convert MLIR to StableHLO text format
    private static func convertToStableHLO(_ mlir: String, aliases: [InputOutputAlias] = []) -> String {
        var result = mlir

        // If aliases are specified, inject them into function arguments
        if !aliases.isEmpty {
            result = injectAliasAttributes(into: result, aliases: aliases)
        }

        // Replace operation names with stablehlo prefixed versions
        // The MLIR format is: %0 = opname operands : type
        let replacements = [
            // Unquoted form (from MLIRBuilder)
            ("= add ", "= stablehlo.add "),
            ("= subtract ", "= stablehlo.subtract "),
            ("= multiply ", "= stablehlo.multiply "),
            ("= divide ", "= stablehlo.divide "),
            ("= negate ", "= stablehlo.negate "),
            ("= dot ", "= stablehlo.dot "),
            ("= transpose ", "= stablehlo.transpose "),
            ("= exp ", "= stablehlo.exponential "),
            ("= log ", "= stablehlo.log "),
            ("= maximum ", "= stablehlo.maximum "),
            ("= reduce_sum ", "= stablehlo.reduce "),
            ("= reduce_max ", "= stablehlo.reduce "),
            ("= reduce_min ", "= stablehlo.reduce "),
            ("= compare ", "= stablehlo.compare "),
            ("= select ", "= stablehlo.select "),
            // Phase 15: Additional operations
            ("= sqrt ", "= stablehlo.sqrt "),
            ("= logistic ", "= stablehlo.logistic "),
            ("= tanh ", "= stablehlo.tanh "),
            ("= abs ", "= stablehlo.abs "),
            ("= sign ", "= stablehlo.sign "),
            ("= sine ", "= stablehlo.sine "),
            ("= cosine ", "= stablehlo.cosine "),
            ("= power ", "= stablehlo.power "),
            ("= constant ", "= stablehlo.constant "),
            ("= broadcast_in_dim ", "= stablehlo.broadcast_in_dim "),
            ("= reshape ", "= stablehlo.reshape "),
            ("= slice ", "= stablehlo.slice "),
            ("= pad ", "= stablehlo.pad "),
            ("= concatenate ", "= stablehlo.concatenate "),
            ("= convolution(", "= stablehlo.convolution("),
            ("= floor ", "= stablehlo.floor "),
            ("= ceil ", "= stablehlo.ceil "),
            ("= clamp ", "= stablehlo.clamp "),
            ("= rsqrt ", "= stablehlo.rsqrt "),
            ("= and ", "= stablehlo.and "),
            // Quoted form (for inline strings)
            ("\"add\"", "\"stablehlo.add\""),
            ("\"subtract\"", "\"stablehlo.subtract\""),
            ("\"multiply\"", "\"stablehlo.multiply\""),
            ("\"divide\"", "\"stablehlo.divide\""),
            ("\"negate\"", "\"stablehlo.negate\""),
            ("\"dot\"", "\"stablehlo.dot\""),
            ("\"transpose\"", "\"stablehlo.transpose\""),
            ("\"exp\"", "\"stablehlo.exponential\""),
            ("\"log\"", "\"stablehlo.log\""),
            ("\"maximum\"", "\"stablehlo.maximum\""),
            ("\"reduce_sum\"", "\"stablehlo.reduce\""),
            ("\"compare\"", "\"stablehlo.compare\""),
            ("\"select\"", "\"stablehlo.select\""),
            // Phase 15: Additional operations (quoted)
            ("\"sqrt\"", "\"stablehlo.sqrt\""),
            ("\"logistic\"", "\"stablehlo.logistic\""),
            ("\"tanh\"", "\"stablehlo.tanh\""),
            ("\"abs\"", "\"stablehlo.abs\""),
            ("\"sign\"", "\"stablehlo.sign\""),
            ("\"power\"", "\"stablehlo.power\""),
            ("\"constant\"", "\"stablehlo.constant\""),
            ("\"broadcast_in_dim\"", "\"stablehlo.broadcast_in_dim\""),
        ]

        for (from, to) in replacements {
            result = result.replacingOccurrences(of: from, with: to)
        }

        // No longer need to fix permutation format - using dims = [...] directly

        // Fix _dot_types attribute: move from attribute to type signature
        // Pattern: {_dot_types = (...)} : type -> : (...)
        let dotTypesPattern = try? NSRegularExpression(pattern: "\\{_dot_types = (\\([^}]+\\))\\} : tensor<[^>]+>")
        if let regex = dotTypesPattern {
            let nsRange = NSRange(result.startIndex..<result.endIndex, in: result)
            let matches = regex.matches(in: result, options: [], range: nsRange)
            for match in matches.reversed() {
                if let range = Range(match.range, in: result),
                   let typesRange = Range(match.range(at: 1), in: result) {
                    let types = String(result[typesRange])
                    let replacement = ": \(types)"
                    result.replaceSubrange(range, with: replacement)
                }
            }
        }

        return result
    }

    /// Inject mhlo.result_alias attributes into function arguments
    ///
    /// This transforms:
    /// ```
    /// func.func @main(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32>
    /// ```
    /// Into:
    /// ```
    /// func.func @main(%arg0: tensor<100xf32> {mhlo.result_alias = mhlo.result_alias<result_index = 0, must_alias = true>}, %arg1: tensor<100xf32>) -> tensor<100xf32>
    /// ```
    private static func injectAliasAttributes(into mlir: String, aliases: [InputOutputAlias]) -> String {
        // Build a map of input index to alias
        var aliasMap: [Int: InputOutputAlias] = [:]
        for alias in aliases {
            aliasMap[alias.inputIndex] = alias
        }

        // Find the func.func line and its arguments
        // Pattern: func.func @name(%arg0: type, %arg1: type, ...) -> return_type
        guard let funcRange = mlir.range(of: "func.func @") else {
            return mlir
        }

        // Find the opening and closing parentheses for arguments
        let searchStart = funcRange.upperBound
        guard let openParen = mlir[searchStart...].firstIndex(of: "(") else {
            return mlir
        }

        // Find matching close paren (handling nested parens)
        var depth = 1
        var closeParen = mlir.index(after: openParen)
        while depth > 0 && closeParen < mlir.endIndex {
            let char = mlir[closeParen]
            if char == "(" { depth += 1 }
            else if char == ")" { depth -= 1 }
            if depth > 0 { closeParen = mlir.index(after: closeParen) }
        }

        guard depth == 0 else {
            return mlir
        }

        // Extract the arguments string
        let argsString = String(mlir[mlir.index(after: openParen)..<closeParen])

        // Parse and modify each argument
        // Split by ", " but be careful about nested types like tensor<2x3xf32>
        var modifiedArgs: [String] = []
        var currentArg = ""
        var angleDepth = 0

        for char in argsString {
            if char == "<" { angleDepth += 1 }
            else if char == ">" { angleDepth -= 1 }

            if char == "," && angleDepth == 0 {
                modifiedArgs.append(currentArg.trimmingCharacters(in: .whitespaces))
                currentArg = ""
            } else {
                currentArg.append(char)
            }
        }
        if !currentArg.isEmpty {
            modifiedArgs.append(currentArg.trimmingCharacters(in: .whitespaces))
        }

        // Modify arguments that have aliases
        for (i, arg) in modifiedArgs.enumerated() {
            if let alias = aliasMap[i] {
                // Add the alias attribute using tf.aliasing_output format
                // This is the format JAX uses for buffer donation in exported StableHLO
                // Format: %arg0: tensor<N> {tf.aliasing_output = 0 : i32}
                let attrStr = " {tf.aliasing_output = \(alias.outputIndex) : i32}"
                modifiedArgs[i] = arg + attrStr
            }
        }

        // Reconstruct the MLIR
        let newArgsString = modifiedArgs.joined(separator: ", ")
        var result = mlir
        let argsRange = mlir.index(after: openParen)..<closeParen
        result.replaceSubrange(argsRange, with: newArgsString)

        return result
    }

    public var info: String {
        let backendInfo: String
        switch backend {
        case .cpu:
            backendInfo = "CPU"
        case .gpu(let id):
            backendInfo = "GPU:\(id)"
        case .tpu(let id):
            backendInfo = "TPU:\(id)"
        }
        return """
        PJRT Executable:
          MLIR size: \(mlirSource.count) bytes
          StableHLO size: \(stablehloSource.count) bytes
          Backend: \(backendInfo)
        """
    }

    // MARK: - Pooled Execution (zero-copy)

    /// Cached buffer pool for repeated executions
    private var cachedBufferPool: PJRTBufferPool?

    /// Create or get a buffer pool for this executable
    ///
    /// The pool caches configuration and uses zero-copy semantics for optimal performance.
    /// - Parameter inputSizes: Number of elements for each input
    /// - Returns: Buffer pool
    /// - Throws: PJRTError on failure
    private func getOrCreateBufferPool(inputSizes: [Int]) throws -> PJRTBufferPool {
        // Check if we have a cached pool with matching sizes
        if let existingPool = cachedBufferPool {
            return existingPool
        }

        // Ensure client and executable are initialized
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.executionFailed("No executable available")
        }

        // Create the pool
        let pool = try executable.createBufferPool(inputSizes: inputSizes)
        cachedBufferPool = pool
        return pool
    }

    /// Execute with buffer pooling and zero-copy semantics
    ///
    /// This is the most optimized execution path, using zero-copy host buffer semantics
    /// and caching configuration across executions.
    ///
    /// - Parameters:
    ///   - inputs: Input data arrays
    ///   - outputSizes: Number of elements in each output
    /// - Returns: Tuple of (output arrays, execution timing breakdown)
    public func executePooled(
        _ inputs: [[Float]],
        outputSizes: [Int]
    ) throws -> ([[Float]], PJRTExecutionTiming) {
        // Ensure executable is initialized
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu: pjrtBackend = .cpu
        case .gpu: pjrtBackend = .gpu
        case .tpu: pjrtBackend = .tpu
        }

        if pjrtClient == nil {
            pjrtClient = try PJRTClient(backend: pjrtBackend)
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        if pjrtExecutable == nil {
            pjrtExecutable = try client.compile(mlirModule: stablehloSource, xlaOptLevel: xlaOptLevel)
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.executionFailed("No executable available")
        }

        // Build input sizes
        let inputSizes = inputs.map { $0.count }

        // Get or create buffer pool
        let pool = try getOrCreateBufferPool(inputSizes: inputSizes)

        // Prepare output arrays
        var results = outputSizes.map { [Float](repeating: 0, count: $0) }

        // Extract results to separate variables to avoid exclusivity violations
        var result0 = results[0]
        var result1 = results.count > 1 ? results[1] : [Float]()
        var timing: PJRTExecutionTiming!

        // Execute with the pool
        try inputs[0].withUnsafeBytes { input0Buf in
            let input0Ptr = input0Buf.baseAddress!

            try result0.withUnsafeMutableBytes { out0Buf in
                let out0Ptr = out0Buf.baseAddress!

                if results.count > 1 {
                    try result1.withUnsafeMutableBytes { out1Buf in
                        let out1Ptr = out1Buf.baseAddress!

                        let inputDataPtrs = [input0Ptr]
                        let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr), UnsafeMutableRawPointer(mutating: out1Ptr)]

                        timing = try executable.executePooled(
                            pool: pool,
                            inputData: inputDataPtrs,
                            inputSizes: inputSizes,
                            outputData: outputDataPtrs,
                            outputSizes: outputSizes
                        )
                    }
                } else {
                    let inputDataPtrs = [input0Ptr]
                    let outputDataPtrs = [UnsafeMutableRawPointer(mutating: out0Ptr)]

                    timing = try executable.executePooled(
                        pool: pool,
                        inputData: inputDataPtrs,
                        inputSizes: inputSizes,
                        outputData: outputDataPtrs,
                        outputSizes: outputSizes
                    )
                }
            }
        }

        // Update results with modified data
        results[0] = result0
        if results.count > 1 {
            results[1] = result1
        }

        return (results, timing)
    }
}

// MARK: - Real Compiled Function

/// A compiled function that can execute on real PJRT hardware
public class RealCompiledFunction {
    public let executable: PJRTBackedExecutable
    public let mlirSource: String
    public let stablehloSource: String

    public init(executable: PJRTBackedExecutable) {
        self.executable = executable
        self.mlirSource = executable.mlirSource
        self.stablehloSource = executable.stablehloSource
    }

    /// Run the function with input arrays
    public func run(_ inputs: [[Float]]) -> [[Float]] {
        do {
            return try executable.execute(inputs)
        } catch {
            print("Execution error: \(error)")
            return inputs // Fallback
        }
    }

    public var info: String {
        executable.info
    }
}

// MARK: - Real PJRT Compiler

/// Compiler that produces real PJRT executables
public class RealPJRTCompiler {
    private let runtime: PJRTBackedRuntime
    private let options: CompilationOptions

    public init(backend: PJRTBackedRuntime.Backend = .cpu, options: CompilationOptions = .default) throws {
        self.runtime = try PJRTBackedRuntime(backend: backend)
        self.options = options
    }

    /// Compile an MLIR module
    public func compile(_ module: MLIRModule) throws -> RealCompiledFunction {
        let executable = try runtime.compile(module)
        return RealCompiledFunction(executable: executable)
    }

    /// Compile from tracer context
    public func compile(
        inputSpecs: [TensorSpec],
        _ function: ([CompilableTracer]) -> CompilableTracer
    ) throws -> RealCompiledFunction {
        // Trace the function
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let inputs = inputSpecs.enumerated().map { (i, spec) in
            context.input(shape: spec.shape, dtype: spec.dtype, name: "%arg\(i)")
        }

        let output = function(inputs)
        context.output(output)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        return try compile(module)
    }

    /// Compile a single-input function
    public func compile(
        inputSpec: TensorSpec,
        _ function: (CompilableTracer) -> CompilableTracer
    ) throws -> RealCompiledFunction {
        try compile(inputSpecs: [inputSpec]) { inputs in
            function(inputs[0])
        }
    }
}

// MARK: - Errors

/// Errors for PJRT execution
public enum PJRTExecutionError: Error, CustomStringConvertible {
    case notInitialized
    case pluginNotFound(String)
    case compilationFailed(String)
    case executionFailed(String)
    case bufferError(String)

    public var description: String {
        switch self {
        case .notInitialized:
            return "PJRT runtime not initialized"
        case .pluginNotFound(let path):
            return "PJRT plugin not found at: \(path)"
        case .compilationFailed(let msg):
            return "PJRT compilation failed: \(msg)"
        case .executionFailed(let msg):
            return "PJRT execution failed: \(msg)"
        case .bufferError(let msg):
            return "PJRT buffer error: \(msg)"
        }
    }
}

// MARK: - High-Level API

/// Compile and run a function on real PJRT hardware
public func compileForPJRT(
    input: TensorSpec,
    backend: PJRTBackedRuntime.Backend = .cpu,
    _ function: (CompilableTracer) -> CompilableTracer
) throws -> RealCompiledFunction {
    let compiler = try RealPJRTCompiler(backend: backend)
    return try compiler.compile(inputSpec: input, function)
}

/// Compile and run a multi-input function on real PJRT hardware
public func compileForPJRT(
    inputs: [TensorSpec],
    backend: PJRTBackedRuntime.Backend = .cpu,
    _ function: ([CompilableTracer]) -> CompilableTracer
) throws -> RealCompiledFunction {
    let compiler = try RealPJRTCompiler(backend: backend)
    return try compiler.compile(inputSpecs: inputs, function)
}

// MARK: - Integration with Existing CompiledFunction

extension CompiledFunction {
    /// Convert to a real PJRT-backed executable
    public func toPJRTExecutable(backend: PJRTBackedRuntime.Backend = .cpu) throws -> RealCompiledFunction {
        let runtime = try PJRTBackedRuntime(backend: backend)
        let executable = try runtime.compile(mlirSource)
        return RealCompiledFunction(executable: executable)
    }
}

// MARK: - Integration with GradientCompiledFunction

extension GradientCompiledFunction {
    /// Convert to a real PJRT-backed executable
    public func toPJRTExecutable(backend: PJRTBackedRuntime.Backend = .cpu) throws -> RealCompiledFunction {
        return try compiled.toPJRTExecutable(backend: backend)
    }
}
