// Profiler.swift - High-level Swift API for TSL profiling
//
// This module provides a user-friendly interface for profiling SwiftIR computations,
// compatible with TensorBoard and XProf visualization tools.
//
// The profiler uses TSL's ProfilerSession which automatically starts collecting
// data on creation and provides real XSpace protobuf data.
//
// Usage:
//   // Simple tracing
//   try Profiler.trace("/tmp/profile") {
//       // Your computation here
//   }
//
//   // Manual control
//   try Profiler.startTrace("/tmp/profile")
//   // Your computation
//   try Profiler.stopTrace()
//
//   // Named scopes (appear in TensorBoard Trace Viewer)
//   Profiler.scope("forward_pass") {
//       // Appears as "forward_pass" in TensorBoard Trace Viewer
//   }

import Foundation

// MARK: - Profiler Session

/// A profiler session for capturing execution traces
/// TSL ProfilerSession automatically starts collecting on creation
public final class ProfilerSession {
    private let handle: ProfilerSessionHandle
    private let bindings: ProfilerBindings

    /// Create a new profiler session with default options
    /// Note: TSL ProfilerSession starts collecting automatically on creation
    public init() throws {
        self.bindings = ProfilerBindings.shared

        if !bindings.isLoaded {
            try bindings.load()
        }

        self.handle = try bindings.createSession()
    }

    /// Create a new profiler session with custom options
    public init(options: ProfilerOptions) throws {
        self.bindings = ProfilerBindings.shared

        if !bindings.isLoaded {
            try bindings.load()
        }

        self.handle = try bindings.createSession(options: options)
    }

    deinit {
        bindings.destroySession(handle)
    }

    /// Start capturing trace data (no-op for TSL, session starts automatically)
    public func start() throws {
        // TSL ProfilerSession starts automatically on creation
    }

    /// Stop capturing and export to a directory
    /// - Parameter logDir: Directory for TensorBoard-compatible output
    public func stopAndExport(to logDir: String) throws {
        try bindings.stopAndExport(handle, logDir: logDir)
    }

    /// Collect data and return raw XSpace protobuf
    public func collectData() throws -> Data {
        return try bindings.collectData(handle)
    }

    /// Stop capturing and return raw XSpace data (legacy alias for collectData)
    public func stopAndGetData() throws -> Data {
        return try collectData()
    }

    /// Check if the session is OK
    public var active: Bool {
        bindings.sessionStatus(handle) == 0
    }
}

// MARK: - Static Profiler API

/// Main profiler interface providing static methods for easy tracing
public enum Profiler {
    /// Current active profiler session (for internal use)
    /// Protected by lock - disable concurrency warnings
    nonisolated(unsafe) private static var currentSession: ProfilerSession?
    nonisolated(unsafe) private static var _pendingLogDir: String?
    private static let lock = NSLock()

    // MARK: - Trace Capture

    /// Start capturing a trace
    /// - Parameter logDir: Directory where trace files will be written
    public static func startTrace(_ logDir: String) throws {
        lock.lock()
        defer { lock.unlock() }

        if currentSession != nil {
            throw ProfilerError.sessionAlreadyActive
        }

        let session = try ProfilerSession()
        try session.start()
        currentSession = session

        // Store logDir for stopTrace
        _pendingLogDir = logDir
    }

    /// Stop the current trace and export
    public static func stopTrace() throws {
        lock.lock()
        defer { lock.unlock() }

        guard let session = currentSession, let logDir = _pendingLogDir else {
            throw ProfilerError.sessionNotActive
        }

        try session.stopAndExport(to: logDir)
        currentSession = nil
        _pendingLogDir = nil
    }

    /// Capture a trace around a block of code
    /// - Parameters:
    ///   - logDir: Directory for trace output
    ///   - block: Code to profile
    /// - Returns: Result of the block
    /// - Note: This function is marked `throws` because it starts/stops tracing
    public static func trace<T>(_ logDir: String, _ block: () throws -> T) throws -> T {
        try startTrace(logDir)

        defer {
            do {
                try stopTrace()
            } catch {
                print("[SwiftIR Profiler] Warning: Failed to stop trace: \(error)")
            }
        }

        return try block()
    }

    /// Capture a trace around an async block of code
    @available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
    public static func trace<T>(_ logDir: String, _ block: () async throws -> T) async throws -> T {
        try startTrace(logDir)

        defer {
            do {
                try stopTrace()
            } catch {
                print("[SwiftIR Profiler] Warning: Failed to stop trace: \(error)")
            }
        }

        return try await block()
    }

    // MARK: - Named Scopes (TraceMe)

    /// Execute a block within a named scope that appears in the trace
    /// - Parameters:
    ///   - name: Name to display in the Trace Viewer
    ///   - level: Importance level (1=important, 2=moderate, 3=verbose)
    ///   - block: Code to execute within the scope
    /// - Returns: Result of the block
    @discardableResult
    public static func scope<T>(_ name: String, level: Int32 = 1, _ block: () throws -> T) rethrows -> T {
        let handle = ProfilerBindings.shared.traceMeStart(name, level: level)
        defer { ProfilerBindings.shared.traceMeStop(handle) }
        return try block()
    }

    /// Execute a block within a named scope with metadata
    /// Note: Metadata is currently ignored by the TSL profiler
    @discardableResult
    public static func scope<T>(
        _ name: String,
        level: Int32 = 1,
        metadata: (key: String, value: String)?,
        _ block: () throws -> T
    ) rethrows -> T {
        // Note: TSL TraceMe doesn't directly support metadata, use name with info
        var fullName = name
        if let meta = metadata {
            fullName = "\(name) [\(meta.key)=\(meta.value)]"
        }
        let handle = ProfilerBindings.shared.traceMeStart(fullName, level: level)
        defer { ProfilerBindings.shared.traceMeStop(handle) }
        return try block()
    }

    // MARK: - Utility Functions

    /// Check if the TSL profiler (full XSpace support) is available
    public static var isTSLAvailable: Bool {
        ProfilerBindings.shared.isTSLAvailable()
    }

    /// Get profiler library version
    public static var version: String {
        ProfilerBindings.shared.getVersion()
    }

    /// Initialize the profiler library
    /// - Parameter path: Optional explicit path to libswiftir_profiler.so
    public static func initialize(from path: String? = nil) throws {
        try ProfilerBindings.shared.load(from: path)
    }
}

// MARK: - Scoped Trace Helper

/// RAII-style trace scope that automatically ends when deallocated
public final class ScopedTrace {
    private let handle: UInt64

    /// Create a scoped trace
    /// - Parameters:
    ///   - name: Name to display in the Trace Viewer
    ///   - level: Importance level (1=important, 2=moderate, 3=verbose)
    public init(_ name: String, level: Int32 = 1) {
        self.handle = ProfilerBindings.shared.traceMeStart(name, level: level)
    }

    /// Create a scoped trace with metadata
    /// Note: Metadata is encoded in the name for TSL compatibility
    public init(_ name: String, level: Int32 = 1, metadata: (key: String, value: String)?) {
        var fullName = name
        if let meta = metadata {
            fullName = "\(name) [\(meta.key)=\(meta.value)]"
        }
        self.handle = ProfilerBindings.shared.traceMeStart(fullName, level: level)
    }

    deinit {
        ProfilerBindings.shared.traceMeStop(handle)
    }
}

// MARK: - traced() Free Function

/// Convenience function for creating a named trace scope
/// - Parameters:
///   - name: Name to display in the Trace Viewer
///   - level: Importance level (1=important, 2=moderate, 3=verbose)
///   - block: Code to execute within the scope
/// - Returns: Result of the block
@discardableResult
public func traced<T>(_ name: String, level: Int32 = 1, _ block: () throws -> T) rethrows -> T {
    try Profiler.scope(name, level: level, block)
}

/// Async version of traced
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
@discardableResult
public func traced<T>(_ name: String, level: Int32 = 1, _ block: () async throws -> T) async rethrows -> T {
    let handle = ProfilerBindings.shared.traceMeStart(name, level: level)
    defer { ProfilerBindings.shared.traceMeStop(handle) }
    return try await block()
}

// MARK: - Step Markers for TensorBoard Overview Page

/// Create a training step marker that TensorBoard recognizes for step timing
///
/// TensorBoard's Overview Page requires step markers to calculate step time.
/// This function creates a trace event with the name "TrainStep" and step number
/// in a format TensorBoard recognizes.
///
/// Usage:
/// ```swift
/// for step in 0..<numSteps {
///     try trainStep(step) {
///         // Your training code here
///     }
/// }
/// ```
///
/// - Parameters:
///   - step: The step number (0-indexed)
///   - block: The training step code to execute
/// - Returns: Result of the block
@discardableResult
public func trainStep<T>(_ step: Int, _ block: () throws -> T) rethrows -> T {
    // TensorBoard looks for "TrainStep" or similar names with step_num
    // Format: "TrainStep#step_num=N,_r=1#" where _r=1 indicates it's a traced step
    let name = "TrainStep#step_num=\(step),_r=1#"
    let handle = ProfilerBindings.shared.traceMeStart(name, level: 1)
    defer { ProfilerBindings.shared.traceMeStop(handle) }
    return try block()
}

/// Async version of trainStep
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
@discardableResult
public func trainStep<T>(_ step: Int, _ block: () async throws -> T) async rethrows -> T {
    let name = "TrainStep#step_num=\(step),_r=1#"
    let handle = ProfilerBindings.shared.traceMeStart(name, level: 1)
    defer { ProfilerBindings.shared.traceMeStop(handle) }
    return try await block()
}

/// Create a custom step marker with a specific name
///
/// This allows you to create step markers with custom names while still
/// being recognized by TensorBoard for step timing.
///
/// - Parameters:
///   - name: The name of the step (e.g., "Train", "Eval", "Inference")
///   - step: The step number (0-indexed)
///   - block: The step code to execute
/// - Returns: Result of the block
@discardableResult
public func step<T>(_ name: String, _ stepNum: Int, _ block: () throws -> T) rethrows -> T {
    // Format recognized by TensorBoard: "Name#step_num=N,_r=1#"
    let traceName = "\(name)#step_num=\(stepNum),_r=1#"
    let handle = ProfilerBindings.shared.traceMeStart(traceName, level: 1)
    defer { ProfilerBindings.shared.traceMeStop(handle) }
    return try block()
}

/// Async version of step
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
@discardableResult
public func step<T>(_ name: String, _ stepNum: Int, _ block: () async throws -> T) async rethrows -> T {
    let traceName = "\(name)#step_num=\(stepNum),_r=1#"
    let handle = ProfilerBindings.shared.traceMeStart(traceName, level: 1)
    defer { ProfilerBindings.shared.traceMeStop(handle) }
    return try await block()
}
