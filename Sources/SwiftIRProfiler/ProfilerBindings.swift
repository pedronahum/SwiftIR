// ProfilerBindings.swift - Dynamic bindings for TSL profiler via dlopen
//
// This module provides pure Swift bindings to the TSL profiler library,
// enabling TensorBoard-compatible profiling in Jupyter/REPL environments.
//
// Usage:
//   try ProfilerBindings.shared.load()
//   let session = try ProfilerBindings.shared.createSession()
//   // ... do work with TraceMe annotations ...
//   let data = try ProfilerBindings.shared.collectData(session)
//   try ProfilerBindings.shared.exportToFile(data, filepath: "/tmp/profile/trace.xspace.pb")
//
// The new TSL-based library provides real XSpace profiling data compatible
// with TensorBoard's profiler plugin.

import Foundation

// MARK: - Error Types

/// Errors that can occur during profiler operations
public enum ProfilerError: Error, CustomStringConvertible {
    case libraryNotLoaded
    case libraryLoadFailed(path: String, error: String)
    case symbolNotFound(symbol: String, error: String)
    case sessionCreationFailed(error: String)
    case sessionAlreadyActive
    case sessionNotActive
    case exportFailed(error: String)
    case invalidArgument(message: String)
    case unsupported(feature: String)
    case internalError(message: String)
    case collectDataFailed

    public var description: String {
        switch self {
        case .libraryNotLoaded:
            return "Profiler library not loaded. Call ProfilerBindings.shared.load() first."
        case .libraryLoadFailed(let path, let error):
            return "Failed to load profiler library from '\(path)': \(error)"
        case .symbolNotFound(let symbol, let error):
            return "Symbol '\(symbol)' not found: \(error)"
        case .sessionCreationFailed(let error):
            return "Failed to create profiler session: \(error)"
        case .sessionAlreadyActive:
            return "Profiler session is already active"
        case .sessionNotActive:
            return "Profiler session is not active"
        case .exportFailed(let error):
            return "Failed to export profile: \(error)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .unsupported(let feature):
            return "Feature not supported: \(feature)"
        case .internalError(let message):
            return "Internal error: \(message)"
        case .collectDataFailed:
            return "Failed to collect profiler data"
        }
    }
}

// MARK: - Error Code Mapping

/// Maps to profiler status codes
public enum ProfilerErrorCode: Int32 {
    case ok = 0
    case invalidArgument = 1
    case notInitialized = 2
    case alreadyActive = 3
    case notActive = 4
    case exportFailed = 5
    case internalError = 6
    case unsupported = 7
}

// MARK: - Profiler Options

/// Configuration options for a profiler session
public struct ProfilerOptions {
    /// Host tracer level (0-3): Controls detail of host-side traces
    /// 0: Disabled, 1: Minimal, 2: Moderate (default), 3: Verbose
    public var hostTracerLevel: Int32

    /// Device tracer level (0-2): Controls device-side tracing
    /// 0: Disabled, 1: Enabled (default), 2: Verbose
    public var deviceTracerLevel: Int32

    /// Python tracer level (0-1): For Python stack traces
    /// 0: Disabled (default for Swift), 1: Enabled
    public var pythonTracerLevel: Int32

    /// Whether to include HLO proto in the trace
    public var includeHLOProto: Bool

    /// Duration hint in milliseconds (0 = no limit)
    public var durationMs: UInt64

    /// Create default options
    public init() {
        self.hostTracerLevel = 2
        self.deviceTracerLevel = 1
        self.pythonTracerLevel = 0
        self.includeHLOProto = true
        self.durationMs = 0
    }

    /// Create custom options
    public init(
        hostTracerLevel: Int32 = 2,
        deviceTracerLevel: Int32 = 1,
        pythonTracerLevel: Int32 = 0,
        includeHLOProto: Bool = true,
        durationMs: UInt64 = 0
    ) {
        self.hostTracerLevel = hostTracerLevel
        self.deviceTracerLevel = deviceTracerLevel
        self.pythonTracerLevel = pythonTracerLevel
        self.includeHLOProto = includeHLOProto
        self.durationMs = durationMs
    }
}

// MARK: - C Structure Mirror

/// Mirrors SwiftIRProfilerOptions from C header
/// Note: This struct must match the C layout exactly
@frozen
public struct CProfilerOptions {
    public var host_tracer_level: Int32
    public var device_tracer_level: Int32
    public var python_tracer_level: Int32
    public var include_hlo_proto: Int32  // Use Int32 for C bool compatibility
    public var duration_ms: UInt64

    public init(from options: ProfilerOptions) {
        self.host_tracer_level = options.hostTracerLevel
        self.device_tracer_level = options.deviceTracerLevel
        self.python_tracer_level = options.pythonTracerLevel
        self.include_hlo_proto = options.includeHLOProto ? 1 : 0
        self.duration_ms = options.durationMs
    }
}

// MARK: - Opaque Handle Types

/// Opaque handle to a profiler session
public struct ProfilerSessionHandle: Equatable {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}

/// Opaque handle to a TraceMe scope
public struct TraceMeHandle: Equatable {
    public let ptr: UnsafeMutableRawPointer?

    public init(_ ptr: UnsafeMutableRawPointer?) {
        self.ptr = ptr
    }

    public var isNull: Bool { ptr == nil }
}


// MARK: - Function Type Definitions

/// C function pointer types for TSL profiler operations
/// Matches the API exported by libswiftir_profiler.so
public enum ProfilerFunctions {
    // Version and capability
    public typealias Version = @convention(c) () -> UnsafePointer<CChar>?
    public typealias IsTSLAvailable = @convention(c) () -> Bool

    // Session management
    public typealias CreateSession = @convention(c) () -> UnsafeMutableRawPointer?
    public typealias CreateSessionWithOptions = @convention(c) (
        Int32,  // host_tracer_level
        Int32,  // device_tracer_level
        Bool    // include_dataset_ops
    ) -> UnsafeMutableRawPointer?
    public typealias DestroySession = @convention(c) (UnsafeMutableRawPointer?) -> Void
    public typealias SessionStatus = @convention(c) (UnsafeMutableRawPointer?) -> Int32

    // Data collection
    public typealias CollectData = @convention(c) (
        UnsafeMutableRawPointer?,  // session
        UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>  // out_buffer
    ) -> Int64  // returns size or -1 on error
    public typealias FreeBuffer = @convention(c) (UnsafeMutablePointer<UInt8>?) -> Void

    // TraceMe annotations
    public typealias TraceMeStart = @convention(c) (
        UnsafePointer<CChar>?,  // name
        Int32  // level
    ) -> UInt64  // returns opaque handle
    public typealias TraceMeStop = @convention(c) (UInt64) -> Void  // takes handle
    public typealias TraceMeInstant = @convention(c) (
        UnsafePointer<CChar>?,  // name
        Int32  // level
    ) -> Void

    // File export
    public typealias ExportToFile = @convention(c) (
        UnsafePointer<UInt8>?,  // data
        Int64,  // size
        UnsafePointer<CChar>?  // filepath
    ) -> Int32  // returns 0 on success
}

// MARK: - Profiler Bindings

/// Dynamic bindings for the TSL profiler library (libswiftir_profiler.so)
public final class ProfilerBindings: @unchecked Sendable {
    /// Shared singleton instance
    nonisolated(unsafe) public static let shared = ProfilerBindings()

    /// Whether the library has been loaded
    public private(set) var isLoaded = false

    /// Library handle from dlopen
    private var libraryHandle: UnsafeMutableRawPointer?

    // Function pointers for TSL profiler API
    private var _version: UnsafeMutableRawPointer?
    private var _isTSLAvailable: UnsafeMutableRawPointer?
    private var _createSession: UnsafeMutableRawPointer?
    private var _createSessionWithOptions: UnsafeMutableRawPointer?
    private var _destroySession: UnsafeMutableRawPointer?
    private var _sessionStatus: UnsafeMutableRawPointer?
    private var _collectData: UnsafeMutableRawPointer?
    private var _freeBuffer: UnsafeMutableRawPointer?
    private var _traceMeStart: UnsafeMutableRawPointer?
    private var _traceMeStop: UnsafeMutableRawPointer?
    private var _traceMeInstant: UnsafeMutableRawPointer?
    private var _exportToFile: UnsafeMutableRawPointer?

    private init() {}

    /// Load the profiler library
    /// - Parameter path: Optional explicit path to libswiftir_profiler.so
    public func load(from path: String? = nil) throws {
        guard !isLoaded else { return }

        let libPath = path ?? findLibraryPath()

        guard let handle = dlopen(libPath, RTLD_NOW | RTLD_LOCAL) else {
            let error = String(cString: dlerror())
            throw ProfilerError.libraryLoadFailed(path: libPath, error: error)
        }

        libraryHandle = handle

        // Load all function pointers
        try loadSymbols()

        isLoaded = true
    }

    /// Unload the profiler library
    public func unload() {
        guard isLoaded, let handle = libraryHandle else { return }
        dlclose(handle)
        libraryHandle = nil
        isLoaded = false
        clearSymbols()
    }

    // MARK: - Library Path Discovery

    private func findLibraryPath() -> String {
        // Check environment variable first
        if let envPath = ProcessInfo.processInfo.environment["SWIFTIR_DEPS"] {
            let path = "\(envPath)/lib/libswiftir_profiler.so"
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        // Check common locations
        let searchPaths = [
            "/opt/swiftir-deps/lib/libswiftir_profiler.so",
            "/usr/local/lib/libswiftir_profiler.so",
            "./lib/libswiftir_profiler.so",
            "../lib/libswiftir_profiler.so",
        ]

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        // Default to the standard location
        return "/opt/swiftir-deps/lib/libswiftir_profiler.so"
    }

    // MARK: - Symbol Loading

    private func loadSymbols() throws {
        guard let handle = libraryHandle else {
            throw ProfilerError.libraryNotLoaded
        }

        // Load TSL profiler API symbols
        _version = try loadSymbol(handle, "SwiftIRProfiler_Version")
        _isTSLAvailable = try loadSymbol(handle, "SwiftIRProfiler_IsTSLAvailable")
        _createSession = try loadSymbol(handle, "SwiftIRProfiler_CreateSession")
        _createSessionWithOptions = try loadSymbol(handle, "SwiftIRProfiler_CreateSessionWithOptions")
        _destroySession = try loadSymbol(handle, "SwiftIRProfiler_DestroySession")
        _sessionStatus = try loadSymbol(handle, "SwiftIRProfiler_SessionStatus")
        _collectData = try loadSymbol(handle, "SwiftIRProfiler_CollectData")
        _freeBuffer = try loadSymbol(handle, "SwiftIRProfiler_FreeBuffer")
        _traceMeStart = try loadSymbol(handle, "SwiftIRProfiler_TraceMeStart")
        _traceMeStop = try loadSymbol(handle, "SwiftIRProfiler_TraceMeStop")
        _traceMeInstant = try loadSymbol(handle, "SwiftIRProfiler_TraceMeInstant")
        _exportToFile = try loadSymbol(handle, "SwiftIRProfiler_ExportToFile")
    }

    private func loadSymbol(_ handle: UnsafeMutableRawPointer, _ name: String) throws -> UnsafeMutableRawPointer {
        guard let sym = dlsym(handle, name) else {
            let error = String(cString: dlerror())
            throw ProfilerError.symbolNotFound(symbol: name, error: error)
        }
        return sym
    }

    private func clearSymbols() {
        _version = nil
        _isTSLAvailable = nil
        _createSession = nil
        _createSessionWithOptions = nil
        _destroySession = nil
        _sessionStatus = nil
        _collectData = nil
        _freeBuffer = nil
        _traceMeStart = nil
        _traceMeStop = nil
        _traceMeInstant = nil
        _exportToFile = nil
    }

    // MARK: - Session Management

    /// Create a new profiler session with default options
    public func createSession() throws -> ProfilerSessionHandle {
        guard isLoaded, let fn = _createSession else {
            throw ProfilerError.libraryNotLoaded
        }

        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.CreateSession.self)
        guard let ptr = typedFn() else {
            throw ProfilerError.sessionCreationFailed(error: "Failed to create profiler session")
        }

        return ProfilerSessionHandle(ptr)
    }

    /// Create a new profiler session with custom options
    public func createSession(options: ProfilerOptions) throws -> ProfilerSessionHandle {
        guard isLoaded, let fn = _createSessionWithOptions else {
            throw ProfilerError.libraryNotLoaded
        }

        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.CreateSessionWithOptions.self)
        guard let ptr = typedFn(
            options.hostTracerLevel,
            options.deviceTracerLevel,
            false  // include_dataset_ops - not relevant for Swift
        ) else {
            throw ProfilerError.sessionCreationFailed(error: "Failed to create profiler session with options")
        }

        return ProfilerSessionHandle(ptr)
    }

    /// Destroy a profiler session
    public func destroySession(_ session: ProfilerSessionHandle) {
        guard isLoaded, let fn = _destroySession else { return }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.DestroySession.self)
        typedFn(session.ptr)
    }

    /// Get the status of a profiler session
    /// - Returns: 0 if OK, negative value on error
    public func sessionStatus(_ session: ProfilerSessionHandle) -> Int32 {
        guard isLoaded, let fn = _sessionStatus else { return -1 }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.SessionStatus.self)
        return typedFn(session.ptr)
    }

    // MARK: - Data Collection

    /// Collect profiler data and return as serialized XSpace protobuf
    public func collectData(_ session: ProfilerSessionHandle) throws -> Data {
        guard isLoaded, let fn = _collectData, let freeFn = _freeBuffer else {
            throw ProfilerError.libraryNotLoaded
        }

        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.CollectData.self)
        let typedFreeFn = unsafeBitCast(freeFn, to: ProfilerFunctions.FreeBuffer.self)

        var dataPtr: UnsafeMutablePointer<UInt8>?
        let size = typedFn(session.ptr, &dataPtr)

        if size < 0 {
            throw ProfilerError.collectDataFailed
        }

        guard let ptr = dataPtr, size > 0 else {
            throw ProfilerError.internalError(message: "No data returned")
        }

        let data = Data(bytes: ptr, count: Int(size))
        typedFreeFn(ptr)
        return data
    }

    /// Export data to a file
    public func exportToFile(_ data: Data, filepath: String) throws {
        guard isLoaded, let fn = _exportToFile else {
            throw ProfilerError.libraryNotLoaded
        }

        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.ExportToFile.self)

        let result = data.withUnsafeBytes { dataPtr in
            filepath.withCString { cPath in
                typedFn(
                    dataPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    Int64(data.count),
                    cPath
                )
            }
        }

        if result != 0 {
            throw ProfilerError.exportFailed(error: "Failed to export to file: error code \(result)")
        }
    }

    // MARK: - TraceMe Annotations

    /// Begin a named trace scope
    /// - Returns: An opaque handle to pass to traceMeStop
    public func traceMeStart(_ name: String, level: Int32 = 1) -> UInt64 {
        guard isLoaded, let fn = _traceMeStart else {
            return 0
        }

        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.TraceMeStart.self)
        return name.withCString { cName in
            typedFn(cName, level)
        }
    }

    /// End a trace scope
    public func traceMeStop(_ handle: UInt64) {
        guard isLoaded, let fn = _traceMeStop, handle != 0 else { return }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.TraceMeStop.self)
        typedFn(handle)
    }

    /// Record an instant trace event
    public func traceMeInstant(_ name: String, level: Int32 = 1) {
        guard isLoaded, let fn = _traceMeInstant else { return }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.TraceMeInstant.self)
        name.withCString { cName in
            typedFn(cName, level)
        }
    }

    // MARK: - Utility Functions

    /// Get profiler library version
    public func getVersion() -> String {
        guard isLoaded, let fn = _version else { return "unknown" }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.Version.self)
        guard let cstr = typedFn() else { return "unknown" }
        return String(cString: cstr)
    }

    /// Check if TSL profiler is available
    public func isTSLAvailable() -> Bool {
        guard isLoaded, let fn = _isTSLAvailable else { return false }
        let typedFn = unsafeBitCast(fn, to: ProfilerFunctions.IsTSLAvailable.self)
        return typedFn()
    }

    // MARK: - Legacy API Compatibility

    /// Legacy: Begin a trace scope (returns TraceMeHandle for compatibility)
    @available(*, deprecated, message: "Use traceMeStart(_:level:) -> UInt64 instead")
    public func traceMeStart(_ name: String, level: Int32 = 1, metadataKey: String?, metadataValue: String?) -> TraceMeHandle {
        // Ignore metadata for now - TSL TraceMe doesn't support it directly
        let handle = traceMeStart(name, level: level)
        // Store handle in a pointer for compatibility
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
        ptr.storeBytes(of: handle, as: UInt64.self)
        return TraceMeHandle(ptr)
    }

    /// Legacy: End a trace scope (takes TraceMeHandle for compatibility)
    @available(*, deprecated, message: "Use traceMeStop(_: UInt64) instead")
    public func traceMeStop(_ trace: TraceMeHandle) {
        guard let ptr = trace.ptr else { return }
        let handle = ptr.load(as: UInt64.self)
        traceMeStop(handle)
        ptr.deallocate()
    }

    /// Legacy: Check if a session is currently active
    public func isActive(_ session: ProfilerSessionHandle) -> Bool {
        return sessionStatus(session) == 0
    }

    /// Legacy: Start collecting trace data (TSL sessions auto-start)
    public func start(_ session: ProfilerSessionHandle) throws {
        // TSL ProfilerSession starts automatically on creation
        // This is a no-op for compatibility
    }

    /// Legacy: Stop collecting and export trace to a directory
    public func stopAndExport(_ session: ProfilerSessionHandle, logDir: String) throws {
        let data = try collectData(session)

        // Create directory if needed
        try FileManager.default.createDirectory(
            atPath: logDir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        // Export with timestamp
        let timestamp = Int(Date().timeIntervalSince1970 * 1000)
        let filename = "\(logDir)/swiftir.\(timestamp).xspace.pb"
        try exportToFile(data, filepath: filename)
    }

    /// Legacy: Stop collecting and get raw XSpace data
    public func stopAndGetData(_ session: ProfilerSessionHandle) throws -> Data {
        return try collectData(session)
    }
}
