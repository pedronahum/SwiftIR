// ProfilerBindingsDynamic.swift - Dynamic library bindings for PJRT profiler
//
// This module uses dlopen/dlsym to load profiler functions from libPJRTSimpleWrapper.so.
// This ensures we share the same g_api global as SwiftIRJupyter when running in Jupyter/Colab.
//
// The key insight: PJRTSimpleWrapper.c contains global state (g_api) that stores the PJRT API
// pointer after plugin load. If we statically link PJRTSimpleWrapper.c, we get our OWN copy
// of g_api which is never initialized. By using dlopen, we share the SAME g_api as SwiftIRJupyter.

import Foundation

#if os(Linux)
import Glibc
#elseif os(macOS)
import Darwin
#endif

// MARK: - Dynamic Profiler Bindings

/// Dynamic bindings to PJRT profiler functions via dlopen
/// This ensures we share the same global state as SwiftIRJupyter
public final class ProfilerBindingsDynamic: @unchecked Sendable {

    /// Shared instance
    public static let shared = ProfilerBindingsDynamic()

    /// Whether bindings are loaded
    public private(set) var isLoaded = false

    /// Library handle
    private var handle: UnsafeMutableRawPointer?

    // Function pointer types
    private typealias HasProfilerExtension = @convention(c) () -> Bool
    private typealias GetProfilerApi = @convention(c) () -> UnsafeMutableRawPointer?
    private typealias ProfilerCreate = @convention(c) (UnsafePointer<CChar>?, Int, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
    private typealias ProfilerStart = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias ProfilerStop = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias ProfilerCollectData = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UInt8>?, UnsafeMutablePointer<Int>) -> Int32
    private typealias ProfilerDestroy = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias HasTraceMeApi = @convention(c) () -> Bool
    private typealias TraceMeStart = @convention(c) (UnsafePointer<CChar>, Int32) -> Int64
    private typealias TraceMeStop = @convention(c) (Int64) -> Void
    private typealias TraceMeActive = @convention(c) (Int32) -> Bool
    private typealias TraceMeInstant = @convention(c) (UnsafePointer<CChar>, Int32) -> Void

    // Function pointers
    private var _hasProfilerExtension: HasProfilerExtension?
    private var _getProfilerApi: GetProfilerApi?
    private var _profilerCreate: ProfilerCreate?
    private var _profilerStart: ProfilerStart?
    private var _profilerStop: ProfilerStop?
    private var _profilerCollectData: ProfilerCollectData?
    private var _profilerDestroy: ProfilerDestroy?
    private var _hasTraceMeApi: HasTraceMeApi?
    private var _traceMeStart: TraceMeStart?
    private var _traceMeStop: TraceMeStop?
    private var _traceMeActive: TraceMeActive?
    private var _traceMeInstant: TraceMeInstant?

    private init() {}

    /// Load profiler bindings from libPJRTSimpleWrapper.so
    /// This must be called AFTER SwiftIRJupyter has loaded the library
    public func load() throws {
        guard !isLoaded else { return }

        // Get SDK path from environment
        let sdkPath = ProcessInfo.processInfo.environment["SWIFTIR_DEPS"] ?? "/opt/swiftir-deps"

        // First try to load PJRTProtoHelper (dependency)
        let protoHelperPath = "\(sdkPath)/lib/libPJRTProtoHelper.so"
        if let protoHandle = dlopen(protoHelperPath, RTLD_NOW | RTLD_GLOBAL) {
            // Loaded successfully, keep handle alive
            _ = protoHandle
        }

        // Load PJRTSimpleWrapper with RTLD_GLOBAL so symbols are shared
        let wrapperPath = "\(sdkPath)/lib/libPJRTSimpleWrapper.so"
        guard let h = dlopen(wrapperPath, RTLD_NOW | RTLD_GLOBAL) else {
            let error = String(cString: dlerror())
            throw ProfilerDynamicError.libraryLoadFailed(path: wrapperPath, error: error)
        }

        handle = h

        // Load function pointers
        func loadSymbol<T>(_ name: String) -> T? {
            guard let sym = dlsym(h, name) else { return nil }
            return unsafeBitCast(sym, to: T.self)
        }

        _hasProfilerExtension = loadSymbol("PJRT_HasProfilerExtension")
        _getProfilerApi = loadSymbol("PJRT_GetProfilerApi")
        _profilerCreate = loadSymbol("PJRT_ProfilerCreate")
        _profilerStart = loadSymbol("PJRT_ProfilerStart")
        _profilerStop = loadSymbol("PJRT_ProfilerStop")
        _profilerCollectData = loadSymbol("PJRT_ProfilerCollectData")
        _profilerDestroy = loadSymbol("PJRT_ProfilerDestroy")
        _hasTraceMeApi = loadSymbol("PJRT_HasTraceMeApi")
        _traceMeStart = loadSymbol("PJRT_TraceMeStart")
        _traceMeStop = loadSymbol("PJRT_TraceMeStop")
        _traceMeActive = loadSymbol("PJRT_TraceMeActive")
        _traceMeInstant = loadSymbol("PJRT_TraceMeInstant")

        isLoaded = true
    }

    // MARK: - Profiler Extension

    /// Check if PJRT profiler extension is available
    public func hasProfilerExtension() -> Bool {
        guard let fn = _hasProfilerExtension else { return false }
        return fn()
    }

    /// Get the profiler API pointer
    public func getProfilerApi() -> UnsafeMutableRawPointer? {
        guard let fn = _getProfilerApi else { return nil }
        return fn()
    }

    /// Create a profiler instance
    public func profilerCreate(options: UnsafePointer<CChar>?, optionsSize: Int) throws -> UnsafeMutableRawPointer {
        guard let fn = _profilerCreate else {
            throw ProfilerDynamicError.notLoaded
        }

        var profiler: UnsafeMutableRawPointer?
        let result = fn(options, optionsSize, &profiler)

        guard result == 0, let p = profiler else {
            throw ProfilerDynamicError.createFailed(code: result)
        }

        return p
    }

    /// Start profiling
    public func profilerStart(_ profiler: UnsafeMutableRawPointer) throws {
        guard let fn = _profilerStart else {
            throw ProfilerDynamicError.notLoaded
        }

        let result = fn(profiler)
        guard result == 0 else {
            throw ProfilerDynamicError.startFailed(code: result)
        }
    }

    /// Stop profiling
    public func profilerStop(_ profiler: UnsafeMutableRawPointer) throws {
        guard let fn = _profilerStop else {
            throw ProfilerDynamicError.notLoaded
        }

        let result = fn(profiler)
        guard result == 0 else {
            throw ProfilerDynamicError.stopFailed(code: result)
        }
    }

    /// Collect profiler data
    public func profilerCollectData(_ profiler: UnsafeMutableRawPointer, buffer: UnsafeMutablePointer<UInt8>?, size: inout Int) throws {
        guard let fn = _profilerCollectData else {
            throw ProfilerDynamicError.notLoaded
        }

        let result = fn(profiler, buffer, &size)
        guard result == 0 else {
            throw ProfilerDynamicError.collectFailed(code: result)
        }
    }

    /// Destroy profiler
    public func profilerDestroy(_ profiler: UnsafeMutableRawPointer) {
        guard let fn = _profilerDestroy else { return }
        _ = fn(profiler)
    }

    // MARK: - TraceMe API

    /// Check if TraceMe API is available
    public func hasTraceMeApi() -> Bool {
        guard let fn = _hasTraceMeApi else { return false }
        return fn()
    }

    /// Start a trace activity
    public func traceMeStart(_ name: String, level: Int32) -> Int64 {
        guard let fn = _traceMeStart else { return 0 }
        return name.withCString { cName in
            fn(cName, level)
        }
    }

    /// Stop a trace activity
    public func traceMeStop(_ activityId: Int64) {
        guard let fn = _traceMeStop else { return }
        fn(activityId)
    }

    /// Check if tracing is active
    public func traceMeActive(level: Int32) -> Bool {
        guard let fn = _traceMeActive else { return false }
        return fn(level)
    }

    /// Record an instant trace event
    public func traceMeInstant(_ name: String, level: Int32) {
        guard let fn = _traceMeInstant else { return }
        name.withCString { cName in
            fn(cName, level)
        }
    }
}

// MARK: - Errors

public enum ProfilerDynamicError: Error, CustomStringConvertible {
    case libraryLoadFailed(path: String, error: String)
    case notLoaded
    case createFailed(code: Int32)
    case startFailed(code: Int32)
    case stopFailed(code: Int32)
    case collectFailed(code: Int32)

    public var description: String {
        switch self {
        case .libraryLoadFailed(let path, let error):
            return "Failed to load library '\(path)': \(error)"
        case .notLoaded:
            return "Profiler bindings not loaded"
        case .createFailed(let code):
            return "Failed to create profiler (error \(code))"
        case .startFailed(let code):
            return "Failed to start profiler (error \(code))"
        case .stopFailed(let code):
            return "Failed to stop profiler (error \(code))"
        case .collectFailed(let code):
            return "Failed to collect profiler data (error \(code))"
        }
    }
}
