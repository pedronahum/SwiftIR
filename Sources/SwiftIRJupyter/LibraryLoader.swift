// SwiftIRJupyter - Pure Swift bridge for Jupyter/Colab
// Uses dlopen/dlsym to load native libraries at runtime
// No C++ interop required - works in LLDB REPL

import Foundation

#if os(Linux)
import Glibc
#elseif os(macOS)
import Darwin
#endif

// MARK: - Library Loader

/// Manages dynamic loading of native libraries
public final class LibraryLoader: @unchecked Sendable {

    /// Shared instance
    nonisolated(unsafe) public static let shared = LibraryLoader()

    /// Loaded library handles
    private var handles: [String: UnsafeMutableRawPointer] = [:]

    /// SDK path (set via environment or explicitly)
    public var sdkPath: String {
        ProcessInfo.processInfo.environment["SWIFTIR_DEPS"] ?? "/opt/swiftir-deps"
    }

    private init() {}

    /// Load a library by name (without lib prefix or .so suffix)
    @discardableResult
    public func load(_ name: String) throws -> UnsafeMutableRawPointer {
        if let handle = handles[name] {
            return handle
        }

        let libPath = "\(sdkPath)/lib/lib\(name).so"

        guard let handle = dlopen(libPath, RTLD_NOW | RTLD_GLOBAL) else {
            let error = String(cString: dlerror())
            throw SwiftIRJupyterError.libraryLoadFailed(name: name, path: libPath, error: error)
        }

        handles[name] = handle
        return handle
    }

    /// Load a library by full path
    @discardableResult
    public func loadPath(_ path: String) throws -> UnsafeMutableRawPointer {
        let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent

        if let handle = handles[name] {
            return handle
        }

        guard let handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL) else {
            let error = String(cString: dlerror())
            throw SwiftIRJupyterError.libraryLoadFailed(name: name, path: path, error: error)
        }

        handles[name] = handle
        return handle
    }

    /// Get a symbol from a loaded library
    public func symbol<T>(_ name: String, from library: String) throws -> T {
        guard let handle = handles[library] else {
            throw SwiftIRJupyterError.libraryNotLoaded(name: library)
        }

        guard let sym = dlsym(handle, name) else {
            let error = String(cString: dlerror())
            throw SwiftIRJupyterError.symbolNotFound(symbol: name, library: library, error: error)
        }

        return unsafeBitCast(sym, to: T.self)
    }

    /// Get a symbol from any loaded library
    public func symbol<T>(_ name: String) throws -> T {
        // Search through loaded libraries first (most reliable)
        for (_, handle) in handles {
            if let sym = dlsym(handle, name) {
                return unsafeBitCast(sym, to: T.self)
            }
        }

        // Try RTLD_DEFAULT as fallback (searches all loaded libraries)
        // RTLD_DEFAULT is ((void*)0) on Linux/glibc
        if let sym = dlsym(nil, name) {
            return unsafeBitCast(sym, to: T.self)
        }

        throw SwiftIRJupyterError.symbolNotFound(symbol: name, library: "any", error: "Symbol not found in any loaded library")
    }

    /// Unload all libraries
    public func unloadAll() {
        for (_, handle) in handles {
            dlclose(handle)
        }
        handles.removeAll()
    }

    /// Check if a library is loaded
    public func isLoaded(_ name: String) -> Bool {
        handles[name] != nil
    }

    /// List all loaded libraries
    public var loadedLibraries: [String] {
        Array(handles.keys)
    }
}

// MARK: - Errors

public enum SwiftIRJupyterError: Error, CustomStringConvertible {
    case libraryLoadFailed(name: String, path: String, error: String)
    case libraryNotLoaded(name: String)
    case symbolNotFound(symbol: String, library: String, error: String)
    case mlirError(message: String)
    case pjrtError(code: Int32, message: String)
    case invalidState(message: String)
    case initializationFailed(message: String)

    public var description: String {
        switch self {
        case .libraryLoadFailed(let name, let path, let error):
            return "Failed to load library '\(name)' from '\(path)': \(error)"
        case .libraryNotLoaded(let name):
            return "Library '\(name)' is not loaded"
        case .symbolNotFound(let symbol, let library, let error):
            return "Symbol '\(symbol)' not found in library '\(library)': \(error)"
        case .mlirError(let message):
            return "MLIR error: \(message)"
        case .pjrtError(let code, let message):
            return "PJRT error (\(code)): \(message)"
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .initializationFailed(let message):
            return "Initialization failed: \(message)"
        }
    }
}
