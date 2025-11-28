// PJRTPlugin.swift - Dynamic PJRT plugin loading
// Copyright 2024 SwiftIR Project
//
// Loads PJRT plugins dynamically using dlopen/dlsym.

import Foundation

#if os(Linux) || os(macOS)
import Glibc
#endif

/// A dynamically loaded PJRT plugin
///
/// `PJRTPlugin` wraps a shared library containing a PJRT implementation.
/// It handles loading the plugin with `dlopen` and finding the `GetPjrtApi`
/// symbol to obtain the PJRT C API.
///
/// ## Usage
/// ```swift
/// // Load plugin for detected accelerator
/// let plugin = try PJRTPlugin.load(for: .cpu)
///
/// // Or load from explicit path
/// let plugin = try PJRTPlugin(path: "/path/to/plugin.so", acceleratorType: .cpu)
/// ```
///
/// The plugin is automatically unloaded when the `PJRTPlugin` instance is deallocated.
public final class PJRTPlugin: @unchecked Sendable {

    /// Handle from dlopen
    public let handle: UnsafeMutableRawPointer

    /// Pointer to the PJRT API struct
    public let api: UnsafeRawPointer

    /// Path to the loaded plugin
    public let path: String

    /// Accelerator type this plugin supports
    public let acceleratorType: AcceleratorType

    /// PJRT API version (major.minor)
    public let apiVersion: (major: Int, minor: Int)

    // MARK: - Initialization

    /// Load a PJRT plugin from the given path
    ///
    /// - Parameters:
    ///   - path: Path to the shared library (.so file)
    ///   - acceleratorType: The accelerator type this plugin supports
    /// - Throws: `SwiftIRError` if loading fails
    public init(path: String, acceleratorType: AcceleratorType) throws {
        self.path = path
        self.acceleratorType = acceleratorType

        // Load the shared library
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let error = String(cString: dlerror())
            throw SwiftIRError.pluginLoadFailed(path: path, reason: error)
        }
        self.handle = handle

        // Find GetPjrtApi symbol
        guard let symbol = dlsym(handle, "GetPjrtApi") else {
            dlclose(handle)
            throw SwiftIRError.symbolNotFound(symbol: "GetPjrtApi", path: path)
        }

        // Call GetPjrtApi() to get the API struct
        // The function signature is: const PJRT_Api* GetPjrtApi()
        typealias GetPjrtApiFn = @convention(c) () -> UnsafeRawPointer?
        let getPjrtApi = unsafeBitCast(symbol, to: GetPjrtApiFn.self)

        guard let api = getPjrtApi() else {
            dlclose(handle)
            throw SwiftIRError.pluginLoadFailed(path: path, reason: "GetPjrtApi() returned null")
        }
        self.api = api

        // Extract API version
        // The PJRT_Api struct starts with:
        //   size_t struct_size;
        //   void* extension_start;
        //   size_t pjrt_api_version.major;
        //   size_t pjrt_api_version.minor;
        let versionPtr = api.advanced(by: MemoryLayout<Int>.size * 2) // Skip struct_size and extension_start
        let major = versionPtr.load(as: Int.self)
        let minor = versionPtr.advanced(by: MemoryLayout<Int>.size).load(as: Int.self)
        self.apiVersion = (major: major, minor: minor)
    }

    deinit {
        dlclose(handle)
    }

    // MARK: - Convenience Loading

    /// Load the PJRT plugin for the given accelerator type
    ///
    /// This method uses `RuntimeDetector.findPluginPath` to locate the plugin.
    ///
    /// - Parameter accelerator: The accelerator type to load
    /// - Returns: A loaded PJRTPlugin
    /// - Throws: `SwiftIRError.pluginNotFound` if no plugin is found
    public static func load(for accelerator: AcceleratorType) throws -> PJRTPlugin {
        guard let path = RuntimeDetector.findPluginPath(for: accelerator) else {
            let searchPaths = RuntimeDetector.getSearchPaths(for: accelerator)
            throw SwiftIRError.pluginNotFound(accelerator: accelerator, searchedPaths: searchPaths)
        }

        return try PJRTPlugin(path: path, acceleratorType: accelerator)
    }

    /// Load the plugin for the auto-detected best accelerator
    ///
    /// - Returns: A loaded PJRTPlugin for the best available accelerator
    /// - Throws: `SwiftIRError` if loading fails
    public static func loadBest() throws -> PJRTPlugin {
        let accelerator = RuntimeDetector.detect()
        return try load(for: accelerator)
    }

    // MARK: - Info

    /// Description of the loaded plugin
    public var description: String {
        return "PJRTPlugin(\(acceleratorType), v\(apiVersion.major).\(apiVersion.minor), \(path))"
    }
}

// MARK: - CustomStringConvertible

extension PJRTPlugin: CustomStringConvertible {}
