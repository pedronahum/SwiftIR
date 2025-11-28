// PJRTClientUnified.swift - Unified PJRT client creation API
// Copyright 2024 SwiftIR Project
//
// Provides auto-detection and unified client creation API similar to JAX.

import Foundation

// Re-export commonly used types
public typealias PJRTBackend = AcceleratorType

/// Unified PJRT client factory
///
/// Provides a JAX-like API for creating PJRT clients with automatic
/// accelerator detection.
///
/// ## Usage
/// ```swift
/// // Auto-detect best accelerator
/// let client = try PJRTClientFactory.create()
///
/// // Explicitly use TPU
/// let tpuClient = try PJRTClientFactory.create(.tpu)
///
/// // Get runtime info
/// PJRTClientFactory.printRuntimeInfo()
/// ```
public struct PJRTClientFactory {

    // MARK: - Client Creation

    /// Create a PJRT client with automatic accelerator detection
    ///
    /// Priority: Environment override > TPU > GPU > CPU
    ///
    /// - Parameter accelerator: Optional explicit accelerator type.
    ///                         If nil, automatically detects the best available.
    /// - Returns: Backend type string suitable for PJRTClient initialization
    /// - Throws: `SwiftIRError` if the accelerator is not available or loading fails
    public static func create(_ accelerator: AcceleratorType? = nil) throws -> String {
        let targetAccelerator = accelerator ?? RuntimeDetector.detect()

        print("SwiftIR: Using \(targetAccelerator) backend")

        // Validate availability
        switch targetAccelerator {
        case .tpu:
            guard RuntimeDetector.isTPUAvailable() else {
                throw SwiftIRError.acceleratorNotAvailable(.tpu)
            }
        case .gpu:
            guard RuntimeDetector.isGPUAvailable() else {
                throw SwiftIRError.acceleratorNotAvailable(.gpu)
            }
        case .cpu:
            // CPU is always available
            break
        }

        // Verify plugin exists
        guard let pluginPath = RuntimeDetector.findPluginPath(for: targetAccelerator) else {
            throw SwiftIRError.pluginNotFound(
                accelerator: targetAccelerator,
                searchedPaths: RuntimeDetector.getSearchPaths(for: targetAccelerator)
            )
        }

        print("SwiftIR: Plugin found at \(pluginPath)")

        // Return the backend string that PJRTClient expects
        return targetAccelerator.rawValue.lowercased()
    }

    /// Create a CPU client
    ///
    /// - Returns: Backend string "cpu"
    /// - Throws: `SwiftIRError` if CPU plugin is not found
    public static func createCPU() throws -> String {
        return try create(.cpu)
    }

    /// Create a GPU client
    ///
    /// - Returns: Backend string "gpu"
    /// - Throws: `SwiftIRError` if GPU is not available or plugin not found
    public static func createGPU() throws -> String {
        return try create(.gpu)
    }

    /// Create a TPU client
    ///
    /// - Returns: Backend string "tpu"
    /// - Throws: `SwiftIRError` if TPU is not available or plugin not found
    public static func createTPU() throws -> String {
        return try create(.tpu)
    }

    // MARK: - Runtime Info

    /// Get the currently detected accelerator
    public static var detectedAccelerator: AcceleratorType {
        return RuntimeDetector.detect()
    }

    /// Get detailed runtime information
    public static var runtimeInfo: RuntimeInfo {
        return RuntimeDetector.getRuntimeInfo()
    }

    /// Print runtime information to stdout
    public static func printRuntimeInfo() {
        RuntimeDetector.printInfo()
    }

    /// Check if a specific accelerator is available
    public static func isAvailable(_ accelerator: AcceleratorType) -> Bool {
        switch accelerator {
        case .cpu:
            return RuntimeDetector.findPluginPath(for: .cpu) != nil
        case .gpu:
            return RuntimeDetector.isGPUAvailable() && RuntimeDetector.findPluginPath(for: .gpu) != nil
        case .tpu:
            return RuntimeDetector.isTPUAvailable() && RuntimeDetector.findPluginPath(for: .tpu) != nil
        }
    }

    /// List all available accelerators
    public static var availableAccelerators: [AcceleratorType] {
        return AcceleratorType.allCases.filter { isAvailable($0) }
    }
}

// MARK: - Environment Configuration

/// Configure SwiftIR runtime via environment
public struct SwiftIREnvironment {
    /// Set the preferred accelerator
    ///
    /// Equivalent to setting `PJRT_DEVICE` environment variable
    public static func setPreferredAccelerator(_ accelerator: AcceleratorType) {
        setenv("PJRT_DEVICE", accelerator.rawValue, 1)
    }

    /// Set the SwiftIR home directory
    ///
    /// Used for finding bundled plugins
    public static func setSwiftIRHome(_ path: String) {
        setenv("SWIFTIR_HOME", path, 1)
    }

    /// Set a custom CPU plugin path
    public static func setCPUPluginPath(_ path: String) {
        setenv("PJRT_CPU_PLUGIN_PATH", path, 1)
    }

    /// Set a custom GPU plugin path
    public static func setGPUPluginPath(_ path: String) {
        setenv("PJRT_GPU_PLUGIN_PATH", path, 1)
    }

    /// Set a custom TPU plugin path
    public static func setTPUPluginPath(_ path: String) {
        setenv("TPU_LIBRARY_PATH", path, 1)
    }
}
