//===-- Backend.swift - Execution Backend Protocol -------*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime
// Protocol for compilation and execution backends
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import Foundation

/// Execution backend protocol
///
/// Defines the interface for compiling and executing MLIR modules.
/// Implementations can target different hardware (CPU, GPU, TPU) and
/// use different compilation strategies (MLIR JIT, XLA, custom).
public protocol Backend {
    /// The name of this backend (e.g., "MLIR-CPU", "XLA-GPU")
    var name: String { get }

    /// Compile an MLIR module into an executable
    func compile(module: MLIRModule) throws -> Executable

    /// Check if this backend is available on the current system
    var isAvailable: Bool { get }
}

/// Compiled executable that can be invoked with tensor inputs
public protocol Executable {
    /// Execute the compiled function with given inputs
    ///
    /// - Parameter inputs: Array of input tensors matching function signature
    /// - Returns: Array of output tensors
    /// - Throws: ExecutionError if execution fails
    func execute(inputs: [Tensor]) throws -> [Tensor]

    /// The entry point function name
    var entryPoint: String { get }
}

/// Runtime errors during execution
public enum ExecutionError: Error, CustomStringConvertible {
    case compilationFailed(String)
    case invalidInputCount(expected: Int, got: Int)
    case invalidInputShape(index: Int, expected: [Int64], got: [Int64])
    case executionFailed(String)
    case backendUnavailable(String)

    public var description: String {
        switch self {
        case .compilationFailed(let msg):
            return "Compilation failed: \(msg)"
        case .invalidInputCount(let expected, let got):
            return "Invalid input count: expected \(expected), got \(got)"
        case .invalidInputShape(let index, let expected, let got):
            return "Invalid shape for input \(index): expected \(expected), got \(got)"
        case .executionFailed(let msg):
            return "Execution failed: \(msg)"
        case .backendUnavailable(let msg):
            return "Backend unavailable: \(msg)"
        }
    }
}

/// Device type for execution
public enum DeviceType {
    case cpu
    case gpu(index: Int)
    case tpu(index: Int)

    public var description: String {
        switch self {
        case .cpu:
            return "CPU"
        case .gpu(let index):
            return "GPU:\(index)"
        case .tpu(let index):
            return "TPU:\(index)"
        }
    }
}

/// Backend factory for creating appropriate backends
public enum BackendFactory {
    /// Get the default backend for the current system
    public static func defaultBackend() -> Backend {
        // For now, always return MLIR CPU backend
        // In the future, detect GPU/TPU availability
        return MLIRCPUBackend()
    }

    /// Get a backend for a specific device
    public static func backend(for device: DeviceType) -> Backend? {
        switch device {
        case .cpu:
            return MLIRCPUBackend()
        case .gpu, .tpu:
            // XLA backend would go here in the future
            return nil
        }
    }

    /// List all available backends
    public static func availableBackends() -> [Backend] {
        var backends: [Backend] = []

        let cpuBackend = MLIRCPUBackend()
        if cpuBackend.isAvailable {
            backends.append(cpuBackend)
        }

        // Future: Check for XLA GPU/TPU availability

        return backends
    }
}
