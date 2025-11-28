// Errors.swift - SwiftIR runtime error types
// Copyright 2024 SwiftIR Project
//
// Defines error types for runtime detection and PJRT client creation.

import Foundation

/// Errors that can occur during SwiftIR runtime operations
public enum SwiftIRError: Error, LocalizedError, Equatable, Sendable {
    /// No PJRT plugin found for the requested accelerator
    case pluginNotFound(accelerator: AcceleratorType, searchedPaths: [String])

    /// Failed to load a PJRT plugin with dlopen
    case pluginLoadFailed(path: String, reason: String)

    /// Required symbol not found in the loaded plugin
    case symbolNotFound(symbol: String, path: String)

    /// Failed to create a PJRT client
    case clientCreationFailed(accelerator: AcceleratorType, reason: String)

    /// Requested accelerator is not available on this system
    case acceleratorNotAvailable(AcceleratorType)

    /// PJRT compilation failed
    case compilationFailed(reason: String)

    /// PJRT execution failed
    case executionFailed(reason: String)

    // MARK: - LocalizedError

    public var errorDescription: String? {
        switch self {
        case .pluginNotFound(let accelerator, let paths):
            return """
                No \(accelerator) PJRT plugin found.
                Searched paths:
                \(paths.map { "  - \($0)" }.joined(separator: "\n"))

                To fix:
                \(installationHint(for: accelerator))
                """

        case .pluginLoadFailed(let path, let reason):
            return "Failed to load PJRT plugin at '\(path)': \(reason)"

        case .symbolNotFound(let symbol, let path):
            return "Symbol '\(symbol)' not found in plugin at '\(path)'. The plugin may be incompatible."

        case .clientCreationFailed(let accelerator, let reason):
            return "Failed to create \(accelerator) PJRT client: \(reason)"

        case .acceleratorNotAvailable(let accelerator):
            return """
                \(accelerator) is not available on this system.
                \(availabilityHint(for: accelerator))
                """

        case .compilationFailed(let reason):
            return "PJRT compilation failed: \(reason)"

        case .executionFailed(let reason):
            return "PJRT execution failed: \(reason)"
        }
    }

    // MARK: - Helpful Hints

    private func installationHint(for accelerator: AcceleratorType) -> String {
        switch accelerator {
        case .cpu:
            return """
                CPU plugin should be bundled with SwiftIR.
                Set SWIFTIR_HOME or PJRT_CPU_PLUGIN_PATH environment variable.
                """
        case .gpu:
            return """
                Install JAX with CUDA support:
                  pip install jax[cuda12]
                Or set PJRT_GPU_PLUGIN_PATH to the xla_cuda_plugin.so location.
                """
        case .tpu:
            return """
                TPU plugin (libtpu.so) is pre-installed on:
                  - Google Colab TPU runtime
                  - Cloud TPU VMs
                Set TPU_LIBRARY_PATH if using a custom location.
                """
        }
    }

    private func availabilityHint(for accelerator: AcceleratorType) -> String {
        switch accelerator {
        case .cpu:
            return "CPU should always be available. Check your SwiftIR installation."
        case .gpu:
            return """
                GPU requires:
                  - NVIDIA GPU hardware
                  - CUDA drivers installed
                  - CUDA toolkit (optional)
                Run 'nvidia-smi' to check GPU status.
                """
        case .tpu:
            return """
                TPU requires:
                  - Google Colab TPU runtime, or
                  - Cloud TPU VM
                Check for /dev/accel* devices.
                """
        }
    }
}

// MARK: - Equatable Conformance

extension SwiftIRError {
    public static func == (lhs: SwiftIRError, rhs: SwiftIRError) -> Bool {
        switch (lhs, rhs) {
        case (.pluginNotFound(let a1, let p1), .pluginNotFound(let a2, let p2)):
            return a1 == a2 && p1 == p2
        case (.pluginLoadFailed(let p1, let r1), .pluginLoadFailed(let p2, let r2)):
            return p1 == p2 && r1 == r2
        case (.symbolNotFound(let s1, let p1), .symbolNotFound(let s2, let p2)):
            return s1 == s2 && p1 == p2
        case (.clientCreationFailed(let a1, let r1), .clientCreationFailed(let a2, let r2)):
            return a1 == a2 && r1 == r2
        case (.acceleratorNotAvailable(let a1), .acceleratorNotAvailable(let a2)):
            return a1 == a2
        case (.compilationFailed(let r1), .compilationFailed(let r2)):
            return r1 == r2
        case (.executionFailed(let r1), .executionFailed(let r2)):
            return r1 == r2
        default:
            return false
        }
    }
}
