// AcceleratorType.swift - Hardware accelerator enumeration
// Copyright 2024 SwiftIR Project
//
// Defines the supported accelerator types for PJRT execution.

import Foundation

/// Supported hardware accelerator types for SwiftIR execution
///
/// SwiftIR can execute on CPU, GPU (NVIDIA CUDA), or TPU backends.
/// Use `RuntimeDetector.detect()` to automatically detect the best available
/// accelerator, or explicitly specify one when creating a `PJRTClient`.
///
/// ## Usage
/// ```swift
/// // Auto-detect best accelerator
/// let client = try PJRTClient.create()
///
/// // Explicitly use TPU
/// let tpuClient = try PJRTClient.create(.tpu)
/// ```
public enum AcceleratorType: String, CustomStringConvertible, Equatable, Hashable, Codable, CaseIterable, Sendable {
    /// CPU backend - always available, uses pjrt_c_api_cpu_plugin.so
    case cpu = "CPU"

    /// GPU backend - requires NVIDIA GPU and CUDA, uses xla_cuda_plugin.so
    case gpu = "GPU"

    /// TPU backend - requires Google TPU hardware, uses libtpu.so
    case tpu = "TPU"

    // MARK: - CustomStringConvertible

    public var description: String { rawValue }

    // MARK: - Plugin Information

    /// The expected filename for this accelerator's PJRT plugin
    public var pluginName: String {
        switch self {
        case .cpu: return "pjrt_c_api_cpu_plugin.so"
        case .gpu: return "xla_cuda_plugin.so"
        case .tpu: return "libtpu.so"
        }
    }

    /// Human-readable description of the accelerator
    public var displayName: String {
        switch self {
        case .cpu: return "CPU"
        case .gpu: return "NVIDIA GPU (CUDA)"
        case .tpu: return "Google TPU"
        }
    }

    /// Whether this accelerator typically requires special hardware
    public var requiresSpecialHardware: Bool {
        switch self {
        case .cpu: return false
        case .gpu, .tpu: return true
        }
    }
}
