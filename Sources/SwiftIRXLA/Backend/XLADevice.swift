//===-- XLADevice.swift - XLA Device Abstraction -----------*- Swift -*-===//
//
// SwiftIR - Phase 11: XLA Backend Integration
// Device abstraction for XLA execution (GPU/TPU/CPU)
//
//===------------------------------------------------------------------===//

import Foundation

/// Represents an XLA-compatible execution device
public struct XLADevice {
    /// Device type
    public enum DeviceType: String {
        case cpu = "CPU"
        case gpu = "GPU"
        case tpu = "TPU"
    }

    /// Device identifier
    public let id: Int

    /// Device type
    public let type: DeviceType

    /// Device name (e.g., "NVIDIA Tesla V100")
    public let name: String

    /// Total memory in bytes
    public let memoryBytes: Int64

    /// Whether this is the default device
    public let isDefault: Bool

    public init(id: Int, type: DeviceType, name: String, memoryBytes: Int64, isDefault: Bool = false) {
        self.id = id
        self.type = type
        self.name = name
        self.memoryBytes = memoryBytes
        self.isDefault = isDefault
    }

    /// Human-readable description
    public var description: String {
        let memoryGB = Double(memoryBytes) / 1_000_000_000.0
        return "\(type.rawValue):\(id) - \(name) (\(String(format: "%.1f", memoryGB)) GB)"
    }
}

/// XLA device manager for discovering and selecting execution devices
public class XLADeviceManager: @unchecked Sendable {
    /// Singleton instance
    public static let shared = XLADeviceManager()

    private var cachedDevices: [XLADevice]?

    private init() {}

    /// Enumerate all available XLA devices
    ///
    /// This will discover GPUs, TPUs, and CPU devices available for XLA execution.
    /// In the current implementation, we start with CPU-based execution via LLVM.
    /// Future versions will integrate with PJRT for true XLA device enumeration.
    ///
    /// - Returns: Array of available devices
    public func enumerateDevices() -> [XLADevice] {
        if let cached = cachedDevices {
            return cached
        }

        var devices: [XLADevice] = []

        // CPU device is always available
        let cpuDevice = XLADevice(
            id: 0,
            type: .cpu,
            name: "CPU (LLVM Execution Engine)",
            memoryBytes: getSystemMemory(),
            isDefault: true
        )
        devices.append(cpuDevice)

        // TODO: Add GPU enumeration via Metal/CUDA
        // This will be implemented when we integrate with real XLA via PJRT
        #if os(macOS)
        // On macOS, we could potentially use Metal
        // devices.append(contentsOf: enumerateMetalDevices())
        #endif

        // TODO: Add TPU enumeration
        // This requires PJRT client library integration

        cachedDevices = devices
        return devices
    }

    /// Get the default device for execution
    public func getDefaultDevice() -> XLADevice {
        return enumerateDevices().first { $0.isDefault } ?? enumerateDevices()[0]
    }

    /// Select a device by type
    public func selectDevice(type: XLADevice.DeviceType) -> XLADevice? {
        return enumerateDevices().first { $0.type == type }
    }

    /// Select a device by ID
    public func selectDevice(id: Int) -> XLADevice? {
        return enumerateDevices().first { $0.id == id }
    }

    /// Get system memory in bytes
    private func getSystemMemory() -> Int64 {
        #if os(macOS) || os(Linux)
        var size: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &size, &len, nil, 0)
        return Int64(size)
        #else
        return 8_000_000_000 // Default to 8GB
        #endif
    }

    // Future: Metal device enumeration for macOS GPU
    #if os(macOS)
    private func enumerateMetalDevices() -> [XLADevice] {
        // TODO: Use Metal framework to enumerate GPUs
        // import Metal
        // let devices = MTLCopyAllDevices()
        return []
    }
    #endif
}

/// Device selection options for XLA compilation
public struct XLADeviceOptions {
    /// Target device
    public let device: XLADevice

    /// Whether to allow fallback to CPU
    public let allowCPUFallback: Bool

    /// Memory limit for execution (nil = no limit)
    public let memoryLimitBytes: Int64?

    public init(
        device: XLADevice,
        allowCPUFallback: Bool = true,
        memoryLimitBytes: Int64? = nil
    ) {
        self.device = device
        self.allowCPUFallback = allowCPUFallback
        self.memoryLimitBytes = memoryLimitBytes
    }

    /// Create options for default device
    public static func defaultOptions() -> XLADeviceOptions {
        return XLADeviceOptions(device: XLADeviceManager.shared.getDefaultDevice())
    }
}
