//===-- MetalRuntime.swift - Metal Runtime Wrapper --------*- Swift -*-===//
//
// SwiftIR - Phase 11C: SPIR-V Integration
// Metal compute runtime for executing shaders on Apple platforms
//
//===------------------------------------------------------------------===//

import Foundation

#if os(macOS) || os(iOS) || os(tvOS)
import Metal

/// Metal device wrapper
///
/// Provides a unified interface for Metal compute operations.
public class MetalDevice {
    /// Underlying Metal device
    public let device: MTLDevice

    /// Command queue for compute operations
    public let commandQueue: MTLCommandQueue

    /// Device ID
    public let id: Int

    /// Device properties
    public struct Properties {
        public let name: String
        public let isLowPower: Bool
        public let isRemovable: Bool
        public let hasUnifiedMemory: Bool
        public let maxThreadsPerThreadgroup: MTLSize
        public let maxBufferLength: Int
        public let recommendedMaxWorkingSetSize: UInt64

        init(device: MTLDevice) {
            self.name = device.name
            self.isLowPower = device.isLowPower
            self.isRemovable = device.isRemovable
            self.hasUnifiedMemory = device.hasUnifiedMemory
            self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup
            self.maxBufferLength = device.maxBufferLength
            self.recommendedMaxWorkingSetSize = device.recommendedMaxWorkingSetSize
        }
    }

    public let properties: Properties

    /// Initialize with a Metal device
    ///
    /// - Parameter device: MTLDevice to wrap
    /// - Throws: MetalError if command queue creation fails
    init(id: Int, device: MTLDevice) throws {
        self.id = id
        self.device = device
        self.properties = Properties(device: device)

        guard let queue = device.makeCommandQueue() else {
            throw MetalError.deviceCreationFailed("Failed to create command queue")
        }
        self.commandQueue = queue
    }

    /// Human-readable description
    public var description: String {
        var desc = properties.name
        if properties.hasUnifiedMemory {
            desc += " (Unified Memory)"
        }
        if properties.isLowPower {
            desc += " (Low Power)"
        }
        return desc
    }
}

/// Metal buffer wrapper
///
/// Represents a buffer in Metal memory.
public class MetalBuffer {
    /// Parent device
    public let device: MetalDevice

    /// Underlying Metal buffer
    public let buffer: MTLBuffer

    /// Buffer size in bytes
    public var size: Int {
        return buffer.length
    }

    init(device: MetalDevice, buffer: MTLBuffer) {
        self.device = device
        self.buffer = buffer
    }

    /// Upload data from host to device
    ///
    /// - Parameter data: Host memory pointer
    /// - Throws: MetalError if transfer fails
    public func upload(data: UnsafeRawPointer, length: Int) throws {
        guard length <= size else {
            throw MetalError.transferFailed("Data size \(length) exceeds buffer size \(size)")
        }

        // Copy to buffer's contents
        let contents = buffer.contents()
        memcpy(contents, data, length)

        // On unified memory systems (Apple Silicon), this is a no-op
        // On discrete GPUs, this synchronizes the data
        #if os(macOS)
        if !device.properties.hasUnifiedMemory {
            buffer.didModifyRange(0..<length)
        }
        #endif
    }

    /// Download data from device to host
    ///
    /// - Parameters:
    ///   - data: Host memory pointer (destination)
    ///   - length: Number of bytes to copy
    /// - Throws: MetalError if transfer fails
    public func download(data: UnsafeMutableRawPointer, length: Int) throws {
        guard length <= size else {
            throw MetalError.transferFailed("Data size \(length) exceeds buffer size \(size)")
        }

        let contents = buffer.contents()
        memcpy(data, contents, length)
    }
}

/// Metal executable (compute pipeline)
///
/// Represents a compiled Metal shader function.
public class MetalExecutable {
    /// Parent device
    public let device: MetalDevice

    /// Metal library containing the shader
    public let library: MTLLibrary

    /// Compute function
    public let function: MTLFunction

    /// Compute pipeline state
    public let pipeline: MTLComputePipelineState

    /// Execution statistics
    public private(set) var executionCount: Int = 0

    /// Initialize from Metal Shading Language source
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - source: MSL source code
    ///   - functionName: Name of compute kernel function
    /// - Throws: MetalError if compilation fails
    init(device: MetalDevice, source: String, functionName: String = "main") throws {
        self.device = device

        // Compile MSL source
        do {
            self.library = try device.device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalError.shaderCompilationFailed("MSL compilation failed: \(error)")
        }

        // Get compute function
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalError.shaderCompilationFailed("Function '\(functionName)' not found in library")
        }
        self.function = function

        // Create compute pipeline
        do {
            self.pipeline = try device.device.makeComputePipelineState(function: function)
        } catch {
            throw MetalError.pipelineCreationFailed("Failed to create compute pipeline: \(error)")
        }
    }

    /// Initialize from SPIR-V binary (via spirv-cross)
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - spirvBinary: SPIR-V binary
    /// - Throws: MetalError if conversion/compilation fails
    static func fromSPIRV(device: MetalDevice, spirvBinary: [UInt32]) throws -> MetalExecutable {
        // TODO: Use spirv-cross to convert SPIR-V → MSL
        // 1. Call spirv_cross_compile_to_msl()
        // 2. Get MSL source string
        // 3. Compile MSL source

        print("⚠️  SPIR-V → MSL conversion stub")
        print("   Note: Full implementation requires spirv-cross integration")

        // For now, create a simple pass-through kernel
        let mslSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void main(device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
            output[id] = input[id];
        }
        """

        return try MetalExecutable(device: device, source: mslSource, functionName: "main")
    }

    /// Execute the compute shader
    ///
    /// - Parameters:
    ///   - buffers: Input/output buffers
    ///   - threadgroups: Number of threadgroups to dispatch
    ///   - threadsPerThreadgroup: Threads per threadgroup
    /// - Throws: MetalError if execution fails
    public func execute(
        buffers: [MetalBuffer],
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) throws {
        // Create command buffer
        guard let commandBuffer = device.commandQueue.makeCommandBuffer() else {
            throw MetalError.executionFailed("Failed to create command buffer")
        }

        // Create compute encoder
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.executionFailed("Failed to create compute encoder")
        }

        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer.buffer, offset: 0, index: index)
        }

        // Dispatch
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check for errors
        if commandBuffer.status == .error {
            if let error = commandBuffer.error {
                throw MetalError.executionFailed("Command buffer error: \(error)")
            } else {
                throw MetalError.executionFailed("Command buffer failed with unknown error")
            }
        }

        executionCount += 1
    }
}

/// Metal runtime manager
///
/// Handles Metal device enumeration.
public class MetalRuntime {
    /// Enumerate all available Metal devices
    ///
    /// - Returns: Array of available Metal devices
    /// - Throws: MetalError if enumeration fails
    public static func enumerateDevices() throws -> [MetalDevice] {
        print("🔍 Enumerating Metal devices...")

        var devices: [MetalDevice] = []

        #if os(macOS)
        // On macOS, enumerate all devices
        let mtlDevices = MTLCopyAllDevices()
        for (index, mtlDevice) in mtlDevices.enumerated() {
            do {
                let device = try MetalDevice(id: index, device: mtlDevice)
                devices.append(device)
                print("   [\(index)] \(device.description)")
            } catch {
                print("   ⚠️  Failed to initialize device \(index): \(error)")
            }
        }
        #else
        // On iOS/tvOS, use default device
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevicesAvailable
        }
        let device = try MetalDevice(id: 0, device: mtlDevice)
        devices.append(device)
        print("   [0] \(device.description)")
        #endif

        if devices.isEmpty {
            throw MetalError.noDevicesAvailable
        }

        print("   Found \(devices.count) Metal device(s)")
        return devices
    }
}

/// Metal runtime errors
public enum MetalError: Error {
    case noDevicesAvailable
    case deviceCreationFailed(String)
    case bufferCreationFailed(String)
    case shaderCompilationFailed(String)
    case pipelineCreationFailed(String)
    case executionFailed(String)
    case transferFailed(String)

    public var localizedDescription: String {
        switch self {
        case .noDevicesAvailable:
            return "No Metal devices available"
        case .deviceCreationFailed(let msg):
            return "Device creation failed: \(msg)"
        case .bufferCreationFailed(let msg):
            return "Buffer creation failed: \(msg)"
        case .shaderCompilationFailed(let msg):
            return "Shader compilation failed: \(msg)"
        case .pipelineCreationFailed(let msg):
            return "Pipeline creation failed: \(msg)"
        case .executionFailed(let msg):
            return "Execution failed: \(msg)"
        case .transferFailed(let msg):
            return "Transfer failed: \(msg)"
        }
    }
}

// MARK: - Helper Extensions

extension MetalDevice {
    /// Create a buffer on this device
    ///
    /// - Parameters:
    ///   - size: Buffer size in bytes
    ///   - options: Metal resource options
    /// - Returns: Metal buffer
    /// - Throws: MetalError if creation fails
    public func createBuffer(
        size: Int,
        options: MTLResourceOptions = .storageModeShared
    ) throws -> MetalBuffer {
        guard let buffer = device.makeBuffer(length: size, options: options) else {
            throw MetalError.bufferCreationFailed("Failed to allocate buffer of size \(size)")
        }
        return MetalBuffer(device: self, buffer: buffer)
    }

    /// Create an executable from MSL source
    ///
    /// - Parameters:
    ///   - source: Metal Shading Language source
    ///   - functionName: Name of compute kernel function
    /// - Returns: Metal executable
    /// - Throws: MetalError if compilation fails
    public func createExecutable(source: String, functionName: String = "main") throws -> MetalExecutable {
        return try MetalExecutable(device: self, source: source, functionName: functionName)
    }

    /// Create an executable from SPIR-V binary
    ///
    /// - Parameter spirvBinary: SPIR-V binary
    /// - Returns: Metal executable
    /// - Throws: MetalError if conversion/compilation fails
    public func createExecutableFromSPIRV(spirvBinary: [UInt32]) throws -> MetalExecutable {
        return try MetalExecutable.fromSPIRV(device: self, spirvBinary: spirvBinary)
    }
}

#else
// Non-Apple platforms: provide stub types

public class MetalDevice {
    public var description: String { "Metal not available" }
}

public class MetalBuffer {}

public class MetalExecutable {}

public class MetalRuntime {
    public static func enumerateDevices() throws -> [MetalDevice] {
        throw MetalError.noDevicesAvailable
    }
}

public enum MetalError: Error {
    case noDevicesAvailable
    case deviceCreationFailed(String)
    case bufferCreationFailed(String)
    case shaderCompilationFailed(String)
    case pipelineCreationFailed(String)
    case executionFailed(String)
    case transferFailed(String)

    public var localizedDescription: String {
        return "Metal not available on this platform"
    }
}

#endif
