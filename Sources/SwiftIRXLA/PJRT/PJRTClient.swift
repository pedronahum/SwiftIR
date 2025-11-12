//===-- PJRTClient.swift - PJRT Client Wrapper ------------*- Swift -*-===//
//
// SwiftIR - Phase 11: PJRT Integration
// Swift wrapper for PJRT C API client
//
//===------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import PJRTCAPI  // C module defined in PJRTCWrappers/include/module.modulemap

/// PJRT Client for managing devices and executing computations
///
/// The PJRTClient is the main entry point for PJRT operations.
/// It manages device enumeration, buffer creation, and program compilation.
public class PJRTClient {
    /// Backend type for the client
    public enum Backend: String {
        case cpu = "cpu"
        case gpu = "gpu"
        case tpu = "tpu"

        /// Get the default plugin path for this backend
        func pluginPath() -> String {
            switch self {
            case .cpu:
                // Try several common locations
                let paths = [
                    "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_cpu_plugin.dylib",
                    "./lib/pjrt_c_api_cpu_plugin.dylib",
                    "../lib/pjrt_c_api_cpu_plugin.dylib",
                ]
                for path in paths {
                    if FileManager.default.fileExists(atPath: path) {
                        return path
                    }
                }
                return paths[0] // Return first as default
            case .gpu:
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_gpu_plugin.dylib"
            case .tpu:
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_tpu_plugin.dylib"
            }
        }
    }

    /// Low-level C API wrapper
    internal var capi: PJRTCAPI?

    /// Opaque handle to the underlying PJRT_Client
    private var handle: OpaquePointer?

    /// Backend this client is connected to
    public let backend: Backend

    /// Platform name (e.g., "cpu", "cuda", "rocm")
    public private(set) var platformName: String

    /// Available devices
    public private(set) var devices: [PJRTDevice] = []

    /// Addressable devices (devices this client can use)
    public private(set) var addressableDevices: [PJRTDevice] = []

    /// Initialize a PJRT client for the specified backend
    ///
    /// - Parameter backend: The backend to use (CPU, GPU, or TPU)
    /// - Throws: PJRTError if client creation fails
    public init(backend: Backend) throws {
        self.backend = backend
        self.platformName = backend.rawValue

        // Load the PJRT plugin
        let pluginPath = backend.pluginPath()
        print("Loading PJRT \(backend.rawValue.uppercased()) plugin...")
        print("  Path: \(pluginPath)")

        do {
            // Initialize the C API wrapper
            self.capi = try PJRTCAPI(backend: backend.rawValue, pluginPath: pluginPath)

            // Create the client
            guard let capi = self.capi else {
                throw PJRTError.clientCreationFailed("Failed to initialize C API")
            }

            self.handle = try capi.createClient()

            // Get platform name
            self.platformName = try capi.getPlatformName(self.handle!)

            // Enumerate devices
            let deviceHandles = try capi.getAddressableDevices(self.handle!)

            for deviceHandle in deviceHandles {
                let deviceId = try capi.getDeviceId(deviceHandle)
                let deviceKind = try capi.getDeviceKind(deviceHandle)

                let device = PJRTDevice(
                    id: Int(deviceId),
                    kind: deviceKind,
                    client: self,
                    handle: deviceHandle
                )

                self.devices.append(device)
                self.addressableDevices.append(device)
            }

            print("✓ PJRT Client initialized successfully")
            print("  Platform: \(platformName)")
            print("  Devices: \(devices.count)")
            for device in devices {
                print("    - \(device.description)")
            }

        } catch let error as PJRTError {
            print("✗ PJRT Client initialization failed: \(error.localizedDescription)")
            throw error
        } catch {
            print("✗ PJRT Client initialization failed: \(error)")
            throw PJRTError.clientCreationFailed("\(error)")
        }
    }

    deinit {
        // Destroy the client
        if let handle = handle, let capi = capi {
            capi.destroyClient(handle)
        }
    }

    /// Get device by ID
    ///
    /// - Parameter id: Device ID
    /// - Returns: Device if found
    public func device(id: Int) -> PJRTDevice? {
        return devices.first { $0.id == id }
    }

    /// Get the default device (usually device 0)
    public var defaultDevice: PJRTDevice? {
        return addressableDevices.first
    }

    /// Create a buffer from host data
    ///
    /// - Parameters:
    ///   - data: Host data pointer
    ///   - shape: Tensor shape
    ///   - elementType: Element type
    ///   - device: Target device (nil = default device)
    /// - Returns: PJRT buffer
    /// - Throws: PJRTError if buffer creation fails
    public func createBuffer(
        data: UnsafeRawPointer,
        shape: [Int],
        elementType: PJRTElementType,
        device: PJRTDevice? = nil
    ) throws -> PJRTBuffer {
        let targetDevice = device ?? defaultDevice
        guard let targetDevice = targetDevice else {
            throw PJRTError.noDeviceAvailable
        }

        guard let capi = self.capi, let clientHandle = self.handle else {
            throw PJRTError.bufferCreationFailed("Client not initialized")
        }

        guard let deviceHandle = targetDevice.handle else {
            throw PJRTError.bufferCreationFailed("Device handle not available")
        }

        // Convert shape to Int64 array for C API
        let dims = shape.map { Int64($0) }

        // Call PJRT C API wrapper
        let bufferHandle = try capi.createBuffer(
            client: clientHandle,
            data: data,
            type: elementType.toCType,
            dims: dims,
            device: deviceHandle
        )

        // Create PJRTBuffer with the handle
        let buffer = PJRTBuffer(
            shape: shape,
            elementType: elementType,
            device: targetDevice
        )
        buffer.handle = bufferHandle

        return buffer
    }

    /// Compile a program
    ///
    /// - Parameters:
    ///   - mlirModule: MLIR module containing StableHLO operations
    ///   - devices: Devices to compile for (nil = all addressable devices)
    /// - Returns: Loaded executable
    /// - Throws: PJRTError if compilation fails
    public func compile(
        mlirModule: String,
        devices: [PJRTDevice]? = nil
    ) throws -> PJRTLoadedExecutable {
        let targetDevices = devices ?? addressableDevices

        guard let capi = self.capi, let clientHandle = self.handle else {
            throw PJRTError.compilationFailed("Client not initialized")
        }

        // Call PJRT C API wrapper to compile the MLIR module
        let executableHandle = try capi.compile(
            client: clientHandle,
            mlirModule: mlirModule
        )

        // Create PJRTLoadedExecutable with the handle
        let executable = PJRTLoadedExecutable(
            client: self,
            devices: targetDevices
        )
        executable.handle = executableHandle

        return executable
    }
}

/// PJRT Device representation
public class PJRTDevice {
    /// Device ID
    public let id: Int

    /// Device kind (e.g., "CPU", "GPU", "TPU")
    public let kind: String

    /// Client that owns this device
    public weak var client: PJRTClient?

    /// Opaque handle to PJRT_Device
    internal var handle: OpaquePointer?

    init(id: Int, kind: String, client: PJRTClient, handle: OpaquePointer? = nil) {
        self.id = id
        self.kind = kind
        self.client = client
        self.handle = handle
    }

    /// Human-readable description
    public var description: String {
        return "\(kind):\(id)"
    }
}

/// PJRT element types
public enum PJRTElementType {
    case pred      // Boolean
    case s8, s16, s32, s64    // Signed integers
    case u8, u16, u32, u64    // Unsigned integers
    case f16, f32, f64        // Floating point
    case bf16                 // Brain float 16
    case c64, c128            // Complex

    /// Size in bytes
    public var sizeInBytes: Int {
        switch self {
        case .pred, .s8, .u8: return 1
        case .s16, .u16, .f16, .bf16: return 2
        case .s32, .u32, .f32: return 4
        case .s64, .u64, .f64, .c64: return 8
        case .c128: return 16
        }
    }

    /// Convert to SW_PJRT_Buffer_Type for C API
    internal var toCType: SW_PJRT_Buffer_Type {
        switch self {
        case .pred: return SW_PJRT_Buffer_Type_PRED
        case .s8: return SW_PJRT_Buffer_Type_S8
        case .s16: return SW_PJRT_Buffer_Type_S16
        case .s32: return SW_PJRT_Buffer_Type_S32
        case .s64: return SW_PJRT_Buffer_Type_S64
        case .u8: return SW_PJRT_Buffer_Type_U8
        case .u16: return SW_PJRT_Buffer_Type_U16
        case .u32: return SW_PJRT_Buffer_Type_U32
        case .u64: return SW_PJRT_Buffer_Type_U64
        case .f16: return SW_PJRT_Buffer_Type_F16
        case .f32: return SW_PJRT_Buffer_Type_F32
        case .f64: return SW_PJRT_Buffer_Type_F64
        case .bf16: return SW_PJRT_Buffer_Type_BF16
        case .c64: return SW_PJRT_Buffer_Type_C64
        case .c128: return SW_PJRT_Buffer_Type_C128
        }
    }
}

/// PJRT buffer (device memory)
public class PJRTBuffer {
    /// Shape of the buffer
    public let shape: [Int]

    /// Element type
    public let elementType: PJRTElementType

    /// Device this buffer resides on
    public let device: PJRTDevice

    /// Opaque handle to PJRT_Buffer
    internal var handle: OpaquePointer?

    /// Track if we incremented the reference count
    /// Only buffers from execute() have incremented ref count
    /// Buffers from createBuffer() do NOT need DecRef
    private var didIncRef: Bool = false

    init(shape: [Int], elementType: PJRTElementType, device: PJRTDevice, didIncRef: Bool = false) {
        self.shape = shape
        self.elementType = elementType
        self.device = device
        self.didIncRef = didIncRef
    }

    deinit {
        // Destroy the buffer if we have a handle
        if let handle = handle, let capi = device.client?.capi {
            // CRITICAL: Only decrease reference count if we incremented it
            // Buffers from execute() need DecRef, buffers from createBuffer() do NOT
            if didIncRef {
                try? capi.bufferDecRef(handle)
            }
            capi.destroyBuffer(handle)
        }
    }

    /// Explicitly destroy the buffer
    /// Call this to ensure cleanup happens at a specific time
    public func destroy() {
        if let handle = handle, let capi = device.client?.capi {
            // CRITICAL: Only decrease reference count if we incremented it
            if didIncRef {
                try? capi.bufferDecRef(handle)
            }
            capi.destroyBuffer(handle)
            self.handle = nil  // Mark as destroyed
        }
    }

    /// Total number of elements
    public var elementCount: Int {
        return shape.reduce(1, *)
    }

    /// Size in bytes
    public var sizeInBytes: Int {
        return elementCount * elementType.sizeInBytes
    }

    /// Copy buffer contents to host
    ///
    /// - Parameter destination: Host memory pointer
    /// - Throws: PJRTError if transfer fails
    public func toHost(destination: UnsafeMutableRawPointer) throws {
        guard let handle = handle else {
            throw PJRTError.bufferTransferFailed("Buffer handle not available")
        }

        guard let capi = device.client?.capi else {
            throw PJRTError.bufferTransferFailed("Client not available")
        }

        // Call PJRT C API wrapper to transfer buffer to host
        try capi.bufferToHost(
            buffer: handle,
            destination: destination,
            size: sizeInBytes
        )
    }
}

/// PJRT loaded executable
public class PJRTLoadedExecutable {
    /// Client that compiled this executable
    public weak var client: PJRTClient?

    /// Devices this executable can run on
    public let devices: [PJRTDevice]

    /// Opaque handle to PJRT_LoadedExecutable
    internal var handle: OpaquePointer?

    /// Execution statistics
    public private(set) var executionCount: Int = 0

    init(client: PJRTClient, devices: [PJRTDevice]) {
        self.client = client
        self.devices = devices
    }

    deinit {
        // Destroy the executable if we have a handle
        if let handle = handle, let capi = client?.capi {
            capi.destroyExecutable(handle)
        }
    }

    /// Explicitly destroy the executable
    /// Call this to ensure cleanup happens at a specific time
    public func destroy() {
        if let handle = handle, let capi = client?.capi {
            capi.destroyExecutable(handle)
            self.handle = nil  // Mark as destroyed
        }
    }

    /// Execute the program
    ///
    /// - Parameters:
    ///   - arguments: Input buffers
    ///   - device: Device to execute on (nil = first device)
    /// - Returns: Output buffers
    /// - Throws: PJRTError if execution fails
    public func execute(
        arguments: [PJRTBuffer],
        device: PJRTDevice? = nil
    ) throws -> [PJRTBuffer] {
        let targetDevice = device ?? devices.first
        guard let targetDevice = targetDevice else {
            throw PJRTError.noDeviceAvailable
        }

        guard let handle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        guard let capi = client?.capi else {
            throw PJRTError.executionFailed("Client not available")
        }

        // Extract buffer handles from arguments
        let inputHandles = try arguments.map { arg -> OpaquePointer in
            guard let handle = arg.handle else {
                throw PJRTError.executionFailed("Input buffer handle not available")
            }
            return handle
        }

        // Call PJRT C API wrapper to execute
        let outputHandles = try capi.execute(
            executable: handle,
            inputs: inputHandles
        )

        executionCount += 1

        // Wrap output handles in PJRTBuffer objects
        // Query actual buffer metadata from PJRT
        let outputBuffers = try outputHandles.map { outputHandle -> PJRTBuffer in
            // NOTE: Not incrementing reference count - PJRT gives us ownership
            // If this causes issues, uncomment: try capi.bufferIncRef(outputHandle)
            // and set didIncRef=true below

            // Query dimensions
            let dims = try capi.getBufferDimensions(buffer: outputHandle)

            // Query size in bytes
            let sizeInBytes = try capi.getBufferOnDeviceSizeInBytes(buffer: outputHandle)

            // Create buffer with actual metadata (convert Int64 dims to Int)
            // Set didIncRef=false since we didn't call bufferIncRef
            let buffer = PJRTBuffer(
                shape: dims.map { Int($0) },
                elementType: .f32,  // Still assuming f32 for now - could add element type query
                device: targetDevice,
                didIncRef: false  // Did NOT increment ref count - PJRT owns the buffer
            )
            buffer.handle = outputHandle
            return buffer
        }

        return outputBuffers
    }
}

/// PJRT errors
public enum PJRTError: Error {
    case clientCreationFailed(String)
    case deviceNotFound(Int)
    case noDeviceAvailable
    case bufferCreationFailed(String)
    case bufferTransferFailed(String)
    case compilationFailed(String)
    case executionFailed(String)
    case transferFailed(String)

    public var localizedDescription: String {
        switch self {
        case .clientCreationFailed(let msg):
            return "PJRT client creation failed: \(msg)"
        case .deviceNotFound(let id):
            return "Device not found: \(id)"
        case .noDeviceAvailable:
            return "No device available"
        case .bufferCreationFailed(let msg):
            return "Buffer creation failed: \(msg)"
        case .bufferTransferFailed(let msg):
            return "Buffer transfer failed: \(msg)"
        case .compilationFailed(let msg):
            return "Compilation failed: \(msg)"
        case .executionFailed(let msg):
            return "Execution failed: \(msg)"
        case .transferFailed(let msg):
            return "Transfer failed: \(msg)"
        }
    }
}
