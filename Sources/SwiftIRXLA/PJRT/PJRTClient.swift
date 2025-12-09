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
                // Try several common locations for both macOS and Linux
                #if os(macOS)
                let ext = "dylib"
                let systemPaths = ["/Users/pedro/programming/swift/SwiftIR/lib"]
                #else
                let ext = "so"
                let systemPaths = ["/opt/swiftir-deps/lib", "/usr/local/lib"]
                #endif

                let paths = systemPaths.map { "\($0)/pjrt_c_api_cpu_plugin.\(ext)" } + [
                    "./lib/pjrt_c_api_cpu_plugin.\(ext)",
                    "../lib/pjrt_c_api_cpu_plugin.\(ext)",
                ]
                for path in paths {
                    if FileManager.default.fileExists(atPath: path) {
                        return path
                    }
                }
                return paths[0] // Return first as default
            case .gpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_gpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_gpu_plugin.so"
                #endif
            case .tpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_tpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_tpu_plugin.so"
                #endif
            }
        }
    }

    /// Low-level C API wrapper
    internal var capi: PJRTCAPI?

    /// Opaque handle to the underlying PJRT_Client
    private var handle: OpaquePointer?

    /// Public accessor for the client handle (for hot path execution)
    public var clientHandle: OpaquePointer? { handle }

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

    /// Host buffer semantics for buffer creation
    public enum HostBufferSemantics {
        /// Data is copied during the call, buffer can be modified after return (default)
        case immutableOnlyDuringCall
        /// Data must remain valid until transfer completes
        case immutableUntilTransferCompletes
        /// Zero-copy: host memory used directly, must remain valid for buffer lifetime
        /// WARNING: Only use when you guarantee the host buffer outlives the PJRT buffer
        case zeroCopy
        /// Mutable zero-copy: for buffer donation - memory can be reused for output
        /// Use with executeWithDonation() to enable buffer reuse for matching shapes
        case mutableZeroCopy
    }

    /// Create a buffer from host data
    ///
    /// - Parameters:
    ///   - data: Host data pointer
    ///   - shape: Tensor shape
    ///   - elementType: Element type
    ///   - device: Target device (nil = default device)
    ///   - semantics: Host buffer semantics (default: immutableOnlyDuringCall)
    /// - Returns: PJRT buffer
    /// - Throws: PJRTError if buffer creation fails
    public func createBuffer(
        data: UnsafeRawPointer,
        shape: [Int],
        elementType: PJRTElementType,
        device: PJRTDevice? = nil,
        semantics: HostBufferSemantics = .immutableOnlyDuringCall
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

        // Map public semantics to internal enum
        let capiSemantics: PJRTCAPI.HostBufferSemantics
        switch semantics {
        case .immutableOnlyDuringCall: capiSemantics = .immutableOnlyDuringCall
        case .immutableUntilTransferCompletes: capiSemantics = .immutableUntilTransferCompletes
        case .zeroCopy: capiSemantics = .zeroCopy
        case .mutableZeroCopy: capiSemantics = .mutableZeroCopy
        }

        // Call PJRT C API wrapper with semantics
        let bufferHandle = try capi.createBuffer(
            client: clientHandle,
            data: data,
            type: elementType.toCType,
            dims: dims,
            device: deviceHandle,
            semantics: capiSemantics
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

    /// Create a buffer using fast path with cached descriptor
    ///
    /// When `isSameShape` is true and the slot was previously used, this avoids
    /// re-initializing the PJRT args structure and only updates the data pointer.
    /// Use this for repeated same-shape inputs to reduce overhead.
    ///
    /// - Parameters:
    ///   - data: Host data pointer
    ///   - shape: Tensor shape
    ///   - elementType: Element type
    ///   - device: Target device (nil = default device)
    ///   - semantics: Host buffer semantics
    ///   - cachedSlot: Cache slot index (0-15)
    ///   - isSameShape: Whether shape matches previous call for this slot
    /// - Returns: PJRT buffer
    public func createBufferFast(
        data: UnsafeRawPointer,
        shape: [Int],
        elementType: PJRTElementType,
        device: PJRTDevice? = nil,
        semantics: HostBufferSemantics = .immutableOnlyDuringCall,
        cachedSlot: Int,
        isSameShape: Bool
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

        // Map public semantics to internal enum
        let capiSemantics: PJRTCAPI.HostBufferSemantics
        switch semantics {
        case .immutableOnlyDuringCall: capiSemantics = .immutableOnlyDuringCall
        case .immutableUntilTransferCompletes: capiSemantics = .immutableUntilTransferCompletes
        case .zeroCopy: capiSemantics = .zeroCopy
        case .mutableZeroCopy: capiSemantics = .mutableZeroCopy
        }

        // Call fast PJRT C API wrapper with cached slot
        let bufferHandle = try capi.createBufferFast(
            client: clientHandle,
            data: data,
            type: elementType.toCType,
            dims: dims,
            device: deviceHandle,
            semantics: capiSemantics,
            cachedSlot: cachedSlot,
            isSameShape: isSameShape
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

    /// Create multiple buffers in a single batched call
    ///
    /// This reduces FFI overhead by batching multiple buffer creations together
    /// into a single C call. Use this when creating several input buffers.
    ///
    /// - Parameters:
    ///   - dataArrays: Array of host data arrays
    ///   - shapes: Array of tensor shapes
    ///   - elementType: Element type (same for all buffers)
    ///   - device: Target device (nil = default device)
    ///   - semantics: Host buffer semantics (default: immutableOnlyDuringCall)
    /// - Returns: Array of PJRT buffers
    /// - Throws: PJRTError if buffer creation fails
    public func createBuffersBatched(
        dataArrays: [[Float]],
        shapes: [[Int]],
        elementType: PJRTElementType = .f32,
        device: PJRTDevice? = nil,
        semantics: HostBufferSemantics = .immutableOnlyDuringCall
    ) throws -> [PJRTBuffer] {
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

        guard dataArrays.count == shapes.count else {
            throw PJRTError.bufferCreationFailed("Mismatched data and shapes arrays")
        }

        let numBuffers = dataArrays.count
        guard numBuffers > 0 else {
            return []
        }

        // Convert shapes to Int64 arrays
        let dimsArrays = shapes.map { $0.map { Int64($0) } }

        // Map semantics
        let capiSemantics: PJRTCAPI.HostBufferSemantics
        switch semantics {
        case .immutableOnlyDuringCall: capiSemantics = .immutableOnlyDuringCall
        case .immutableUntilTransferCompletes: capiSemantics = .immutableUntilTransferCompletes
        case .zeroCopy: capiSemantics = .zeroCopy
        case .mutableZeroCopy: capiSemantics = .mutableZeroCopy
        }

        // Collect data pointers
        var dataPointers: [UnsafeRawPointer] = []
        dataPointers.reserveCapacity(numBuffers)

        // Need to call the batched API with pointers into the arrays
        // We'll use withUnsafeBytes for each array
        var bufferHandles: [OpaquePointer] = []

        // Create types array (all same type)
        let types = [SW_PJRT_Buffer_Type](repeating: elementType.toCType, count: numBuffers)

        // We need a different approach - collect pointers manually
        try dataArrays[0].withUnsafeBytes { ptr0 in
            if numBuffers == 1 {
                // Single buffer case
                let handles = try capi.createBuffersBatched(
                    client: clientHandle,
                    dataPointers: [ptr0.baseAddress!],
                    types: types,
                    dimsArrays: dimsArrays,
                    device: deviceHandle,
                    semantics: capiSemantics
                )
                bufferHandles = handles
            } else if numBuffers == 2 {
                try dataArrays[1].withUnsafeBytes { ptr1 in
                    let handles = try capi.createBuffersBatched(
                        client: clientHandle,
                        dataPointers: [ptr0.baseAddress!, ptr1.baseAddress!],
                        types: types,
                        dimsArrays: dimsArrays,
                        device: deviceHandle,
                        semantics: capiSemantics
                    )
                    bufferHandles = handles
                }
            } else {
                // For more than 2 buffers, fall back to sequential creation
                // to avoid deeply nested closures
                for (i, data) in dataArrays.enumerated() {
                    let buffer = try data.withUnsafeBytes { ptr in
                        try createBuffer(
                            data: ptr.baseAddress!,
                            shape: shapes[i],
                            elementType: elementType,
                            device: targetDevice,
                            semantics: semantics
                        )
                    }
                    bufferHandles.append(buffer.handle!)
                }
            }
        }

        // Create PJRTBuffer wrappers
        var buffers: [PJRTBuffer] = []
        buffers.reserveCapacity(numBuffers)

        for (i, handle) in bufferHandles.enumerated() {
            let buffer = PJRTBuffer(
                shape: shapes[i],
                elementType: elementType,
                device: targetDevice
            )
            buffer.handle = handle
            buffers.append(buffer)
        }

        return buffers
    }

    /// Compile a program
    ///
    /// - Parameters:
    ///   - mlirModule: MLIR module containing StableHLO operations
    ///   - devices: Devices to compile for (nil = all addressable devices)
    ///   - xlaOptLevel: XLA backend optimization level (default = use XLA's default)
    /// - Returns: Loaded executable
    /// - Throws: PJRTError if compilation fails
    public func compile(
        mlirModule: String,
        devices: [PJRTDevice]? = nil,
        xlaOptLevel: XLAOptimizationLevel = .default
    ) throws -> PJRTLoadedExecutable {
        let targetDevices = devices ?? addressableDevices

        guard let capi = self.capi, let clientHandle = self.handle else {
            throw PJRTError.compilationFailed("Client not initialized")
        }

        // Call PJRT C API wrapper to compile the MLIR module
        let executableHandle = try capi.compile(
            client: clientHandle,
            mlirModule: mlirModule,
            xlaOptLevel: xlaOptLevel
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

    /// Public accessor for the device handle (for hot path execution)
    public var deviceHandle: OpaquePointer? { handle }

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

    /// Copy buffer contents to host (synchronous)
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

    /// Copy buffer contents to host (asynchronous)
    /// Returns immediately, caller must await the event before accessing destination
    ///
    /// - Parameter destination: Host memory pointer (must remain valid until event completes)
    /// - Returns: Event to await before accessing destination data
    /// - Throws: PJRTError if transfer initiation fails
    public func toHostAsync(destination: UnsafeMutableRawPointer) throws -> PJRTEvent? {
        guard let handle = handle else {
            throw PJRTError.bufferTransferFailed("Buffer handle not available")
        }

        guard let capi = device.client?.capi else {
            throw PJRTError.bufferTransferFailed("Client not available")
        }

        // Call PJRT C API wrapper to initiate async transfer
        let eventHandle = try capi.bufferToHostAsync(
            buffer: handle,
            destination: destination,
            size: sizeInBytes
        )

        return PJRTEvent(handle: eventHandle, client: device.client)
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

    /// Public accessor for the executable handle (for buffer pool creation)
    public var executableHandle: OpaquePointer? { handle }

    /// Execution statistics
    public private(set) var executionCount: Int = 0

    /// Cached output metadata (shapes and element counts) - populated after first execution
    private var cachedOutputMetadata: [(shape: [Int], elementCount: Int)]?

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

        // Use cached metadata if available (avoids C API calls after first execution)
        if let cached = cachedOutputMetadata, cached.count == outputHandles.count {
            // Fast path: use cached metadata
            var outputBuffers: [PJRTBuffer] = []
            outputBuffers.reserveCapacity(outputHandles.count)

            for (i, outputHandle) in outputHandles.enumerated() {
                let buffer = PJRTBuffer(
                    shape: cached[i].shape,
                    elementType: .f32,
                    device: targetDevice,
                    didIncRef: false
                )
                buffer.handle = outputHandle
                outputBuffers.append(buffer)
            }
            return outputBuffers
        }

        // First execution: query metadata and cache it
        var metadata: [(shape: [Int], elementCount: Int)] = []
        metadata.reserveCapacity(outputHandles.count)

        let outputBuffers = try outputHandles.map { outputHandle -> PJRTBuffer in
            // Query dimensions (only on first execution)
            let dims = try capi.getBufferDimensions(buffer: outputHandle)
            let shape = dims.map { Int($0) }
            let elementCount = shape.reduce(1, *)

            // Cache the metadata
            metadata.append((shape: shape, elementCount: elementCount))

            let buffer = PJRTBuffer(
                shape: shape,
                elementType: .f32,
                device: targetDevice,
                didIncRef: false
            )
            buffer.handle = outputHandle
            return buffer
        }

        // Store cached metadata for subsequent calls
        cachedOutputMetadata = metadata

        return outputBuffers
    }

    /// Execute the program with buffer donation support
    ///
    /// Buffer donation allows input buffers to be reused for outputs with matching shapes.
    /// This avoids allocating new output buffers and can significantly improve performance.
    ///
    /// - Parameters:
    ///   - arguments: Input buffers
    ///   - device: Device to execute on (nil = first device)
    ///   - nonDonatableIndices: Input indices that should NOT be donated (empty = donate all eligible)
    /// - Returns: Output buffers
    /// - Throws: PJRTError if execution fails
    /// - Note: After execution, donated input buffers become invalid and must not be used.
    ///         For donation to work, the MLIR module must include input_output_alias attributes.
    public func executeWithDonation(
        arguments: [PJRTBuffer],
        device: PJRTDevice? = nil,
        nonDonatableIndices: [Int] = []
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

        // Convert indices to Int64 for C API
        let indices = nonDonatableIndices.map { Int64($0) }

        // Call PJRT C API wrapper with donation support
        let outputHandles = try capi.executeWithDonation(
            executable: handle,
            inputs: inputHandles,
            nonDonatableIndices: indices
        )

        executionCount += 1

        // Use cached metadata if available (avoids C API calls after first execution)
        if let cached = cachedOutputMetadata, cached.count == outputHandles.count {
            // Fast path: use cached metadata
            var outputBuffers: [PJRTBuffer] = []
            outputBuffers.reserveCapacity(outputHandles.count)

            for (i, outputHandle) in outputHandles.enumerated() {
                let buffer = PJRTBuffer(
                    shape: cached[i].shape,
                    elementType: .f32,
                    device: targetDevice,
                    didIncRef: false
                )
                buffer.handle = outputHandle
                outputBuffers.append(buffer)
            }
            return outputBuffers
        }

        // First execution: query metadata and cache it
        var metadata: [(shape: [Int], elementCount: Int)] = []
        metadata.reserveCapacity(outputHandles.count)

        let outputBuffers = try outputHandles.map { outputHandle -> PJRTBuffer in
            // Query dimensions (only on first execution)
            let dims = try capi.getBufferDimensions(buffer: outputHandle)
            let shape = dims.map { Int($0) }
            let elementCount = shape.reduce(1, *)

            // Cache the metadata
            metadata.append((shape: shape, elementCount: elementCount))

            let buffer = PJRTBuffer(
                shape: shape,
                elementType: .f32,
                device: targetDevice,
                didIncRef: false
            )
            buffer.handle = outputHandle
            return buffer
        }

        // Store cached metadata for subsequent calls
        cachedOutputMetadata = metadata

        return outputBuffers
    }

    /// Execute and transfer outputs to host in a single optimized call
    /// This reduces FFI overhead by combining execute + async D2H + await into one C call.
    ///
    /// - Parameters:
    ///   - arguments: Input buffers
    ///   - outputSizes: Size in bytes for each expected output
    /// - Returns: Array of Float arrays containing the output data
    /// - Throws: PJRTError if execution fails
    public func executeAndTransfer(
        arguments: [PJRTBuffer],
        outputSizes: [Int]
    ) throws -> [[Float]] {
        guard let handle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        guard let capi = client?.capi else {
            throw PJRTError.executionFailed("Client not available")
        }

        // Extract buffer handles from arguments
        let inputHandles = try arguments.compactMap { arg -> OpaquePointer? in
            arg.handle
        }

        return try capi.executeAndTransfer(
            executable: handle,
            inputBuffers: inputHandles,
            outputSizes: outputSizes
        )
    }

    /// Execute with hot path optimization: H2D + Execute + D2H in a single FFI call
    ///
    /// This is the most optimized path for repeated execution with the same shapes.
    /// It combines buffer creation, execution, and transfer all in one C call to
    /// minimize FFI overhead.
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input float data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of mutable raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    ///   - semantics: Host buffer semantics (default: .zeroCopy for CPU)
    /// - Throws: PJRTError if execution fails
    public func executeHotPath(
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int],
        semantics: PJRTClient.HostBufferSemantics = .zeroCopy
    ) throws {
        guard let handle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        guard let client = client,
              let clientHandle = client.clientHandle,
              let capi = client.capi else {
            throw PJRTError.executionFailed("Client not available")
        }

        guard let device = devices.first,
              let deviceHandle = device.deviceHandle else {
            throw PJRTError.noDeviceAvailable
        }

        try capi.executeHotPath(
            client: clientHandle,
            executable: handle,
            device: deviceHandle,
            inputData: inputData,
            inputSizes: inputSizes,
            outputData: outputData,
            outputSizes: outputSizes,
            semantics: PJRTCAPI.HostBufferSemantics(semantics)
        )

        executionCount += 1
    }

    // MARK: - Profiled Execution

    /// Execute with detailed timing breakdown
    /// This profiles the complete execution path: H2D + Execute + D2H
    ///
    /// Use this to identify performance bottlenecks in the PJRT execution pipeline.
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input float data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    ///   - semantics: Host buffer semantics (default: zeroCopy)
    /// - Returns: PJRTExecutionTiming with breakdown of where time was spent
    public func executeProfiled(
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int],
        semantics: PJRTClient.HostBufferSemantics = .zeroCopy
    ) throws -> PJRTExecutionTiming {
        guard let capi = client?.capi,
              let clientHandle = client?.clientHandle else {
            throw PJRTError.clientCreationFailed("Client not available")
        }

        guard let device = devices.first,
              let deviceHandle = device.deviceHandle else {
            throw PJRTError.noDeviceAvailable
        }

        guard let execHandle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        let timing = try capi.executeProfiled(
            client: clientHandle,
            executable: execHandle,
            device: deviceHandle,
            inputData: inputData,
            inputSizes: inputSizes,
            outputData: outputData,
            outputSizes: outputSizes,
            semantics: PJRTCAPI.HostBufferSemantics(semantics)
        )

        executionCount += 1
        return timing
    }

    /// Execute with OnReady callbacks instead of blocking await
    /// This uses PJRT_Event_OnReady for D2H completion instead of PJRT_Event_Await.
    /// Returns timing breakdown for comparison with blocking version.
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    ///   - semantics: Host buffer semantics (default: zeroCopy)
    /// - Returns: PJRTExecutionTiming with breakdown of where time was spent
    public func executeWithCallbacks(
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int],
        semantics: PJRTClient.HostBufferSemantics = .zeroCopy
    ) throws -> PJRTExecutionTiming {
        guard let capi = client?.capi,
              let clientHandle = client?.clientHandle else {
            throw PJRTError.clientCreationFailed("Client not available")
        }

        guard let device = devices.first,
              let deviceHandle = device.deviceHandle else {
            throw PJRTError.noDeviceAvailable
        }

        guard let execHandle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        let timing = try pjrtExecuteWithCallbacks(
            client: clientHandle,
            executable: execHandle,
            device: deviceHandle,
            inputData: inputData,
            inputSizes: inputSizes,
            outputData: outputData,
            outputSizes: outputSizes,
            semantics: semantics
        )

        executionCount += 1
        return timing
    }

    /// Create a buffer pool for this executable
    ///
    /// The pool uses zero-copy semantics for optimal performance with repeated
    /// executions of the same input sizes.
    ///
    /// - Parameter inputSizes: Number of elements for each input
    /// - Returns: Buffer pool for pooled execution
    /// - Throws: PJRTError on failure
    public func createBufferPool(inputSizes: [Int]) throws -> PJRTBufferPool {
        guard let clientHandle = client?.clientHandle else {
            throw PJRTError.clientCreationFailed("Client not available")
        }

        guard let device = devices.first,
              let deviceHandle = device.deviceHandle else {
            throw PJRTError.noDeviceAvailable
        }

        guard let execHandle = handle else {
            throw PJRTError.executionFailed("Executable handle not available")
        }

        return try PJRTBufferPool(
            client: clientHandle,
            device: deviceHandle,
            executable: execHandle,
            inputSizes: inputSizes
        )
    }

    /// Execute using a pre-created buffer pool with zero-copy semantics
    ///
    /// This method uses zero-copy host buffer semantics for optimal performance.
    /// The pool should be created once and reused for multiple executions.
    ///
    /// - Parameters:
    ///   - pool: Buffer pool created for this executable
    ///   - inputData: Array of raw pointers to input data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    /// - Returns: PJRTExecutionTiming with breakdown of where time was spent
    /// - Throws: PJRTError on failure
    public func executePooled(
        pool: PJRTBufferPool,
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int]
    ) throws -> PJRTExecutionTiming {
        let timing = try pool.execute(
            inputData: inputData,
            inputSizes: inputSizes,
            outputData: outputData,
            outputSizes: outputSizes
        )

        executionCount += 1
        return timing
    }
}

/// PJRT Event handle for async operations
public class PJRTEvent {
    /// Opaque handle to PJRT event
    internal var handle: OpaquePointer?

    /// Reference to client for cleanup
    private weak var client: PJRTClient?

    init(handle: OpaquePointer?, client: PJRTClient?) {
        self.handle = handle
        self.client = client
    }

    deinit {
        // Destroy any un-awaited event
        if let handle = handle, let capi = client?.capi {
            capi.eventDestroy(handle)
        }
    }

    /// Wait for the event to complete
    /// - Throws: PJRTError if await fails
    public func await() throws {
        guard let handle = handle else { return }
        guard let capi = client?.capi else {
            throw PJRTError.bufferTransferFailed("Client not available for event await")
        }
        try capi.eventAwait(handle)
    }

    /// Wait for the event to complete and release resources
    /// - Throws: PJRTError if await fails
    public func awaitAndDestroy() throws {
        guard let handle = handle else { return }
        guard let capi = client?.capi else {
            throw PJRTError.bufferTransferFailed("Client not available for event await")
        }
        try capi.eventAwaitAndDestroy(handle)
        self.handle = nil  // Mark as consumed
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
