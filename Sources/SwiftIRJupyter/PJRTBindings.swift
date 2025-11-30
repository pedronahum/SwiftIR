// SwiftIRJupyter - PJRT Bindings via dlopen
// Pure Swift implementation - no C++ interop

import Foundation

// MARK: - PJRT Error Codes

public enum PJRTErrorCode: Int32 {
    case ok = 0
    case cancelled = 1
    case unknown = 2
    case invalidArgument = 3
    case deadlineExceeded = 4
    case notFound = 5
    case alreadyExists = 6
    case permissionDenied = 7
    case resourceExhausted = 8
    case failedPrecondition = 9
    case aborted = 10
    case outOfRange = 11
    case unimplemented = 12
    case `internal` = 13
    case unavailable = 14
    case dataLoss = 15
    case unauthenticated = 16
}

// MARK: - PJRT Buffer Types

public enum PJRTBufferType: Int32 {
    case pred = 0
    case s8 = 1
    case s16 = 2
    case s32 = 3
    case s64 = 4
    case u8 = 5
    case u16 = 6
    case u32 = 7
    case u64 = 8
    case f16 = 9
    case f32 = 10
    case f64 = 11
    case bf16 = 12
    case c64 = 13
    case c128 = 14
}

// MARK: - PJRT Function Types

public enum PJRTFunctions {
    // Plugin loading
    public typealias LoadPlugin = @convention(c) (UnsafePointer<CChar>) -> Int32
    public typealias UnloadPlugin = @convention(c) () -> Void
    public typealias GetLastError = @convention(c) () -> UnsafePointer<CChar>?

    // Error handling
    public typealias GetErrorCode = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    public typealias GetErrorMessage = @convention(c) (UnsafeMutableRawPointer?) -> UnsafePointer<CChar>?
    public typealias DestroyError = @convention(c) (UnsafeMutableRawPointer?) -> Void

    // Client management
    public typealias CreateClient = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
    public typealias DestroyClient = @convention(c) (UnsafeMutableRawPointer?) -> Void
    public typealias GetPlatformName = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafePointer<CChar>?>) -> Int32

    // Device enumeration
    public typealias GetAddressableDevices = @convention(c) (
        UnsafeMutableRawPointer?,
        UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutableRawPointer?>?>,
        UnsafeMutablePointer<Int>
    ) -> Int32
    public typealias GetDeviceId = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Int32>) -> Int32
    public typealias GetDeviceKind = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafePointer<CChar>?>) -> Int32

    // Buffer management
    public typealias CreateBuffer = @convention(c) (
        UnsafeMutableRawPointer?,       // client
        UnsafeRawPointer?,              // data
        Int32,                          // type
        UnsafePointer<Int64>,           // dims
        Int,                            // num_dims
        UnsafeMutableRawPointer?,       // device
        UnsafeMutablePointer<UnsafeMutableRawPointer?> // out_buffer
    ) -> Int32
    public typealias DestroyBuffer = @convention(c) (UnsafeMutableRawPointer?) -> Void
    public typealias BufferToHost = @convention(c) (
        UnsafeMutableRawPointer?,       // buffer
        UnsafeMutableRawPointer?,       // out_data
        Int                             // data_size
    ) -> Int32
    public typealias GetBufferDimensions = @convention(c) (
        UnsafeMutableRawPointer?,
        UnsafeMutablePointer<UnsafePointer<Int64>?>,
        UnsafeMutablePointer<Int>
    ) -> Int32
    public typealias GetBufferOnDeviceSizeInBytes = @convention(c) (
        UnsafeMutableRawPointer?,
        UnsafeMutablePointer<Int>
    ) -> Int32

    // Compilation & Execution
    public typealias CompileWrapper = @convention(c) (
        UnsafeMutableRawPointer?,       // client
        UnsafePointer<CChar>,           // mlir_module
        UnsafeMutablePointer<UnsafeMutableRawPointer?> // out_executable
    ) -> Int32
    public typealias DestroyExecutable = @convention(c) (UnsafeMutableRawPointer?) -> Void
    public typealias ExecuteWrapper = @convention(c) (
        UnsafeMutableRawPointer?,       // executable
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, // inputs
        Int,                            // num_inputs
        UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutableRawPointer?>?>, // out_outputs
        UnsafeMutablePointer<Int>       // out_num_outputs
    ) -> Int32
}

// MARK: - PJRT Bindings

/// Dynamic bindings to PJRT C API (via PJRTSimpleWrapper)
public final class PJRTBindings: @unchecked Sendable {

    /// Shared instance
    nonisolated(unsafe) public static let shared = PJRTBindings()

    /// Whether bindings are loaded
    public private(set) var isLoaded = false

    /// Whether a plugin is loaded
    public private(set) var isPluginLoaded = false

    // Function pointers
    private var _loadPlugin: PJRTFunctions.LoadPlugin?
    private var _unloadPlugin: PJRTFunctions.UnloadPlugin?
    private var _getLastError: PJRTFunctions.GetLastError?
    private var _getErrorCode: PJRTFunctions.GetErrorCode?
    private var _getErrorMessage: PJRTFunctions.GetErrorMessage?
    private var _destroyError: PJRTFunctions.DestroyError?
    private var _createClient: PJRTFunctions.CreateClient?
    private var _destroyClient: PJRTFunctions.DestroyClient?
    private var _getPlatformName: PJRTFunctions.GetPlatformName?
    private var _getAddressableDevices: PJRTFunctions.GetAddressableDevices?
    private var _getDeviceId: PJRTFunctions.GetDeviceId?
    private var _getDeviceKind: PJRTFunctions.GetDeviceKind?
    private var _createBuffer: PJRTFunctions.CreateBuffer?
    private var _destroyBuffer: PJRTFunctions.DestroyBuffer?
    private var _bufferToHost: PJRTFunctions.BufferToHost?
    private var _getBufferDimensions: PJRTFunctions.GetBufferDimensions?
    private var _getBufferOnDeviceSizeInBytes: PJRTFunctions.GetBufferOnDeviceSizeInBytes?
    private var _compileWrapper: PJRTFunctions.CompileWrapper?
    private var _destroyExecutable: PJRTFunctions.DestroyExecutable?
    private var _executeWrapper: PJRTFunctions.ExecuteWrapper?

    private init() {}

    /// Load PJRT bindings from the native library
    public func load() throws {
        guard !isLoaded else { return }

        let loader = LibraryLoader.shared

        // Load PJRTProtoHelper first (dependency of PJRTSimpleWrapper)
        try loader.load("PJRTProtoHelper")

        // Load PJRTSimpleWrapper which provides the wrapper functions
        try loader.load("PJRTSimpleWrapper")

        // Load function pointers
        _loadPlugin = try loader.symbol("PJRT_LoadPlugin")
        _unloadPlugin = try loader.symbol("PJRT_UnloadPlugin")
        _getLastError = try loader.symbol("PJRT_GetLastError")
        _getErrorCode = try loader.symbol("PJRT_GetErrorCode")
        _getErrorMessage = try loader.symbol("PJRT_GetErrorMessage")
        _destroyError = try loader.symbol("PJRT_DestroyError")
        _createClient = try loader.symbol("PJRT_CreateClient")
        _destroyClient = try loader.symbol("PJRT_DestroyClient")
        _getPlatformName = try loader.symbol("PJRT_GetPlatformName")
        _getAddressableDevices = try loader.symbol("PJRT_GetAddressableDevices")
        _getDeviceId = try loader.symbol("PJRT_GetDeviceId")
        _getDeviceKind = try loader.symbol("PJRT_GetDeviceKind")
        _createBuffer = try loader.symbol("PJRT_CreateBuffer")
        _destroyBuffer = try loader.symbol("PJRT_DestroyBuffer")
        _bufferToHost = try loader.symbol("PJRT_BufferToHost")
        _getBufferDimensions = try loader.symbol("PJRT_GetBufferDimensions")
        _getBufferOnDeviceSizeInBytes = try loader.symbol("PJRT_GetBufferOnDeviceSizeInBytes")
        _compileWrapper = try loader.symbol("PJRT_CompileWrapper")
        _destroyExecutable = try loader.symbol("PJRT_DestroyExecutable")
        _executeWrapper = try loader.symbol("PJRT_ExecuteWrapper")

        isLoaded = true
    }

    // MARK: - Plugin Loading

    public func loadPlugin(path: String) throws {
        guard let fn = _loadPlugin else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        let code = fn(path)
        if code != PJRTErrorCode.ok.rawValue {
            let errorMsg = _getLastError?().map { String(cString: $0) } ?? "Unknown error"
            throw SwiftIRJupyterError.pjrtError(code: code, message: errorMsg)
        }
        isPluginLoaded = true
    }

    public func unloadPlugin() {
        _unloadPlugin?()
        isPluginLoaded = false
    }

    // MARK: - Client Management

    public func createClient() throws -> UnsafeMutableRawPointer {
        guard let fn = _createClient else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var client: UnsafeMutableRawPointer?
        let code = fn(&client)

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to create client")
        }

        guard let result = client else {
            throw SwiftIRJupyterError.pjrtError(code: -1, message: "Client is null")
        }

        return result
    }

    public func destroyClient(_ client: UnsafeMutableRawPointer) {
        _destroyClient?(client)
    }

    public func getPlatformName(_ client: UnsafeMutableRawPointer) throws -> String {
        guard let fn = _getPlatformName else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var namePtr: UnsafePointer<CChar>?
        let code = fn(client, &namePtr)

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to get platform name")
        }

        return namePtr.map { String(cString: $0) } ?? "unknown"
    }

    // MARK: - Device Enumeration

    public func getAddressableDevices(_ client: UnsafeMutableRawPointer) throws -> [UnsafeMutableRawPointer] {
        guard let fn = _getAddressableDevices else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var devicesPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
        var numDevices: Int = 0

        let code = fn(client, &devicesPtr, &numDevices)

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to get devices")
        }

        guard let devices = devicesPtr else {
            return []
        }

        var result: [UnsafeMutableRawPointer] = []
        for i in 0..<numDevices {
            if let device = devices[i] {
                result.append(device)
            }
        }

        return result
    }

    public func getDeviceId(_ device: UnsafeMutableRawPointer) throws -> Int32 {
        guard let fn = _getDeviceId else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var deviceId: Int32 = 0
        let code = fn(device, &deviceId)

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to get device ID")
        }

        return deviceId
    }

    public func getDeviceKind(_ device: UnsafeMutableRawPointer) throws -> String {
        guard let fn = _getDeviceKind else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var kindPtr: UnsafePointer<CChar>?
        let code = fn(device, &kindPtr)

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to get device kind")
        }

        return kindPtr.map { String(cString: $0) } ?? "unknown"
    }

    // MARK: - Buffer Management

    public func createBuffer<T>(
        client: UnsafeMutableRawPointer,
        data: [T],
        type: PJRTBufferType,
        shape: [Int64],
        device: UnsafeMutableRawPointer
    ) throws -> UnsafeMutableRawPointer {
        guard let fn = _createBuffer else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var buffer: UnsafeMutableRawPointer?

        let code = data.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                fn(
                    client,
                    dataPtr.baseAddress,
                    type.rawValue,
                    shapePtr.baseAddress!,
                    shape.count,
                    device,
                    &buffer
                )
            }
        }

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to create buffer")
        }

        guard let result = buffer else {
            throw SwiftIRJupyterError.pjrtError(code: -1, message: "Buffer is null")
        }

        return result
    }

    public func destroyBuffer(_ buffer: UnsafeMutableRawPointer) {
        _destroyBuffer?(buffer)
    }

    public func bufferToHost<T>(_ buffer: UnsafeMutableRawPointer, into array: inout [T]) throws {
        guard let fn = _bufferToHost else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        let size = array.count * MemoryLayout<T>.stride

        let code = array.withUnsafeMutableBytes { ptr in
            fn(buffer, ptr.baseAddress, size)
        }

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Failed to transfer buffer to host")
        }
    }

    // MARK: - Compilation & Execution

    public func compile(client: UnsafeMutableRawPointer, mlirModule: String) throws -> UnsafeMutableRawPointer {
        guard let fn = _compileWrapper else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var executable: UnsafeMutableRawPointer?

        let code = mlirModule.withCString { modulePtr in
            fn(client, modulePtr, &executable)
        }

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Compilation failed")
        }

        guard let result = executable else {
            throw SwiftIRJupyterError.pjrtError(code: -1, message: "Executable is null")
        }

        return result
    }

    public func destroyExecutable(_ executable: UnsafeMutableRawPointer) {
        _destroyExecutable?(executable)
    }

    public func execute(
        executable: UnsafeMutableRawPointer,
        inputs: [UnsafeMutableRawPointer]
    ) throws -> [UnsafeMutableRawPointer] {
        guard let fn = _executeWrapper else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT bindings not loaded")
        }

        var outputsPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
        var numOutputs: Int = 0

        var inputsCopy = inputs.map { Optional($0) }

        let code = inputsCopy.withUnsafeMutableBufferPointer { inputsBuffer in
            fn(
                executable,
                inputsBuffer.baseAddress,
                inputs.count,
                &outputsPtr,
                &numOutputs
            )
        }

        if code != PJRTErrorCode.ok.rawValue {
            throw SwiftIRJupyterError.pjrtError(code: code, message: "Execution failed")
        }

        guard let outputs = outputsPtr else {
            return []
        }

        var result: [UnsafeMutableRawPointer] = []
        for i in 0..<numOutputs {
            if let output = outputs[i] {
                result.append(output)
            }
        }

        return result
    }
}
