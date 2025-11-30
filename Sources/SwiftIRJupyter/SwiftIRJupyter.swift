// SwiftIRJupyter - High-level API for Jupyter/Colab
// Pure Swift - works in LLDB REPL without C++ interop

import Foundation

// MARK: - SwiftIRJupyter Main API

/// Main entry point for SwiftIR in Jupyter/Colab notebooks
public final class SwiftIRJupyter: @unchecked Sendable {

    /// Shared instance
    nonisolated(unsafe) public static let shared = SwiftIRJupyter()

    /// Whether SwiftIR is initialized
    public private(set) var isInitialized = false

    /// SDK path
    public var sdkPath: String {
        LibraryLoader.shared.sdkPath
    }

    private init() {}

    /// Initialize SwiftIR for Jupyter usage
    /// - Parameter pluginPath: Optional path to PJRT plugin. If nil, uses CPU plugin from SDK.
    public func initialize(pluginPath: String? = nil) throws {
        guard !isInitialized else { return }

        print("🚀 Initializing SwiftIRJupyter...")
        print("   SDK path: \(sdkPath)")

        // Load MLIR bindings
        print("   Loading MLIR bindings...")
        try MLIRBindings.shared.load()
        print("   ✅ MLIR bindings loaded")

        // Load PJRT bindings
        print("   Loading PJRT bindings...")
        try PJRTBindings.shared.load()
        print("   ✅ PJRT bindings loaded")

        // Load PJRT plugin
        let plugin = pluginPath ?? "\(sdkPath)/lib/pjrt_c_api_cpu_plugin.so"
        print("   Loading PJRT plugin: \(plugin)")
        try PJRTBindings.shared.loadPlugin(path: plugin)
        print("   ✅ PJRT plugin loaded")

        isInitialized = true
        print("✅ SwiftIRJupyter initialized successfully!")
    }

    /// Print system information
    public func printInfo() {
        let initStr = isInitialized ? "Yes" : "No"
        let mlirStr = MLIRBindings.shared.isLoaded ? "Yes" : "No"
        let pjrtStr = PJRTBindings.shared.isLoaded ? "Yes" : "No"
        let pluginStr = PJRTBindings.shared.isPluginLoaded ? "Yes" : "No"

        print("SwiftIR Jupyter Info")
        print("====================")
        print("SDK Path:      \(sdkPath)")
        print("Initialized:   \(initStr)")
        print("MLIR Loaded:   \(mlirStr)")
        print("PJRT Loaded:   \(pjrtStr)")
        print("Plugin Loaded: \(pluginStr)")
    }
}

// MARK: - MLIRContext (High-level wrapper)

/// High-level MLIR context wrapper
public final class JupyterMLIRContext {
    private let ctx: MLIRContextRef
    private let bindings = MLIRBindings.shared

    public init() throws {
        guard bindings.isLoaded else {
            throw SwiftIRJupyterError.invalidState(message: "MLIR bindings not loaded. Call SwiftIRJupyter.shared.initialize() first.")
        }
        self.ctx = try bindings.contextCreate()
        bindings.contextLoadAllDialects(ctx)
    }

    deinit {
        bindings.contextDestroy(ctx)
    }

    /// Create a new empty module
    public func createModule() throws -> JupyterMLIRModule {
        try JupyterMLIRModule(context: ctx)
    }

    /// Get F32 type
    public func f32Type() throws -> MLIRTypeRef {
        try bindings.f32TypeGet(ctx)
    }

    /// Get F64 type
    public func f64Type() throws -> MLIRTypeRef {
        try bindings.f64TypeGet(ctx)
    }

    /// Get integer type
    public func integerType(bitwidth: UInt32) throws -> MLIRTypeRef {
        try bindings.integerTypeGet(ctx, bitwidth: bitwidth)
    }

    /// Get index type
    public func indexType() throws -> MLIRTypeRef {
        try bindings.indexTypeGet(ctx)
    }

    /// Internal context reference
    internal var ref: MLIRContextRef { ctx }
}

// MARK: - MLIRModule (High-level wrapper)

/// High-level MLIR module wrapper
public final class JupyterMLIRModule {
    private let module: MLIRModuleRef
    private let bindings = MLIRBindings.shared

    internal init(context: MLIRContextRef) throws {
        let loc = try bindings.locationUnknownGet(context)
        self.module = try bindings.moduleCreateEmpty(loc)
    }

    deinit {
        bindings.moduleDestroy(module)
    }

    /// Print the module as MLIR text
    public func print() throws -> String {
        let op = try bindings.moduleGetOperation(module)
        return try bindings.operationPrint(op)
    }

    /// Verify the module
    public func verify() throws -> Bool {
        let op = try bindings.moduleGetOperation(module)
        return bindings.operationVerify(op)
    }

    /// Internal module reference
    internal var ref: MLIRModuleRef { module }
}

// MARK: - PJRTClient (High-level wrapper)

/// High-level PJRT client wrapper
public final class JupyterPJRTClient {
    private let client: UnsafeMutableRawPointer
    private let bindings = PJRTBindings.shared
    private var devices: [UnsafeMutableRawPointer] = []

    public init() throws {
        guard bindings.isLoaded && bindings.isPluginLoaded else {
            throw SwiftIRJupyterError.invalidState(message: "PJRT not initialized. Call SwiftIRJupyter.shared.initialize() first.")
        }
        self.client = try bindings.createClient()
        self.devices = try bindings.getAddressableDevices(client)
    }

    deinit {
        bindings.destroyClient(client)
    }

    /// Platform name (e.g., "cpu", "cuda")
    public var platformName: String {
        (try? bindings.getPlatformName(client)) ?? "unknown"
    }

    /// Number of devices
    public var deviceCount: Int {
        devices.count
    }

    /// Get device info
    public func deviceInfo(index: Int) throws -> (id: Int32, kind: String) {
        guard index < devices.count else {
            throw SwiftIRJupyterError.invalidState(message: "Device index out of range")
        }
        let device = devices[index]
        let id = try bindings.getDeviceId(device)
        let kind = try bindings.getDeviceKind(device)
        return (id, kind)
    }

    /// First device (convenience)
    public var firstDevice: UnsafeMutableRawPointer? {
        devices.first
    }

    /// Create a buffer from Float array
    public func createBuffer(data: [Float], shape: [Int64]) throws -> JupyterPJRTBuffer {
        guard let device = firstDevice else {
            throw SwiftIRJupyterError.invalidState(message: "No devices available")
        }
        let bufferPtr = try bindings.createBuffer(
            client: client,
            data: data,
            type: .f32,
            shape: shape,
            device: device
        )
        return JupyterPJRTBuffer(buffer: bufferPtr, shape: shape, type: .f32)
    }

    /// Create a buffer from Int32 array
    public func createBuffer(data: [Int32], shape: [Int64]) throws -> JupyterPJRTBuffer {
        guard let device = firstDevice else {
            throw SwiftIRJupyterError.invalidState(message: "No devices available")
        }
        let bufferPtr = try bindings.createBuffer(
            client: client,
            data: data,
            type: .s32,
            shape: shape,
            device: device
        )
        return JupyterPJRTBuffer(buffer: bufferPtr, shape: shape, type: .s32)
    }

    /// Compile an MLIR module
    public func compile(mlir: String) throws -> JupyterPJRTExecutable {
        let execPtr = try bindings.compile(client: client, mlirModule: mlir)
        return JupyterPJRTExecutable(executable: execPtr)
    }

    /// Print client info
    public func printInfo() {
        print("PJRT Client:")
        print("  Platform: \(platformName)")
        print("  Devices: \(deviceCount)")
        for i in 0..<deviceCount {
            if let info = try? deviceInfo(index: i) {
                print("    [\(i)] ID=\(info.id), Kind=\(info.kind)")
            }
        }
    }

    /// Internal client reference
    internal var ref: UnsafeMutableRawPointer { client }
}

// MARK: - PJRTBuffer (High-level wrapper)

/// High-level PJRT buffer wrapper
public final class JupyterPJRTBuffer {
    fileprivate let buffer: UnsafeMutableRawPointer
    private let bindings = PJRTBindings.shared
    public let shape: [Int64]
    public let type: PJRTBufferType

    internal init(buffer: UnsafeMutableRawPointer, shape: [Int64], type: PJRTBufferType) {
        self.buffer = buffer
        self.shape = shape
        self.type = type
    }

    deinit {
        bindings.destroyBuffer(buffer)
    }

    /// Total number of elements
    public var elementCount: Int {
        shape.reduce(1) { $0 * Int($1) }
    }

    /// Transfer to host as Float array
    public func toHostFloat() throws -> [Float] {
        guard type == .f32 else {
            throw SwiftIRJupyterError.invalidState(message: "Buffer type is not F32")
        }
        var result = [Float](repeating: 0, count: elementCount)
        try bindings.bufferToHost(buffer, into: &result)
        return result
    }

    /// Transfer to host as Int32 array
    public func toHostInt32() throws -> [Int32] {
        guard type == .s32 else {
            throw SwiftIRJupyterError.invalidState(message: "Buffer type is not S32")
        }
        var result = [Int32](repeating: 0, count: elementCount)
        try bindings.bufferToHost(buffer, into: &result)
        return result
    }

    /// Internal buffer reference
    internal var ref: UnsafeMutableRawPointer { buffer }
}

// MARK: - PJRTExecutable (High-level wrapper)

/// High-level PJRT executable wrapper
public final class JupyterPJRTExecutable {
    fileprivate let executable: UnsafeMutableRawPointer
    private let bindings = PJRTBindings.shared

    internal init(executable: UnsafeMutableRawPointer) {
        self.executable = executable
    }

    deinit {
        bindings.destroyExecutable(executable)
    }

    /// Execute with input buffers
    public func execute(inputs: [JupyterPJRTBuffer]) throws -> [UnsafeMutableRawPointer] {
        let inputPtrs = inputs.map { $0.buffer }
        return try bindings.execute(executable: executable, inputs: inputPtrs)
    }

    /// Internal executable reference
    internal var ref: UnsafeMutableRawPointer { executable }
}

// MARK: - Convenience Extensions

extension SwiftIRJupyter {
    /// Create a new MLIR context
    public func createContext() throws -> JupyterMLIRContext {
        try JupyterMLIRContext()
    }

    /// Create a new PJRT client
    public func createClient() throws -> JupyterPJRTClient {
        try JupyterPJRTClient()
    }
}
