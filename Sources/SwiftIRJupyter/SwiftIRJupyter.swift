// SwiftIRJupyter - High-level API for Jupyter/Colab
// Pure Swift - works in LLDB REPL without C++ interop

import Foundation

// MARK: - Backend Types

/// Accelerator backend types for SwiftIRJupyter
public enum JupyterBackend: String, CaseIterable, Sendable {
    case cpu = "cpu"
    case gpu = "gpu"
    case tpu = "tpu"

    /// Human-readable description
    public var description: String {
        switch self {
        case .cpu: return "CPU"
        case .gpu: return "GPU (CUDA)"
        case .tpu: return "TPU"
        }
    }
}

// MARK: - SwiftIRJupyter Main API

/// Main entry point for SwiftIR in Jupyter/Colab notebooks
public final class SwiftIRJupyter: @unchecked Sendable {

    /// Shared instance
    nonisolated(unsafe) public static let shared = SwiftIRJupyter()

    /// Whether SwiftIR is initialized
    public private(set) var isInitialized = false

    /// Current backend (set after initialization)
    public private(set) var currentBackend: JupyterBackend?

    /// SDK path
    public var sdkPath: String {
        LibraryLoader.shared.sdkPath
    }

    private init() {}

    // MARK: - Backend Detection

    /// Check if TPU hardware is available
    ///
    /// Detection methods:
    /// 1. Check for `/dev/accel*` devices (TPU chips on TPU VMs)
    /// 2. Check Colab TPU environment variables (COLAB_TPU_1VM, TPU_ACCELERATOR_TYPE)
    /// 3. Check for `libtpu.so` in known locations
    public static func isTPUAvailable() -> Bool {
        // Method 1: Check /dev/accel* devices (TPU chips on TPU VMs)
        for i in 0..<8 {
            if FileManager.default.fileExists(atPath: "/dev/accel\(i)") {
                return true
            }
        }

        // Method 2: Check Colab TPU environment
        if ProcessInfo.processInfo.environment["COLAB_TPU_1VM"] != nil {
            if findPluginPath(for: .tpu) != nil {
                return true
            }
        }

        // Check TPU_ACCELERATOR_TYPE (e.g., "v5e-1")
        if ProcessInfo.processInfo.environment["TPU_ACCELERATOR_TYPE"] != nil {
            if findPluginPath(for: .tpu) != nil {
                return true
            }
        }

        // Method 3: Check TPU_NAME env (Cloud TPU)
        if let tpuName = ProcessInfo.processInfo.environment["TPU_NAME"], !tpuName.isEmpty {
            return true
        }

        return false
    }

    /// Check if NVIDIA GPU is available
    ///
    /// Detection methods:
    /// 1. Check for `/dev/nvidia*` devices
    /// 2. Check for CUDA libraries
    /// 3. Check `CUDA_VISIBLE_DEVICES` environment
    public static func isGPUAvailable() -> Bool {
        // Method 1: Check /dev/nvidia* devices
        let nvidiaDevices = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
        for device in nvidiaDevices {
            if FileManager.default.fileExists(atPath: device) {
                return true
            }
        }

        // Method 2: Check CUDA libraries
        let cudaLibs = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        ]
        for lib in cudaLibs {
            if FileManager.default.fileExists(atPath: lib) {
                return true
            }
        }

        // Method 3: Check CUDA_VISIBLE_DEVICES
        if let cudaDevices = ProcessInfo.processInfo.environment["CUDA_VISIBLE_DEVICES"] {
            if !cudaDevices.isEmpty && cudaDevices != "-1" {
                return true
            }
        }

        return false
    }

    /// Detect the best available backend
    ///
    /// Priority: Environment override > TPU > GPU > CPU
    public static func detectBestBackend() -> JupyterBackend {
        // Check environment override first
        if let pjrtDevice = ProcessInfo.processInfo.environment["PJRT_DEVICE"]?.uppercased() {
            switch pjrtDevice {
            case "TPU":
                if isTPUAvailable() { return .tpu }
            case "GPU", "CUDA":
                if isGPUAvailable() { return .gpu }
            case "CPU":
                return .cpu
            default:
                break
            }
        }

        // Auto-detect based on availability
        if isTPUAvailable() { return .tpu }
        if isGPUAvailable() { return .gpu }
        return .cpu
    }

    /// Find the PJRT plugin path for a given backend
    public static func findPluginPath(for backend: JupyterBackend) -> String? {
        let searchPaths = getSearchPaths(for: backend)

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        return nil
    }

    /// Get the list of paths to search for a plugin
    private static func getSearchPaths(for backend: JupyterBackend) -> [String] {
        let sdkPath = LibraryLoader.shared.sdkPath

        switch backend {
        case .tpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["TPU_LIBRARY_PATH"],
                // System locations (Colab TPU runtime)
                "/lib/libtpu.so",
                "/usr/local/lib/libtpu.so",
                // Colab TPU v5e location (installed via pip)
                "/usr/local/lib/python3.10/dist-packages/libtpu/libtpu.so",
                "/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so",
                "/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so",
                // User Python package locations
                NSString(string: "~/.local/lib/python3.10/site-packages/libtpu/libtpu.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.11/site-packages/libtpu/libtpu.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.12/site-packages/libtpu/libtpu.so").expandingTildeInPath,
            ].compactMap { $0 }

        case .gpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["PJRT_GPU_PLUGIN_PATH"],
                // SDK location
                "\(sdkPath)/lib/pjrt_c_api_gpu_plugin.so",
                "\(sdkPath)/lib/xla_cuda_plugin.so",
                // System locations
                "/usr/local/lib/pjrt_c_api_gpu_plugin.so",
                "/usr/local/lib/xla_cuda_plugin.so",
                "/opt/swiftir-deps/lib/pjrt_c_api_gpu_plugin.so",
                "/opt/swiftir-deps/lib/xla_cuda_plugin.so",
                // JAX CUDA plugin locations
                NSString(string: "~/.local/lib/python3.10/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so").expandingTildeInPath,
            ].compactMap { $0 }

        case .cpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["PJRT_CPU_PLUGIN_PATH"],
                // SDK location (primary)
                "\(sdkPath)/lib/pjrt_c_api_cpu_plugin.so",
                // System locations
                "/usr/local/lib/pjrt_c_api_cpu_plugin.so",
                "/opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so",
                // Local directory
                "./lib/pjrt_c_api_cpu_plugin.so",
                // JAX CPU plugin locations
                NSString(string: "~/.local/lib/python3.10/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.11/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so").expandingTildeInPath,
                NSString(string: "~/.local/lib/python3.12/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so").expandingTildeInPath,
            ].compactMap { $0 }
        }
    }

    // MARK: - Initialization

    /// Initialize SwiftIR with auto-detected backend
    ///
    /// Automatically detects the best available backend (TPU > GPU > CPU)
    /// and initializes SwiftIR with it.
    public func initialize() throws {
        let backend = Self.detectBestBackend()
        try initialize(backend: backend)
    }

    /// Initialize SwiftIR for Jupyter usage with specific backend
    /// - Parameter backend: The backend to use (cpu, gpu, or tpu)
    public func initialize(backend: JupyterBackend) throws {
        guard !isInitialized else { return }

        // Find plugin path
        guard let pluginPath = Self.findPluginPath(for: backend) else {
            throw SwiftIRJupyterError.initializationFailed(
                message: "Could not find PJRT plugin for \(backend.description). " +
                         "Searched paths: \(Self.getSearchPaths(for: backend).joined(separator: ", "))"
            )
        }

        try initialize(pluginPath: pluginPath, backend: backend)
    }

    /// Initialize SwiftIR for Jupyter usage with explicit plugin path
    /// - Parameters:
    ///   - pluginPath: Path to PJRT plugin
    ///   - backend: Optional backend type (for logging, auto-detected from path if nil)
    public func initialize(pluginPath: String, backend: JupyterBackend? = nil) throws {
        guard !isInitialized else { return }

        // Determine backend from path if not specified
        let detectedBackend: JupyterBackend
        if let backend = backend {
            detectedBackend = backend
        } else if pluginPath.contains("libtpu") {
            detectedBackend = .tpu
        } else if pluginPath.contains("gpu") || pluginPath.contains("cuda") {
            detectedBackend = .gpu
        } else {
            detectedBackend = .cpu
        }

        print("Initializing SwiftIRJupyter...")
        print("   SDK path: \(sdkPath)")
        print("   Backend: \(detectedBackend.description)")

        // Load MLIR bindings
        print("   Loading MLIR bindings...")
        try MLIRBindings.shared.load()
        print("   MLIR bindings loaded")

        // Load PJRT bindings
        print("   Loading PJRT bindings...")
        try PJRTBindings.shared.load()
        print("   PJRT bindings loaded")

        // Load PJRT plugin
        print("   Loading PJRT plugin: \(pluginPath)")
        try PJRTBindings.shared.loadPlugin(path: pluginPath)
        print("   PJRT plugin loaded")

        currentBackend = detectedBackend
        isInitialized = true
        print("SwiftIRJupyter initialized successfully with \(detectedBackend.description)!")
    }

    // MARK: - Convenience Initializers

    /// Initialize with CPU backend
    public func initializeCPU() throws {
        try initialize(backend: .cpu)
    }

    /// Initialize with GPU (CUDA) backend
    public func initializeGPU() throws {
        try initialize(backend: .gpu)
    }

    /// Initialize with TPU backend
    public func initializeTPU() throws {
        try initialize(backend: .tpu)
    }

    /// Print system information
    public func printInfo() {
        let initStr = isInitialized ? "Yes" : "No"
        let mlirStr = MLIRBindings.shared.isLoaded ? "Yes" : "No"
        let pjrtStr = PJRTBindings.shared.isLoaded ? "Yes" : "No"
        let pluginStr = PJRTBindings.shared.isPluginLoaded ? "Yes" : "No"
        let backendStr = currentBackend?.description ?? "Not initialized"

        print("SwiftIR Jupyter Info")
        print("====================")
        print("SDK Path:      \(sdkPath)")
        print("Initialized:   \(initStr)")
        print("Backend:       \(backendStr)")
        print("MLIR Loaded:   \(mlirStr)")
        print("PJRT Loaded:   \(pjrtStr)")
        print("Plugin Loaded: \(pluginStr)")
        print("")
        print("Hardware Detection:")
        print("  TPU Available: \(Self.isTPUAvailable() ? "Yes" : "No")")
        print("  GPU Available: \(Self.isGPUAvailable() ? "Yes" : "No")")
        print("  Best Backend:  \(Self.detectBestBackend().description)")
        print("")
        print("Plugin Paths:")
        for backend in JupyterBackend.allCases {
            if let path = Self.findPluginPath(for: backend) {
                print("  \(backend.description): \(path)")
            } else {
                print("  \(backend.description): Not found")
            }
        }
    }

    /// Print hardware detection summary (static, can be called before initialization)
    public static func printHardwareInfo() {
        print("SwiftIR Hardware Detection")
        print("==========================")
        print("TPU Available: \(isTPUAvailable() ? "Yes" : "No")")
        print("GPU Available: \(isGPUAvailable() ? "Yes" : "No")")
        print("Best Backend:  \(detectBestBackend().description)")
        print("")
        print("Plugin Paths:")
        for backend in JupyterBackend.allCases {
            if let path = findPluginPath(for: backend) {
                print("  \(backend.description): \(path)")
            } else {
                print("  \(backend.description): Not found")
            }
        }
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
