# CLAUDE.md - SwiftIR Runtime Detection Implementation Guide

## Project Context

SwiftIR is a Swift framework for building and executing MLIR/StableHLO code via PJRT. 
This document specifies the implementation of automatic CPU/GPU/TPU detection and unified client creation,
enabling seamless execution on Google Colab and Cloud TPU VMs.

**Repository:** https://github.com/pedronahum/SwiftIR
**Related:** https://github.com/pedronahum/swift-jupyter (Swift kernel for Colab)

## Goals

1. Auto-detect available accelerators (CPU, GPU, TPU)
2. Provide unified `PJRTClient.create()` API (similar to JAX)
3. Support Google Colab CPU/GPU/TPU runtimes
4. Support Cloud TPU VMs
5. Create prebuilt binaries for easy installation

---

## Implementation Tasks

### Phase 1: Create SwiftIRRuntime Module

Create `Sources/SwiftIRRuntime/` with these files:

#### AcceleratorType.swift
```swift
public enum AcceleratorType: String, CustomStringConvertible, Equatable, Hashable, Codable {
    case cpu = "CPU"
    case gpu = "GPU"
    case tpu = "TPU"
    
    public var description: String { rawValue }
    
    public var pluginName: String {
        switch self {
        case .cpu: return "pjrt_c_api_cpu_plugin.so"
        case .gpu: return "xla_cuda_plugin.so"
        case .tpu: return "libtpu.so"
        }
    }
}
```

#### Errors.swift
```swift
public enum SwiftIRError: Error, LocalizedError {
    case pluginNotFound(accelerator: AcceleratorType, searchedPaths: [String])
    case pluginLoadFailed(path: String, reason: String)
    case symbolNotFound(symbol: String, path: String)
    case clientCreationFailed(accelerator: AcceleratorType, reason: String)
    case acceleratorNotAvailable(AcceleratorType)
    
    public var errorDescription: String? {
        switch self {
        case .pluginNotFound(let acc, let paths):
            return "No \(acc) plugin found. Searched: \(paths.joined(separator: ", "))"
        case .pluginLoadFailed(let path, let reason):
            return "Failed to load plugin at \(path): \(reason)"
        case .symbolNotFound(let symbol, let path):
            return "Symbol '\(symbol)' not found in \(path)"
        case .clientCreationFailed(let acc, let reason):
            return "Failed to create \(acc) client: \(reason)"
        case .acceleratorNotAvailable(let acc):
            return "\(acc) is not available on this system"
        }
    }
}
```

---

### Phase 2: Implement RuntimeDetector

#### RuntimeDetector.swift

```swift
import Foundation

public struct RuntimeInfo {
    public let detectedAccelerator: AcceleratorType
    public let tpuAvailable: Bool
    public let gpuAvailable: Bool
    public let tpuCoreCount: Int?
    public let environmentVariables: [String: String]
    public let detectedPluginPaths: [AcceleratorType: String]
}

public struct RuntimeDetector {
    
    /// Detect best available accelerator
    /// Priority: PJRT_DEVICE env > TPU > GPU > CPU
    public static func detect() -> AcceleratorType {
        // Check environment override
        if let pjrtDevice = ProcessInfo.processInfo.environment["PJRT_DEVICE"]?.uppercased() {
            switch pjrtDevice {
            case "TPU": return .tpu
            case "GPU", "CUDA": return .gpu
            case "CPU": return .cpu
            default: break
            }
        }
        
        if isTPUAvailable() { return .tpu }
        if isGPUAvailable() { return .gpu }
        return .cpu
    }
    
    /// Check if TPU is available
    public static func isTPUAvailable() -> Bool {
        // Method 1: Check /dev/accel* devices (TPU chips)
        for i in 0..<8 {
            if FileManager.default.fileExists(atPath: "/dev/accel\(i)") {
                return true
            }
        }
        
        // Method 2: Check libtpu.so locations
        if findPluginPath(for: .tpu) != nil {
            return true
        }
        
        // Method 3: Check TPU_NAME env (Cloud TPU)
        if ProcessInfo.processInfo.environment["TPU_NAME"] != nil {
            return true
        }
        
        return false
    }
    
    /// Check if GPU (NVIDIA CUDA) is available
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
        if let cudaDevices = ProcessInfo.processInfo.environment["CUDA_VISIBLE_DEVICES"],
           !cudaDevices.isEmpty, cudaDevices != "-1" {
            return true
        }
        
        // Method 4: Try nvidia-smi (fallback)
        return checkNvidiaSmi()
    }
    
    private static func checkNvidiaSmi() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/nvidia-smi")
        process.arguments = ["-L"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
    
    /// Find plugin path for given accelerator
    public static func findPluginPath(for accelerator: AcceleratorType) -> String? {
        let searchPaths: [String?]
        
        switch accelerator {
        case .tpu:
            searchPaths = [
                ProcessInfo.processInfo.environment["TPU_LIBRARY_PATH"],
                "/lib/libtpu.so",
                "/usr/local/lib/libtpu.so",
                expandPath("~/.local/lib/python3.10/site-packages/libtpu/libtpu.so"),
                expandPath("~/.local/lib/python3.11/site-packages/libtpu/libtpu.so")
            ]
        case .gpu:
            searchPaths = [
                ProcessInfo.processInfo.environment["PJRT_GPU_PLUGIN_PATH"],
                "/usr/local/lib/pjrt_c_api_gpu_plugin.so",
                expandPath("~/.local/lib/python3.10/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"),
                expandPath("~/.local/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"),
                swiftirPath("lib/pjrt_c_api_gpu_plugin.so")
            ]
        case .cpu:
            searchPaths = [
                ProcessInfo.processInfo.environment["PJRT_CPU_PLUGIN_PATH"],
                "/usr/local/lib/pjrt_c_api_cpu_plugin.so",
                swiftirPath("lib/pjrt_c_api_cpu_plugin.so"),
                "./lib/pjrt_c_api_cpu_plugin.so"
            ]
        }
        
        for path in searchPaths.compactMap({ $0 }) {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return nil
    }
    
    /// Get detailed runtime information
    public static func getRuntimeInfo() -> RuntimeInfo {
        let envVars = ["PJRT_DEVICE", "TPU_NAME", "TPU_LIBRARY_PATH", 
                       "CUDA_VISIBLE_DEVICES", "SWIFTIR_HOME"]
        var envDict: [String: String] = [:]
        for key in envVars {
            if let value = ProcessInfo.processInfo.environment[key] {
                envDict[key] = value
            }
        }
        
        var pluginPaths: [AcceleratorType: String] = [:]
        for acc in [AcceleratorType.cpu, .gpu, .tpu] {
            if let path = findPluginPath(for: acc) {
                pluginPaths[acc] = path
            }
        }
        
        let tpuCores: Int? = isTPUAvailable() ? countTPUCores() : nil
        
        return RuntimeInfo(
            detectedAccelerator: detect(),
            tpuAvailable: isTPUAvailable(),
            gpuAvailable: isGPUAvailable(),
            tpuCoreCount: tpuCores,
            environmentVariables: envDict,
            detectedPluginPaths: pluginPaths
        )
    }
    
    private static func expandPath(_ path: String) -> String {
        if path.hasPrefix("~") {
            return NSString(string: path).expandingTildeInPath
        }
        return path
    }
    
    private static func swiftirPath(_ subpath: String) -> String? {
        guard let home = ProcessInfo.processInfo.environment["SWIFTIR_HOME"] else {
            return nil
        }
        return "\(home)/\(subpath)"
    }
    
    private static func countTPUCores() -> Int {
        var count = 0
        for i in 0..<8 {
            if FileManager.default.fileExists(atPath: "/dev/accel\(i)") {
                count += 1
            }
        }
        return max(count * 2, 8) // Each chip has 2 cores, minimum 8 on v2-8
    }
}
```

---

### Phase 3: Implement PJRTPlugin

#### PJRTPlugin.swift

```swift
import Foundation

public class PJRTPlugin {
    public let handle: UnsafeMutableRawPointer
    public let api: UnsafePointer<PJRT_Api>
    public let path: String
    public let acceleratorType: AcceleratorType
    
    public init(path: String, acceleratorType: AcceleratorType) throws {
        self.path = path
        self.acceleratorType = acceleratorType
        
        // Load the shared library
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let error = String(cString: dlerror())
            throw SwiftIRError.pluginLoadFailed(path: path, reason: error)
        }
        self.handle = handle
        
        // Get GetPjrtApi symbol
        guard let symbol = dlsym(handle, "GetPjrtApi") else {
            dlclose(handle)
            throw SwiftIRError.symbolNotFound(symbol: "GetPjrtApi", path: path)
        }
        
        // Call GetPjrtApi()
        typealias GetPjrtApiFn = @convention(c) () -> UnsafePointer<PJRT_Api>?
        let getPjrtApi = unsafeBitCast(symbol, to: GetPjrtApiFn.self)
        
        guard let api = getPjrtApi() else {
            dlclose(handle)
            throw SwiftIRError.pluginLoadFailed(path: path, reason: "GetPjrtApi returned null")
        }
        self.api = api
    }
    
    deinit {
        dlclose(handle)
    }
    
    /// Load plugin for given accelerator type
    public static func load(for accelerator: AcceleratorType) throws -> PJRTPlugin {
        guard let path = RuntimeDetector.findPluginPath(for: accelerator) else {
            let searchPaths = getSearchPaths(for: accelerator)
            throw SwiftIRError.pluginNotFound(accelerator: accelerator, searchedPaths: searchPaths)
        }
        return try PJRTPlugin(path: path, acceleratorType: accelerator)
    }
    
    private static func getSearchPaths(for accelerator: AcceleratorType) -> [String] {
        // Return the paths that would be searched (for error messages)
        switch accelerator {
        case .tpu:
            return ["/lib/libtpu.so", "/usr/local/lib/libtpu.so", "$TPU_LIBRARY_PATH"]
        case .gpu:
            return ["/usr/local/lib/pjrt_c_api_gpu_plugin.so", "$PJRT_GPU_PLUGIN_PATH"]
        case .cpu:
            return ["/usr/local/lib/pjrt_c_api_cpu_plugin.so", "$SWIFTIR_HOME/lib/"]
        }
    }
}
```

---

### Phase 4: Unified PJRTClient API

#### PJRTClient+Unified.swift (Extension to existing PJRTClient)

```swift
public extension PJRTClient {
    
    /// The accelerator type this client is using
    private(set) var acceleratorType: AcceleratorType
    
    /// Create PJRT client with automatic device detection
    static func create(_ accelerator: AcceleratorType? = nil) throws -> PJRTClient {
        let targetAccelerator = accelerator ?? RuntimeDetector.detect()
        
        print("SwiftIR: Using \(targetAccelerator) backend")
        
        switch targetAccelerator {
        case .tpu:
            guard RuntimeDetector.isTPUAvailable() else {
                throw SwiftIRError.acceleratorNotAvailable(.tpu)
            }
            return try createTPU()
        case .gpu:
            guard RuntimeDetector.isGPUAvailable() else {
                throw SwiftIRError.acceleratorNotAvailable(.gpu)
            }
            return try createGPU()
        case .cpu:
            return try createCPU()
        }
    }
    
    /// Create TPU client
    static func createTPU() throws -> PJRTClient {
        let plugin = try PJRTPlugin.load(for: .tpu)
        return try createClientFromPlugin(plugin)
    }
    
    /// Create GPU client
    static func createGPU() throws -> PJRTClient {
        let plugin = try PJRTPlugin.load(for: .gpu)
        return try createClientFromPlugin(plugin)
    }
    
    /// Create CPU client (ensure this follows same pattern)
    static func createCPU() throws -> PJRTClient {
        let plugin = try PJRTPlugin.load(for: .cpu)
        return try createClientFromPlugin(plugin)
    }
    
    /// Internal: Create client from loaded plugin
    private static func createClientFromPlugin(_ plugin: PJRTPlugin) throws -> PJRTClient {
        // Prepare PJRT_Client_Create_Args
        var createArgs = PJRT_Client_Create_Args()
        createArgs.struct_size = MemoryLayout<PJRT_Client_Create_Args>.size
        
        // Call PJRT_Client_Create through the API
        guard let createFn = plugin.api.pointee.PJRT_Client_Create else {
            throw SwiftIRError.clientCreationFailed(
                accelerator: plugin.acceleratorType,
                reason: "PJRT_Client_Create function not found in plugin"
            )
        }
        
        let error = createFn(&createArgs)
        if let error = error {
            // Extract error message from PJRT_Error
            let message = extractPJRTErrorMessage(error, api: plugin.api)
            throw SwiftIRError.clientCreationFailed(
                accelerator: plugin.acceleratorType,
                reason: message
            )
        }
        
        guard let clientHandle = createArgs.client else {
            throw SwiftIRError.clientCreationFailed(
                accelerator: plugin.acceleratorType,
                reason: "Client handle is null"
            )
        }
        
        // Create and return PJRTClient
        // NOTE: Adapt this to your existing PJRTClient initializer
        return PJRTClient(
            handle: clientHandle,
            api: plugin.api,
            plugin: plugin,
            acceleratorType: plugin.acceleratorType
        )
    }
    
    private static func extractPJRTErrorMessage(_ error: UnsafeMutablePointer<PJRT_Error>, api: UnsafePointer<PJRT_Api>) -> String {
        // Use PJRT_Error_Message to get error string
        if let getMessageFn = api.pointee.PJRT_Error_Message {
            var args = PJRT_Error_Message_Args()
            args.struct_size = MemoryLayout<PJRT_Error_Message_Args>.size
            args.error = error
            getMessageFn(&args)
            if let message = args.message {
                return String(cString: message)
            }
        }
        return "Unknown error"
    }
}
```

---

### Phase 5: Update Package.swift

Add the new module:

```swift
// In Package.swift targets array:
.target(
    name: "SwiftIRRuntime",
    dependencies: ["PJRTCWrappers"],
    path: "Sources/SwiftIRRuntime"
),

// Update main SwiftIR target to depend on SwiftIRRuntime
.target(
    name: "SwiftIR",
    dependencies: [
        "SwiftIRCore",
        "SwiftIRRuntime",  // Add this
        // ... other dependencies
    ]
),

// Add test target
.testTarget(
    name: "SwiftIRRuntimeTests",
    dependencies: ["SwiftIRRuntime"]
),
```

---

## Tests

### Unit Tests (Tests/SwiftIRRuntimeTests/)

```swift
// AcceleratorTypeTests.swift
import XCTest
@testable import SwiftIRRuntime

final class AcceleratorTypeTests: XCTestCase {
    func testRawValues() {
        XCTAssertEqual(AcceleratorType.cpu.rawValue, "CPU")
        XCTAssertEqual(AcceleratorType.gpu.rawValue, "GPU")
        XCTAssertEqual(AcceleratorType.tpu.rawValue, "TPU")
    }
    
    func testPluginNames() {
        XCTAssertEqual(AcceleratorType.cpu.pluginName, "pjrt_c_api_cpu_plugin.so")
        XCTAssertEqual(AcceleratorType.gpu.pluginName, "xla_cuda_plugin.so")
        XCTAssertEqual(AcceleratorType.tpu.pluginName, "libtpu.so")
    }
}

// RuntimeDetectorTests.swift
final class RuntimeDetectorTests: XCTestCase {
    func testDetectReturnsValidType() {
        let result = RuntimeDetector.detect()
        XCTAssertTrue([.cpu, .gpu, .tpu].contains(result))
    }
    
    func testGetRuntimeInfoNotNil() {
        let info = RuntimeDetector.getRuntimeInfo()
        XCTAssertNotNil(info.detectedAccelerator)
    }
    
    func testEnvironmentOverride() {
        // This test requires setting env var before process starts
        // or using a mock - document this limitation
    }
}
```

---

## Success Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| F1 | `detect()` returns `.cpu` on CPU-only | Run on Colab CPU runtime |
| F2 | `detect()` returns `.gpu` on GPU | Run on Colab GPU runtime |
| F3 | `detect()` returns `.tpu` on TPU | Run on Colab TPU v2 runtime |
| F4 | `PJRT_DEVICE` env overrides detection | Set env, verify override works |
| F5 | `PJRTClient.create()` auto-detects | Call without args, verify correct client |
| F6 | `create(.tpu)` fails gracefully on CPU | Verify throws `acceleratorNotAvailable` |
| F7 | StableHLO compiles on all backends | Compile simple function |
| F8 | Execution produces correct results | Run add([1,2,3], [4,5,6]) = [5,7,9] |

---

## Build System for Prebuilt Binaries

### GitHub Actions Workflow (.github/workflows/build-release.yml)

See the full workflow in the JSON spec file. Key steps:

1. Build on Ubuntu 22.04 with Swift 6.0
2. Build/download MLIR dependencies from StableHLO
3. Build SwiftIR in release mode
4. Download CPU PJRT plugin from JAX
5. Package as `SwiftIR-{version}-linux-x86_64.tar.gz`
6. Create GitHub release with artifacts

### Colab Installation

```python
# Cell 1: Download SwiftIR
!curl -L https://github.com/pedronahum/SwiftIR/releases/latest/download/SwiftIR-linux-x86_64.tar.gz | tar xz
%env LD_LIBRARY_PATH=/content/SwiftIR/lib:$LD_LIBRARY_PATH
%env SWIFTIR_HOME=/content/SwiftIR

# Cell 2: (In Swift kernel)
import SwiftIR
let client = try PJRTClient.create()
print("Ready on \(client.acceleratorType)")
```

---

## Important Notes

1. **libtpu.so is proprietary** - Cannot bundle, but it's pre-installed on TPU runtimes
2. **GPU plugin is large** - Users install via `pip install jax[cuda12]`
3. **CPU plugin CAN be bundled** - It's open source
4. **Colab TPU v2** has local TPU access (not remote like old TPU Node)
5. **Check PJRT API version compatibility** via `api->pjrt_api_version`

---

## References

- PJRT Integration Guide: https://openxla.org/xla/pjrt/pjrt_integration
- PJRT C API Header: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
- Colab TPU VM Migration: https://github.com/googlecolab/colabtools/issues/4481
- TPU Runtimes: https://cloud.google.com/tpu/docs/runtimes
