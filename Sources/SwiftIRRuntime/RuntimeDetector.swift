// RuntimeDetector.swift - Automatic accelerator detection
// Copyright 2024 SwiftIR Project
//
// Detects available hardware accelerators (CPU, GPU, TPU) and finds PJRT plugins.

import Foundation

/// Information about the detected runtime environment
public struct RuntimeInfo: Sendable {
    /// The best detected accelerator (based on priority: TPU > GPU > CPU)
    public let detectedAccelerator: AcceleratorType

    /// Whether TPU hardware is available
    public let tpuAvailable: Bool

    /// Whether GPU (NVIDIA CUDA) is available
    public let gpuAvailable: Bool

    /// Number of TPU cores if available
    public let tpuCoreCount: Int?

    /// Relevant environment variables
    public let environmentVariables: [String: String]

    /// Detected plugin paths for each accelerator type
    public let detectedPluginPaths: [AcceleratorType: String]

    /// Human-readable summary of the runtime
    public var summary: String {
        var lines: [String] = []
        lines.append("SwiftIR Runtime Info")
        lines.append("====================")
        lines.append("Detected accelerator: \(detectedAccelerator)")
        lines.append("TPU available: \(tpuAvailable)\(tpuCoreCount.map { " (\($0) cores)" } ?? "")")
        lines.append("GPU available: \(gpuAvailable)")
        lines.append("")
        lines.append("Plugin paths:")
        for (acc, path) in detectedPluginPaths.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            lines.append("  \(acc): \(path)")
        }
        if !environmentVariables.isEmpty {
            lines.append("")
            lines.append("Environment:")
            for (key, value) in environmentVariables.sorted(by: { $0.key < $1.key }) {
                lines.append("  \(key)=\(value)")
            }
        }
        return lines.joined(separator: "\n")
    }
}

/// Detects available hardware accelerators and PJRT plugins
///
/// `RuntimeDetector` provides static methods to:
/// - Detect available hardware (TPU, GPU, CPU)
/// - Find PJRT plugin paths
/// - Get detailed runtime information
///
/// ## Detection Priority
/// When using `detect()`, the priority is:
/// 1. Environment override (`PJRT_DEVICE`)
/// 2. TPU (if available)
/// 3. GPU (if available)
/// 4. CPU (always available)
///
/// ## Example
/// ```swift
/// // Auto-detect best accelerator
/// let accelerator = RuntimeDetector.detect()
/// print("Using: \(accelerator)")
///
/// // Get detailed info
/// let info = RuntimeDetector.getRuntimeInfo()
/// print(info.summary)
/// ```
public struct RuntimeDetector {

    // MARK: - Main Detection API

    /// Detect the best available accelerator
    ///
    /// Priority: Environment override > TPU > GPU > CPU
    ///
    /// - Returns: The best available accelerator type
    public static func detect() -> AcceleratorType {
        // Check environment override first
        if let pjrtDevice = ProcessInfo.processInfo.environment["PJRT_DEVICE"]?.uppercased() {
            switch pjrtDevice {
            case "TPU":
                return .tpu
            case "GPU", "CUDA":
                return .gpu
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

    // MARK: - TPU Detection

    /// Check if TPU hardware is available
    ///
    /// Detection methods:
    /// 1. Check for `/dev/accel*` devices (TPU chips)
    /// 2. Check for `libtpu.so` in known locations
    /// 3. Check `TPU_NAME` environment variable
    ///
    /// - Returns: `true` if TPU is available
    public static func isTPUAvailable() -> Bool {
        // Method 1: Check /dev/accel* devices (TPU chips on TPU VMs)
        for i in 0..<8 {
            if FileManager.default.fileExists(atPath: "/dev/accel\(i)") {
                return true
            }
        }

        // Method 2: Check libtpu.so in known locations
        if findPluginPath(for: .tpu) != nil {
            return true
        }

        // Method 3: Check TPU_NAME env (Cloud TPU)
        if ProcessInfo.processInfo.environment["TPU_NAME"] != nil {
            return true
        }

        return false
    }

    /// Count available TPU cores
    ///
    /// - Returns: Number of TPU cores, or nil if no TPU
    public static func countTPUCores() -> Int? {
        guard isTPUAvailable() else { return nil }

        var chipCount = 0
        for i in 0..<8 {
            if FileManager.default.fileExists(atPath: "/dev/accel\(i)") {
                chipCount += 1
            }
        }

        // Each TPU chip has 2 cores
        // TPU v2-8 has 4 chips = 8 cores
        // TPU v3-8 has 4 chips = 8 cores
        if chipCount > 0 {
            return chipCount * 2
        }

        // Default to 8 cores for Cloud TPU (v2-8/v3-8)
        return 8
    }

    // MARK: - GPU Detection

    /// Check if NVIDIA GPU is available
    ///
    /// Detection methods:
    /// 1. Check for `/dev/nvidia*` devices
    /// 2. Check for CUDA libraries
    /// 3. Check `CUDA_VISIBLE_DEVICES` environment
    /// 4. Run `nvidia-smi` as fallback
    ///
    /// - Returns: `true` if GPU is available
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
            // CUDA_VISIBLE_DEVICES="" or "-1" means no GPU
            if !cudaDevices.isEmpty && cudaDevices != "-1" {
                return true
            }
        }

        // Method 4: Try nvidia-smi (fallback)
        return checkNvidiaSmi()
    }

    /// Run nvidia-smi to check GPU availability
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

    // MARK: - Plugin Path Discovery

    /// Find the PJRT plugin path for a given accelerator
    ///
    /// Searches environment variables and common installation locations.
    ///
    /// - Parameter accelerator: The accelerator type to find
    /// - Returns: Path to the plugin, or nil if not found
    public static func findPluginPath(for accelerator: AcceleratorType) -> String? {
        let searchPaths = getSearchPaths(for: accelerator)

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        return nil
    }

    /// Get the list of paths to search for a plugin
    ///
    /// - Parameter accelerator: The accelerator type
    /// - Returns: List of paths to search
    public static func getSearchPaths(for accelerator: AcceleratorType) -> [String] {
        switch accelerator {
        case .tpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["TPU_LIBRARY_PATH"],
                // System locations (Colab TPU runtime)
                "/lib/libtpu.so",
                "/usr/local/lib/libtpu.so",
                // Python package locations
                expandPath("~/.local/lib/python3.10/site-packages/libtpu/libtpu.so"),
                expandPath("~/.local/lib/python3.11/site-packages/libtpu/libtpu.so"),
                expandPath("~/.local/lib/python3.12/site-packages/libtpu/libtpu.so"),
            ].compactMap { $0 }

        case .gpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["PJRT_GPU_PLUGIN_PATH"],
                // System locations
                "/usr/local/lib/pjrt_c_api_gpu_plugin.so",
                "/usr/local/lib/xla_cuda_plugin.so",
                // JAX CUDA plugin locations
                expandPath("~/.local/lib/python3.10/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"),
                expandPath("~/.local/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"),
                expandPath("~/.local/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"),
                // SwiftIR bundled (if ever)
                swiftirPath("lib/pjrt_c_api_gpu_plugin.so"),
                swiftirPath("lib/xla_cuda_plugin.so"),
            ].compactMap { $0 }

        case .cpu:
            return [
                // Environment override
                ProcessInfo.processInfo.environment["PJRT_CPU_PLUGIN_PATH"],
                // SwiftIR bundled location (primary)
                swiftirPath("lib/pjrt_c_api_cpu_plugin.so"),
                // System locations
                "/usr/local/lib/pjrt_c_api_cpu_plugin.so",
                "/opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so",
                // Local directory
                "./lib/pjrt_c_api_cpu_plugin.so",
                // JAX CPU plugin locations
                expandPath("~/.local/lib/python3.10/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so"),
                expandPath("~/.local/lib/python3.11/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so"),
                expandPath("~/.local/lib/python3.12/site-packages/jaxlib/pjrt_c_api_cpu_plugin.so"),
            ].compactMap { $0 }
        }
    }

    // MARK: - Runtime Info

    /// Get detailed information about the runtime environment
    ///
    /// - Returns: RuntimeInfo with detected accelerators and paths
    public static func getRuntimeInfo() -> RuntimeInfo {
        // Collect environment variables
        let envKeys = [
            "PJRT_DEVICE",
            "TPU_NAME",
            "TPU_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "SWIFTIR_HOME",
            "PJRT_CPU_PLUGIN_PATH",
            "PJRT_GPU_PLUGIN_PATH"
        ]

        var envDict: [String: String] = [:]
        for key in envKeys {
            if let value = ProcessInfo.processInfo.environment[key] {
                envDict[key] = value
            }
        }

        // Find plugin paths
        var pluginPaths: [AcceleratorType: String] = [:]
        for acc in AcceleratorType.allCases {
            if let path = findPluginPath(for: acc) {
                pluginPaths[acc] = path
            }
        }

        return RuntimeInfo(
            detectedAccelerator: detect(),
            tpuAvailable: isTPUAvailable(),
            gpuAvailable: isGPUAvailable(),
            tpuCoreCount: countTPUCores(),
            environmentVariables: envDict,
            detectedPluginPaths: pluginPaths
        )
    }

    // MARK: - Path Helpers

    /// Expand ~ in paths to home directory
    private static func expandPath(_ path: String) -> String {
        if path.hasPrefix("~") {
            return NSString(string: path).expandingTildeInPath
        }
        return path
    }

    /// Get path relative to SWIFTIR_HOME
    private static func swiftirPath(_ subpath: String) -> String? {
        guard let home = ProcessInfo.processInfo.environment["SWIFTIR_HOME"] else {
            return nil
        }
        return "\(home)/\(subpath)"
    }
}

// MARK: - Convenience Extensions

extension RuntimeDetector {
    /// Print runtime information to stdout
    public static func printInfo() {
        print(getRuntimeInfo().summary)
    }
}
