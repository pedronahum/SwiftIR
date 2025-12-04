// ProfilerDemo.swift - Demo of SwiftIR PJRT profiler for TensorBoard
//
// This demonstrates profiling actual XLA/PJRT computations for TensorBoard visualization.
// The PJRT profiler extension captures XLA compilation, HLO modules, buffer transfers,
// and kernel execution with full XLA internal metrics.
//
// With the new TraceMe API, Swift traces now appear in the SAME profile as XLA internal
// traces, making it easy to correlate user code with XLA operations in TensorBoard.
//
// Run with:
//   SWIFTIR_DEPS=/opt/swiftir-deps LD_LIBRARY_PATH=/opt/swiftir-deps/lib swift run ProfilerDemo

import SwiftIRProfiler
import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders
import SwiftIRStableHLO
import SwiftIRXLA
import Foundation

@main
struct ProfilerDemo {
    static func main() throws {
        print("SwiftIR PJRT Profiler Demo for TensorBoard")
        print("==========================================")
        print("Profiling real XLA/PJRT computations with PJRT extension\n")

        // ============================================================
        // Initialize PJRT first - this loads the plugin with profiler
        // ============================================================

        print("Initializing PJRT CPU Client...")
        let cpuClient = try PJRTClient(backend: .cpu)
        print("✅ PJRT CPU Client initialized")
        print("   Platform: \(cpuClient.platformName)")

        guard let device = cpuClient.addressableDevices.first else {
            print("❌ No devices available")
            return
        }

        // ============================================================
        // Now check if PJRT profiler extension is available
        // ============================================================

        print("\nChecking PJRT profiler extension...")
        let hasPJRTProfiler = PJRTProfiler.isAvailable
        let hasTraceMeApi = PJRTProfiler.hasTraceMeApi
        print("PJRT Profiler Extension: \(hasPJRTProfiler ? "✅ Available" : "❌ Not available")")
        print("PJRT TraceMe API: \(hasTraceMeApi ? "✅ Available (Swift traces in same profile)" : "❌ Not available (fallback to separate traces)")")

        // Create profiler - use PJRT profiler if available, otherwise fall back
        var pjrtProfiler: PJRTProfiler? = nil
        var legacySession: ProfilerSession? = nil

        if hasPJRTProfiler {
            print("Using PJRT profiler extension (captures XLA internals)")
            pjrtProfiler = try PJRTProfiler.create()
            try pjrtProfiler!.start()
        } else {
            print("Falling back to legacy TSL profiler")
            try Profiler.initialize()
            legacySession = try ProfilerSession()
            try legacySession!.start()
        }
        print("Profiler started\n")

        // ============================================================
        // Run multiple profiled XLA computations with step markers
        // ============================================================

        // Use pjrtTrainStep and pjrtTraced which automatically use PJRT TraceMe
        // when available, ensuring Swift traces appear in the same profile as
        // XLA internal traces (compilation, HLO, execution).

        // Warmup run (step 0)
        try pjrtTrainStep(0) {
            try pjrtTraced("warmup") {
                try runMatMul(client: cpuClient, device: device, size: 64, name: "warmup")
            }
        }

        // Training steps with step markers for TensorBoard Overview Page
        // Each pjrtTrainStep() creates a marker that TensorBoard recognizes
        for stepNum in 1...5 {
            try pjrtTrainStep(stepNum) {
                let size = 128 * stepNum  // Increasing sizes: 128, 256, 384, 512, 640
                try pjrtTraced("train_step_\(stepNum)") {
                    try runMatMul(client: cpuClient, device: device, size: min(size, 512), name: "step_\(stepNum)")
                }
            }
        }

        // ============================================================
        // Collect and export profiling data
        // ============================================================

        print("\n" + String(repeating: "=", count: 50))
        print("Collecting profiler data...")

        let data: Data
        if let profiler = pjrtProfiler {
            try profiler.stop()
            data = try profiler.collectData()
        } else if let session = legacySession {
            data = try session.collectData()
        } else {
            print("❌ No profiler active")
            return
        }

        print("Collected \(data.count) bytes of XSpace data")

        // Export to TensorBoard-compatible format
        let logDir = "/tmp/swiftir_profiler_test"
        let pluginDir = "\(logDir)/plugins/profile/run_1"

        try FileManager.default.createDirectory(
            atPath: pluginDir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        // TensorBoard expects: <host>.xplane.pb (without timestamp for simpler host detection)
        let hostname = ProcessInfo.processInfo.hostName
        let filepath = "\(pluginDir)/\(hostname).xplane.pb"

        if pjrtProfiler != nil {
            try PJRTProfiler.exportToFile(data, filepath: filepath)
        } else {
            try ProfilerBindings.shared.exportToFile(data, filepath: filepath)
        }
        print("Exported XSpace to: \(filepath)")

        print("\n" + String(repeating: "=", count: 50))
        print("View in TensorBoard:")
        print("  tensorboard --logdir=\(logDir)")
        print("  Open http://localhost:6006 -> Profile tab")
        print(String(repeating: "=", count: 50))
    }

    /// Run a matrix multiplication of given size
    static func runMatMul(client: PJRTClient, device: PJRTDevice, size: Int, name: String) throws {
        print("  Running \(name) matmul [\(size)x\(size)]...")

        // Create input matrices
        var matrixA = [Float](repeating: 0.0, count: size * size)
        var matrixB = [Float](repeating: 0.0, count: size * size)
        for i in 0..<(size * size) {
            matrixA[i] = Float.random(in: -1.0...1.0)
            matrixB[i] = Float.random(in: -1.0...1.0)
        }

        // Create PJRT buffers - using pjrtTraced for unified profiling
        let bufferA = try pjrtTraced("create_buffer_A") {
            try matrixA.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [size, size],
                    elementType: .f32,
                    device: device
                )
            }
        }

        let bufferB = try pjrtTraced("create_buffer_B") {
            try matrixB.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [size, size],
                    elementType: .f32,
                    device: device
                )
            }
        }

        // Build StableHLO program
        let tensorType = TensorType(shape: [size, size])

        let module = StableHLOModule(name: "matmul_\(size)") {
            StableHLOFunction(
                name: "main",
                parameters: [
                    Parameter(name: "arg0", type: tensorType),
                    Parameter(name: "arg1", type: tensorType)
                ],
                returnType: tensorType
            ) {
                Return(DotGeneral(
                    "arg0", "arg1",
                    lhsType: tensorType,
                    rhsType: tensorType,
                    resultType: tensorType,
                    contractingDims: (1, 0)
                ))
            }
        }

        let mlirProgram = module.build()

        // Compile - this is captured by the profiler
        // Using pjrtTraced ensures this trace appears alongside XLA compilation traces
        let executable = try pjrtTraced("compile") {
            try client.compile(
                mlirModule: mlirProgram,
                devices: client.addressableDevices
            )
        }

        // Execute - this is captured by the profiler
        // The trace will show execution time alongside XLA kernel execution
        let outputs = try pjrtTraced("execute") {
            try executable.execute(
                arguments: [bufferA, bufferB],
                device: device
            )
        }

        // Read result back
        var result = [Float](repeating: 0.0, count: size * size)
        try pjrtTraced("read_result") {
            try result.withUnsafeMutableBytes { ptr in
                try outputs.first!.toHost(destination: ptr.baseAddress!)
            }
        }

        print("    ✅ Result: [\(size)x\(size)] first element = \(result[0])")
    }

}
