//===-- PJRT_Example.swift - PJRT Integration Example ----*- Swift -*-===//
//
// SwiftIR - Phase 11: PJRT Integration Example
// Demonstrates using PJRT for GPU/TPU execution
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA
import Foundation

/// Comprehensive PJRT integration example
///
/// This demonstrates:
/// 1. Loading PJRT plugins (CPU/GPU)
/// 2. Device enumeration
/// 3. Buffer management (host ↔ device transfers)
/// 4. Program compilation
/// 5. Execution on accelerators
func runPJRTExample() {
    print("=" * 70)
    print("PJRT Integration Example - True XLA Acceleration")
    print("=" * 70)

    // MARK: - Stage 1: Plugin Loading

    print("\n📦 Stage 1: Loading PJRT Plugins")
    print("   PJRT allows dynamic loading of backend-specific plugins")
    print("   Plugins available: CPU, GPU (CUDA/ROCm), TPU")

    do {
        // Try to create a CPU client first
        let cpuClient = try PJRTClient(backend: .cpu)
        print("\n   ✅ CPU Client created")
        print("   Platform: \(cpuClient.platformName)")
        print("   Devices: \(cpuClient.devices.count)")

        for device in cpuClient.devices {
            print("      • \(device.description)")
        }

        // MARK: - Stage 2: Device Enumeration

        print("\n🔍 Stage 2: Device Enumeration")
        print("   Addressable devices (can execute on):")
        for device in cpuClient.addressableDevices {
            print("      • \(device.description)")
        }

        guard let defaultDevice = cpuClient.defaultDevice else {
            print("   ❌ No default device available")
            return
        }
        print("\n   Using device: \(defaultDevice.description)")

        // MARK: - Stage 3: Buffer Management

        print("\n💾 Stage 3: Buffer Management")
        print("   Creating test tensor on host: [2, 3] float32")

        // Create host data: simple 2x3 matrix
        var hostData: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]

        var inputBuffer: PJRTBuffer?
        hostData.withUnsafeBytes { ptr in
            do {
                let buffer = try cpuClient.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [2, 3],
                    elementType: .f32,
                    device: defaultDevice
                )

                print("   ✅ Buffer created")
                print("   Shape: \(buffer.shape)")
                print("   Type: F32")
                print("   Size: \(buffer.sizeInBytes) bytes")
                print("   Device: \(buffer.device.description)")

                inputBuffer = buffer

            } catch {
                print("   ❌ Buffer creation failed: \(error)")
            }
        }

        // MARK: - Stage 4: Program Compilation

        print("\n🔨 Stage 4: Program Compilation")
        print("   Compiling StableHLO program to XLA HLO")

        // Create a simple StableHLO computation
        let context = MLIRContext()
        // Load only the dialects we need (avoid GPU/SPIRV which aren't linked)
        _ = loadStablehloDialect(context)
        _ = context.loadDialect("func")
        _ = context.loadDialect("arith")

        let module = MLIRModule(context: context)

        // In a real scenario, we would build a complete StableHLO program here
        // For now, demonstrate the compilation API

        print("   Program: element-wise multiplication")
        print("   Input:  tensor<2x3xf32>")
        print("   Output: tensor<2x3xf32>")

        do {
            // Create a minimal valid StableHLO program
            // This is a simple identity function that returns its input
            let mlirText = """
            module @jit_add {
              func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                return %arg0 : tensor<2x3xf32>
              }
            }
            """

            let executable = try cpuClient.compile(
                mlirModule: mlirText,
                devices: cpuClient.addressableDevices
            )

            print("\n   ✅ Compilation successful!")
            print("   Target devices: \(executable.devices.count)")

            // MARK: - Stage 5: Execution

            print("\n⚡ Stage 5: Execution")
            print("   Running compiled program on device")

            // Execute with the input buffer we created
            guard let inputBuffer = inputBuffer else {
                print("   ⚠️  No input buffer available for execution")
                return
            }

            let outputs = try executable.execute(
                arguments: [inputBuffer],
                device: defaultDevice
            )

            print("   ✅ Execution complete!")
            print("   Execution count: \(executable.executionCount)")
            print("   Outputs: \(outputs.count) buffers")

            // Try to read back the output
            if let outputBuffer = outputs.first {
                print("\n   📤 Reading output buffer:")
                print("   Shape: \(outputBuffer.shape)")
                print("   Type: \(outputBuffer.elementType)")
                print("   Size: \(outputBuffer.sizeInBytes) bytes")

                // Allocate space for output data (2x3 matrix = 6 floats)
                var outputData = [Float](repeating: 0.0, count: 6)
                try outputData.withUnsafeMutableBytes { ptr in
                    try outputBuffer.toHost(destination: ptr.baseAddress!)
                }

                print("   Output values: \(outputData)")
                print("   Expected: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] (identity)")
            }

        } catch {
            print("   ⚠️  Compilation/Execution: \(error)")
            print("   Note: Full implementation requires PJRT library linkage")
        }

        // MARK: - GPU Example (if available)

        print("\n🎮 GPU Execution Path (Future)")
        print("   When GPU plugin is available:")
        print("   1. Load GPU plugin: PJRTClient(backend: .gpu)")
        print("   2. Enumerate GPU devices (CUDA/ROCm)")
        print("   3. Compile with GPU-specific optimizations")
        print("   4. Execute on GPU with kernel fusion")

    } catch {
        print("   ❌ Client creation failed: \(error)")
    }

    // MARK: - Integration Architecture

    print("\n🏗️  PJRT Integration Architecture")
    print("\n   SwiftIR Application")
    print("         ↓")
    print("   PJRTClient.swift (Swift API)")
    print("         ↓")
    print("   PJRTWrapper.h (C bindings)")
    print("         ↓")
    print("   libpjrt_c_api_*.so (Plugin library)")
    print("         ↓")
    print("   XLA Compiler + Runtime")
    print("         ↓")
    print("   GPU/TPU Hardware")

    // MARK: - Performance Expectations

    print("\n📊 Performance Expectations")
    print("\n   CPU (via PJRT):")
    print("   • Throughput: ~10-50 GFLOPS")
    print("   • Latency: 1-10ms for small tensors")
    print("   • Optimization: Vectorization + loop fusion")

    print("\n   GPU (CUDA/ROCm):")
    print("   • Throughput: ~1-10 TFLOPS")
    print("   • Latency: 0.1-1ms for small tensors")
    print("   • Optimization: Kernel fusion + autotuning")
    print("   • Memory: Unified memory or explicit transfers")

    print("\n   TPU:")
    print("   • Throughput: ~100+ TFLOPS")
    print("   • Latency: <0.1ms for small tensors")
    print("   • Optimization: Systolic array utilization")
    print("   • Memory: High-bandwidth memory (HBM)")

    // MARK: - Next Steps

    print("\n🛣️  Integration Roadmap")
    print("\n   Current Status:")
    print("   ✅ PJRT C API headers integrated")
    print("   ✅ Swift wrapper layer complete")
    print("   ✅ Plugin loading infrastructure ready")
    print("   ✅ Device enumeration API")
    print("   ✅ Buffer management API")
    print("   ✅ Compilation API")

    print("\n   To Enable Full Functionality:")
    print("   1. Build PJRT CPU plugin:")
    print("      cd /path/to/xla")
    print("      bazel build //xla/pjrt/c:pjrt_c_api_cpu")
    print("")
    print("   2. Copy plugin to SwiftIR:")
    print("      cp bazel-bin/xla/pjrt/c/libpjrt_c_api_cpu.so \\")
    print("         /path/to/SwiftIR/lib/")
    print("")
    print("   3. Update Package.swift linker settings:")
    print("      .unsafeFlags([\"-L/path/to/SwiftIR/lib\",")
    print("                    \"-lpjrt_c_api_cpu\"])")
    print("")
    print("   4. Replace stub implementations in PJRTClient.swift")

    print("\n   For GPU Support:")
    print("   1. Build PJRT GPU plugin:")
    print("      bazel build //xla/pjrt/c:pjrt_c_api_gpu")
    print("")
    print("   2. Ensure CUDA/ROCm installed")
    print("   3. Copy GPU plugin and update linker")

    // MARK: - Comparison with Other Backends

    print("\n🔬 Backend Comparison")
    print("\n   Current (LLVM ExecutionEngine):")
    print("   ✓ Works today")
    print("   ✓ No external dependencies")
    print("   ✓ Good for prototyping")
    print("   ✗ CPU only")
    print("   ✗ No distributed execution")

    print("\n   With PJRT:")
    print("   ✓ GPU/TPU support")
    print("   ✓ Production-ready performance")
    print("   ✓ Distributed training")
    print("   ✓ Framework compatibility (JAX/TF)")
    print("   ✗ Requires external library")

    // MARK: - Example Use Cases

    print("\n💡 Real-World Use Cases")

    print("\n   1. Large Model Training:")
    print("      • Multi-GPU data parallelism")
    print("      • Gradient accumulation")
    print("      • Mixed precision training")

    print("\n   2. Production Inference:")
    print("      • Low-latency serving")
    print("      • Batch processing")
    print("      • Model quantization")

    print("\n   3. Research Prototyping:")
    print("      • Rapid iteration")
    print("      • Custom operators")
    print("      • Algorithm exploration")

    print("\n   4. Edge Deployment:")
    print("      • Mobile GPU (Metal/OpenCL)")
    print("      • Embedded accelerators")
    print("      • Power-efficient inference")

    print("\n" + "=" * 70)
    print("✨ PJRT Integration Example Complete!")
    print("=" * 70)
}

// MARK: - Helper Extension

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run the example
runPJRTExample()
