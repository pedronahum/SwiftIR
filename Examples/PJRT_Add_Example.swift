//===-- PJRT_Add_Example.swift - PJRT Addition Example ----*- Swift -*-===//
//
// SwiftIR - PJRT Integration: Real Computation Example
// Demonstrates element-wise addition with XLA optimization
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders
import SwiftIRStableHLO
import SwiftIRXLA
import Foundation

/// Element-wise addition example using PJRT/XLA
///
/// This demonstrates:
/// 1. Real computation (not just identity)
/// 2. Multiple input buffers
/// 3. XLA compiler optimizations
/// 4. Verifiable results
func runPJRTAddExample() {
    print("=" * 70)
    print("PJRT Element-Wise Addition Example")
    print("Demonstrating Real XLA Computation")
    print("=" * 70)

    do {
        // Create CPU client
        let cpuClient = try PJRTClient(backend: .cpu)
        print("\n✅ PJRT CPU Client initialized")
        print("   Platform: \(cpuClient.platformName)")
        print("   Devices: \(cpuClient.devices.count)")

        guard let device = cpuClient.defaultDevice else {
            print("❌ No default device available")
            return
        }

        // MARK: - Create Input Tensors

        print("\n📊 Creating Input Tensors")

        // First tensor: [1, 2, 3, 4, 5, 6] shaped as 2x3
        var inputA: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        // Second tensor: [10, 20, 30, 40, 50, 60] shaped as 2x3
        var inputB: [Float] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        print("   Input A: \(inputA)")
        print("   Input B: \(inputB)")
        print("   Shape: [2, 3] (2x3 matrix)")

        // Create buffers for both inputs
        var bufferA: PJRTBuffer?
        var bufferB: PJRTBuffer?

        inputA.withUnsafeBytes { ptrA in
            do {
                bufferA = try cpuClient.createBuffer(
                    data: ptrA.baseAddress!,
                    shape: [2, 3],
                    elementType: .f32,
                    device: device
                )
                print("\n   ✅ Buffer A created: [2, 3] f32, 24 bytes")
            } catch {
                print("   ❌ Failed to create buffer A: \(error)")
            }
        }

        inputB.withUnsafeBytes { ptrB in
            do {
                bufferB = try cpuClient.createBuffer(
                    data: ptrB.baseAddress!,
                    shape: [2, 3],
                    elementType: .f32,
                    device: device
                )
                print("   ✅ Buffer B created: [2, 3] f32, 24 bytes")
            } catch {
                print("   ❌ Failed to create buffer B: \(error)")
            }
        }

        guard let bufferA = bufferA, let bufferB = bufferB else {
            print("   ❌ Failed to create input buffers")
            return
        }

        // MARK: - Compile StableHLO Program

        print("\n🔨 Compiling StableHLO Addition Program")
        print("   Operation: output = input_a + input_b")
        print("   XLA will optimize this to vectorized CPU instructions")

        // NOTE: SwiftIR has comprehensive MLIR builder APIs (see StablehloOps.swift, Func.swift)
        // For complex programs, you can use:
        //   - Stablehlo.add(lhs, rhs, ...) for StableHLO operations
        //   - Func.function(...).setBody(...).build() for function construction
        //   - MLIRModule, MLIRBlock, OperationBuilder for full programmatic control
        //
        // For this simple example, we use a straightforward MLIR string:

        let mlirProgram = """
        module @jit_add {
          func.func public @main(
            %arg0: tensor<2x3xf32>,
            %arg1: tensor<2x3xf32>
          ) -> tensor<2x3xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        print("   ✅ StableHLO program ready for XLA compilation")

        let executable = try cpuClient.compile(
            mlirModule: mlirProgram,
            devices: cpuClient.addressableDevices
        )

        print("   ✅ Compilation successful!")
        print("   XLA has optimized the addition for CPU execution")

        // MARK: - Execute

        print("\n⚡ Executing on Device")

        let outputs = try executable.execute(
            arguments: [bufferA, bufferB],
            device: device
        )

        print("   ✅ Execution complete!")
        print("   Execution count: \(executable.executionCount)")

        // MARK: - Read Results

        guard let outputBuffer = outputs.first else {
            print("   ❌ No output buffer returned")
            return
        }

        print("\n📤 Reading Results")
        print("   Output shape: \(outputBuffer.shape)")
        print("   Output type: \(outputBuffer.elementType)")
        print("   Output size: \(outputBuffer.sizeInBytes) bytes")

        var outputData = [Float](repeating: 0.0, count: 6)
        try outputData.withUnsafeMutableBytes { ptr in
            try outputBuffer.toHost(destination: ptr.baseAddress!)
        }

        print("\n🎯 Results:")
        print("   Input A:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]")
        print("   Input B:  [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]")
        print("   Output:   \(outputData)")
        print("   Expected: [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]")

        // Verify correctness
        let expected: [Float] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        let allCorrect = zip(outputData, expected).allSatisfy { abs($0 - $1) < 0.001 }

        if allCorrect {
            print("\n   ✅ VERIFICATION PASSED! All values correct!")
        } else {
            print("\n   ❌ VERIFICATION FAILED! Values don't match expected")
        }

        // MARK: - XLA Optimization Details

        print("\n🚀 XLA Optimization Benefits")
        print("\n   What XLA Did:")
        print("   1. Analyzed the addition operation")
        print("   2. Generated vectorized CPU instructions (SIMD)")
        print("   3. Optimized memory access patterns")
        print("   4. Eliminated unnecessary copies")
        print("   5. Fused operations into single kernel")

        print("\n   Performance Benefits:")
        print("   • CPU SIMD vectorization (4-8 ops/instruction)")
        print("   • Cache-friendly memory layout")
        print("   • No Python/Swift overhead in hot loop")
        print("   • Ready for GPU/TPU with same code")

        // MARK: - Matrix View

        print("\n📐 Matrix View (2x3):")
        print("   Input A:")
        print("     [ \(inputA[0])  \(inputA[1])  \(inputA[2]) ]")
        print("     [ \(inputA[3])  \(inputA[4])  \(inputA[5]) ]")
        print("\n   Input B:")
        print("     [ \(inputB[0])  \(inputB[1])  \(inputB[2]) ]")
        print("     [ \(inputB[3])  \(inputB[4])  \(inputB[5]) ]")
        print("\n   Output:")
        print("     [ \(outputData[0])  \(outputData[1])  \(outputData[2]) ]")
        print("     [ \(outputData[3])  \(outputData[4])  \(outputData[5]) ]")

    } catch {
        print("❌ Error: \(error)")
    }

    print("\n" + "=" * 70)
    print("✨ PJRT Addition Example Complete!")
    print("=" * 70)
}

// MARK: - Helper Extension

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run the example
runPJRTAddExample()
