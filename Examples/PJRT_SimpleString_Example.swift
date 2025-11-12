//===-- PJRT_SimpleString_Example.swift - Simple MLIR String Example --*- Swift -*-===//
//
// SwiftIR - "Poor Man's MLIR"
// Demonstrates the basic approach: hand-written MLIR strings
//
// This example shows the simplest way to use PJRT with XLA:
// - Write MLIR StableHLO as strings
// - Compile and execute with PJRT
// - No fancy DSLs or builders needed
//
//===--------------------------------------------------------------------------===//

import Foundation
import SwiftIRXLA

print("========================================")
print("SwiftIR: Simple String-Based MLIR")
print("The 'Poor Man's' Approach")
print("========================================")

do {
    // Step 1: Initialize PJRT CPU client
    print("\n[1/5] Initializing PJRT CPU client...")
    let client = try PJRTClient(backend: .cpu)
    guard let device = client.defaultDevice else {
        print("ERROR: No device available")
        exit(1)
    }
    print("      Device: \(device.description)")

    // Step 2: Write MLIR as a string (the "poor man's" way)
    print("\n[2/5] Writing MLIR StableHLO as a string...")

    // Simple computation: element-wise add two vectors
    // Result[i] = A[i] + B[i]
    let mlirProgram = """
    module @simple_add {
      func.func public @main(
        %arg0: tensor<4xf32>,
        %arg1: tensor<4xf32>
      ) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """

    print("      ✓ MLIR program written")
    print("\n      Generated MLIR:")
    for line in mlirProgram.split(separator: "\n") {
        print("        \(line)")
    }

    // Step 3: Create input buffers
    print("\n[3/5] Creating input buffers...")
    let vectorA: [Float] = [1.0, 2.0, 3.0, 4.0]
    let vectorB: [Float] = [10.0, 20.0, 30.0, 40.0]

    var bufferA: PJRTBuffer!
    var bufferB: PJRTBuffer!

    vectorA.withUnsafeBytes { ptr in
        bufferA = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [4],
            elementType: .f32,
            device: device
        )
    }

    vectorB.withUnsafeBytes { ptr in
        bufferB = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [4],
            elementType: .f32,
            device: device
        )
    }

    guard let bufferA = bufferA, let bufferB = bufferB else {
        print("      ERROR: Failed to create buffers")
        exit(1)
    }

    print("      ✓ Buffer A: \(vectorA)")
    print("      ✓ Buffer B: \(vectorB)")

    // Step 4: Compile the MLIR string
    print("\n[4/5] Compiling MLIR program...")
    let executable = try client.compile(
        mlirModule: mlirProgram,
        devices: [device]
    )
    print("      ✓ Compilation successful")

    // Step 5: Execute and get results
    print("\n[5/5] Executing on XLA...")
    let outputs = try executable.execute(
        arguments: [bufferA, bufferB],
        device: device
    )

    guard let outputBuffer = outputs.first else {
        print("      ERROR: No output")
        exit(1)
    }

    // Read results back to host
    var result = [Float](repeating: 0.0, count: 4)
    try result.withUnsafeMutableBytes { ptr in
        try outputBuffer.toHost(destination: ptr.baseAddress!)
    }

    print("      ✓ Execution complete")
    print("\n========================================")
    print("RESULTS")
    print("========================================")
    print("A      = \(vectorA)")
    print("B      = \(vectorB)")
    print("A + B  = \(result)")
    print("\n✓ Success!")
    print("========================================")

    // Cleanup
    outputBuffer.destroy()
    executable.destroy()
    bufferA.destroy()
    bufferB.destroy()

} catch {
    print("\nERROR: \(error)")
    exit(1)
}
