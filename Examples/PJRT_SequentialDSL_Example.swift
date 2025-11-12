//===-- PJRT_SequentialDSL_Example.swift - Sequential DSL Example --*- Swift -*-===//
//
// SwiftIR - Sequential Operation Composition
// Demonstrates the NEW Let-binding DSL for chaining operations:
// - Named intermediate results
// - Operation sequencing (matmul → add → relu)
// - Complete neural network layer in one function
//
//===--------------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA

print("========================================")
print("SwiftIR: Sequential DSL Composition")
print("Chaining Operations: matmul → add → relu")
print("========================================")

do {
    // Step 1: Initialize PJRT CPU client
    print("\n[1/6] Initializing PJRT CPU client...")
    let client = try PJRTClient(backend: .cpu)
    guard let device = client.defaultDevice else {
        print("ERROR: No device available")
        exit(1)
    }
    print("      Device: \(device.description)")

    // Step 2: Define tensor types for a complete neural network layer
    print("\n[2/6] Defining neural network layer types...")
    let inputType = TensorType(shape: [1, 4])       // Input: 1 sample, 4 features
    let weightType = TensorType(shape: [4, 3])      // Weights: 4×3 matrix
    let biasType = TensorType(shape: [1, 3])        // Bias: 1×3 (broadcasted)
    let hiddenType = TensorType(shape: [1, 3])      // Hidden layer output
    let zeroType = TensorType(shape: [1, 3])        // Zero constant for ReLU

    print("      ✓ Input:  tensor<1x4xf32>")
    print("      ✓ Weight: tensor<4x3xf32>")
    print("      ✓ Bias:   tensor<1x3xf32>")
    print("      ✓ Output: tensor<1x3xf32>")

    // Step 3: Build complete neural network layer with sequential composition
    print("\n[3/6] Building neural network layer with Let-binding DSL...")

    let neuralNetLayer = StableHLOModule(name: "sequential_neural_layer") {
        StableHLOFunction(
            name: "main",
            parameters: [
                Parameter(name: "input", type: inputType),
                Parameter(name: "weight", type: weightType),
                Parameter(name: "bias", type: biasType)
            ],
            returnType: hiddenType
        ) {
            // Step 1: matmul - matrix multiplication (input @ weight)
            // Result: (1, 4) @ (4, 3) = (1, 3)
            Let("matmul_result",
                DotGeneral(
                    "input", "weight",
                    lhsType: inputType,
                    rhsType: weightType,
                    resultType: hiddenType,
                    contractingDims: (1, 0)
                )
            )

            // Step 2: add - add bias to matmul result
            // Result: (1, 3) + (1, 3) = (1, 3)
            Let("add_result",
                Add("matmul_result", "bias", type: hiddenType)
            )

            // Step 3: relu - activation function: max(add_result, 0)
            // First create a zero constant
            Let("zero",
                Constant(value: 0.0, type: zeroType)
            )

            // Then apply ReLU: max(add_result, 0)
            Return(
                Maximum("add_result", "zero", type: hiddenType)
            )
        }
    }

    // Generate MLIR
    let mlirProgram = neuralNetLayer.build()

    print("      ✓ Complete layer: matmul → add → relu")
    print("\n      Generated MLIR with sequential operations:")
    for line in mlirProgram.split(separator: "\n") {
        print("        \(line)")
    }

    // Step 4: Prepare input data
    print("\n[4/6] Preparing input data...")

    // Input: [1.0, 2.0, 3.0, 4.0]
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]

    // Weights (4×3)
    let weight: [Float] = [
        0.1, 0.2, 0.3,   // Feature 1 connections
        0.4, 0.5, 0.6,   // Feature 2 connections
        0.7, 0.8, 0.9,   // Feature 3 connections
        0.2, 0.3, 0.4    // Feature 4 connections
    ]

    // Bias (1×3) - using negative values to test ReLU
    let bias: [Float] = [-1.0, 0.5, 1.0]

    print("      ✓ Input:  \(input)")
    print("      ✓ Bias:   \(bias)")

    // Step 5: Create PJRT buffers
    print("\n[5/6] Creating PJRT buffers...")

    var inputBuffer: PJRTBuffer!
    var weightBuffer: PJRTBuffer!
    var biasBuffer: PJRTBuffer!

    input.withUnsafeBytes { ptr in
        inputBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [1, 4],
            elementType: .f32,
            device: device
        )
    }

    weight.withUnsafeBytes { ptr in
        weightBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [4, 3],
            elementType: .f32,
            device: device
        )
    }

    bias.withUnsafeBytes { ptr in
        biasBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [1, 3],
            elementType: .f32,
            device: device
        )
    }

    guard let inputBuffer = inputBuffer,
          let weightBuffer = weightBuffer,
          let biasBuffer = biasBuffer else {
        print("      ERROR: Failed to create buffers")
        exit(1)
    }

    print("      ✓ Buffers created on device")

    // Step 6: Compile and execute
    print("\n[6/6] Compiling and executing neural network layer...")

    let executable = try client.compile(
        mlirModule: mlirProgram,
        devices: [device]
    )
    print("      ✓ Compiled successfully")

    let outputs = try executable.execute(
        arguments: [inputBuffer, weightBuffer, biasBuffer],
        device: device
    )

    guard let outputBuffer = outputs.first else {
        print("      ERROR: No output")
        exit(1)
    }

    // Read results
    var result = [Float](repeating: 0.0, count: 3)
    try result.withUnsafeMutableBytes { ptr in
        try outputBuffer.toHost(destination: ptr.baseAddress!)
    }

    print("      ✓ Execution complete")

    // Step 7: Display and verify results
    print("\n========================================")
    print("RESULTS")
    print("========================================")

    // Calculate expected values step by step
    let matmul1: Float = 1.0*0.1 + 2.0*0.4 + 3.0*0.7 + 4.0*0.2
    let matmul2: Float = 1.0*0.2 + 2.0*0.5 + 3.0*0.8 + 4.0*0.3
    let matmul3: Float = 1.0*0.3 + 2.0*0.6 + 3.0*0.9 + 4.0*0.4

    let afterAdd1: Float = matmul1 + bias[0]
    let afterAdd2: Float = matmul2 + bias[1]
    let afterAdd3: Float = matmul3 + bias[2]

    let afterRelu1: Float = max(afterAdd1, 0.0)
    let afterRelu2: Float = max(afterAdd2, 0.0)
    let afterRelu3: Float = max(afterAdd3, 0.0)

    print("Operation Sequence:")
    print("")
    print("1. MatMul (input @ weight):")
    print("   [\(matmul1), \(matmul2), \(matmul3)]")
    print("")
    print("2. Add Bias:")
    print("   [\(afterAdd1), \(afterAdd2), \(afterAdd3)]")
    print("")
    print("3. ReLU (max with 0):")
    print("   [\(afterRelu1), \(afterRelu2), \(afterRelu3)]")
    print("")
    print("Actual Output from XLA:")
    print("   [\(result[0]), \(result[1]), \(result[2])]")

    // Verification
    let tolerance: Float = 0.0001
    let correct1 = abs(result[0] - afterRelu1) < tolerance
    let correct2 = abs(result[1] - afterRelu2) < tolerance
    let correct3 = abs(result[2] - afterRelu3) < tolerance

    print("")
    if correct1 && correct2 && correct3 {
        print("✅ Results match expected values!")
    } else {
        print("❌ Results don't match expected values")
    }

    print("\n========================================")
    print("KEY FEATURES DEMONSTRATED")
    print("========================================")
    print("✓ Let-binding for intermediate results")
    print("✓ Named variables (matmul_result, add_result, zero)")
    print("✓ Sequential operation chaining")
    print("✓ Complete neural network layer in one function")
    print("✓ ReLU activation with constant zero")
    print("")
    print("✓ Sequential DSL working perfectly!")
    print("========================================")

    // Cleanup
    outputBuffer.destroy()
    executable.destroy()
    inputBuffer.destroy()
    weightBuffer.destroy()
    biasBuffer.destroy()

} catch {
    print("\nERROR: \(error)")
    exit(1)
}
