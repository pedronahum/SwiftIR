//===-- PJRT_NeuralNet_Example.swift - Neural Network Example --*- Swift -*-===//
//
// SwiftIR - End-to-End Neural Network
// Demonstrates a complete ML workflow with:
// - 2-layer neural network (4 → 3 → 2)
// - ReLU activation in hidden layer
// - Sigmoid activation in output layer
// - Forward pass inference
//
//===--------------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA

print("========================================")
print("SwiftIR: Neural Network Example")
print("2-Layer Network (4 → 3 → 2)")
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

    // Step 2: Define network architecture
    print("\n[2/6] Defining network architecture...")
    print("      Input:  4 features")
    print("      Hidden: 3 neurons (ReLU activation)")
    print("      Output: 2 classes (Sigmoid activation)")

    // Define tensor types for each layer
    let inputType = TensorType(shape: [1, 4])      // Batch of 1, 4 features
    let weight1Type = TensorType(shape: [4, 3])    // First layer: 4×3
    let bias1Type = TensorType(shape: [3])         // Hidden layer bias
    let hidden1Type = TensorType(shape: [1, 3])    // Hidden layer output
    let weight2Type = TensorType(shape: [3, 2])    // Second layer: 3×2
    let bias2Type = TensorType(shape: [2])         // Output layer bias
    let outputType = TensorType(shape: [1, 2])     // Final output

    // Step 3: Build neural network computation with DSL
    print("\n[3/6] Building neural network with StableHLO DSL...")

    let neuralNet = StableHLOModule(name: "simple_neural_net") {
        StableHLOFunction(
            name: "main",
            parameters: [
                Parameter(name: "input", type: inputType),      // Input data
                Parameter(name: "weight1", type: weight1Type)   // Layer 1 weights
            ],
            returnType: hidden1Type  // Return layer 1 output (1x3) for now
        ) {
            // Layer 1: input @ weight1
            // Result shape: (1, 4) @ (4, 3) = (1, 3)
            // Note: For a complete neural network, we would also add:
            // - bias addition (requires broadcast operation)
            // - ReLU activation (requires maximum operation)
            // - Layer 2 operations
            Return(DotGeneral(
                "input", "weight1",
                lhsType: inputType,
                rhsType: weight1Type,
                resultType: hidden1Type,
                contractingDims: (1, 0)
            ))
        }
    }

    // Generate MLIR
    let mlirProgram = neuralNet.build()

    print("      ✓ Neural network structure defined")
    print("\n      Generated MLIR (layer 1 only for now):")
    for line in mlirProgram.split(separator: "\n").prefix(15) {
        print("        \(line)")
    }

    // Step 4: Prepare input data and weights
    print("\n[4/6] Preparing input data and weights...")

    // Input: Single sample with 4 features
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]

    // Layer 1 weights (4×3) - small random-like values
    let weight1: [Float] = [
        0.1, 0.2, 0.3,   // Feature 1 connections
        0.4, 0.5, 0.6,   // Feature 2 connections
        0.7, 0.8, 0.9,   // Feature 3 connections
        0.2, 0.3, 0.4    // Feature 4 connections
    ]

    // Layer 1 bias (3)
    let bias1: [Float] = [0.1, 0.2, 0.3]

    // Layer 2 weights (3×2)
    let weight2: [Float] = [
        0.5, 0.6,        // Hidden neuron 1 connections
        0.7, 0.8,        // Hidden neuron 2 connections
        0.9, 1.0         // Hidden neuron 3 connections
    ]

    // Layer 2 bias (2)
    let bias2: [Float] = [0.1, 0.2]

    print("      ✓ Input:  \(input)")
    print("      ✓ Weights initialized")

    // Step 5: Create PJRT buffers
    print("\n[5/6] Creating PJRT buffers...")

    var inputBuffer: PJRTBuffer!
    var weight1Buffer: PJRTBuffer!

    input.withUnsafeBytes { ptr in
        inputBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [1, 4],
            elementType: .f32,
            device: device
        )
    }

    weight1.withUnsafeBytes { ptr in
        weight1Buffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [4, 3],
            elementType: .f32,
            device: device
        )
    }

    guard let inputBuffer = inputBuffer, let weight1Buffer = weight1Buffer else {
        print("      ERROR: Failed to create buffers")
        exit(1)
    }

    print("      ✓ Buffers created on device")

    // Step 6: Compile and execute
    print("\n[6/6] Compiling and running network...")

    let executable = try client.compile(
        mlirModule: mlirProgram,
        devices: [device]
    )
    print("      ✓ Network compiled")

    // For now we only have layer 1 (matrix multiply)
    // In a complete implementation, we would have:
    // 1. Layer 1 matmul + bias + ReLU
    // 2. Layer 2 matmul + bias + Sigmoid

    // Execute just layer 1 for now
    let outputs = try executable.execute(
        arguments: [inputBuffer, weight1Buffer],
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

    // Step 7: Display results
    print("\n========================================")
    print("RESULTS")
    print("========================================")
    print("Layer 1 Output (before activation):")
    print("  Hidden layer: [\(result[0]), \(result[1]), \(result[2])]")
    print("")
    print("Expected calculation:")
    let expected1 = 1.0*0.1 + 2.0*0.4 + 3.0*0.7 + 4.0*0.2
    let expected2 = 1.0*0.2 + 2.0*0.5 + 3.0*0.8 + 4.0*0.3
    let expected3 = 1.0*0.3 + 2.0*0.6 + 3.0*0.9 + 4.0*0.4
    print("  h1 = 1.0*0.1 + 2.0*0.4 + 3.0*0.7 + 4.0*0.2 = \(expected1)")
    print("  h2 = 1.0*0.2 + 2.0*0.5 + 3.0*0.8 + 4.0*0.3 = \(expected2)")
    print("  h3 = 1.0*0.3 + 2.0*0.6 + 3.0*0.9 + 4.0*0.4 = \(expected3)")

    print("\n========================================")
    print("NEXT STEPS")
    print("========================================")
    print("This example shows layer 1 computation.")
    print("To complete the neural network, we need:")
    print("  1. Sequential operation composition in DSL")
    print("  2. Broadcast operations for bias addition")
    print("  3. ReLU activation (max(x, 0))")
    print("  4. Sigmoid activation (logistic)")
    print("")
    print("All the activation ops are implemented!")
    print("What's needed is better DSL composition.")
    print("\n✓ Neural network foundation working!")
    print("========================================")

    // Cleanup
    outputBuffer.destroy()
    executable.destroy()
    inputBuffer.destroy()
    weight1Buffer.destroy()

} catch {
    print("\nERROR: \(error)")
    exit(1)
}
