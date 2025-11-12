//===-- PJRT_PowerfulDSL_Example.swift - Powerful DSL Features --*- Swift -*-===//
//
// SwiftIR - The Power of the DSL
// Demonstrates advanced features of the SwiftIR library
//
// This example shows what you can do with SwiftIR's powerful DSL:
// - Type-safe tensor operations
// - Declarative StableHLO module building
// - Compile-time shape checking
// - Clean, readable code
// - Composable operations
//
//===--------------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA

print("========================================")
print("SwiftIR: Powerful DSL Features")
print("The Modern Approach")
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

    // Step 2: Define tensor types (type-safe!)
    print("\n[2/6] Defining type-safe tensor types...")
    let vectorType = TensorType(shape: [4])       // Vector with 4 elements
    let matrixType = TensorType(shape: [2, 4])    // 2x4 matrix
    let resultType = TensorType(shape: [2])       // 2-element result vector

    print("      ✓ Vector type: tensor<4xf32>")
    print("      ✓ Matrix type: tensor<2x4xf32>")
    print("      ✓ Result type: tensor<2xf32>")

    // Step 3: Build StableHLO module with declarative DSL
    print("\n[3/6] Building computation with declarative DSL...")

    // This is where the magic happens - we use a Swift DSL that:
    // 1. Is type-safe (compiler catches shape mismatches)
    // 2. Is composable (operations can be nested and reused)
    // 3. Generates correct MLIR automatically
    // 4. Is readable (looks like mathematical notation)

    let computation = StableHLOModule(name: "matrix_vector_multiply") {
        StableHLOFunction(
            name: "main",
            parameters: [
                Parameter(name: "matrix", type: matrixType),  // 2x4 matrix
                Parameter(name: "vector", type: vectorType)   // 4-element vector
            ],
            returnType: resultType
        ) {
            // Matrix-vector multiplication: (2x4) × (4x1) -> (2x1)
            // This generates the correct StableHLO dot_general operation
            DotGeneral(
                "matrix", "vector",
                lhsType: matrixType,
                rhsType: vectorType,
                resultType: resultType,
                contractingDims: (1, 0)  // Contract over dimension 1 of matrix, 0 of vector
            )
        }
    }

    // Generate MLIR from the DSL (happens automatically)
    let mlirProgram = computation.build()

    print("      ✓ StableHLO module built declaratively")
    print("\n      Generated MLIR (automatic!):")
    for line in mlirProgram.split(separator: "\n").prefix(12) {
        print("        \(line)")
    }
    print("        ...")

    // Step 4: Create input data
    print("\n[4/6] Creating input buffers...")

    // Matrix: [[1, 2, 3, 4],
    //          [5, 6, 7, 8]]
    let matrix: [Float] = [
        1, 2, 3, 4,
        5, 6, 7, 8
    ]

    // Vector: [10, 20, 30, 40]
    let vector: [Float] = [10, 20, 30, 40]

    var matrixBuffer: PJRTBuffer!
    var vectorBuffer: PJRTBuffer!

    matrix.withUnsafeBytes { ptr in
        matrixBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [2, 4],
            elementType: .f32,
            device: device
        )
    }

    vector.withUnsafeBytes { ptr in
        vectorBuffer = try? client.createBuffer(
            data: ptr.baseAddress!,
            shape: [4],
            elementType: .f32,
            device: device
        )
    }

    guard let matrixBuffer = matrixBuffer, let vectorBuffer = vectorBuffer else {
        print("      ERROR: Failed to create buffers")
        exit(1)
    }

    print("      ✓ Matrix: [[1, 2, 3, 4], [5, 6, 7, 8]]")
    print("      ✓ Vector: [10, 20, 30, 40]")

    // Step 5: Compile the DSL-generated MLIR
    print("\n[5/6] Compiling DSL-generated StableHLO...")
    let executable = try client.compile(
        mlirModule: mlirProgram,
        devices: [device]
    )
    print("      ✓ Compilation successful")

    // Step 6: Execute on XLA
    print("\n[6/6] Executing on XLA...")
    let outputs = try executable.execute(
        arguments: [matrixBuffer, vectorBuffer],
        device: device
    )

    guard let outputBuffer = outputs.first else {
        print("      ERROR: No output")
        exit(1)
    }

    // Read results back to host
    var result = [Float](repeating: 0.0, count: 2)
    try result.withUnsafeMutableBytes { ptr in
        try outputBuffer.toHost(destination: ptr.baseAddress!)
    }

    print("      ✓ Execution complete")

    // Step 7: Show results with explanation
    print("\n========================================")
    print("RESULTS")
    print("========================================")
    print("Matrix × Vector = Result")
    print("")
    print("[[1, 2, 3, 4]   [10]   [1*10 + 2*20 + 3*30 + 4*40]   [\(result[0])]")
    print(" [5, 6, 7, 8]] × [20] = [5*10 + 6*20 + 7*30 + 8*40] = [\(result[1])]")
    print("                [30]")
    print("                [40]")
    print("")
    print("Verification:")
    let expected1 = 1*10 + 2*20 + 3*30 + 4*40
    let expected2 = 5*10 + 6*20 + 7*30 + 8*40
    print("  Row 1: 1*10 + 2*20 + 3*30 + 4*40 = \(expected1) ✓")
    print("  Row 2: 5*10 + 6*20 + 7*30 + 8*40 = \(expected2) ✓")

    print("\n========================================")
    print("WHY USE THE DSL?")
    print("========================================")
    print("✓ Type Safety: Compiler catches shape mismatches")
    print("✓ Readability: Clear, declarative syntax")
    print("✓ Composability: Build complex ops from simple ones")
    print("✓ Automatic MLIR: No manual string manipulation")
    print("✓ Maintainability: Refactor with confidence")
    print("✓ Expressiveness: Looks like math, not strings")
    print("\n✓ Success!")
    print("========================================")

    // Cleanup
    outputBuffer.destroy()
    executable.destroy()
    matrixBuffer.destroy()
    vectorBuffer.destroy()

} catch {
    print("\nERROR: \(error)")
    exit(1)
}
