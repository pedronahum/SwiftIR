// Minimal test: Can approaches 1 and 2 coexist (skipping approach 3)?
// This tests if the DSL approach (2) causes issues when combined with approach 1

import Foundation
import SwiftIRXLA
import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO

func runApproaches1and2Test() throws {
    print("Testing: Can approaches 1 and 2 coexist?")
    print(String(repeating: "=", count: 60))

    // Initialize PJRT client
    print("\n📱 Initializing PJRT CPU Client...")
    let cpuClient = try PJRTClient(backend: .cpu)
    guard let device = cpuClient.defaultDevice else {
        throw PJRTError.noDeviceAvailable
    }

    // Matrix dimensions
    let rowsA = 4, colsA = 3, colsB = 2

    // Approach 1: MLIR Ops (Manual StableHLO string)
    print("\n--- Approach 1: MLIR Ops ---")
    let mlirProgram1 = """
    module @test {
      func.func public @main(
        %arg0: tensor<4x3xf32>,
        %arg1: tensor<3x2xf32>
      ) -> tensor<4x2xf32> {
        %0 = stablehlo.dot_general %arg0, %arg1,
          contracting_dims = [1] x [0]
          : (tensor<4x3xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
        return %0 : tensor<4x2xf32>
      }
    }
    """

    // Create input buffers for approach 1
    let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    let matrixB: [Float] = [10, 20, 30, 40, 50, 60]

    var bufferA: PJRTBuffer!
    var bufferB: PJRTBuffer!

    matrixA.withUnsafeBytes { ptr in
        bufferA = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [rowsA, colsA],
            elementType: .f32,
            device: device
        )
    }

    matrixB.withUnsafeBytes { ptr in
        bufferB = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [colsA, colsB],
            elementType: .f32,
            device: device
        )
    }

    print("✅ Input buffers created for approach 1")

    let executable1 = try cpuClient.compile(mlirModule: mlirProgram1, devices: [device])
    print("✅ Executable 1 compiled")

    let outputs1 = try executable1.execute(arguments: [bufferA, bufferB], device: device)
    print("✅ Executable 1 executed")

    guard let outputBuffer1 = outputs1.first else {
        print("❌ No output from executable 1")
        return
    }

    var result1 = [Float](repeating: 0.0, count: 8)
    try result1.withUnsafeMutableBytes { ptr in
        try outputBuffer1.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 1: \(result1[0]), \(result1[1]), \(result1[2]), \(result1[3])")

    // Cleanup approach 1
    outputBuffer1.destroy()
    executable1.destroy()
    bufferA.destroy()
    bufferB.destroy()
    print("✅ Approach 1 cleaned up")

    // Approach 2: StableHLO DSL
    print("\n--- Approach 2: StableHLO DSL ---")

    // Define tensor types for DSL
    let tensorA = TensorType(shape: [rowsA, colsA])
    let tensorB = TensorType(shape: [colsA, colsB])
    let tensorC = TensorType(shape: [rowsA, colsB])

    let module2 = StableHLOModule(name: "dsl_matmul") {
        StableHLOFunction(
            name: "main",
            parameters: [
                Parameter(name: "arg0", type: tensorA),
                Parameter(name: "arg1", type: tensorB)
            ],
            returnType: tensorC
        ) {
            Return(DotGeneral(
                "arg0", "arg1",
                lhsType: tensorA,
                rhsType: tensorB,
                resultType: tensorC,
                contractingDims: (1, 0)
            ))
        }
    }

    let mlirProgram2 = module2.build()
    print("✅ StableHLO program built with DSL")

    // Create fresh buffers for approach 2
    var bufferA2: PJRTBuffer!
    var bufferB2: PJRTBuffer!

    matrixA.withUnsafeBytes { ptr in
        bufferA2 = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [rowsA, colsA],
            elementType: .f32,
            device: device
        )
    }

    matrixB.withUnsafeBytes { ptr in
        bufferB2 = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [colsA, colsB],
            elementType: .f32,
            device: device
        )
    }

    print("✅ Input buffers created for approach 2")

    let executable2 = try cpuClient.compile(mlirModule: mlirProgram2, devices: [device])
    print("✅ Executable 2 compiled")

    let outputs2 = try executable2.execute(arguments: [bufferA2, bufferB2], device: device)
    print("✅ Executable 2 executed")

    guard let outputBuffer2 = outputs2.first else {
        print("❌ No output from executable 2")
        return
    }

    var result2 = [Float](repeating: 0.0, count: 8)
    try result2.withUnsafeMutableBytes { ptr in
        try outputBuffer2.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 2: \(result2[0]), \(result2[1]), \(result2[2]), \(result2[3])")

    // Cleanup approach 2
    outputBuffer2.destroy()
    executable2.destroy()
    bufferA2.destroy()
    bufferB2.destroy()
    print("✅ Approach 2 cleaned up")

    // Test if we can create a string after all operations
    print("\n--- Testing string creation after all operations ---")
    let testString = String(repeating: "=", count: 60)
    print(testString)
    print("🎉 SUCCESS! Both approaches 1 and 2 worked!")
    print(String(repeating: "=", count: 60))
}

// Run the test
do {
    try runApproaches1and2Test()
} catch {
    print("❌ Test failed: \(error)")
    exit(1)
}
