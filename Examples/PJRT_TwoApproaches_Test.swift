// Minimal test: Can approaches 1 and 3 coexist (skipping approach 2)?
// This tests if the issue is specific to having all 3 approaches

import Foundation
import SwiftIRXLA

func runTwoApproachesTest() throws {
    print("Testing: Can approaches 1 and 3 coexist?")
    print(String(repeating: "=", count: 60))

    // Initialize PJRT client
    print("\n📱 Initializing PJRT CPU Client...")
    let cpuClient = try PJRTClient(backend: .cpu)
    guard let device = cpuClient.defaultDevice else {
        throw PJRTError.noDeviceAvailable
    }

    // Simple computation: dot product
    let mlirProgram = """
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

    // Create input buffers
    let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    let matrixB: [Float] = [10, 20, 30, 40, 50, 60]

    var bufferA: PJRTBuffer!
    var bufferB: PJRTBuffer!

    matrixA.withUnsafeBytes { ptr in
        bufferA = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [4, 3],
            elementType: .f32,
            device: device
        )
    }

    matrixB.withUnsafeBytes { ptr in
        bufferB = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [3, 2],
            elementType: .f32,
            device: device
        )
    }

    print("✅ Input buffers created")

    // Test 1: Approach 1 (similar to MLOps approach in MultiPath)
    print("\n--- Approach 1: First executable ---")
    let executable1 = try cpuClient.compile(mlirModule: mlirProgram, devices: [device])
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

    // Recreate input buffers for approach 3
    print("\n--- Recreating buffers for approach 3 ---")
    var bufferA3: PJRTBuffer!
    var bufferB3: PJRTBuffer!

    matrixA.withUnsafeBytes { ptr in
        bufferA3 = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [4, 3],
            elementType: .f32,
            device: device
        )
    }

    matrixB.withUnsafeBytes { ptr in
        bufferB3 = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [3, 2],
            elementType: .f32,
            device: device
        )
    }

    print("✅ Buffers recreated")

    // Test 2: Approach 3 (similar to Manual approach in MultiPath)
    print("\n--- Approach 3: Second executable ---")
    let executable3 = try cpuClient.compile(mlirModule: mlirProgram, devices: [device])
    print("✅ Executable 3 compiled")

    let outputs3 = try executable3.execute(arguments: [bufferA3, bufferB3], device: device)
    print("✅ Executable 3 executed")

    guard let outputBuffer3 = outputs3.first else {
        print("❌ No output from executable 3")
        return
    }

    var result3 = [Float](repeating: 0.0, count: 8)
    try result3.withUnsafeMutableBytes { ptr in
        try outputBuffer3.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 3: \(result3[0]), \(result3[1]), \(result3[2]), \(result3[3])")

    // Cleanup approach 3
    outputBuffer3.destroy()
    executable3.destroy()
    bufferA3.destroy()
    bufferB3.destroy()
    print("✅ Approach 3 cleaned up")

    // Test if we can create a string after all operations
    print("\n--- Testing string creation after all operations ---")
    let testString = String(repeating: "=", count: 60)
    print(testString)
    print("🎉 SUCCESS! Both approaches worked and string creation succeeded!")
    print(String(repeating: "=", count: 60))
}

// Run the test
do {
    try runTwoApproachesTest()
} catch {
    print("❌ Test failed: \(error)")
    exit(1)
}
