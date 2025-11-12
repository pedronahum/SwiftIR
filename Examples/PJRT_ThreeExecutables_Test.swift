// Minimal test: Can we create 3 executables safely?
// This tests if multiple PJRTLoadedExecutable objects can coexist

import Foundation
import SwiftIRXLA

func runThreeExecutablesTest() throws {
    print("Testing: Can 3 executables coexist?")
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
        %arg0: tensor<2x3xf32>,
        %arg1: tensor<3x2xf32>
      ) -> tensor<2x2xf32> {
        %0 = stablehlo.dot_general %arg0, %arg1,
          contracting_dims = [1] x [0]
          : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }
    }
    """

    // Create input buffers ONCE
    let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
    let matrixB: [Float] = [10, 20, 30, 40, 50, 60]

    var bufferA: PJRTBuffer!
    var bufferB: PJRTBuffer!

    matrixA.withUnsafeBytes { ptr in
        bufferA = try? cpuClient.createBuffer(
            data: ptr.baseAddress!,
            shape: [2, 3],
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

    print("\n✅ Input buffers created")

    // Test 1: Create executable1, execute, read result, destroy IMMEDIATELY
    print("\n--- Test 1: Executable 1 ---")
    let executable1 = try cpuClient.compile(mlirModule: mlirProgram, devices: [device])
    print("✅ Executable 1 compiled")

    let outputs1 = try executable1.execute(arguments: [bufferA, bufferB], device: device)
    print("✅ Executable 1 executed")

    guard let outputBuffer1 = outputs1.first else {
        print("❌ No output from executable 1")
        return
    }

    var result1 = [Float](repeating: 0.0, count: 4)
    try result1.withUnsafeMutableBytes { ptr in
        try outputBuffer1.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 1: \(result1.prefix(2))")

    // Explicit cleanup
    outputBuffer1.destroy()
    executable1.destroy()
    print("✅ Executable 1 destroyed")

    // Test 2: Create executable2, execute, read result, destroy IMMEDIATELY
    print("\n--- Test 2: Executable 2 ---")
    let executable2 = try cpuClient.compile(mlirModule: mlirProgram, devices: [device])
    print("✅ Executable 2 compiled")

    let outputs2 = try executable2.execute(arguments: [bufferA, bufferB], device: device)
    print("✅ Executable 2 executed")

    guard let outputBuffer2 = outputs2.first else {
        print("❌ No output from executable 2")
        return
    }

    var result2 = [Float](repeating: 0.0, count: 4)
    try result2.withUnsafeMutableBytes { ptr in
        try outputBuffer2.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 2: \(result2.prefix(2))")

    // Explicit cleanup
    outputBuffer2.destroy()
    executable2.destroy()
    print("✅ Executable 2 destroyed")

    // Test 3: Create executable3, execute, read result, destroy IMMEDIATELY
    print("\n--- Test 3: Executable 3 ---")
    let executable3 = try cpuClient.compile(mlirModule: mlirProgram, devices: [device])
    print("✅ Executable 3 compiled")

    let outputs3 = try executable3.execute(arguments: [bufferA, bufferB], device: device)
    print("✅ Executable 3 executed")

    guard let outputBuffer3 = outputs3.first else {
        print("❌ No output from executable 3")
        return
    }

    var result3 = [Float](repeating: 0.0, count: 4)
    print("About to read result3...")
    try result3.withUnsafeMutableBytes { ptr in
        try outputBuffer3.toHost(destination: ptr.baseAddress!)
    }
    print("✅ Result 3: \(result3.prefix(2))")

    // Explicit cleanup
    outputBuffer3.destroy()
    executable3.destroy()
    print("✅ Executable 3 destroyed")

    print("\n" + String(repeating: "=", count: 60))
    print("🎉 SUCCESS! All 3 executables worked!")
    print(String(repeating: "=", count: 60))
}

// Run the test
do {
    try runThreeExecutablesTest()
} catch {
    print("❌ Test failed: \(error)")
    exit(1)
}
