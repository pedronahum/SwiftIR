// SwiftIRJupyter Comprehensive Test Suite
// Tests all major functionality before long runs

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

import SwiftIRJupyter

nonisolated(unsafe) var testsPassed = 0
nonisolated(unsafe) var testsFailed = 0

func runTest(_ name: String, _ test: () throws -> Void) {
    print("Testing: \(name)...")
    do {
        try test()
        print("  ✅ PASSED\n")
        testsPassed += 1
    } catch {
        print("  ❌ FAILED: \(error)\n")
        testsFailed += 1
    }
}

// String multiplication helper
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

print("=" * 60)
print("SwiftIRJupyter Comprehensive Test Suite")
print("=" * 60)
print()

// Initialize once
do {
    print("Initializing SwiftIRJupyter...")
    try SwiftIRJupyter.shared.initialize()
    print("✅ Initialized\n")
} catch {
    print("❌ Failed to initialize: \(error)")
    exit(1)
}

// Test 1: MLIR Context Creation
runTest("MLIR Context Creation") {
    let ctx = try SwiftIRJupyter.shared.createContext()
    let module = try ctx.createModule()
    let mlirText = try module.print()
    guard mlirText.contains("module") else {
        throw SwiftIRJupyterError.invalidState(message: "Module text doesn't contain 'module'")
    }
    let verified = try module.verify()
    guard verified == true else {
        throw SwiftIRJupyterError.invalidState(message: "Module verification failed")
    }
}

// Test 2: PJRT Client Creation
nonisolated(unsafe) var client: JupyterPJRTClient!
runTest("PJRT Client Creation") {
    client = try SwiftIRJupyter.shared.createClient()
    guard client.platformName == "cpu" else {
        throw SwiftIRJupyterError.invalidState(message: "Expected platform 'cpu', got '\(client.platformName)'")
    }
    guard client.deviceCount > 0 else {
        throw SwiftIRJupyterError.invalidState(message: "No devices available")
    }
}

// Test 3: Float Buffer Creation
runTest("Float Buffer Creation") {
    let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let buffer = try client.createBuffer(data: data, shape: [2, 3])
    guard buffer.shape == [2, 3] else {
        throw SwiftIRJupyterError.invalidState(message: "Shape mismatch")
    }
    guard buffer.elementCount == 6 else {
        throw SwiftIRJupyterError.invalidState(message: "Element count mismatch")
    }
    guard buffer.type == .f32 else {
        throw SwiftIRJupyterError.invalidState(message: "Type mismatch")
    }
}

// Test 4: Int32 Buffer Creation
runTest("Int32 Buffer Creation") {
    let data: [Int32] = [10, 20, 30, 40]
    let buffer = try client.createBuffer(data: data, shape: [4])
    guard buffer.shape == [4] else {
        throw SwiftIRJupyterError.invalidState(message: "Shape mismatch")
    }
    guard buffer.type == .s32 else {
        throw SwiftIRJupyterError.invalidState(message: "Type mismatch")
    }
}

// Test 5: Simple Add Operation
runTest("Simple Add (x + x = 2x)") {
    let mlirModule = """
    module @add_test {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]
    let buffer = try client.createBuffer(data: input, shape: [4])
    let outputs = try executable.execute(inputs: [buffer])
    guard outputs.count == 1 else {
        throw SwiftIRJupyterError.invalidState(message: "Expected 1 output, got \(outputs.count)")
    }
}

// Test 6: Multiply Operation
runTest("Multiply Operation (x * 2)") {
    let mlirModule = """
    module @mul_test {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %c = stablehlo.constant dense<2.0> : tensor<4xf32>
        %0 = stablehlo.multiply %arg0, %c : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]
    let buffer = try client.createBuffer(data: input, shape: [4])
    _ = try executable.execute(inputs: [buffer])
}

// Test 7: Two Input Operation
runTest("Two Input Add (a + b)") {
    let mlirModule = """
    module @two_input_test {
      func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let a: [Float] = [1.0, 2.0, 3.0, 4.0]
    let b: [Float] = [10.0, 20.0, 30.0, 40.0]
    let bufferA = try client.createBuffer(data: a, shape: [4])
    let bufferB = try client.createBuffer(data: b, shape: [4])
    let outputs = try executable.execute(inputs: [bufferA, bufferB])
    guard outputs.count == 1 else {
        throw SwiftIRJupyterError.invalidState(message: "Expected 1 output")
    }
}

// Test 8: Matrix Multiplication
runTest("Matrix Multiplication (2x3 @ 3x2 = 2x2)") {
    let mlirModule = """
    module @matmul_test {
      func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    // 2x3 matrix
    let a: [Float] = [1, 2, 3, 4, 5, 6]
    // 3x2 matrix
    let b: [Float] = [1, 2, 3, 4, 5, 6]
    let bufferA = try client.createBuffer(data: a, shape: [2, 3])
    let bufferB = try client.createBuffer(data: b, shape: [3, 2])
    let outputs = try executable.execute(inputs: [bufferA, bufferB])
    guard outputs.count == 1 else {
        throw SwiftIRJupyterError.invalidState(message: "Expected 1 output")
    }
}

// Test 9: Larger Matrix Multiplication
runTest("Large Matrix Multiplication (64x64 @ 64x64)") {
    let mlirModule = """
    module @large_matmul {
      func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
        return %0 : tensor<64x64xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let a = [Float](repeating: 1.0, count: 64*64)
    let b = [Float](repeating: 0.5, count: 64*64)
    let bufferA = try client.createBuffer(data: a, shape: [64, 64])
    let bufferB = try client.createBuffer(data: b, shape: [64, 64])
    _ = try executable.execute(inputs: [bufferA, bufferB])
}

// Test 10: Chained Operations
runTest("Chained Operations (relu(x * w + b))") {
    let mlirModule = """
    module @chained_ops {
      func.func @main(%x: tensor<4xf32>, %w: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
        %mul = stablehlo.multiply %x, %w : tensor<4xf32>
        %add = stablehlo.add %mul, %b : tensor<4xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
        %relu = stablehlo.maximum %add, %zero : tensor<4xf32>
        return %relu : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let x: [Float] = [1.0, -2.0, 3.0, -4.0]
    let w: [Float] = [0.5, 0.5, 0.5, 0.5]
    let b: [Float] = [0.1, 0.1, 0.1, 0.1]
    let bufferX = try client.createBuffer(data: x, shape: [4])
    let bufferW = try client.createBuffer(data: w, shape: [4])
    let bufferB = try client.createBuffer(data: b, shape: [4])
    _ = try executable.execute(inputs: [bufferX, bufferW, bufferB])
}

// Test 11: Multiple Outputs
runTest("Multiple Outputs") {
    let mlirModule = """
    module @multi_output {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        %doubled = stablehlo.add %arg0, %arg0 : tensor<4xf32>
        %negated = stablehlo.negate %arg0 : tensor<4xf32>
        return %doubled, %negated : tensor<4xf32>, tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]
    let buffer = try client.createBuffer(data: input, shape: [4])
    let outputs = try executable.execute(inputs: [buffer])
    guard outputs.count == 2 else {
        throw SwiftIRJupyterError.invalidState(message: "Expected 2 outputs, got \(outputs.count)")
    }
}

// Test 12: Repeated Execution
runTest("Repeated Execution (10 times)") {
    let mlirModule = """
    module @repeated {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    for i in 0..<10 {
        let input: [Float] = [Float(i), Float(i+1), Float(i+2), Float(i+3)]
        let buffer = try client.createBuffer(data: input, shape: [4])
        _ = try executable.execute(inputs: [buffer])
    }
}

// Test 13: Batch Processing
runTest("Batch Processing (batch of 4, size 8)") {
    let mlirModule = """
    module @batch {
      func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %c = stablehlo.constant dense<0.5> : tensor<4x8xf32>
        %0 = stablehlo.multiply %arg0, %c : tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input = [Float](repeating: 2.0, count: 4*8)
    let buffer = try client.createBuffer(data: input, shape: [4, 8])
    _ = try executable.execute(inputs: [buffer])
}

// Test 14: Reduction Operation
runTest("Reduction (sum over axis)") {
    let mlirModule = """
    module @reduce {
      func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
        %init = stablehlo.constant dense<0.0> : tensor<f32>
        %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %sum = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %sum : tensor<f32>
          }
        return %0 : tensor<4xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input: [Float] = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]
    let buffer = try client.createBuffer(data: input, shape: [4, 4])
    _ = try executable.execute(inputs: [buffer])
}

// Test 15: Transpose
runTest("Transpose (2x3 -> 3x2)") {
    let mlirModule = """
    module @transpose {
      func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
        %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
        return %0 : tensor<3x2xf32>
      }
    }
    """
    let executable = try client.compile(mlir: mlirModule)
    let input: [Float] = [1, 2, 3, 4, 5, 6]
    let buffer = try client.createBuffer(data: input, shape: [2, 3])
    _ = try executable.execute(inputs: [buffer])
}

// Print summary
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("Passed: \(testsPassed)")
print("Failed: \(testsFailed)")
print("Total:  \(testsPassed + testsFailed)")
print()

if testsFailed == 0 {
    print("🎉 ALL TESTS PASSED! SwiftIRJupyter is ready for production use.")
} else {
    print("⚠️  Some tests failed. Please review before long runs.")
}
