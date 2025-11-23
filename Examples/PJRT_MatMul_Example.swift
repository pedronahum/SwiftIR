//===-- PJRT_MatMul_Example.swift - PJRT Matrix Multiply --*- Swift -*-===//
//
// SwiftIR - PJRT Integration: StableHLO Matrix Multiplication
// Side-by-side demonstration of PJRT with pure StableHLO operations
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders
import SwiftIRStableHLO
import SwiftIRXLA
import Foundation

/// Matrix multiplication example using PJRT/XLA with StableHLO
///
/// This demonstrates:
/// 1. StableHLO dot_general operation (matrix multiplication)
/// 2. Multiple inputs with different shapes
/// 3. XLA's optimized matrix multiplication kernels
/// 4. Side-by-side comparison: naive Swift vs XLA-optimized execution
func runPJRTMatMulExample() throws {
    print(String(repeating: "=", count: 70))
    print("PJRT Matrix Multiplication Example")
    print("Side-by-Side: StableHLO vs Naive Swift")
    print(String(repeating: "=", count: 70))

    // MARK: - Initialize PJRT Client

    print("\nInitializing PJRT CPU Client...")


    let cpuClient = try PJRTClient(backend: .cpu)
    print("✅ PJRT CPU Client initialized")
    print("   Platform: \(cpuClient.platformName)")
    print("   Devices: \(cpuClient.addressableDevices.count)")

    guard let device = cpuClient.addressableDevices.first else {
        print("❌ No devices available")
        return
    }

    // MARK: - Define Large Matrices

    print("\n📊 Matrix Definitions - Large Scale for Performance Testing")

    // Matrix A: 512x1024 (512 rows, 1024 columns)
    let rowsA = 512
    let colsA = 1024
    var matrixA = [Float](repeating: 0.0, count: rowsA * colsA)

    // Initialize with random values for realistic computation
    for i in 0..<(rowsA * colsA) {
        matrixA[i] = Float.random(in: -1.0...1.0)
    }

    // Matrix B: 1024x256 (1024 rows, 256 columns)
    let rowsB = 1024
    let colsB = 256
    var matrixB = [Float](repeating: 0.0, count: rowsB * colsB)

    for i in 0..<(rowsB * colsB) {
        matrixB[i] = Float.random(in: -1.0...1.0)
    }

    print("   Matrix A: [\(rowsA), \(colsA)] - \(rowsA) rows, \(colsA) columns")
    print("   Matrix B: [\(rowsB), \(colsB)] - \(rowsB) rows, \(colsB) columns")
    print("   Result:   [\(rowsA), \(colsB)] - Matrix A × Matrix B")
    print("   Total operations: ~\(2 * rowsA * colsB * colsA / 1_000_000) million FLOPs")

    // MARK: - Part 1: XLA-Optimized Execution

    print("\n" + String(repeating: "=", count: 70))
    print("PART 1: XLA-Optimized StableHLO Execution")
    print(String(repeating: "=", count: 70))

    print("\n📤 Creating PJRT Buffers")

    var bufferA: PJRTBuffer?
    var bufferB: PJRTBuffer?

    matrixA.withUnsafeBytes { ptrA in
        do {
            bufferA = try cpuClient.createBuffer(
                data: ptrA.baseAddress!,
                shape: [rowsA, colsA],
                elementType: .f32,
                device: device
            )
            print("   ✅ Buffer A created: [\(rowsA), \(colsA)] f32")
        } catch {
            print("   ❌ Failed to create buffer A: \(error)")
        }
    }

    matrixB.withUnsafeBytes { ptrB in
        do {
            bufferB = try cpuClient.createBuffer(
                data: ptrB.baseAddress!,
                shape: [rowsB, colsB],
                elementType: .f32,
                device: device
            )
            print("   ✅ Buffer B created: [\(rowsB), \(colsB)] f32")
        } catch {
            print("   ❌ Failed to create buffer B: \(error)")
        }
    }

    guard let bufferA = bufferA, let bufferB = bufferB else {
        print("   ❌ Failed to create input buffers")
        return
    }

    print("\n🔨 Building StableHLO Program with Swift DSL")
    print("   Using declarative StableHLO builder (SwiftUI-like syntax)")
    print("   XLA will generate optimized BLAS kernels")

    // Build StableHLO program using Swift result builder DSL
    // This provides a type-safe, declarative way to construct StableHLO programs
    let tensorA = TensorType(shape: [rowsA, colsA])
    let tensorB = TensorType(shape: [rowsB, colsB])
    let tensorC = TensorType(shape: [rowsA, colsB])

    let module = StableHLOModule(name: "jit_matmul") {
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

    let mlirProgram = module.build()

    print("   ✅ StableHLO program built with declarative DSL")
    print("   Input shapes: [\(rowsA), \(colsA)] × [\(rowsB), \(colsB)]")
    print("   Output shape: [\(rowsA), \(colsB)]")

    let startCompile = Date()
    let executable = try cpuClient.compile(
        mlirModule: mlirProgram,
        devices: cpuClient.addressableDevices
    )
    let compileTime = Date().timeIntervalSince(startCompile)

    print("   ✅ Compilation successful!")
    print("   Compile time: \(String(format: "%.3f", compileTime * 1000)) ms")

    print("\n⚡ Executing Matrix Multiplication on XLA")

    let startExecute = Date()
    let outputs = try executable.execute(
        arguments: [bufferA, bufferB],
        device: device
    )
    let executeTime = Date().timeIntervalSince(startExecute)

    print("   ✅ Execution complete!")
    print("   Execute time: \(String(format: "%.3f", executeTime * 1000)) ms")

    guard let outputBuffer = outputs.first else {
        print("   ❌ No output buffer returned")
        return
    }

    print("\n📤 Reading XLA Results")

    let resultSize = rowsA * colsB
    var xlaResult = [Float](repeating: 0.0, count: resultSize)
    try xlaResult.withUnsafeMutableBytes { ptr in
        try outputBuffer.toHost(destination: ptr.baseAddress!)
    }

    print("   Output shape: \(outputBuffer.shape)")
    print("   Output type: \(outputBuffer.elementType)")
    print("   Result elements: \(resultSize)")

    // MARK: - Part 2: Naive Swift Execution

    print("\n" + String(repeating: "=", count: 70))
    print("PART 2: Naive Swift Matrix Multiplication (for comparison)")
    print(String(repeating: "=", count: 70))

    print("\n🔢 Computing with naive triple-nested loop...")
    print("   (This may take a while for large matrices...)")

    let startNaive = Date()
    var naiveResult = [Float](repeating: 0.0, count: resultSize)

    // C[i,k] = sum_j(A[i,j] * B[j,k])
    for i in 0..<rowsA {           // Rows of A
        for k in 0..<colsB {       // Columns of B
            var sum: Float = 0.0
            for j in 0..<colsA {   // Inner dimension (= rowsB)
                let a_val = matrixA[i * colsA + j]
                let b_val = matrixB[j * colsB + k]
                sum += a_val * b_val
            }
            naiveResult[i * colsB + k] = sum
        }
    }

    let naiveTime = Date().timeIntervalSince(startNaive)

    print("   ✅ Naive computation complete!")
    print("   Compute time: \(String(format: "%.3f", naiveTime * 1000)) ms")

    // MARK: - Results Comparison

    print("\n" + String(repeating: "=", count: 70))
    print("RESULTS COMPARISON")
    print(String(repeating: "=", count: 70))

    print("\n📊 Matrix Shapes:")
    print("   Input A:  [\(rowsA), \(colsA)]")
    print("   Input B:  [\(rowsB), \(colsB)]")
    print("   Output:   [\(rowsA), \(colsB)]")

    print("\n🎯 Sample Results (first 3×3 elements):")
    print("   XLA Result:")
    for i in 0..<min(3, rowsA) {
        var row = "     [ "
        for j in 0..<min(3, colsB) {
            row += String(format: "%7.3f ", xlaResult[i * colsB + j])
        }
        row += "]"
        print(row)
    }

    print("\n   Naive Swift Result:")
    for i in 0..<min(3, rowsA) {
        var row = "     [ "
        for j in 0..<min(3, colsB) {
            row += String(format: "%7.3f ", naiveResult[i * colsB + j])
        }
        row += "]"
        print(row)
    }

    // Verification
    let tolerance: Float = 0.01  // Slightly higher tolerance for floating point
    var allMatch = true
    var mismatchCount = 0
    for i in 0..<resultSize {
        if abs(xlaResult[i] - naiveResult[i]) > tolerance {
            allMatch = false
            mismatchCount += 1
            if mismatchCount <= 5 {  // Show only first 5 mismatches
                print("\n   ⚠️  Mismatch at index \(i): XLA=\(xlaResult[i]), Naive=\(naiveResult[i])")
            }
        }
    }

    if mismatchCount > 5 {
        print("\n   ⚠️  ... and \(mismatchCount - 5) more mismatches")
    }

    if allMatch {
        print("\n   ✅ VERIFICATION PASSED! XLA and naive results match!")
    } else {
        print("\n   ❌ VERIFICATION FAILED! Results do not match!")
    }

    // MARK: - Performance Analysis

    print("\n" + String(repeating: "=", count: 70))
    print("PERFORMANCE ANALYSIS")
    print(String(repeating: "=", count: 70))

    print("\n⏱️  Execution Times:")
    print("   XLA Compile:     \(String(format: "%8.3f", compileTime * 1000)) ms")
    print("   XLA Execute:     \(String(format: "%8.3f", executeTime * 1000)) ms")
    print("   XLA Total:       \(String(format: "%8.3f", (compileTime + executeTime) * 1000)) ms")
    print("   Naive Swift:     \(String(format: "%8.3f", naiveTime * 1000)) ms")

    if naiveTime > executeTime {
        let speedup = naiveTime / executeTime
        print("\n   🚀 XLA execution is \(String(format: "%.1f", speedup))x faster than naive Swift!")

        // Calculate GFLOPS
        let totalOps = Double(2 * rowsA * colsB * colsA)  // 2 ops per multiply-add
        let xlaGflops = totalOps / (executeTime * 1_000_000_000)
        let naiveGflops = totalOps / (naiveTime * 1_000_000_000)

        print("   XLA Performance:    \(String(format: "%.2f", xlaGflops)) GFLOPS")
        print("   Naive Performance:  \(String(format: "%.2f", naiveGflops)) GFLOPS")
    } else {
        print("\n   Note: For very small matrices, compilation overhead dominates")
        print("   XLA benefits become clear with larger matrices or repeated execution")
    }

    print("\n💡 XLA Optimizations Applied:")
    print("   • BLAS-optimized matrix multiplication kernels")
    print("   • SIMD vectorization (AVX/NEON instructions)")
    print("   • Cache-friendly memory access patterns")
    print("   • Loop unrolling and instruction pipelining")
    print("   • Potential multi-threading for larger matrices")

    print("\n📝 StableHLO Operation Details:")
    print("   Operation: stablehlo.dot_general")
    print("   Purpose: General dot product (includes matmul, dot, etc.)")
    print("   Contracting dims: [1] × [0]  (inner dimension)")
    print("   Batch dims: [] × []  (no batching)")
    print("   Input shapes: tensor<\(rowsA)x\(colsA)xf32> × tensor<\(rowsB)x\(colsB)xf32>")
    print("   Output shape: tensor<\(rowsA)x\(colsB)xf32>")

    print("\n🎓 Key Takeaways:")
    print("   1. StableHLO provides portable ML operations")
    print("   2. XLA compiles to optimized machine code")
    print("   3. Same StableHLO code runs on CPU/GPU/TPU")
    print("   4. PJRT provides runtime execution interface")
    print("   5. Significant performance gains for larger matrices")

    print("\n" + String(repeating: "=", count: 70))
    print("✨ PJRT Matrix Multiplication Example Complete!")
    print(String(repeating: "=", count: 70))
}

// MARK: - Entry Point

@main
struct PJRTMatMulExample {
    static func main() {
        do {
            try runPJRTMatMulExample()
        } catch {
            print("\n❌ Error: \(error)")
        }
    }
}
