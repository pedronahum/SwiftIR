//===-- main.swift - Run JAX's StableHLO via SwiftIR PJRT --*- Swift -*-===//
//
// This benchmark runs JAX's exact StableHLO through SwiftIR's PJRT execution
// to isolate whether the performance difference is in:
// 1. The generated IR (JAX vs SwiftIR)
// 2. The execution path (JAX's XLA client vs SwiftIR's PJRT bindings)
//
// If running JAX's MLIR through SwiftIR is as slow as SwiftIR's own MLIR,
// then the bottleneck is in SwiftIR's execution path.
//
// If running JAX's MLIR through SwiftIR is as fast as JAX,
// then the bottleneck is in SwiftIR's IR generation.
//
//===----------------------------------------------------------------===//

import SwiftIR
import SwiftIRXLA
import Foundation

// MARK: - JAX's StableHLO (copied from jax-python/hlo_dumps/black_scholes_fwd_grad_stablehlo.txt)

let jaxStableHLO_1000 = """
module @jit_price_and_delta_explicit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1000xf32>) -> (tensor<1000xf32>, tensor<1000xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.sqrt %cst : tensor<f32>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<f32>
    %1 = stablehlo.multiply %cst_0, %0 : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %3 = stablehlo.divide %arg0, %2 : tensor<1000xf32>
    %4 = stablehlo.log %3 : tensor<1000xf32>
    %cst_2 = stablehlo.constant dense<7.000000e-02> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %6 = stablehlo.add %4, %5 : tensor<1000xf32>
    %7 = stablehlo.convert %1 : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %9 = stablehlo.divide %6, %8 : tensor<1000xf32>
    %10 = stablehlo.convert %1 : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %12 = stablehlo.subtract %9, %11 : tensor<1000xf32>
    %cst_3 = stablehlo.constant dense<-5.000000e-02> : tensor<f32>
    %13 = stablehlo.exponential %cst_3 : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<1000xf32>
    %16 = stablehlo.negate %15 : tensor<1000xf32>
    %17 = stablehlo.exponential %16 : tensor<1000xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %19 = stablehlo.add %18, %17 : tensor<1000xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %21 = stablehlo.divide %20, %19 : tensor<1000xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<1000xf32>
    %24 = stablehlo.multiply %21, %23 : tensor<1000xf32>
    %cst_8 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %26 = stablehlo.multiply %12, %25 : tensor<1000xf32>
    %27 = stablehlo.negate %26 : tensor<1000xf32>
    %28 = stablehlo.exponential %27 : tensor<1000xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %30 = stablehlo.add %29, %28 : tensor<1000xf32>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %32 = stablehlo.divide %31, %30 : tensor<1000xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %34 = stablehlo.subtract %33, %32 : tensor<1000xf32>
    %35 = stablehlo.multiply %32, %34 : tensor<1000xf32>
    %36 = stablehlo.multiply %arg0, %21 : tensor<1000xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %37 = stablehlo.multiply %cst_12, %13 : tensor<f32>
    %38 = stablehlo.convert %37 : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %40 = stablehlo.multiply %39, %32 : tensor<1000xf32>
    %41 = stablehlo.subtract %36, %40 : tensor<1000xf32>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %43 = stablehlo.negate %42 : tensor<1000xf32>
    %44 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %45 = stablehlo.multiply %44, %43 : tensor<1000xf32>
    %46 = stablehlo.multiply %arg0, %42 : tensor<1000xf32>
    %47 = stablehlo.multiply %42, %21 : tensor<1000xf32>
    %48 = stablehlo.multiply %45, %35 : tensor<1000xf32>
    %cst_14 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<1000xf32>
    %51 = stablehlo.multiply %46, %24 : tensor<1000xf32>
    %cst_15 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<1000xf32>
    %54 = stablehlo.add %50, %53 : tensor<1000xf32>
    %55 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %56 = stablehlo.divide %54, %55 : tensor<1000xf32>
    %57 = stablehlo.divide %56, %3 : tensor<1000xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %59 = stablehlo.divide %57, %58 : tensor<1000xf32>
    %60 = stablehlo.add %47, %59 : tensor<1000xf32>
    return %41, %60 : tensor<1000xf32>, tensor<1000xf32>
  }
}
"""

// Function to generate JAX StableHLO for different batch sizes
func generateJAXStableHLO(batchSize: Int) -> String {
    return jaxStableHLO_1000.replacingOccurrences(of: "1000", with: "\(batchSize)")
}

// MARK: - Timing

func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - Benchmark

struct BenchmarkResult {
    let batchSize: Int
    let compileTimeMs: Double
    let executeTimeUs: Double
    let minExecuteTimeUs: Double
    let throughput: Double
}

func runBenchmark(batchSize: Int, trials: Int) throws -> BenchmarkResult {
    let mlir = generateJAXStableHLO(batchSize: batchSize)

    // Compile
    let compileStart = currentTime()
    let runtime = try PJRTBackedRuntime(backend: .cpu)
    let executable = try runtime.compile(mlir)
    let compileTime = (currentTime() - compileStart) * 1000  // ms

    // Create input data
    var spots = [Float](repeating: 0, count: batchSize)
    for i in 0..<batchSize {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }

    // Warmup
    for _ in 0..<10 {
        let _ = try executable.execute([spots])
    }

    // Benchmark
    var times: [Double] = []

    for _ in 0..<trials {
        let start = currentTime()
        let _ = try executable.execute([spots])
        times.append(currentTime() - start)
    }

    let avgTime = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTime = times.min()! * 1_000_000
    let throughput = Double(batchSize) / (avgTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        compileTimeMs: compileTime,
        executeTimeUs: avgTime,
        minExecuteTimeUs: minTime,
        throughput: throughput
    )
}

// MARK: - Main

@main
struct JAXMLIRBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     JAX StableHLO via SwiftIR PJRT Benchmark                                 ║
        ║                                                                              ║
        ║     Running JAX's exact StableHLO through SwiftIR's PJRT execution           ║
        ║     to isolate whether bottleneck is in IR or execution path                 ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        do {
            let trials = 100
            let batchSizes = [1000, 10000, 100000]

            var results: [BenchmarkResult] = []

            for size in batchSizes {
                print("  Running batch size: \(size)...")
                let result = try runBenchmark(batchSize: size, trials: trials)
                results.append(result)
                print("    Compile:    \(String(format: "%.1f", result.compileTimeMs)) ms")
                print("    Execute:    \(String(format: "%.1f", result.executeTimeUs)) μs")
                print("    Throughput: \(String(format: "%.2f", result.throughput)) K/s")
                print()
            }

            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  RESULTS: JAX StableHLO via SwiftIR PJRT                                     ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

            """)

            print("  ┌────────────┬──────────────┬──────────────┬──────────────────┐")
            print("  │ Batch Size │ Compile (ms) │ Execute (μs) │   Throughput     │")
            print("  ├────────────┼──────────────┼──────────────┼──────────────────┤")
            for result in results {
                let batchStr = String(format: "%10d", result.batchSize)
                let compileStr = String(format: "%10.1f", result.compileTimeMs)
                let execStr = String(format: "%10.1f", result.executeTimeUs)
                let throughputStr = String(format: "%12.2f K/s", result.throughput)
                print("  │ \(batchStr) │ \(compileStr) │ \(execStr) │ \(throughputStr) │")
            }
            print("  └────────────┴──────────────┴──────────────┴──────────────────┘")
            print()

            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  COMPARISON                                                                  ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

              If JAX MLIR via SwiftIR PJRT is:

              - SLOW (~2,500μs at 100K): Bottleneck is SwiftIR's PJRT execution path
                → Need to optimize data transfer / PJRT bindings

              - FAST (~730μs at 100K): Bottleneck is SwiftIR's IR generation
                → Need to optimize DifferentiableTracer / code generation

              Reference times at 100K batch:
              - JAX native:       ~729 μs
              - SwiftIR native:   ~2,486 μs

            """)

        } catch {
            print("ERROR: \(error)")
        }
    }
}
