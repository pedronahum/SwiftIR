//===-- main.swift - JAX StableHLO executed in SwiftIR PJRT --*- Swift -*-===//
//
// JAX StableHLO in SwiftIR PJRT Benchmark
//
// This test isolates the performance gap between JAX and SwiftIR by using
// JAX's EXACT StableHLO IR and executing it through SwiftIR's PJRT path.
//
// If SwiftIR matches JAX performance with identical IR, the gap is due to
// IR generation differences. If SwiftIR is still slower, the gap is due to
// PJRT execution overhead (buffer management, Swift↔C bridging, etc.)
//
//===--------------------------------------------------------------------===//

import SwiftIR
import SwiftIRXLA
import Foundation

// MARK: - Timing

func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - JAX's StableHLO (copied from JAX output)
// This is the EXACT IR that JAX generates for Black-Scholes fwd+grad

/// Generate JAX-style StableHLO for a given batch size
/// This replicates JAX's generated IR with pre-computed constants
func generateJAXStableHLO(batchSize: Int) -> String {
    // JAX pre-computes: rate + 0.5*vol^2 = 0.05 + 0.5*0.04 = 0.07
    // JAX pre-computes: strike * exp(-rate*maturity) = 100 * exp(-0.05) = 95.1229477
    // JAX computes exp(-0.05) as scalar first
    return """
    module @jax_stablehlo_in_swift attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<\(batchSize)xf32>) -> (tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>) {
        %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %0 = stablehlo.sqrt %cst : tensor<f32>
        %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<f32>
        %1 = stablehlo.multiply %cst_0, %0 : tensor<f32>
        %cst_1 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
        %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %3 = stablehlo.divide %arg0, %2 : tensor<\(batchSize)xf32>
        %4 = stablehlo.log %3 : tensor<\(batchSize)xf32>
        %cst_2 = stablehlo.constant dense<7.000000e-02> : tensor<f32>
        %5 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %6 = stablehlo.add %4, %5 : tensor<\(batchSize)xf32>
        %7 = stablehlo.convert %1 : tensor<f32>
        %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %9 = stablehlo.divide %6, %8 : tensor<\(batchSize)xf32>
        %10 = stablehlo.convert %1 : tensor<f32>
        %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %12 = stablehlo.subtract %9, %11 : tensor<\(batchSize)xf32>
        %cst_3 = stablehlo.constant dense<-5.000000e-02> : tensor<f32>
        %13 = stablehlo.exponential %cst_3 : tensor<f32>
        %cst_4 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
        %14 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %15 = stablehlo.multiply %9, %14 : tensor<\(batchSize)xf32>
        %16 = stablehlo.negate %15 : tensor<\(batchSize)xf32>
        %17 = stablehlo.exponential %16 : tensor<\(batchSize)xf32>
        %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %18 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %19 = stablehlo.add %18, %17 : tensor<\(batchSize)xf32>
        %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %20 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %21 = stablehlo.divide %20, %19 : tensor<\(batchSize)xf32>
        %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %22 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %23 = stablehlo.subtract %22, %21 : tensor<\(batchSize)xf32>
        %24 = stablehlo.multiply %21, %23 : tensor<\(batchSize)xf32>
        %cst_8 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
        %25 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %26 = stablehlo.multiply %12, %25 : tensor<\(batchSize)xf32>
        %27 = stablehlo.negate %26 : tensor<\(batchSize)xf32>
        %28 = stablehlo.exponential %27 : tensor<\(batchSize)xf32>
        %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %29 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %30 = stablehlo.add %29, %28 : tensor<\(batchSize)xf32>
        %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %31 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %32 = stablehlo.divide %31, %30 : tensor<\(batchSize)xf32>
        %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %33 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %34 = stablehlo.subtract %33, %32 : tensor<\(batchSize)xf32>
        %35 = stablehlo.multiply %32, %34 : tensor<\(batchSize)xf32>
        %36 = stablehlo.multiply %arg0, %21 : tensor<\(batchSize)xf32>
        %cst_12 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
        %37 = stablehlo.multiply %cst_12, %13 : tensor<f32>
        %38 = stablehlo.convert %37 : tensor<f32>
        %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %40 = stablehlo.multiply %39, %32 : tensor<\(batchSize)xf32>
        %41 = stablehlo.subtract %36, %40 : tensor<\(batchSize)xf32>
        %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %42 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %43 = stablehlo.negate %42 : tensor<\(batchSize)xf32>
        %44 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %45 = stablehlo.multiply %44, %43 : tensor<\(batchSize)xf32>
        %46 = stablehlo.multiply %arg0, %42 : tensor<\(batchSize)xf32>
        %47 = stablehlo.multiply %42, %21 : tensor<\(batchSize)xf32>
        %48 = stablehlo.multiply %45, %35 : tensor<\(batchSize)xf32>
        %cst_14 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
        %49 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %50 = stablehlo.multiply %48, %49 : tensor<\(batchSize)xf32>
        %51 = stablehlo.multiply %46, %24 : tensor<\(batchSize)xf32>
        %cst_15 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
        %52 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %53 = stablehlo.multiply %51, %52 : tensor<\(batchSize)xf32>
        %54 = stablehlo.add %50, %53 : tensor<\(batchSize)xf32>
        %55 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %56 = stablehlo.divide %54, %55 : tensor<\(batchSize)xf32>
        %57 = stablehlo.divide %56, %3 : tensor<\(batchSize)xf32>
        %cst_16 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
        %58 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %59 = stablehlo.divide %57, %58 : tensor<\(batchSize)xf32>
        %60 = stablehlo.add %47, %59 : tensor<\(batchSize)xf32>
        return %41, %60 : tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>
      }
    }
    """
}

// MARK: - Benchmark Result

struct BenchmarkResult {
    let batchSize: Int
    let compileTimeMs: Double
    let avgTimeUs: Double
    let minTimeUs: Double
    let stdTimeUs: Double
    let throughputK: Double
    let samplePrices: [Float]
    let sampleDeltas: [Float]
}

// MARK: - Run Benchmark

func runBenchmark(batchSize: Int, trials: Int) throws -> BenchmarkResult {
    // Generate JAX-style StableHLO
    let stablehlo = generateJAXStableHLO(batchSize: batchSize)

    // Create PJRT client
    let client = try PJRTClient(backend: .cpu)

    // Measure compile time
    let compileStart = currentTime()
    let executable = try client.compile(mlirModule: stablehlo)
    let compileTime = (currentTime() - compileStart) * 1000

    guard let device = client.defaultDevice else {
        throw PJRTError.noDeviceAvailable
    }

    // Create input data (same as JAX benchmark)
    var spots = [Float](repeating: 0, count: batchSize)
    for i in 0..<batchSize {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }

    // Warmup
    for _ in 0..<10 {
        let inputBuffer = try spots.withUnsafeBytes { ptr in
            try client.createBuffer(
                data: ptr.baseAddress!,
                shape: [batchSize],
                elementType: .f32,
                device: device,
                semantics: .zeroCopy
            )
        }
        let outputs = try executable.execute(arguments: [inputBuffer])
        // Just discard
        _ = outputs
    }

    // Benchmark
    var times: [Double] = []
    var samplePrices: [Float] = []
    var sampleDeltas: [Float] = []

    for i in 0..<trials {
        // Create input buffer
        let inputBuffer = try spots.withUnsafeBytes { ptr in
            try client.createBuffer(
                data: ptr.baseAddress!,
                shape: [batchSize],
                elementType: .f32,
                device: device,
                semantics: .zeroCopy
            )
        }

        let start = currentTime()
        let outputs = try executable.execute(arguments: [inputBuffer])

        // Read results back using async transfers for better performance
        var prices = [Float](repeating: 0, count: batchSize)
        var deltas = [Float](repeating: 0, count: batchSize)

        // Initiate both transfers asynchronously
        let event0 = try prices.withUnsafeMutableBytes { ptr in
            try outputs[0].toHostAsync(destination: ptr.baseAddress!)
        }
        let event1 = try deltas.withUnsafeMutableBytes { ptr in
            try outputs[1].toHostAsync(destination: ptr.baseAddress!)
        }

        // Await both transfers
        try event0?.awaitAndDestroy()
        try event1?.awaitAndDestroy()

        let elapsed = currentTime() - start
        times.append(elapsed)

        if i == 0 {
            samplePrices = Array(prices.prefix(5))
            sampleDeltas = Array(deltas.prefix(5))
        }
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        compileTimeMs: compileTime,
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK,
        samplePrices: samplePrices,
        sampleDeltas: sampleDeltas
    )
}

// MARK: - Main

@main
struct JAXStableHLOBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     JAX StableHLO in SwiftIR PJRT Benchmark                                  ║
        ║                                                                              ║
        ║     Tests if SwiftIR's PJRT path matches JAX performance                     ║
        ║     when using IDENTICAL StableHLO IR from JAX                               ║
        ║                                                                              ║
        ║     If SwiftIR matches JAX: Gap is from IR generation                        ║
        ║     If SwiftIR is slower: Gap is from PJRT execution overhead                ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        do {
            let trials = 100
            let batchSizes = [100, 1000, 10000, 100000, 1000000]

            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Running JAX StableHLO through SwiftIR PJRT                  ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()
            print("  Trials per benchmark: \(trials)")
            print()

            var results: [BenchmarkResult] = []

            for size in batchSizes {
                print("  Running batch size: \(size)...")
                let result = try runBenchmark(batchSize: size, trials: trials)
                results.append(result)
                print("    Compile:    \(String(format: "%.1f", result.compileTimeMs)) ms")
                print("    Execute:    \(String(format: "%.1f", result.avgTimeUs)) μs (±\(String(format: "%.1f", result.stdTimeUs)))")
                print("    Throughput: \(String(format: "%.2f", result.throughputK)) K/s")
                print()
            }

            // Print execution time table
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  EXECUTION TIME (JAX StableHLO via SwiftIR PJRT)                              ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

            """)

            print("  ┌────────────┬──────────────────┬──────────────────┬──────────────────┐")
            print("  │ Batch Size │   Avg Time (μs)  │   Min Time (μs)  │   Throughput     │")
            print("  ├────────────┼──────────────────┼──────────────────┼──────────────────┤")
            for result in results {
                let batchStr = String(format: "%10d", result.batchSize)
                let timeStr = String(format: "%14.1f", result.avgTimeUs)
                let minTimeStr = String(format: "%14.1f", result.minTimeUs)
                let throughputStr = String(format: "%12.2f K/s", result.throughputK)
                print("  │ \(batchStr) │ \(timeStr) │ \(minTimeStr) │ \(throughputStr) │")
            }
            print("  └────────────┴──────────────────┴──────────────────┴──────────────────┘")
            print()

            // Sample results
            if let first = results.first {
                print("""
                ╔══════════════════════════════════════════════════════════════════════════════╗
                ║  SAMPLE RESULTS (batch size \(first.batchSize))                                             ║
                ╚══════════════════════════════════════════════════════════════════════════════╝

                  Spot prices:     80.0, 80.4, 80.8, 81.2, 81.6
                  Call prices:     \(first.samplePrices.map { String(format: "%.4f", $0) }.joined(separator: ", "))
                  Deltas (∂C/∂S):  \(first.sampleDeltas.map { String(format: "%.4f", $0) }.joined(separator: ", "))

                """)
            }

            // Comparison at 100K
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  GAP ANALYSIS (at batch size 100,000)                                         ║
            ╚══════════════════════════════════════════════════════════════════════════════╝
            """)

            if let r = results.first(where: { $0.batchSize == 100000 }) {
                print()
                print("  JAX StableHLO via SwiftIR PJRT:")
                print("    Execute time:    \(String(format: "%.1f", r.avgTimeUs)) μs")
                print("    Throughput:      \(String(format: "%.0f", r.throughputK)) K options/sec")
                print()
                print("  Compare with JAX native execution (~793 μs):")
                let ratio = r.avgTimeUs / 793.0
                print("    Ratio:           \(String(format: "%.2f", ratio))x")
                print()
                print("  Interpretation:")
                if ratio < 1.1 {
                    print("    ✓ SwiftIR PJRT matches JAX! Gap is from IR generation differences.")
                } else if ratio < 1.3 {
                    print("    ~ SwiftIR PJRT is ~\(String(format: "%.0f", (ratio - 1) * 100))% slower than JAX")
                    print("      Some gap is from PJRT execution overhead (buffer management, etc.)")
                } else {
                    print("    ✗ SwiftIR PJRT is significantly slower (\(String(format: "%.1f", ratio))x)")
                    print("      Major gap is from PJRT execution overhead, not IR differences.")
                }
                print()
                print("  Compare with SwiftIR-Traced (~1,006 μs):")
                print("    If this is faster: SwiftIR IR generation has optimization potential")
                print("    If similar: IR differences are NOT the primary cause of the gap")
            }
            print()

        } catch {
            print("ERROR: \(error)")
        }
    }
}
