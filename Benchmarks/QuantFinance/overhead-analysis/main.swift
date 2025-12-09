//===-- main.swift - SwiftIR Overhead Analysis -----------*- Swift -*-===//
//
// This benchmark isolates different sources of overhead in SwiftIR's PJRT path
// compared to JAX:
//
// 1. Execution-only: Reuse all buffers, measure pure XLA execution
// 2. Input creation: Measure overhead of creating input buffers
// 3. Output reading: Measure overhead of reading outputs to host
// 4. Full path: Measure complete execute-and-read cycle
//
//===--------------------------------------------------------------------===//

import SwiftIRXLA
import Foundation

// MARK: - Timing

@inline(__always)
func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - JAX StableHLO

func generateJAXStableHLO(batchSize: Int) -> String {
    return """
    module @overhead_analysis attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
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
        %20 = stablehlo.divide %18, %19 : tensor<\(batchSize)xf32>
        %21 = stablehlo.subtract %18, %20 : tensor<\(batchSize)xf32>
        %22 = stablehlo.multiply %20, %21 : tensor<\(batchSize)xf32>
        %23 = stablehlo.multiply %12, %14 : tensor<\(batchSize)xf32>
        %24 = stablehlo.negate %23 : tensor<\(batchSize)xf32>
        %25 = stablehlo.exponential %24 : tensor<\(batchSize)xf32>
        %26 = stablehlo.add %18, %25 : tensor<\(batchSize)xf32>
        %27 = stablehlo.divide %18, %26 : tensor<\(batchSize)xf32>
        %28 = stablehlo.subtract %18, %27 : tensor<\(batchSize)xf32>
        %29 = stablehlo.multiply %27, %28 : tensor<\(batchSize)xf32>
        %30 = stablehlo.multiply %arg0, %20 : tensor<\(batchSize)xf32>
        %cst_6 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
        %31 = stablehlo.multiply %cst_6, %13 : tensor<f32>
        %32 = stablehlo.convert %31 : tensor<f32>
        %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<\(batchSize)xf32>
        %34 = stablehlo.multiply %33, %27 : tensor<\(batchSize)xf32>
        %35 = stablehlo.subtract %30, %34 : tensor<\(batchSize)xf32>
        %36 = stablehlo.negate %18 : tensor<\(batchSize)xf32>
        %37 = stablehlo.multiply %33, %36 : tensor<\(batchSize)xf32>
        %38 = stablehlo.multiply %arg0, %18 : tensor<\(batchSize)xf32>
        %39 = stablehlo.multiply %18, %20 : tensor<\(batchSize)xf32>
        %40 = stablehlo.multiply %37, %29 : tensor<\(batchSize)xf32>
        %41 = stablehlo.multiply %40, %14 : tensor<\(batchSize)xf32>
        %42 = stablehlo.multiply %38, %22 : tensor<\(batchSize)xf32>
        %43 = stablehlo.multiply %42, %14 : tensor<\(batchSize)xf32>
        %44 = stablehlo.add %41, %43 : tensor<\(batchSize)xf32>
        %45 = stablehlo.divide %44, %8 : tensor<\(batchSize)xf32>
        %46 = stablehlo.divide %45, %3 : tensor<\(batchSize)xf32>
        %47 = stablehlo.divide %46, %2 : tensor<\(batchSize)xf32>
        %48 = stablehlo.add %39, %47 : tensor<\(batchSize)xf32>
        return %35, %48 : tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>
      }
    }
    """
}

// MARK: - Main

@main
struct OverheadAnalysis {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     SwiftIR PJRT Overhead Analysis                                           ║
        ║                                                                              ║
        ║     Isolating sources of overhead vs JAX:                                    ║
        ║     1. Pure XLA execution (reuse buffers)                                    ║
        ║     2. Input buffer creation overhead                                        ║
        ║     3. Output D2H transfer overhead                                          ║
        ║     4. Full execute-and-read cycle                                           ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        do {
            let trials = 500  // More trials for accurate timing
            let batchSize = 100000

            print("  Batch size: \(batchSize)")
            print("  Trials: \(trials)")
            print()

            // Prepare
            let stablehlo = generateJAXStableHLO(batchSize: batchSize)
            let client = try PJRTClient(backend: .cpu)
            let executable = try client.compile(mlirModule: stablehlo)

            guard let device = client.defaultDevice else {
                throw PJRTError.noDeviceAvailable
            }

            // Create input data
            var spots = [Float](repeating: 0, count: batchSize)
            for i in 0..<batchSize {
                spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
            }

            // Preallocate output arrays
            var prices = [Float](repeating: 0, count: batchSize)
            var deltas = [Float](repeating: 0, count: batchSize)

            // Create persistent input buffer for reuse tests
            let persistentInputBuffer = try spots.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [batchSize],
                    elementType: .f32,
                    device: device,
                    semantics: .zeroCopy
                )
            }

            // Warmup
            print("  Warming up...")
            for _ in 0..<50 {
                let outputs = try executable.execute(arguments: [persistentInputBuffer])
                try prices.withUnsafeMutableBytes { ptr in
                    try outputs[0].toHost(destination: ptr.baseAddress!)
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // Test 1: Pure XLA execution with buffer reuse
            // ═══════════════════════════════════════════════════════════════
            print("\n  [1] Pure XLA execution (reuse input buffer, discard outputs):")
            var times1: [Double] = []
            times1.reserveCapacity(trials)

            for _ in 0..<trials {
                let start = currentTime()
                let _ = try executable.execute(arguments: [persistentInputBuffer])
                times1.append(currentTime() - start)
            }

            let avg1 = times1.reduce(0, +) / Double(trials) * 1_000_000
            let min1 = times1.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg1)) μs, Min: \(String(format: "%.1f", min1)) μs")

            // ═══════════════════════════════════════════════════════════════
            // Test 2: Input buffer creation overhead
            // ═══════════════════════════════════════════════════════════════
            print("\n  [2] Input buffer creation (zeroCopy):")
            var times2: [Double] = []
            times2.reserveCapacity(trials)

            for _ in 0..<trials {
                let start = currentTime()
                let _ = try spots.withUnsafeBytes { ptr in
                    try client.createBuffer(
                        data: ptr.baseAddress!,
                        shape: [batchSize],
                        elementType: .f32,
                        device: device,
                        semantics: .zeroCopy
                    )
                }
                times2.append(currentTime() - start)
            }

            let avg2 = times2.reduce(0, +) / Double(trials) * 1_000_000
            let min2 = times2.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg2)) μs, Min: \(String(format: "%.1f", min2)) μs")

            // ═══════════════════════════════════════════════════════════════
            // Test 3: Output D2H transfer overhead (sync)
            // ═══════════════════════════════════════════════════════════════
            print("\n  [3] Output D2H transfer (2 outputs, sync):")
            var times3: [Double] = []
            times3.reserveCapacity(trials)

            // Get outputs once
            let outputs = try executable.execute(arguments: [persistentInputBuffer])

            for _ in 0..<trials {
                let start = currentTime()
                try prices.withUnsafeMutableBytes { ptr in
                    try outputs[0].toHost(destination: ptr.baseAddress!)
                }
                try deltas.withUnsafeMutableBytes { ptr in
                    try outputs[1].toHost(destination: ptr.baseAddress!)
                }
                times3.append(currentTime() - start)
            }

            let avg3 = times3.reduce(0, +) / Double(trials) * 1_000_000
            let min3 = times3.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg3)) μs, Min: \(String(format: "%.1f", min3)) μs")

            // ═══════════════════════════════════════════════════════════════
            // Test 4: Output D2H transfer overhead (async)
            // ═══════════════════════════════════════════════════════════════
            print("\n  [4] Output D2H transfer (2 outputs, async):")
            var times4: [Double] = []
            times4.reserveCapacity(trials)

            for _ in 0..<trials {
                let start = currentTime()
                let event0 = try prices.withUnsafeMutableBytes { ptr in
                    try outputs[0].toHostAsync(destination: ptr.baseAddress!)
                }
                let event1 = try deltas.withUnsafeMutableBytes { ptr in
                    try outputs[1].toHostAsync(destination: ptr.baseAddress!)
                }
                try event0?.awaitAndDestroy()
                try event1?.awaitAndDestroy()
                times4.append(currentTime() - start)
            }

            let avg4 = times4.reduce(0, +) / Double(trials) * 1_000_000
            let min4 = times4.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg4)) μs, Min: \(String(format: "%.1f", min4)) μs")

            // ═══════════════════════════════════════════════════════════════
            // Test 5: Full path with buffer reuse
            // ═══════════════════════════════════════════════════════════════
            print("\n  [5] Full path (reuse input, execute, read outputs sync):")
            var times5: [Double] = []
            times5.reserveCapacity(trials)

            for _ in 0..<trials {
                let start = currentTime()
                let outs = try executable.execute(arguments: [persistentInputBuffer])
                try prices.withUnsafeMutableBytes { ptr in
                    try outs[0].toHost(destination: ptr.baseAddress!)
                }
                try deltas.withUnsafeMutableBytes { ptr in
                    try outs[1].toHost(destination: ptr.baseAddress!)
                }
                times5.append(currentTime() - start)
            }

            let avg5 = times5.reduce(0, +) / Double(trials) * 1_000_000
            let min5 = times5.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg5)) μs, Min: \(String(format: "%.1f", min5)) μs")

            // ═══════════════════════════════════════════════════════════════
            // Test 6: Full path with fresh input buffer each time
            // ═══════════════════════════════════════════════════════════════
            print("\n  [6] Full path (fresh input buffer, execute, read outputs):")
            var times6: [Double] = []
            times6.reserveCapacity(trials)

            for _ in 0..<trials {
                let start = currentTime()
                let inputBuffer = try spots.withUnsafeBytes { ptr in
                    try client.createBuffer(
                        data: ptr.baseAddress!,
                        shape: [batchSize],
                        elementType: .f32,
                        device: device,
                        semantics: .zeroCopy
                    )
                }
                let outs = try executable.execute(arguments: [inputBuffer])
                try prices.withUnsafeMutableBytes { ptr in
                    try outs[0].toHost(destination: ptr.baseAddress!)
                }
                try deltas.withUnsafeMutableBytes { ptr in
                    try outs[1].toHost(destination: ptr.baseAddress!)
                }
                times6.append(currentTime() - start)
            }

            let avg6 = times6.reduce(0, +) / Double(trials) * 1_000_000
            let min6 = times6.min()! * 1_000_000
            print("      Avg: \(String(format: "%.1f", avg6)) μs, Min: \(String(format: "%.1f", min6)) μs")

            // Summary
            print("""

            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  OVERHEAD BREAKDOWN (100K batch)                                             ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

              Component                                    Avg (μs)    Min (μs)
              ───────────────────────────────────────────────────────────────────
              [1] Pure XLA execution                      \(String(format: "%8.1f", avg1))    \(String(format: "%8.1f", min1))
              [2] Input buffer creation (zeroCopy)        \(String(format: "%8.1f", avg2))    \(String(format: "%8.1f", min2))
              [3] Output D2H (2 outputs, sync)            \(String(format: "%8.1f", avg3))    \(String(format: "%8.1f", min3))
              [4] Output D2H (2 outputs, async)           \(String(format: "%8.1f", avg4))    \(String(format: "%8.1f", min4))
              [5] Full (reuse input + sync D2H)           \(String(format: "%8.1f", avg5))    \(String(format: "%8.1f", min5))
              [6] Full (fresh input + sync D2H)           \(String(format: "%8.1f", avg6))    \(String(format: "%8.1f", min6))
              ───────────────────────────────────────────────────────────────────

              Overhead Analysis:
              - Input buffer overhead: [6] - [5] = \(String(format: "%.1f", avg6 - avg5)) μs
              - D2H overhead: [5] - [1] = \(String(format: "%.1f", avg5 - avg1)) μs
              - Total overhead vs pure XLA: [6] - [1] = \(String(format: "%.1f", avg6 - avg1)) μs

              JAX reference at 100K: ~779 μs
              SwiftIR best case [5]: \(String(format: "%.1f", avg5)) μs
              Gap to close: \(String(format: "%.1f", avg5 - 779)) μs (\(String(format: "%.0f", (avg5 / 779 - 1) * 100))% overhead)

            """)

        } catch {
            print("ERROR: \(error)")
        }
    }
}
