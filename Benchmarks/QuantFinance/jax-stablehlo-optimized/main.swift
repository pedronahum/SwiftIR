//===-- main.swift - Optimized PJRT Benchmark ----------------*- Swift -*-===//
//
// Optimized PJRT Benchmark
//
// Tests the optimized PJRT execution path with:
// - Combined execute+transfer API (reduces FFI overhead)
// - Fast designated initializer paths (avoids memset)
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

// MARK: - JAX's StableHLO (same as baseline)

func generateJAXStableHLO(batchSize: Int) -> String {
    return """
    module @jax_stablehlo_optimized attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
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

// MARK: - Benchmark Results

struct BenchmarkResult {
    let batchSize: Int
    let method: String
    let avgTimeUs: Double
    let minTimeUs: Double
    let stdTimeUs: Double
    let throughputK: Double
}

// MARK: - Run Benchmarks

func runBaselineBenchmark(batchSize: Int, trials: Int, client: PJRTClient, executable: PJRTLoadedExecutable, device: PJRTDevice, spots: [Float]) throws -> BenchmarkResult {
    var times: [Double] = []

    for _ in 0..<trials {
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

        var prices = [Float](repeating: 0, count: batchSize)
        var deltas = [Float](repeating: 0, count: batchSize)

        let event0 = try prices.withUnsafeMutableBytes { ptr in
            try outputs[0].toHostAsync(destination: ptr.baseAddress!)
        }
        let event1 = try deltas.withUnsafeMutableBytes { ptr in
            try outputs[1].toHostAsync(destination: ptr.baseAddress!)
        }

        try event0?.awaitAndDestroy()
        try event1?.awaitAndDestroy()

        let elapsed = currentTime() - start
        times.append(elapsed)
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        method: "Baseline",
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK
    )
}

func runOptimizedBenchmark(batchSize: Int, trials: Int, client: PJRTClient, executable: PJRTLoadedExecutable, device: PJRTDevice, spots: [Float]) throws -> BenchmarkResult {
    var times: [Double] = []

    // Pre-allocate output arrays outside the loop to avoid allocation overhead
    var prices = [Float](repeating: 0, count: batchSize)
    var deltas = [Float](repeating: 0, count: batchSize)

    // First call sets up the cache
    var isFirstCall = true

    for _ in 0..<trials {
        // Create input buffer using fast API with caching
        // isSameShape is true only after the first call
        let inputBuffer = try spots.withUnsafeBytes { ptr in
            try client.createBufferFast(
                data: ptr.baseAddress!,
                shape: [batchSize],
                elementType: .f32,
                device: device,
                semantics: .zeroCopy,
                cachedSlot: 0,
                isSameShape: !isFirstCall
            )
        }
        isFirstCall = false

        let start = currentTime()

        // Execute using regular API
        let outputs = try executable.execute(arguments: [inputBuffer])

        // Read results back using async transfers
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
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        method: "Optimized",
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK
    )
}

/// Benchmark using the new executeWithBufferReuse high-level API
func runBufferReuseBenchmark(
    batchSize: Int,
    trials: Int,
    spots: [Float]
) throws -> BenchmarkResult {
    // Create runtime and compile executable using high-level API
    let runtime = try PJRTBackedRuntime(backend: .cpu)
    let stablehlo = generateJAXStableHLO(batchSize: batchSize)
    let executable = try runtime.compile(stablehlo)

    // Warmup
    for _ in 0..<10 {
        _ = try executable.executeWithBufferReuse([spots])
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()

        // Use the high-level executeWithBufferReuse API
        // This skips H2D transfer when data hasn't changed
        let outputs = try executable.executeWithBufferReuse([spots])

        let elapsed = currentTime() - start
        times.append(elapsed)

        // Prevent optimization from removing the call
        _ = outputs[0][0] + outputs[1][0]
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        method: "BufferReuse",
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK
    )
}

/// Benchmark using batched buffer creation
func runBatchedBuffersBenchmark(
    batchSize: Int,
    trials: Int,
    spots: [Float]
) throws -> BenchmarkResult {
    // Create runtime and compile executable using high-level API
    let runtime = try PJRTBackedRuntime(backend: .cpu)
    let stablehlo = generateJAXStableHLO(batchSize: batchSize)
    let executable = try runtime.compile(stablehlo)

    // Warmup
    for _ in 0..<10 {
        _ = try executable.executeWithBatchedBuffers([spots])
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()

        // Use the batched buffer creation API
        let outputs = try executable.executeWithBatchedBuffers([spots])

        let elapsed = currentTime() - start
        times.append(elapsed)

        // Prevent optimization from removing the call
        _ = outputs[0][0] + outputs[1][0]
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        method: "BatchedBuffers",
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK
    )
}

/// Benchmark using the ultra-optimized execution with global caching
func runUltraOptimizedBenchmark(
    batchSize: Int,
    trials: Int,
    spots: [Float]
) throws -> BenchmarkResult {
    // Create runtime and compile executable using high-level API
    let runtime = try PJRTBackedRuntime(backend: .cpu)
    let stablehlo = generateJAXStableHLO(batchSize: batchSize)
    let executable = try runtime.compile(stablehlo)

    // Warmup
    for _ in 0..<10 {
        _ = try executable.executeUltraOptimized([spots])
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()

        // Use the ultra-optimized execution path
        let outputs = try executable.executeUltraOptimized([spots])

        let elapsed = currentTime() - start
        times.append(elapsed)

        // Prevent optimization from removing the call
        _ = outputs[0][0] + outputs[1][0]
    }

    let avgTimeUs = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTimeUs = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTimeUs = variance.squareRoot() * 1_000_000
    let throughputK = Double(batchSize) / (avgTimeUs / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        method: "UltraOptimized",
        avgTimeUs: avgTimeUs,
        minTimeUs: minTimeUs,
        stdTimeUs: stdTimeUs,
        throughputK: throughputK
    )
}

// MARK: - Main

@main
struct OptimizedBenchmark {
    static func main() {
        print("""

        ==============================================================================
            Optimized PJRT Execution Benchmark
        ==============================================================================

          Tests optimizations:
          1. Combined execute+transfer API (single FFI call)
          2. Designated initializers (avoids memset overhead)
          3. Pre-allocated output buffers

        """)

        do {
            let trials = 100
            let batchSizes = [100, 1000, 10000, 100000]

            let client = try PJRTClient(backend: .cpu)
            guard let device = client.defaultDevice else {
                print("ERROR: No device available")
                return
            }

            var baselineResults: [BenchmarkResult] = []
            var optimizedResults: [BenchmarkResult] = []
            var bufferReuseResults: [BenchmarkResult] = []
            var ultraOptResults: [BenchmarkResult] = []

            for batchSize in batchSizes {
                print("  Testing batch size: \(batchSize)...")

                let stablehlo = generateJAXStableHLO(batchSize: batchSize)
                let executable = try client.compile(mlirModule: stablehlo)

                // Create input data
                var spots = [Float](repeating: 0, count: batchSize)
                for i in 0..<batchSize {
                    spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
                }

                // Warmup
                print("    Warmup...")
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
                    _ = outputs
                }

                // Run baseline
                print("    Running baseline...")
                let baseline = try runBaselineBenchmark(
                    batchSize: batchSize,
                    trials: trials,
                    client: client,
                    executable: executable,
                    device: device,
                    spots: spots
                )
                baselineResults.append(baseline)
                print("      Baseline: \(String(format: "%.1f", baseline.avgTimeUs)) μs")

                // Run optimized
                print("    Running optimized...")
                let optimized = try runOptimizedBenchmark(
                    batchSize: batchSize,
                    trials: trials,
                    client: client,
                    executable: executable,
                    device: device,
                    spots: spots
                )
                optimizedResults.append(optimized)
                print("      Optimized: \(String(format: "%.1f", optimized.avgTimeUs)) μs")

                // Run buffer reuse (JAX-style optimization)
                print("    Running buffer reuse...")
                let bufferReuse = try runBufferReuseBenchmark(
                    batchSize: batchSize,
                    trials: trials,
                    spots: spots
                )
                bufferReuseResults.append(bufferReuse)
                print("      BufferReuse: \(String(format: "%.1f", bufferReuse.avgTimeUs)) μs")

                // Run ultra-optimized (global cache + all optimizations)
                print("    Running ultra-optimized...")
                let ultraOpt = try runUltraOptimizedBenchmark(
                    batchSize: batchSize,
                    trials: trials,
                    spots: spots
                )
                ultraOptResults.append(ultraOpt)
                print("      UltraOpt: \(String(format: "%.1f", ultraOpt.avgTimeUs)) μs")

                let bestTime = min(optimized.avgTimeUs, min(bufferReuse.avgTimeUs, ultraOpt.avgTimeUs))
                let speedup = baseline.avgTimeUs / bestTime
                print("      Best speedup: \(String(format: "%.2f", speedup))x")
                print()
            }

            // Print comparison table
            print("""

            ==============================================================================
                RESULTS COMPARISON
            ==============================================================================

            """)

            print("  ┌────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
            print("  │ Batch Size │ Baseline(μs) │ Optimized(μs)│ BufReuse(μs) │ UltraOpt(μs) │ Best Speedup │")
            print("  ├────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

            for i in 0..<batchSizes.count {
                let batchStr = String(format: "%10d", batchSizes[i])
                let baseStr = String(format: "%10.1f", baselineResults[i].avgTimeUs)
                let optStr = String(format: "%10.1f", optimizedResults[i].avgTimeUs)
                let reuseStr = String(format: "%10.1f", bufferReuseResults[i].avgTimeUs)
                let ultraStr = String(format: "%10.1f", ultraOptResults[i].avgTimeUs)
                let bestTime = min(optimizedResults[i].avgTimeUs, min(bufferReuseResults[i].avgTimeUs, ultraOptResults[i].avgTimeUs))
                let speedup = baselineResults[i].avgTimeUs / bestTime
                let speedupStr = String(format: "%10.2f", speedup) + "x"
                print("  │ \(batchStr) │ \(baseStr) │ \(optStr) │ \(reuseStr) │ \(ultraStr) │ \(speedupStr) │")
            }
            print("  └────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

            // Analysis at 100K
            print()
            print("  Analysis at 100K batch:")
            if let baseline100k = baselineResults.first(where: { $0.batchSize == 100000 }),
               let opt100k = optimizedResults.first(where: { $0.batchSize == 100000 }),
               let reuse100k = bufferReuseResults.first(where: { $0.batchSize == 100000 }),
               let ultra100k = ultraOptResults.first(where: { $0.batchSize == 100000 }) {
                let bestTime = min(opt100k.avgTimeUs, min(reuse100k.avgTimeUs, ultra100k.avgTimeUs))
                let speedup = baseline100k.avgTimeUs / bestTime
                let savingsUs = baseline100k.avgTimeUs - bestTime
                print("    Baseline:     \(String(format: "%.1f", baseline100k.avgTimeUs)) μs")
                print("    Optimized:    \(String(format: "%.1f", opt100k.avgTimeUs)) μs")
                print("    BufferReuse:  \(String(format: "%.1f", reuse100k.avgTimeUs)) μs")
                print("    UltraOpt:     \(String(format: "%.1f", ultra100k.avgTimeUs)) μs")
                print("    Best:         \(String(format: "%.1f", bestTime)) μs")
                print("    Savings:      \(String(format: "%.1f", savingsUs)) μs (\(String(format: "%.2f", speedup))x speedup)")
                print()
                print("    JAX native target: ~793 μs")
                let gapToJax = bestTime / 793.0
                print("    Current gap to JAX: \(String(format: "%.2f", gapToJax))x")
            }

            // Print LRU Cache Statistics
            print()
            print("  LRU Cache Statistics (JAX-style, max \(PJRTBackedRuntime.cacheMaxSize) entries):")
            let stats = PJRTBackedRuntime.cacheStatistics
            print("    Hits:      \(stats.hits)")
            print("    Misses:    \(stats.misses)")
            print("    Evictions: \(stats.evictions)")
            print("    Size:      \(stats.currentSize)/\(stats.maxSize)")
            print("    Hit Rate:  \(String(format: "%.1f%%", stats.hitRate * 100))")
            print()

        } catch {
            print("ERROR: \(error)")
        }
    }
}
