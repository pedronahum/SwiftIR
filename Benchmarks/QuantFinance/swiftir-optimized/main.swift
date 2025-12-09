//===-- main.swift - SwiftIR Buffer Semantics Benchmark ----*- Swift -*-===//
//
// SwiftIR Execution Optimization Benchmark
//
// This benchmark compares execution methods and buffer semantics:
// 1. copy (kImmutableOnlyDuringCall) - Data copied immediately, safest
// 2. asyncTransfer (kImmutableUntilTransferCompletes) - Async transfer
// 3. zeroCopy (kImmutableZeroCopy) - No copy, host memory used directly
// 4. auto - Automatically selects best based on backend
// 5. executeAsync - Fully optimized path with async D2H and metadata caching
//
// The optimal choice depends on:
// - Backend type (CPU can use zeroCopy, GPU/TPU typically need copy)
// - Data lifetime (zeroCopy requires data to outlive the buffer)
// - Batch size (overhead matters more for small batches)
//
//===----------------------------------------------------------------===//

import SwiftIR
import SwiftIRXLA
import Foundation
import _Differentiation

// MARK: - Timing

func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - Benchmark Result Structure

struct BenchmarkResult {
    let name: String
    let batchSize: Int
    let executeTimeUs: Double
    let minExecuteTimeUs: Double
    let stdExecuteTimeUs: Double
    let throughput: Double
}

// MARK: - Run benchmark with specific semantics

func runBenchmarkWithSemantics(
    name: String,
    size: Int,
    trials: Int,
    executable: PJRTBackedExecutable,
    semantics: BufferSemantics
) throws -> BenchmarkResult {
    var spots = [Float](repeating: 0, count: size)
    for i in 0..<size {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }
    // Note: seed is embedded as a constant in the IR, only 1 input needed

    // Clear cache before each semantics test
    executable.clearCache()

    // Warmup
    for _ in 0..<10 {
        let _ = try executable.executeWithSemantics([spots], semantics: semantics)
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()
        let _ = try executable.executeWithSemantics([spots], semantics: semantics)
        times.append(currentTime() - start)
    }

    let avgTime = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTime = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTime = variance.squareRoot() * 1_000_000
    let throughput = Double(size) / (avgTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        name: name,
        batchSize: size,
        executeTimeUs: avgTime,
        minExecuteTimeUs: minTime,
        stdExecuteTimeUs: stdTime,
        throughput: throughput
    )
}

// MARK: - Run benchmark with executeAsync (fully optimized)

func runBenchmarkAsync(
    name: String,
    size: Int,
    trials: Int,
    executable: PJRTBackedExecutable
) throws -> BenchmarkResult {
    var spots = [Float](repeating: 0, count: size)
    for i in 0..<size {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }
    // Note: seed is embedded as a constant in the IR, only 1 input needed

    // Clear cache before test (first run caches metadata)
    executable.clearCache()

    // Warmup - also populates cached output metadata
    for _ in 0..<10 {
        let _ = try executable.executeAsync([spots])
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()
        let _ = try executable.executeAsync([spots])
        times.append(currentTime() - start)
    }

    let avgTime = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTime = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTime = variance.squareRoot() * 1_000_000
    let throughput = Double(size) / (avgTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        name: name,
        batchSize: size,
        executeTimeUs: avgTime,
        minExecuteTimeUs: minTime,
        stdExecuteTimeUs: stdTime,
        throughput: throughput
    )
}

// MARK: - Run benchmark with executeAsyncWithDonation (buffer donation)

func runBenchmarkAsyncWithDonation(
    name: String,
    size: Int,
    trials: Int,
    executable: PJRTBackedExecutable
) throws -> BenchmarkResult {
    // Clear cache before test
    executable.clearCache()

    // Configure input-output aliasing for buffer donation
    // This tells XLA that input 0 can be reused for output 0
    // The gradient function returns one output (dLoss/dSpot) with same shape as input spots
    executable.configureSimpleAliases(inputIndices: [0])

    // Debug: print a snippet of the MLIR to verify aliasing (only for small batches)
    if size == 1000 {
        let snippet = String(executable.stablehloSource.prefix(400))
        print("  DEBUG: MLIR with aliasing:\n\(snippet)...\n")
    }

    // Warmup - also populates cached output metadata
    for _ in 0..<10 {
        var spots = [Float](repeating: 0, count: size)
        for i in 0..<size {
            spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
        }
        // Note: seed is embedded as a constant in the IR, only 1 input needed
        // Donate the first input (spots) - its buffer can be reused for output
        let _ = try executable.executeAsyncWithDonation([spots], donateInputIndices: [0])
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        // Create fresh arrays each iteration since donation invalidates inputs
        var spots = [Float](repeating: 0, count: size)
        for i in 0..<size {
            spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
        }
        // Note: seed is embedded as a constant in the IR, only 1 input needed

        let start = currentTime()
        // Donate spots buffer (index 0) - can be reused for output
        let _ = try executable.executeAsyncWithDonation([spots], donateInputIndices: [0])
        times.append(currentTime() - start)
    }

    let avgTime = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTime = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTime = variance.squareRoot() * 1_000_000
    let throughput = Double(size) / (avgTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        name: name,
        batchSize: size,
        executeTimeUs: avgTime,
        minExecuteTimeUs: minTime,
        stdExecuteTimeUs: stdTime,
        throughput: throughput
    )
}

// MARK: - Run benchmark with executeHotPath (single FFI call)

func runBenchmarkHotPath(
    name: String,
    size: Int,
    trials: Int,
    executable: PJRTBackedExecutable
) throws -> BenchmarkResult {
    var spots = [Float](repeating: 0, count: size)
    for i in 0..<size {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }

    // Clear cache before test (first run caches metadata)
    executable.clearCache()

    // Output sizes: gradient function returns [output, gradient] each with same size as input
    let outputSizes = [size, size]

    // Warmup
    for _ in 0..<10 {
        let _ = try executable.executeHotPath([spots], outputSizes: outputSizes)
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<trials {
        let start = currentTime()
        let _ = try executable.executeHotPath([spots], outputSizes: outputSizes)
        times.append(currentTime() - start)
    }

    let avgTime = times.reduce(0, +) / Double(trials) * 1_000_000
    let minTime = times.min()! * 1_000_000
    let mean = times.reduce(0, +) / Double(trials)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdTime = variance.squareRoot() * 1_000_000
    let throughput = Double(size) / (avgTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        name: name,
        batchSize: size,
        executeTimeUs: avgTime,
        minExecuteTimeUs: minTime,
        stdExecuteTimeUs: stdTime,
        throughput: throughput
    )
}

// MARK: - Main

@main
struct SwiftIRBufferSemanticsBenchmark {
    static func main() {
        print("""

        =====================================================================

             SwiftIR Buffer Semantics Benchmark

             Comparing PJRT host buffer semantics:

             1. copy        - Immediate copy, safe, no lifetime requirements
             2. asyncTransfer - Async copy, data must remain valid until done
             3. zeroCopy    - No copy, fastest but data must outlive buffer
             4. auto        - Auto-selects best for backend (zeroCopy on CPU)

             Backend: CPU (can directly access host memory)

        =====================================================================

        """)

        do {
            let trials = 100
            let batchSizes = [1000, 10000, 100000]

            // Store results for summary table
            var allResults: [Int: [BenchmarkResult]] = [:]

            for size in batchSizes {
                print("=== Batch Size: \(size) ===\n")

                // Compile using compileGradientForPJRT
                print("  Compiling gradient function...")
                let compileStart = currentTime()

                let gradFunc = try compileGradientForPJRT(
                    input: TensorSpec(shape: [size], dtype: .float32),
                    backend: .cpu
                ) { spot in
                    let strike = createConstant(100.0, shape: [size], dtype: .float32)
                    let rate = createConstant(0.05, shape: [size], dtype: .float32)
                    let maturity = createConstant(1.0, shape: [size], dtype: .float32)
                    let volatility = createConstant(0.2, shape: [size], dtype: .float32)

                    let stdev = volatility * diffSqrt(maturity)
                    let logSK = diffLog(spot / strike)
                    let half = createConstant(0.5, shape: [size], dtype: .float32)
                    let drift = (rate + volatility * volatility * half) * maturity
                    let d1 = (logSK + drift) / stdev
                    let d2 = d1 - stdev
                    let df = diffExp(diffNegate(rate) * maturity)

                    let factor = createConstant(1.702, shape: [size], dtype: .float32)
                    let nd1 = diffSigmoid(d1 * factor)
                    let nd2 = diffSigmoid(d2 * factor)

                    return spot * nd1 - strike * df * nd2
                }

                let compileTime = (currentTime() - compileStart) * 1000
                print("  Compile time: \(String(format: "%.1f", compileTime)) ms\n")

                // Get the underlying executable for direct benchmark
                let executable = gradFunc.executable

                // Test all semantics
                var results: [BenchmarkResult] = []

                for (semantics, label) in [
                    (BufferSemantics.copy, "copy (ImmutableOnlyDuringCall)"),
                    (BufferSemantics.asyncTransfer, "asyncTransfer (ImmutableUntilTransferCompletes)"),
                    (BufferSemantics.zeroCopy, "zeroCopy (ImmutableZeroCopy)"),
                    (BufferSemantics.auto, "auto (zeroCopy on CPU)")
                ] {
                    let result = try runBenchmarkWithSemantics(
                        name: label,
                        size: size,
                        trials: trials,
                        executable: executable,
                        semantics: semantics
                    )
                    results.append(result)

                    print("  \(label):")
                    print("    Avg: \(String(format: "%7.1f", result.executeTimeUs)) us")
                    print("    Min: \(String(format: "%7.1f", result.minExecuteTimeUs)) us")
                    print("    Std: \(String(format: "%7.1f", result.stdExecuteTimeUs)) us")
                    print("    Throughput: \(String(format: "%.0f", result.throughput)) K/s\n")
                }

                // Test executeAsync (fully optimized path)
                let asyncResult = try runBenchmarkAsync(
                    name: "executeAsync (async D2H + cached metadata)",
                    size: size,
                    trials: trials,
                    executable: executable
                )
                results.append(asyncResult)

                print("  \(asyncResult.name):")
                print("    Avg: \(String(format: "%7.1f", asyncResult.executeTimeUs)) us")
                print("    Min: \(String(format: "%7.1f", asyncResult.minExecuteTimeUs)) us")
                print("    Std: \(String(format: "%7.1f", asyncResult.stdExecuteTimeUs)) us")
                print("    Throughput: \(String(format: "%.0f", asyncResult.throughput)) K/s\n")

                // Test executeAsyncWithDonation (buffer donation)
                let donationResult = try runBenchmarkAsyncWithDonation(
                    name: "executeAsyncWithDonation (buffer reuse)",
                    size: size,
                    trials: trials,
                    executable: executable
                )
                results.append(donationResult)

                print("  \(donationResult.name):")
                print("    Avg: \(String(format: "%7.1f", donationResult.executeTimeUs)) us")
                print("    Min: \(String(format: "%7.1f", donationResult.minExecuteTimeUs)) us")
                print("    Std: \(String(format: "%7.1f", donationResult.stdExecuteTimeUs)) us")
                print("    Throughput: \(String(format: "%.0f", donationResult.throughput)) K/s\n")

                // Test executeHotPath (single FFI call: H2D + Execute + D2H)
                let hotPathResult = try runBenchmarkHotPath(
                    name: "executeHotPath (single FFI call)",
                    size: size,
                    trials: trials,
                    executable: executable
                )
                results.append(hotPathResult)

                print("  \(hotPathResult.name):")
                print("    Avg: \(String(format: "%7.1f", hotPathResult.executeTimeUs)) us")
                print("    Min: \(String(format: "%7.1f", hotPathResult.minExecuteTimeUs)) us")
                print("    Std: \(String(format: "%7.1f", hotPathResult.stdExecuteTimeUs)) us")
                print("    Throughput: \(String(format: "%.0f", hotPathResult.throughput)) K/s\n")

                // Calculate speedups relative to copy
                let copyTime = results[0].executeTimeUs
                print("  Speedups vs copy:")
                for result in results.dropFirst() {
                    let speedup = copyTime / result.executeTimeUs
                    print("    \(result.name): \(String(format: "%.2f", speedup))x")
                }
                print("")

                allResults[size] = results
            }

            // Print summary table
            print("""
            =====================================================================

                 SUMMARY TABLE (Average execution time in microseconds)

            """)

            // Header
            print(String(format: "  %-45s %10s %10s %10s",
                "Semantics", "1K", "10K", "100K"))
            print("  " + String(repeating: "-", count: 77))

            // Data rows
            let semanticsNames = ["copy", "asyncTransfer", "zeroCopy", "auto", "executeAsync", "donation", "hotPath"]
            for (i, name) in semanticsNames.enumerated() {
                let t1k = allResults[1000]?[i].executeTimeUs ?? 0
                let t10k = allResults[10000]?[i].executeTimeUs ?? 0
                let t100k = allResults[100000]?[i].executeTimeUs ?? 0
                print(String(format: "  %-45s %10.1f %10.1f %10.1f",
                    name, t1k, t10k, t100k))
            }

            print("""

            =====================================================================

                 RECOMMENDATIONS

                 For best latency on CPU, use executeAsync():
                 - Auto-selects optimal buffer semantics (zeroCopy on CPU)
                 - Caches output metadata (avoids PJRT queries after first run)
                 - Uses async D2H transfers
                 - Preallocates output arrays

                 CPU Backend:
                 - executeAsync() provides best latency with reusable input arrays
                 - Buffer donation on CPU adds overhead (array recreation) without
                   memory savings since zeroCopy already avoids copies

                 GPU/TPU Backend:
                 - executeAsyncWithDonation() provides memory savings
                 - Buffer donation eliminates device memory allocation for outputs
                 - tf.aliasing_output attribute enables XLA buffer reuse
                 - Use when inputs won't be reused after execution

                 Buffer Donation (GPU/TPU):
                 - Primary benefit is MEMORY savings, not necessarily latency
                 - Requires configureSimpleAliases() for compile-time aliasing
                 - Most beneficial when input/output shapes match exactly

            =====================================================================
            """)

        } catch {
            print("ERROR: \(error)")
        }
    }
}
