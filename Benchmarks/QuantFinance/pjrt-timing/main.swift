//===-- main.swift - PJRT Timing Breakdown Benchmark ------*- Swift -*-===//
//
// This benchmark profiles WHERE time is spent in PJRT execution
// to identify the source of the ~200μs gap vs JAX at 100K batch size.
//
// Breakdown measured:
// 1. H2D Buffer Creation - Creating PJRT buffers from host data
// 2. Execute - Kernel execution time
// 3. D2H Initiate - Starting async transfer from device to host
// 4. D2H Await - Waiting for transfer completion
// 5. Buffer Destroy - Cleanup time
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

// MARK: - Main

@main
struct PJRTTimingBreakdown {
    static func main() {
        print("""

        =====================================================================

             PJRT Timing Breakdown Profiler

             Measuring individual PJRT C API call times

        =====================================================================

        """)

        do {
            let size = 100_000
            let trials = 50

            // Compile the gradient function
            print("Compiling gradient function for \(size) elements...")
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
            print("Compile time: \(String(format: "%.1f", compileTime)) ms\n")

            // Get executable
            let executable = gradFunc.executable

            // Prepare input data
            var spots = [Float](repeating: 0, count: size)
            for i in 0..<size {
                spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
            }

            let outputSizes = [size, size]

            // Warmup
            print("Warming up...")
            for _ in 0..<20 {
                let (_, _) = try executable.executeProfiled(
                    [spots],
                    outputSizes: outputSizes
                )
            }

            print("\n=== PJRT TIMING BREAKDOWN ===\n")
            print("Batch size: \(size)")
            print("Trials: \(trials)\n")

            // Collect timing data
            var h2dTimes: [Double] = []
            var executeTimes: [Double] = []
            var d2hInitTimes: [Double] = []
            var d2hAwaitTimes: [Double] = []
            var bufferDestroyTimes: [Double] = []
            var totalTimes: [Double] = []

            for _ in 0..<trials {
                let (_, timing) = try executable.executeProfiled(
                    [spots],
                    outputSizes: outputSizes
                )

                h2dTimes.append(timing.h2dCreateUs)
                executeTimes.append(timing.executeUs)
                d2hInitTimes.append(timing.d2hInitiateUs)
                d2hAwaitTimes.append(timing.d2hAwaitUs)
                bufferDestroyTimes.append(timing.bufferDestroyUs)
                totalTimes.append(timing.totalUs)
            }

            // Calculate statistics
            func stats(_ arr: [Double]) -> (avg: Double, min: Double, max: Double, std: Double) {
                let avg = arr.reduce(0, +) / Double(arr.count)
                let min = arr.min()!
                let max = arr.max()!
                let variance = arr.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(arr.count)
                let std = variance.squareRoot()
                return (avg, min, max, std)
            }

            let h2dStats = stats(h2dTimes)
            let execStats = stats(executeTimes)
            let d2hInitStats = stats(d2hInitTimes)
            let d2hAwaitStats = stats(d2hAwaitTimes)
            let destroyStats = stats(bufferDestroyTimes)
            let totalStats = stats(totalTimes)

            // Print results
            print("Component               Avg (μs)    Min (μs)    Max (μs)    Std (μs)    %Total")
            print("-" * 85)

            func printRow(_ name: String, _ s: (avg: Double, min: Double, max: Double, std: Double)) {
                let pct = (s.avg / totalStats.avg) * 100
                print(String(format: "%-22s %9.1f   %9.1f   %9.1f   %9.1f   %6.1f%%",
                    name, s.avg, s.min, s.max, s.std, pct))
            }

            printRow("H2D Buffer Create", h2dStats)
            printRow("Execute", execStats)
            printRow("D2H Initiate", d2hInitStats)
            printRow("D2H Await", d2hAwaitStats)
            printRow("Buffer Destroy", destroyStats)
            print("-" * 85)
            printRow("TOTAL", totalStats)

            // Analysis
            let overheadUs = totalStats.avg - execStats.avg
            let jaxBaseline = 782.0  // μs

            print("""

            =====================================================================

                 ANALYSIS

                 JAX baseline:           \(String(format: "%.1f", jaxBaseline)) μs
                 SwiftIR total:          \(String(format: "%.1f", totalStats.avg)) μs
                 SwiftIR min:            \(String(format: "%.1f", totalStats.min)) μs

                 Gap vs JAX (avg):       \(String(format: "%.1f", totalStats.avg - jaxBaseline)) μs
                 Gap vs JAX (min):       \(String(format: "%.1f", totalStats.min - jaxBaseline)) μs

                 Pure kernel time:       \(String(format: "%.1f", execStats.avg)) μs (\(String(format: "%.1f", execStats.avg / totalStats.avg * 100))% of total)
                 Overhead (non-kernel):  \(String(format: "%.1f", overheadUs)) μs (\(String(format: "%.1f", overheadUs / totalStats.avg * 100))% of total)

                 Breakdown of overhead:
                   H2D:      \(String(format: "%.1f", h2dStats.avg)) μs
                   D2H:      \(String(format: "%.1f", d2hInitStats.avg + d2hAwaitStats.avg)) μs (init: \(String(format: "%.1f", d2hInitStats.avg)), await: \(String(format: "%.1f", d2hAwaitStats.avg)))
                   Cleanup:  \(String(format: "%.1f", destroyStats.avg)) μs

            =====================================================================
            """)

            // Recommendations based on findings
            print("\n=== RECOMMENDATIONS ===\n")

            if h2dStats.avg > 50 {
                print("- H2D buffer creation is significant (\(String(format: "%.1f", h2dStats.avg)) μs)")
                print("  Consider: Buffer pooling, reusing buffers for same shapes")
            }

            if d2hAwaitStats.avg > d2hInitStats.avg * 2 {
                print("- D2H await dominates D2H time")
                print("  This is expected - async execution overlaps with compute")
            }

            if destroyStats.avg > 30 {
                print("- Buffer cleanup overhead is notable (\(String(format: "%.1f", destroyStats.avg)) μs)")
                print("  Consider: Lazy buffer destruction, deferred cleanup")
            }

            if totalStats.avg - totalStats.min > 200 {
                print("- High variance between runs (spread: \(String(format: "%.1f", totalStats.max - totalStats.min)) μs)")
                print("  Consider: CPU pinning, warmup improvements")
            }

            // ========================================
            // ASYNC CALLBACK COMPARISON
            // ========================================
            print("\n=== ASYNC CALLBACK COMPARISON ===\n")
            print("Comparing blocking PJRT_Event_Await vs PJRT_Event_OnReady callbacks...\n")

            // Warmup for callback-based execution
            print("Warming up callback-based execution...")
            for _ in 0..<20 {
                let (_, _) = try executable.executeWithCallbacks(
                    [spots],
                    outputSizes: outputSizes
                )
            }

            // Collect callback-based timing data
            var callbackTotalTimes: [Double] = []
            var callbackAwaitTimes: [Double] = []

            for _ in 0..<trials {
                let (_, timing) = try executable.executeWithCallbacks(
                    [spots],
                    outputSizes: outputSizes
                )

                callbackTotalTimes.append(timing.totalUs)
                callbackAwaitTimes.append(timing.d2hAwaitUs)
            }

            let callbackTotalStats = stats(callbackTotalTimes)
            let callbackAwaitStats = stats(callbackAwaitTimes)

            print("Callback-based execution:")
            print("  Total:     Avg \(String(format: "%.1f", callbackTotalStats.avg)) μs, Min \(String(format: "%.1f", callbackTotalStats.min)) μs")
            print("  Wait time: Avg \(String(format: "%.1f", callbackAwaitStats.avg)) μs, Min \(String(format: "%.1f", callbackAwaitStats.min)) μs")

            print("\nBlocking execution (from above):")
            print("  Total:     Avg \(String(format: "%.1f", totalStats.avg)) μs, Min \(String(format: "%.1f", totalStats.min)) μs")
            print("  D2H Await: Avg \(String(format: "%.1f", d2hAwaitStats.avg)) μs, Min \(String(format: "%.1f", d2hAwaitStats.min)) μs")

            let speedup = totalStats.avg / callbackTotalStats.avg
            let improvement = totalStats.avg - callbackTotalStats.avg

            print("""

            =====================================================================

                 ASYNC CALLBACK RESULTS

                 Blocking total:         \(String(format: "%.1f", totalStats.avg)) μs
                 Callback total:         \(String(format: "%.1f", callbackTotalStats.avg)) μs

                 Improvement:            \(String(format: "%.1f", improvement)) μs (\(String(format: "%.1f", (improvement / totalStats.avg) * 100))%)
                 Speedup:                \(String(format: "%.2f", speedup))x

                 JAX baseline:           782 μs
                 Gap vs JAX (callback):  \(String(format: "%.1f", callbackTotalStats.avg - 782)) μs

            =====================================================================
            """)

            // ========================================
            // POOLED EXECUTION COMPARISON
            // ========================================
            print("\n=== POOLED EXECUTION (ZERO-COPY) ===\n")
            print("Comparing regular execution vs pooled execution with zero-copy semantics...\n")

            // Warmup for pooled execution
            print("Warming up pooled execution...")
            for _ in 0..<20 {
                let (_, _) = try gradFunc.executable.executePooled(
                    [spots],
                    outputSizes: outputSizes
                )
            }

            // Collect pooled timing data
            var pooledTotalTimes: [Double] = []
            var pooledH2dTimes: [Double] = []

            for _ in 0..<trials {
                let (_, timing) = try gradFunc.executable.executePooled(
                    [spots],
                    outputSizes: outputSizes
                )

                pooledTotalTimes.append(timing.totalUs)
                pooledH2dTimes.append(timing.h2dCreateUs)
            }

            let pooledTotalStats = stats(pooledTotalTimes)
            let pooledH2dStats = stats(pooledH2dTimes)

            print("Pooled (zero-copy) execution:")
            print("  Total:     Avg \(String(format: "%.1f", pooledTotalStats.avg)) μs, Min \(String(format: "%.1f", pooledTotalStats.min)) μs")
            print("  H2D time:  Avg \(String(format: "%.1f", pooledH2dStats.avg)) μs, Min \(String(format: "%.1f", pooledH2dStats.min)) μs")

            print("\nRegular profiled execution (from above):")
            print("  Total:     Avg \(String(format: "%.1f", totalStats.avg)) μs, Min \(String(format: "%.1f", totalStats.min)) μs")
            print("  H2D time:  Avg \(String(format: "%.1f", h2dStats.avg)) μs, Min \(String(format: "%.1f", h2dStats.min)) μs")

            let pooledSpeedup = totalStats.avg / pooledTotalStats.avg
            let pooledImprovement = totalStats.avg - pooledTotalStats.avg
            let h2dImprovement = h2dStats.avg - pooledH2dStats.avg

            print("""

            =====================================================================

                 POOLED EXECUTION RESULTS

                 Regular total:          \(String(format: "%.1f", totalStats.avg)) μs
                 Pooled total:           \(String(format: "%.1f", pooledTotalStats.avg)) μs

                 Total improvement:      \(String(format: "%.1f", pooledImprovement)) μs (\(String(format: "%.1f", (pooledImprovement / totalStats.avg) * 100))%)
                 H2D improvement:        \(String(format: "%.1f", h2dImprovement)) μs
                 Speedup:                \(String(format: "%.2f", pooledSpeedup))x

                 JAX baseline:           782 μs
                 Gap vs JAX (pooled):    \(String(format: "%.1f", pooledTotalStats.avg - 782)) μs

            =====================================================================
            """)

            // Final summary
            print("\n=== FINAL SUMMARY ===\n")
            print("Method                    Avg (μs)    Min (μs)    Gap vs JAX")
            print("-" * 60)
            print(String(format: "%-25s %8.1f    %8.1f    %8.1f μs", "Regular (blocking)", totalStats.avg, totalStats.min, totalStats.avg - 782))
            print(String(format: "%-25s %8.1f    %8.1f    %8.1f μs", "Async callbacks", callbackTotalStats.avg, callbackTotalStats.min, callbackTotalStats.avg - 782))
            print(String(format: "%-25s %8.1f    %8.1f    %8.1f μs", "Pooled (zero-copy)", pooledTotalStats.avg, pooledTotalStats.min, pooledTotalStats.avg - 782))
            print("-" * 60)
            print("JAX baseline: 782 μs")

        } catch {
            print("ERROR: \(error)")
        }
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        return String(repeating: lhs, count: rhs)
    }
}
