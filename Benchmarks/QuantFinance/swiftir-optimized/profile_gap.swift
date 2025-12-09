//===-- profile_gap.swift - Profile SwiftIR vs JAX Gap ------*- Swift -*-===//
//
// This benchmark profiles WHERE time is spent in the SwiftIR execution path
// to understand the ~128μs gap vs JAX at 100K batch size.
//
// Breakdown:
// 1. Pure XLA kernel execution time (via PJRT execute only)
// 2. H2D buffer creation time
// 3. D2H transfer time
// 4. Swift wrapper overhead
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
struct ProfileGap {
    static func main() {
        print("""

        =====================================================================

             SwiftIR Performance Gap Profiler

             Breaking down WHERE time is spent in execution

        =====================================================================

        """)

        do {
            let size = 100_000
            let trials = 100

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

            // Warmup
            print("Warming up...")
            for _ in 0..<20 {
                let _ = try executable.executeAsync([spots])
            }

            print("\n=== PROFILING BREAKDOWN ===\n")

            // ========================================
            // 1. Profile full executeAsync path
            // ========================================
            print("1. Full executeAsync (end-to-end):")
            var fullTimes: [Double] = []
            for _ in 0..<trials {
                let start = currentTime()
                let _ = try executable.executeAsync([spots])
                fullTimes.append(currentTime() - start)
            }
            let fullAvg = fullTimes.reduce(0, +) / Double(trials) * 1_000_000
            let fullMin = fullTimes.min()! * 1_000_000
            print("   Avg: \(String(format: "%.1f", fullAvg)) μs")
            print("   Min: \(String(format: "%.1f", fullMin)) μs")

            // ========================================
            // 2. Profile with detailed timing via executeWithTiming
            // ========================================
            print("\n2. Detailed breakdown (using executeWithTiming):")

            // We need to access the lower-level APIs to get timing breakdown
            // Let's measure individual components

            // 2a. Array preparation overhead (Swift side)
            print("\n   2a. Swift array preparation overhead:")
            var arrayPrepTimes: [Double] = []
            for _ in 0..<trials {
                let start = currentTime()
                // Simulate what executeAsync does: wrap in array, check count
                let inputs = [spots]
                let _ = inputs[0].count
                arrayPrepTimes.append(currentTime() - start)
            }
            let arrayPrepAvg = arrayPrepTimes.reduce(0, +) / Double(trials) * 1_000_000
            print("       Avg: \(String(format: "%.3f", arrayPrepAvg)) μs (negligible)")

            // 2b. withUnsafeBytes overhead
            print("\n   2b. withUnsafeBytes closure overhead:")
            var unsafeTimes: [Double] = []
            for _ in 0..<trials {
                let start = currentTime()
                spots.withUnsafeBytes { ptr in
                    let _ = ptr.baseAddress
                }
                unsafeTimes.append(currentTime() - start)
            }
            let unsafeAvg = unsafeTimes.reduce(0, +) / Double(trials) * 1_000_000
            print("       Avg: \(String(format: "%.3f", unsafeAvg)) μs (negligible)")

            // ========================================
            // 3. Compare with executeHotPath (single FFI call)
            // ========================================
            print("\n3. executeHotPath (single FFI call):")
            executable.clearCache()

            // Warmup hot path
            let outputSizes = [size, size]
            for _ in 0..<20 {
                let _ = try executable.executeHotPath([spots], outputSizes: outputSizes)
            }

            var hotPathTimes: [Double] = []
            for _ in 0..<trials {
                let start = currentTime()
                let _ = try executable.executeHotPath([spots], outputSizes: outputSizes)
                hotPathTimes.append(currentTime() - start)
            }
            let hotPathAvg = hotPathTimes.reduce(0, +) / Double(trials) * 1_000_000
            let hotPathMin = hotPathTimes.min()! * 1_000_000
            print("   Avg: \(String(format: "%.1f", hotPathAvg)) μs")
            print("   Min: \(String(format: "%.1f", hotPathMin)) μs")

            // ========================================
            // 4. Profile executeWithSemantics variants
            // ========================================
            print("\n4. Buffer semantics comparison:")
            executable.clearCache()

            for (semantics, name) in [
                (BufferSemantics.zeroCopy, "zeroCopy"),
                (BufferSemantics.copy, "copy")
            ] {
                // Warmup
                for _ in 0..<10 {
                    let _ = try executable.executeWithSemantics([spots], semantics: semantics)
                }

                var times: [Double] = []
                for _ in 0..<trials {
                    let start = currentTime()
                    let _ = try executable.executeWithSemantics([spots], semantics: semantics)
                    times.append(currentTime() - start)
                }
                let avg = times.reduce(0, +) / Double(trials) * 1_000_000
                let min = times.min()! * 1_000_000
                print("   \(name): Avg \(String(format: "%.1f", avg)) μs, Min \(String(format: "%.1f", min)) μs")
            }

            // ========================================
            // 5. Summary and gap analysis
            // ========================================
            print("""

            =====================================================================

                 GAP ANALYSIS SUMMARY

                 JAX baseline (from earlier):     ~782 μs
                 SwiftIR executeAsync:            ~\(String(format: "%.0f", fullAvg)) μs
                 SwiftIR executeHotPath:          ~\(String(format: "%.0f", hotPathAvg)) μs

                 Gap vs JAX:                      ~\(String(format: "%.0f", fullAvg - 782)) μs

                 OBSERVATIONS:
                 - Swift array overhead: negligible (<1 μs)
                 - withUnsafeBytes overhead: negligible (<1 μs)

                 The gap is likely in:
                 1. XLA kernel code generation differences
                 2. PJRT buffer management overhead
                 3. Differences in how constants are handled

                 NEXT STEPS:
                 - Compare generated StableHLO (SwiftIR vs JAX)
                 - Check XLA compilation flags
                 - Profile PJRT C API calls individually

            =====================================================================
            """)

            // ========================================
            // 6. Dump MLIR for comparison
            // ========================================
            print("\n6. Generated StableHLO (first 1000 chars):")
            let mlir = executable.stablehloSource
            print(String(mlir.prefix(1000)))
            print("...")
            print("\nFull MLIR has \(mlir.count) characters, \(mlir.components(separatedBy: "\n").count) lines")

            // Count operations
            let ops = ["constant", "broadcast_in_dim", "multiply", "divide", "add", "subtract",
                      "log", "exp", "sqrt", "negate", "logistic"]
            print("\nOperation counts:")
            for op in ops {
                let count = mlir.components(separatedBy: "stablehlo.\(op)").count - 1
                if count > 0 {
                    print("   stablehlo.\(op): \(count)")
                }
            }

        } catch {
            print("ERROR: \(error)")
        }
    }
}
