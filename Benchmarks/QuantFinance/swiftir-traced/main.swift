//===-- main.swift - SwiftIR Traced Quantitative Finance --*- Swift -*-===//
//
// SwiftIR Traced Quantitative Finance Benchmark
//
// This demonstrates the PROPER SwiftIR API using DifferentiableTracer:
// - Write Black-Scholes using Swift operators (+, -, *, /, etc.)
// - DifferentiableTracer traces the computation graph
// - Generates StableHLO MLIR automatically
// - Executes via PJRT/XLA with proper gradient compilation
//
// Uses compileGradientForPJRT for working forward+gradient execution.
//
// Key features:
// - Separates compile time from execution time
// - Async D2H transfers with preallocated outputs
// - Can dump MLIR for comparison with JAX's HLO
//
//===----------------------------------------------------------------===//

import SwiftIR
import SwiftIRXLA
import Foundation
import _Differentiation

// MARK: - Command line arguments

let dumpMLIR = CommandLine.arguments.contains("--dump-mlir")
let compareOptLevels = CommandLine.arguments.contains("--compare-opt-levels")
let mlirOutputDir = "mlir_dumps"

// Parse MLIR optimization level from command line
// Default is now .none (JAX-style) for best runtime performance
func parseMLIROptimizationLevel() -> MLIROptimizationLevel {
    if CommandLine.arguments.contains("--opt=none") { return .none }
    if CommandLine.arguments.contains("--opt=minimal") { return .minimal }
    if CommandLine.arguments.contains("--opt=reduced") { return .reduced }
    if CommandLine.arguments.contains("--opt=standard") { return .standard }
    if CommandLine.arguments.contains("--opt=maximum") { return .maximum }
    return .none  // default: JAX-style (let XLA optimize)
}

// Parse XLA backend optimization level from command line
// Default is .default (use XLA's default, typically O2)
func parseXLAOptimizationLevel() -> XLAOptimizationLevel {
    if CommandLine.arguments.contains("--xla=default") { return .default }
    if CommandLine.arguments.contains("--xla=0") { return .O0 }
    if CommandLine.arguments.contains("--xla=1") { return .O1 }
    if CommandLine.arguments.contains("--xla=2") { return .O2 }
    if CommandLine.arguments.contains("--xla=3") { return .O3 }
    return .default
}

let selectedMLIROptLevel = parseMLIROptimizationLevel()
let selectedXLAOptLevel = parseXLAOptimizationLevel()

// MARK: - Timing

func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - Benchmark Result Structure

struct BenchmarkResult {
    let batchSize: Int
    let compileTimeMs: Double
    let forwardGradTimeUs: Double
    let minForwardGradTimeUs: Double
    let stdForwardGradTimeUs: Double
    let forwardGradThroughput: Double
    let samplePrices: [Float]
    let sampleDeltas: [Float]
}

// MARK: - Run benchmark for a specific batch size

func runBenchmarkForBatchSize(
    _ size: Int,
    trials: Int,
    dumpMLIR: Bool = false,
    mlirOptLevel: MLIROptimizationLevel = .none,
    xlaOptLevel: XLAOptimizationLevel = .default
) throws -> BenchmarkResult {
    // Compile the gradient function using the proper SwiftIR API
    let compileStart = currentTime()

    let gradFunc = try compileGradientForPJRT(
        input: TensorSpec(shape: [size], dtype: .float32),
        backend: .cpu,
        mlirOptimization: mlirOptLevel,
        xlaOptimization: xlaOptLevel
    ) { spot in
        // Black-Scholes loss function inline with parametric batch size
        let strike = createConstant(100.0, shape: [size], dtype: .float32)
        let rate = createConstant(0.05, shape: [size], dtype: .float32)
        let maturity = createConstant(1.0, shape: [size], dtype: .float32)
        let volatility = createConstant(0.2, shape: [size], dtype: .float32)

        // Black-Scholes call pricing
        let stdev = volatility * diffSqrt(maturity)
        let logSK = diffLog(spot / strike)
        let half = createConstant(0.5, shape: [size], dtype: .float32)
        let drift = (rate + volatility * volatility * half) * maturity
        let d1 = (logSK + drift) / stdev
        let d2 = d1 - stdev
        let df = diffExp(diffNegate(rate) * maturity)

        // Normal CDF approximation using sigmoid
        let factor = createConstant(1.702, shape: [size], dtype: .float32)
        let nd1 = diffSigmoid(d1 * factor)
        let nd2 = diffSigmoid(d2 * factor)

        return spot * nd1 - strike * df * nd2
    }

    let compileTime = (currentTime() - compileStart) * 1000  // ms

    // Dump MLIR if requested
    if dumpMLIR && size == 1000 {
        // Create output directory
        try? FileManager.default.createDirectory(atPath: mlirOutputDir, withIntermediateDirectories: true)
        let mlirPath = "\(mlirOutputDir)/black_scholes_fwd_grad_stablehlo.txt"
        try gradFunc.writeMLIR(to: mlirPath)
        print("  Wrote MLIR to: \(mlirPath)")
    }

    // Create input data
    var spots = [Float](repeating: 0, count: size)
    for i in 0..<size {
        spots[i] = Float(80.0 + 40.0 * Double(i % 100) / 100.0)
    }
    // Seed is now embedded as constant 1.0 in the IR (like JAX)
    // No need to create or pass a seed array!

    // Warmup - using async execution path
    for _ in 0..<10 {
        let _ = try gradFunc.forwardWithGradientAsync(spots)
    }

    // Benchmark - pure execution time (no compilation)
    var forwardTimes: [Double] = []
    var samplePrices: [Float] = []
    var sampleDeltas: [Float] = []

    for i in 0..<trials {
        // Forward + Gradient pass (async optimized path)
        let fwdStart = currentTime()
        let result = try gradFunc.forwardWithGradientAsync(spots)
        let fwdTime = currentTime() - fwdStart
        forwardTimes.append(fwdTime)

        if i == 0 {
            samplePrices = Array(result.0.prefix(5))
            sampleDeltas = Array(result.1.prefix(5))
        }
    }

    let avgForwardGrad = forwardTimes.reduce(0, +) / Double(trials) * 1_000_000
    let minForwardGrad = forwardTimes.min()! * 1_000_000
    let mean = forwardTimes.reduce(0, +) / Double(trials)
    let variance = forwardTimes.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trials)
    let stdForwardGrad = variance.squareRoot() * 1_000_000
    let forwardGradThroughput = Double(size) / (avgForwardGrad / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: size,
        compileTimeMs: compileTime,
        forwardGradTimeUs: avgForwardGrad,
        minForwardGradTimeUs: minForwardGrad,
        stdForwardGradTimeUs: stdForwardGrad,
        forwardGradThroughput: forwardGradThroughput,
        samplePrices: samplePrices,
        sampleDeltas: sampleDeltas
    )
}

// MARK: - Main

@main
struct SwiftIRTracedBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     SwiftIR Traced Quantitative Finance Benchmark                            ║
        ║     (DifferentiableTracer + compileGradientForPJRT)                          ║
        ║     (Separated Compile/Execute Timing)                                       ║
        ║                                                                              ║
        ║     Write Swift code -> Auto-generate MLIR -> Execute via XLA                ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        // Show optimization levels
        let mlirOptLevelName: String = {
            switch selectedMLIROptLevel {
            case .none: return "none (JAX-style, let XLA optimize)"
            case .minimal: return "minimal (DCE only)"
            case .reduced: return "reduced (CSE + DCE)"
            case .standard: return "standard (7-pass pipeline)"
            case .maximum: return "maximum (7-pass + extra iterations)"
            }
        }()
        let xlaOptLevelName: String = {
            switch selectedXLAOptLevel {
            case .default: return "default (use XLA's default, typically O2)"
            case .O0: return "O0 (no optimization, fastest compile)"
            case .O1: return "O1 (basic optimization)"
            case .O2: return "O2 (standard optimization)"
            case .O3: return "O3 (maximum optimization)"
            }
        }()
        print("  MLIR Optimization: \(mlirOptLevelName)")
        print("  XLA Optimization:  \(xlaOptLevelName)")

        if dumpMLIR {
            print("  MLIR dumping enabled (--dump-mlir)")
        }
        print()

        do {
            let trials = 100
            let batchSizes = [100, 1000, 10000, 100000, 1000000]

            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Running Benchmarks at Multiple Batch Sizes                  ║")
            print("║  (Compile and Execute times measured SEPARATELY)             ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()
            print("  Trials per benchmark: \(trials)")
            print()

            var results: [BenchmarkResult] = []

            for size in batchSizes {
                print("  Running batch size: \(size)...")
                let result = try runBenchmarkForBatchSize(size, trials: trials, dumpMLIR: dumpMLIR, mlirOptLevel: selectedMLIROptLevel, xlaOptLevel: selectedXLAOptLevel)
                results.append(result)
                print("    Compile:    \(String(format: "%.1f", result.compileTimeMs)) ms")
                print("    Execute:    \(String(format: "%.1f", result.forwardGradTimeUs)) μs (±\(String(format: "%.1f", result.stdForwardGradTimeUs)))")
                print("    Throughput: \(String(format: "%.2f", result.forwardGradThroughput)) K/s")
                print()
            }

            // Print tables
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  COMPILATION TIME                                                             ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

            """)

            print("  ┌────────────┬────────────────┐")
            print("  │ Batch Size │  Compile (ms)  │")
            print("  ├────────────┼────────────────┤")
            for result in results {
                let batchStr = String(format: "%10d", result.batchSize)
                let compileStr = String(format: "%12.1f", result.compileTimeMs)
                print("  │ \(batchStr) │ \(compileStr) │")
            }
            print("  └────────────┴────────────────┘")
            print()

            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  EXECUTION TIME (pure execution, no compilation)                              ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

            """)

            print("  ┌────────────┬──────────────────┬──────────────────┬──────────────────┐")
            print("  │ Batch Size │   Avg Time (μs)  │   Min Time (μs)  │   Throughput     │")
            print("  ├────────────┼──────────────────┼──────────────────┼──────────────────┤")
            for result in results {
                let batchStr = String(format: "%10d", result.batchSize)
                let timeStr = String(format: "%14.1f", result.forwardGradTimeUs)
                let minTimeStr = String(format: "%14.1f", result.minForwardGradTimeUs)
                let throughputStr = String(format: "%12.2f K/s", result.forwardGradThroughput)
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

            // Comparison summary
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  COMPARISON SUMMARY (at batch size 100,000)                                   ║
            ╚══════════════════════════════════════════════════════════════════════════════╝
            """)

            if let r = results.first(where: { $0.batchSize == 100000 }) {
                print()
                print("  SwiftIR-Traced:")
                print("    Compile time:    \(String(format: "%.1f", r.compileTimeMs)) ms")
                print("    Execute time:    \(String(format: "%.1f", r.forwardGradTimeUs)) μs")
                print("    Throughput:      \(String(format: "%.0f", r.forwardGradThroughput)) K options/sec")
                print()
                print("  Compare with JAX at 100K batch:")
                print("    Compile time:    ~343 ms (lowering + XLA compile)")
                print("    Execute time:    ~729 μs")
                print("    Throughput:      ~137,162 K/s")
                print()
                print("  Compare with Swift _Differentiation: ~803 K/s at 100K batch")
            }
            print()

            if dumpMLIR {
                print("  MLIR dumps saved to: \(mlirOutputDir)/")
                print()
            }

        } catch {
            print("ERROR: \(error)")
        }
    }
}
