//===-- main.swift - Pure Swift with _Differentiation Protocol --*- Swift -*-===//
//
// Pure Swift Quantitative Finance Benchmark with Automatic Differentiation
//
// This uses Swift's _Differentiation protocol for gradient computation:
// - @differentiable(reverse) annotation for functions
// - valueWithGradient for automatic gradient computation
// - No MLIR, No XLA - pure Swift with autodiff
//
// Compare this with:
// - pure-swift/           : Baseline (analytical Greeks, no autodiff)
// - swiftir-traced/       : SwiftIR DifferentiableTracer + XLA compilation
// - swiftir-shardy/       : SwiftIR with Shardy sharding
//
//===----------------------------------------------------------------------===//

#if os(macOS) || os(iOS)
import Foundation
#else
import Glibc
#endif
import _Differentiation

// MARK: - Timing

func currentTime() -> Double {
    #if os(macOS) || os(iOS)
    return CFAbsoluteTimeGetCurrent()
    #else
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
    #endif
}

// MARK: - Formatting

func formatDouble(_ value: Double, decimals: Int) -> String {
    #if os(macOS) || os(iOS)
    return String(format: "%.\(decimals)f", value)
    #else
    let multiplier = pow(10.0, Double(decimals))
    let rounded = (value * multiplier).rounded() / multiplier
    let intPart = Int(rounded)
    let fracPart = abs(rounded - Double(intPart)) * multiplier
    let fracInt = Int(fracPart.rounded())
    if decimals == 0 { return "\(intPart)" }
    var fracStr = "\(fracInt)"
    while fracStr.count < decimals { fracStr = "0" + fracStr }
    return "\(intPart).\(fracStr)"
    #endif
}

func padLeft(_ s: String, _ width: Int) -> String {
    if s.count >= width { return s }
    return String(repeating: " ", count: width - s.count) + s
}

func formatInt(_ value: Int, width: Int) -> String {
    return padLeft("\(value)", width)
}

// MARK: - Black-Scholes Implementation with @differentiable

/// Standard normal CDF (sigmoid approximation for differentiability)
/// Uses N(x) ≈ 1 / (1 + exp(-1.702 * x)) which is smooth and differentiable
@differentiable(reverse)
@inlinable
func normCDF(_ d: Double) -> Double {
    return 1.0 / (1.0 + exp(-1.702 * d))
}

/// Standard normal PDF (for reference - analytical formulas)
@inlinable
func normPDF(_ x: Double) -> Double {
    return exp(-0.5 * x * x) / 2.5066282746310002
}

/// Black-Scholes call option price using @differentiable
/// This allows automatic gradient computation via Swift's autodiff
@differentiable(reverse)
@inlinable
func blackScholesCall(
    spot: Double,
    strike: Double,
    rate: Double,
    maturity: Double,
    volatility: Double
) -> Double {
    let stdev = volatility * maturity.squareRoot()
    let d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity) / stdev
    let d2 = d1 - stdev
    let df = exp(-rate * maturity)

    return spot * normCDF(d1) - strike * df * normCDF(d2)
}

/// Loss function for gradient computation - computes call price
/// The gradient w.r.t. spot gives us Delta (∂C/∂S)
@differentiable(reverse)
@inlinable
func blackScholesLoss(
    spot: Double,
    strike: Double = 100.0,
    rate: Double = 0.05,
    maturity: Double = 1.0,
    volatility: Double = 0.2
) -> Double {
    return blackScholesCall(
        spot: spot,
        strike: strike,
        rate: rate,
        maturity: maturity,
        volatility: volatility
    )
}

// MARK: - Benchmark Results Structure

struct BenchmarkResult {
    let batchSize: Int
    let pricingTimeUs: Double
    let pricingThroughput: Double  // Koptions/sec
    let gradientTimeUs: Double
    let gradientThroughput: Double  // Koptions/sec
    let samplePrices: [Double]
    let sampleDeltas: [Double]
}

// MARK: - Benchmark Function

func runBenchmark(batchSize: Int, trials: Int) -> BenchmarkResult {
    // Create input data
    var spots = [Double](repeating: 0, count: batchSize)
    for i in 0..<batchSize {
        spots[i] = 80.0 + 40.0 * Double(i % 100) / 100.0
    }

    // Fixed parameters
    let strike = 100.0
    let rate = 0.05
    let maturity = 1.0
    let volatility = 0.2

    // Warmup
    for _ in 0..<10 {
        for i in 0..<batchSize {
            let _ = blackScholesCall(spot: spots[i], strike: strike, rate: rate,
                                     maturity: maturity, volatility: volatility)
        }
    }

    // Benchmark: Pricing only
    var pricingTimes: [Double] = []
    var samplePrices: [Double] = []

    for trial in 0..<trials {
        let start = currentTime()

        for i in 0..<batchSize {
            let price = blackScholesCall(spot: spots[i], strike: strike, rate: rate,
                                         maturity: maturity, volatility: volatility)
            if trial == 0 && i < 5 {
                samplePrices.append(price)
            }
        }

        let elapsed = currentTime() - start
        pricingTimes.append(elapsed)
    }

    // Warmup for gradients
    for _ in 0..<10 {
        for i in 0..<batchSize {
            let _ = valueWithGradient(at: spots[i]) { s in
                blackScholesLoss(spot: s, strike: strike, rate: rate,
                                maturity: maturity, volatility: volatility)
            }
        }
    }

    // Benchmark: Gradient (Delta) computation
    var gradientTimes: [Double] = []
    var sampleDeltas: [Double] = []

    for trial in 0..<trials {
        let start = currentTime()

        for i in 0..<batchSize {
            let (_, grad) = valueWithGradient(at: spots[i]) { s in
                blackScholesLoss(spot: s, strike: strike, rate: rate,
                                maturity: maturity, volatility: volatility)
            }
            if trial == 0 && i < 5 {
                sampleDeltas.append(grad)
            }
        }

        let elapsed = currentTime() - start
        gradientTimes.append(elapsed)
    }

    // Calculate statistics
    let avgPricingTime = pricingTimes.reduce(0, +) / Double(trials) * 1_000_000  // μs
    let avgGradientTime = gradientTimes.reduce(0, +) / Double(trials) * 1_000_000  // μs
    let pricingThroughput = Double(batchSize) / (avgPricingTime / 1_000_000) / 1000.0
    let gradientThroughput = Double(batchSize) / (avgGradientTime / 1_000_000) / 1000.0

    return BenchmarkResult(
        batchSize: batchSize,
        pricingTimeUs: avgPricingTime,
        pricingThroughput: pricingThroughput,
        gradientTimeUs: avgGradientTime,
        gradientThroughput: gradientThroughput,
        samplePrices: samplePrices,
        sampleDeltas: sampleDeltas
    )
}

// MARK: - Main

@main
struct PureSwiftDifferentiableBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     Pure Swift + _Differentiation Quantitative Finance Benchmark             ║
        ║     (Swift Autodiff - No MLIR, No XLA)                                       ║
        ║                                                                              ║
        ║     Uses @differentiable(reverse) and valueWithGradient                      ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        let trials = 100
        let batchSizes = [100, 1000, 10_000, 100_000]

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Running Benchmarks at Multiple Batch Sizes                  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print("  Trials per benchmark: \(trials)")
        print()

        var results: [BenchmarkResult] = []

        for batchSize in batchSizes {
            print("  Running batch size: \(batchSize)...")
            let result = runBenchmark(batchSize: batchSize, trials: trials)
            results.append(result)
            print("    Pricing:  \(formatDouble(result.pricingTimeUs, decimals: 1)) μs, \(formatDouble(result.pricingThroughput, decimals: 2)) Koptions/sec")
            print("    Gradient: \(formatDouble(result.gradientTimeUs, decimals: 1)) μs, \(formatDouble(result.gradientThroughput, decimals: 2)) Koptions/sec")
            print()
        }

        // Print comparison tables
        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  PRICING PERFORMANCE TABLE                                                   ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        print("  ┌────────────┬──────────────────┬──────────────────┐")
        print("  │ Batch Size │     Time (μs)    │   Throughput     │")
        print("  ├────────────┼──────────────────┼──────────────────┤")
        for result in results {
            let batchStr = formatInt(result.batchSize, width: 10)
            let timeStr = padLeft(formatDouble(result.pricingTimeUs, decimals: 1), 14)
            let throughputStr = padLeft(formatDouble(result.pricingThroughput, decimals: 2) + " K/s", 16)
            print("  │ \(batchStr) │ \(timeStr) │ \(throughputStr) │")
        }
        print("  └────────────┴──────────────────┴──────────────────┘")
        print()

        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  GRADIENT (DELTA) PERFORMANCE TABLE                                          ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        print("  ┌────────────┬──────────────────┬──────────────────┐")
        print("  │ Batch Size │     Time (μs)    │   Throughput     │")
        print("  ├────────────┼──────────────────┼──────────────────┤")
        for result in results {
            let batchStr = formatInt(result.batchSize, width: 10)
            let timeStr = padLeft(formatDouble(result.gradientTimeUs, decimals: 1), 14)
            let throughputStr = padLeft(formatDouble(result.gradientThroughput, decimals: 2) + " K/s", 16)
            print("  │ \(batchStr) │ \(timeStr) │ \(throughputStr) │")
        }
        print("  └────────────┴──────────────────┴──────────────────┘")
        print()

        // Sample results
        if let first = results.first {
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  SAMPLE RESULTS (first 5 options from batch size \(first.batchSize))                         ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

              Spot prices:     80.0, 80.4, 80.8, 81.2, 81.6
              Call prices:     \(first.samplePrices.map { formatDouble($0, decimals: 4) }.joined(separator: ", "))
              Deltas (∂C/∂S):  \(first.sampleDeltas.map { formatDouble($0, decimals: 4) }.joined(separator: ", "))

            """)
        }

        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  Implementation Notes                                                        ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        This benchmark uses Swift's _Differentiation protocol:

        1. Functions are annotated with @differentiable(reverse):
           ```swift
           @differentiable(reverse)
           func blackScholesCall(spot: Double, ...) -> Double {
               let stdev = volatility * maturity.squareRoot()
               let d1 = (log(spot / strike) + ...) / stdev
               return spot * normCDF(d1) - strike * df * normCDF(d2)
           }
           ```

        2. Gradients are computed using valueWithGradient:
           ```swift
           let (price, delta) = valueWithGradient(at: spot) { s in
               blackScholesCall(spot: s, strike: 100, ...)
           }
           ```

        Compare with:
        - pure-swift/            : Analytical Greeks (no autodiff)
        - swiftir-traced/        : DifferentiableTracer + XLA compilation
        - swiftir-shardy/        : SwiftIR with Shardy sharding

        """)
    }
}
