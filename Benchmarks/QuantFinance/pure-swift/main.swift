/// Pure Swift Quantitative Finance Benchmark
///
/// This is a BASELINE implementation using only standard Swift.
/// No MLIR, no XLA, no JIT compilation - just pure Swift for comparison.
///
/// Ported from NVIDIA's accelerated-quant-finance repository:
/// https://github.com/NVIDIA/accelerated-quant-finance
///
/// Compare this with the swiftir-xla version to see the performance
/// benefits of JIT compilation and hardware acceleration.

#if os(macOS) || os(iOS)
import Foundation
#else
import Glibc
#endif

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

// MARK: - Black-Scholes Implementation

/// Standard normal CDF (Abramowitz & Stegun approximation)
func normCDF(_ d: Double) -> Double {
    let a1 = 0.319381530
    let a2 = -0.356563782
    let a3 = 1.781477937
    let a4 = -1.821255978
    let a5 = 1.330274429
    let sqrt2pi = 2.5066282746310002

    let k = 1.0 / (1.0 + 0.2316419 * abs(d))
    let pdf = exp(-0.5 * d * d) / sqrt2pi
    let cnd = pdf * k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))

    return d > 0 ? 1.0 - cnd : cnd
}

/// Black-Scholes call option price
func blackScholesCall(spot: Double, strike: Double, rate: Double,
                       maturity: Double, volatility: Double) -> Double {
    guard volatility > 1e-8 && maturity > 1e-8 && strike > 1e-8 && spot > 1e-8 else {
        return maturity <= 1e-8 ? max(spot - strike, 0) : 0
    }

    let stdev = volatility * maturity.squareRoot()
    let d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity) / stdev
    let d2 = d1 - stdev
    let df = exp(-rate * maturity)

    return spot * normCDF(d1) - strike * df * normCDF(d2)
}

/// Black-Scholes put option price
func blackScholesPut(spot: Double, strike: Double, rate: Double,
                      maturity: Double, volatility: Double) -> Double {
    let call = blackScholesCall(spot: spot, strike: strike, rate: rate,
                                 maturity: maturity, volatility: volatility)
    return call - spot + strike * exp(-rate * maturity)
}

// MARK: - Greeks (Analytical Formulas)

func normPDF(_ x: Double) -> Double {
    return exp(-0.5 * x * x) / 2.5066282746310002
}

struct Greeks {
    let price: Double
    let delta: Double
    let gamma: Double
    let vega: Double
    let theta: Double
}

func computeGreeks(spot: Double, strike: Double, rate: Double,
                   maturity: Double, volatility: Double) -> Greeks {
    let stdev = volatility * maturity.squareRoot()
    let d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * maturity) / stdev
    let d2 = d1 - stdev
    let df = exp(-rate * maturity)

    let price = spot * normCDF(d1) - strike * df * normCDF(d2)
    let delta = normCDF(d1)
    let gamma = normPDF(d1) / (spot * stdev)
    let vega = spot * normPDF(d1) * maturity.squareRoot()
    let theta = -spot * normPDF(d1) * volatility / (2 * maturity.squareRoot())
                - rate * strike * df * normCDF(d2)

    return Greeks(price: price, delta: delta, gamma: gamma, vega: vega, theta: theta)
}

// MARK: - Benchmarks

/// Benchmark 1: Batch option pricing
func benchmarkBatchPricing(numOptions: Int) -> (time: Double, throughput: Double) {
    let spot = 100.0
    let rate = 0.05

    // Generate random-ish option parameters
    var strikes = [Double](repeating: 0, count: numOptions)
    var maturities = [Double](repeating: 0, count: numOptions)
    var volatilities = [Double](repeating: 0, count: numOptions)

    for i in 0..<numOptions {
        strikes[i] = spot * (0.8 + 0.4 * Double(i % 20) / 20.0)
        maturities[i] = 0.1 + 2.0 * Double(i % 100) / 100.0
        volatilities[i] = 0.1 + 0.3 * Double(i % 30) / 30.0
    }

    let startTime = currentTime()

    var totalPrice = 0.0
    for i in 0..<numOptions {
        let callPrice = blackScholesCall(spot: spot, strike: strikes[i], rate: rate,
                                          maturity: maturities[i], volatility: volatilities[i])
        let putPrice = blackScholesPut(spot: spot, strike: strikes[i], rate: rate,
                                        maturity: maturities[i], volatility: volatilities[i])
        totalPrice += callPrice + putPrice
    }

    let elapsed = currentTime() - startTime
    let throughput = Double(numOptions * 2) / elapsed / 1000.0  // Koptions/sec

    // Use totalPrice to prevent optimization
    _ = totalPrice

    return (elapsed, throughput)
}

/// Benchmark 2: Greek computation
func benchmarkGreeks(numOptions: Int) -> (time: Double, throughput: Double) {
    let spot = 100.0
    let rate = 0.05

    var strikes = [Double](repeating: 0, count: numOptions)
    var maturities = [Double](repeating: 0, count: numOptions)
    var volatilities = [Double](repeating: 0, count: numOptions)

    for i in 0..<numOptions {
        strikes[i] = spot * (0.8 + 0.4 * Double(i % 20) / 20.0)
        maturities[i] = 0.1 + 2.0 * Double(i % 100) / 100.0
        volatilities[i] = 0.1 + 0.3 * Double(i % 30) / 30.0
    }

    let startTime = currentTime()

    var totalDelta = 0.0
    for i in 0..<numOptions {
        let greeks = computeGreeks(spot: spot, strike: strikes[i], rate: rate,
                                    maturity: maturities[i], volatility: volatilities[i])
        totalDelta += greeks.delta
    }

    let elapsed = currentTime() - startTime
    let throughput = Double(numOptions) / elapsed / 1000.0

    _ = totalDelta

    return (elapsed, throughput)
}

/// Benchmark 3: Monte Carlo path simulation
func benchmarkMonteCarlo(numPaths: Int, numSteps: Int) -> (time: Double, pathsPerSec: Double) {
    let s0 = 100.0
    let rate = 0.05
    let volatility = 0.2
    let dt = 1.0 / 365.0

    let drift = (rate - 0.5 * volatility * volatility) * dt
    let diffusion = volatility * dt.squareRoot()

    let startTime = currentTime()

    var finalSum = 0.0
    var rngState: UInt64 = 12345

    for _ in 0..<numPaths {
        var spot = s0
        for _ in 0..<numSteps {
            // Simple xorshift RNG
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u1 = Double(rngState) / Double(UInt64.max)
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let u2 = Double(rngState) / Double(UInt64.max)

            // Box-Muller
            let z = (-2.0 * log(max(u1, 1e-10))).squareRoot() * cos(2.0 * .pi * u2)
            spot *= exp(drift + diffusion * z)
        }
        finalSum += spot
    }

    let elapsed = currentTime() - startTime
    let pathsPerSec = Double(numPaths) / elapsed

    _ = finalSum

    return (elapsed, pathsPerSec)
}

// MARK: - Main

@main
struct PureSwiftBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     Pure Swift Quantitative Finance Benchmark                                ║
        ║     (BASELINE - No MLIR, No XLA, No JIT)                                     ║
        ║                                                                              ║
        ║     Compare with swiftir-xla version for performance comparison              ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        // Warmup
        print("Warming up...")
        _ = benchmarkBatchPricing(numOptions: 1000)
        _ = benchmarkGreeks(numOptions: 1000)
        _ = benchmarkMonteCarlo(numPaths: 100, numSteps: 100)
        print()

        // Benchmark 1: Batch Pricing
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Benchmark 1: Black-Scholes Batch Pricing                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        for numOptions in [100, 1_000, 10_000, 100_000] {
            let (time, throughput) = benchmarkBatchPricing(numOptions: numOptions)
            let timeUs = time * 1_000_000
            print("  \(numOptions) options: \(formatDouble(timeUs, decimals: 1)) μs, \(formatDouble(throughput, decimals: 2)) Koptions/sec")
        }
        print()

        // Benchmark 2: Greeks (analytical Delta)
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Benchmark 2: Greek Computation (Analytical Delta)           ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        for numOptions in [100, 1_000, 10_000, 100_000] {
            let (time, throughput) = benchmarkGreeks(numOptions: numOptions)
            let timeUs = time * 1_000_000
            print("  \(numOptions) options: \(formatDouble(timeUs, decimals: 1)) μs, \(formatDouble(throughput, decimals: 2)) Koptions/sec")
        }
        print()

        // Benchmark 3: Monte Carlo
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Benchmark 3: Monte Carlo Path Simulation                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        for (paths, steps) in [(1000, 252), (10_000, 252), (100_000, 252)] {
            let (time, pathsPerSec) = benchmarkMonteCarlo(numPaths: paths, numSteps: steps)
            print("  \(paths) paths x \(steps) steps: \(formatDouble(time * 1000, decimals: 2)) ms, \(formatDouble(pathsPerSec / 1000, decimals: 2)) Kpaths/sec")
        }
        print()

        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  Summary                                                                     ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        This is the PURE SWIFT baseline. Performance is limited by:
        - Interpreted execution (no JIT compilation)
        - Single-threaded execution
        - No SIMD vectorization
        - No hardware acceleration

        Run the swiftir-xla version to compare with:
        - JIT-compiled StableHLO -> XLA
        - Automatic vectorization
        - GPU acceleration (if available)

        """)
    }
}
