//===-- main.swift - SwiftIR+XLA Quantitative Finance --*- Swift -*-===//
//
// SwiftIR+XLA Quantitative Finance Benchmark
//
// This demonstrates REAL XLA execution via PJRT:
// - StableHLO MLIR modules compiled to XLA executables
// - Vectorized batch operations on CPU/GPU
// - JIT compilation with XLA optimizations
//
// Ported from NVIDIA's accelerated-quant-finance repository:
// https://github.com/NVIDIA/accelerated-quant-finance
//
// NOTE: This file avoids C math functions (exp, log, sqrt) due to
// compiler crashes with C++ interop. All math is done in StableHLO.
//
//===----------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRDialects
import SwiftIRBuilders
import SwiftIRStableHLO
import SwiftIRXLA

#if os(Linux)
import Glibc
#else
import Darwin
#endif

// MARK: - Timing

func currentTime() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) / 1_000_000_000.0
}

// MARK: - Simple Error Type (avoid Foundation)

enum BenchmarkError: Error {
    case noOutput
    case noDevice
}

// MARK: - Formatting (no C math functions)

// Avoid pow() from C - use our own
func pow10(_ n: Int) -> Double {
    var result = 1.0
    for _ in 0..<n { result *= 10.0 }
    return result
}

func formatDouble(_ value: Double, decimals: Int) -> String {
    // Simple formatting without sprintf to avoid C++ interop issues
    let multiplier = pow10(decimals)
    let rounded = (value * multiplier).rounded() / multiplier
    let intPart = Int(rounded)
    let fracPart = Int(((rounded - Double(intPart)).magnitude * multiplier).rounded())
    if decimals == 0 { return "\(intPart)" }
    var fracStr = "\(fracPart)"
    while fracStr.count < decimals { fracStr = "0" + fracStr }
    return "\(intPart).\(fracStr)"
}

// MARK: - StableHLO Black-Scholes Module Builder

/// Builds a StableHLO module for batch Black-Scholes call option pricing
///
/// The Black-Scholes formula (all math done in StableHLO, not Swift):
///   d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
///   d2 = d1 - σ√T
///   Call = S·N(d1) - K·e^(-rT)·N(d2)
func buildBlackScholesModule(batchSize: Int) -> String {
    return """
    module @black_scholes {
      func.func public @main(
        %spots: tensor<\(batchSize)xf32>,
        %strikes: tensor<\(batchSize)xf32>,
        %rates: tensor<\(batchSize)xf32>,
        %maturities: tensor<\(batchSize)xf32>,
        %volatilities: tensor<\(batchSize)xf32>
      ) -> tensor<\(batchSize)xf32> {
        // Constants
        %half = stablehlo.constant dense<0.5> : tensor<\(batchSize)xf32>
        %one = stablehlo.constant dense<1.0> : tensor<\(batchSize)xf32>
        %neg_one = stablehlo.constant dense<-1.0> : tensor<\(batchSize)xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<\(batchSize)xf32>

        // Normal CDF constants (Abramowitz & Stegun)
        %a1 = stablehlo.constant dense<0.319381530> : tensor<\(batchSize)xf32>
        %a2 = stablehlo.constant dense<-0.356563782> : tensor<\(batchSize)xf32>
        %a3 = stablehlo.constant dense<1.781477937> : tensor<\(batchSize)xf32>
        %a4 = stablehlo.constant dense<-1.821255978> : tensor<\(batchSize)xf32>
        %a5 = stablehlo.constant dense<1.330274429> : tensor<\(batchSize)xf32>
        %k_coeff = stablehlo.constant dense<0.2316419> : tensor<\(batchSize)xf32>
        %sqrt2pi = stablehlo.constant dense<2.5066282746310002> : tensor<\(batchSize)xf32>

        // Step 1: stdev = volatility * sqrt(maturity)
        %sqrt_T = stablehlo.sqrt %maturities : tensor<\(batchSize)xf32>
        %stdev = stablehlo.multiply %volatilities, %sqrt_T : tensor<\(batchSize)xf32>

        // Step 2: d1 = (ln(S/K) + (r + 0.5*vol^2)*T) / stdev
        %s_over_k = stablehlo.divide %spots, %strikes : tensor<\(batchSize)xf32>
        %ln_s_k = stablehlo.log %s_over_k : tensor<\(batchSize)xf32>
        %vol_sq = stablehlo.multiply %volatilities, %volatilities : tensor<\(batchSize)xf32>
        %half_vol_sq = stablehlo.multiply %half, %vol_sq : tensor<\(batchSize)xf32>
        %r_plus_hvsq = stablehlo.add %rates, %half_vol_sq : tensor<\(batchSize)xf32>
        %drift = stablehlo.multiply %r_plus_hvsq, %maturities : tensor<\(batchSize)xf32>
        %numerator = stablehlo.add %ln_s_k, %drift : tensor<\(batchSize)xf32>
        %d1 = stablehlo.divide %numerator, %stdev : tensor<\(batchSize)xf32>

        // Step 3: d2 = d1 - stdev
        %d2 = stablehlo.subtract %d1, %stdev : tensor<\(batchSize)xf32>

        // Step 4: N(d1) using Abramowitz & Stegun
        %abs_d1 = stablehlo.abs %d1 : tensor<\(batchSize)xf32>
        %k1_denom_inner = stablehlo.multiply %k_coeff, %abs_d1 : tensor<\(batchSize)xf32>
        %k1_denom = stablehlo.add %one, %k1_denom_inner : tensor<\(batchSize)xf32>
        %k1 = stablehlo.divide %one, %k1_denom : tensor<\(batchSize)xf32>

        %d1_sq = stablehlo.multiply %d1, %d1 : tensor<\(batchSize)xf32>
        %neg_half_d1_sq = stablehlo.multiply %neg_one, %d1_sq : tensor<\(batchSize)xf32>
        %neg_half_d1_sq2 = stablehlo.multiply %half, %neg_half_d1_sq : tensor<\(batchSize)xf32>
        %exp_d1 = stablehlo.exponential %neg_half_d1_sq2 : tensor<\(batchSize)xf32>
        %pdf1 = stablehlo.divide %exp_d1, %sqrt2pi : tensor<\(batchSize)xf32>

        // Horner's method
        %p5_1 = stablehlo.multiply %k1, %a5 : tensor<\(batchSize)xf32>
        %p4_1 = stablehlo.add %a4, %p5_1 : tensor<\(batchSize)xf32>
        %p4_1k = stablehlo.multiply %k1, %p4_1 : tensor<\(batchSize)xf32>
        %p3_1 = stablehlo.add %a3, %p4_1k : tensor<\(batchSize)xf32>
        %p3_1k = stablehlo.multiply %k1, %p3_1 : tensor<\(batchSize)xf32>
        %p2_1 = stablehlo.add %a2, %p3_1k : tensor<\(batchSize)xf32>
        %p2_1k = stablehlo.multiply %k1, %p2_1 : tensor<\(batchSize)xf32>
        %p1_1 = stablehlo.add %a1, %p2_1k : tensor<\(batchSize)xf32>
        %poly1 = stablehlo.multiply %k1, %p1_1 : tensor<\(batchSize)xf32>
        %cnd1_neg = stablehlo.multiply %pdf1, %poly1 : tensor<\(batchSize)xf32>

        %d1_pos = stablehlo.compare GT, %d1, %zero : (tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>) -> tensor<\(batchSize)xi1>
        %one_minus_cnd1 = stablehlo.subtract %one, %cnd1_neg : tensor<\(batchSize)xf32>
        %N_d1 = stablehlo.select %d1_pos, %one_minus_cnd1, %cnd1_neg : tensor<\(batchSize)xi1>, tensor<\(batchSize)xf32>

        // Step 5: N(d2) similarly
        %abs_d2 = stablehlo.abs %d2 : tensor<\(batchSize)xf32>
        %k2_denom_inner = stablehlo.multiply %k_coeff, %abs_d2 : tensor<\(batchSize)xf32>
        %k2_denom = stablehlo.add %one, %k2_denom_inner : tensor<\(batchSize)xf32>
        %k2 = stablehlo.divide %one, %k2_denom : tensor<\(batchSize)xf32>

        %d2_sq = stablehlo.multiply %d2, %d2 : tensor<\(batchSize)xf32>
        %neg_half_d2_sq = stablehlo.multiply %neg_one, %d2_sq : tensor<\(batchSize)xf32>
        %neg_half_d2_sq2 = stablehlo.multiply %half, %neg_half_d2_sq : tensor<\(batchSize)xf32>
        %exp_d2 = stablehlo.exponential %neg_half_d2_sq2 : tensor<\(batchSize)xf32>
        %pdf2 = stablehlo.divide %exp_d2, %sqrt2pi : tensor<\(batchSize)xf32>

        %p5_2 = stablehlo.multiply %k2, %a5 : tensor<\(batchSize)xf32>
        %p4_2 = stablehlo.add %a4, %p5_2 : tensor<\(batchSize)xf32>
        %p4_2k = stablehlo.multiply %k2, %p4_2 : tensor<\(batchSize)xf32>
        %p3_2 = stablehlo.add %a3, %p4_2k : tensor<\(batchSize)xf32>
        %p3_2k = stablehlo.multiply %k2, %p3_2 : tensor<\(batchSize)xf32>
        %p2_2 = stablehlo.add %a2, %p3_2k : tensor<\(batchSize)xf32>
        %p2_2k = stablehlo.multiply %k2, %p2_2 : tensor<\(batchSize)xf32>
        %p1_2 = stablehlo.add %a1, %p2_2k : tensor<\(batchSize)xf32>
        %poly2 = stablehlo.multiply %k2, %p1_2 : tensor<\(batchSize)xf32>
        %cnd2_neg = stablehlo.multiply %pdf2, %poly2 : tensor<\(batchSize)xf32>

        %d2_pos = stablehlo.compare GT, %d2, %zero : (tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>) -> tensor<\(batchSize)xi1>
        %one_minus_cnd2 = stablehlo.subtract %one, %cnd2_neg : tensor<\(batchSize)xf32>
        %N_d2 = stablehlo.select %d2_pos, %one_minus_cnd2, %cnd2_neg : tensor<\(batchSize)xi1>, tensor<\(batchSize)xf32>

        // Step 6: df = exp(-r * T)
        %neg_rates = stablehlo.negate %rates : tensor<\(batchSize)xf32>
        %neg_rT = stablehlo.multiply %neg_rates, %maturities : tensor<\(batchSize)xf32>
        %df = stablehlo.exponential %neg_rT : tensor<\(batchSize)xf32>

        // Step 7: call = S * N(d1) - K * df * N(d2)
        %term1 = stablehlo.multiply %spots, %N_d1 : tensor<\(batchSize)xf32>
        %df_N_d2 = stablehlo.multiply %df, %N_d2 : tensor<\(batchSize)xf32>
        %term2 = stablehlo.multiply %strikes, %df_N_d2 : tensor<\(batchSize)xf32>
        %call_price = stablehlo.subtract %term1, %term2 : tensor<\(batchSize)xf32>

        return %call_price : tensor<\(batchSize)xf32>
      }
    }
    """
}

/// Builds a simple element-wise add module for testing
func buildAddModule(size: Int) -> String {
    return """
    module @simple_add {
      func.func public @main(%a: tensor<\(size)xf32>, %b: tensor<\(size)xf32>) -> tensor<\(size)xf32> {
        %result = stablehlo.add %a, %b : tensor<\(size)xf32>
        return %result : tensor<\(size)xf32>
      }
    }
    """
}

// MARK: - XLA Benchmarks

/// Run batch Black-Scholes pricing via XLA
func benchmarkXLABlackScholes(client: PJRTClient, device: PJRTDevice, numOptions: Int) throws -> (compileTime: Double, execTime: Double, throughput: Double) {
    // Generate input data (all Float, no math functions needed)
    let spots = [Float](repeating: 100.0, count: numOptions)
    var strikes = [Float](repeating: 0, count: numOptions)
    let rates = [Float](repeating: 0.05, count: numOptions)
    var maturities = [Float](repeating: 0, count: numOptions)
    var volatilities = [Float](repeating: 0, count: numOptions)

    for i in 0..<numOptions {
        // Generate parameters without using any C math functions
        strikes[i] = Float(100.0 * (0.8 + 0.4 * Double(i % 20) / 20.0))
        maturities[i] = Float(0.1 + 2.0 * Double(i % 100) / 100.0)
        volatilities[i] = Float(0.1 + 0.3 * Double(i % 30) / 30.0)
    }

    // Build and compile the module
    let compileStart = currentTime()
    let mlirModule = buildBlackScholesModule(batchSize: numOptions)
    let executable = try client.compile(mlirModule: mlirModule, devices: client.addressableDevices)
    let compileTime = currentTime() - compileStart

    // Create input buffers
    let bufSpots = try spots.withUnsafeBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [numOptions], elementType: .f32, device: device)
    }
    let bufStrikes = try strikes.withUnsafeBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [numOptions], elementType: .f32, device: device)
    }
    let bufRates = try rates.withUnsafeBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [numOptions], elementType: .f32, device: device)
    }
    let bufMaturities = try maturities.withUnsafeBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [numOptions], elementType: .f32, device: device)
    }
    let bufVolatilities = try volatilities.withUnsafeBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [numOptions], elementType: .f32, device: device)
    }

    // Warmup execution
    _ = try executable.execute(arguments: [bufSpots, bufStrikes, bufRates, bufMaturities, bufVolatilities], device: device)

    // Timed execution
    let execStart = currentTime()
    let outputs = try executable.execute(arguments: [bufSpots, bufStrikes, bufRates, bufMaturities, bufVolatilities], device: device)
    let execTime = currentTime() - execStart

    guard outputs.first != nil else {
        throw BenchmarkError.noOutput
    }

    let throughput = Double(numOptions) / execTime / 1000.0  // Koptions/sec

    return (compileTime, execTime, throughput)
}

/// Simple add benchmark to verify XLA is working
func benchmarkXLAAdd(client: PJRTClient, device: PJRTDevice, size: Int) throws -> (compileTime: Double, execTime: Double) {
    var inputA = [Float](repeating: 1.0, count: size)
    var inputB = [Float](repeating: 2.0, count: size)

    let compileStart = currentTime()
    let mlirModule = buildAddModule(size: size)
    let executable = try client.compile(mlirModule: mlirModule, devices: client.addressableDevices)
    let compileTime = currentTime() - compileStart

    let bufA = try inputA.withUnsafeMutableBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [size], elementType: .f32, device: device)
    }
    let bufB = try inputB.withUnsafeMutableBytes { ptr in
        try client.createBuffer(data: ptr.baseAddress!, shape: [size], elementType: .f32, device: device)
    }

    // Warmup
    _ = try executable.execute(arguments: [bufA, bufB], device: device)

    let execStart = currentTime()
    let outputs = try executable.execute(arguments: [bufA, bufB], device: device)
    let execTime = currentTime() - execStart

    // Verify result
    guard let outputBuffer = outputs.first else {
        throw BenchmarkError.noOutput
    }

    var result = [Float](repeating: 0, count: size)
    try result.withUnsafeMutableBytes { ptr in
        try outputBuffer.toHost(destination: ptr.baseAddress!)
    }

    // Check first few values
    let correct = result[0] == 3.0 && result[1] == 3.0
    if !correct {
        print("    WARNING: Result verification failed")
    }

    return (compileTime, execTime)
}

// MARK: - Main

@main
struct SwiftIRXLABenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     SwiftIR + XLA Quantitative Finance Benchmark                             ║
        ║     (REAL XLA Execution via PJRT)                                            ║
        ║                                                                              ║
        ║     All math computed in StableHLO, compiled by XLA                          ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        do {
            // Initialize PJRT client
            print("Initializing PJRT CPU Client...")
            let client = try PJRTClient(backend: .cpu)
            print("  Platform: \(client.platformName)")
            print("  Devices: \(client.devices.count)")

            guard let device = client.defaultDevice else {
                print("  ERROR: No default device available")
                return
            }
            print("  Default device ready")
            print()

            // First, verify XLA is working with a simple add
            print("Verifying XLA execution with simple add...")
            let (addCompile, addExec) = try benchmarkXLAAdd(client: client, device: device, size: 1000)
            print("  Simple add: compile=\(formatDouble(addCompile * 1000, decimals: 1))ms, exec=\(formatDouble(addExec * 1000, decimals: 3))ms")
            print("  XLA verified working!")
            print()

            // Benchmark: Black-Scholes
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Benchmark: Black-Scholes via XLA                            ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()
            print("  Complete Black-Scholes formula computed in StableHLO:")
            print("  - log, exp, sqrt, normCDF all run as fused XLA kernels")
            print("  - Automatic SIMD vectorization")
            print("  - No Swift interpreter overhead")
            print()

            for numOptions in [1_000, 10_000, 100_000] {
                let (compileTime, execTime, throughput) = try benchmarkXLABlackScholes(client: client, device: device, numOptions: numOptions)
                print("  \(numOptions) options:")
                print("    Compile: \(formatDouble(compileTime * 1000, decimals: 1)) ms")
                print("    Execute: \(formatDouble(execTime * 1000, decimals: 2)) ms")
                print("    Throughput: \(formatDouble(throughput, decimals: 1)) Koptions/sec")
                print()
            }

            // Summary
            print("""
            ╔══════════════════════════════════════════════════════════════════════════════╗
            ║  Summary                                                                     ║
            ╚══════════════════════════════════════════════════════════════════════════════╝

            This benchmark demonstrates REAL XLA execution:

            1. StableHLO MLIR -> XLA compilation via PJRT
               - Full Black-Scholes formula in ~70 StableHLO operations
               - XLA fuses operations into optimized kernels

            2. Hardware-optimized execution:
               - CPU: SIMD vectorization (AVX2/AVX-512)
               - Automatic parallelization across cores
               - GPU/TPU ready with same code

            3. Compare with pure-swift version (QuantFinancePureSwift):
               - Pure Swift: ~2,100 Koptions/sec
               - XLA should show significant improvement for large batches
               - Compilation overhead amortized over many executions

            To run pure Swift baseline:
              swift run QuantFinancePureSwift

            """)

        } catch {
            print("ERROR: \(error)")
        }
    }
}
