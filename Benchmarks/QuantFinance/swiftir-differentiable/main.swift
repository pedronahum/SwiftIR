//===-- main.swift - SwiftIR Differentiable Quantitative Finance --*- Swift -*-===//
//
// SwiftIR Differentiable Quantitative Finance Benchmark
//
// This demonstrates the SwiftIR DifferentiableTracer API:
// - Write Black-Scholes using Swift operators (+, -, *, /, etc.)
// - DifferentiableTracer traces the computation graph
// - Generates StableHLO MLIR automatically
// - Supports Swift's automatic differentiation (@differentiable)
//
// DifferentiableTracer is the "Trojan Horse" - it looks like a number to Swift's AD
// but actually builds computation graphs for XLA execution.
//
//===------------------------------------------------------------------------===//

import SwiftIR
import _Differentiation

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

// MARK: - Black-Scholes Implementation using DifferentiableTracer

/// Standard Normal CDF approximation using sigmoid-based approximation
///
/// This uses a simpler approximation that avoids conditionals:
/// N(x) ≈ sigmoid(1.702 * x)  (fast approximation)
///
/// For demonstration purposes, we use a smooth approximation
/// that works well for option pricing.
@differentiable(reverse)
func normalCDF(_ d: DifferentiableTracer) -> DifferentiableTracer {
    // Simple sigmoid approximation: N(x) ≈ 1 / (1 + exp(-1.702 * x))
    // This is differentiable and avoids the conditional branch issue
    let scale = createConstant(1.702, shape: withoutDerivative(at: d.shape), dtype: withoutDerivative(at: d.dtype))
    return diffSigmoid(d * scale)
}

/// Black-Scholes Call Option Pricing
///
/// This is the core pricing function written using natural Swift operators.
/// DifferentiableTracer will trace all operations and generate the corresponding MLIR.
///
/// Formula:
///   d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
///   d2 = d1 - σ√T
///   Call = S·N(d1) - K·e^(-rT)·N(d2)
@differentiable(reverse)
func blackScholesCall(
    spot: DifferentiableTracer,
    strike: DifferentiableTracer,
    rate: DifferentiableTracer,
    maturity: DifferentiableTracer,
    volatility: DifferentiableTracer
) -> DifferentiableTracer {
    let half = createConstant(0.5, shape: withoutDerivative(at: spot.shape), dtype: withoutDerivative(at: spot.dtype))

    // Standard deviation of returns: stdev = σ√T
    let stdev = volatility * diffSqrt(maturity)

    // d1 = (ln(S/K) + (r + 0.5*σ²)*T) / (σ√T)
    let logSK = diffLog(spot / strike)
    let drift = (rate + volatility * volatility * half) * maturity
    let d1 = (logSK + drift) / stdev

    // d2 = d1 - σ√T
    let d2 = d1 - stdev

    // Discount factor: df = exp(-r*T)
    let df = diffExp(diffNegate(rate * maturity))

    // Call price = S*N(d1) - K*df*N(d2)
    let callPrice = spot * normalCDF(d1) - strike * df * normalCDF(d2)

    return callPrice
}

// MARK: - Helper: Format tensor type

func tensorTypeString(shape: [Int], dtype: DType) -> String {
    if shape.isEmpty {
        return "tensor<\(dtype.rawValue)>"
    } else {
        return "tensor<\(shape.map(String.init).joined(separator: "x"))x\(dtype.rawValue)>"
    }
}

// MARK: - Main

@main
struct SwiftIRDifferentiableBenchmark {
    static func main() {
        print("""

        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                                                                              ║
        ║     SwiftIR Differentiable Quantitative Finance                              ║
        ║     (DifferentiableTracer API with Swift AD Support)                         ║
        ║                                                                              ║
        ║     Write Swift code -> Auto-generate MLIR -> Ready for XLA                  ║
        ║     Supports: @differentiable, pullback, gradient composition                ║
        ║                                                                              ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        """)

        // Benchmark with 10000 options (same as other implementations)
        let batchSize = 10000

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Tracing Black-Scholes with DifferentiableTracer             ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print("  Writing Black-Scholes in natural Swift syntax...")
        print("  DifferentiableTracer automatically traces the computation graph.")
        print()

        // Create MLIR builder and set as current
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let tensorType = tensorTypeString(shape: [batchSize], dtype: .float32)

        // Add function arguments
        builder.addArgument(name: "%arg0", type: tensorType)  // spot
        builder.addArgument(name: "%arg1", type: tensorType)  // strike
        builder.addArgument(name: "%arg2", type: tensorType)  // rate
        builder.addArgument(name: "%arg3", type: tensorType)  // maturity
        builder.addArgument(name: "%arg4", type: tensorType)  // volatility

        // Create symbolic inputs (5 input tensors)
        let spot = DifferentiableTracer(irValue: "%arg0", shape: [batchSize], dtype: .float32)
        let strike = DifferentiableTracer(irValue: "%arg1", shape: [batchSize], dtype: .float32)
        let rate = DifferentiableTracer(irValue: "%arg2", shape: [batchSize], dtype: .float32)
        let maturity = DifferentiableTracer(irValue: "%arg3", shape: [batchSize], dtype: .float32)
        let volatility = DifferentiableTracer(irValue: "%arg4", shape: [batchSize], dtype: .float32)

        // Trace the Black-Scholes computation
        print("  Tracing Black-Scholes formula...")
        let callPrice = blackScholesCall(
            spot: spot,
            strike: strike,
            rate: rate,
            maturity: maturity,
            volatility: volatility
        )

        // Set output
        builder.setResults([callPrice.irValue])

        // Build MLIR module
        print("  Building MLIR module...")
        let mlirModule = builder.build(functionName: "main")

        // Convert to StableHLO
        let emitter = StableHLOEmitter()
        do {
            let stableHLOModule = try emitter.convert(mlirModule)
            let stableHLO = stableHLOModule.mlirText

            print("  Generated StableHLO: \(stableHLO.count) bytes")
            print("  Operations: \(mlirModule.operationCount)")
            print()

            // Show the full generated MLIR for debugging
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Generated StableHLO MLIR                                    ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()
            print(stableHLO)
            print()

            // PJRT Execution
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Compiling and Executing via PJRT                            ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()

            do {
                let runtime = try PJRTBackedRuntime(backend: .cpu)
                print("  Runtime: \(runtime.info)")

                print("  Compiling MLIR to XLA executable...")
                let compileStart = currentTime()
                let executable = try runtime.compile(stableHLO)
                let compileTime = currentTime() - compileStart
                print("  Compiled successfully!")
                print("  Compile time: \(String(format: "%.2f", compileTime * 1000)) ms")
                print()

                // Create input data
                print("  Creating input data (\(batchSize) options)...")
                var spots = [Float](repeating: 100.0, count: batchSize)
                var strikes = [Float](repeating: 0, count: batchSize)
                var rates = [Float](repeating: 0.05, count: batchSize)
                var maturities = [Float](repeating: 0, count: batchSize)
                var volatilities = [Float](repeating: 0, count: batchSize)

                for i in 0..<batchSize {
                    strikes[i] = Float(100.0 * (0.8 + 0.4 * Double(i % 20) / 20.0))
                    maturities[i] = Float(0.1 + 2.0 * Double(i % 100) / 100.0)
                    volatilities[i] = Float(0.1 + 0.3 * Double(i % 30) / 30.0)
                }

                // Warmup
                print("  Warming up...")
                _ = try executable.execute([spots, strikes, rates, maturities, volatilities])

                // Timed execution
                print("  Executing via XLA (timed)...")
                let execStart = currentTime()
                let outputs = try executable.execute([spots, strikes, rates, maturities, volatilities])
                let execTime = currentTime() - execStart

                print("  Execution complete! Got \(outputs.count) output(s)")
                print("  Execution time: \(String(format: "%.4f", execTime * 1000)) ms")

                let throughput = Double(batchSize) / execTime / 1000.0
                print("  Throughput: \(String(format: "%.2f", throughput)) Koptions/sec")
                print()

            } catch {
                print("  PJRT execution error: \(error)")
                print()
            }

        } catch {
            print("  Error converting to StableHLO: \(error)")
            print("  Falling back to generic MLIR...")
            print()
            print(mlirModule.mlirText)
            print()
        }

        // Demonstrate gradient computation capability
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Gradient Computation (Swift AD Integration)                 ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        print("  DifferentiableTracer supports Swift's automatic differentiation!")
        print("  You can compute gradients of Black-Scholes (the Greeks) using pullback:")
        print()
        print("    // Get the VJP (value and pullback)")
        print("    let (price, pullback) = valueWithPullback(at: spot) { s in")
        print("        blackScholesCall(spot: s, strike: k, rate: r, maturity: t, volatility: v)")
        print("    }")
        print("    // Compute Delta by running the pullback with seed=1")
        print("    let delta = pullback(DifferentiableTracer.one)")
        print()
        print("  The gradient computation generates additional MLIR for the backward pass.")
        print()

        // Now demonstrate gradient tracing using pullback
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Tracing Gradient (Delta) via Pullback                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        // Reset builder for gradient computation
        let gradBuilder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = gradBuilder

        // Add function arguments for gradient computation
        gradBuilder.addArgument(name: "%arg0", type: tensorType)  // spot
        gradBuilder.addArgument(name: "%arg1", type: tensorType)  // strike
        gradBuilder.addArgument(name: "%arg2", type: tensorType)  // rate
        gradBuilder.addArgument(name: "%arg3", type: tensorType)  // maturity
        gradBuilder.addArgument(name: "%arg4", type: tensorType)  // volatility

        // Create inputs
        let spot2 = DifferentiableTracer(irValue: "%arg0", shape: [batchSize], dtype: .float32)
        let strike2 = DifferentiableTracer(irValue: "%arg1", shape: [batchSize], dtype: .float32)
        let rate2 = DifferentiableTracer(irValue: "%arg2", shape: [batchSize], dtype: .float32)
        let maturity2 = DifferentiableTracer(irValue: "%arg3", shape: [batchSize], dtype: .float32)
        let volatility2 = DifferentiableTracer(irValue: "%arg4", shape: [batchSize], dtype: .float32)

        // Use Swift's AD pullback mechanism to trace the backward pass
        print("  Computing gradient with respect to spot (Delta) via pullback...")

        // Get value and pullback for the Black-Scholes function
        let (price, pullback) = valueWithPullback(at: spot2) { s in
            blackScholesCall(
                spot: s,
                strike: strike2,
                rate: rate2,
                maturity: maturity2,
                volatility: volatility2
            )
        }

        // Create seed of ones for the backward pass
        let seed = createConstant(1.0, shape: [batchSize], dtype: .float32)

        // Run the pullback to get Delta (∂Call/∂Spot)
        let delta = pullback(seed)

        // Set outputs (both price and delta)
        gradBuilder.setResults([price.irValue, delta.irValue])

        // Build and convert
        let gradModule = gradBuilder.build(functionName: "black_scholes_with_delta")

        do {
            let gradStableHLOModule = try emitter.convert(gradModule)
            let gradStableHLO = gradStableHLOModule.mlirText

            print("  Generated gradient StableHLO: \(gradStableHLO.count) bytes")
            print("  Operations (forward + backward): \(gradModule.operationCount)")
            print()

            // Show the gradient MLIR (truncated if too long)
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║  Gradient StableHLO (Forward + Backward Pass)                ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()
            if gradStableHLO.count > 5000 {
                let truncated = String(gradStableHLO.prefix(5000))
                print(truncated)
                print("\n... [truncated, \(gradStableHLO.count) total bytes] ...")
            } else {
                print(gradStableHLO)
            }
            print()

        } catch {
            print("  Error converting gradient to StableHLO: \(error)")
            print("  Falling back to generic MLIR...")
            print()

            let mlirText = gradModule.mlirText
            if mlirText.count > 5000 {
                print(String(mlirText.prefix(5000)))
                print("\n... [truncated, \(mlirText.count) total bytes] ...")
            } else {
                print(mlirText)
            }
            print()
        }

        // Summary
        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║  Summary                                                                     ║
        ╚══════════════════════════════════════════════════════════════════════════════╝

        This example demonstrates the DifferentiableTracer API:

        1. Write computations using natural Swift syntax with @differentiable:
           ```swift
           @differentiable(reverse)
           func blackScholesCall(spot: DifferentiableTracer, ...) -> DifferentiableTracer {
               let stdev = volatility * diffSqrt(maturity)
               let d1 = (diffLog(spot / strike) + drift) / stdev
               return spot * normalCDF(d1) - strike * df * normalCDF(d2)
           }
           ```

        2. DifferentiableTracer integrates with Swift's automatic differentiation:
           - Use valueWithPullback to get forward value + backward function
           - Use pullback to compute Greeks (Delta, Gamma, Vega, etc.)
           - Use gradient composition for higher-order derivatives

        3. The traced graph is compiled to optimized StableHLO:
           - Full Black-Scholes with normalCDF (forward pass)
           - Automatic gradient computation via AD (backward pass)
           - Ready for XLA optimization

        4. Key differences from JTracer (SwiftIRJupyter):
           - DifferentiableTracer: Supports Swift AD (@differentiable, pullback)
           - JTracer: Simpler API without AD integration
           - Both generate valid StableHLO MLIR

        Compare with:
        - pure-swift/            : Pure Swift baseline (no MLIR)
        - swiftir-xla/           : Hand-written MLIR strings
        - swiftir-traced/        : JTracer (simple tracing, no AD)
        - swiftir-differentiable/: THIS! DifferentiableTracer with Swift AD

        """)

        // Clean up
        DifferentiableTracer.currentBuilder = nil
    }
}
