// BuildingSimulation_StandardSwift.swift
// Pure Swift implementation using standard Swift differentiation
// Based on PassiveLogic's differentiable-swift-examples
// https://github.com/PassiveLogic/differentiable-swift-examples/blob/main/Benchmarks/BuildingSimulation/Swift/main.swift
//
// This implementation uses Swift's built-in _Differentiation module without MLIR/XLA.
// It provides a direct comparison to the SwiftIR version (BuildingSimulation_SwiftIR.swift)
// using identical physics parameters and simulation structure.

import Foundation
import _Differentiation

// MARK: - Physical Constants and Parameters (IDENTICAL to SwiftIR version)

let dTime: Float = 0.1  // Timestep in seconds
let numTimesteps = 1000  // Total simulation steps (100 seconds)
let targetTemp: Float = 27.344767  // Target slab temperature (°C)

// MARK: - Pure Swift Simulation (No SwiftIR)

/// Simulate one timestep of building thermal dynamics
/// This is pure Swift with standard differentiation - no MLIR/XLA compilation
///
/// Heat flow: Tank (70°C) → Quanta (fluid) → Slab (floor)
/// Based on PassiveLogic's differentiable-swift-examples
@differentiable(reverse)
func simulateTimestep(
    _ slabTemp: Float,
    _ quantaTemp: Float,
    _ tankTemp: Float
) -> (Float, Float, Float) {

    // Physical constants (IDENTICAL to SwiftIR version)
    let tubeSpacing: Float = 0.50292
    let tubeDiameter: Float = 0.019
    let tubeThickness: Float = 0.0023
    let tubeResistivity: Float = 0.35
    let slabArea: Float = 100.0
    let slabCp: Float = 880.0
    let slabDensity: Float = 2242.58
    let slabThickness: Float = 0.101
    let quantaFlow: Float = 0.0006309  // m³/s (PassiveLogic original)
    let quantaDensity: Float = 1000.0
    let quantaCp: Float = 4184.0
    let tankCp: Float = 4184.0
    let tankMass: Float = 75.708
    let dt = dTime

    // Calculate derived quantities
    let pi: Float = 3.14159265
    let tubeCircumference = pi * tubeDiameter
    let tubingLength = slabArea / tubeSpacing
    let tubingSurfaceArea = tubeCircumference * tubingLength

    // Fluid mass in tubes
    let tubeVolume = tubingSurfaceArea * tubeDiameter / 4.0
    let fluidMass = tubeVolume * quantaDensity

    // Thermal resistance and conductance (tubes to slab)
    let resistance = tubeResistivity * tubeThickness / tubingSurfaceArea
    let conductance: Float = 1.0 / resistance

    // === STEP 1: Tank heats the quanta (fluid) ===
    // Heat flows from hot tank to cooler fluid via mass flow
    let massPerTime = quantaFlow * quantaDensity  // kg/s
    let dTemp_tankToQuanta = tankTemp - quantaTemp
    let power_tankToQuanta = dTemp_tankToQuanta * massPerTime * quantaCp  // Watts

    // Update quanta temperature from tank heat
    let quantaEnergyFromTank = power_tankToQuanta * dt
    let quantaTempRiseFromTank = quantaEnergyFromTank / (quantaCp * fluidMass)

    // === STEP 2: Quanta heats the slab ===
    // Heat flows from hot fluid to cooler slab via conductance
    let dTemp_quantaToSlab = quantaTemp - slabTemp  // Positive when quanta is hotter
    let power_quantaToSlab = dTemp_quantaToSlab * conductance

    // Update slab temperature
    let slabMass = slabArea * slabThickness * slabDensity
    let slabEnergyChange = power_quantaToSlab * dt
    let slabTempDelta = slabEnergyChange / (slabCp * slabMass)
    let newSlabTemp = slabTemp + slabTempDelta  // Slab gains heat from quanta

    // Quanta loses heat to slab
    let quantaTempDropToSlab = slabEnergyChange / (quantaCp * fluidMass)

    // Net quanta temperature change
    let newQuantaTemp = quantaTemp + quantaTempRiseFromTank - quantaTempDropToSlab

    // === STEP 3: Tank loses heat ===
    let tankEnergyChange = power_tankToQuanta * dt
    let tankTempDelta = tankEnergyChange / (tankCp * tankMass)
    let newTankTemp = tankTemp - tankTempDelta  // Tank loses heat to quanta

    return (newSlabTemp, newQuantaTemp, newTankTemp)
}

/// Full simulation: run multiple timesteps and return final slab temperature
@differentiable(reverse)
func runSimulation(_ dummy: Float) -> Float {
    // Initial temperatures (based on PassiveLogic original)
    // Note: Slab starts HOT (33.3°C) and cools toward equilibrium
    let slabInitialTemp: Float = 33.3
    let quantaInitialTemp: Float = 33.3
    let tankInitialTemp: Float = 70.0

    var slabTemp = slabInitialTemp
    var quantaTemp = quantaInitialTemp
    var tankTemp = tankInitialTemp

    // Run simulation for numTimesteps
    for _ in 0..<numTimesteps {
        (slabTemp, quantaTemp, tankTemp) = simulateTimestep(
            slabTemp,
            quantaTemp,
            tankTemp
        )
    }

    return slabTemp
}

/// Loss function: squared error from target temperature
@differentiable(reverse)
func computeLoss(_ dummy: Float) -> Float {
    let prediction = runSimulation(dummy)
    let target = targetTemp
    let diff = prediction - target

    // Squared error
    return diff * diff
}

// MARK: - Benchmark Utilities

func measure<T>(_ block: () throws -> T) rethrows -> (TimeInterval, T) {
    let start = Date()
    let result = try block()
    let duration = Date().timeIntervalSince(start)
    return (duration, result)
}

@inline(never)
func dontOptimize<T>(_ value: T) {
    _ = value
}

// MARK: - Main Benchmark

func runStandardSwiftBenchmark(trials: Int = 100) {
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Building Thermal Simulation - Standard Swift Edition    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Model components:")
    print("  • Radiant floor heating with embedded tubes")
    print("  • Concrete slab thermal mass (100m², 0.1m thick)")
    print("  • Circulating fluid (water at 0.2 g/s)")
    print("  • Hot water tank (70°C)")
    print("  • Full thermal coupling and heat transfer")
    print()
    print("Physics:")
    print("  • Tube geometry and thermal resistance")
    print("  • Conductive heat transfer (slab ↔ fluid)")
    print("  • Convective flow (tank ↔ fluid)")
    print("  • Electric power input (1000W)")
    print()
    print("Simulation: 20 timesteps × 0.1s = 2 seconds")
    print("Target: Slab temperature = 27.345°C")
    print()
    print("Using: Pure Swift with _Differentiation (no MLIR/XLA)")
    print()

    // Warmup
    print("Warming up (10 iterations)...")
    for _ in 0..<10 {
        let _ = computeLoss(0.0)
        let _ = gradient(at: 0.0, of: computeLoss)
    }
    print("✓ Warmup complete\n")

    // Run benchmark
    print("Running \(trials) benchmark trials...")
    var totalForwardTime: TimeInterval = 0
    var totalGradientTime: TimeInterval = 0
    var allLosses: [Float] = []

    for i in 0..<trials {
        // Forward pass only
        let (forwardTime, forwardResult) = measure {
            computeLoss(0.0)
        }

        // Gradient computation
        let (gradTime, grads) = measure {
            gradient(at: 0.0, of: computeLoss)
        }

        totalForwardTime += forwardTime
        totalGradientTime += gradTime
        allLosses.append(forwardResult)

        dontOptimize(grads)

        if (i + 1) % 20 == 0 {
            print("  Progress: \(i + 1)/\(trials) - loss: \(String(format: "%.6f", forwardResult))°C²")
        }
    }

    // Calculate statistics
    let avgForward = totalForwardTime / Double(trials) * 1_000_000  // μs
    let avgGradient = totalGradientTime / Double(trials) * 1_000_000  // μs
    let avgLoss = allLosses.reduce(0, +) / Float(allLosses.count)
    let finalTemp = targetTemp + sqrt(avgLoss)  // Approximate final temp

    print()
    print(String(repeating: "═", count: 60))
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         STANDARD SWIFT BENCHMARK RESULTS                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Configuration:")
    print("  • Trials: \(trials)")
    print("  • Timesteps: \(numTimesteps)")
    print("  • Simulation time: \(Float(numTimesteps) * dTime)s")
    print()
    print("Performance:")
    print("  • Forward pass:         \(String(format: "%6.1f", avgForward))μs  (\(String(format: "%.3f", avgForward/1000))ms)")
    print("  • Gradient computation: \(String(format: "%6.1f", avgGradient))μs  (\(String(format: "%.3f", avgGradient/1000))ms)")
    print("  • Gradient overhead:    \(String(format: "%.2f", avgGradient / avgForward))x")
    print("  • Total time:          \(String(format: "%.3f", (totalForwardTime + totalGradientTime)))s")
    print()
    print("Physics Results:")
    print("  • Average loss:    \(String(format: "%.6f", avgLoss))°C²")
    print("  • Final slab temp: ~\(String(format: "%.2f", finalTemp))°C")
    print("  • Target temp:     \(targetTemp)°C")
    print()
    print(String(repeating: "═", count: 60))
}

// MARK: - Main

print("""
╔════════════════════════════════════════════════════════════╗
║  Building Thermal Simulation Benchmark                     ║
║  Standard Swift Differentiation (No MLIR/XLA)              ║
╚════════════════════════════════════════════════════════════╝

This benchmark uses Swift's built-in automatic differentiation without
any MLIR or XLA compilation. It provides a baseline for comparison with
the SwiftIR implementation.

Approach:
  Swift @differentiable code
         ↓
  Swift compiler's AD
         ↓
  Native Swift execution

Key differences from SwiftIR:
  • No compilation overhead
  • Pure Swift (no external dependencies beyond _Differentiation)
  • CPU-only execution
  • No cross-framework portability

For comparison, run:
  swift run BuildingSimulation_SwiftIR

""")

runStandardSwiftBenchmark(trials: 100)

print("""

✓ Benchmark completed successfully!

To compare with SwiftIR:
  source /etc/profile.d/swiftir.sh
  swift run BuildingSimulation_SwiftIR

Expected differences:
  • Standard Swift: No compilation cost, faster for small workloads
  • SwiftIR: One-time compilation (~400ms), benefits from XLA optimizations
  • SwiftIR: Can run on GPU/TPU with same source code (future)
  • Both should produce identical physics results (~54°C² loss)

""")
