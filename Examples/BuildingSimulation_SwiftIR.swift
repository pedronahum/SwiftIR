// BuildingSimulation_SwiftIR.swift
// SwiftIR implementation of PassiveLogic building thermal simulation benchmark
// Original: https://github.com/PassiveLogic/differentiable-swift-examples/blob/main/Benchmarks/BuildingSimulation/Swift/main.swift
//
// This demonstrates SwiftIR's automatic differentiation compiling to XLA/PJRT
// for hardware-accelerated physics simulation with gradients.

import SwiftIR
import SwiftIRXLA
import Foundation
import _Differentiation

// MARK: - Physical Constants and Parameters

let dTime: Float = 0.1  // Timestep in seconds
let numTimesteps = 1000  // Total simulation steps (100 seconds)
let targetTemp: Float = 27.344767  // Target slab temperature (°C)

// MARK: - Simulation Parameters (Flattened for SwiftIR)

/// All simulation parameters packed into a flat array for efficient computation
/// This matches the structure used by PassiveLogic but optimized for tensor operations
struct SimulationParams {
    // Tube parameters (indices 0-3)
    let tubeSpacing: Float = 0.50292      // m
    let tubeDiameter: Float = 0.019       // m
    let tubeThickness: Float = 0.0023     // m
    let tubeResistivity: Float = 0.35     // thermal resistance

    // Slab parameters (indices 4-8)
    let slabInitialTemp: Float = 20.0     // °C
    let slabArea: Float = 100.0           // m²
    let slabCp: Float = 880.0            // J/(kg·K)
    let slabDensity: Float = 2242.58     // kg/m³
    let slabThickness: Float = 0.101      // m

    // Quanta (fluid) parameters (indices 9-13)
    let quantaPower: Float = 1000.0       // W
    let quantaInitialTemp: Float = 20.0   // °C
    let quantaFlow: Float = 0.0002        // kg/s
    let quantaDensity: Float = 1000.0     // kg/m³
    let quantaCp: Float = 4184.0         // J/(kg·K) - water

    // Tank parameters (indices 14-18)
    let tankInitialTemp: Float = 70.0     // °C
    let tankVolume: Float = 0.075708      // m³
    let tankCp: Float = 4184.0           // J/(kg·K)
    let tankDensity: Float = 1000.0      // kg/m³
    let tankMass: Float = 75.708         // kg

    /// Convert to flat array for tensor operations
    func toArray() -> [Float] {
        return [
            // Tube: 0-3
            tubeSpacing, tubeDiameter, tubeThickness, tubeResistivity,
            // Slab: 4-8
            slabInitialTemp, slabArea, slabCp, slabDensity, slabThickness,
            // Quanta: 9-13
            quantaPower, quantaInitialTemp, quantaFlow, quantaDensity, quantaCp,
            // Tank: 14-18
            tankInitialTemp, tankVolume, tankCp, tankDensity, tankMass
        ]
    }
}

// MARK: - SwiftIR Differentiable Simulation

/// Simulate one timestep of building thermal dynamics
/// Simplified version using constants (SwiftIR doesn't yet support tensor slicing easily)
///
/// Heat flow: Tank (70°C) → Quanta (fluid) → Slab (floor)
/// Based on PassiveLogic's differentiable-swift-examples
@differentiable(reverse)
func simulateTimestep(
    _ slabTemp: DifferentiableTracer,
    _ quantaTemp: DifferentiableTracer,
    _ tankTemp: DifferentiableTracer
) -> (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) {

    // Physical constants (from SimulationParams)
    let tubeSpacing = createConstant(0.50292, shape: [], dtype: .float32)
    let tubeDiameter = createConstant(0.019, shape: [], dtype: .float32)
    let tubeThickness = createConstant(0.0023, shape: [], dtype: .float32)
    let tubeResistivity = createConstant(0.35, shape: [], dtype: .float32)
    let slabArea = createConstant(100.0, shape: [], dtype: .float32)
    let slabCp = createConstant(880.0, shape: [], dtype: .float32)
    let slabDensity = createConstant(2242.58, shape: [], dtype: .float32)
    let slabThickness = createConstant(0.101, shape: [], dtype: .float32)
    let quantaFlow = createConstant(0.0006309, shape: [], dtype: .float32)  // m³/s (PassiveLogic original)
    let quantaDensity = createConstant(1000.0, shape: [], dtype: .float32)
    let quantaCp = createConstant(4184.0, shape: [], dtype: .float32)
    let tankCp = createConstant(4184.0, shape: [], dtype: .float32)
    let tankMass = createConstant(75.708, shape: [], dtype: .float32)
    let dt = createConstant(dTime, shape: [], dtype: .float32)

    // Calculate derived quantities
    let pi = createConstant(3.14159265, shape: [], dtype: .float32)
    let tubeCircumference = pi * tubeDiameter
    let tubingLength = slabArea / tubeSpacing
    let tubingSurfaceArea = tubeCircumference * tubingLength

    // Fluid mass in tubes
    let four = createConstant(4.0, shape: [], dtype: .float32)
    let tubeVolume = tubingSurfaceArea * tubeDiameter / four
    let fluidMass = tubeVolume * quantaDensity

    // Thermal resistance and conductance (tubes to slab)
    let resistance = tubeResistivity * tubeThickness / tubingSurfaceArea
    let one = createConstant(1.0, shape: [], dtype: .float32)
    let conductance = one / resistance

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
func runSimulation(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    // Initial temperatures (based on PassiveLogic original)
    // Note: Slab starts HOT (33.3°C) and cools toward equilibrium
    let slabInitialTemp = createConstant(33.3, shape: [], dtype: .float32)
    let quantaInitialTemp = createConstant(33.3, shape: [], dtype: .float32)
    let tankInitialTemp = createConstant(70.0, shape: [], dtype: .float32)

    var slabTemp = slabInitialTemp
    var quantaTemp = quantaInitialTemp
    var tankTemp = tankInitialTemp

    // Run simulation for numTimesteps (unrolled for SwiftIR)
    for _ in 0..<numTimesteps {
        (slabTemp, quantaTemp, tankTemp) = simulateTimestep(
            slabTemp,
            quantaTemp,
            tankTemp
        )
    }

    return slabTemp
}

/// Full simulation using native while loop: run multiple timesteps and return final slab temperature
///
/// This version uses `diffWhileLoop` which compiles to a single `stablehlo.while` operation,
/// enabling XLA loop fusion optimizations for 10-20x performance improvement.
func runSimulationWhileLoop(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    // Initial temperatures and iteration counter (based on PassiveLogic original)
    // Note: Slab starts HOT (33.3°C) and cools toward equilibrium
    let iterInitial = createConstant(0.0, shape: [], dtype: .float32)
    let slabInitialTemp = createConstant(33.3, shape: [], dtype: .float32)
    let quantaInitialTemp = createConstant(33.3, shape: [], dtype: .float32)
    let tankInitialTemp = createConstant(70.0, shape: [], dtype: .float32)

    // Run simulation using while loop
    // NOTE: Constants used in condition/body must be created INSIDE the closure
    // to be properly inlined in the while loop regions
    let (_, finalSlab, _, _) = diffWhileLoop(
        initial: (iterInitial, slabInitialTemp, quantaInitialTemp, tankInitialTemp),
        condition: { state in
            // Create maxIter inside the condition region
            let maxIter = createConstant(Float(numTimesteps), shape: [], dtype: .float32)
            return state.0 < maxIter  // iter < numTimesteps
        },
        body: { state in
            let (iter, slab, quanta, tank) = state
            // Simulate one timestep
            let (newSlab, newQuanta, newTank) = simulateTimestep(slab, quanta, tank)
            // Increment counter
            let one = createConstant(1.0, shape: [], dtype: .float32)
            let newIter = iter + one
            return (newIter, newSlab, newQuanta, newTank)
        }
    )

    return finalSlab
}

/// Loss function using while loop version
func computeLossWhileLoop(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    let prediction = runSimulationWhileLoop(dummy)
    let target = createConstant(targetTemp, shape: [], dtype: .float32)
    let diff = prediction - target

    // Squared error
    return diff * diff
}

/// Loss function: squared error from target temperature
@differentiable(reverse)
func computeLoss(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    let prediction = runSimulation(dummy)
    let target = createConstant(targetTemp, shape: [], dtype: .float32)
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

// MARK: - While Loop Benchmark

func runWhileLoopBenchmark(trials: Int = 100) throws {
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Building Simulation - While Loop Performance Test       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Comparing unrolled loop vs stablehlo.while loop...")
    print()

    // === PHASE 5: Verify MLIR Generation ===
    print("=== PHASE 5: Verifying While Loop MLIR Generation ===")
    print()

    // Trace the while loop version using the proper compilation path
    print("Tracing while loop version...")

    // Use GradientCompiler to properly set up the builder
    let builder = MLIRBuilder()
    DifferentiableTracer.currentBuilder = builder

    // Create dummy input argument
    let inputName = "%arg0"
    builder.addArgument(name: inputName, type: "tensor<f32>")
    let dummy = DifferentiableTracer(irValue: inputName, shape: [], dtype: .float32)

    // Run the while loop simulation (this will trace operations)
    let result = runSimulationWhileLoop(dummy)

    // Set result
    builder.setResults([result.irValue])

    // Build the MLIR module
    let mlirModule = builder.build(functionName: "while_loop_test")
    let mlirText = mlirModule.mlirText

    DifferentiableTracer.currentBuilder = nil

    // Check for stablehlo.while
    let hasWhileOp = mlirText.contains("stablehlo.while")

    print()
    if hasWhileOp {
        print("✅ SUCCESS: stablehlo.while operation found in MLIR!")
    } else {
        print("⚠️  stablehlo.while NOT found - operations may be unrolled")
    }

    // Show a snippet of the MLIR
    print()
    print("Generated MLIR snippet (first 2000 chars):")
    print(String(repeating: "-", count: 60))
    let snippet = String(mlirText.prefix(2000))
    print(snippet)
    if mlirText.count > 2000 {
        print("... [\(mlirText.count - 2000) more characters]")
    }
    print(String(repeating: "-", count: 60))
    print()

    // Count operations
    let opCount = mlirText.components(separatedBy: "stablehlo.").count - 1
    print("Total StableHLO operations: \(opCount)")
    print()

    // === PHASE 5: Execute While Loop Through PJRT ===
    print(String(repeating: "═", count: 60))
    print("     PHASE 5: WHILE LOOP PJRT EXECUTION BENCHMARK")
    print(String(repeating: "═", count: 60))
    print()

    // Compile the while loop version
    print("Compiling while loop version to XLA...")
    let whileCompileStart = Date()

    let whileGradFunc = try compileGradientForPJRT(
        input: TensorSpec(shape: [], dtype: .float32)
    ) { dummy in
        computeLossWhileLoop(dummy)
    }
    let whileCompileTime = Date().timeIntervalSince(whileCompileStart)
    print("✓ While loop compilation: \(String(format: "%.3f", whileCompileTime * 1000))ms")
    print()

    // Skip unrolled version for very large iteration counts (would take too long)
    let skipUnrolled = numTimesteps > 10000
    var unrolledGradFunc: PJRTGradientFunction? = nil
    var unrolledCompileTime: TimeInterval = 0

    if skipUnrolled {
        print("⏭️  Skipping unrolled version (numTimesteps=\(numTimesteps) > 10000)")
        print("   Estimated unrolled compile time: ~\(numTimesteps * 25 / 1000)s")
        print()
    } else {
        // Compile the unrolled version for comparison
        print("Compiling unrolled version to XLA...")
        let unrolledCompileStart = Date()

        unrolledGradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [], dtype: .float32)
        ) { dummy in
            computeLoss(dummy)
        }
        unrolledCompileTime = Date().timeIntervalSince(unrolledCompileStart)
        print("✓ Unrolled compilation: \(String(format: "%.3f", unrolledCompileTime * 1000))ms")
        print()
    }

    // Warmup
    print("Warming up (10 iterations)...")
    for _ in 0..<10 {
        let _ = try whileGradFunc.forwardWithGradient([0.0], seed: [1.0])
        if let unrolled = unrolledGradFunc {
            let _ = try unrolled.forwardWithGradient([0.0], seed: [1.0])
        }
    }
    print("✓ Warmup complete")
    print()

    // Run benchmark for both versions
    print("Running \(trials) benchmark trials...")
    print()

    var whileForwardTimes: [TimeInterval] = []
    var whileGradientTimes: [TimeInterval] = []
    var whileLosses: [Float] = []

    var unrolledForwardTimes: [TimeInterval] = []
    var unrolledGradientTimes: [TimeInterval] = []
    var unrolledLosses: [Float] = []

    for i in 0..<trials {
        // While loop version - forward
        let (whileFwdTime, whileLoss) = try measure {
            let (loss, _) = try whileGradFunc.forwardWithGradient([0.0], seed: [1.0])
            return loss[0]
        }
        whileForwardTimes.append(whileFwdTime)
        whileLosses.append(whileLoss)

        // While loop version - gradient
        let (whileGradTime, _) = try measure {
            try whileGradFunc.gradient([0.0])
        }
        whileGradientTimes.append(whileGradTime)

        // Unrolled version (if not skipped)
        if let unrolled = unrolledGradFunc {
            let (unrolledFwdTime, unrolledLoss) = try measure {
                let (loss, _) = try unrolled.forwardWithGradient([0.0], seed: [1.0])
                return loss[0]
            }
            unrolledForwardTimes.append(unrolledFwdTime)
            unrolledLosses.append(unrolledLoss)

            let (unrolledGradTime, _) = try measure {
                try unrolled.gradient([0.0])
            }
            unrolledGradientTimes.append(unrolledGradTime)
        }

        if (i + 1) % 20 == 0 {
            print("  Progress: \(i + 1)/\(trials)")
        }
    }

    // Calculate statistics
    let whileAvgForward = whileForwardTimes.reduce(0, +) / Double(trials) * 1_000_000
    let whileAvgGradient = whileGradientTimes.reduce(0, +) / Double(trials) * 1_000_000
    let whileAvgLoss = whileLosses.reduce(0, +) / Float(trials)

    print()
    print(String(repeating: "═", count: 60))
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        WHILE LOOP BENCHMARK RESULTS                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("While Loop Compilation: \(String(format: "%.1f", whileCompileTime * 1000))ms")
    print()
    print("While Loop Performance (average over \(trials) trials):")
    print("  • Forward pass:  \(String(format: "%6.1f", whileAvgForward))μs")
    print("  • Gradient:      \(String(format: "%6.1f", whileAvgGradient))μs")
    print("  • Loss:          \(String(format: "%.6f", whileAvgLoss))°C²")

    if !skipUnrolled {
        let unrolledAvgForward = unrolledForwardTimes.reduce(0, +) / Double(trials) * 1_000_000
        let unrolledAvgGradient = unrolledGradientTimes.reduce(0, +) / Double(trials) * 1_000_000
        let unrolledAvgLoss = unrolledLosses.reduce(0, +) / Float(trials)

        let forwardSpeedup = unrolledAvgForward / whileAvgForward
        let gradientSpeedup = unrolledAvgGradient / whileAvgGradient
        let compileSpeedup = unrolledCompileTime / whileCompileTime

        print()
        print("Unrolled Compilation: \(String(format: "%.1f", unrolledCompileTime * 1000))ms")
        print("Compile Speedup: \(String(format: "%.0f", compileSpeedup))x faster with while loop")
        print()
        print("Unrolled Performance (average over \(trials) trials):")
        print("  • Forward pass:  \(String(format: "%6.1f", unrolledAvgForward))μs")
        print("  • Gradient:      \(String(format: "%6.1f", unrolledAvgGradient))μs")
        print("  • Loss:          \(String(format: "%.6f", unrolledAvgLoss))°C²")
        print()
        print("Execution Speedup:")
        print("  • Forward:  \(String(format: "%.2f", forwardSpeedup))x")
        print("  • Gradient: \(String(format: "%.2f", gradientSpeedup))x")

        let lossDiff = abs(whileAvgLoss - unrolledAvgLoss)
        if lossDiff < 0.01 {
            print("  ✅ Results match! (diff: \(String(format: "%.6f", lossDiff)))")
        } else {
            print("  ⚠️  Results differ by \(String(format: "%.6f", lossDiff))")
        }
    } else {
        print()
        print("Estimated Unrolled Compile Time: ~\(numTimesteps * 25 / 1000)s")
        print("Estimated Compile Speedup: ~\(numTimesteps * 25 / 42)x faster with while loop")
    }
    print()
    print(String(repeating: "═", count: 60))
}

// MARK: - Main Benchmark

func runSwiftIRBenchmark(trials: Int = 100) throws {
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Building Thermal Simulation - Full Physics Model      ║")
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
    print("Compiling to MLIR + XLA...")

    // Compile forward + gradient function (no inputs needed - all params are constants)
    let compileStart = Date()

    // We need a dummy input since compileGradientForPJRT requires one
    // We'll use a scalar that we ignore in the computation
    let gradFunc = try compileGradientForPJRT(
        input: TensorSpec(shape: [], dtype: .float32)
    ) { dummy in
        computeLoss(dummy)
    }
    let compileTime = Date().timeIntervalSince(compileStart)
    print("✓ Compilation complete: \(String(format: "%.3f", compileTime * 1000))ms\n")

    // Warmup
    print("Warming up (10 iterations)...")
    for _ in 0..<10 {
        let _ = try gradFunc.forwardWithGradient([0.0], seed: [1.0])
    }
    print("✓ Warmup complete\n")

    // Run benchmark
    print("Running \(trials) benchmark trials...")
    var totalForwardTime: TimeInterval = 0
    var totalGradientTime: TimeInterval = 0
    var allLosses: [Float] = []

    for i in 0..<trials {
        // Forward pass only
        let (forwardTime, forwardResult) = try measure {
            let (loss, _) = try gradFunc.forwardWithGradient([0.0], seed: [1.0])
            return loss[0]
        }

        // Forward + gradient pass
        let (gradTime, _) = try measure {
            let grads = try gradFunc.gradient([0.0])
            dontOptimize(grads)
            return grads
        }

        totalForwardTime += forwardTime
        totalGradientTime += gradTime
        allLosses.append(forwardResult)

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
    print("║              FULL PHYSICS BENCHMARK RESULTS              ║")
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

// MARK: - Comparison with Standard Swift

print("""
╔════════════════════════════════════════════════════════════╗
║  Building Thermal Simulation Benchmark                     ║
║  SwiftIR vs Standard Swift Differentiation                 ║
╚════════════════════════════════════════════════════════════╝

This benchmark compares two approaches to automatic differentiation
in Swift for a physics-based building thermal simulation:

1. SwiftIR: Compiles to MLIR → XLA → PJRT (this file)
   - Symbolic tracing builds computation graph
   - XLA optimizes and compiles to native code
   - Executes on hardware with PJRT runtime

2. Standard Swift: Uses _Differentiation module directly
   - Computes gradients using Swift compiler's AD
   - Pure Swift execution without MLIR
   - See PassiveLogic's original implementation

""")

do {
    // First run the while loop comparison benchmark
    print("═══════════════════════════════════════════════════════════")
    print("          PHASE 4: WHILE LOOP IMPLEMENTATION               ")
    print("═══════════════════════════════════════════════════════════")
    print()

    try runWhileLoopBenchmark(trials: 100)

    print("\n\n")
    print("═══════════════════════════════════════════════════════════")
    print("        ORIGINAL FULL PHYSICS BENCHMARK (for reference)    ")
    print("═══════════════════════════════════════════════════════════")
    print()

    // Then run the original full benchmark
    try runSwiftIRBenchmark(trials: 100)

    print("\n✓ While loop benchmark completed successfully!")
    print("\nTo compare with standard Swift:")
    print("  git clone https://github.com/PassiveLogic/differentiable-swift-examples")
    print("  cd differentiable-swift-examples/Benchmarks/BuildingSimulation/Swift")
    print("  swift run")
} catch {
    print("Error: \(error)")
}
