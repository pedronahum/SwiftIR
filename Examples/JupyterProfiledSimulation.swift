// JupyterProfiledSimulation.swift
// Building Thermal Simulation with TensorBoard Profiling
//
// This example demonstrates:
// 1. SwiftIRJupyter's auto-tracing (JTracer) for computational graphs
// 2. PJRT Profiler integration for TensorBoard visualization
// 3. Step markers for TensorBoard Overview Page
// 4. Custom trace annotations for correlating Swift code with XLA ops
//
// Run with:
//   SWIFTIR_DEPS=/opt/swiftir-deps LD_LIBRARY_PATH=/opt/swiftir-deps/lib swift run JupyterProfiledSimulation
//
// View in TensorBoard:
//   tensorboard --logdir=/tmp/swiftir_jupyter_profile --port=6006

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

import SwiftIRJupyter
import SwiftIRProfiler
import _Differentiation
import Foundation

// MARK: - String Helpers

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// MARK: - Physical Constants

let dTime: Float = 0.1      // Timestep in seconds
let numTimesteps = 10       // Number of simulation steps per training epoch
let targetTemp: Float = 27.344767  // Target slab temperature

// MARK: - Simulation Parameters

struct SimulationParams {
    // Tube parameters
    let tubeSpacing: Double = 0.50292
    let tubeDiameter: Double = 0.019
    let tubeThickness: Double = 0.0023
    let tubeResistivity: Double = 0.35

    // Slab parameters
    let slabArea: Double = 100.0
    let slabCp: Double = 880.0
    let slabDensity: Double = 2242.58
    let slabThickness: Double = 0.101

    // Quanta (fluid) parameters
    let quantaFlow: Double = 0.0006309
    let quantaDensity: Double = 1000.0
    let quantaCp: Double = 4184.0

    // Tank parameters
    let tankCp: Double = 4184.0
    let tankMass: Double = 75.708
}

let params = SimulationParams()

// MARK: - Simulation Step (Traced)

/// Simulate one timestep of building thermal dynamics with tracing
func simulateTimestep(
    slabTemp: JTracer,
    quantaTemp: JTracer,
    tankTemp: JTracer
) -> (JTracer, JTracer, JTracer) {

    // Physical constants as JTracer values
    let tubeSpacing = JTracer(value: params.tubeSpacing, shape: .scalar, dtype: .float32)
    let tubeDiameter = JTracer(value: params.tubeDiameter, shape: .scalar, dtype: .float32)
    let tubeThickness = JTracer(value: params.tubeThickness, shape: .scalar, dtype: .float32)
    let tubeResistivity = JTracer(value: params.tubeResistivity, shape: .scalar, dtype: .float32)
    let slabArea = JTracer(value: params.slabArea, shape: .scalar, dtype: .float32)
    let slabCp = JTracer(value: params.slabCp, shape: .scalar, dtype: .float32)
    let slabDensity = JTracer(value: params.slabDensity, shape: .scalar, dtype: .float32)
    let slabThickness = JTracer(value: params.slabThickness, shape: .scalar, dtype: .float32)
    let quantaFlow = JTracer(value: params.quantaFlow, shape: .scalar, dtype: .float32)
    let quantaDensity = JTracer(value: params.quantaDensity, shape: .scalar, dtype: .float32)
    let quantaCp = JTracer(value: params.quantaCp, shape: .scalar, dtype: .float32)
    let tankCp = JTracer(value: params.tankCp, shape: .scalar, dtype: .float32)
    let tankMass = JTracer(value: params.tankMass, shape: .scalar, dtype: .float32)
    let dt = JTracer(value: Double(dTime), shape: .scalar, dtype: .float32)

    // Calculate derived quantities
    let pi = JTracer(value: 3.14159265, shape: .scalar, dtype: .float32)
    let tubeCircumference = pi * tubeDiameter
    let tubingLength = slabArea / tubeSpacing
    let tubingSurfaceArea = tubeCircumference * tubingLength

    // Fluid mass in tubes
    let four = JTracer(value: 4.0, shape: .scalar, dtype: .float32)
    let tubeVolume = tubingSurfaceArea * tubeDiameter / four
    let fluidMass = tubeVolume * quantaDensity

    // Thermal resistance and conductance
    let resistance = tubeResistivity * tubeThickness / tubingSurfaceArea
    let one = JTracer(value: 1.0, shape: .scalar, dtype: .float32)
    let conductance = one / resistance

    // STEP 1: Tank heats quanta
    let massPerTime = quantaFlow * quantaDensity
    let dTemp_tankToQuanta = tankTemp - quantaTemp
    let power_tankToQuanta = dTemp_tankToQuanta * massPerTime * quantaCp
    let quantaEnergyFromTank = power_tankToQuanta * dt
    let quantaTempRiseFromTank = quantaEnergyFromTank / (quantaCp * fluidMass)

    // STEP 2: Quanta heats slab
    let dTemp_quantaToSlab = quantaTemp - slabTemp
    let power_quantaToSlab = dTemp_quantaToSlab * conductance
    let slabMass = slabArea * slabThickness * slabDensity
    let slabEnergyChange = power_quantaToSlab * dt
    let slabTempDelta = slabEnergyChange / (slabCp * slabMass)
    let newSlabTemp = slabTemp + slabTempDelta

    // Quanta loses heat to slab
    let quantaTempDropToSlab = slabEnergyChange / (quantaCp * fluidMass)
    let newQuantaTemp = quantaTemp + quantaTempRiseFromTank - quantaTempDropToSlab

    // STEP 3: Tank loses heat
    let tankEnergyChange = power_tankToQuanta * dt
    let tankTempDelta = tankEnergyChange / (tankCp * tankMass)
    let newTankTemp = tankTemp - tankTempDelta

    return (newSlabTemp, newQuantaTemp, newTankTemp)
}

/// Run full simulation using native while loop
func runSimulation(dummy: JTracer) -> JTracer {
    let iterInitial = JTracer(value: 0.0, shape: .scalar, dtype: .float32)
    let slabInitialTemp = JTracer(value: 33.3, shape: .scalar, dtype: .float32)
    let quantaInitialTemp = JTracer(value: 33.3, shape: .scalar, dtype: .float32)
    let tankInitialTemp = JTracer(value: 70.0, shape: .scalar, dtype: .float32)

    let (_, finalSlab, _, _) = jWhileLoop4(
        initial: (iterInitial, slabInitialTemp, quantaInitialTemp, tankInitialTemp),
        condition: { state in
            let maxIter = JTracer(value: Double(numTimesteps), shape: .scalar, dtype: .float32)
            return state.0 < maxIter
        },
        body: { state in
            let (iter, slab, quanta, tank) = state
            let (newSlab, newQuanta, newTank) = simulateTimestep(
                slabTemp: slab,
                quantaTemp: quanta,
                tankTemp: tank
            )
            let one = JTracer(value: 1.0, shape: .scalar, dtype: .float32)
            let newIter = iter + one
            return (newIter, newSlab, newQuanta, newTank)
        }
    )

    return finalSlab
}

/// Loss function: squared error from target temperature
func computeLoss(dummy: JTracer) -> JTracer {
    let prediction = runSimulation(dummy: dummy)
    let target = JTracer(value: Double(targetTemp), shape: .scalar, dtype: .float32)
    let diff = prediction - target
    return diff * diff
}

// MARK: - Main Entry Point

@main
struct JupyterProfiledSimulation {
    static func main() throws {
        print("=" * 60)
        print("Building Thermal Simulation with TensorBoard Profiling")
        print("SwiftIRJupyter + PJRT Profiler Integration")
        print("=" * 60)
        print()

        // ============================================================
        // 1. Initialize SwiftIRJupyter
        // ============================================================

        print("Initializing SwiftIRJupyter...")
        try SwiftIRJupyter.shared.initialize()
        print("Backend: \(SwiftIRJupyter.shared.currentBackend?.description ?? "Unknown")")
        print()

        // ============================================================
        // 2. Check and Start PJRT Profiler
        // ============================================================

        print("-" * 60)
        print("Setting up TensorBoard Profiling")
        print("-" * 60)

        let hasPJRTProfiler = PJRTProfiler.isAvailable
        let hasTraceMeApi = PJRTProfiler.hasTraceMeApi

        print("PJRT Profiler Extension: \(hasPJRTProfiler ? "Available" : "Not available")")
        print("PJRT TraceMe API: \(hasTraceMeApi ? "Available (Swift traces in same profile)" : "Not available")")

        var profiler: PJRTProfiler? = nil

        if hasPJRTProfiler {
            print("Starting PJRT profiler...")
            profiler = try PJRTProfiler.create()
            try profiler!.start()
            print("Profiler started!")
        } else {
            print("WARNING: PJRT profiler not available - no TensorBoard profile will be generated")
        }
        print()

        // ============================================================
        // 3. Trace the Simulation (Multiple Training Steps)
        // ============================================================

        print("-" * 60)
        print("Running Profiled Simulation")
        print("-" * 60)
        print()

        // Run multiple "training epochs" with step markers
        // Each epoch traces the simulation and computes loss
        let numEpochs = 5

        for epoch in 0..<numEpochs {
            // Use pjrtTrainStep for TensorBoard Overview Page step detection
            try pjrtTrainStep(epoch) {
                print("Epoch \(epoch + 1)/\(numEpochs):")

                // Trace graph construction
                try pjrtTraced("trace_graph") {
                    let ctx = JTracingContext()
                    let dummy = ctx.input(shape: JTensorShape([]), dtype: .float32)
                    let loss = computeLoss(dummy: dummy)
                    ctx.output(loss)

                    // Build MLIR module
                    let mlir = try pjrtTraced("build_mlir") {
                        ctx.buildModule(name: "building_sim_epoch_\(epoch)")
                    }

                    print("  - Traced simulation (\(mlir.count) bytes MLIR)")

                    // Record graph size as instant event
                    if PJRTProfiler.hasTraceMeApi {
                        PJRTProfiler.traceMeInstant("mlir_size#bytes=\(mlir.count)#", level: 1)
                    }
                }

                // Simulate some actual computation time
                try pjrtTraced("simulate_computation") {
                    // In a real scenario, this would be XLA execution
                    // For now, just add some delay to show in profile
                    Thread.sleep(forTimeInterval: 0.01 * Double(epoch + 1))
                }

                print("  - Completed")
            }
        }

        print()

        // ============================================================
        // 4. Demonstrate Hierarchical Tracing
        // ============================================================

        print("-" * 60)
        print("Demonstrating Hierarchical Tracing")
        print("-" * 60)

        try pjrtTraced("hierarchical_demo") {
            try pjrtTraced("outer_scope") {
                try pjrtTraced("inner_scope_1") {
                    Thread.sleep(forTimeInterval: 0.005)
                }
                try pjrtTraced("inner_scope_2") {
                    Thread.sleep(forTimeInterval: 0.003)
                }
            }

            try pjrtTraced("parallel_operations") {
                // These will show as sequential in single-threaded trace
                for i in 1...3 {
                    try pjrtTraced("operation_\(i)") {
                        Thread.sleep(forTimeInterval: 0.002)
                    }
                }
            }
        }

        print("Hierarchical tracing complete")
        print()

        // ============================================================
        // 5. Collect and Export Profile Data
        // ============================================================

        if let profiler = profiler {
            print("-" * 60)
            print("Exporting TensorBoard Profile")
            print("-" * 60)

            try profiler.stop()
            let data = try profiler.collectData()
            print("Collected \(data.count) bytes of XSpace data")

            // Create TensorBoard log directory structure
            let logDir = "/tmp/swiftir_jupyter_profile"
            let pluginDir = "\(logDir)/plugins/profile/run_1"

            try FileManager.default.createDirectory(
                atPath: pluginDir,
                withIntermediateDirectories: true,
                attributes: nil
            )

            // Export with hostname for TensorBoard host detection
            let hostname = ProcessInfo.processInfo.hostName
            let filepath = "\(pluginDir)/\(hostname).xplane.pb"

            try PJRTProfiler.exportToFile(data, filepath: filepath)
            print("Exported to: \(filepath)")

            print()
            print("=" * 60)
            print("TensorBoard Instructions")
            print("=" * 60)
            print()
            print("1. Start TensorBoard:")
            print("   tensorboard --logdir=\(logDir) --port=6006")
            print()
            print("2. Open in browser:")
            print("   http://localhost:6006")
            print()
            print("3. Navigate to Profile tab to see:")
            print("   - Overview Page: Step timing, input pipeline")
            print("   - Trace Viewer: Timeline of all operations")
            print("   - TensorFlow Stats: Op-level statistics")
            print()
            print("Key traces to look for:")
            print("   - TrainStep#step_num=N#: Training epoch markers")
            print("   - trace_graph: Graph construction time")
            print("   - build_mlir: MLIR generation time")
            print("   - simulate_computation: Simulated XLA execution")
            print("   - hierarchical_demo: Nested trace scopes")
            print()
        }

        // ============================================================
        // 6. Final Statistics
        // ============================================================

        print("=" * 60)
        print("Profiled Simulation Complete!")
        print("=" * 60)
        print()
        print("Statistics:")
        print("  - Backend: \(SwiftIRJupyter.shared.currentBackend?.description ?? "Unknown")")
        print("  - Training epochs: \(numEpochs)")
        print("  - Timesteps per epoch: \(numTimesteps)")
        print("  - Profiler available: \(hasPJRTProfiler)")
        print("  - TraceMe API available: \(hasTraceMeApi)")
        print()
    }
}
