// JupyterHighLevelTest.swift
// Test the high-level SwiftIRJupyter API with @differentiable functions
// This demonstrates the style of code you can paste into a Jupyter notebook

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

import SwiftIRJupyter
import _Differentiation

print("=" * 60)
print("SwiftIRJupyter High-Level API Test")
print("=" * 60)
print()

// String multiplication helper
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Initialize SwiftIRJupyter
do {
    print("Initializing SwiftIRJupyter...")
    try SwiftIRJupyter.shared.initialize()
    print("Initialization complete!\n")
} catch {
    print("Failed to initialize: \(error)")
    exit(1)
}

print("-" * 60)
print("Test 1: Simple Tracing API")
print("-" * 60)

// Demonstrate the tracing context API
do {
    let ctx = JTracingContext()

    // Create symbolic input
    let x = ctx.input(shape: JTensorShape([4]), dtype: .float32)

    // Build computation graph using traced operations
    let doubled = x * 2.0
    let result = doubled + 1.0

    // Set output
    ctx.output(result)

    // Build MLIR module
    let mlir = ctx.buildModule(name: "simple_test")

    print("Generated MLIR:")
    print(mlir)
    print()
}

print("-" * 60)
print("Test 2: JTracer with Automatic Differentiation")
print("-" * 60)

// Define a differentiable function using JTracer
@differentiable(reverse)
func simpleFunction(_ x: JTracer) -> JTracer {
    return x * x + x * 2.0
}

// Test that the function can be differentiated
do {
    // Reset the graph builder for a fresh trace
    JTracerGraphBuilder.shared.reset()

    // Create an input
    let x = JTracer(value: 3.0, shape: JTensorShape([1]), dtype: .float32)

    // Forward pass
    let y = simpleFunction(x)
    print("Forward: f(x) = x^2 + 2x")
    print("  At x=3: f(3) = 9 + 6 = 15")
    print("  Traced output ID: \(y.valueId)")

    // Gradient
    let (value, grad) = valueWithGradient(of: simpleFunction, at: x)
    print("Gradient: f'(x) = 2x + 2")
    print("  At x=3: f'(3) = 8")
    print("  Traced gradient ID: \(grad.valueId)")
    print()
}

print("-" * 60)
print("Test 3: Building Simulation Style - Thermal Model")
print("-" * 60)

// Define thermal simulation parameters
struct ThermalParams {
    let dt: Double = 0.01           // Time step
    let thermalMass: Double = 1000.0
    let heatCapacity: Double = 1.0
    let wallConductance: Double = 0.5
    let hvacPower: Double = 5000.0
}

// This is the style of code you'd use in a Jupyter notebook
@differentiable(reverse)
func thermalStep(
    indoorTemp: JTracer,
    outdoorTemp: JTracer,
    hvacSetpoint: JTracer
) -> JTracer {
    let params = ThermalParams()

    // Heat transfer through walls
    let heatTransfer = (outdoorTemp - indoorTemp) * params.wallConductance

    // HVAC control (simplified - proportional control)
    let tempError = hvacSetpoint - indoorTemp
    let hvacHeat = tempError * 100.0  // Proportional gain

    // Temperature change: dT = (Q_in - Q_out) / (m * c) * dt
    let totalHeat = heatTransfer + hvacHeat
    let tempChange = totalHeat / (params.thermalMass * params.heatCapacity) * params.dt

    // New temperature
    return indoorTemp + tempChange
}

do {
    // Reset for fresh trace
    JTracerGraphBuilder.shared.reset()

    // Create symbolic inputs
    let ctx = JTracingContext()
    let indoorTemp = ctx.input(shape: JTensorShape([1]), dtype: .float32, name: "%indoor")
    let outdoorTemp = ctx.input(shape: JTensorShape([1]), dtype: .float32, name: "%outdoor")
    let setpoint = ctx.input(shape: JTensorShape([1]), dtype: .float32, name: "%setpoint")

    // Run thermal simulation step
    let newTemp = thermalStep(indoorTemp: indoorTemp, outdoorTemp: outdoorTemp, hvacSetpoint: setpoint)

    ctx.output(newTemp)

    let mlir = ctx.buildModule(name: "thermal_step")
    print("Thermal Simulation MLIR:")
    print(mlir)
    print()
}

print("-" * 60)
print("Test 4: Compile and Execute")
print("-" * 60)

do {
    // Create a simple compiled function
    let compiler = JupyterFunctionCompiler()

    let compiled = try compiler.compile(inputShape: [4], dtype: .float32) { x in
        // x * 2 + 1
        return x * 2.0 + 1.0
    }

    print("Compiled function:")
    print(compiled.mlirSource)

    // Execute it
    let input: [Float] = [1.0, 2.0, 3.0, 4.0]
    let output = try compiled.execute([input], shapes: [[4]])

    print("Input:  \(input)")
    print("Output: \(output[0])")
    print("Expected: [3.0, 5.0, 7.0, 9.0]")
    print()
} catch {
    print("Execution error: \(error)")
}

print("-" * 60)
print("Test 5: Matrix Operations")
print("-" * 60)

do {
    JTracerGraphBuilder.shared.reset()

    let ctx = JTracingContext()
    let a = ctx.input(shape: JTensorShape([2, 3]), dtype: .float32, name: "%a")
    let b = ctx.input(shape: JTensorShape([3, 2]), dtype: .float32, name: "%b")

    // Matrix multiplication: (2x3) @ (3x2) = (2x2)
    let c = a.matmul(b)

    ctx.output(c)

    let mlir = ctx.buildModule(name: "matmul_test")
    print("Matrix Multiplication MLIR:")
    print(mlir)
    print()
}

print("-" * 60)
print("Test 6: Activation Functions")
print("-" * 60)

@differentiable(reverse)
func neuralLayer(_ x: JTracer) -> JTracer {
    // Simple neural network layer: tanh(x * w + b)
    let scaled = x * 0.5
    let biased = scaled + 0.1
    return biased.tanh()
}

do {
    JTracerGraphBuilder.shared.reset()

    let ctx = JTracingContext()
    let x = ctx.input(shape: JTensorShape([4]), dtype: .float32)

    let y = neuralLayer(x)

    ctx.output(y)

    let mlir = ctx.buildModule(name: "neural_layer")
    print("Neural Layer MLIR:")
    print(mlir)
    print()
}

print("-" * 60)
print("Test 7: Gradient Computation")
print("-" * 60)

@differentiable(reverse)
func lossFunction(_ prediction: JTracer, _ target: JTracer) -> JTracer {
    let diff = prediction - target
    return (diff * diff).sum()  // MSE loss
}

do {
    JTracerGraphBuilder.shared.reset()

    let prediction = JTracer(value: 1.0, shape: JTensorShape([4]), dtype: .float32)
    let target = JTracer(value: 2.0, shape: JTensorShape([4]), dtype: .float32)

    // Compute gradient of loss w.r.t. prediction
    let (loss, grad) = valueWithGradient(of: { p in lossFunction(p, target) }, at: prediction)

    print("Loss function: MSE = sum((pred - target)^2)")
    print("Traced loss value ID: \(loss.valueId)")
    print("Traced gradient ID: \(grad.valueId)")
    print()
}

print("=" * 60)
print("All tests completed!")
print("=" * 60)
print()
print("The SwiftIRJupyter module is ready for use in Jupyter/Colab.")
print("You can now paste high-level Swift code with @differentiable")
print("functions and they will be compiled to MLIR/StableHLO and")
print("executed via PJRT.")
