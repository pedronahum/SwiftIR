// Phase10ValidationTests.swift
// Tests for Phase 10: Tracer Integration with Backend Compilation

import XCTest
@testable import SwiftIR

final class Phase10ValidationTests: XCTestCase {

    // MARK: - CompilableTracer Basic Tests

    func testCompilableTracerCreation() {
        let tracer = CompilableTracer(irValue: "%x", shape: [10, 20], dtype: .float32)

        XCTAssertEqual(tracer.irValue, "%x")
        XCTAssertEqual(tracer.shape, [10, 20])
        XCTAssertEqual(tracer.dtype, .float32)
    }

    func testCompilableTracerZero() {
        let zero = CompilableTracer.zero

        XCTAssertEqual(zero.irValue, "zeros")
        XCTAssertEqual(zero.shape, [])
    }

    // MARK: - Tracing Context Tests

    func testTracingContextInputCreation() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let input = context.input(shape: [32, 784], dtype: .float32)

        XCTAssertEqual(input.shape, [32, 784])
        XCTAssertEqual(input.dtype, .float32)
        XCTAssertTrue(input.irValue.contains("arg"))

        CompilableTracer.currentBuilder = nil
    }

    func testTracingContextMultipleInputs() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10, 20], dtype: .float32)
        let y = context.input(shape: [10, 20], dtype: .float32)

        XCTAssertNotEqual(x.irValue, y.irValue)

        CompilableTracer.currentBuilder = nil
    }

    func testTracingContextBuildModule() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        context.output(x)

        let module = context.buildModule(name: "identity")

        XCTAssertEqual(module.functionName, "identity")
        XCTAssertEqual(module.arguments.count, 1)

        CompilableTracer.currentBuilder = nil
    }

    // MARK: - Arithmetic Operations Tests

    func testTracerAddition() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = context.input(shape: [10], dtype: .float32)
        let z = x + y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(context.builder.operations.count, 1)
        XCTAssertEqual(context.builder.operations[0].opName, "add")

        CompilableTracer.currentBuilder = nil
    }

    func testTracerSubtraction() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = context.input(shape: [10], dtype: .float32)
        let z = x - y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "subtract")

        CompilableTracer.currentBuilder = nil
    }

    func testTracerMultiplication() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = context.input(shape: [10], dtype: .float32)
        let z = x * y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "multiply")

        CompilableTracer.currentBuilder = nil
    }

    func testTracerDivision() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = context.input(shape: [10], dtype: .float32)
        let z = x / y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "divide")

        CompilableTracer.currentBuilder = nil
    }

    func testTracerNegation() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let z = -x

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "negate")

        CompilableTracer.currentBuilder = nil
    }

    // MARK: - Math Operations Tests

    func testMatmul() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let a = context.input(shape: [10, 20], dtype: .float32)
        let b = context.input(shape: [20, 30], dtype: .float32)
        let c = matmul(a, b)

        XCTAssertEqual(c.shape, [10, 30])
        XCTAssertEqual(context.builder.operations[0].opName, "dot")

        CompilableTracer.currentBuilder = nil
    }

    func testTranspose() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10, 20], dtype: .float32)
        let y = transpose(x)

        XCTAssertEqual(y.shape, [20, 10])
        XCTAssertEqual(context.builder.operations[0].opName, "transpose")

        CompilableTracer.currentBuilder = nil
    }

    func testRelu() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = relu(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "maximum")

        CompilableTracer.currentBuilder = nil
    }

    func testExp() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = exp(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "exp")

        CompilableTracer.currentBuilder = nil
    }

    func testLog() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10], dtype: .float32)
        let y = log(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(context.builder.operations[0].opName, "log")

        CompilableTracer.currentBuilder = nil
    }

    func testSum() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10, 20], dtype: .float32)
        let y = sum(x, axes: [0])

        XCTAssertEqual(y.shape, [20])
        XCTAssertEqual(context.builder.operations[0].opName, "reduce_sum")

        CompilableTracer.currentBuilder = nil
    }

    func testSumAll() {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10, 20], dtype: .float32)
        let y = sum(x)

        XCTAssertEqual(y.shape, [1])
        XCTAssertEqual(context.builder.operations[0].opName, "reduce_sum")

        CompilableTracer.currentBuilder = nil
    }

    // MARK: - Function Compiler Tests

    func testCompileSimpleFunction() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [10], dtype: .float32) { x in
            return x + x
        }

        XCTAssertFalse(compiled.mlirSource.isEmpty)
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.add"))
    }

    func testCompileChainedOperations() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [10], dtype: .float32) { x in
            let y = x * x
            let z = y + x
            return z
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.multiply"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.add"))
    }

    func testCompileMultiInputFunction() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(
            inputSpecs: [
                (shape: [10], dtype: .float32),
                (shape: [10], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] + inputs[1]
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.add"))
    }

    func testCompileMatmulFunction() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(
            inputSpecs: [
                (shape: [32, 784], dtype: .float32),
                (shape: [784, 128], dtype: .float32)
            ]
        ) { inputs in
            return matmul(inputs[0], inputs[1])
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.dot_general"))
    }

    func testCompileNeuralNetworkLayer() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(
            inputSpecs: [
                (shape: [32, 784], dtype: .float32),  // x
                (shape: [784, 128], dtype: .float32), // w
                (shape: [128], dtype: .float32)      // b
            ]
        ) { inputs in
            let x = inputs[0]
            let w = inputs[1]
            let b = inputs[2]

            let xw = matmul(x, w)
            let xwb = xw + b
            return relu(xwb)
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.dot_general"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.add"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.maximum"))
    }

    func testCompileWithOptions() throws {
        let options = CompilationOptions(
            target: .cuda(deviceId: 0),
            optimizationLevel: .aggressive
        )
        let compiler = FunctionCompiler(options: options)

        let compiled = try compiler.compile(inputShape: [10], dtype: .float32) { x in
            return exp(x)
        }

        XCTAssertTrue(compiled.info.contains("CUDA"))
        XCTAssertTrue(compiled.info.contains("O3"))
    }

    // MARK: - Gradient Function Tests

    func testCompileWithGradients() throws {
        let compiler = FunctionCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
        XCTAssertFalse(gradFunc.info.isEmpty)
    }

    // MARK: - Convenience API Tests

    func testTraceConvenienceAPI() throws {
        let compiled = try trace(
            input: (shape: [10], dtype: .float32)
        ) { x in
            return -x
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.negate"))
    }

    func testTraceMultiInputConvenienceAPI() throws {
        let compiled = try trace(
            inputs: [
                (shape: [10], dtype: .float32),
                (shape: [10], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] * inputs[1]
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.multiply"))
    }

    // MARK: - TensorSpec Tests

    func testTensorSpecCreation() {
        let spec = TensorSpec(shape: [32, 784], dtype: .float32)

        XCTAssertEqual(spec.shape, [32, 784])
        XCTAssertEqual(spec.dtype, .float32)
    }

    func testTensorSpecVariadic() {
        let spec = TensorSpec.tensor(32, 784, dtype: .float32)

        XCTAssertEqual(spec.shape, [32, 784])
        XCTAssertEqual(spec.dtype, .float32)
    }

    // MARK: - Complex Graph Tests

    func testMLPForwardPass() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(
            inputSpecs: [
                (shape: [32, 784], dtype: .float32),  // x
                (shape: [784, 256], dtype: .float32), // w1
                (shape: [256], dtype: .float32),     // b1
                (shape: [256, 10], dtype: .float32), // w2
                (shape: [10], dtype: .float32)      // b2
            ]
        ) { inputs in
            let x = inputs[0]
            let w1 = inputs[1]
            let b1 = inputs[2]
            let w2 = inputs[3]
            let b2 = inputs[4]

            // First layer
            let h1 = matmul(x, w1)
            let h1b = h1 + b1
            let a1 = relu(h1b)

            // Second layer
            let h2 = matmul(a1, w2)
            let output = h2 + b2

            return output
        }

        // Should have: 2 matmuls, 2 adds, 1 relu
        let mlir = compiled.stablehloSource
        XCTAssertTrue(mlir.contains("stablehlo.dot_general"))
        XCTAssertTrue(mlir.contains("stablehlo.add"))
        XCTAssertTrue(mlir.contains("stablehlo.maximum"))
    }

    func testSoftmaxComputation() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [32, 10], dtype: .float32) { x in
            // Softmax: exp(x) / sum(exp(x))
            let expX = exp(x)
            let sumExp = sum(expX, axes: [1])
            // Note: This is simplified, real softmax needs broadcasting
            return expX
        }

        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.exponential"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.reduce"))
    }

    // MARK: - Module Generation Tests

    func testGeneratedModuleStructure() throws {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [10, 20], dtype: .float32)
        let y = context.input(shape: [10, 20], dtype: .float32)
        let z = x + y
        context.output(z)

        let module = context.buildModule(name: "add_function")

        XCTAssertEqual(module.functionName, "add_function")
        XCTAssertEqual(module.arguments.count, 2)
        XCTAssertEqual(module.operations.count, 1)
        XCTAssertEqual(module.results.count, 1)

        let mlir = module.mlirText
        XCTAssertTrue(mlir.contains("func.func @add_function"))
        XCTAssertTrue(mlir.contains("return"))

        CompilableTracer.currentBuilder = nil
    }

    // MARK: - DType Tests

    func testFloat64Tracer() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [10], dtype: .float64) { x in
            return x * x
        }

        // DType rawValue is used in MLIR
        XCTAssertTrue(compiled.mlirSource.contains("f64") || compiled.mlirSource.contains("float64"))
    }

    func testFloat16Tracer() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [10], dtype: .float16) { x in
            return x + x
        }

        // DType rawValue is used in MLIR
        XCTAssertTrue(compiled.mlirSource.contains("f16") || compiled.mlirSource.contains("float16"))
    }
}
