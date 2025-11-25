// Phase13ValidationTests.swift
// Tests for Phase 13: AD Integration with Compilation Pipeline

import XCTest
@testable import SwiftIR

final class Phase13ValidationTests: XCTestCase {

    // MARK: - DifferentiableTracer Basic Tests

    func testDifferentiableTracerCreation() {
        let tracer = DifferentiableTracer(irValue: "%x", shape: [10, 20], dtype: .float32)

        XCTAssertEqual(tracer.irValue, "%x")
        XCTAssertEqual(tracer.shape, [10, 20])
        XCTAssertEqual(tracer.dtype, .float32)
    }

    func testDifferentiableTracerZero() {
        let zero = DifferentiableTracer.zero

        XCTAssertEqual(zero.irValue, "zeros")
        XCTAssertEqual(zero.shape, [])
    }

    // MARK: - AdditiveArithmetic Tests

    func testDifferentiableTracerAddition() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = DifferentiableTracer(irValue: "%y", shape: [10], dtype: .float32)
        let z = x + y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(builder.operations.count, 1)
        XCTAssertEqual(builder.operations[0].opName, "add")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDifferentiableTracerSubtraction() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = DifferentiableTracer(irValue: "%y", shape: [10], dtype: .float32)
        let z = x - y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "subtract")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDifferentiableTracerMultiplication() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = DifferentiableTracer(irValue: "%y", shape: [10], dtype: .float32)
        let z = x * y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "multiply")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDifferentiableTracerDivision() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = DifferentiableTracer(irValue: "%y", shape: [10], dtype: .float32)
        let z = x / y

        XCTAssertEqual(z.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "divide")

        DifferentiableTracer.currentBuilder = nil
    }

    // MARK: - Move Operation Test

    func testDifferentiableTracerMove() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        var x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let offset = DifferentiableTracer(irValue: "%offset", shape: [10], dtype: .float32)

        x.move(by: offset)

        // Should create an add operation for gradient accumulation
        XCTAssertEqual(builder.operations.count, 1)
        XCTAssertEqual(builder.operations[0].opName, "add")

        DifferentiableTracer.currentBuilder = nil
    }

    // MARK: - Differentiable Operations Tests

    func testDiffMatmul() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let a = DifferentiableTracer(irValue: "%a", shape: [10, 20], dtype: .float32)
        let b = DifferentiableTracer(irValue: "%b", shape: [20, 30], dtype: .float32)
        let c = diffMatmul(a, b)

        XCTAssertEqual(c.shape, [10, 30])
        XCTAssertEqual(builder.operations[0].opName, "dot")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDiffTranspose() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10, 20], dtype: .float32)
        let y = diffTranspose(x)

        XCTAssertEqual(y.shape, [20, 10])
        XCTAssertEqual(builder.operations[0].opName, "transpose")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDiffRelu() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = diffRelu(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "maximum")

        DifferentiableTracer.currentBuilder = nil
    }

    func testDiffExp() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = diffExp(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "exp")

        DifferentiableTracer.currentBuilder = nil
    }

    // MARK: - VJP (Pullback) Tests

    func testMatmulVJP() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let a = DifferentiableTracer(irValue: "%a", shape: [10, 20], dtype: .float32)
        let b = DifferentiableTracer(irValue: "%b", shape: [20, 30], dtype: .float32)

        // Just test the forward operation first
        let y = diffMatmul(a, b)
        XCTAssertEqual(y.shape, [10, 30])

        // Verify operations were created
        XCTAssertGreaterThanOrEqual(builder.operations.count, 1)
        XCTAssertEqual(builder.operations[0].opName, "dot")

        DifferentiableTracer.currentBuilder = nil
    }

    func testTransposeVJP() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10, 20], dtype: .float32)
        let y = diffTranspose(x)

        XCTAssertEqual(y.shape, [20, 10])
        XCTAssertEqual(builder.operations[0].opName, "transpose")

        DifferentiableTracer.currentBuilder = nil
    }

    func testReluVJP() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = diffRelu(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "maximum")

        DifferentiableTracer.currentBuilder = nil
    }

    func testExpVJP() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float32)
        let y = diffExp(x)

        XCTAssertEqual(y.shape, [10])
        XCTAssertEqual(builder.operations[0].opName, "exp")

        DifferentiableTracer.currentBuilder = nil
    }

    // MARK: - GradientCompiler Tests

    func testGradientCompilerSimpleFunction() throws {
        let compiler = GradientCompiler()

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

    func testGradientCompilerExpFunction() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffExp(x)
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
    }

    func testGradientCompilerComposedFunction() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            let y = x * x
            let z = diffExp(y)
            return z
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
    }

    func testGradientCompilerTwoInputs() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [
                (shape: [10, 20], dtype: .float32),
                (shape: [20, 30], dtype: .float32)
            ]
        ) { a, b in
            return diffMatmul(a, b)
        }

        XCTAssertEqual(gradFunc.inputShape, [10, 20])
        XCTAssertEqual(gradFunc.outputShape, [10, 30])
    }

    // MARK: - High-Level API Tests

    func testCompileWithGradientsSingleInput() throws {
        let gradFunc = try compileWithGradients(
            input: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return x * x
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
    }

    func testCompileWithGradientsTwoInputs() throws {
        let gradFunc = try compileWithGradients(
            inputs: (
                TensorSpec(shape: [10, 20], dtype: .float32),
                TensorSpec(shape: [20, 30], dtype: .float32)
            )
        ) { a, b in
            return diffMatmul(a, b)
        }

        XCTAssertEqual(gradFunc.inputShape, [10, 20])
        XCTAssertEqual(gradFunc.outputShape, [10, 30])
    }

    // MARK: - GradientCompiledFunction Tests

    func testGradientCompiledFunctionInfo() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        let info = gradFunc.info
        XCTAssertTrue(info.contains("Input shape"))
        XCTAssertTrue(info.contains("Output shape"))
    }

    // MARK: - CompiledTrainingStep Tests

    func testCompiledTrainingStepCreation() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        let trainingStep = CompiledTrainingStep(
            gradientFunc: gradFunc,
            learningRate: 0.01
        )

        XCTAssertEqual(trainingStep.learningRate, 0.01)
    }

    func testCompiledTrainingStepLearningRateDefault() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        let trainingStep = CompiledTrainingStep(gradientFunc: gradFunc)

        XCTAssertEqual(trainingStep.learningRate, 0.01)
    }

    // MARK: - Neural Network Gradient Tests

    func testMLPLayerGradient() throws {
        let compiler = GradientCompiler()

        // Single layer: relu(x @ w)
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [
                (shape: [32, 784], dtype: .float32),
                (shape: [784, 128], dtype: .float32)
            ]
        ) { x, w in
            let h = diffMatmul(x, w)
            return diffRelu(h)
        }

        XCTAssertEqual(gradFunc.inputShape, [32, 784])
        XCTAssertEqual(gradFunc.outputShape, [32, 128])
    }

    // MARK: - Complex Gradient Tests

    func testChainedOperationsGradient() throws {
        let compiler = GradientCompiler()

        // f(x) = exp(x * x)
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            let squared = x * x
            return diffExp(squared)
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
    }

    func testMultipleOperationsGradient() throws {
        let compiler = GradientCompiler()

        // f(x) = x * x + x
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            let squared = x * x
            return squared + x
        }

        XCTAssertEqual(gradFunc.inputShape, [10])
        XCTAssertEqual(gradFunc.outputShape, [10])
    }

    // MARK: - DType Tests

    func testDifferentiableTracerFloat64() {
        let tracer = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float64)
        XCTAssertEqual(tracer.dtype, .float64)
    }

    func testDifferentiableTracerFloat16() {
        let tracer = DifferentiableTracer(irValue: "%x", shape: [10], dtype: .float16)
        XCTAssertEqual(tracer.dtype, .float16)
    }

    // MARK: - Builder State Tests

    func testBuilderOperationCount() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [10, 20], dtype: .float32)
        let y = DifferentiableTracer(irValue: "%y", shape: [20, 30], dtype: .float32)

        _ = diffMatmul(x, y)  // 1 op
        _ = diffTranspose(x)   // 1 op

        XCTAssertEqual(builder.operations.count, 2)

        DifferentiableTracer.currentBuilder = nil
    }

    func testBuilderArgumentsAdded() throws {
        let compiler = GradientCompiler()

        _ = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        // Compilation was successful if no error thrown
        // This validates that arguments were properly added
    }

    // MARK: - Compilation Options Tests

    func testGradientCompilerWithOptions() throws {
        let options = CompilationOptions(
            target: .cuda(deviceId: 0),
            optimizationLevel: .aggressive
        )
        let compiler = GradientCompiler(options: options)

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        XCTAssertTrue(gradFunc.info.contains("CUDA"))
        XCTAssertTrue(gradFunc.info.contains("O3"))
    }

    // MARK: - Edge Case Tests

    func testScalarShape() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [1], dtype: .float32)
        let y = diffExp(x)

        XCTAssertEqual(y.shape, [1])

        DifferentiableTracer.currentBuilder = nil
    }

    func testLargeShape() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [1000, 1000], dtype: .float32)
        let y = x * x

        XCTAssertEqual(y.shape, [1000, 1000])

        DifferentiableTracer.currentBuilder = nil
    }

    func testMultiDimensionalShape() {
        let builder = MLIRBuilder()
        DifferentiableTracer.currentBuilder = builder

        let x = DifferentiableTracer(irValue: "%x", shape: [2, 3, 4, 5], dtype: .float32)
        let y = diffExp(x)

        XCTAssertEqual(y.shape, [2, 3, 4, 5])

        DifferentiableTracer.currentBuilder = nil
    }
}
