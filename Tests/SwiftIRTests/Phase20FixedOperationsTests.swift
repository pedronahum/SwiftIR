// Phase20FixedOperationsTests.swift
// Tests for Phase 20: Fixed sigmoid/sqrt/constant StableHLO generation

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase20FixedOperationsTests: XCTestCase {

    // MARK: - Constant Tests

    func testConstantMLIRGeneration() throws {
        // Test that constant generates valid MLIR
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [4],
            dtype: .float32
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return x * two
        }

        // Verify MLIR contains properly formatted constant
        let mlir = gradFunc.compiled.mlirSource
        XCTAssertTrue(mlir.contains("constant dense<"), "MLIR should have constant with dense format")
    }

    func testConstantGradientOnPJRT() throws {
        // f(x) = x * 2
        // f'(x) = 2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return x * two
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be 2 everywhere
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 2.0, accuracy: 0.01)
        }
    }

    func testConstantTimesSquare() throws {
        // f(x) = 3 * x²
        // f'(x) = 6x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let three = createConstant(3.0, shape: shape, dtype: dtype)
            return three * x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 6x
        XCTAssertEqual(gradient[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 12.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 18.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 24.0, accuracy: 0.01)
    }

    // MARK: - Sigmoid Tests

    func testSigmoidGradient() throws {
        // f(x) = sigmoid(x)
        // f'(x) = sigmoid(x) * (1 - sigmoid(x))
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSigmoid(x)
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // At x=0: sigmoid(0) = 0.5, grad = 0.25
        XCTAssertEqual(gradient[0], 0.25, accuracy: 0.01)
        // At x=1: sigmoid(1) ≈ 0.731, grad ≈ 0.197
        XCTAssertEqual(gradient[1], 0.197, accuracy: 0.02)
    }

    func testSigmoidForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSigmoid(x)
        }

        let input: [Float] = [0.0, 1.0, -1.0, 100.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // sigmoid(0) = 0.5
        XCTAssertEqual(output[0], 0.5, accuracy: 0.01)
        // sigmoid(large) ≈ 1
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)
    }

    // MARK: - Sqrt Tests

    func testSqrtGradient() throws {
        // f(x) = sqrt(x)
        // f'(x) = 1 / (2 * sqrt(x))
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSqrt(x)
        }

        let input: [Float] = [1.0, 4.0, 9.0, 16.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 1/(2*sqrt(x)) = [0.5, 0.25, 0.167, 0.125]
        XCTAssertEqual(gradient[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.25, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0/6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.125, accuracy: 0.01)
    }

    func testSqrtForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSqrt(x)
        }

        let input: [Float] = [1.0, 4.0, 9.0, 16.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // sqrt values
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)
    }

    // MARK: - Combined Tests

    func testSigmoidWithConstant() throws {
        // f(x) = sigmoid(x) * 2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return diffSigmoid(x) * two
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Output at x=0: sigmoid(0) * 2 = 1.0
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)

        // Gradient at x=0: 2 * sigmoid'(0) = 2 * 0.25 = 0.5
        XCTAssertEqual(gradient[0], 0.5, accuracy: 0.01)
    }

    func testSqrtComposed() throws {
        // f(x) = sqrt(x) * x = x^1.5
        // f'(x) = 1.5 * x^0.5 = 1.5 * sqrt(x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSqrt(x) * x
        }

        let input: [Float] = [1.0, 4.0, 9.0, 16.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 1.5 * sqrt(x)
        XCTAssertEqual(gradient[0], 1.5, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 3.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 4.5, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 6.0, accuracy: 0.01)
    }

    // MARK: - MSE Loss Test

    func testMSEStyleLoss() throws {
        // f(x) = (x - target)²
        // f'(x) = 2(x - target)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let target = createConstant(2.0, shape: shape, dtype: dtype)
            let diff = x - target
            return diff * diff
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // d/dx (x - 2)² = 2(x - 2)
        XCTAssertEqual(gradient[0], -2.0, accuracy: 0.01)  // 2(1-2) = -2
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)   // 2(2-2) = 0
        XCTAssertEqual(gradient[2], 2.0, accuracy: 0.01)   // 2(3-2) = 2
        XCTAssertEqual(gradient[3], 4.0, accuracy: 0.01)   // 2(4-2) = 4
    }

    // MARK: - Constant Binary Op Tests

    func testAddConstant() throws {
        // f(x) = x + 2
        // f'(x) = 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return x + two
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: [3, 4, 5, 6]
        XCTAssertEqual(output[0], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 5.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 6.0, accuracy: 0.01)

        // Gradient: [1, 1, 1, 1]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 1.0, accuracy: 0.01)
        }
    }

    func testSubtractConstant() throws {
        // f(x) = x - 3
        // f'(x) = 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let three = createConstant(3.0, shape: shape, dtype: dtype)
            return x - three
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: [-2, -1, 0, 1]
        XCTAssertEqual(output[0], -2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], -1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)

        // Gradient: [1, 1, 1, 1]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 1.0, accuracy: 0.01)
        }
    }

    func testDivideByConstant() throws {
        // f(x) = x / 2
        // f'(x) = 0.5
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return x / two
        }

        let input: [Float] = [2.0, 4.0, 6.0, 8.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: [1, 2, 3, 4]
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)

        // Gradient: [0.5, 0.5, 0.5, 0.5]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.5, accuracy: 0.01)
        }
    }

    func testConstantMinusX() throws {
        // f(x) = 5 - x
        // f'(x) = -1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let five = createConstant(5.0, shape: shape, dtype: dtype)
            return five - x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: [4, 3, 2, 1]
        XCTAssertEqual(output[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)

        // Gradient: [-1, -1, -1, -1]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], -1.0, accuracy: 0.01)
        }
    }
}
