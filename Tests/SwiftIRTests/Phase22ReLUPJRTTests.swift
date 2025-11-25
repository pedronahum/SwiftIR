// Phase22ReLUPJRTTests.swift
// Tests for Phase 22: ReLU gradients on PJRT

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase22ReLUPJRTTests: XCTestCase {

    // MARK: - Basic ReLU Tests

    func testReLUForward() throws {
        // Test ReLU forward pass: max(0, x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRelu(x)
        }

        let input: [Float] = [-2.0, -1.0, 0.0, 1.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // ReLU(-2) = 0, ReLU(-1) = 0, ReLU(0) = 0, ReLU(1) = 1
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)
    }

    func testReLUGradient() throws {
        // f(x) = ReLU(x)
        // f'(x) = 1 if x > 0, else 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRelu(x)
        }

        let input: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient is 0 for negative inputs, 1 for positive
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    func testReLUGradientAtZero() throws {
        // Test gradient at exactly zero (should be 0 since we use x > 0)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRelu(x)
        }

        let input: [Float] = [0.0, 0.0, 0.0, 0.0]
        let gradient = try gradFunc.gradient(input)

        // At x=0, gradient is 0 (since 0 > 0 is false)
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.01)
        }
    }

    // MARK: - Composed ReLU Tests

    func testReLUTimesX() throws {
        // f(x) = ReLU(x) * x
        // For x > 0: f(x) = x^2, f'(x) = 2x
        // For x <= 0: f(x) = 0, f'(x) = 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRelu(x) * x
        }

        let input: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: ReLU(x) * x
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)   // 0 * -2 = 0
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)   // 0 * -1 = 0
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)   // 1 * 1 = 1
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)   // 2 * 2 = 4

        // Gradient: For x > 0, d/dx(x^2) = 2x
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 4.0, accuracy: 0.01)
    }

    func testReLUWithConstant() throws {
        // f(x) = ReLU(x) * 2
        // f'(x) = 2 if x > 0, else 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            return diffRelu(x) * two
        }

        let input: [Float] = [-1.0, 0.0, 1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)

        // Gradient
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 2.0, accuracy: 0.01)
    }

    func testReLUPlusBias() throws {
        // f(x) = ReLU(x) + bias
        // f'(x) = 1 if x > 0, else 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let bias = createConstant(1.0, shape: shape, dtype: dtype)
            return diffRelu(x) + bias
        }

        let input: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: ReLU(x) + 1
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 3.0, accuracy: 0.01)

        // Gradient: same as ReLU
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - Neural Network Style Tests

    func testReLUAfterLinear() throws {
        // f(x) = ReLU(x * 2 - 1)
        // For x * 2 - 1 > 0, i.e., x > 0.5: f'(x) = 2
        // For x <= 0.5: f'(x) = 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let two = createConstant(2.0, shape: shape, dtype: dtype)
            let one = createConstant(1.0, shape: shape, dtype: dtype)
            return diffRelu(x * two - one)
        }

        let input: [Float] = [0.0, 0.5, 1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: ReLU(2x - 1)
        // x=0: ReLU(-1) = 0
        // x=0.5: ReLU(0) = 0
        // x=1: ReLU(1) = 1
        // x=2: ReLU(3) = 3
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 3.0, accuracy: 0.01)

        // Gradient: 2 if 2x - 1 > 0, else 0
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 2.0, accuracy: 0.01)
    }

    func testChainedReLU() throws {
        // f(x) = ReLU(ReLU(x) - 1)
        // For x > 1: f(x) = x - 1, f'(x) = 1
        // For 0 < x <= 1: f(x) = 0, f'(x) = 0
        // For x <= 0: f(x) = 0, f'(x) = 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let shape = withoutDerivative(at: x.shape)
            let dtype = withoutDerivative(at: x.dtype)
            let one = createConstant(1.0, shape: shape, dtype: dtype)
            return diffRelu(diffRelu(x) - one)
        }

        let input: [Float] = [-1.0, 0.5, 1.0, 2.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)   // ReLU(ReLU(-1) - 1) = ReLU(-1) = 0
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)   // ReLU(0.5 - 1) = ReLU(-0.5) = 0
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)   // ReLU(1 - 1) = ReLU(0) = 0
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)   // ReLU(2 - 1) = ReLU(1) = 1

        // Gradient
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - 2D Tensor Tests

    func testReLU2D() throws {
        // Test ReLU on 2D tensor
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffRelu(x)
        }

        let input: [Float] = [-1.0, 0.0, 1.0, -2.0, 2.0, -3.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1, 1, 1])

        // Forward
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 0.0, accuracy: 0.01)

        // Gradient
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[4], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[5], 0.0, accuracy: 0.01)
    }

    // MARK: - MLIR Generation Test

    func testReLUMLIRGeneration() throws {
        // Test that ReLU generates valid MLIR
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [4],
            dtype: .float32
        ) { x in
            return diffRelu(x)
        }

        let mlir = gradFunc.compiled.mlirSource

        // Should contain maximum operation for ReLU (raw or stablehlo prefixed)
        XCTAssertTrue(mlir.contains("maximum"), "MLIR should contain maximum for ReLU")
        // Should contain compare for gradient
        XCTAssertTrue(mlir.contains("compare"), "MLIR should contain compare for ReLU gradient")
        // Should contain select for gradient
        XCTAssertTrue(mlir.contains("select"), "MLIR should contain select for ReLU gradient")
    }
}
