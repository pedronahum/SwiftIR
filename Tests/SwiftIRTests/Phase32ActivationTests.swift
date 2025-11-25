// Phase32ActivationTests.swift
// Tests for Phase 32: Additional Activation Functions

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase32ActivationTests: XCTestCase {

    // MARK: - LeakyReLU Tests

    func testLeakyReLUForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLeakyReLU(x, alpha: 0.1)
        }

        let input: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // LeakyReLU: x if x > 0, else 0.1 * x
        XCTAssertEqual(output[0], -0.2, accuracy: 0.01)  // 0.1 * -2
        XCTAssertEqual(output[1], -0.1, accuracy: 0.01)  // 0.1 * -1
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 2.0, accuracy: 0.01)
    }

    func testLeakyReLUGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLeakyReLU(x, alpha: 0.1)
        }

        let input: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient: 1 if x > 0, else alpha
        XCTAssertEqual(gradient[0], 0.1, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.1, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - ELU Tests

    func testELUForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffELU(x, alpha: 1.0)
        }

        let input: [Float] = [-2.0, 0.0, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // ELU: x if x > 0, else alpha * (exp(x) - 1)
        XCTAssertEqual(output[0], 1.0 * (exp(-2.0) - 1.0), accuracy: 0.01)  // ≈ -0.865
        XCTAssertEqual(output[1], 1.0 * (exp(0.0) - 1.0), accuracy: 0.01)   // 0
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 2.0, accuracy: 0.01)
    }

    func testELUGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffELU(x, alpha: 1.0)
        }

        let input: [Float] = [-1.0, 0.5, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient: 1 if x > 0, else alpha * exp(x)
        XCTAssertEqual(gradient[0], exp(-1.0), accuracy: 0.01)  // ≈ 0.368
        XCTAssertEqual(gradient[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - SiLU/Swish Tests

    func testSiLUForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSiLU(x)
        }

        let input: [Float] = [-2.0, 0.0, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // SiLU: x * sigmoid(x)
        func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
        XCTAssertEqual(output[0], -2.0 * sigmoid(-2.0), accuracy: 0.01)  // ≈ -0.238
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.0 * sigmoid(1.0), accuracy: 0.01)    // ≈ 0.731
        XCTAssertEqual(output[3], 2.0 * sigmoid(2.0), accuracy: 0.01)    // ≈ 1.762
    }

    func testSiLUGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffSiLU(x)
        }

        let input: [Float] = [0.0, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should exist and be reasonable
        XCTAssertEqual(gradient.count, 3)
        // At x=0: grad = sigmoid(0) * (1 + 0 - 0) = 0.5
        XCTAssertEqual(gradient[0], 0.5, accuracy: 0.01)
    }

    // MARK: - GELU Tests

    func testGELUForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffGELU(x)
        }

        let input: [Float] = [-2.0, 0.0, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // At x=0: GELU(0) = 0
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        // GELU is approximately ReLU-like: negative for negative x, positive for positive x
        XCTAssertLessThan(output[0], 0.0)
        XCTAssertGreaterThan(output[2], 0.0)
        XCTAssertGreaterThan(output[3], 0.0)
    }

    func testGELUGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffGELU(x)
        }

        let input: [Float] = [0.0, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should exist
        XCTAssertEqual(gradient.count, 3)
        // At x=0: grad ≈ 0.5
        XCTAssertEqual(gradient[0], 0.5, accuracy: 0.05)
    }

    // MARK: - Softplus Tests

    func testSoftplusForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSoftplus(x)
        }

        let input: [Float] = [-2.0, 0.0, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Softplus: log(1 + exp(x))
        XCTAssertEqual(output[0], log(1.0 + exp(-2.0)), accuracy: 0.01)  // ≈ 0.127
        XCTAssertEqual(output[1], log(2.0), accuracy: 0.01)              // ≈ 0.693
        XCTAssertEqual(output[2], log(1.0 + exp(1.0)), accuracy: 0.01)   // ≈ 1.313
        XCTAssertEqual(output[3], log(1.0 + exp(2.0)), accuracy: 0.01)   // ≈ 2.127
    }

    func testSoftplusGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSoftplus(x)
        }

        let input: [Float] = [-2.0, 0.0, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient = sigmoid(x)
        func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
        XCTAssertEqual(gradient[0], sigmoid(-2.0), accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[2], sigmoid(1.0), accuracy: 0.01)
        XCTAssertEqual(gradient[3], sigmoid(2.0), accuracy: 0.01)
    }

    // MARK: - LogSoftmax Tests

    func testLogSoftmaxForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffLogSoftmax(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1])

        // LogSoftmax values should be negative (log of probabilities < 1)
        XCTAssertLessThan(output[0], 0.0)
        XCTAssertLessThan(output[1], 0.0)
        XCTAssertLessThan(output[2], 0.0)

        // exp(log_softmax) should sum to 1
        let probs = output.map { exp($0) }
        let sum = probs.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01)
    }

    func testLogSoftmaxGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffLogSoftmax(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should exist
        XCTAssertEqual(gradient.count, 3)
    }
}
