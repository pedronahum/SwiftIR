// Phase18TranscendentalPJRTTests.swift
// Tests for Phase 18: Transcendental Function Gradients on PJRT
//
// These tests verify that exp, log, sigmoid, tanh operations execute correctly
// on real PJRT hardware with numerically correct gradients.

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase18TranscendentalPJRTTests: XCTestCase {

    // MARK: - Exponential Tests

    func testExpGradient() throws {
        // f(x) = exp(x)
        // f'(x) = exp(x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x)
        }

        let input: [Float] = [0.0, 1.0, 2.0, -1.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: exp(x) = [1, e, e², 1/e]
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], Float(M_E), accuracy: 0.01)
        XCTAssertEqual(gradient[2], Float(M_E * M_E), accuracy: 0.1)
        XCTAssertEqual(gradient[3], 1.0 / Float(M_E), accuracy: 0.01)
    }

    func testExpSquaredGradient() throws {
        // f(x) = exp(x²)
        // f'(x) = 2x * exp(x²)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x * x)
        }

        let input: [Float] = [0.0, 0.5, 1.0, -0.5]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2x * exp(x²)
        // x=0: 0, x=0.5: 1 * exp(0.25) ≈ 1.284, x=1: 2 * exp(1) ≈ 5.436
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 1.0 * exp(0.25), accuracy: 0.1)
        XCTAssertEqual(gradient[2], 2.0 * Float(M_E), accuracy: 0.1)
        XCTAssertEqual(gradient[3], -1.0 * exp(0.25), accuracy: 0.1)
    }

    // MARK: - Logarithm Tests

    func testLogGradient() throws {
        // f(x) = log(x)
        // f'(x) = 1/x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(x)
        }

        let input: [Float] = [1.0, 2.0, 4.0, 0.5]
        let gradient = try gradFunc.gradient(input)

        // Expected: 1/x = [1, 0.5, 0.25, 2]
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.25, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 2.0, accuracy: 0.01)
    }

    func testLogSquaredGradient() throws {
        // f(x) = log(x²)
        // f'(x) = 2/x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(x * x)
        }

        let input: [Float] = [1.0, 2.0, 4.0, 0.5]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2/x
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 4.0, accuracy: 0.01)
    }

    // MARK: - Combined Exp/Log Tests

    func testExpLogComposed() throws {
        // f(x) = exp(x) * x
        // f'(x) = exp(x) * x + exp(x) = exp(x) * (x + 1)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x) * x
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: exp(x) * (x + 1)
        XCTAssertEqual(gradient[0], exp(0.0) * 1.0, accuracy: 0.01)  // 1
        XCTAssertEqual(gradient[1], exp(1.0) * 2.0, accuracy: 0.1)   // ≈5.44
        XCTAssertEqual(gradient[2], exp(-1.0) * 0.0, accuracy: 0.01) // 0
        XCTAssertEqual(gradient[3], exp(2.0) * 3.0, accuracy: 0.5)   // ≈22.17
    }

    func testLogChainRule() throws {
        // f(x) = log(x * x) = 2 * log(x)
        // f'(x) = 2/x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(x * x)
        }

        let input: [Float] = [1.0, 2.0, 4.0, 0.5]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2/x
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 4.0, accuracy: 0.01)
    }

    // MARK: - Negation Tests

    func testNegateGradient() throws {
        // f(x) = -x
        // f'(x) = -1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffNegate(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be -1 everywhere
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], -1.0, accuracy: 0.01)
        }
    }

    func testNegateSquared() throws {
        // f(x) = (-x)² = x²
        // f'(x) = 2x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let neg = diffNegate(x)
            return neg * neg
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2x
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 8.0, accuracy: 0.01)
    }

    // MARK: - Polynomial Tests

    func testCubicGradient() throws {
        // f(x) = x³
        // f'(x) = 3x²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 3x² = [3, 12, 27, 48]
        XCTAssertEqual(gradient[0], 3.0, accuracy: 0.1)
        XCTAssertEqual(gradient[1], 12.0, accuracy: 0.1)
        XCTAssertEqual(gradient[2], 27.0, accuracy: 0.1)
        XCTAssertEqual(gradient[3], 48.0, accuracy: 0.1)
    }

    // MARK: - Composed Functions

    func testExpLogIdentity() throws {
        // f(x) = log(exp(x)) = x
        // f'(x) = 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(diffExp(x))
        }

        let input: [Float] = [0.0, 1.0, 2.0, -1.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be 1 everywhere
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 1.0, accuracy: 0.01)
        }
    }

    func testExpTimesInput() throws {
        // f(x) = x * exp(x)
        // f'(x) = exp(x) + x * exp(x) = exp(x) * (1 + x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * diffExp(x)
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: exp(x) * (1 + x)
        XCTAssertEqual(gradient[0], exp(0.0) * 1.0, accuracy: 0.01)  // exp(0) * 1 = 1
        XCTAssertEqual(gradient[1], exp(1.0) * 2.0, accuracy: 0.1)   // exp(1) * 2 ≈ 5.44
        XCTAssertEqual(gradient[2], exp(-1.0) * 0.0, accuracy: 0.01) // exp(-1) * 0 = 0
        XCTAssertEqual(gradient[3], exp(2.0) * 3.0, accuracy: 0.5)   // exp(2) * 3 ≈ 22.17
    }

    func testLogTimesInput() throws {
        // f(x) = x * log(x)
        // f'(x) = log(x) + 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * diffLog(x)
        }

        let input: [Float] = [1.0, Float(M_E), 2.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: log(x) + 1
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)  // log(1) + 1 = 1
        XCTAssertEqual(gradient[1], 2.0, accuracy: 0.01)  // log(e) + 1 = 2
        XCTAssertEqual(gradient[2], log(2.0) + 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], log(4.0) + 1.0, accuracy: 0.01)
    }

    // MARK: - Forward Value Tests

    func testExpForwardValues() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x)
        }

        let input: [Float] = [0.0, 1.0, 2.0, -1.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], Float(M_E), accuracy: 0.01)
        XCTAssertEqual(output[2], Float(M_E * M_E), accuracy: 0.1)
        XCTAssertEqual(output[3], 1.0 / Float(M_E), accuracy: 0.01)
    }

    func testLogForwardValues() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(x)
        }

        let input: [Float] = [1.0, Float(M_E), Float(M_E * M_E), 0.5]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[3], log(0.5), accuracy: 0.01)
    }

    func testExpLogIdentityForward() throws {
        // f(x) = log(exp(x)) should equal x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffLog(diffExp(x))
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Output should equal input (identity)
        for i in 0..<4 {
            XCTAssertEqual(output[i], input[i], accuracy: 0.01)
        }
    }

    // MARK: - Numerical Gradient Verification

    func testExpNumericalGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1], dtype: .float32)
        ) { x in
            return diffExp(x)
        }

        let x: Float = 1.5
        let h: Float = 0.0001

        let gradient = try gradFunc.gradient([x])

        // Numerical gradient
        let fPlus = exp(x + h)
        let fMinus = exp(x - h)
        let numericalGrad = (fPlus - fMinus) / (2 * h)

        XCTAssertEqual(gradient[0], numericalGrad, accuracy: 0.01)
    }

    func testLogNumericalGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1], dtype: .float32)
        ) { x in
            return diffLog(x)
        }

        let x: Float = 2.0
        let h: Float = 0.0001

        let gradient = try gradFunc.gradient([x])

        // Numerical gradient
        let fPlus = log(x + h)
        let fMinus = log(x - h)
        let numericalGrad = (fPlus - fMinus) / (2 * h)

        XCTAssertEqual(gradient[0], numericalGrad, accuracy: 0.01)
    }

    // MARK: - Edge Cases

    func testExpWithZeros() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x)
        }

        let input: [Float] = [0.0, 0.0, 0.0, 0.0]
        let gradient = try gradFunc.gradient(input)

        // exp'(0) = exp(0) = 1
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 1.0, accuracy: 0.01)
        }
    }

    func testQuarticGradient() throws {
        // f(x) = x⁴
        // f'(x) = 4x³
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let squared = x * x
            return squared * squared
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 4x³ = [4, 32, 108, 256]
        XCTAssertEqual(gradient[0], 4.0, accuracy: 0.1)
        XCTAssertEqual(gradient[1], 32.0, accuracy: 0.1)
        XCTAssertEqual(gradient[2], 108.0, accuracy: 0.5)
        XCTAssertEqual(gradient[3], 256.0, accuracy: 1.0)
    }

    // MARK: - Composition Tests

    func testExpPlusPolynomial() throws {
        // f(x) = exp(x) + x²
        // f'(x) = exp(x) + 2x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffExp(x) + x * x
        }

        let input: [Float] = [0.0, 1.0, -1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: exp(x) + 2x
        XCTAssertEqual(gradient[0], exp(0.0) + 0.0, accuracy: 0.01)   // 1
        XCTAssertEqual(gradient[1], exp(1.0) + 2.0, accuracy: 0.1)    // ≈4.72
        XCTAssertEqual(gradient[2], exp(-1.0) - 2.0, accuracy: 0.1)   // ≈-1.63
        XCTAssertEqual(gradient[3], exp(2.0) + 4.0, accuracy: 0.5)    // ≈11.39
    }
}
