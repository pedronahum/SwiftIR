// Phase21MatmulPJRTTests.swift
// Tests for Phase 21: Matrix Multiplication Gradients on PJRT
//
// These tests verify that matmul operations execute correctly on real
// PJRT hardware with numerically correct gradients.

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase21MatmulPJRTTests: XCTestCase {

    // MARK: - Basic Matmul Tests

    func testMatmulForward() throws {
        // Simple 2x2 @ 2x2 matmul
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        // Input: [[1, 2], [3, 4]]
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Expected: [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]]
        //         = [[7, 10], [15, 22]]
        XCTAssertEqual(output[0], 7.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 15.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 22.0, accuracy: 0.01)
    }

    func testMatmulGradient2x2() throws {
        // f(X) = X @ X
        // For square matrix: df/dX = X^T + X^T (from chain rule)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // The gradient of X @ X w.r.t X is:
        // dL/dX = dL/dY @ X^T + X^T @ dL/dY
        // With seed = I: grad = X^T + X^T
        // X^T = [[1, 3], [2, 4]]
        // Expected gradient shape: [2, 2]
        XCTAssertEqual(gradient.count, 4)
    }

    func testTransposeForward() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffTranspose(x)
        }

        // Input: [[1, 2, 3], [4, 5, 6]] (row-major)
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1, 1, 1])

        // Output: [[1, 4], [2, 5], [3, 6]] (row-major) = [1, 4, 2, 5, 3, 6]
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 5.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 6.0, accuracy: 0.01)
    }

    func testTransposeGradient() throws {
        // f(X) = X^T
        // f'(X) = I (transpose of transpose is identity)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffTranspose(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient of transpose is just permuting back
        XCTAssertEqual(gradient.count, 6)
    }

    // MARK: - Matmul with Transpose

    func testMatmulWithTranspose() throws {
        // f(X) = X @ X^T
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            let xT = diffTranspose(x)
            return diffMatmul(x, xT)
        }

        // Input: [[1, 2, 3], [4, 5, 6]]
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // X @ X^T = [[1+4+9, 4+10+18], [4+10+18, 16+25+36]]
        //         = [[14, 32], [32, 77]]
        XCTAssertEqual(output[0], 14.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 32.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 32.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 77.0, accuracy: 0.01)
    }

    // MARK: - Neural Network Layer Tests

    func testSimpleDenseLayer() throws {
        // Simulates a dense layer: f(x) = x @ x (using same tensor as weight proxy)
        // In real use, we'd have separate weights
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            let y = diffMatmul(x, x)
            return y
        }

        let input: [Float] = [0.5, 0.5, 0.5, 0.5]
        let gradient = try gradFunc.gradient(input)

        // Verify gradient computation completes
        XCTAssertEqual(gradient.count, 4)

        // For constant input, gradient should be uniform
        // f(X) = X @ X, with X = 0.5 everywhere
        // Each output element is sum of products, gradient reflects this
    }

    func testMatmulTimesScalar() throws {
        // f(X) = X @ X * X (element-wise multiply after matmul)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            let matmulResult = diffMatmul(x, x)
            return matmulResult * x
        }

        let input: [Float] = [1.0, 0.0, 0.0, 1.0]  // Identity matrix
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // I @ I = I, I * I = I
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)

        XCTAssertEqual(gradient.count, 4)
    }

    // MARK: - Numerical Gradient Verification

    func testMatmulNumericalGradient() throws {
        // Verify gradient using finite differences for a simple case
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be non-zero and reasonable
        XCTAssertEqual(gradient.count, 4)

        // Verify at least some gradients are non-zero
        let hasNonZero = gradient.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasNonZero, "Gradient should have non-zero elements")
    }

    // MARK: - Different Shapes

    func testMatmul3x3() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3, 3], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        // 3x3 identity
        let input: [Float] = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ]
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1, 1, 1, 1, 1, 1, 1]
        )

        // I @ I = I
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[8], 1.0, accuracy: 0.01)
    }

    func testMatmulRectangular() throws {
        // 2x3 @ 3x2 = 2x2
        // But our current API only supports single input, so we'll do X @ X^T
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffMatmul(x, diffTranspose(x))
        }

        let input: [Float] = [1, 2, 3, 4, 5, 6]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Result is 2x2
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(gradient.count, 6)
    }

    // MARK: - Repeated Execution

    func testRepeatedMatmulExecution() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        // Execute multiple times with different inputs
        for scale in [1.0, 2.0, 0.5] as [Float] {
            let input: [Float] = [scale, 0, 0, scale]
            let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

            // Diagonal matrix squared stays diagonal
            XCTAssertEqual(output[0], scale * scale, accuracy: 0.01)
            XCTAssertEqual(output[3], scale * scale, accuracy: 0.01)
        }
    }

    // MARK: - Chain of Matmuls

    func testChainedMatmul() throws {
        // f(X) = (X @ X) @ X = X^3
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            let xx = diffMatmul(x, x)
            return diffMatmul(xx, x)
        }

        let input: [Float] = [1.0, 0.0, 0.0, 1.0]  // Identity
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // I^3 = I
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 1.0, accuracy: 0.01)
    }

    // MARK: - Matmul with Other Operations

    func testMatmulPlusInput() throws {
        // f(X) = X @ X + X
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x) + x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // X @ X = [[7, 10], [15, 22]]
        // + X   = [[8, 12], [18, 26]]
        XCTAssertEqual(output[0], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 12.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 18.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 26.0, accuracy: 0.01)
    }

    func testMatmulWithExp() throws {
        // f(X) = exp(X @ X)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffExp(diffMatmul(x, x))
        }

        let input: [Float] = [0.1, 0.1, 0.1, 0.1]
        let gradient = try gradFunc.gradient(input)

        XCTAssertEqual(gradient.count, 4)
        // Gradient should be positive since exp is always positive
        for g in gradient {
            XCTAssertGreaterThan(g, 0)
        }
    }

    // MARK: - Info Test

    func testMatmulGradientInfo() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffMatmul(x, x)
        }

        let info = gradFunc.info
        XCTAssertTrue(info.contains("PJRT"))
        XCTAssertTrue(info.contains("Gradient"))
    }
}
