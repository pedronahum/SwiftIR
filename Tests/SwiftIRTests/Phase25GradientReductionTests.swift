// Phase25GradientReductionTests.swift
// Tests for Phase 25: Gradient reduction for broadcasting

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase25GradientReductionTests: XCTestCase {

    // MARK: - Addition Broadcasting Gradient Tests

    func testAddBroadcast1DTo2DGradientReduction() throws {
        // f(x, bias) = x + bias
        // x is [2, 4], bias is [4] -> broadcast bias to [2, 4]
        // Gradient for bias should be summed over dimension 0
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, bias in
            return x + bias
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  // [2, 4]
        let bias: [Float] = [1.0, 2.0, 3.0, 4.0]  // [4]

        let (output, gradX, gradBias) = try gradFunc.forwardWithGradients(
            input, bias,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: add bias to each row
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[6], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[7], 12.0, accuracy: 0.01)

        // gradX should be all 1s (same shape as x)
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 1.0, accuracy: 0.01)
        }

        // gradBias should be reduced from [2, 4] to [4]
        // Sum over first dimension: [1+1, 1+1, 1+1, 1+1] = [2, 2, 2, 2]
        for i in 0..<4 {
            XCTAssertEqual(gradBias[i], 2.0, accuracy: 0.01, "gradBias[\(i)] should be 2.0")
        }
    }

    func testAddBroadcastScalarTo1DGradientReduction() throws {
        // f(x, scalar) = x + scalar
        // x is [4], scalar is [1] -> broadcast scalar to [4]
        // Gradient for scalar should be summed over all elements
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [1], dtype: .float32))
        ) { x, scalar in
            return x + scalar
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let scalar: [Float] = [10.0]

        let (output, gradX, gradScalar) = try gradFunc.forwardWithGradients(
            input, scalar,
            seed: [1, 1, 1, 1]
        )

        // Forward: [11, 12, 13, 14]
        XCTAssertEqual(output[0], 11.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 12.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 13.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 14.0, accuracy: 0.01)

        // gradX should be all 1s
        for i in 0..<4 {
            XCTAssertEqual(gradX[i], 1.0, accuracy: 0.01)
        }

        // gradScalar should be sum of all gradients = 4
        XCTAssertEqual(gradScalar[0], 4.0, accuracy: 0.01, "gradScalar should be sum of all seed gradients")
    }

    // MARK: - Multiplication Broadcasting Gradient Tests

    func testMultiplyBroadcast1DTo2DGradientReduction() throws {
        // f(x, scale) = x * scale
        // x is [2, 4], scale is [4] -> broadcast scale to [2, 4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, scale in
            return x * scale
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  // [2, 4]
        let scale: [Float] = [2.0, 2.0, 2.0, 2.0]  // [4]

        let (output, gradX, gradScale) = try gradFunc.forwardWithGradients(
            input, scale,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: scale each element by 2
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[7], 16.0, accuracy: 0.01)

        // gradX = scale (broadcast) = [2, 2, 2, 2, 2, 2, 2, 2]
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 2.0, accuracy: 0.01)
        }

        // gradScale = sum over first dimension of (dy * x)
        // = [1*1 + 1*5, 1*2 + 1*6, 1*3 + 1*7, 1*4 + 1*8]
        // = [6, 8, 10, 12]
        XCTAssertEqual(gradScale[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradScale[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(gradScale[2], 10.0, accuracy: 0.01)
        XCTAssertEqual(gradScale[3], 12.0, accuracy: 0.01)
    }

    func testMultiplyBroadcastWithNonUniformSeed() throws {
        // Test with non-uniform seed to verify reduction works correctly
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 3], dtype: .float32),
                    TensorSpec(shape: [3], dtype: .float32))
        ) { x, scale in
            return x * scale
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  // [2, 3]
        let scale: [Float] = [1.0, 2.0, 3.0]  // [3]

        let (_, _, gradScale) = try gradFunc.forwardWithGradients(
            input, scale,
            seed: [1, 2, 3, 4, 5, 6]  // Non-uniform seed
        )

        // gradScale = sum over first dimension of (dy * x)
        // = [1*1 + 4*4, 2*2 + 5*5, 3*3 + 6*6]
        // = [1 + 16, 4 + 25, 9 + 36]
        // = [17, 29, 45]
        XCTAssertEqual(gradScale[0], 17.0, accuracy: 0.01)
        XCTAssertEqual(gradScale[1], 29.0, accuracy: 0.01)
        XCTAssertEqual(gradScale[2], 45.0, accuracy: 0.01)
    }

    // MARK: - Subtraction Broadcasting Gradient Tests

    func testSubtractBroadcastGradientReduction() throws {
        // f(x, bias) = x - bias
        // x is [2, 4], bias is [4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, bias in
            return x - bias
        }

        let input: [Float] = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]  // [2, 4]
        let bias: [Float] = [1.0, 2.0, 3.0, 4.0]  // [4]

        let (output, gradX, gradBias) = try gradFunc.forwardWithGradients(
            input, bias,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: subtract bias from each row
        XCTAssertEqual(output[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)

        // gradX should be all 1s
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 1.0, accuracy: 0.01)
        }

        // gradBias should be -2 (negative of sum)
        for i in 0..<4 {
            XCTAssertEqual(gradBias[i], -2.0, accuracy: 0.01, "gradBias[\(i)] should be -2.0")
        }
    }

    // MARK: - Division Broadcasting Gradient Tests

    func testDivideBroadcastGradientReduction() throws {
        // f(x, divisor) = x / divisor
        // x is [2, 4], divisor is [4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, divisor in
            return x / divisor
        }

        let input: [Float] = [2.0, 4.0, 6.0, 8.0, 4.0, 8.0, 12.0, 16.0]  // [2, 4]
        let divisor: [Float] = [2.0, 2.0, 2.0, 2.0]  // [4]

        let (output, gradX, gradDivisor) = try gradFunc.forwardWithGradients(
            input, divisor,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: divide each element by 2
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[7], 8.0, accuracy: 0.01)

        // gradX = 1/divisor (broadcast) = [0.5, 0.5, ...]
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 0.5, accuracy: 0.01)
        }

        // gradDivisor = sum over dim 0 of (-dy * x / divisor^2)
        // = sum([−1*2/4, −1*4/4, −1*6/4, −1*8/4], [−1*4/4, −1*8/4, −1*12/4, −1*16/4])
        // = [-0.5-1.0, -1.0-2.0, -1.5-3.0, -2.0-4.0]
        // = [-1.5, -3.0, -4.5, -6.0]
        XCTAssertEqual(gradDivisor[0], -1.5, accuracy: 0.01)
        XCTAssertEqual(gradDivisor[1], -3.0, accuracy: 0.01)
        XCTAssertEqual(gradDivisor[2], -4.5, accuracy: 0.01)
        XCTAssertEqual(gradDivisor[3], -6.0, accuracy: 0.01)
    }

    // MARK: - Complex Expression Tests

    func testNeuralNetworkLayerWithBroadcasting() throws {
        // Simulate: output = x * weights + bias
        // x is [2, 4], weights is [4], bias is [4]
        // Both weights and bias are broadcast to [2, 4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, weights in
            return x * weights
        }

        let activations: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  // [2, 4]
        let weights: [Float] = [0.5, 0.5, 0.5, 0.5]  // [4]

        let (output, gradX, gradWeights) = try gradFunc.forwardWithGradients(
            activations, weights,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: scale by 0.5
        XCTAssertEqual(output[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(output[7], 4.0, accuracy: 0.01)

        // gradX = weights (broadcast)
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 0.5, accuracy: 0.01)
        }

        // gradWeights = sum over batch dimension of (dy * x)
        // = [1*1 + 1*5, 1*2 + 1*6, 1*3 + 1*7, 1*4 + 1*8]
        // = [6, 8, 10, 12]
        XCTAssertEqual(gradWeights[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradWeights[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(gradWeights[2], 10.0, accuracy: 0.01)
        XCTAssertEqual(gradWeights[3], 12.0, accuracy: 0.01)
    }

    // MARK: - Same Shape Tests (No Reduction)

    func testSameShapeNoReduction() throws {
        // When shapes match, no reduction should occur
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x + y
        }

        let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input2: [Float] = [5.0, 6.0, 7.0, 8.0]

        let (_, grad1, grad2) = try gradFunc.forwardWithGradients(
            input1, input2,
            seed: [1, 1, 1, 1]
        )

        // Gradients should be same shape as inputs
        for i in 0..<4 {
            XCTAssertEqual(grad1[i], 1.0, accuracy: 0.01)
            XCTAssertEqual(grad2[i], 1.0, accuracy: 0.01)
        }
    }

    // MARK: - 3D Broadcasting Tests

    func testBroadcast1DTo3D() throws {
        // Test broadcasting [4] to [2, 3, 4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 3, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, bias in
            return x + bias
        }

        // Create input of size 2*3*4 = 24
        let input: [Float] = Array(repeating: 1.0, count: 24)
        let bias: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (_, _, gradBias) = try gradFunc.forwardWithGradients(
            input, bias,
            seed: Array(repeating: 1.0, count: 24)
        )

        // gradBias should be reduced from [2, 3, 4] to [4]
        // Sum over dimensions 0 and 1: each element summed 2*3 = 6 times
        for i in 0..<4 {
            XCTAssertEqual(gradBias[i], 6.0, accuracy: 0.01, "gradBias[\(i)] should be 6.0")
        }
    }
}
