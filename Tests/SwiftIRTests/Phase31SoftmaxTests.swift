// Phase31SoftmaxTests.swift
// Tests for Phase 31: Softmax and Cross-Entropy Loss

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase31SoftmaxTests: XCTestCase {

    // MARK: - Softmax Forward Tests

    func testSoftmaxForward() throws {
        // Test softmax forward pass
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSoftmax(x)
        }

        // Equal logits should give uniform distribution
        let input: [Float] = [1.0, 1.0, 1.0, 1.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // softmax([1,1,1,1]) = [0.25, 0.25, 0.25, 0.25]
        XCTAssertEqual(output.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(output[i], 0.25, accuracy: 0.01)
        }
    }

    func testSoftmaxNumericalStability() throws {
        // Test with large values - should not overflow
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffSoftmax(x)
        }

        // Large values that would overflow without max subtraction
        let input: [Float] = [1000.0, 1000.0, 1000.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1])

        // Should still be uniform distribution
        for i in 0..<3 {
            XCTAssertEqual(output[i], 1.0/3.0, accuracy: 0.01)
        }
    }

    func testSoftmaxDistribution() throws {
        // Test that output sums to 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSoftmax(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Sum should be 1
        let sum = output.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01)

        // Larger input should have larger probability
        XCTAssertGreaterThan(output[3], output[2])
        XCTAssertGreaterThan(output[2], output[1])
        XCTAssertGreaterThan(output[1], output[0])
    }

    func testSoftmaxGradient() throws {
        // Test softmax gradient
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffSoftmax(x)
        }

        let input: [Float] = [1.0, 2.0, 3.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should exist and have correct shape
        XCTAssertEqual(gradient.count, 3)

        // With uniform seed, gradients should sum to 0
        // (softmax preserves sum, so uniform gradient contribution is 0)
        let gradSum = gradient.reduce(0, +)
        XCTAssertEqual(gradSum, 0.0, accuracy: 0.01)
    }

    // MARK: - Cross-Entropy Tests

    func testCrossEntropyForward() throws {
        // Test cross-entropy loss
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [3], dtype: .float32),
                    TensorSpec(shape: [3], dtype: .float32))
        ) { pred, target in
            return diffCrossEntropy(pred, target)
        }

        // Perfect prediction: pred = target
        let pred: [Float] = [0.7, 0.2, 0.1]
        let target: [Float] = [1.0, 0.0, 0.0]  // One-hot for class 0

        let (output, _, _) = try gradFunc.forwardWithGradients(
            pred, target,
            seed: [1]  // Scalar output
        )

        // Loss = -1.0 * log(0.7) = 0.357
        XCTAssertEqual(output[0], 0.357, accuracy: 0.05)
    }

    func testCrossEntropyPerfectPrediction() throws {
        // Perfect prediction should have low loss
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [3], dtype: .float32),
                    TensorSpec(shape: [3], dtype: .float32))
        ) { pred, target in
            return diffCrossEntropy(pred, target)
        }

        // Nearly perfect prediction
        let pred: [Float] = [0.99, 0.005, 0.005]
        let target: [Float] = [1.0, 0.0, 0.0]

        let (output, _, _) = try gradFunc.forwardWithGradients(
            pred, target,
            seed: [1]
        )

        // Loss should be close to 0
        XCTAssertLessThan(output[0], 0.02)
    }

    func testSoftmaxCrossEntropyForward() throws {
        // Test combined softmax + cross-entropy
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [3], dtype: .float32),
                    TensorSpec(shape: [3], dtype: .float32))
        ) { logits, target in
            return diffSoftmaxCrossEntropy(logits, target)
        }

        // Logits that favor class 0
        let logits: [Float] = [2.0, 1.0, 0.0]
        let target: [Float] = [1.0, 0.0, 0.0]  // One-hot for class 0

        let (output, gradLogits, _) = try gradFunc.forwardWithGradients(
            logits, target,
            seed: [1]
        )

        // Loss should be positive
        XCTAssertGreaterThan(output[0], 0)

        // Gradient should exist
        XCTAssertEqual(gradLogits.count, 3)

        // Gradient = softmax(logits) - target
        // For class 0: grad < 0 (push up)
        // For class 1,2: grad > 0 (push down)
        print("Softmax CE gradient: \(gradLogits)")
    }

    func testSoftmaxCrossEntropyGradient() throws {
        // The gradient of softmax CE is: probs - targets
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [3], dtype: .float32),
                    TensorSpec(shape: [3], dtype: .float32))
        ) { logits, target in
            return diffSoftmaxCrossEntropy(logits, target)
        }

        // Logits that give uniform distribution
        let logits: [Float] = [0.0, 0.0, 0.0]
        let target: [Float] = [1.0, 0.0, 0.0]  // One-hot for class 0

        let (_, gradLogits, _) = try gradFunc.forwardWithGradients(
            logits, target,
            seed: [1]
        )

        // softmax([0,0,0]) = [1/3, 1/3, 1/3]
        // grad = [1/3 - 1, 1/3 - 0, 1/3 - 0] = [-2/3, 1/3, 1/3]
        XCTAssertEqual(gradLogits[0], -2.0/3.0, accuracy: 0.05)
        XCTAssertEqual(gradLogits[1], 1.0/3.0, accuracy: 0.05)
        XCTAssertEqual(gradLogits[2], 1.0/3.0, accuracy: 0.05)
    }
}
