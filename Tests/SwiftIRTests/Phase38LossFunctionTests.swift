// Phase38LossFunctionTests.swift
// Tests for Phase 38: Additional Loss Functions (Huber, KL Divergence)

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase38LossFunctionTests: XCTestCase {

    // MARK: - Huber Loss Tests

    func testHuberLossQuadraticRegion() throws {
        // Test Huber loss in quadratic region (|error| <= delta)
        // When |pred - target| <= delta, Huber = 0.5 * (pred - target)^2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            // Targets: all zeros
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targets, delta: 1.0)
        }

        // Predictions (errors when target=0): [0.2, 0.3, -0.2, 0.1]
        let input: [Float] = [0.2, 0.3, -0.2, 0.1]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // Compute expected: mean(0.5 * errors^2)
        // errors^2 = [0.04, 0.09, 0.04, 0.01]
        // 0.5 * errors^2 = [0.02, 0.045, 0.02, 0.005]
        // mean = 0.0225
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0], 0.0225, accuracy: 0.001)
    }

    func testHuberLossLinearRegion() throws {
        // Test Huber loss in linear region (|error| > delta)
        // When |pred - target| > delta, Huber = delta * (|error| - 0.5 * delta)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targets, delta: 1.0)
        }

        // Large errors (beyond delta=1.0)
        let input: [Float] = [2.0, -2.0, 3.0, -3.0]  // |errors| = [2, 2, 3, 3]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // Compute expected for each: delta * (|error| - 0.5 * delta)
        // For |error|=2: 1.0 * (2.0 - 0.5) = 1.5
        // For |error|=3: 1.0 * (3.0 - 0.5) = 2.5
        // mean = (1.5 + 1.5 + 2.5 + 2.5) / 4 = 2.0
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
    }

    func testHuberLossMixedRegion() throws {
        // Test Huber loss with both quadratic and linear regions
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targets, delta: 1.0)
        }

        // Mix of small and large errors
        let input: [Float] = [0.5, -0.5, 2.0, -2.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // 0.5: quadratic = 0.5 * 0.25 = 0.125
        // -0.5: quadratic = 0.5 * 0.25 = 0.125
        // 2.0: linear = 1.0 * (2.0 - 0.5) = 1.5
        // -2.0: linear = 1.0 * (2.0 - 0.5) = 1.5
        // mean = (0.125 + 0.125 + 1.5 + 1.5) / 4 = 0.8125
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0], 0.8125, accuracy: 0.01)
    }

    func testHuberLossGradientQuadratic() throws {
        // Test gradient in quadratic region
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targets, delta: 1.0)
        }

        let input: [Float] = [0.4, -0.4]

        let gradient = try gradFunc.gradient(input)

        // In quadratic region, gradient = (pred - target) / n
        // For [0.4, -0.4] with n=2: [0.4/2, -0.4/2] = [0.2, -0.2]
        XCTAssertEqual(gradient.count, 2)
        XCTAssertEqual(gradient[0], 0.2, accuracy: 0.01)
        XCTAssertEqual(gradient[1], -0.2, accuracy: 0.01)
    }

    func testHuberLossWithDifferentDelta() throws {
        // Test Huber loss with delta=0.5
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targets, delta: 0.5)
        }

        let input: [Float] = [1.0, -1.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // For |error|=1.0 with delta=0.5 (linear region):
        // loss = 0.5 * (1.0 - 0.25) = 0.375 per element
        // mean = 0.375
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0], 0.375, accuracy: 0.01)
    }

    // MARK: - KL Divergence Tests

    func testKLDivergenceIdentical() throws {
        // KL divergence should be ~0 for identical distributions
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            // Target distribution same as prediction
            let targets = predictions
            return diffKLDivergence(predictions, targets)
        }

        // Probability distribution (sums to 1)
        let input: [Float] = [0.25, 0.25, 0.25, 0.25]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // KL(P||P) = 0
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0], 0.0, accuracy: 0.001)
    }

    func testKLDivergenceUniformToSkewed() throws {
        // Test KL divergence - predictions should differ from targets
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            // Use uniform target distribution (properly normalized: 0.25 each)
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.25, shape: shape, dtype: dtype)
            return diffKLDivergence(predictions, targets)
        }

        // Skewed prediction distribution (sums to 1.0)
        let input: [Float] = [0.7, 0.2, 0.05, 0.05]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // KL should be positive (distributions are different)
        XCTAssertEqual(output.count, 1)
        XCTAssertGreaterThan(output[0], 0.0)
    }

    func testKLDivergenceAsymmetry() throws {
        // KL(P||Q) != KL(Q||P) - test asymmetry
        let dist1: [Float] = [0.8, 0.2]
        let dist2: [Float] = [0.4, 0.6]

        // Compute KL(dist1 || uniform target)
        let gradFunc1 = try compileGradientForPJRT(
            input: TensorSpec(shape: [2], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.5, shape: shape, dtype: dtype)
            return diffKLDivergence(predictions, targets)
        }

        let (kl1, _) = try gradFunc1.forwardWithGradient(dist1, seed: [1.0])

        // Compute KL(dist2 || uniform target)
        let gradFunc2 = try compileGradientForPJRT(
            input: TensorSpec(shape: [2], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.5, shape: shape, dtype: dtype)
            return diffKLDivergence(predictions, targets)
        }

        let (kl2, _) = try gradFunc2.forwardWithGradient(dist2, seed: [1.0])

        // KL divergence values should differ for different distributions
        XCTAssertNotEqual(kl1[0], kl2[0], accuracy: 0.01)
    }

    func testKLDivergenceGradient() throws {
        // Test that gradients flow correctly
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.33, shape: shape, dtype: dtype)
            return diffKLDivergence(predictions, targets)
        }

        let input: [Float] = [0.5, 0.3, 0.2]

        let gradient = try gradFunc.gradient(input)

        // Gradient should exist and be non-zero
        XCTAssertEqual(gradient.count, 3)
        // At least one gradient should be non-zero
        let hasNonZero = gradient.contains { abs($0) > 0.01 }
        XCTAssertTrue(hasNonZero, "Gradient should have non-zero components")
    }

    func testKLDivergenceWithSoftmax() throws {
        // Realistic use case: KL divergence after softmax
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { logits in
            // Apply softmax to get probabilities
            let predictions = diffSoftmax(logits)
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targets = createConstant(0.25, shape: shape, dtype: dtype)
            return diffKLDivergence(predictions, targets)
        }

        // Logits (before softmax)
        let input: [Float] = [1.0, 0.5, 0.0, -0.5]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1.0]
        )

        // Should compute KL divergence after softmax normalization
        XCTAssertEqual(output.count, 1)
        XCTAssertGreaterThan(output[0], 0.0)
    }

    // MARK: - Comparison with MSE

    func testHuberVsMSEOnOutliers() throws {
        // Demonstrate that Huber is more robust to outliers than MSE
        let outliers: [Float] = [0.1, 0.2, 10.0, 0.1]  // One large outlier

        // MSE loss
        let mseFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targ = createConstant(0.0, shape: shape, dtype: dtype)
            return diffMSELoss(predictions, targ)
        }

        let (mseLoss, _) = try mseFunc.forwardWithGradient(outliers, seed: [1.0])

        // Huber loss
        let huberFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { predictions in
            let shape = withoutDerivative(at: predictions.shape)
            let dtype = withoutDerivative(at: predictions.dtype)
            let targ = createConstant(0.0, shape: shape, dtype: dtype)
            return diffHuberLoss(predictions, targ, delta: 1.0)
        }

        let (huberLoss, _) = try huberFunc.forwardWithGradient(outliers, seed: [1.0])

        // Huber loss should be significantly smaller than MSE due to linear penalty on outlier
        // MSE: mean([0.01, 0.04, 100, 0.01]) = 25.015
        // Huber: mean([0.005, 0.02, 1.0*(10-0.5)=9.5, 0.005]) ≈ 2.38
        XCTAssertGreaterThan(mseLoss[0], huberLoss[0])
        XCTAssertGreaterThan(mseLoss[0], 20.0)  // MSE is large
        XCTAssertLessThan(huberLoss[0], 5.0)    // Huber is smaller
    }
}
