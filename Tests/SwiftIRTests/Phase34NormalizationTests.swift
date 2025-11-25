// Phase34NormalizationTests.swift
// Tests for Phase 34: Normalization Layers

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase34NormalizationTests: XCTestCase {

    // MARK: - BatchNorm2D Tests

    func testBatchNorm2DForward() throws {
        // Input: [1, 2, 2, 2] - 1 batch, 2x2 spatial, 2 channels
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 2, 2, 2], dtype: .float32)
        ) { x in
            let scale = withoutDerivative(at: createConstant(1.0, shape: [2], dtype: .float32))
            let bias = withoutDerivative(at: createConstant(0.0, shape: [2], dtype: .float32))
            return diffBatchNorm2D(x, scale: scale, bias: bias, epsilon: 1e-5)
        }

        // Values with some variance
        let input: [Float] = [
            1, 5, 2, 6,  // row 0: channel 0: [1,2], channel 1: [5,6]
            3, 7, 4, 8   // row 1: channel 0: [3,4], channel 1: [7,8]
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 8)
        )

        // Check normalized output
        XCTAssertEqual(output.count, 8)

        // Channel 0: [1,2,3,4] mean=2.5, after normalization should have mean ~0
        let channel0Indices = [0, 2, 4, 6]
        let channel0Mean = channel0Indices.map { output[$0] }.reduce(0, +) / 4.0
        XCTAssertEqual(channel0Mean, 0.0, accuracy: 0.1)
    }

    // MARK: - LayerNorm Tests

    func testLayerNormForward() throws {
        // Input: [2, 3] - 2 samples, 3 features each
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            let scale = withoutDerivative(at: createConstant(1.0, shape: [3], dtype: .float32))
            let bias = withoutDerivative(at: createConstant(0.0, shape: [3], dtype: .float32))
            return diffLayerNorm(x, scale: scale, bias: bias, epsilon: 1e-5)
        }

        let input: [Float] = [
            1, 2, 3,  // Sample 0
            4, 5, 6   // Sample 1
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 6)
        )

        XCTAssertEqual(output.count, 6)

        // Check that each sample is normalized independently
        // Sample 0: mean of [1,2,3] should normalize to mean ~0
        let sample0 = Array(output[0..<3])
        let mean0 = sample0.reduce(0, +) / Float(sample0.count)
        XCTAssertEqual(mean0, 0.0, accuracy: 0.1)

        // Sample 1: mean of [4,5,6] should normalize to mean ~0
        let sample1 = Array(output[3..<6])
        let mean1 = sample1.reduce(0, +) / Float(sample1.count)
        XCTAssertEqual(mean1, 0.0, accuracy: 0.1)
    }

    // MARK: - GroupNorm Tests

    func testGroupNormForward() throws {
        // Input: [1, 2, 2, 4] - 1 batch, 2x2 spatial, 4 channels
        // 2 groups of 2 channels each
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 2, 2, 4], dtype: .float32)
        ) { x in
            let scale = withoutDerivative(at: createConstant(1.0, shape: [4], dtype: .float32))
            let bias = withoutDerivative(at: createConstant(0.0, shape: [4], dtype: .float32))
            return diffGroupNorm(x, scale: scale, bias: bias, numGroups: 2, epsilon: 1e-5)
        }

        let input: [Float] = [
            1, 2, 5, 6,  // Position (0,0): channels 0-3
            1, 2, 5, 6,  // Position (0,1)
            3, 4, 7, 8,  // Position (1,0)
            3, 4, 7, 8   // Position (1,1)
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 16)
        )

        XCTAssertEqual(output.count, 16)

        // Group 0 (channels 0-1): [1,2,1,2,3,4,3,4] should normalize
        let group0Values = [output[0], output[1], output[4], output[5], output[8], output[9], output[12], output[13]]
        let mean0 = group0Values.reduce(0, +) / Float(group0Values.count)
        XCTAssertEqual(mean0, 0.0, accuracy: 0.1)

        // Group 1 (channels 2-3): [5,6,5,6,7,8,7,8] should normalize
        let group1Values = [output[2], output[3], output[6], output[7], output[10], output[11], output[14], output[15]]
        let mean1 = group1Values.reduce(0, +) / Float(group1Values.count)
        XCTAssertEqual(mean1, 0.0, accuracy: 0.1)
    }
}
