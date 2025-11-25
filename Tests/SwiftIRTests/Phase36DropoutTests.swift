// Phase36DropoutTests.swift
// Tests for Phase 36: Dropout

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase36DropoutTests: XCTestCase {

    // MARK: - Dropout Inference Tests

    func testDropoutInferenceMode() throws {
        // During inference, dropout should scale by (1-p)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.5, training: false)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1, 1]
        )

        // In inference mode with p=0.5, output should be scaled by 0.5
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(output[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.5, accuracy: 0.01)
        XCTAssertEqual(output[3], 2.0, accuracy: 0.01)
    }

    func testDropoutInferenceZeroProbability() throws {
        // With p=0, dropout should be identity
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.0, training: false)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1, 1]
        )

        // With p=0, output should equal input
        XCTAssertEqual(output, input)
    }

    // MARK: - Dropout Training Tests

    func testDropoutTrainingMode() throws {
        // During training (without RNG), dropout scales by 1/(1-p)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.5, training: true)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1, 1]
        )

        // In training mode with p=0.5, output should be scaled by 1/0.5 = 2.0
        // (This is simplified without actual random masking)
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 8.0, accuracy: 0.01)
    }

    func testDropoutTrainingZeroProbability() throws {
        // With p=0 in training, should be identity
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.0, training: true)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1, 1]
        )

        // With p=0, output should equal input even in training
        XCTAssertEqual(output, input)
    }

    // MARK: - Dropout on Multi-dimensional Tensors

    func testDropoutOn2DTensor() throws {
        // Test dropout on 2D tensor
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.3, training: false)
        }

        let input: [Float] = [1, 2, 3, 4, 5, 6]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 6)
        )

        XCTAssertEqual(output.count, 6)

        // With p=0.3 in inference, scale by 0.7
        XCTAssertEqual(output[0], 0.7, accuracy: 0.01)
        XCTAssertEqual(output[5], 4.2, accuracy: 0.01)
    }

    func testDropoutHighProbability() throws {
        // Test with high dropout probability
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3], dtype: .float32)
        ) { x in
            return diffDropout(x, probability: 0.9, training: false)
        }

        let input: [Float] = [10.0, 20.0, 30.0]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: [1, 1, 1]
        )

        // With p=0.9 in inference, scale by 0.1
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 3.0, accuracy: 0.01)
    }
}
