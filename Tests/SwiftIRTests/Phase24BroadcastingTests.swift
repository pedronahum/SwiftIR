// Phase24BroadcastingTests.swift
// Tests for Phase 24: Broadcasting support

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase24BroadcastingTests: XCTestCase {

    // MARK: - Basic Broadcasting Tests

    func testBroadcastScalarTo1D() throws {
        // Broadcast [1] to [4] in addition
        // f(x) = x + bias where x is [4], bias is [1] -> broadcast to [4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [1], dtype: .float32))
        ) { x, bias in
            return x + bias
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let bias: [Float] = [10.0]

        let (output, gradX, gradBias) = try gradFunc.forwardWithGradients(
            input, bias,
            seed: [1, 1, 1, 1]
        )

        // Forward: [11, 12, 13, 14]
        XCTAssertEqual(output[0], 11.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 12.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 13.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 14.0, accuracy: 0.01)

        // Gradients
        for i in 0..<4 {
            XCTAssertEqual(gradX[i], 1.0, accuracy: 0.01)
        }
    }

    func testBroadcast1DTo2D() throws {
        // Broadcast [4] to [2, 4] in addition
        // x is [2, 4], bias is [4] -> broadcast bias to [2, 4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, bias in
            return x + bias
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  // [2, 4]
        let bias: [Float] = [1.0, 2.0, 3.0, 4.0]  // [4]

        let (output, gradX, _) = try gradFunc.forwardWithGradients(
            input, bias,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: add bias to each row
        // Row 0: [1+1, 2+2, 3+3, 4+4] = [2, 4, 6, 8]
        // Row 1: [5+1, 6+2, 7+3, 8+4] = [6, 8, 10, 12]
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[6], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[7], 12.0, accuracy: 0.01)

        // gradX should be all 1s
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 1.0, accuracy: 0.01)
        }
    }

    func testBroadcastMultiply() throws {
        // f(x, scale) = x * scale
        // x is [2, 4], scale is [4] -> broadcast scale to [2, 4]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, scale in
            return x * scale
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let scale: [Float] = [2.0, 2.0, 2.0, 2.0]

        let (output, gradX, _) = try gradFunc.forwardWithGradients(
            input, scale,
            seed: [1, 1, 1, 1, 1, 1, 1, 1]
        )

        // Forward: scale each element by 2
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[4], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 12.0, accuracy: 0.01)
        XCTAssertEqual(output[6], 14.0, accuracy: 0.01)
        XCTAssertEqual(output[7], 16.0, accuracy: 0.01)

        // gradX = scale = [2, 2, 2, 2, 2, 2, 2, 2]
        for i in 0..<8 {
            XCTAssertEqual(gradX[i], 2.0, accuracy: 0.01)
        }
    }

    // MARK: - Same Shape Tests (No Broadcasting)

    func testSameShapeNobroadcast() throws {
        // When shapes match, no broadcasting should occur
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x + y
        }

        let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input2: [Float] = [5.0, 6.0, 7.0, 8.0]

        let (output, grad1, grad2) = try gradFunc.forwardWithGradients(
            input1, input2,
            seed: [1, 1, 1, 1]
        )

        // Forward
        XCTAssertEqual(output[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 12.0, accuracy: 0.01)

        // Gradients
        for i in 0..<4 {
            XCTAssertEqual(grad1[i], 1.0, accuracy: 0.01)
            XCTAssertEqual(grad2[i], 1.0, accuracy: 0.01)
        }
    }

    // MARK: - Existing Tests Should Still Pass

    func testSingleInputNobroadcast() throws {
        // Make sure single input functions still work
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // f'(x) = 2x
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 8.0, accuracy: 0.01)
    }

    func testConstantWithSingleInput() throws {
        // f(x) = x + 2 (constant should work with broadcasting)
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

        // Forward
        XCTAssertEqual(output[0], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 5.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 6.0, accuracy: 0.01)

        // Gradient
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 1.0, accuracy: 0.01)
        }
    }
}
