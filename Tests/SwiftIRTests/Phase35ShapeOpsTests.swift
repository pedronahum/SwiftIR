// Phase35ShapeOpsTests.swift
// Tests for Phase 35: Shape Operations (Permute, Flatten, Reshape)

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase35ShapeOpsTests: XCTestCase {

    // MARK: - Permute Tests

    func testPermuteNHWCtoNCHW() throws {
        // Test converting NHWC (TensorFlow format) to NCHW (PyTorch format)
        // Input: [1, 2, 3, 4] -> NHWC (1 batch, 2 height, 3 width, 4 channels)
        // Output: [1, 4, 2, 3] -> NCHW (1 batch, 4 channels, 2 height, 3 width)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 2, 3, 4], dtype: .float32)
        ) { x in
            return diffPermute(x, dims: [0, 3, 1, 2])  // NHWC -> NCHW
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)

        // Verify shape transformation
        // Input[0,0,0,0] should map to Output[0,0,0,0]
        // Input[0,0,0,1] should map to Output[0,1,0,0]
        // Input layout: [batch=1][height=2][width=3][channel=4]
        // Output layout: [batch=1][channel=4][height=2][width=3]

        // First element of first channel should be input[0,0,0,0]
        XCTAssertEqual(output[0], input[0])
        // First element of second channel should be input[0,0,0,1]
        XCTAssertEqual(output[6], input[1])  // Stride through height*width=6
    }

    func testPermuteIdentity() throws {
        // Test identity permutation [0,1,2]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4], dtype: .float32)
        ) { x in
            return diffPermute(x, dims: [0, 1, 2])
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        // Identity permutation should return same values
        XCTAssertEqual(output, input)
    }

    func testPermuteMatrixTranspose() throws {
        // Test simple matrix transpose [1, 0]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return diffPermute(x, dims: [1, 0])
        }

        let input: [Float] = [
            1, 2, 3,  // Row 0
            4, 5, 6   // Row 1
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 6)
        )

        let expected: [Float] = [
            1, 4,  // Column 0
            2, 5,  // Column 1
            3, 6   // Column 2
        ]

        XCTAssertEqual(output, expected)
    }

    // MARK: - Flatten Tests

    func testFlattenMiddleDims() throws {
        // Flatten middle dimensions: [2, 3, 4, 5] -> [2, 12, 5]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4, 5], dtype: .float32)
        ) { x in
            return diffFlatten(x, startDim: 1, endDim: 2)
        }

        let inputSize = 2 * 3 * 4 * 5
        let input: [Float] = Array(0..<inputSize).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: inputSize)
        )

        // Should flatten to 2 * 12 * 5 = 120 elements
        XCTAssertEqual(output.count, 120)

        // First element should be preserved
        XCTAssertEqual(output[0], input[0])
    }

    func testFlattenAll() throws {
        // Flatten entire tensor: [2, 3, 4] -> [24]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4], dtype: .float32)
        ) { x in
            return diffFlatten(x, startDim: 0, endDim: -1)
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)
        XCTAssertEqual(output, input)  // Flattening doesn't change data order
    }

    func testFlattenBatchPreserving() throws {
        // Flatten all except batch: [2, 3, 4] -> [2, 12]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4], dtype: .float32)
        ) { x in
            return diffFlatten(x, startDim: 1, endDim: -1)
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)

        // Batch 0 starts at index 0, batch 1 starts at index 12
        XCTAssertEqual(output[0], input[0])
        XCTAssertEqual(output[12], input[12])
    }

    // MARK: - Reshape Tests

    func testReshapeSimple() throws {
        // Reshape [2, 3, 4] -> [6, 4]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4], dtype: .float32)
        ) { x in
            return diffReshape(x, shape: [6, 4])
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)
        XCTAssertEqual(output, input)  // Data order preserved
    }

    func testReshapeTo1D() throws {
        // Reshape [2, 3, 4] -> [24]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3, 4], dtype: .float32)
        ) { x in
            return diffReshape(x, shape: [24])
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)
        XCTAssertEqual(output, input)
    }

    func testReshapeHigherDim() throws {
        // Reshape [24] -> [2, 3, 4]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [24], dtype: .float32)
        ) { x in
            return diffReshape(x, shape: [2, 3, 4])
        }

        let input: [Float] = Array(0..<24).map(Float.init)

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 24)
        )

        XCTAssertEqual(output.count, 24)
        XCTAssertEqual(output, input)
    }
}
