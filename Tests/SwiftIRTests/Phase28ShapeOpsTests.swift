// Phase28ShapeOpsTests.swift
// Tests for Phase 28: Slice and concatenate operations

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase28ShapeOpsTests: XCTestCase {

    // MARK: - Slice Tests

    func testSlice1DForward() throws {
        // Slice [1:3] from [1, 2, 3, 4]
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSlice(x, starts: [1], limits: [3], strides: nil)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1])

        // Should get [2, 3]
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 3.0, accuracy: 0.01)
    }

    func testSlice1DGradient() throws {
        // Gradient should pad back to original shape
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSlice(x, starts: [1], limits: [3], strides: nil)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be [0, 1, 1, 0] (padded back)
        XCTAssertEqual(gradient.count, 4)
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
    }

    func testSlice2DForward() throws {
        // Slice [0:2, 1:3] from [2, 4] tensor
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 4], dtype: .float32)
        ) { x in
            return diffSlice(x, starts: [0, 1], limits: [2, 3], strides: nil)
        }

        // Input: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Should get [[2, 3], [6, 7]]
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 7.0, accuracy: 0.01)
    }

    func testSlice2DGradient() throws {
        // Gradient should pad back to original shape
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 4], dtype: .float32)
        ) { x in
            return diffSlice(x, starts: [0, 1], limits: [2, 3], strides: nil)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be [[0, 1, 1, 0], [0, 1, 1, 0]]
        XCTAssertEqual(gradient.count, 8)
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[4], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[5], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[6], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[7], 0.0, accuracy: 0.01)
    }

    func testSliceFirstRow() throws {
        // Slice first row [0:1, :] from [2, 4] tensor
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 4], dtype: .float32)
        ) { x in
            return diffSlice(x, starts: [0, 0], limits: [1, 4], strides: nil)
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Should get [1, 2, 3, 4]
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)

        // Gradient should be [[1, 1, 1, 1], [0, 0, 0, 0]]
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[4], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[7], 0.0, accuracy: 0.01)
    }

    func testSliceWithMath() throws {
        // f(x) = sum(slice(x))
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let sliced = diffSlice(x, starts: [1], limits: [3], strides: nil)
            return sliced * sliced  // Square the sliced elements
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1])

        // output = [4, 9] (squares of [2, 3])
        XCTAssertEqual(output[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 9.0, accuracy: 0.01)

        // gradient: d/dx(x^2) = 2x, padded back
        // = [0, 4, 6, 0]
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
    }
}
