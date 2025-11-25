// Phase33PoolingTests.swift
// Tests for Phase 33: MaxPool2D and AvgPool2D operations

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase33PoolingTests: XCTestCase {

    // MARK: - MaxPool2D Tests

    func testMaxPool2DForward() throws {
        // 4x4 input with 2x2 window, stride 2 -> 2x2 output
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 4, 4, 1], dtype: .float32)
        ) { x in
            return diffMaxPool2D(x, windowSize: [2, 2], strides: [2, 2], padding: "VALID")
        }

        // Input with distinct values
        let input: [Float] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ]
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 4)
        )

        // MaxPool with 2x2 windows:
        // [1,2,5,6] -> 6
        // [3,4,7,8] -> 8
        // [9,10,13,14] -> 14
        // [11,12,15,16] -> 16
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 14.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 16.0, accuracy: 0.01)
    }

    func testMaxPool2DOutputShape() throws {
        // Test output shape calculation
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 6, 6, 1], dtype: .float32)
        ) { x in
            return diffMaxPool2D(x, windowSize: [2, 2], strides: [2, 2], padding: "VALID")
        }

        let input = Array(repeating: Float(1.0), count: 36)
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 9)
        )

        // 6x6 with 2x2 window, stride 2 -> 3x3 = 9 elements
        XCTAssertEqual(output.count, 9)
    }

    // MARK: - AvgPool2D Tests

    func testAvgPool2DForward() throws {
        // 4x4 input with 2x2 window, stride 2 -> 2x2 output
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 4, 4, 1], dtype: .float32)
        ) { x in
            return diffAvgPool2D(x, windowSize: [2, 2], strides: [2, 2], padding: "VALID")
        }

        // Input with all ones
        let input = Array(repeating: Float(1.0), count: 16)
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 4)
        )

        // Average of 4 ones = 1.0
        XCTAssertEqual(output.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(output[i], 1.0, accuracy: 0.01)
        }
    }

    func testAvgPool2DValues() throws {
        // Test with distinct values
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 4, 4, 1], dtype: .float32)
        ) { x in
            return diffAvgPool2D(x, windowSize: [2, 2], strides: [2, 2], padding: "VALID")
        }

        let input: [Float] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ]
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 4)
        )

        // AvgPool with 2x2 windows:
        // [1,2,5,6] -> (1+2+5+6)/4 = 3.5
        // [3,4,7,8] -> (3+4+7+8)/4 = 5.5
        // [9,10,13,14] -> (9+10+13+14)/4 = 11.5
        // [11,12,15,16] -> (11+12+15+16)/4 = 13.5
        XCTAssertEqual(output.count, 4)
        XCTAssertEqual(output[0], 3.5, accuracy: 0.01)
        XCTAssertEqual(output[1], 5.5, accuracy: 0.01)
        XCTAssertEqual(output[2], 11.5, accuracy: 0.01)
        XCTAssertEqual(output[3], 13.5, accuracy: 0.01)
    }

    func testAvgPool2DOutputShape() throws {
        // Test output shape calculation
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 6, 6, 1], dtype: .float32)
        ) { x in
            return diffAvgPool2D(x, windowSize: [2, 2], strides: [2, 2], padding: "VALID")
        }

        let input = Array(repeating: Float(1.0), count: 36)
        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 9)
        )

        // 6x6 with 2x2 window, stride 2 -> 3x3 = 9 elements
        XCTAssertEqual(output.count, 9)
    }
}
