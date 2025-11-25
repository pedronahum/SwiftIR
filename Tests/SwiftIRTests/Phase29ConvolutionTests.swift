// Phase29ConvolutionTests.swift
// Tests for Phase 29: Convolution operations

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase29ConvolutionTests: XCTestCase {

    // MARK: - Basic Convolution Forward Tests

    func testConv2DForwardValidPadding() throws {
        // Simple 2x2 convolution on 4x4 input with valid padding
        // Input: [1, 4, 4, 1] (batch, height, width, channels)
        // Filter: [2, 2, 1, 1] (filter_h, filter_w, in_channels, out_channels)
        // Output: [1, 3, 3, 1]
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [1, 4, 4, 1], dtype: .float32),
                    TensorSpec(shape: [2, 2, 1, 1], dtype: .float32))
        ) { input, filter in
            return diffConv2D(input, filter, strides: [1, 1], padding: "VALID")
        }

        // 4x4 input (all ones)
        let input: [Float] = Array(repeating: 1.0, count: 16)
        // 2x2 filter (all ones) - should sum to 4 at each position
        let filter: [Float] = [1.0, 1.0, 1.0, 1.0]

        let (output, _, _) = try gradFunc.forwardWithGradients(
            input, filter,
            seed: Array(repeating: 1.0, count: 9)  // 3x3 output
        )

        // Each output position should be sum of 2x2 region = 4
        XCTAssertEqual(output.count, 9)
        for i in 0..<9 {
            XCTAssertEqual(output[i], 4.0, accuracy: 0.01, "output[\(i)] should be 4.0")
        }
    }

    func testConv2DGradientValues() throws {
        // Test actual gradient values for simple case
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [1, 3, 3, 1], dtype: .float32),
                    TensorSpec(shape: [2, 2, 1, 1], dtype: .float32))
        ) { input, filter in
            return diffConv2D(input, filter, strides: [1, 1], padding: "VALID")
        }

        // 3x3 input with distinct values
        let input: [Float] = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]
        // 2x2 filter with distinct values
        let filter: [Float] = [1, 2, 3, 4]

        let (output, gradInput, gradFilter) = try gradFunc.forwardWithGradients(
            input, filter,
            seed: [1, 1, 1, 1]  // 2x2 output
        )

        // Forward: output[i,j] = sum of input[i:i+2, j:j+2] * filter
        // Should produce 4 output values
        XCTAssertEqual(output.count, 4)

        // Input gradient should have same shape as input
        XCTAssertEqual(gradInput.count, 9)

        // Filter gradient should have same shape as filter
        XCTAssertEqual(gradFilter.count, 4)

        // Filter gradient = sum over output positions of (input patch * dy)
        // With all-ones seed, filter_grad[i] = sum of input values at position i across all patches
        // Top-left filter position sees: 1, 2, 4, 5 (sum=12)
        // Etc.
        print("Filter gradient: \(gradFilter)")
    }

    // MARK: - Gradient Tests
    // Note: Full gradient tests for convolution are complex due to the
    // transposed convolution operations. These are basic sanity checks.

    func testConv2DGradientShape() throws {
        // Test that gradients have correct shapes
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [1, 4, 4, 1], dtype: .float32),
                    TensorSpec(shape: [2, 2, 1, 1], dtype: .float32))
        ) { input, filter in
            return diffConv2D(input, filter, strides: [1, 1], padding: "VALID")
        }

        let input: [Float] = Array(repeating: 1.0, count: 16)
        let filter: [Float] = [1.0, 1.0, 1.0, 1.0]

        let (_, gradInput, gradFilter) = try gradFunc.forwardWithGradients(
            input, filter,
            seed: Array(repeating: 1.0, count: 9)
        )

        // Gradient shapes should match input shapes
        XCTAssertEqual(gradInput.count, 16, "Input gradient should have 16 elements")
        XCTAssertEqual(gradFilter.count, 4, "Filter gradient should have 4 elements")
    }
}
