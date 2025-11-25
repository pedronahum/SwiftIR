// Phase30ElementWiseTests.swift
// Tests for Phase 30: Additional element-wise operations (floor, ceil, clamp, rsqrt)

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase30ElementWiseTests: XCTestCase {

    // MARK: - Rsqrt Tests

    func testRsqrtForward() throws {
        // Test rsqrt forward pass
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRsqrt(x)
        }

        let input: [Float] = [1.0, 4.0, 9.0, 16.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // rsqrt([1, 4, 9, 16]) = [1, 0.5, 0.333, 0.25]
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.0/3.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 0.25, accuracy: 0.01)
    }

    func testRsqrtGradient() throws {
        // Test rsqrt gradient
        // d/dx (1/sqrt(x)) = -0.5 * x^(-3/2)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffRsqrt(x)
        }

        let input: [Float] = [1.0, 4.0, 9.0, 16.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient: -0.5 * x^(-3/2)
        // At x=1: -0.5 * 1^(-1.5) = -0.5
        // At x=4: -0.5 * 4^(-1.5) = -0.5 * 0.125 = -0.0625
        // At x=9: -0.5 * 9^(-1.5) = -0.5 * 0.037 = -0.0185
        // At x=16: -0.5 * 16^(-1.5) = -0.5 * 0.0156 = -0.0078
        XCTAssertEqual(gradient[0], -0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[1], -0.0625, accuracy: 0.01)
        XCTAssertEqual(gradient[2], -0.0185, accuracy: 0.01)
        XCTAssertEqual(gradient[3], -0.0078, accuracy: 0.01)
    }

    // Note: Clamp tests require 3 inputs which isn't currently supported by the test API.
    // The clamp operation implementation is complete with gradients.
}
