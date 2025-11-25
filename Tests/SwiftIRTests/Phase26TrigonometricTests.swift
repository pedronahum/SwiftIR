// Phase26TrigonometricTests.swift
// Tests for Phase 26: Trigonometric operations (sin, cos)

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase26TrigonometricTests: XCTestCase {

    // MARK: - Sine Tests

    func testSineForward() throws {
        // Test sin forward pass
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSin(x)
        }

        let input: [Float] = [0.0, Float.pi / 2, Float.pi, 3 * Float.pi / 2]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // sin(0) = 0, sin(π/2) = 1, sin(π) = 0, sin(3π/2) = -1
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], -1.0, accuracy: 0.01)
    }

    func testSineGradient() throws {
        // d/dx sin(x) = cos(x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSin(x)
        }

        let input: [Float] = [0.0, Float.pi / 2, Float.pi, 3 * Float.pi / 2]
        let gradient = try gradFunc.gradient(input)

        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(3π/2) = 0
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], -1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
    }

    // MARK: - Cosine Tests

    func testCosineForward() throws {
        // Test cos forward pass
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffCos(x)
        }

        let input: [Float] = [0.0, Float.pi / 2, Float.pi, 3 * Float.pi / 2]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(3π/2) = 0
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], -1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 0.0, accuracy: 0.01)
    }

    func testCosineGradient() throws {
        // d/dx cos(x) = -sin(x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffCos(x)
        }

        let input: [Float] = [0.0, Float.pi / 2, Float.pi, 3 * Float.pi / 2]
        let gradient = try gradFunc.gradient(input)

        // -sin(0) = 0, -sin(π/2) = -1, -sin(π) = 0, -sin(3π/2) = 1
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], -1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - Combined Tests

    func testSinCosIdentity() throws {
        // Test sin²(x) + cos²(x) = 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let sinX = diffSin(x)
            let cosX = diffCos(x)
            return sinX * sinX + cosX * cosX
        }

        let input: [Float] = [0.0, 0.5, 1.0, 2.0]
        let (output, _) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Should be 1.0 for all inputs
        for i in 0..<4 {
            XCTAssertEqual(output[i], 1.0, accuracy: 0.01, "sin²(x) + cos²(x) should be 1.0")
        }
    }

    func testSinCosIdentityGradient() throws {
        // d/dx (sin²(x) + cos²(x)) = 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let sinX = diffSin(x)
            let cosX = diffCos(x)
            return sinX * sinX + cosX * cosX
        }

        let input: [Float] = [0.0, 0.5, 1.0, 2.0]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be 0 everywhere
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.01, "Gradient of sin²+cos² should be 0")
        }
    }

    func testSinCosCombined() throws {
        // f(x) = sin(x) * cos(x)
        // f'(x) = cos²(x) - sin²(x) = cos(2x)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return diffSin(x) * diffCos(x)
        }

        let input: [Float] = [0.0, Float.pi / 4, Float.pi / 2, Float.pi]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: [1, 1, 1, 1])

        // Forward: sin(x)*cos(x)
        // sin(0)*cos(0) = 0, sin(π/4)*cos(π/4) = 0.5, sin(π/2)*cos(π/2) = 0, sin(π)*cos(π) = 0
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(output[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 0.0, accuracy: 0.01)

        // Gradient: cos(2x)
        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(2π) = 1
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], -1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)
    }

    // MARK: - 2D Tensor Tests

    func testSine2D() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffSin(x)
        }

        let input: [Float] = [0.0, Float.pi / 2, Float.pi, 3 * Float.pi / 2]
        let gradient = try gradFunc.gradient(input)

        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(3π/2) = 0
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], -1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
    }
}
