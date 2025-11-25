// Phase16GradientExecutionTests.swift
// Tests for Phase 16: DifferentiableTracer + PJRT Gradient Execution
//
// These tests verify that Swift AD gradients execute on real PJRT hardware
// with numerically correct results.

import XCTest
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase16GradientExecutionTests: XCTestCase {

    // MARK: - Basic Gradient Tests

    func testSquareGradient() throws {
        // f(x) = x²
        // f'(x) = 2x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2x = [2, 4, 6, 8]
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 6.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 8.0, accuracy: 0.001)
    }

    func testCubeGradient() throws {
        // f(x) = x³ = x * x * x
        // f'(x) = 3x²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 3x² = [3, 12, 27, 48]
        XCTAssertEqual(gradient[0], 3.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 12.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 27.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 48.0, accuracy: 0.01)
    }

    func testAdditionGradient() throws {
        // f(x) = x + x = 2x
        // f'(x) = 2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x + x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: [2, 2, 2, 2]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 2.0, accuracy: 0.001)
        }
    }

    func testSubtractionGradient() throws {
        // f(x) = x - x = 0
        // f'(x) = 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x - x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: [0, 0, 0, 0]
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.001)
        }
    }

    // MARK: - Forward + Backward Tests

    func testForwardWithGradient() throws {
        // f(x) = x²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [2.0, 3.0, 4.0, 5.0]
        let seed: [Float] = [1.0, 1.0, 1.0, 1.0]
        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: seed)

        // Forward: x² = [4, 9, 16, 25]
        XCTAssertEqual(output[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(output[1], 9.0, accuracy: 0.001)
        XCTAssertEqual(output[2], 16.0, accuracy: 0.001)
        XCTAssertEqual(output[3], 25.0, accuracy: 0.001)

        // Gradient: 2x = [4, 6, 8, 10]
        XCTAssertEqual(gradient[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 6.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 8.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 10.0, accuracy: 0.001)
    }

    func testScaledGradient() throws {
        // Test with non-unit seed (scaled gradient)
        // f(x) = x²
        // dy/dx = 2x, scaled by seed
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let seed: [Float] = [2.0, 2.0, 2.0, 2.0]  // Scale by 2
        let (_, gradient) = try gradFunc.forwardWithGradient(input, seed: seed)

        // Expected: 2 * 2x = [4, 8, 12, 16]
        XCTAssertEqual(gradient[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 8.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 12.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 16.0, accuracy: 0.001)
    }

    // MARK: - Polynomial Gradient Tests

    func testPolynomialGradient() throws {
        // f(x) = x² + x
        // f'(x) = 2x + 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x + x
        }

        let input: [Float] = [0.0, 1.0, 2.0, 3.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2x + 1 = [1, 3, 5, 7]
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 3.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 5.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 7.0, accuracy: 0.001)
    }

    func testQuadraticGradient() throws {
        // f(x) = x² - 2x
        // f'(x) = 2x - 2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x - x - x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2x - 2 = [0, 2, 4, 6]
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 4.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 6.0, accuracy: 0.001)
    }

    // MARK: - Numerical Gradient Verification

    func testNumericalGradientVerification() throws {
        // Verify gradient using finite differences
        // f(x) = x²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1], dtype: .float32)
        ) { x in
            return x * x
        }

        let x: Float = 3.0
        let h: Float = 0.0001

        // Forward pass
        let (_, gradient) = try gradFunc.forwardWithGradient([x], seed: [1.0])

        // Numerical gradient: (f(x+h) - f(x-h)) / 2h
        let fPlus = (x + h) * (x + h)
        let fMinus = (x - h) * (x - h)
        let numericalGrad = (fPlus - fMinus) / (2 * h)

        // Should match: f'(3) = 6
        XCTAssertEqual(gradient[0], numericalGrad, accuracy: 0.01)
        XCTAssertEqual(gradient[0], 6.0, accuracy: 0.001)
    }

    // MARK: - Chain Rule Tests

    func testChainRule() throws {
        // f(x) = (x + x) * (x + x) = 4x²
        // f'(x) = 8x
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let doubled = x + x
            return doubled * doubled
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 8x = [8, 16, 24, 32]
        XCTAssertEqual(gradient[0], 8.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 16.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], 24.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], 32.0, accuracy: 0.001)
    }

    // MARK: - Conversion Tests

    func testGradientCompiledFunctionToPJRT() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [4],
            dtype: .float32
        ) { x in
            return x * x
        }

        // Convert to PJRT
        let pjrtFunc = try gradFunc.toPJRTGradientFunction(backend: .cpu)

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try pjrtFunc.gradient(input)

        // Expected: 2x
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.001)
    }

    // MARK: - Edge Cases

    func testZeroInput() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [0.0, 0.0, 0.0, 0.0]
        let gradient = try gradFunc.gradient(input)

        // f'(0) = 0
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.001)
        }
    }

    func testNegativeInput() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [-1.0, -2.0, -3.0, -4.0]
        let gradient = try gradFunc.gradient(input)

        // f'(x) = 2x = [-2, -4, -6, -8]
        XCTAssertEqual(gradient[0], -2.0, accuracy: 0.001)
        XCTAssertEqual(gradient[1], -4.0, accuracy: 0.001)
        XCTAssertEqual(gradient[2], -6.0, accuracy: 0.001)
        XCTAssertEqual(gradient[3], -8.0, accuracy: 0.001)
    }

    // MARK: - Repeated Execution Tests

    func testRepeatedGradientExecution() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        // Execute multiple times
        for i in 1...3 {
            let scale = Float(i)
            let input: [Float] = [scale, scale * 2, scale * 3, scale * 4]
            let gradient = try gradFunc.gradient(input)

            // f'(x) = 2x
            XCTAssertEqual(gradient[0], 2 * scale, accuracy: 0.001)
            XCTAssertEqual(gradient[1], 4 * scale, accuracy: 0.001)
            XCTAssertEqual(gradient[2], 6 * scale, accuracy: 0.001)
            XCTAssertEqual(gradient[3], 8 * scale, accuracy: 0.001)
        }
    }

    // MARK: - Complex Gradient Tests

    func testFifthPowerGradient() throws {
        // f(x) = x⁵ = x * x * x * x * x
        // f'(x) = 5x⁴
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x * x * x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 5x⁴ = [5, 80, 405, 1280]
        XCTAssertEqual(gradient[0], 5.0, accuracy: 0.1)
        XCTAssertEqual(gradient[1], 80.0, accuracy: 1.0)
        XCTAssertEqual(gradient[2], 405.0, accuracy: 5.0)
        XCTAssertEqual(gradient[3], 1280.0, accuracy: 10.0)
    }

    // MARK: - Info Tests

    func testPJRTGradientFunctionInfo() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let info = gradFunc.info
        XCTAssertTrue(info.contains("PJRT"))
        XCTAssertTrue(info.contains("Gradient"))
        XCTAssertTrue(info.contains("Input shape"))
    }

    // MARK: - Higher-Degree Polynomial Tests

    func testFourthPowerGradient() throws {
        // f(x) = x⁴ = (x²)²
        // f'(x) = 4x³
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let squared = x * x
            return squared * squared
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        // Expected: 4x³ = [4, 32, 108, 256]
        XCTAssertEqual(gradient[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 32.0, accuracy: 0.1)
        XCTAssertEqual(gradient[2], 108.0, accuracy: 0.5)
        XCTAssertEqual(gradient[3], 256.0, accuracy: 1.0)
    }
}
