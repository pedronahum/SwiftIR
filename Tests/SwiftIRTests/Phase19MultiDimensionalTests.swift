// Phase19MultiDimensionalTests.swift
// Tests for Phase 19: Multi-dimensional Tensor Support
//
// These tests verify that 2D and higher-dimensional tensors work correctly
// on real PJRT hardware with proper gradient computation.

import XCTest
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase19MultiDimensionalTests: XCTestCase {

    // MARK: - 2D Tensor Basic Operations

    func test2DSquareGradient() throws {
        // f(X) = X² where X is 2x3
        // f'(X) = 2X
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return x * x
        }

        // 2x3 matrix flattened
        let input: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2X
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 8.0, accuracy: 0.01)
        XCTAssertEqual(gradient[4], 10.0, accuracy: 0.01)
        XCTAssertEqual(gradient[5], 12.0, accuracy: 0.01)
    }

    func test2DAdditionGradient() throws {
        // f(X) = X + X = 2X
        // f'(X) = 2
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return x + x
        }

        let input: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]
        let gradient = try gradFunc.gradient(input)

        // All gradients should be 2
        for i in 0..<6 {
            XCTAssertEqual(gradient[i], 2.0, accuracy: 0.01)
        }
    }

    func test2DCubeGradient() throws {
        // f(X) = X³
        // f'(X) = 3X²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return x * x * x
        }

        let input: [Float] = [
            1.0, 2.0,
            3.0, 4.0
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 3X² = [3, 12, 27, 48]
        XCTAssertEqual(gradient[0], 3.0, accuracy: 0.1)
        XCTAssertEqual(gradient[1], 12.0, accuracy: 0.1)
        XCTAssertEqual(gradient[2], 27.0, accuracy: 0.1)
        XCTAssertEqual(gradient[3], 48.0, accuracy: 0.1)
    }

    // MARK: - Batch Operations

    func testBatchSquareGradient() throws {
        // Batch of 4 samples, each with 3 features
        // f(X) = X²
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4, 3], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [
            1.0, 2.0, 3.0,   // Sample 0
            2.0, 3.0, 4.0,   // Sample 1
            3.0, 4.0, 5.0,   // Sample 2
            4.0, 5.0, 6.0    // Sample 3
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2X
        let expected: [Float] = [
            2.0, 4.0, 6.0,
            4.0, 6.0, 8.0,
            6.0, 8.0, 10.0,
            8.0, 10.0, 12.0
        ]

        for i in 0..<12 {
            XCTAssertEqual(gradient[i], expected[i], accuracy: 0.01)
        }
    }

    func testBatchPolynomial() throws {
        // f(X) = X² + X
        // f'(X) = 2X + 1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3, 2], dtype: .float32)
        ) { x in
            return x * x + x
        }

        let input: [Float] = [
            0.0, 1.0,   // Sample 0
            2.0, 3.0,   // Sample 1
            4.0, 5.0    // Sample 2
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2X + 1
        let expected: [Float] = [
            1.0, 3.0,   // 2*0+1, 2*1+1
            5.0, 7.0,   // 2*2+1, 2*3+1
            9.0, 11.0   // 2*4+1, 2*5+1
        ]

        for i in 0..<6 {
            XCTAssertEqual(gradient[i], expected[i], accuracy: 0.01)
        }
    }

    // MARK: - 2D Transcendental Operations

    func test2DExpGradient() throws {
        // f(X) = exp(X)
        // f'(X) = exp(X)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffExp(x)
        }

        let input: [Float] = [
            0.0, 1.0,
            -1.0, 0.5
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: exp(X)
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)  // exp(0)
        XCTAssertEqual(gradient[1], Float(M_E), accuracy: 0.01)  // exp(1)
        XCTAssertEqual(gradient[2], 1.0 / Float(M_E), accuracy: 0.01)  // exp(-1)
        XCTAssertEqual(gradient[3], exp(0.5), accuracy: 0.01)  // exp(0.5)
    }

    func test2DNegateGradient() throws {
        // f(X) = -X
        // f'(X) = -1
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffNegate(x)
        }

        let input: [Float] = [
            1.0, 2.0,
            3.0, 4.0
        ]
        let gradient = try gradFunc.gradient(input)

        // Gradient should be -1 everywhere
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], -1.0, accuracy: 0.01)
        }
    }

    func test2DLogGradient() throws {
        // f(X) = log(X)
        // f'(X) = 1/X
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffLog(x)
        }

        let input: [Float] = [
            1.0, 2.0,
            4.0, 0.5
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 1/X
        XCTAssertEqual(gradient[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.5, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.25, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 2.0, accuracy: 0.01)
    }

    // MARK: - Square Matrix Operations

    func testSquareMatrixGradient() throws {
        // 3x3 matrix
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3, 3], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 2X
        for i in 0..<9 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }

    // MARK: - Different Shapes

    func testTallMatrixGradient() throws {
        // 4x2 matrix (more rows than columns)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4, 2], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0
        ]
        let gradient = try gradFunc.gradient(input)

        for i in 0..<8 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }

    func testWideMatrixGradient() throws {
        // 2x4 matrix (more columns than rows)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        ]
        let gradient = try gradFunc.gradient(input)

        for i in 0..<8 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }

    // MARK: - Forward + Gradient Tests

    func test2DForwardWithGradient() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]
        let seed: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        let (output, gradient) = try gradFunc.forwardWithGradient(input, seed: seed)

        // Forward: X²
        let expectedOutput: [Float] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
        for i in 0..<6 {
            XCTAssertEqual(output[i], expectedOutput[i], accuracy: 0.01)
        }

        // Gradient: 2X
        for i in 0..<6 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }

    // MARK: - Chain Rule with 2D

    func test2DChainRule() throws {
        // f(X) = (X + X) * (X + X) = 4X²
        // f'(X) = 8X
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            let doubled = x + x
            return doubled * doubled
        }

        let input: [Float] = [
            1.0, 2.0,
            3.0, 4.0
        ]
        let gradient = try gradFunc.gradient(input)

        // Expected: 8X
        XCTAssertEqual(gradient[0], 8.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 16.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 24.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], 32.0, accuracy: 0.01)
    }

    // MARK: - Repeated Execution

    func test2DRepeatedExecution() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return x * x
        }

        // Execute multiple times with different inputs
        for scale in 1...3 {
            let s = Float(scale)
            let input: [Float] = [s, 2*s, 3*s, 4*s]
            let gradient = try gradFunc.gradient(input)

            for i in 0..<4 {
                XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
            }
        }
    }

    // MARK: - Edge Cases

    func test2DWithZeros() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [0.0, 0.0, 0.0, 0.0]
        let gradient = try gradFunc.gradient(input)

        // f'(0) = 0
        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.01)
        }
    }

    func test2DWithNegatives() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [-1.0, -2.0, -3.0, -4.0]
        let gradient = try gradFunc.gradient(input)

        // f'(x) = 2x
        XCTAssertEqual(gradient[0], -2.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], -4.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], -6.0, accuracy: 0.01)
        XCTAssertEqual(gradient[3], -8.0, accuracy: 0.01)
    }

    // MARK: - Larger Tensors

    func testLarger2DTensor() throws {
        // 8x8 matrix
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [8, 8], dtype: .float32)
        ) { x in
            return x * x
        }

        var input: [Float] = []
        for i in 0..<64 {
            input.append(Float(i + 1))
        }

        let gradient = try gradFunc.gradient(input)

        // Spot check a few values
        XCTAssertEqual(gradient[0], 2.0, accuracy: 0.01)   // 2*1
        XCTAssertEqual(gradient[7], 16.0, accuracy: 0.01)  // 2*8
        XCTAssertEqual(gradient[63], 128.0, accuracy: 0.1) // 2*64
    }

    // MARK: - Composed 2D Operations

    func test2DComposedExpSquare() throws {
        // f(X) = exp(X²)
        // f'(X) = 2X * exp(X²)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 2], dtype: .float32)
        ) { x in
            return diffExp(x * x)
        }

        let input: [Float] = [0.0, 0.5, 1.0, 0.25]
        let gradient = try gradFunc.gradient(input)

        // At x=0: grad = 0
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)

        // At x=0.5: grad = 1.0 * exp(0.25) ≈ 1.284
        XCTAssertEqual(gradient[1], 1.0 * exp(0.25), accuracy: 0.1)
    }

    // MARK: - Info Tests

    func test2DGradientFunctionInfo() throws {
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [2, 3], dtype: .float32)
        ) { x in
            return x * x
        }

        let info = gradFunc.info
        XCTAssertTrue(info.contains("PJRT"))
        XCTAssertTrue(info.contains("Input shape"))
    }

    // MARK: - Single Element 2D

    func test1x1MatrixGradient() throws {
        // 1x1 matrix is essentially a scalar
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 1], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [3.0]
        let gradient = try gradFunc.gradient(input)

        XCTAssertEqual(gradient[0], 6.0, accuracy: 0.01)  // 2*3
    }

    // MARK: - Row/Column Vectors

    func testRowVectorGradient() throws {
        // 1xN row vector
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 4], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }

    func testColumnVectorGradient() throws {
        // Nx1 column vector
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4, 1], dtype: .float32)
        ) { x in
            return x * x
        }

        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let gradient = try gradFunc.gradient(input)

        for i in 0..<4 {
            XCTAssertEqual(gradient[i], 2.0 * input[i], accuracy: 0.01)
        }
    }
}
