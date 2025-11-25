// Phase23MultiInputTests.swift
// Tests for Phase 23: Multi-input gradient support

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase23MultiInputTests: XCTestCase {

    // MARK: - Basic Two-Input Tests

    func testTwoInputAddition() throws {
        // f(x, y) = x + y
        // df/dx = 1, df/dy = 1
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

        // Forward: [6, 8, 10, 12]
        XCTAssertEqual(output[0], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 12.0, accuracy: 0.01)

        // Gradients should be [1, 1, 1, 1] for both
        for i in 0..<4 {
            XCTAssertEqual(grad1[i], 1.0, accuracy: 0.01)
            XCTAssertEqual(grad2[i], 1.0, accuracy: 0.01)
        }
    }

    func testTwoInputMultiplication() throws {
        // f(x, y) = x * y
        // df/dx = y, df/dy = x
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x * y
        }

        let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input2: [Float] = [2.0, 3.0, 4.0, 5.0]

        let (grad1, grad2) = try gradFunc.gradients(input1, input2)

        // df/dx = y = [2, 3, 4, 5]
        XCTAssertEqual(grad1[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(grad1[1], 3.0, accuracy: 0.01)
        XCTAssertEqual(grad1[2], 4.0, accuracy: 0.01)
        XCTAssertEqual(grad1[3], 5.0, accuracy: 0.01)

        // df/dy = x = [1, 2, 3, 4]
        XCTAssertEqual(grad2[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(grad2[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(grad2[2], 3.0, accuracy: 0.01)
        XCTAssertEqual(grad2[3], 4.0, accuracy: 0.01)
    }

    func testTwoInputSubtraction() throws {
        // f(x, y) = x - y
        // df/dx = 1, df/dy = -1
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x - y
        }

        let input1: [Float] = [5.0, 6.0, 7.0, 8.0]
        let input2: [Float] = [1.0, 2.0, 3.0, 4.0]

        let (output, grad1, grad2) = try gradFunc.forwardWithGradients(
            input1, input2,
            seed: [1, 1, 1, 1]
        )

        // Forward: [4, 4, 4, 4]
        for i in 0..<4 {
            XCTAssertEqual(output[i], 4.0, accuracy: 0.01)
        }

        // df/dx = 1, df/dy = -1
        for i in 0..<4 {
            XCTAssertEqual(grad1[i], 1.0, accuracy: 0.01)
            XCTAssertEqual(grad2[i], -1.0, accuracy: 0.01)
        }
    }

    func testTwoInputDivision() throws {
        // f(x, y) = x / y
        // df/dx = 1/y, df/dy = -x/y²
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x / y
        }

        let input1: [Float] = [4.0, 9.0, 16.0, 25.0]
        let input2: [Float] = [2.0, 3.0, 4.0, 5.0]

        let (grad1, grad2) = try gradFunc.gradients(input1, input2)

        // df/dx = 1/y = [0.5, 0.333, 0.25, 0.2]
        XCTAssertEqual(grad1[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(grad1[1], 1.0/3.0, accuracy: 0.01)
        XCTAssertEqual(grad1[2], 0.25, accuracy: 0.01)
        XCTAssertEqual(grad1[3], 0.2, accuracy: 0.01)

        // df/dy = -x/y² = [-1, -1, -1, -1]
        XCTAssertEqual(grad2[0], -1.0, accuracy: 0.01)
        XCTAssertEqual(grad2[1], -1.0, accuracy: 0.01)
        XCTAssertEqual(grad2[2], -1.0, accuracy: 0.01)
        XCTAssertEqual(grad2[3], -1.0, accuracy: 0.01)
    }

    // MARK: - Neural Network Style Tests

    func testLinearLayer() throws {
        // f(x, w) = x * w (elementwise for simplicity)
        // This simulates a simple linear transformation
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, w in
            return x * w
        }

        let activations: [Float] = [1.0, 2.0, 3.0, 4.0]
        let weights: [Float] = [0.5, 0.5, 0.5, 0.5]

        let (output, gradX, gradW) = try gradFunc.forwardWithGradients(
            activations, weights,
            seed: [1, 1, 1, 1]
        )

        // Forward: x * w = [0.5, 1, 1.5, 2]
        XCTAssertEqual(output[0], 0.5, accuracy: 0.01)
        XCTAssertEqual(output[1], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.5, accuracy: 0.01)
        XCTAssertEqual(output[3], 2.0, accuracy: 0.01)

        // df/dx = w = [0.5, 0.5, 0.5, 0.5]
        for i in 0..<4 {
            XCTAssertEqual(gradX[i], 0.5, accuracy: 0.01)
        }

        // df/dw = x = [1, 2, 3, 4]
        XCTAssertEqual(gradW[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradW[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradW[2], 3.0, accuracy: 0.01)
        XCTAssertEqual(gradW[3], 4.0, accuracy: 0.01)
    }

    func testMSELoss() throws {
        // f(pred, target) = (pred - target)²
        // df/dpred = 2(pred - target)
        // df/dtarget = -2(pred - target)
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { pred, target in
            let diff = pred - target
            return diff * diff
        }

        let predictions: [Float] = [1.0, 2.0, 3.0, 4.0]
        let targets: [Float] = [2.0, 2.0, 2.0, 2.0]

        let (output, gradPred, gradTarget) = try gradFunc.forwardWithGradients(
            predictions, targets,
            seed: [1, 1, 1, 1]
        )

        // Forward: (pred - target)² = [1, 0, 1, 4]
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)

        // df/dpred = 2(pred - target) = [-2, 0, 2, 4]
        XCTAssertEqual(gradPred[0], -2.0, accuracy: 0.01)
        XCTAssertEqual(gradPred[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradPred[2], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradPred[3], 4.0, accuracy: 0.01)

        // df/dtarget = -2(pred - target) = [2, 0, -2, -4]
        XCTAssertEqual(gradTarget[0], 2.0, accuracy: 0.01)
        XCTAssertEqual(gradTarget[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradTarget[2], -2.0, accuracy: 0.01)
        XCTAssertEqual(gradTarget[3], -4.0, accuracy: 0.01)
    }

    // MARK: - Complex Expressions

    func testComplexExpression() throws {
        // f(x, y) = x² * y + x * y²
        // df/dx = 2xy + y²
        // df/dy = x² + 2xy
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [4], dtype: .float32),
                    TensorSpec(shape: [4], dtype: .float32))
        ) { x, y in
            return x * x * y + x * y * y
        }

        let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input2: [Float] = [2.0, 1.0, 2.0, 1.0]

        let (grad1, grad2) = try gradFunc.gradients(input1, input2)

        // df/dx = 2xy + y²
        // At (1,2): 2*1*2 + 4 = 8
        // At (2,1): 2*2*1 + 1 = 5
        // At (3,2): 2*3*2 + 4 = 16
        // At (4,1): 2*4*1 + 1 = 9
        XCTAssertEqual(grad1[0], 8.0, accuracy: 0.01)
        XCTAssertEqual(grad1[1], 5.0, accuracy: 0.01)
        XCTAssertEqual(grad1[2], 16.0, accuracy: 0.01)
        XCTAssertEqual(grad1[3], 9.0, accuracy: 0.01)

        // df/dy = x² + 2xy
        // At (1,2): 1 + 2*1*2 = 5
        // At (2,1): 4 + 2*2*1 = 8
        // At (3,2): 9 + 2*3*2 = 21
        // At (4,1): 16 + 2*4*1 = 24
        XCTAssertEqual(grad2[0], 5.0, accuracy: 0.01)
        XCTAssertEqual(grad2[1], 8.0, accuracy: 0.01)
        XCTAssertEqual(grad2[2], 21.0, accuracy: 0.01)
        XCTAssertEqual(grad2[3], 24.0, accuracy: 0.01)
    }

    // MARK: - 2D Tensor Tests

    func testTwoInput2D() throws {
        // Test with 2D tensors
        let gradFunc = try compileGradientForPJRT(
            inputs: (TensorSpec(shape: [2, 2], dtype: .float32),
                    TensorSpec(shape: [2, 2], dtype: .float32))
        ) { x, y in
            return x * y
        }

        let input1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let input2: [Float] = [4.0, 3.0, 2.0, 1.0]

        let (output, grad1, grad2) = try gradFunc.forwardWithGradients(
            input1, input2,
            seed: [1, 1, 1, 1]
        )

        // Forward: elementwise multiply
        XCTAssertEqual(output[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 6.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 4.0, accuracy: 0.01)

        // Gradients
        XCTAssertEqual(grad1[0], 4.0, accuracy: 0.01)
        XCTAssertEqual(grad2[0], 1.0, accuracy: 0.01)
    }
}
