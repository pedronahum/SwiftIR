// Phase37GatherScatterTests.swift
// Tests for Phase 37: Gather/Scatter for Embedding

import XCTest
import _Differentiation
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase37GatherScatterTests: XCTestCase {

    // MARK: - Gather Tests (Embedding Lookup)

    func testGatherSingleIndex() throws {
        // Simple embedding lookup: select one row
        // Input: [5, 3] embedding matrix (5 vocab, 3 dim)
        // Indices: [1] (single index, select row 0)
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [5, 3], dtype: .float32)
        ) { embeddings in
            // Index 0 (as float, will be cast to int32)
            let indices = withoutDerivative(at: createConstant(0.0, shape: [1], dtype: .int32))
            return diffGather(embeddings, indices: indices)
        }

        // Embedding matrix
        let input: [Float] = [
            1, 2, 3,    // Row 0
            4, 5, 6,    // Row 1
            7, 8, 9,    // Row 2
            10, 11, 12, // Row 3
            13, 14, 15  // Row 4
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 3)  // 1 row × 3 columns
        )

        // Should gather row 0
        XCTAssertEqual(output.count, 3)
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)  // Row 0, col 0
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)  // Row 0, col 1
        XCTAssertEqual(output[2], 3.0, accuracy: 0.01)  // Row 0, col 2
    }

    func testGatherSecondRow() throws {
        // Gather row 1 from embedding matrix
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3, 2], dtype: .float32)
        ) { embeddings in
            // Index 1
            let indices = withoutDerivative(at: createConstant(1.0, shape: [1], dtype: .int32))
            return diffGather(embeddings, indices: indices)
        }

        let input: [Float] = [
            1, 2,  // Row 0
            3, 4,  // Row 1
            5, 6   // Row 2
        ]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 2)
        )

        // Should gather row 1: [3, 4]
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output[0], 3.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 4.0, accuracy: 0.01)
    }

    func testGatherGradient() throws {
        // Test that gradient propagates correctly
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [3, 2], dtype: .float32)
        ) { embeddings in
            // Gather row 1
            let indices = withoutDerivative(at: createConstant(1.0, shape: [1], dtype: .int32))
            let gathered = diffGather(embeddings, indices: indices)
            // Sum to get a scalar loss
            return diffSum(gathered)
        }

        let input: [Float] = [
            1, 2,  // Row 0
            3, 4,  // Row 1
            5, 6   // Row 2
        ]

        let gradient = try gradFunc.gradient(input)

        // Gradient should be 1.0 for row 1, 0.0 for rows 0 and 2
        XCTAssertEqual(gradient.count, 6)
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)  // Row 0, col 0: not gathered
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)  // Row 0, col 1
        XCTAssertEqual(gradient[2], 1.0, accuracy: 0.01)  // Row 1, col 0: gathered
        XCTAssertEqual(gradient[3], 1.0, accuracy: 0.01)  // Row 1, col 1: gathered
        XCTAssertEqual(gradient[4], 0.0, accuracy: 0.01)  // Row 2, col 0: not gathered
        XCTAssertEqual(gradient[5], 0.0, accuracy: 0.01)  // Row 2, col 1
    }

    // MARK: - Scatter Tests

    func testScatterSingleUpdate() throws {
        // Scatter a single update to a specific position
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 2], dtype: .float32)
        ) { updates in
            let base = withoutDerivative(at: createConstant(0.0, shape: [3, 2], dtype: .float32))
            let indices = withoutDerivative(at: createConstant(1.0, shape: [1], dtype: .int32))
            return diffScatter(base, indices: indices, updates: updates)
        }

        let input: [Float] = [10, 20]  // Update for index 1

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 6)  // 3 × 2
        )

        // Output shape: [3, 2] = 6 elements
        XCTAssertEqual(output.count, 6)

        // Row 0: all zeros
        XCTAssertEqual(output[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 0.0, accuracy: 0.01)

        // Row 1: [10, 20]
        XCTAssertEqual(output[2], 10.0, accuracy: 0.01)
        XCTAssertEqual(output[3], 20.0, accuracy: 0.01)

        // Row 2: all zeros
        XCTAssertEqual(output[4], 0.0, accuracy: 0.01)
        XCTAssertEqual(output[5], 0.0, accuracy: 0.01)
    }

    func testScatterToFirstRow() throws {
        // Scatter update to row 0
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [1, 3], dtype: .float32)
        ) { updates in
            let base = withoutDerivative(at: createConstant(0.0, shape: [4, 3], dtype: .float32))
            let indices = withoutDerivative(at: createConstant(0.0, shape: [1], dtype: .int32))
            return diffScatter(base, indices: indices, updates: updates)
        }

        let input: [Float] = [1, 2, 3]

        let (output, _) = try gradFunc.forwardWithGradient(
            input,
            seed: Array(repeating: 1.0, count: 12)
        )

        XCTAssertEqual(output.count, 12)

        // Row 0: [1, 2, 3]
        XCTAssertEqual(output[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(output[1], 2.0, accuracy: 0.01)
        XCTAssertEqual(output[2], 3.0, accuracy: 0.01)

        // Other rows: zeros
        for i in 3..<12 {
            XCTAssertEqual(output[i], 0.0, accuracy: 0.01)
        }
    }

    // MARK: - Round-trip Tests (Gather then Scatter gradient)

    func testGatherScatterRoundTrip() throws {
        // Test that gather gradient (which uses scatter) works correctly
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [4, 3], dtype: .float32)
        ) { embeddings in
            let indices = withoutDerivative(at: createConstant(2.0, shape: [1], dtype: .int32))
            let gathered = diffGather(embeddings, indices: indices)
            // Sum all gathered values
            return diffSum(gathered)
        }

        let input: [Float] = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ]

        let gradient = try gradFunc.gradient(input)

        // Gradient should be 1.0 for row 2, 0.0 for other rows
        XCTAssertEqual(gradient.count, 12)

        // Row 0: not gathered, gradient = 0
        XCTAssertEqual(gradient[0], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[1], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[2], 0.0, accuracy: 0.01)

        // Row 1: not gathered, gradient = 0
        XCTAssertEqual(gradient[3], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[4], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[5], 0.0, accuracy: 0.01)

        // Row 2: gathered, so gradient = 1
        XCTAssertEqual(gradient[6], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[7], 1.0, accuracy: 0.01)
        XCTAssertEqual(gradient[8], 1.0, accuracy: 0.01)

        // Row 3: not gathered, gradient = 0
        XCTAssertEqual(gradient[9], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[10], 0.0, accuracy: 0.01)
        XCTAssertEqual(gradient[11], 0.0, accuracy: 0.01)
    }

    func testEmbeddingLayerSimulation() throws {
        // Simulate a simple embedding layer: lookup + downstream computation
        let gradFunc = try compileGradientForPJRT(
            input: TensorSpec(shape: [10, 8], dtype: .float32)  // 10 vocab, 8 dim
        ) { embeddings in
            // Lookup token 1
            let indices = withoutDerivative(at: createConstant(1.0, shape: [1], dtype: .int32))
            let gathered = diffGather(embeddings, indices: indices)  // [1, 8]

            // Apply a simple transformation: multiply by 2
            let scale = withoutDerivative(at: createConstant(2.0, shape: [1, 8], dtype: .float32))
            let scaled = gathered * scale

            // Sum to scalar
            return diffSum(scaled)
        }

        let input: [Float] = Array(0..<80).map(Float.init)

        let gradient = try gradFunc.gradient(input)

        XCTAssertEqual(gradient.count, 80)

        // Token 1 (row 1, indices 8-15): gradient = 2.0 (due to scale)
        for i in 8..<16 {
            XCTAssertEqual(gradient[i], 2.0, accuracy: 0.01)
        }

        // Other tokens: gradient = 0.0
        for i in 0..<8 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.01)
        }
        for i in 16..<80 {
            XCTAssertEqual(gradient[i], 0.0, accuracy: 0.01)
        }
    }
}
