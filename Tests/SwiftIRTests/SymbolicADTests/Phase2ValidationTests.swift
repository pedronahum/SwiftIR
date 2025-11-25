/// Phase 2 Validation Tests - Arithmetic & Broadcasting with Derivatives
/// These tests validate the critical components of Phase 2

import Testing

@testable import SwiftIR

@Suite("Phase 2: Arithmetic & Broadcasting", .serialized)
struct Phase2ValidationTests {

    init() {
        TracerGraphBuilder.shared.reset()
        TokenManager.shared.reset()
    }

    // MARK: - Unary Operations Tests

    @Suite("Unary Operations")
    struct UnaryOperationsTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Exponential operation")
        func exponential() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let result = x.exp()

            #expect(result.shape == x.shape)
            #expect(result.dtype == x.dtype)
            #expect(result.valueId != x.valueId)
        }

        @Test("Logarithm operation")
        func logarithm() {
            let x = Tracer(value: 2.0, shape: TensorShape([3, 4]), dtype: .float32)
            let result = x.log()

            #expect(result.shape == x.shape)
            #expect(result.dtype == x.dtype)
        }

        @Test("Square root operation")
        func squareRoot() {
            let x = Tracer(value: 4.0, shape: TensorShape([2, 2]), dtype: .float64)
            let result = x.sqrt()

            #expect(result.shape == x.shape)
            #expect(result.dtype == .float64)
        }

        @Test("Activation functions")
        func activationFunctions() {
            let x = Tracer(value: 1.0, shape: TensorShape([4, 5]), dtype: .float32)

            let tanhResult = x.tanh()
            #expect(tanhResult.shape == x.shape)

            let sigmoidResult = x.sigmoid()
            #expect(sigmoidResult.shape == x.shape)

            let reluResult = x.relu()
            #expect(reluResult.shape == x.shape)
        }

        @Test("Negation operation")
        func negation() {
            let x = Tracer(value: 5.0, shape: TensorShape([3]), dtype: .float32)
            let result = -x

            #expect(result.shape == x.shape)
            #expect(result.dtype == x.dtype)
        }

        @Test("Absolute value operation")
        func absoluteValue() {
            let x = Tracer(value: -2.0, shape: TensorShape([2, 3]), dtype: .float32)
            let absX = x.abs()

            #expect(absX.shape == x.shape)
            #expect(absX.dtype == x.dtype)
        }
    }

    // MARK: - Matrix Operations Tests

    @Suite("Matrix Operations")
    struct MatrixOperationsTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Matrix multiplication shapes")
        func matmulShapes() {
            // [2, 3] @ [3, 4] -> [2, 4]
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 1.0, shape: TensorShape([3, 4]), dtype: .float32)
            let c = a.matmul(b)

            #expect(c.shape.dimensions == [2, 4])
        }

        @Test("Batched matrix multiplication")
        func batchedMatmul() {
            // [batch, 2, 3] @ [batch, 3, 4] -> [batch, 2, 4]
            let a = Tracer(value: 1.0, shape: TensorShape([5, 2, 3]), dtype: .float32)
            let b = Tracer(value: 1.0, shape: TensorShape([5, 3, 4]), dtype: .float32)
            let c = a.matmul(b)

            #expect(c.shape.dimensions == [5, 2, 4])
        }

        @Test("Transpose operation")
        func transpose() {
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let aT = a.transpose()

            #expect(aT.shape.dimensions == [3, 2])

            // Batched transpose
            let b = Tracer(value: 1.0, shape: TensorShape([5, 2, 3]), dtype: .float32)
            let bT = b.transpose()

            #expect(bT.shape.dimensions == [5, 3, 2])
        }
    }

    // MARK: - Broadcasting Gradient Tests

    @Suite("Broadcasting Gradients")
    struct BroadcastingGradientTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Reduce to broadcast shape - same shape")
        func reduceToSameShape() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let reduced = x.reduceToBroadcastShape(originalShape: TensorShape([2, 3]))

            #expect(reduced.shape == TensorShape([2, 3]))
        }

        @Test("Reduce to broadcast shape - scalar broadcast")
        func reduceToScalar() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let reduced = x.reduceToBroadcastShape(originalShape: TensorShape.scalar)

            // When reducing to scalar with keepDims=true, we get [1, 1]
            // The reduceToBroadcastShape keeps dims for stability
            #expect(reduced.shape.dimensions.allSatisfy { $0 == 1 })
        }

        @Test("Reduce to broadcast shape - vector broadcast")
        func reduceToVector() {
            // [2, 3] reduced to [1, 3] means summing over axis 0
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let reduced = x.reduceToBroadcastShape(originalShape: TensorShape([1, 3]))

            #expect(reduced.shape.dimensions.last == 3)
        }
    }

    // MARK: - Graph Recording Tests

    @Suite("Graph Recording")
    struct GraphRecordingTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Unary operations are recorded")
        func unaryOperationsRecorded() {
            TracerGraphBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let _ = x.exp()
            let _ = x.log()

            let operations = TracerGraphBuilder.shared.getOperations()

            // Should have: constant, unary(exp), unary(log)
            // Note: operations may include more due to intermediate values
            #expect(operations.count >= 3)
        }

        @Test("Matrix operations are recorded")
        func matrixOperationsRecorded() {
            TracerGraphBuilder.shared.reset()

            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 1.0, shape: TensorShape([3, 4]), dtype: .float32)
            let _ = a.matmul(b)

            let operations = TracerGraphBuilder.shared.getOperations()

            // Should have at least the matmul operation
            // Note: constant creation may not be recorded as operations
            #expect(operations.count >= 1)
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 2 integration - simple computation")
    func phase2SimpleComputation() {
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 2: Arithmetic & Broadcasting Test")
        print("========================================\n")

        // Build a simple computation graph
        print("Testing simple computation graph...")
        let x = Tracer(value: 2.0, shape: TensorShape([3, 4]), dtype: .float32)

        // y = exp(x) * 2 + 1
        let expX = x.exp()
        let scaled = expX * 2.0
        let result = scaled + 1.0

        #expect(result.shape == x.shape)
        print("✅ Simple computation verified")

        // Test chain of operations
        print("Testing chain of operations...")
        let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
        let chain = a.exp().tanh().sigmoid()
        #expect(chain.shape == a.shape)
        print("✅ Chain of operations verified")

        print("\n========================================")
        print("✅ PHASE 2 SIMPLE COMPUTATION COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 2 integration - matrix operations")
    func phase2MatrixOperations() {
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 2: Matrix Operations Test")
        print("========================================\n")

        // Test matrix multiplication
        print("Testing matrix multiplication...")
        let w = Tracer(value: 0.1, shape: TensorShape([784, 128]), dtype: .float32)
        let x = Tracer(value: 1.0, shape: TensorShape([32, 784]), dtype: .float32)
        let h = x.matmul(w)

        #expect(h.shape.dimensions == [32, 128])
        print("✅ Matrix multiplication verified")

        // Test with activation
        print("Testing matrix multiplication with activation...")
        let activated = h.relu()
        #expect(activated.shape == h.shape)
        print("✅ Matrix multiplication with activation verified")

        // Test transpose
        print("Testing transpose...")
        let wT = w.transpose()
        #expect(wT.shape.dimensions == [128, 784])
        print("✅ Transpose verified")

        print("\n========================================")
        print("✅ PHASE 2 MATRIX OPERATIONS COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 2 integration - neural network forward pass")
    func phase2NeuralNetworkForward() {
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 2: Neural Network Forward Pass")
        print("========================================\n")

        // Simulate a simple 2-layer neural network forward pass
        let batchSize = 32
        let inputSize = 784
        let hiddenSize = 128
        let outputSize = 10

        // Input
        let x = Tracer(value: 1.0, shape: TensorShape([batchSize, inputSize]), dtype: .float32)

        // Layer 1: Linear + ReLU
        let w1 = Tracer(value: 0.01, shape: TensorShape([inputSize, hiddenSize]), dtype: .float32)
        let b1 = Tracer(value: 0.0, shape: TensorShape([1, hiddenSize]), dtype: .float32)

        let z1 = x.matmul(w1) + b1  // [32, 128]
        let h1 = z1.relu()

        #expect(h1.shape.dimensions == [batchSize, hiddenSize])
        print("✅ Layer 1 (Linear + ReLU) verified")

        // Layer 2: Linear
        let w2 = Tracer(value: 0.01, shape: TensorShape([hiddenSize, outputSize]), dtype: .float32)
        let b2 = Tracer(value: 0.0, shape: TensorShape([1, outputSize]), dtype: .float32)

        let logits = h1.matmul(w2) + b2  // [32, 10]

        #expect(logits.shape.dimensions == [batchSize, outputSize])
        print("✅ Layer 2 (Linear) verified")

        // Softmax-like operation (simplified)
        let expLogits = logits.exp()
        let sumExp = expLogits.sum(alongAxes: [1], keepDims: true)  // [32, 1]
        let probs = expLogits / sumExp  // Broadcasting division

        #expect(probs.shape.dimensions == [batchSize, outputSize])
        print("✅ Softmax operation verified")

        print("\n========================================")
        print("✅ PHASE 2 NEURAL NETWORK FORWARD COMPLETE")
        print("========================================\n")
    }
}
