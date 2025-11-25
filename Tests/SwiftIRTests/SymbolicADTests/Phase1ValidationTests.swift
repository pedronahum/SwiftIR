/// Phase 1 Validation Tests - Core Infrastructure
/// These tests validate the critical components of Phase 1

import Testing

@testable import SwiftIR

@Suite("Phase 1: Core Infrastructure", .serialized)
struct Phase1ValidationTests {

    init() {
        // Reset the graph builder before each test
        TracerGraphBuilder.shared.reset()
        TokenManager.shared.reset()
    }

    // MARK: - TensorShape Tests

    @Suite("TensorShape")
    struct TensorShapeTests {

        /// CRITICAL: Verify broadcasting rules match NumPy
        @Test("Broadcasting rules match NumPy")
        func broadcastingRules() {
            // Test case 1: [1, 3] with [3, 1] -> [3, 3]
            let shape1 = TensorShape(dimensions: [1, 3])
            let shape2 = TensorShape(dimensions: [3, 1])

            #expect(
                shape1.isBroadcastCompatible(with: shape2),
                "CRITICAL: [1, 3] and [3, 1] should be broadcast compatible")

            let result = shape1.broadcast(with: shape2)
            #expect(
                result.dimensions == [3, 3],
                "CRITICAL: [1, 3] broadcast with [3, 1] should give [3, 3]")

            // Test case 2: [5, 1, 3] with [1, 3] -> [5, 1, 3]
            let shape3 = TensorShape(dimensions: [5, 1, 3])
            let shape4 = TensorShape(dimensions: [1, 3])

            #expect(shape3.isBroadcastCompatible(with: shape4))

            let result2 = shape3.broadcast(with: shape4)
            #expect(result2.dimensions == [5, 1, 3])

            // Test case 3: [2, 3] with [4, 2, 3] -> [4, 2, 3]
            let shape5 = TensorShape(dimensions: [2, 3])
            let shape6 = TensorShape(dimensions: [4, 2, 3])

            #expect(shape5.isBroadcastCompatible(with: shape6))

            let result3 = shape5.broadcast(with: shape6)
            #expect(result3.dimensions == [4, 2, 3])

            // Test case 4: Incompatible shapes [2, 3] with [4, 5]
            let shape7 = TensorShape(dimensions: [2, 3])
            let shape8 = TensorShape(dimensions: [4, 5])

            #expect(
                !shape7.isBroadcastCompatible(with: shape8),
                "CRITICAL: [2, 3] and [4, 5] should NOT be broadcast compatible")
        }

        @Test("Shape reduction")
        func shapeReduction() {
            let shape = TensorShape(dimensions: [2, 3, 4])

            // Reduce axis 1, keep dims
            let reduced1 = shape.reduced(alongAxes: [1], keepDims: true)
            #expect(reduced1.dimensions == [2, 1, 4])

            // Reduce axis 1, don't keep dims
            let reduced2 = shape.reduced(alongAxes: [1], keepDims: false)
            #expect(reduced2.dimensions == [2, 4])

            // Reduce multiple axes
            let reduced3 = shape.reduced(alongAxes: [0, 2], keepDims: false)
            #expect(reduced3.dimensions == [3])
        }

        @Test("Scalar shape")
        func scalarShape() {
            let scalar = TensorShape.scalar
            #expect(scalar.rank == 0)
            #expect(scalar.dimensions == [])
            #expect(scalar.isFullyKnown)
            #expect(scalar.elementCount == 1)
        }
    }

    // MARK: - DType Tests

    @Suite("DType")
    struct DTypeTests {

        @Test("Type promotion")
        func typePromotion() {
            // Float promotion
            #expect(DType.float32.promoted(with: .float64) == .float64)
            #expect(DType.float16.promoted(with: .float32) == .float32)

            // Int + Float -> Float
            #expect(DType.int32.promoted(with: .float32) == .float32)

            // Complex promotion
            #expect(DType.float32.promoted(with: .complex64) == .complex64)
            #expect(DType.complex64.promoted(with: .complex128) == .complex128)

            // Int promotion
            #expect(DType.int8.promoted(with: .int32) == .int32)
        }

        @Test("Type properties")
        func typeProperties() {
            #expect(DType.float32.isFloatingPoint)
            #expect(!DType.int32.isFloatingPoint)

            #expect(DType.int32.isInteger)
            #expect(!DType.float32.isInteger)

            #expect(DType.uint32.isUnsigned)
            #expect(!DType.int32.isUnsigned)

            #expect(DType.complex64.isComplex)
            #expect(!DType.float32.isComplex)
        }
    }

    // MARK: - Token Tests

    @Suite("Token")
    struct TokenTests {

        init() {
            TokenManager.shared.reset()
        }

        /// CRITICAL: Verify token chaining enforces execution order
        @Test("Token chaining")
        func tokenChaining() {
            let token1 = Token.global
            let token2 = Token.create()
            let token3 = Token.create()

            // Each token should have a unique ID
            #expect(token1.id != token2.id, "CRITICAL: Tokens must have unique IDs")
            #expect(token2.id != token3.id, "CRITICAL: Tokens must have unique IDs")
            #expect(token1.id != token3.id, "CRITICAL: Tokens must have unique IDs")

            // Global token should always be ID 0
            #expect(token1.id == 0, "Global token should have ID 0")
        }

        @Test("Tokenized graph builder")
        func tokenizedGraphBuilder() {
            let builder = TokenizedGraphBuilder()

            // Start with global token
            #expect(builder.token.id == Token.global.id)

            // Execute operations and verify token updates
            var tokenIds: [UInt64] = [builder.token.id]

            let _ = builder.execute { _ in
                let nextToken = Token.create()
                tokenIds.append(nextToken.id)
                return ("result1", nextToken)
            }

            let _ = builder.execute { _ in
                let nextToken = Token.create()
                tokenIds.append(nextToken.id)
                return ("result2", nextToken)
            }

            // Verify tokens are chained (monotonically increasing IDs)
            for i in 1..<tokenIds.count {
                #expect(
                    tokenIds[i] > tokenIds[i - 1],
                    "CRITICAL: Token IDs should be monotonically increasing for proper chaining")
            }
        }
    }

    // MARK: - Tracer Tests

    @Suite("Tracer")
    struct TracerTests {

        init() {
            TracerGraphBuilder.shared.reset()
            TokenManager.shared.reset()
        }

        /// CRITICAL: Verify Tracer maintains value semantics
        @Test("Value semantics")
        func valueSemantics() {
            var x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let y = x  // Create a copy

            // Store original ID
            let originalXId = x.valueId
            let yId = y.valueId

            // They should have the same ID initially (both point to same SSA value)
            #expect(originalXId == yId, "Copy should have same ID before mutation")

            // Mutate x
            x.move(by: Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32))

            // CRITICAL: After mutation, x should have NEW ID, y should have OLD ID
            #expect(
                x.valueId != y.valueId,
                "CRITICAL: After mutation, x should have new SSA value, y should retain original!")

            #expect(y.valueId == yId, "CRITICAL: y should retain its original SSA value!")
        }

        @Test("Arithmetic operations")
        func arithmetic() {
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32)

            // Test addition
            let c = a + b
            #expect(c.shape == TensorShape([2, 3]))
            #expect(c.dtype == .float32)
            #expect(c.valueId != a.valueId)
            #expect(c.valueId != b.valueId)

            // Test subtraction
            let d = a - b
            #expect(d.shape == TensorShape([2, 3]))

            // Test multiplication
            let e = a * b
            #expect(e.shape == TensorShape([2, 3]))

            // Test division
            let f = a / b
            #expect(f.shape == TensorShape([2, 3]))
        }

        @Test("Broadcasting arithmetic")
        func broadcastingArithmetic() {
            // Matrix [2, 3] + Vector [1, 3] -> Matrix [2, 3]
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 2.0, shape: TensorShape([1, 3]), dtype: .float32)

            let c = a + b
            #expect(
                c.shape.dimensions == [2, 3],
                "CRITICAL: Broadcasting should produce correct output shape")
        }

        @Test("Scalar operations")
        func scalarOperations() {
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)

            // Scalar multiplication
            let b = a * 2.0
            #expect(b.shape == TensorShape([2, 3]))

            // Scalar addition
            let c = a + 1.0
            #expect(c.shape == TensorShape([2, 3]))

            // Reverse order
            let d = 2.0 * a
            #expect(d.shape == TensorShape([2, 3]))
        }

        @Test("Reduction operations")
        func reductions() {
            let a = Tracer(value: 1.0, shape: TensorShape([2, 3, 4]), dtype: .float32)

            // Sum along axis 1, keep dims
            let sum1 = a.sum(alongAxes: [1], keepDims: true)
            #expect(sum1.shape.dimensions == [2, 1, 4])

            // Sum along axis 1, don't keep dims
            let sum2 = a.sum(alongAxes: [1], keepDims: false)
            #expect(sum2.shape.dimensions == [2, 4])

            // Mean
            let mean = a.mean(alongAxes: [0, 2], keepDims: false)
            #expect(mean.shape.dimensions == [3])
        }

        @Test("Factory methods")
        func factoryMethods() {
            let zeros = Tracer.zeros(shape: TensorShape([3, 4]), dtype: .float32)
            #expect(zeros.shape == TensorShape([3, 4]))
            #expect(zeros.dtype == .float32)

            let ones = Tracer.ones(shape: TensorShape([2, 5]), dtype: .float64)
            #expect(ones.shape == TensorShape([2, 5]))
            #expect(ones.dtype == .float64)
        }

        @Test("Print with token")
        func printWithToken() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)

            // Print with default global token
            let (printed1, token1) = x.print(label: "first")
            #expect(
                token1.id != Token.global.id,
                "Print should return a new token")

            // Chain another print
            let (printed2, token2) = printed1.print(label: "second", after: token1)
            #expect(
                token2.id > token1.id,
                "CRITICAL: Token chain should be properly ordered")

            // Shapes should be preserved
            #expect(printed1.shape == x.shape)
            #expect(printed2.shape == x.shape)
        }
    }

    // MARK: - Graph Building Tests

    @Suite("Graph Building")
    struct GraphBuildingTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Operation recording")
        func operationRecording() {
            TracerGraphBuilder.shared.reset()

            let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32)
            let _ = a + b

            let operations = TracerGraphBuilder.shared.getOperations()

            // Should have: constant, constant, binary
            #expect(operations.count == 3, "Should record 3 operations")

            // Verify operation types
            switch operations[0] {
            case .constant: break
            default: Issue.record("First operation should be constant")
            }

            switch operations[2] {
            case .binary(_, let op, _, _, _, _):
                #expect(op == .add, "Third operation should be add")
            default: Issue.record("Third operation should be binary")
            }
        }
    }

    // MARK: - Integration Test

    @Test("Phase 1 integration")
    func phase1Integration() {
        print("\n========================================")
        print("Phase 1: Core Infrastructure Validation")
        print("========================================\n")

        // Reset state
        TracerGraphBuilder.shared.reset()
        TokenManager.shared.reset()

        // 1. Value semantics
        print("Testing value semantics...")
        var x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
        let y = x
        let yId = y.valueId
        x.move(by: Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32))
        #expect(x.valueId != y.valueId)
        #expect(y.valueId == yId)
        print("✅ Value semantics verified")

        // 2. Token chaining
        print("Testing token chaining...")
        TokenManager.shared.reset()
        let token1 = Token.global
        let token2 = Token.create()
        #expect(token1.id != token2.id)
        print("✅ Token chaining verified")

        // 3. Broadcasting
        print("Testing broadcasting rules...")
        let shape1 = TensorShape(dimensions: [1, 3])
        let shape2 = TensorShape(dimensions: [3, 1])
        #expect(shape1.isBroadcastCompatible(with: shape2))
        let result = shape1.broadcast(with: shape2)
        #expect(result.dimensions == [3, 3])
        print("✅ Broadcasting rules verified")

        // 4. Arithmetic operations
        print("Testing arithmetic operations...")
        let a = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
        let b = Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32)
        let c = a + b
        #expect(c.shape == TensorShape([2, 3]))

        let d = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
        let e = Tracer(value: 2.0, shape: TensorShape([1, 3]), dtype: .float32)
        let f = d + e
        #expect(f.shape.dimensions == [2, 3])
        print("✅ Arithmetic operations verified")

        print("\n========================================")
        print("✅ PHASE 1 VALIDATION COMPLETE")
        print("========================================\n")
    }
}
