//===-- MLOpsTests.swift - ML Operations Tests -----------*- Swift -*-===//
//
// SwiftIR - Phase 7: ML Operations Tests
// Tests for machine learning operations
//
//===------------------------------------------------------------------===//

import Testing
@testable import SwiftIRXLA
@testable import SwiftIRCore
@testable import SwiftIRTypes
import MLIRCoreWrapper

@Suite("ML Operations")
struct MLOpsTests {

    @Test("Matrix multiplication - matmul")
    func testMatmul() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")

        let builder = IRBuilder(context: ctx)

        // Create two 2x2 tensors for multiplication
        let A = TensorOps.empty(
            shape: [2, 3],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [3, 4],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // C = A @ B should be [2, 4]
        let C = MLOps.matmul(lhs: A, rhs: B, in: builder)

        // Verify result shape
        let resultType = C.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 4)
    }

    @Test("Dot is alias for matmul")
    func testDot() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")

        let builder = IRBuilder(context: ctx)

        let A = TensorOps.empty(
            shape: [4, 5],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [5, 6],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Test that dot works (alias for matmul)
        let C = MLOps.dot(A, B, in: builder)

        let resultType = C.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 6)
    }

    @Test("ReLU activation")
    func testRelu() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [10, 20],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let activated = MLOps.relu(input, in: builder)

        // Verify output has same shape as input
        let resultType = activated.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 10)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 20)
    }

    @Test("Sigmoid activation")
    func testSigmoid() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("math")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [5, 10],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let activated = MLOps.sigmoid(input, in: builder)

        // Verify output shape matches input
        let resultType = activated.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
    }

    @Test("Tanh activation")
    func testTanh() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("math")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [8, 8],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let activated = MLOps.tanh(input, in: builder)

        // Verify operation created successfully
        #expect(!mlirValueIsNullWrapper(activated.handle))
    }

    @Test("Softmax operation")
    func testSoftmax() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("math")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [10, 100],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let normalized = MLOps.softmax(input, in: builder)

        // Verify shape preservation
        let resultType = normalized.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 10)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 100)
    }

    @Test("2D Convolution")
    func testConv2d() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")

        let builder = IRBuilder(context: ctx)

        // Input: [batch=1, height=28, width=28, channels=3]
        let input = TensorOps.empty(
            shape: [1, 28, 28, 3],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Kernel: [kernel_h=3, kernel_w=3, in_channels=3, out_channels=16]
        let kernel = TensorOps.empty(
            shape: [3, 3, 3, 16],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let output = MLOps.conv2d(input: input, kernel: kernel, in: builder)

        // Verify output shape
        let resultType = output.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 1)   // batch
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 3) == 16)  // filters
    }

    @Test("Element-wise addition")
    func testElementWiseAdd() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let A = TensorOps.empty(
            shape: [10, 10],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [10, 10],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let C = MLOps.add(A, B, in: builder)

        // Verify result
        #expect(!mlirValueIsNullWrapper(C.handle))
        let resultType = C.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 10)
    }

    @Test("Element-wise multiplication")
    func testElementWiseMul() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let A = TensorOps.empty(
            shape: [5, 5],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [5, 5],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let C = MLOps.mul(A, B, in: builder)

        #expect(!mlirValueIsNullWrapper(C.handle))
    }

    @Test("Element-wise subtraction")
    func testElementWiseSub() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let A = TensorOps.empty(
            shape: [3, 4],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [3, 4],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let C = MLOps.sub(A, B, in: builder)

        #expect(!mlirValueIsNullWrapper(C.handle))
    }

    @Test("Element-wise division")
    func testElementWiseDiv() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let A = TensorOps.empty(
            shape: [2, 2],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let B = TensorOps.empty(
            shape: [2, 2],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let C = MLOps.div(A, B, in: builder)

        #expect(!mlirValueIsNullWrapper(C.handle))
    }

    @Test("Chained ML operations")
    func testChainedOps() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")
        ctx.loadDialect("math")

        let builder = IRBuilder(context: ctx)

        // Simple neural network layer: relu(W @ x + b)
        let x = TensorOps.empty(
            shape: [10, 784],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let W = TensorOps.empty(
            shape: [784, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let b = TensorOps.empty(
            shape: [10, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Forward pass: W @ x
        let linear = MLOps.matmul(lhs: x, rhs: W, in: builder)

        // Add bias
        let withBias = MLOps.add(linear, b, in: builder)

        // Apply activation
        let activated = MLOps.relu(withBias, in: builder)

        // Verify final output shape
        let resultType = activated.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 10)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 128)
    }

    // MARK: - Phase 8 Tests: Reduction Operations

    @Test("Reduce sum operation")
    func testReduceSum() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [4, 5, 6],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Sum along axis 1
        let sum = MLOps.reduce_sum(input, axes: [1], in: builder)

        let resultType = sum.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 6)
    }

    @Test("Reduce mean operation")
    func testReduceMean() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [10, 20],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let mean = MLOps.reduce_mean(input, axes: [0], keepDims: true, in: builder)

        let resultType = mean.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 1)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 20)
    }

    // MARK: - Phase 8 Tests: Pooling Operations

    @Test("Max pooling 2D")
    func testMaxPool2d() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")

        let builder = IRBuilder(context: ctx)

        // Input: [batch=2, height=28, width=28, channels=3]
        let input = TensorOps.empty(
            shape: [2, 28, 28, 3],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Pool with 2x2 kernel, stride 2
        let pooled = MLOps.max_pool2d(
            input,
            kernelSize: [2, 2],
            strides: [2, 2],
            in: builder
        )

        let resultType = pooled.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 2)   // batch preserved
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 14)  // height halved
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 2) == 14)  // width halved
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 3) == 3)   // channels preserved
    }

    @Test("Average pooling 2D")
    func testAvgPool2d() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [1, 32, 32, 16],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let pooled = MLOps.avg_pool2d(
            input,
            kernelSize: [4, 4],
            strides: [4, 4],
            in: builder
        )

        let resultType = pooled.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 8)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 2) == 8)
    }

    // MARK: - Phase 8 Tests: Normalization

    @Test("Batch normalization")
    func testBatchNorm() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("arith")
        ctx.loadDialect("math")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [32, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let mean = TensorOps.empty(
            shape: [32, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let variance = TensorOps.empty(
            shape: [32, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let normalized = MLOps.batch_norm(
            input,
            mean: mean,
            variance: variance,
            in: builder
        )

        #expect(!mlirValueIsNullWrapper(normalized.handle))
        let resultType = normalized.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 32)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 128)
    }

    @Test("Layer normalization")
    func testLayerNorm() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")
        ctx.loadDialect("math")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [10, 512],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let normalized = MLOps.layer_norm(input, axes: [1], in: builder)

        #expect(!mlirValueIsNullWrapper(normalized.handle))
    }

    // MARK: - Phase 8 Tests: Fluent API

    @Test("Value extension - fluent API")
    func testFluentAPI() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let a = TensorOps.empty(
            shape: [5, 5],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let b = TensorOps.empty(
            shape: [5, 5],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Test fluent chaining
        let result = a.add(b, in: builder).relu(in: builder)

        #expect(!mlirValueIsNullWrapper(result.handle))
    }

    @Test("NNLayer - linear layer")
    func testNNLayerLinear() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [16, 784],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let weights = TensorOps.empty(
            shape: [784, 256],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let bias = TensorOps.empty(
            shape: [16, 256],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let output = NNLayer.linear(input: input, weights: weights, bias: bias, in: builder)

        let resultType = output.getType()
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 16)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 256)
    }

    @Test("NNLayer - dense with activation")
    func testNNLayerDense() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        ctx.loadDialect("linalg")
        ctx.loadDialect("arith")

        let builder = IRBuilder(context: ctx)

        let input = TensorOps.empty(
            shape: [8, 128],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let weights = TensorOps.empty(
            shape: [128, 64],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let bias = TensorOps.empty(
            shape: [8, 64],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        let output = NNLayer.dense(
            input: input,
            weights: weights,
            bias: bias,
            activation: { value, builder in
                MLOps.relu(value, in: builder)
            },
            in: builder
        )

        #expect(!mlirValueIsNullWrapper(output.handle))
    }
}
