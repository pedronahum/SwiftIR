//===-- StablehloIntegrationTest.swift - StableHLO Integration -*-Swift-*-===//
//
// SwiftIR - Phase 11: StableHLO Integration Tests
// Comprehensive end-to-end tests for StableHLO operations
//
//===-----------------------------------------------------------------------===//

import Testing
@testable import SwiftIRCore
@testable import SwiftIRTypes
@testable import SwiftIRStableHLO
@testable import SwiftIRXLA
import MLIRCoreWrapper

@Suite("StableHLO Integration Tests")
struct StablehloIntegrationTests {

    // MARK: - Complete CNN Layer Test

    @Test("Complete CNN layer with conv + relu + pool")
    func testCNNLayer() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=2, height=32, width=32, channels=3]
        let inputType = RankedTensorType(shape: [2, 32, 32, 3], elementType: f32, context: context)
        // Kernel: [3x3, 3 input channels, 64 output channels]
        let kernelType = RankedTensorType(shape: [3, 3, 3, 64], elementType: f32, context: context)

        let block = MLIRBlock(
            arguments: [inputType.typeHandle, kernelType.typeHandle],
            context: context
        )

        let input = block.getArgument(0)
        let kernel = block.getArgument(1)

        // Conv2D with padding and stride
        let conv = StableHLOMLOps.conv2d(
            input: input,
            kernel: kernel,
            strides: [1, 1],
            padding: [(1, 1), (1, 1)],  // same padding
            in: builder
        )

        // ReLU activation
        let relu = StableHLOMLOps.relu(conv, in: builder)

        // Max pooling 2x2
        let pool = StableHLOMLOps.maxPool2d(
            relu,
            windowSize: [2, 2],
            strides: [2, 2],
            padding: [(0, 0), (0, 0)],
            in: builder
        )

        // Verify all operations created successfully
        #expect(!conv.isNull)
        #expect(!relu.isNull)
        #expect(!pool.isNull)

        // Verify output shape [2, 16, 16, 64]
        let poolType = pool.getType()
        #expect(mlirTypeIsARankedTensorWrapper(poolType))
        #expect(mlirShapedTypeGetDimSizeWrapper(poolType, 0) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(poolType, 1) == 16)
        #expect(mlirShapedTypeGetDimSizeWrapper(poolType, 2) == 16)
        #expect(mlirShapedTypeGetDimSizeWrapper(poolType, 3) == 64)
    }

    // MARK: - Batch Normalization Integration

    @Test("Batch norm with conv2d integration")
    func testBatchNormWithConv() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=4, height=28, width=28, channels=32]
        let inputType = RankedTensorType(shape: [4, 28, 28, 32], elementType: f32, context: context)
        // BN params: [32 channels]
        let paramType = RankedTensorType(shape: [32], elementType: f32, context: context)

        let block = MLIRBlock(
            arguments: [
                inputType.typeHandle,
                paramType.typeHandle,
                paramType.typeHandle,
                paramType.typeHandle,
                paramType.typeHandle
            ],
            context: context
        )

        let input = block.getArgument(0)
        let scale = block.getArgument(1)
        let offset = block.getArgument(2)
        let mean = block.getArgument(3)
        let variance = block.getArgument(4)

        // Apply batch normalization
        let bn = StableHLOMLOps.batchNorm(
            input,
            scale: scale,
            offset: offset,
            mean: mean,
            variance: variance,
            epsilon: 1e-5,
            in: builder
        )

        // Then apply activation
        let activated = StableHLOMLOps.relu(bn, in: builder)

        #expect(!bn.isNull)
        #expect(!activated.isNull)

        // Verify shape preserved
        let bnType = bn.getType()
        #expect(mlirTypeIsARankedTensorWrapper(bnType))
        #expect(mlirShapedTypeGetDimSizeWrapper(bnType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(bnType, 3) == 32)
    }

    // MARK: - Reduction Pipeline

    @Test("Multi-stage reduction pipeline")
    func testReductionPipeline() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=8, height=16, width=16, channels=64]
        let inputType = RankedTensorType(shape: [8, 16, 16, 64], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Reduce spatial dimensions (global average pooling)
        let avgPool = StableHLOMLOps.avgPool2d(
            input,
            windowSize: [16, 16],
            strides: [1, 1],
            padding: [(0, 0), (0, 0)],
            in: builder
        )

        // Then reduce to get per-channel statistics
        let channelMean = StableHLOMLOps.reduceMean(avgPool, dimensions: [0], in: builder)
        let channelMax = StableHLOMLOps.reduceMax(avgPool, dimensions: [0], in: builder)

        #expect(!avgPool.isNull)
        #expect(!channelMean.isNull)
        #expect(!channelMax.isNull)

        // Verify final shapes
        let meanType = channelMean.getType()
        #expect(mlirTypeIsARankedTensorWrapper(meanType))
        #expect(mlirShapedTypeGetRankWrapper(meanType) == 3)  // [1, 1, 64]
    }

    // MARK: - Activation Combinations

    @Test("Multiple activation functions in sequence")
    func testActivationSequence() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        let tensorType = RankedTensorType(shape: [10, 10], elementType: f32, context: context)
        let block = MLIRBlock(arguments: [tensorType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Apply different activations
        let relu = StableHLOMLOps.relu(input, in: builder)
        let tanh = StableHLOMLOps.tanh(relu, in: builder)
        let sigmoid = StableHLOMLOps.sigmoid(tanh, in: builder)

        #expect(!relu.isNull)
        #expect(!tanh.isNull)
        #expect(!sigmoid.isNull)

        // All should preserve shape
        for value in [relu, tanh, sigmoid] {
            let type = value.getType()
            #expect(mlirTypeIsARankedTensorWrapper(type))
            #expect(mlirShapedTypeGetDimSizeWrapper(type, 0) == 10)
            #expect(mlirShapedTypeGetDimSizeWrapper(type, 1) == 10)
        }
    }

    // MARK: - Reshape and Matmul Pipeline

    @Test("Reshape followed by matmul (dense layer)")
    func testDenseLayerPipeline() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Feature map: [batch=4, height=8, width=8, channels=64]
        let featureType = RankedTensorType(shape: [4, 8, 8, 64], elementType: f32, context: context)
        // Weights for dense layer: [4096, 128]
        let weightsType = RankedTensorType(shape: [4096, 128], elementType: f32, context: context)

        let block = MLIRBlock(
            arguments: [featureType.typeHandle, weightsType.typeHandle],
            context: context
        )

        let features = block.getArgument(0)
        let weights = block.getArgument(1)

        // Reshape to flatten: [4, 8*8*64] = [4, 4096]
        let flattened = StableHLOMLOps.reshape(features, to: [4, 4096], in: builder)

        // Dense layer (matmul + bias would normally follow)
        let dense = StableHLOMLOps.matmul(lhs: flattened, rhs: weights, in: builder)

        #expect(!flattened.isNull)
        #expect(!dense.isNull)

        // Verify output shape [4, 128]
        let denseType = dense.getType()
        #expect(mlirTypeIsARankedTensorWrapper(denseType))
        #expect(mlirShapedTypeGetDimSizeWrapper(denseType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(denseType, 1) == 128)
    }

    // MARK: - Strided Convolution Test

    @Test("Strided convolution (downsampling)")
    func testStridedConvolution() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=1, height=64, width=64, channels=32]
        let inputType = RankedTensorType(shape: [1, 64, 64, 32], elementType: f32, context: context)
        // Kernel: [3x3, 32 in, 64 out]
        let kernelType = RankedTensorType(shape: [3, 3, 32, 64], elementType: f32, context: context)

        let block = MLIRBlock(
            arguments: [inputType.typeHandle, kernelType.typeHandle],
            context: context
        )

        let input = block.getArgument(0)
        let kernel = block.getArgument(1)

        // Strided conv (2x2 stride) for downsampling
        let conv = StableHLOMLOps.conv2d(
            input: input,
            kernel: kernel,
            strides: [2, 2],
            padding: [(1, 1), (1, 1)],
            in: builder
        )

        #expect(!conv.isNull)

        // Verify downsampled output: [1, 32, 32, 64]
        let convType = conv.getType()
        #expect(mlirTypeIsARankedTensorWrapper(convType))
        #expect(mlirShapedTypeGetDimSizeWrapper(convType, 1) == 32)
        #expect(mlirShapedTypeGetDimSizeWrapper(convType, 2) == 32)
        #expect(mlirShapedTypeGetDimSizeWrapper(convType, 3) == 64)
    }

    // MARK: - Broadcast and Element-wise Ops

    @Test("Broadcast with element-wise operations")
    func testBroadcastElementwise() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Tensor: [batch=4, features=10]
        let tensorType = RankedTensorType(shape: [4, 10], elementType: f32, context: context)
        // Bias: [features=10]
        let biasType = RankedTensorType(shape: [10], elementType: f32, context: context)

        let block = MLIRBlock(
            arguments: [tensorType.typeHandle, biasType.typeHandle],
            context: context
        )

        let tensor = block.getArgument(0)
        let bias = block.getArgument(1)

        // Broadcast bias to match tensor shape
        let broadcastedBias = StableHLOMLOps.broadcast(bias, to: [4, 10], in: builder)

        // Add broadcasted bias
        let result = StableHLOMLOps.add(tensor, broadcastedBias, in: builder)

        #expect(!broadcastedBias.isNull)
        #expect(!result.isNull)

        // Verify final shape
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 10)
    }
}
