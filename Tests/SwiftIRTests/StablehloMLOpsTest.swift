//===-- StablehloMLOpsTest.swift - StableHLO ML Ops Tests -*-Swift-*-===//
//
// SwiftIR - Phase 11: StableHLO ML Operations Tests
// Tests for high-level ML operations built on StableHLO
//
//===--------------------------------------------------------------------===//

import Testing
@testable import SwiftIRCore
@testable import SwiftIRTypes
@testable import SwiftIRStableHLO
@testable import SwiftIRXLA
import MLIRCoreWrapper

@Suite("StableHLO ML Operations")
struct StablehloMLOpsTests {

    // MARK: - Batch Normalization

    @Test("Batch normalization operation")
    func testBatchNormalization() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=4, height=8, width=8, channels=16]
        let inputType = RankedTensorType(shape: [4, 8, 8, 16], elementType: f32, context: context)
        // Scale, offset, mean, variance: [channels=16]
        let paramType = RankedTensorType(shape: [16], elementType: f32, context: context)

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

        // Test batch norm operation
        let result = StableHLOMLOps.batchNorm(
            input,
            scale: scale,
            offset: offset,
            mean: mean,
            variance: variance,
            epsilon: 1e-5,
            in: builder
        )

        #expect(!result.isNull)
    }

    // MARK: - Pooling Operations

    @Test("Max pooling 2D operation")
    func testMaxPool2D() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=1, height=28, width=28, channels=32]
        let inputType = RankedTensorType(shape: [1, 28, 28, 32], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Test max pool with 2x2 window
        let result = StableHLOMLOps.maxPool2d(
            input,
            windowSize: [2, 2],
            strides: [2, 2],
            padding: [(0, 0), (0, 0)],
            in: builder
        )

        #expect(!result.isNull)

        // Output shape should be [1, 14, 14, 32]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 14)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 2) == 14)
    }

    @Test("Average pooling 2D operation")
    func testAvgPool2D() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=2, height=32, width=32, channels=64]
        let inputType = RankedTensorType(shape: [2, 32, 32, 64], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Test avg pool with 2x2 window
        let result = StableHLOMLOps.avgPool2d(
            input,
            windowSize: [2, 2],
            strides: [2, 2],
            padding: [(0, 0), (0, 0)],
            in: builder
        )

        #expect(!result.isNull)

        // Output shape should be [2, 16, 16, 64]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 16)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 2) == 16)
    }

    // MARK: - Reduction Operations

    @Test("Reduce mean operation")
    func testReduceMean() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=4, features=10]
        let inputType = RankedTensorType(shape: [4, 10], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Reduce along batch dimension
        let result = StableHLOMLOps.reduceMean(input, dimensions: [0], in: builder)

        #expect(!result.isNull)

        // Output shape should be [10]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 1)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 10)
    }

    @Test("Reduce max operation")
    func testReduceMax() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=8, height=16, width=16]
        let inputType = RankedTensorType(shape: [8, 16, 16], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Reduce along spatial dimensions
        let result = StableHLOMLOps.reduceMax(input, dimensions: [1, 2], in: builder)

        #expect(!result.isNull)

        // Output shape should be [8]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 1)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 8)
    }

    @Test("Reduce min operation")
    func testReduceMin() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Input: [batch=4, features=20]
        let inputType = RankedTensorType(shape: [4, 20], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        // Reduce along feature dimension
        let result = StableHLOMLOps.reduceMin(input, dimensions: [1], in: builder)

        #expect(!result.isNull)

        // Output shape should be [4]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetRankWrapper(resultType) == 1)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 4)
    }

    // MARK: - Activation Functions

    @Test("ReLU activation")
    func testReLU() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        let inputType = RankedTensorType(shape: [10, 10], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let result = StableHLOMLOps.relu(input, in: builder)

        #expect(!result.isNull)
    }

    @Test("Sigmoid activation")
    func testSigmoid() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        let inputType = RankedTensorType(shape: [8, 8], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let result = StableHLOMLOps.sigmoid(input, in: builder)

        #expect(!result.isNull)
    }

    @Test("Tanh activation")
    func testTanh() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        let inputType = RankedTensorType(shape: [6, 6], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let result = StableHLOMLOps.tanh(input, in: builder)

        #expect(!result.isNull)
    }

    // MARK: - Matrix Operations

    @Test("Matrix multiplication")
    func testMatmul() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let builder = IRBuilder(context: context)

        // Matrix A: [M=4, K=8]
        let lhsType = RankedTensorType(shape: [4, 8], elementType: f32, context: context)
        // Matrix B: [K=8, N=16]
        let rhsType = RankedTensorType(shape: [8, 16], elementType: f32, context: context)

        let block = MLIRBlock(arguments: [lhsType.typeHandle, rhsType.typeHandle], context: context)
        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        let result = StableHLOMLOps.matmul(lhs: lhs, rhs: rhs, in: builder)

        #expect(!result.isNull)

        // Output shape should be [4, 16]
        let resultType = result.getType()
        #expect(mlirTypeIsARankedTensorWrapper(resultType))
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 0) == 4)
        #expect(mlirShapedTypeGetDimSizeWrapper(resultType, 1) == 16)
    }
}
