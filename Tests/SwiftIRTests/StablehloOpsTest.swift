//===-- StablehloOpsTest.swift - StableHLO Operations Tests -*-Swift-*-===//
//
// SwiftIR - Phase 11: StableHLO Operation Tests
// Tests for StableHLO operation builders
//
//===--------------------------------------------------------------------===//

import Testing
@testable import SwiftIRCore
@testable import SwiftIRTypes
@testable import SwiftIRStableHLO
import MLIRCoreWrapper

@Suite("StableHLO Operations")
struct StablehloOpsTests {

    // MARK: - Element-wise Binary Operations

    @Test("StableHLO add operation")
    func testAddOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [2, 3], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        // Create dummy values (we're just testing operation creation, not execution)
        let module = MLIRModule(context: context)
        let region = MLIRRegion()
        let block = MLIRBlock(arguments: [tensorType.typeHandle, tensorType.typeHandle], context: context)

        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        // Create add operation
        let addOp = Stablehlo.add(lhs, rhs, location: location, context: context)

        #expect(!addOp.isNull)
        #expect(addOp.name == "stablehlo.add")
    }

    @Test("StableHLO multiply operation")
    func testMultiplyOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [4, 4], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let module = MLIRModule(context: context)
        let block = MLIRBlock(arguments: [tensorType.typeHandle, tensorType.typeHandle], context: context)

        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        let mulOp = Stablehlo.multiply(lhs, rhs, location: location, context: context)

        #expect(!mulOp.isNull)
        #expect(mulOp.name == "stablehlo.multiply")
    }

    @Test("StableHLO maximum operation")
    func testMaximumOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [10], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [tensorType.typeHandle, tensorType.typeHandle], context: context)

        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        let maxOp = Stablehlo.maximum(lhs, rhs, location: location, context: context)

        #expect(!maxOp.isNull)
        #expect(maxOp.name == "stablehlo.maximum")
    }

    // MARK: - Matrix Operations

    @Test("StableHLO dot operation")
    func testDotOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let lhsType = RankedTensorType(shape: [4, 8], elementType: f32, context: context)
        let rhsType = RankedTensorType(shape: [8, 16], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [4, 16], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [lhsType.typeHandle, rhsType.typeHandle], context: context)

        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        let dotOp = Stablehlo.dot(lhs, rhs, resultType: resultType.typeHandle, location: location, context: context)

        #expect(!dotOp.isNull)
        #expect(dotOp.name == "stablehlo.dot")
    }

    @Test("StableHLO dot_general operation")
    func testDotGeneralOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let lhsType = RankedTensorType(shape: [2, 3, 4], elementType: f32, context: context)
        let rhsType = RankedTensorType(shape: [2, 4, 5], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [2, 3, 5], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [lhsType.typeHandle, rhsType.typeHandle], context: context)

        let lhs = block.getArgument(0)
        let rhs = block.getArgument(1)

        let dotGenOp = Stablehlo.dotGeneral(lhs, rhs, resultType: resultType.typeHandle, location: location, context: context)

        #expect(!dotGenOp.isNull)
        #expect(dotGenOp.name == "stablehlo.dot_general")
    }

    // MARK: - Activation Functions

    @Test("StableHLO exponential operation")
    func testExponentialOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [10, 10], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [tensorType.typeHandle], context: context)
        let input = block.getArgument(0)

        let expOp = Stablehlo.exponential(input, location: location, context: context)

        #expect(!expOp.isNull)
        #expect(expOp.name == "stablehlo.exponential")
    }

    @Test("StableHLO tanh operation")
    func testTanhOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [5, 5], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [tensorType.typeHandle], context: context)
        let input = block.getArgument(0)

        let tanhOp = Stablehlo.tanh(input, location: location, context: context)

        #expect(!tanhOp.isNull)
        #expect(tanhOp.name == "stablehlo.tanh")
    }

    @Test("StableHLO logistic (sigmoid) operation")
    func testLogisticOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [8, 8], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [tensorType.typeHandle], context: context)
        let input = block.getArgument(0)

        let sigmoidOp = Stablehlo.logistic(input, location: location, context: context)

        #expect(!sigmoidOp.isNull)
        #expect(sigmoidOp.name == "stablehlo.logistic")
    }

    // MARK: - Shape Operations

    @Test("StableHLO reshape operation")
    func testReshapeOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let inputType = RankedTensorType(shape: [2, 6], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [3, 4], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let reshapeOp = Stablehlo.reshape(input, resultType: resultType.typeHandle, location: location, context: context)

        #expect(!reshapeOp.isNull)
        #expect(reshapeOp.name == "stablehlo.reshape")
    }

    @Test("StableHLO broadcast_in_dim operation")
    func testBroadcastInDimOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let inputType = RankedTensorType(shape: [3], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [2, 3, 4], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let broadcastOp = Stablehlo.broadcastInDim(input, resultType: resultType.typeHandle, location: location, context: context)

        #expect(!broadcastOp.isNull)
        #expect(broadcastOp.name == "stablehlo.broadcast_in_dim")
    }

    @Test("StableHLO transpose operation")
    func testTransposeOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let inputType = RankedTensorType(shape: [2, 3, 4], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [4, 2, 3], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle], context: context)
        let input = block.getArgument(0)

        let transposeOp = Stablehlo.transpose(input, permutation: [2, 0, 1], resultType: resultType.typeHandle, location: location, context: context)

        #expect(!transposeOp.isNull)
        #expect(transposeOp.name == "stablehlo.transpose")
    }

    // MARK: - Convolution Operations

    @Test("StableHLO convolution operation")
    func testConvolutionOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        // Input: [batch=1, height=28, width=28, channels=1]
        let inputType = RankedTensorType(shape: [1, 28, 28, 1], elementType: f32, context: context)
        // Kernel: [height=3, width=3, in_channels=1, out_channels=32]
        let kernelType = RankedTensorType(shape: [3, 3, 1, 32], elementType: f32, context: context)
        // Output: [batch=1, height=26, width=26, channels=32] (assuming no padding)
        let resultType = RankedTensorType(shape: [1, 26, 26, 32], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle, kernelType.typeHandle], context: context)

        let input = block.getArgument(0)
        let kernel = block.getArgument(1)

        let convOp = Stablehlo.convolution(input, kernel: kernel, resultType: resultType.typeHandle, location: location, context: context)

        #expect(!convOp.isNull)
        #expect(convOp.name == "stablehlo.convolution")
    }

    // MARK: - Reduction Operations

    @Test("StableHLO reduce operation")
    func testReduceOperation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let inputType = RankedTensorType(shape: [4, 8, 16], elementType: f32, context: context)
        let resultType = RankedTensorType(shape: [4, 16], elementType: f32, context: context)
        let scalarType = RankedTensorType(shape: [], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let block = MLIRBlock(arguments: [inputType.typeHandle, scalarType.typeHandle], context: context)

        let input = block.getArgument(0)
        let initValue = block.getArgument(1)

        let reduceOp = Stablehlo.reduce(input, initValue: initValue, dimensions: [1], resultType: resultType.typeHandle, location: location, context: context)

        #expect(!reduceOp.isNull)
        #expect(reduceOp.name == "stablehlo.reduce")
    }

    // MARK: - Integration Test

    @Test("Build simple StableHLO computation")
    func testSimpleComputation() throws {
        let context = MLIRContext()
        context.loadAllDialects()
        _ = loadStablehloDialect(context)

        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [10], elementType: f32, context: context)
        let location = MLIRLocation.unknown(in: context)

        let module = MLIRModule(context: context)
        let block = MLIRBlock(arguments: [tensorType.typeHandle, tensorType.typeHandle], context: context)

        let x = block.getArgument(0)
        let y = block.getArgument(1)

        // Build computation: (x + y) * x
        let addOp = Stablehlo.add(x, y, location: location, context: context)
        block.append(addOp)
        let sum = addOp.getResult(0)

        let mulOp = Stablehlo.multiply(sum, x, location: location, context: context)
        block.append(mulOp)

        // Verify both operations were created
        #expect(!addOp.isNull)
        #expect(!mulOp.isNull)
        #expect(addOp.name == "stablehlo.add")
        #expect(mulOp.name == "stablehlo.multiply")
    }
}
