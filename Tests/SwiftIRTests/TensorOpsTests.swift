//===-- TensorOpsTests.swift - Phase 7 Tensor Tests ------*- Swift -*-===//
//
// SwiftIR - Phase 7: Tensor Operations for ML
// Tests for MLIR Tensor dialect operations
//
//===------------------------------------------------------------------===//

import Testing
@testable import SwiftIRXLA
@testable import SwiftIRCore
@testable import SwiftIRTypes
import MLIRCoreWrapper

@Suite("Phase 7: Tensor Operations")
struct TensorOpsTests {

    @Test("Tensor dialect loading")
    func testTensorDialectLoading() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)
        // If we got here without crashing, the dialect loaded successfully
        #expect(true)
    }

    @Test("Create empty tensor")
    func testCreateEmptyTensor() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)

        let builder = IRBuilder(context: ctx)

        // Create a 2x3 tensor of f32
        let tensor = TensorOps.empty(
            shape: [2, 3],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )

        // Verify we got a value back
        #expect(!mlirValueIsNullWrapper(tensor.handle))

        // Verify the type is a tensor
        let tensorType = tensor.getType()
        #expect(mlirTypeIsARankedTensorWrapper(tensorType))

        // Verify the shape
        #expect(mlirShapedTypeGetRankWrapper(tensorType) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(tensorType, 0) == 2)
        #expect(mlirShapedTypeGetDimSizeWrapper(tensorType, 1) == 3)
    }

    @Test("RankedTensorType creation")
    func testRankedTensorType() {
        let ctx = MLIRContext()

        // Create a tensor type: tensor<4x8xf32>
        let tensorType = RankedTensorType(
            shape: [4, 8],
            elementType: FloatType.f32(context: ctx),
            context: ctx
        )

        // Verify basic properties
        #expect(tensorType.isValid)
        #expect(tensorType.rank == 2)
        #expect(tensorType.shape == [4, 8])
        #expect(tensorType.hasStaticShape)
    }

    @Test("Multiple tensor shapes")
    func testMultipleTensorShapes() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)

        let builder = IRBuilder(context: ctx)

        // Test different tensor shapes
        let shapes: [[Int64]] = [
            [10],           // 1D tensor
            [3, 4],         // 2D tensor
            [2, 3, 4],      // 3D tensor
            [1, 2, 3, 4],   // 4D tensor
        ]

        for shape in shapes {
            let tensor = TensorOps.empty(
                shape: shape,
                elementType: FloatType.f32(context: ctx),
                in: builder
            )

            let tensorType = tensor.getType()
            #expect(mlirShapedTypeGetRankWrapper(tensorType) == Int64(shape.count))

            for (dim, size) in shape.enumerated() {
                #expect(mlirShapedTypeGetDimSizeWrapper(tensorType, dim) == size)
            }
        }
    }

    @Test("Different element types")
    func testDifferentElementTypes() {
        let ctx = MLIRContext()
        TensorDialect.register(with: ctx)

        let builder = IRBuilder(context: ctx)

        // Test f16 tensor
        let f16Tensor = TensorOps.empty(
            shape: [2, 2],
            elementType: FloatType.f16(context: ctx),
            in: builder
        )
        #expect(!mlirValueIsNullWrapper(f16Tensor.handle))

        // Test f32 tensor
        let f32Tensor = TensorOps.empty(
            shape: [2, 2],
            elementType: FloatType.f32(context: ctx),
            in: builder
        )
        #expect(!mlirValueIsNullWrapper(f32Tensor.handle))

        // Test f64 tensor
        let f64Tensor = TensorOps.empty(
            shape: [2, 2],
            elementType: FloatType.f64(context: ctx),
            in: builder
        )
        #expect(!mlirValueIsNullWrapper(f64Tensor.handle))

        // Test integer tensors
        let i32Tensor = TensorOps.empty(
            shape: [2, 2],
            elementType: IntegerType.i32(context: ctx),
            in: builder
        )
        #expect(!mlirValueIsNullWrapper(i32Tensor.handle))
    }

    @Test("SwiftIRXLA version info")
    func testSwiftIRXLAVersion() {
        #expect(SwiftIRXLA.version == "0.1.0-phase7")
        #expect(SwiftIRXLA.phase.contains("Phase 7"))
    }
}
