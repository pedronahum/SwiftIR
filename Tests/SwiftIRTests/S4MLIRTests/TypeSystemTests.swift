import Testing
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct TypeSystemTests {

    // MARK: - Integer Type Tests

    @Test("Create i1 (boolean) type")
    func createBooleanType() {
        let context = MLIRContext()
        let i1 = IntegerType.i1(context: context)

        #expect(i1.isValid)
        #expect(i1.bitWidth == 1)
        #expect(!i1.isSigned)
    }

    @Test("Create i8 type")
    func createI8Type() {
        let context = MLIRContext()
        let i8 = IntegerType.i8(context: context)

        #expect(i8.isValid)
        #expect(i8.bitWidth == 8)
        #expect(!i8.isSigned)
    }

    @Test("Create i16 type")
    func createI16Type() {
        let context = MLIRContext()
        let i16 = IntegerType.i16(context: context)

        #expect(i16.isValid)
        #expect(i16.bitWidth == 16)
        #expect(!i16.isSigned)
    }

    @Test("Create i32 type")
    func createI32Type() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)

        #expect(i32.isValid)
        #expect(i32.bitWidth == 32)
        #expect(!i32.isSigned)
    }

    @Test("Create i64 type")
    func createI64Type() {
        let context = MLIRContext()
        let i64 = IntegerType.i64(context: context)

        #expect(i64.isValid)
        #expect(i64.bitWidth == 64)
        #expect(!i64.isSigned)
    }

    @Test("Create signed integer type")
    func createSignedIntegerType() {
        let context = MLIRContext()
        let si32 = IntegerType(bitWidth: 32, signed: true, context: context)

        #expect(si32.isValid)
        #expect(si32.bitWidth == 32)
        #expect(si32.isSigned)
    }

    @Test("Create custom bit width integer type")
    func createCustomBitWidthIntegerType() {
        let context = MLIRContext()
        let i13 = IntegerType(bitWidth: 13, context: context)

        #expect(i13.isValid)
        #expect(i13.bitWidth == 13)
        #expect(!i13.isSigned)
    }

    // MARK: - Float Type Tests

    @Test("Create f16 type")
    func createF16Type() {
        let context = MLIRContext()
        let f16 = FloatType.f16(context: context)

        #expect(f16.isValid)
        #expect(f16.bitWidth == 16)
    }

    @Test("Create f32 type")
    func createF32Type() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)

        #expect(f32.isValid)
        #expect(f32.bitWidth == 32)
    }

    @Test("Create f64 type")
    func createF64Type() {
        let context = MLIRContext()
        let f64 = FloatType.f64(context: context)

        #expect(f64.isValid)
        #expect(f64.bitWidth == 64)
    }

    // MARK: - Index Type Tests

    @Test("Create index type")
    func createIndexType() {
        let context = MLIRContext()
        let indexType = IndexType(context: context)

        #expect(indexType.isValid)
    }

    // MARK: - Type Reuse and Context Tests

    @Test("Multiple types share same context")
    func multipleTypesShareContext() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)
        let f32 = FloatType.f32(context: context)
        let idx = IndexType(context: context)

        #expect(i32.isValid)
        #expect(f32.isValid)
        #expect(idx.isValid)
    }

    @Test("Types with default context")
    func typesWithDefaultContext() {
        let i32 = IntegerType.i32()
        let f32 = FloatType.f32()
        let idx = IndexType()

        #expect(i32.isValid)
        #expect(f32.isValid)
        #expect(idx.isValid)
    }

    // MARK: - Tensor Type Tests

    @Test("Create ranked tensor type")
    func createRankedTensorType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let tensorType = RankedTensorType(shape: [2, 3, 4], elementType: f32, context: context)

        #expect(tensorType.isValid)
        #expect(tensorType.rank == 3)
        #expect(tensorType.shape == [2, 3, 4])
        #expect(tensorType.hasStaticShape)
    }

    @Test("Create 2D tensor type")
    func create2DTensorType() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)
        let tensorType = RankedTensorType(shape: [10, 20], elementType: i32, context: context)

        #expect(tensorType.isValid)
        #expect(tensorType.rank == 2)
        #expect(tensorType.shape == [10, 20])
        #expect(tensorType.hasStaticShape)
    }

    @Test("Create 1D tensor type")
    func create1DTensorType() {
        let context = MLIRContext()
        let f64 = FloatType.f64(context: context)
        let tensorType = RankedTensorType(shape: [128], elementType: f64, context: context)

        #expect(tensorType.isValid)
        #expect(tensorType.rank == 1)
        #expect(tensorType.shape == [128])
        #expect(tensorType.hasStaticShape)
    }

    @Test("Create unranked tensor type")
    func createUnrankedTensorType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let tensorType = UnrankedTensorType(elementType: f32, context: context)

        #expect(tensorType.isValid)
    }

    // MARK: - MemRef Type Tests

    @Test("Create memref type")
    func createMemRefType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let memrefType = MemRefType(shape: [4, 8], elementType: f32, context: context)

        #expect(memrefType.isValid)
        #expect(memrefType.rank == 2)
        #expect(memrefType.shape == [4, 8])
        #expect(memrefType.hasStaticShape)
    }

    @Test("Create 3D memref type")
    func create3DMemRefType() {
        let context = MLIRContext()
        let i64 = IntegerType.i64(context: context)
        let memrefType = MemRefType(shape: [2, 4, 6], elementType: i64, context: context)

        #expect(memrefType.isValid)
        #expect(memrefType.rank == 3)
        #expect(memrefType.shape == [2, 4, 6])
    }

    // MARK: - Vector Type Tests

    @Test("Create vector type")
    func createVectorType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let vectorType = VectorType(shape: [4], elementType: f32, context: context)

        #expect(vectorType.isValid)
        #expect(vectorType.rank == 1)
        #expect(vectorType.shape == [4])
        #expect(vectorType.hasStaticShape)
    }

    @Test("Create 2D vector type")
    func create2DVectorType() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)
        let vectorType = VectorType(shape: [4, 8], elementType: i32, context: context)

        #expect(vectorType.isValid)
        #expect(vectorType.rank == 2)
        #expect(vectorType.shape == [4, 8])
    }

    // MARK: - Complex Type Composition Tests

    @Test("Multiple shaped types with same element type")
    func multipleShapedTypesWithSameElementType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)

        let tensor = RankedTensorType(shape: [2, 3], elementType: f32, context: context)
        let memref = MemRefType(shape: [2, 3], elementType: f32, context: context)
        let vector = VectorType(shape: [4], elementType: f32, context: context)

        #expect(tensor.isValid)
        #expect(memref.isValid)
        #expect(vector.isValid)

        #expect(tensor.shape == [2, 3])
        #expect(memref.shape == [2, 3])
        #expect(vector.shape == [4])
    }
}
