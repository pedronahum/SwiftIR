import Testing
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct TypeValidationTests {

    // MARK: - Type Validation Tests

    @Test("Validate scalar types as element types")
    func validateScalarElementTypes() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)
        let f32 = FloatType.f32(context: context)
        let idx = IndexType(context: context)

        #expect(TypeValidator.isValidElementType(i32))
        #expect(TypeValidator.isValidElementType(f32))
        #expect(TypeValidator.isValidElementType(idx))
    }

    @Test("Reject shaped types as element types")
    func rejectShapedElementTypes() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let tensor = RankedTensorType(shape: [2, 3], elementType: f32, context: context)

        #expect(!TypeValidator.isValidElementType(tensor))
    }

    @Test("Validate positive shape dimensions")
    func validatePositiveShapes() {
        #expect(TypeValidator.isValidShape([1, 2, 3]))
        #expect(TypeValidator.isValidShape([10, 20]))
        #expect(TypeValidator.isValidShape([]))  // Empty shape is valid
    }

    @Test("Validate dynamic shape dimensions")
    func validateDynamicShapes() {
        #expect(TypeValidator.isValidShape([-1, 2, 3]))  // Dynamic first dim
        #expect(TypeValidator.isValidShape([2, -1, 3]))  // Dynamic middle dim
        #expect(TypeValidator.isValidShape([-1, -1]))     // All dynamic
    }

    @Test("Reject invalid shape dimensions")
    func rejectInvalidShapes() {
        #expect(!TypeValidator.isValidShape([0, 2, 3]))   // Zero dimension
        #expect(!TypeValidator.isValidShape([1, -2, 3]))  // Negative (not -1)
    }

    @Test("Validate well-formed tensor type")
    func validateTensorType() {
        let context = MLIRContext()
        let f32 = FloatType.f32(context: context)
        let tensor = RankedTensorType(shape: [2, 3], elementType: f32, context: context)

        #expect(TypeValidator.isValidTensorType(tensor))
    }

    @Test("Validate well-formed memref type")
    func validateMemRefType() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)
        let memref = MemRefType(shape: [4, 8], elementType: i32, context: context)

        #expect(TypeValidator.isValidMemRefType(memref))
    }

    @Test("Validate well-formed vector type")
    func validateVectorType() {
        let context = MLIRContext()
        let f64 = FloatType.f64(context: context)
        let vector = VectorType(shape: [4], elementType: f64, context: context)

        #expect(TypeValidator.isValidVectorType(vector))
    }

    // MARK: - Shape Inference Tests

    @Test("Infer elementwise shape - same shapes")
    func inferElementwiseSameShapes() {
        let shape = ShapeInference.elementwiseShape([[2, 3], [2, 3]])
        #expect(shape == [2, 3])
    }

    @Test("Infer elementwise shape - with dynamic dimensions")
    func inferElementwiseWithDynamic() {
        let shape = ShapeInference.elementwiseShape([[-1, 3], [2, 3]])
        #expect(shape == [2, 3])  // Concrete dimension wins
    }

    @Test("Reject elementwise with incompatible shapes")
    func rejectIncompatibleElementwise() {
        let shape = ShapeInference.elementwiseShape([[2, 3], [2, 4]])
        #expect(shape == nil)
    }

    @Test("Infer matmul shape")
    func inferMatmulShape() {
        let shape = ShapeInference.matmulShape([2, 3], [3, 4])
        #expect(shape == [2, 4])
    }

    @Test("Infer matmul shape with dynamic dimensions")
    func inferMatmulWithDynamic() {
        let shape = ShapeInference.matmulShape([-1, 3], [3, -1])
        #expect(shape == [-1, -1])
    }

    @Test("Reject matmul with incompatible inner dimensions")
    func rejectIncompatibleMatmul() {
        let shape = ShapeInference.matmulShape([2, 3], [4, 5])
        #expect(shape == nil)
    }

    @Test("Infer broadcast shape - same shapes")
    func inferBroadcastSameShapes() {
        let shape = ShapeInference.broadcastShape([2, 3], [2, 3])
        #expect(shape == [2, 3])
    }

    @Test("Infer broadcast shape - with ones")
    func inferBroadcastWithOnes() {
        let shape = ShapeInference.broadcastShape([1, 3], [2, 3])
        #expect(shape == [2, 3])
    }

    @Test("Infer broadcast shape - different ranks")
    func inferBroadcastDifferentRanks() {
        let shape = ShapeInference.broadcastShape([3], [2, 3])
        #expect(shape == [2, 3])
    }

    @Test("Reject incompatible broadcast shapes")
    func rejectIncompatibleBroadcast() {
        let shape = ShapeInference.broadcastShape([2, 3], [2, 4])
        #expect(shape == nil)
    }

    @Test("Infer reduction shape - single axis")
    func inferReductionSingleAxis() {
        let shape = ShapeInference.reductionShape([2, 3, 4], axes: [1])
        #expect(shape == [2, 4])
    }

    @Test("Infer reduction shape - multiple axes")
    func inferReductionMultipleAxes() {
        let shape = ShapeInference.reductionShape([2, 3, 4], axes: [0, 2])
        #expect(shape == [3])
    }

    @Test("Infer reduction shape - all axes")
    func inferReductionAllAxes() {
        let shape = ShapeInference.reductionShape([2, 3, 4], axes: [0, 1, 2])
        #expect(shape == [])  // Scalar result
    }

    @Test("Reject reduction with invalid axes")
    func rejectInvalidReductionAxes() {
        let shape = ShapeInference.reductionShape([2, 3], axes: [3])
        #expect(shape == nil)
    }

    @Test("Infer reshape shape")
    func inferReshapeShape() {
        let shape = ShapeInference.reshapeShape([2, 3, 4], [6, 4])
        #expect(shape == [6, 4])
    }

    @Test("Infer reshape with dynamic dimension")
    func inferReshapeWithDynamic() {
        let shape = ShapeInference.reshapeShape([2, 3, 4], [-1, 4])
        #expect(shape == [6, 4])
    }

    @Test("Reject reshape with incompatible total elements")
    func rejectIncompatibleReshape() {
        let shape = ShapeInference.reshapeShape([2, 3, 4], [5, 5])
        #expect(shape == nil)
    }

    @Test("Infer transpose shape")
    func inferTransposeShape() {
        let shape = ShapeInference.transposeShape([2, 3, 4], permutation: [2, 0, 1])
        #expect(shape == [4, 2, 3])
    }

    @Test("Reject transpose with invalid permutation")
    func rejectInvalidTranspose() {
        let shape = ShapeInference.transposeShape([2, 3], permutation: [0, 0])  // Duplicate
        #expect(shape == nil)
    }

    @Test("Reject transpose with out of bounds permutation")
    func rejectOutOfBoundsTranspose() {
        let shape = ShapeInference.transposeShape([2, 3], permutation: [0, 5])
        #expect(shape == nil)
    }
}
