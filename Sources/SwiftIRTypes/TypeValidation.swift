/// Type validation and checking utilities for S4MLIR
import SwiftIRCore
import MLIRCoreWrapper

// MARK: - Type Validation

/// Validates that a type is well-formed and compatible
public struct TypeValidator {

    /// Validates that an element type is suitable for shaped types
    public static func isValidElementType<T: MLIRType>(_ type: T) -> Bool {
        // Element types must be scalar types (integer, float, index)
        let handle = type.typeHandle

        // Check if it's a valid scalar type
        return !mlirTypeIsATensorWrapper(handle)
            && !mlirTypeIsAMemRefWrapper(handle)
            && !mlirTypeIsAVectorWrapper(handle)
    }

    /// Validates that a shape is valid (all dimensions > 0 or dynamic)
    public static func isValidShape(_ shape: [Int64]) -> Bool {
        // Empty shape is valid (scalar)
        if shape.isEmpty {
            return true
        }

        // All dimensions must be positive or -1 (dynamic)
        return shape.allSatisfy { $0 > 0 || $0 == -1 }
    }

    /// Checks if two types are compatible (can be used interchangeably)
    public static func areTypesCompatible<T: MLIRType, U: MLIRType>(_ type1: T, _ type2: U) -> Bool {
        let handle1 = type1.typeHandle
        let handle2 = type2.typeHandle

        // For now, use simple pointer equality
        // In the future, this could be enhanced with structural equality
        return handle1.ptr == handle2.ptr
    }

    /// Validates that a tensor type is well-formed
    public static func isValidTensorType(_ tensorType: RankedTensorType) -> Bool {
        guard tensorType.isValid else { return false }
        guard isValidShape(tensorType.shape) else { return false }
        return isValidElementType(tensorType.elementType)
    }

    /// Validates that a memref type is well-formed
    public static func isValidMemRefType(_ memrefType: MemRefType) -> Bool {
        guard memrefType.isValid else { return false }
        guard isValidShape(memrefType.shape) else { return false }
        return isValidElementType(memrefType.elementType)
    }

    /// Validates that a vector type is well-formed
    public static func isValidVectorType(_ vectorType: VectorType) -> Bool {
        guard vectorType.isValid else { return false }
        guard isValidShape(vectorType.shape) else { return false }
        return isValidElementType(vectorType.elementType)
    }
}

// MARK: - Shape Inference

/// Shape inference utilities for tensor operations
public struct ShapeInference {

    /// Infers the output shape for element-wise operations (same shape as inputs)
    public static func elementwiseShape(_ shapes: [[Int64]]) -> [Int64]? {
        guard !shapes.isEmpty else { return nil }

        let firstShape = shapes[0]
        let rank = firstShape.count

        // All shapes must have the same rank
        guard shapes.allSatisfy({ $0.count == rank }) else {
            return nil
        }

        var result = [Int64](repeating: -1, count: rank)

        for dim in 0..<rank {
            let dimSizes = shapes.map { $0[dim] }

            // Find the concrete dimension size (if any)
            let concreteSizes = dimSizes.filter { $0 != -1 }

            if concreteSizes.isEmpty {
                // All dynamic
                result[dim] = -1
            } else {
                // Check all concrete sizes are the same
                guard Set(concreteSizes).count == 1 else {
                    return nil  // Incompatible shapes
                }
                result[dim] = concreteSizes[0]
            }
        }

        return result
    }

    /// Infers the output shape for matrix multiplication (A: [m, k] × B: [k, n] -> [m, n])
    public static func matmulShape(_ lhs: [Int64], _ rhs: [Int64]) -> [Int64]? {
        // Matrices must be 2D
        guard lhs.count == 2 && rhs.count == 2 else {
            return nil
        }

        let m = lhs[0]
        let k1 = lhs[1]
        let k2 = rhs[0]
        let n = rhs[1]

        // Inner dimensions must match (or be dynamic)
        if k1 != -1 && k2 != -1 && k1 != k2 {
            return nil
        }

        return [m, n]
    }

    /// Infers the output shape for broadcast operations
    public static func broadcastShape(_ lhs: [Int64], _ rhs: [Int64]) -> [Int64]? {
        let maxRank = max(lhs.count, rhs.count)
        var result = [Int64](repeating: -1, count: maxRank)

        // Pad shorter shape with 1s on the left
        let lhsPadded = Array(repeating: Int64(1), count: maxRank - lhs.count) + lhs
        let rhsPadded = Array(repeating: Int64(1), count: maxRank - rhs.count) + rhs

        for i in 0..<maxRank {
            let l = lhsPadded[i]
            let r = rhsPadded[i]

            if l == r {
                result[i] = l
            } else if l == 1 {
                result[i] = r
            } else if r == 1 {
                result[i] = l
            } else if l == -1 || r == -1 {
                // One is dynamic, use the other if concrete, otherwise dynamic
                result[i] = (l == -1) ? r : l
            } else {
                // Incompatible shapes
                return nil
            }
        }

        return result
    }

    /// Infers the output shape for reduction operations along specified axes
    public static func reductionShape(_ inputShape: [Int64], axes: [Int]) -> [Int64]? {
        guard !axes.isEmpty else { return inputShape }

        // Check axes are valid
        guard axes.allSatisfy({ $0 >= 0 && $0 < inputShape.count }) else {
            return nil
        }

        // Remove the reduced dimensions
        let axisSet = Set(axes)
        return inputShape.enumerated().compactMap { idx, dim in
            axisSet.contains(idx) ? nil : dim
        }
    }

    /// Infers the output shape for reshape operations
    public static func reshapeShape(_ inputShape: [Int64], _ outputShape: [Int64]) -> [Int64]? {
        // Calculate total number of elements
        let inputElements = inputShape.reduce(1, *)
        let outputElements = outputShape.reduce(1, *)

        // If both are concrete, they must match
        if !inputShape.contains(-1) && !outputShape.contains(-1) {
            guard inputElements == outputElements else {
                return nil
            }
            return outputShape
        }

        // If one dimension is -1, infer it
        if let dynamicIdx = outputShape.firstIndex(of: -1) {
            let knownElements = outputShape.enumerated()
                .filter { $0.offset != dynamicIdx }
                .map { $0.element }
                .reduce(1, *)

            if inputElements % knownElements == 0 {
                var result = outputShape
                result[dynamicIdx] = inputElements / knownElements
                return result
            }
        }

        // Can't infer completely, return the requested shape
        return outputShape
    }

    /// Infers the output shape for transpose operations
    public static func transposeShape(_ inputShape: [Int64], permutation: [Int]) -> [Int64]? {
        guard permutation.count == inputShape.count else {
            return nil
        }

        // Check permutation is valid
        guard Set(permutation).count == permutation.count else {
            return nil  // Duplicate indices
        }
        guard permutation.allSatisfy({ $0 >= 0 && $0 < inputShape.count }) else {
            return nil  // Out of bounds
        }

        return permutation.map { inputShape[$0] }
    }
}
