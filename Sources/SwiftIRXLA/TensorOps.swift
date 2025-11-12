//===-- TensorOps.swift - MLIR Tensor Operations ---------*- Swift -*-===//
//
// SwiftIR - Phase 7: Tensor Operations for ML
// Provides tensor creation and manipulation operations
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import MLIRCoreWrapper

/// Namespace for tensor operations
public enum TensorOps {

    /// Creates an empty tensor with the specified shape and element type
    ///
    /// Generates: `tensor.empty() : tensor<shape x elemType>`
    ///
    /// Example:
    /// ```swift
    /// let tensor = TensorOps.empty(shape: [2, 3], elementType: FloatType.f32())
    /// // Generates: tensor.empty() : tensor<2x3xf32>
    /// ```
    public static func empty<T: MLIRType>(
        shape: [Int64],
        elementType: T,
        in builder: IRBuilder
    ) -> MLIRValue {
        let tensorType = RankedTensorType(shape: shape, elementType: elementType, context: builder.context)

        let op = OperationBuilder(
            name: "tensor.empty",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([tensorType.typeHandle])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Creates a tensor from individual elements
    ///
    /// Generates: `tensor.from_elements %v1, %v2, ... : tensor<shape x elemType>`
    ///
    /// Example:
    /// ```swift
    /// let elements = [v1, v2, v3, v4]
    /// let tensor = TensorOps.fromElements(elements, shape: [2, 2], in: builder)
    /// // Generates: tensor.from_elements %v1, %v2, %v3, %v4 : tensor<2x2xf32>
    /// ```
    public static func fromElements<T: MLIRType>(
        _ elements: [MLIRValue],
        shape: [Int64],
        elementType: T,
        in builder: IRBuilder
    ) -> MLIRValue {
        let tensorType = RankedTensorType(shape: shape, elementType: elementType, context: builder.context)

        let op = OperationBuilder(
            name: "tensor.from_elements",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([tensorType.typeHandle])
        .addOperands(elements)
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Extracts an element from a tensor at the given indices
    ///
    /// Generates: `tensor.extract %tensor[%i0, %i1, ...] : tensor<shape x elemType>`
    ///
    /// Example:
    /// ```swift
    /// let element = TensorOps.extract(from: tensor, at: [i, j], in: builder)
    /// // Generates: tensor.extract %tensor[%i, %j] : tensor<2x3xf32>
    /// ```
    public static func extract(
        from tensor: MLIRValue,
        at indices: [MLIRValue],
        in builder: IRBuilder
    ) -> MLIRValue {
        // Get the tensor's element type
        let tensorTypeHandle = tensor.getType()
        let elementTypeHandle = mlirShapedTypeGetElementTypeWrapper(tensorTypeHandle)

        let op = OperationBuilder(
            name: "tensor.extract",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elementTypeHandle])
        .addOperands([tensor] + indices)
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Inserts an element into a tensor at the given indices
    ///
    /// Generates: `tensor.insert %scalar into %tensor[%i0, %i1, ...] : tensor<shape x elemType>`
    ///
    /// Example:
    /// ```swift
    /// let newTensor = TensorOps.insert(value: scalar, into: tensor, at: [i, j], in: builder)
    /// // Generates: tensor.insert %scalar into %tensor[%i, %j] : tensor<2x3xf32>
    /// ```
    public static func insert(
        value: MLIRValue,
        into tensor: MLIRValue,
        at indices: [MLIRValue],
        in builder: IRBuilder
    ) -> MLIRValue {
        let tensorTypeHandle = tensor.getType()

        let op = OperationBuilder(
            name: "tensor.insert",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([tensorTypeHandle])
        .addOperands([value, tensor] + indices)
        .build()

        builder.insert(op)
        return op.getResult(0)
    }
}
