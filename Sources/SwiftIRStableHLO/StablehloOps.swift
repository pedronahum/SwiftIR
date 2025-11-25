// StablehloOps.swift - Swift wrappers for StableHLO operations
// Copyright 2024 SwiftIR Project
//
// This file provides high-level Swift interfaces for StableHLO operations.
// StableHLO operations are the portable ML operations used by JAX, PyTorch, and TensorFlow.

import SwiftIRCore
import SwiftIRTypes
import MLIRCoreWrapper

// MARK: - StableHLO Dialect

/// The StableHLO dialect namespace
public enum Stablehlo {
    /// Dialect name
    public static let dialectName = "stablehlo"

    /// Ensures the StableHLO dialect is loaded in the context
    public static func load(in context: MLIRContext) {
        _ = context.loadDialect("stablehlo")
    }
}

// MARK: - Element-wise Binary Operations

extension Stablehlo {

    /// Creates a StableHLO add operation: result = lhs + rhs
    ///
    /// Performs element-wise addition of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func add(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.add", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO multiply operation: result = lhs * rhs
    ///
    /// Performs element-wise multiplication of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func multiply(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.multiply", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO subtract operation: result = lhs - rhs
    ///
    /// Performs element-wise subtraction of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func subtract(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.subtract", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO divide operation: result = lhs / rhs
    ///
    /// Performs element-wise division of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand (dividend)
    ///   - rhs: Right-hand side operand (divisor)
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func divide(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.divide", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO maximum operation: result = max(lhs, rhs)
    ///
    /// Performs element-wise maximum of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func maximum(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.maximum", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO minimum operation: result = min(lhs, rhs)
    ///
    /// Performs element-wise minimum of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func minimum(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "stablehlo.minimum", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Matrix Operations

extension Stablehlo {

    /// Creates a StableHLO dot_general operation for matrix multiplication
    ///
    /// This is the core operation for matrix multiplication, supporting batched matmul,
    /// einsum-like contractions, and more.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side tensor
    ///   - rhs: Right-hand side tensor
    ///   - resultType: Result tensor type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func dotGeneral(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // For now, simple 2D matrix multiplication
        // In full implementation, would add dimension_numbers attribute
        return OperationBuilder(name: "stablehlo.dot_general", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO dot operation (simple matrix multiply)
    ///
    /// Simplified version of dot_general for standard matrix multiplication.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side tensor [M, K]
    ///   - rhs: Right-hand side tensor [K, N]
    ///   - resultType: Result tensor type [M, N]
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func dot(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "stablehlo.dot", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Activation Functions

extension Stablehlo {

    /// Creates a StableHLO exponential operation: result = exp(operand)
    ///
    /// - Parameters:
    ///   - operand: Input tensor
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func exponential(
        _ operand: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = operand.getType()
        return OperationBuilder(name: "stablehlo.exponential", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO tanh operation: result = tanh(operand)
    ///
    /// - Parameters:
    ///   - operand: Input tensor
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func tanh(
        _ operand: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = operand.getType()
        return OperationBuilder(name: "stablehlo.tanh", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO logistic (sigmoid) operation: result = 1 / (1 + exp(-x))
    ///
    /// - Parameters:
    ///   - operand: Input tensor
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func logistic(
        _ operand: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = operand.getType()
        return OperationBuilder(name: "stablehlo.logistic", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Reduction Operations

extension Stablehlo {

    /// Creates a StableHLO reduce operation
    ///
    /// Reduces a tensor along specified dimensions using a computation function.
    ///
    /// - Parameters:
    ///   - operand: Input tensor to reduce
    ///   - initValue: Initial value for reduction
    ///   - dimensions: Array of dimensions to reduce along
    ///   - resultType: Result tensor type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func reduce(
        _ operand: MLIRValue,
        initValue: MLIRValue,
        dimensions: [Int64],
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // Note: Full implementation would add body region with reduction computation
        return OperationBuilder(name: "stablehlo.reduce", location: location, context: context)
            .addOperands([operand, initValue])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Shape Operations

extension Stablehlo {

    /// Creates a StableHLO broadcast_in_dim operation
    ///
    /// Broadcasts a tensor to a higher-rank shape by specifying dimension mapping.
    ///
    /// - Parameters:
    ///   - operand: Input tensor to broadcast
    ///   - resultType: Target broadcast type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func broadcastInDim(
        _ operand: MLIRValue,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "stablehlo.broadcast_in_dim", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO reshape operation
    ///
    /// Reshapes a tensor to a new shape with the same number of elements.
    ///
    /// - Parameters:
    ///   - operand: Input tensor
    ///   - resultType: Target shape type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func reshape(
        _ operand: MLIRValue,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "stablehlo.reshape", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }

    /// Creates a StableHLO transpose operation
    ///
    /// Permutes the dimensions of a tensor.
    ///
    /// - Parameters:
    ///   - operand: Input tensor
    ///   - permutation: Dimension permutation
    ///   - resultType: Transposed type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func transpose(
        _ operand: MLIRValue,
        permutation: [Int64],
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // Note: Full implementation would add permutation attribute
        return OperationBuilder(name: "stablehlo.transpose", location: location, context: context)
            .addOperands([operand])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Convolution Operations

extension Stablehlo {

    /// Creates a StableHLO convolution operation
    ///
    /// Performs n-dimensional convolution.
    ///
    /// - Parameters:
    ///   - input: Input tensor [batch, in_height, in_width, in_channels]
    ///   - kernel: Kernel tensor [kernel_height, kernel_width, in_channels, out_channels]
    ///   - resultType: Output type
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func convolution(
        _ input: MLIRValue,
        kernel: MLIRValue,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // Note: Full implementation would add window attributes (stride, padding, dilation)
        return OperationBuilder(name: "stablehlo.convolution", location: location, context: context)
            .addOperands([input, kernel])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Constant Operations

extension Stablehlo {

    /// Creates a StableHLO constant operation
    ///
    /// Creates a constant tensor with the given value.
    ///
    /// - Parameters:
    ///   - value: Attribute containing the constant value
    ///   - resultType: Type of the constant tensor
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation
    public static func constant(
        value: MlirAttribute,
        resultType: MlirType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "stablehlo.constant", location: location, context: context)
            .addAttributes([("value", value)])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Control Flow Operations

extension Stablehlo {

    /// Creates a StableHLO while operation
    ///
    /// Executes a while loop with a condition region and body region.
    /// This is the key operation for efficient iterative algorithms with XLA fusion.
    ///
    /// The `stablehlo.while` operation has two regions:
    /// 1. **Condition region**: Takes loop state, returns i1 (continue or not)
    /// 2. **Body region**: Takes loop state, returns updated loop state
    ///
    /// Example MLIR:
    /// ```mlir
    /// %result = stablehlo.while(%arg0 = %init) : tensor<f32> {
    ///   // Condition region
    ///   ^bb0(%cond_arg: tensor<f32>):
    ///     %cond = stablehlo.compare LT, %cond_arg, %max : tensor<i1>
    ///     stablehlo.return %cond : tensor<i1>
    /// } do {
    ///   // Body region
    ///   ^bb0(%body_arg: tensor<f32>):
    ///     %new_val = stablehlo.add %body_arg, %step : tensor<f32>
    ///     stablehlo.return %new_val : tensor<f32>
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - initialValues: Initial loop carry values
    ///   - conditionRegion: Region that computes loop condition (returns i1)
    ///   - bodyRegion: Region that computes next iteration state
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The while operation
    public static func whileOp(
        initialValues: [MLIRValue],
        conditionRegion: MLIRRegion,
        bodyRegion: MLIRRegion,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultTypes = initialValues.map { $0.getType() }

        return OperationBuilder(name: "stablehlo.while", location: location, context: context)
            .addOperands(initialValues)
            .addResults(resultTypes)
            .addRegions([conditionRegion, bodyRegion])
            .build()
    }

    /// Creates a StableHLO return operation (for regions)
    ///
    /// Returns values from a region (used in while loop condition/body).
    ///
    /// - Parameters:
    ///   - operands: Values to return
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The return operation
    public static func `return`(
        _ operands: [MLIRValue],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "stablehlo.return", location: location, context: context)
            .addOperands(operands)
            .build()
    }

    /// Comparison directions for stablehlo.compare
    public enum ComparisonDirection: String {
        case eq = "EQ"   // Equal
        case ne = "NE"   // Not equal
        case lt = "LT"   // Less than
        case le = "LE"   // Less than or equal
        case gt = "GT"   // Greater than
        case ge = "GE"   // Greater than or equal
    }

    /// Creates a StableHLO compare operation
    ///
    /// Performs element-wise comparison of two tensors.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side operand
    ///   - rhs: Right-hand side operand
    ///   - direction: Comparison direction (EQ, NE, LT, LE, GT, GE)
    ///   - location: Source location for debugging
    ///   - context: The MLIR context
    /// - Returns: The operation (returns tensor of i1)
    public static func compare(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        direction: ComparisonDirection,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // Result type is i1 (boolean) tensor with same shape as inputs
        // For now, we'll create a simple scalar i1 tensor type
        // In full implementation, would extract shape from lhsType and create proper result type
        let i1Type = IntegerType.i1(context: context)
        let resultType = RankedTensorType(shape: [], elementType: i1Type, context: context)

        // Create comparison_direction attribute
        let directionAttr = direction.rawValue.withCString { ptr in
            let strRef = mlirStringRefCreateWrapper(ptr, direction.rawValue.utf8.count)
            return mlirStringAttrGetWrapper(context.handle, strRef)
        }

        return OperationBuilder(name: "stablehlo.compare", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType.typeHandle])
            .addAttributes([("comparison_direction", directionAttr)])
            .build()
    }
}
