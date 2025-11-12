// StableHLOMLOps.swift - ML Operations using StableHLO Dialect
// Copyright 2024 SwiftIR Project
//
// This file provides ML operation builders using the StableHLO dialect,
// mirroring the API of MLOps.swift but targeting XLA execution.

import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRDialects
import MLIRCoreWrapper

/// Machine Learning operations implemented using StableHLO dialect
/// These operations are designed for XLA compilation and GPU/TPU execution
public enum StableHLOMLOps {

    // MARK: - Attribute Helpers

    /// Creates a dense i64 array attribute from an array of integers
    private static func createDenseI64ArrayAttr(_ values: [Int64], context: MLIRContext) -> MlirAttribute {
        var mutableValues = values
        return mutableValues.withUnsafeMutableBufferPointer { ptr in
            mlirDenseI64ArrayAttrGetWrapper(context.handle, values.count, ptr.baseAddress)
        }
    }

    /// Creates a 2D padding attribute for StableHLO operations
    /// Format: [[low_h, high_h], [low_w, high_w]]
    private static func createPaddingAttr(_ padding: [(Int64, Int64)], context: MLIRContext) -> MlirAttribute {
        // Flatten padding to [low_h, high_h, low_w, high_w]
        let flatPadding = padding.flatMap { [$0.0, $0.1] }
        return createDenseI64ArrayAttr(flatPadding, context: context)
    }

    // MARK: - Matrix Operations

    /// Matrix multiplication using StableHLO dot operation
    ///
    /// Performs matrix multiplication: result = lhs @ rhs
    ///
    /// - Parameters:
    ///   - lhs: Left-hand side tensor [M, K]
    ///   - rhs: Right-hand side tensor [K, N]
    ///   - builder: The IR builder
    /// - Returns: Result tensor [M, N]
    public static func matmul(lhs: MLIRValue, rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Infer result shape from operands
        let lhsType = lhs.getType()
        let rhsType = rhs.getType()

        guard mlirTypeIsARankedTensorWrapper(lhsType),
              mlirTypeIsARankedTensorWrapper(rhsType) else {
            fatalError("matmul requires ranked tensor types")
        }

        let lhsRank = mlirShapedTypeGetRankWrapper(lhsType)
        let rhsRank = mlirShapedTypeGetRankWrapper(rhsType)

        guard lhsRank == 2 && rhsRank == 2 else {
            fatalError("matmul requires 2D tensors, got ranks [\(lhsRank), \(rhsRank)]")
        }

        let M = mlirShapedTypeGetDimSizeWrapper(lhsType, 0)
        let N = mlirShapedTypeGetDimSizeWrapper(rhsType, 1)
        let elementType = mlirShapedTypeGetElementTypeWrapper(lhsType)

        let resultType = mlirRankedTensorTypeGetWrapper(context.handle, 2, [M, N], elementType, mlirAttributeGetNullWrapper())

        let dotOp = Stablehlo.dot(lhs, rhs, resultType: resultType, location: loc, context: context)
        return dotOp.getResult(0)
    }

    // MARK: - Element-wise Operations

    /// Element-wise addition using StableHLO
    public static func add(_ lhs: MLIRValue, _ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let addOp = Stablehlo.add(lhs, rhs, location: loc, context: context)
        return addOp.getResult(0)
    }

    /// Element-wise multiplication using StableHLO
    public static func multiply(_ lhs: MLIRValue, _ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let mulOp = Stablehlo.multiply(lhs, rhs, location: loc, context: context)
        return mulOp.getResult(0)
    }

    /// Element-wise subtraction using StableHLO
    public static func subtract(_ lhs: MLIRValue, _ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let subOp = Stablehlo.subtract(lhs, rhs, location: loc, context: context)
        return subOp.getResult(0)
    }

    /// Element-wise division using StableHLO
    public static func divide(_ lhs: MLIRValue, _ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let divOp = Stablehlo.divide(lhs, rhs, location: loc, context: context)
        return divOp.getResult(0)
    }

    /// Element-wise maximum using StableHLO
    public static func maximum(_ lhs: MLIRValue, _ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let maxOp = Stablehlo.maximum(lhs, rhs, location: loc, context: context)
        return maxOp.getResult(0)
    }

    // MARK: - Normalization Operations

    /// Batch normalization using StableHLO
    ///
    /// Normalizes the input tensor across the batch dimension:
    /// output = scale * (input - mean) / sqrt(variance + epsilon) + offset
    ///
    /// - Parameters:
    ///   - input: Input tensor [N, ...] where N is the batch dimension
    ///   - scale: Scale parameter (gamma) - same shape as feature dimensions
    ///   - offset: Offset parameter (beta) - same shape as feature dimensions
    ///   - mean: Batch mean - same shape as feature dimensions
    ///   - variance: Batch variance - same shape as feature dimensions
    ///   - epsilon: Small constant for numerical stability (default: 1e-5)
    ///   - builder: The IR builder
    /// - Returns: Normalized output tensor
    public static func batchNorm(
        _ input: MLIRValue,
        scale: MLIRValue,
        offset: MLIRValue,
        mean: MLIRValue,
        variance: MLIRValue,
        epsilon: Double = 1e-5,
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Create epsilon constant
        let epsilonAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, epsilon)
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let epsilonOp = Stablehlo.constant(value: epsilonAttr, resultType: scalarType, location: loc, context: context)
        let epsilonValue = epsilonOp.getResult(0)

        // variance + epsilon
        let varianceEps = add(variance, epsilonValue, in: builder)

        // sqrt(variance + epsilon)
        let sqrtOp = OperationBuilder(name: "stablehlo.sqrt", location: loc, context: context)
            .addOperands([varianceEps])
            .addResults([variance.getType()])
            .build()
        let stddev = sqrtOp.getResult(0)

        // input - mean
        let centered = subtract(input, mean, in: builder)

        // (input - mean) / sqrt(variance + epsilon)
        let normalized = divide(centered, stddev, in: builder)

        // scale * normalized
        let scaled = multiply(normalized, scale, in: builder)

        // scaled + offset
        return add(scaled, offset, in: builder)
    }

    // MARK: - Activation Functions

    /// ReLU activation: max(0, x) using StableHLO
    public static func relu(_ input: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Create constant zero with same type as input
        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Create zero constant
        let zeroAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, 0.0)

        // Need to create a scalar tensor type for the constant
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let zeroOp = Stablehlo.constant(value: zeroAttr, resultType: scalarType, location: loc, context: context)
        let zero = zeroOp.getResult(0)

        // max(input, 0)
        return maximum(input, zero, in: builder)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)) using StableHLO logistic
    public static func sigmoid(_ input: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let sigmoidOp = Stablehlo.logistic(input, location: loc, context: context)
        return sigmoidOp.getResult(0)
    }

    /// Tanh activation using StableHLO
    public static func tanh(_ input: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)
        let tanhOp = Stablehlo.tanh(input, location: loc, context: context)
        return tanhOp.getResult(0)
    }

    // MARK: - Pooling Operations

    /// 2D Max Pooling using StableHLO
    ///
    /// Performs max pooling over spatial dimensions with NHWC layout
    ///
    /// - Parameters:
    ///   - input: Input tensor [N, H, W, C]
    ///   - windowSize: Size of the pooling window [window_h, window_w]
    ///   - strides: Stride for each spatial dimension [stride_h, stride_w]
    ///   - padding: Padding for each spatial dimension [(pad_h_low, pad_h_high), (pad_w_low, pad_w_high)]
    ///   - builder: The IR builder
    /// - Returns: Pooled output tensor [N, H_out, W_out, C]
    public static func maxPool2d(
        _ input: MLIRValue,
        windowSize: [Int64],
        strides: [Int64] = [1, 1],
        padding: [(Int64, Int64)] = [(0, 0), (0, 0)],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        let inputType = input.getType()
        guard mlirTypeIsARankedTensorWrapper(inputType) else {
            fatalError("maxPool2d requires ranked tensor type")
        }

        let N = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let H = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let W = mlirShapedTypeGetDimSizeWrapper(inputType, 2)
        let C = mlirShapedTypeGetDimSizeWrapper(inputType, 3)

        // Calculate output spatial dimensions
        let H_out = (H + padding[0].0 + padding[0].1 - windowSize[0]) / strides[0] + 1
        let W_out = (W + padding[1].0 + padding[1].1 - windowSize[1]) / strides[1] + 1

        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = mlirRankedTensorTypeGetWrapper(
            context.handle, 4,
            [N, H_out, W_out, C],
            elementType,
            mlirAttributeGetNullWrapper()
        )

        // Create negative infinity as init value for max reduction
        let negInfAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, -Double.infinity)
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let initOp = Stablehlo.constant(value: negInfAttr, resultType: scalarType, location: loc, context: context)
        let initValue = initOp.getResult(0)

        // Create attributes for reduce_window
        // Window dimensions: [1, window_h, window_w, 1] (NHWC format)
        let windowDimsAttr = createDenseI64ArrayAttr([1] + windowSize + [1], context: context)

        // Window strides: [1, stride_h, stride_w, 1]
        let windowStridesAttr = createDenseI64ArrayAttr([1] + strides + [1], context: context)

        // Padding: [[0,0], [pad_h_low, pad_h_high], [pad_w_low, pad_w_high], [0,0]]
        let flatPadding: [Int64] = [0, 0] + padding.flatMap { [$0.0, $0.1] } + [0, 0]
        let paddingAttr = createDenseI64ArrayAttr(flatPadding, context: context)

        let poolOp = OperationBuilder(name: "stablehlo.reduce_window", location: loc, context: context)
            .addOperands([input, initValue])
            .addResults([outputType])
            .addAttributes([
                ("window_dimensions", windowDimsAttr),
                ("window_strides", windowStridesAttr),
                ("padding", paddingAttr)
            ])
            .build()

        return poolOp.getResult(0)
    }

    /// 2D Average Pooling using StableHLO
    ///
    /// Performs average pooling over spatial dimensions with NHWC layout
    ///
    /// - Parameters:
    ///   - input: Input tensor [N, H, W, C]
    ///   - windowSize: Size of the pooling window [window_h, window_w]
    ///   - strides: Stride for each spatial dimension [stride_h, stride_w]
    ///   - padding: Padding for each spatial dimension [(pad_h_low, pad_h_high), (pad_w_low, pad_w_high)]
    ///   - builder: The IR builder
    /// - Returns: Pooled output tensor [N, H_out, W_out, C]
    public static func avgPool2d(
        _ input: MLIRValue,
        windowSize: [Int64],
        strides: [Int64] = [1, 1],
        padding: [(Int64, Int64)] = [(0, 0), (0, 0)],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        let inputType = input.getType()
        guard mlirTypeIsARankedTensorWrapper(inputType) else {
            fatalError("avgPool2d requires ranked tensor type")
        }

        let N = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let H = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let W = mlirShapedTypeGetDimSizeWrapper(inputType, 2)
        let C = mlirShapedTypeGetDimSizeWrapper(inputType, 3)

        // Calculate output spatial dimensions
        let H_out = (H + padding[0].0 + padding[0].1 - windowSize[0]) / strides[0] + 1
        let W_out = (W + padding[1].0 + padding[1].1 - windowSize[1]) / strides[1] + 1

        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = mlirRankedTensorTypeGetWrapper(
            context.handle, 4,
            [N, H_out, W_out, C],
            elementType,
            mlirAttributeGetNullWrapper()
        )

        // Create zero as init value for sum reduction
        let zeroAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, 0.0)
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let initOp = Stablehlo.constant(value: zeroAttr, resultType: scalarType, location: loc, context: context)
        let initValue = initOp.getResult(0)

        // Create attributes for reduce_window
        // Window dimensions: [1, window_h, window_w, 1] (NHWC format)
        let windowDimsAttr = createDenseI64ArrayAttr([1] + windowSize + [1], context: context)

        // Window strides: [1, stride_h, stride_w, 1]
        let windowStridesAttr = createDenseI64ArrayAttr([1] + strides + [1], context: context)

        // Padding: [[0,0], [pad_h_low, pad_h_high], [pad_w_low, pad_w_high], [0,0]]
        let flatPadding: [Int64] = [0, 0] + padding.flatMap { [$0.0, $0.1] } + [0, 0]
        let paddingAttr = createDenseI64ArrayAttr(flatPadding, context: context)

        // Sum reduction for average pooling
        let sumOp = OperationBuilder(name: "stablehlo.reduce_window", location: loc, context: context)
            .addOperands([input, initValue])
            .addResults([outputType])
            .addAttributes([
                ("window_dimensions", windowDimsAttr),
                ("window_strides", windowStridesAttr),
                ("padding", paddingAttr)
            ])
            .build()

        // For average pooling, divide by window size
        let windowArea = windowSize[0] * windowSize[1]
        let divisorAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, Double(windowArea))
        let divisorOp = Stablehlo.constant(value: divisorAttr, resultType: scalarType, location: loc, context: context)
        let divisor = divisorOp.getResult(0)

        return divide(sumOp.getResult(0), divisor, in: builder)
    }

    // MARK: - Convolution Operations

    /// 2D Convolution using StableHLO
    ///
    /// Performs 2D convolution with NHWC (batch, height, width, channels) layout
    ///
    /// - Parameters:
    ///   - input: Input tensor [N, H, W, C_in]
    ///   - kernel: Convolution kernel [KH, KW, C_in, C_out]
    ///   - strides: Stride for each spatial dimension [stride_h, stride_w]
    ///   - padding: Padding for each spatial dimension [(pad_h_low, pad_h_high), (pad_w_low, pad_w_high)]
    ///   - builder: The IR builder
    /// - Returns: Output tensor [N, H_out, W_out, C_out]
    public static func conv2d(
        input: MLIRValue,
        kernel: MLIRValue,
        strides: [Int64] = [1, 1],
        padding: [(Int64, Int64)] = [(0, 0), (0, 0)],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Infer output shape
        let inputType = input.getType()
        let kernelType = kernel.getType()

        guard mlirTypeIsARankedTensorWrapper(inputType),
              mlirTypeIsARankedTensorWrapper(kernelType) else {
            fatalError("conv2d requires ranked tensor types")
        }

        let N = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let H = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let W = mlirShapedTypeGetDimSizeWrapper(inputType, 2)

        let KH = mlirShapedTypeGetDimSizeWrapper(kernelType, 0)
        let KW = mlirShapedTypeGetDimSizeWrapper(kernelType, 1)
        let C_out = mlirShapedTypeGetDimSizeWrapper(kernelType, 3)

        // Calculate output spatial dimensions
        let H_out = (H + padding[0].0 + padding[0].1 - KH) / strides[0] + 1
        let W_out = (W + padding[1].0 + padding[1].1 - KW) / strides[1] + 1

        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = mlirRankedTensorTypeGetWrapper(
            context.handle, 4,
            [N, H_out, W_out, C_out],
            elementType,
            mlirAttributeGetNullWrapper()
        )

        // Create attributes for convolution
        // Window strides: [stride_h, stride_w]
        let windowStridesAttr = createDenseI64ArrayAttr(strides, context: context)

        // Padding: [[pad_h_low, pad_h_high], [pad_w_low, pad_w_high]]
        let paddingAttr = createPaddingAttr(padding, context: context)

        // Build convolution with attributes
        let convOp = OperationBuilder(name: "stablehlo.convolution", location: loc, context: context)
            .addOperands([input, kernel])
            .addResults([outputType])
            .addAttributes([
                ("window_strides", windowStridesAttr),
                ("padding", paddingAttr)
            ])
            .build()

        return convOp.getResult(0)
    }

    // MARK: - Reduction Operations

    /// Reduce sum along specified dimensions
    public static func reduceSum(
        _ input: MLIRValue,
        dimensions: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Create zero initial value
        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let zeroAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, 0.0)

        // Create scalar tensor for init value
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let zeroOp = Stablehlo.constant(value: zeroAttr, resultType: scalarType, location: loc, context: context)
        let zeroValue = zeroOp.getResult(0)

        // Calculate result shape (remove reduced dimensions)
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        var resultShape: [Int64] = []
        for dim in 0..<inputRank {
            if !dimensions.contains(Int64(dim)) {
                resultShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            }
        }

        let resultType = mlirRankedTensorTypeGetWrapper(
            context.handle,
            Int(resultShape.count),
            resultShape,
            elementType,
            mlirAttributeGetNullWrapper()
        )

        let reduceOp = Stablehlo.reduce(
            input,
            initValue: zeroValue,
            dimensions: dimensions,
            resultType: resultType,
            location: loc,
            context: context
        )

        return reduceOp.getResult(0)
    }

    /// Reduce mean along specified dimensions
    public static func reduceMean(
        _ input: MLIRValue,
        dimensions: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // First compute sum
        let sum = reduceSum(input, dimensions: dimensions, in: builder)

        // Calculate number of elements reduced
        var numElements: Int64 = 1
        for dim in dimensions {
            numElements *= mlirShapedTypeGetDimSizeWrapper(inputType, Int(dim))
        }

        // Create divisor constant
        let divisorAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, Double(numElements))
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let loc = MLIRLocation.unknown(in: context)
        let divisorOp = Stablehlo.constant(value: divisorAttr, resultType: scalarType, location: loc, context: context)
        let divisor = divisorOp.getResult(0)

        // Divide sum by number of elements
        return divide(sum, divisor, in: builder)
    }

    /// Reduce max along specified dimensions
    public static func reduceMax(
        _ input: MLIRValue,
        dimensions: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Create negative infinity initial value
        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let negInfAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, -Double.infinity)

        // Create scalar tensor for init value
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let initOp = Stablehlo.constant(value: negInfAttr, resultType: scalarType, location: loc, context: context)
        let initValue = initOp.getResult(0)

        // Calculate result shape (remove reduced dimensions)
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        var resultShape: [Int64] = []
        for dim in 0..<inputRank {
            if !dimensions.contains(Int64(dim)) {
                resultShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            }
        }

        let resultType = mlirRankedTensorTypeGetWrapper(
            context.handle,
            Int(resultShape.count),
            resultShape,
            elementType,
            mlirAttributeGetNullWrapper()
        )

        let reduceOp = Stablehlo.reduce(
            input,
            initValue: initValue,
            dimensions: dimensions,
            resultType: resultType,
            location: loc,
            context: context
        )

        return reduceOp.getResult(0)
    }

    /// Reduce min along specified dimensions
    public static func reduceMin(
        _ input: MLIRValue,
        dimensions: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        // Create positive infinity initial value
        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let posInfAttr = mlirFloatAttrDoubleGetWrapper(context.handle, elementType, Double.infinity)

        // Create scalar tensor for init value
        let scalarType = mlirRankedTensorTypeGetWrapper(context.handle, 0, [], elementType, mlirAttributeGetNullWrapper())
        let initOp = Stablehlo.constant(value: posInfAttr, resultType: scalarType, location: loc, context: context)
        let initValue = initOp.getResult(0)

        // Calculate result shape (remove reduced dimensions)
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        var resultShape: [Int64] = []
        for dim in 0..<inputRank {
            if !dimensions.contains(Int64(dim)) {
                resultShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            }
        }

        let resultType = mlirRankedTensorTypeGetWrapper(
            context.handle,
            Int(resultShape.count),
            resultShape,
            elementType,
            mlirAttributeGetNullWrapper()
        )

        let reduceOp = Stablehlo.reduce(
            input,
            initValue: initValue,
            dimensions: dimensions,
            resultType: resultType,
            location: loc,
            context: context
        )

        return reduceOp.getResult(0)
    }

    // MARK: - Shape Operations

    /// Broadcast a tensor to a new shape
    ///
    /// Note: Currently this is a simplified version that broadcasts to a target shape.
    /// The full broadcastInDim operation requires specifying which dimensions of the
    /// output correspond to which dimensions of the input.
    public static func broadcast(
        _ input: MLIRValue,
        to shape: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)

        let resultType = mlirRankedTensorTypeGetWrapper(
            context.handle,
            Int(shape.count),
            shape,
            elementType,
            mlirAttributeGetNullWrapper()
        )

        // For now, use broadcastInDim without broadcast_dimensions attribute
        // This will need to be enhanced when we add the attribute support
        let broadcastOp = Stablehlo.broadcastInDim(
            input,
            resultType: resultType,
            location: loc,
            context: context
        )

        return broadcastOp.getResult(0)
    }

    /// Reshape a tensor to a new shape
    public static func reshape(
        _ input: MLIRValue,
        to shape: [Int64],
        in builder: IRBuilder
    ) -> MLIRValue {
        let context = builder.context
        let loc = MLIRLocation.unknown(in: context)

        let inputType = input.getType()
        let elementType = mlirShapedTypeGetElementTypeWrapper(inputType)

        let resultType = mlirRankedTensorTypeGetWrapper(
            context.handle,
            Int(shape.count),
            shape,
            elementType,
            mlirAttributeGetNullWrapper()
        )

        let reshapeOp = Stablehlo.reshape(
            input,
            resultType: resultType,
            location: loc,
            context: context
        )

        return reshapeOp.getResult(0)
    }
}
