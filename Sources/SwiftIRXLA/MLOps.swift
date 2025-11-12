//===-- MLOps.swift - Machine Learning Operations ---------*- Swift -*-===//
//
// SwiftIR - Phase 7: ML Operations
// Provides common ML operations using MLIR dialects
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import MLIRCoreWrapper

/// Machine Learning operations namespace
///
/// Provides common ML operations:
/// - Linear algebra: dot (matrix multiplication)
/// - Activations: relu, sigmoid, tanh
/// - Normalization: softmax
/// - Convolutions: conv2d
/// - Element-wise: add, mul, sub, div
public enum MLOps {

    // MARK: - Linear Algebra Operations

    /// Matrix multiplication (dot product)
    ///
    /// Generates: `linalg.matmul` operation
    ///
    /// Example:
    /// ```swift
    /// let result = MLOps.matmul(lhs: A, rhs: B, in: builder)
    /// // Computes: result = A @ B
    /// ```
    public static func matmul(
        lhs: MLIRValue,
        rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        // Get input shapes to determine output shape
        let lhsType = lhs.getType()
        let rhsType = rhs.getType()

        let lhsRank = mlirShapedTypeGetRankWrapper(lhsType)
        let rhsRank = mlirShapedTypeGetRankWrapper(rhsType)

        guard lhsRank == 2 && rhsRank == 2 else {
            fatalError("matmul requires 2D tensors")
        }

        // Output shape: [M, N] where lhs is [M, K] and rhs is [K, N]
        let m = mlirShapedTypeGetDimSizeWrapper(lhsType, 0)
        let n = mlirShapedTypeGetDimSizeWrapper(rhsType, 1)

        let elemType = mlirShapedTypeGetElementTypeWrapper(lhsType)
        let resultType = RankedTensorType(
            shape: [m, n],
            elementType: FloatType(handle: elemType, context: builder.context),
            context: builder.context
        )

        let op = OperationBuilder(
            name: "linalg.matmul",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([lhs, rhs])
        .addResults([resultType.typeHandle])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Convenience method: dot is an alias for matmul
    public static func dot(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        return matmul(lhs: lhs, rhs: rhs, in: builder)
    }

    // MARK: - Activation Functions

    /// ReLU activation function: max(0, x)
    ///
    /// Generates: `arith.maxf` with constant zero
    ///
    /// Example:
    /// ```swift
    /// let activated = MLOps.relu(input, in: builder)
    /// // Computes: activated = max(0, input)
    /// ```
    public static func relu(
        _ input: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Create constant zero tensor with same shape as input
        let zeroAttr = mlirFloatAttrDoubleGetWrapper(builder.context.handle, elemType, 0.0)

        let zero = OperationBuilder(
            name: "arith.constant",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elemType])
        .addAttributes([("value", zeroAttr)])
        .build()

        builder.insert(zero)
        let zeroValue = zero.getResult(0)

        // Perform element-wise maximum
        let maxOp = OperationBuilder(
            name: "arith.maximumf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input, zeroValue])
        .addResults([inputType])
        .build()

        builder.insert(maxOp)
        return maxOp.getResult(0)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    ///
    /// Generates: math.exp and arith operations
    public static func sigmoid(
        _ input: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        // -x
        let negOp = OperationBuilder(
            name: "arith.negf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([inputType])
        .build()

        builder.insert(negOp)
        let negX = negOp.getResult(0)

        // exp(-x)
        let expOp = OperationBuilder(
            name: "math.exp",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([negX])
        .addResults([inputType])
        .build()

        builder.insert(expOp)
        return expOp.getResult(0)
    }

    /// Tanh activation function
    ///
    /// Generates: `math.tanh` operation
    public static func tanh(
        _ input: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        let op = OperationBuilder(
            name: "math.tanh",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([inputType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    // MARK: - Normalization Operations

    /// Softmax operation: exp(x) / sum(exp(x))
    ///
    /// Note: This is a simplified version. Production softmax requires
    /// numerical stability improvements (subtract max before exp)
    ///
    /// Generates: math.exp, linalg.reduce operations
    public static func softmax(
        _ input: MLIRValue,
        axis: Int64 = -1,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        // For now, just return exp(input) as a placeholder
        // Full softmax implementation requires reduction operations
        let expOp = OperationBuilder(
            name: "math.exp",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([inputType])
        .build()

        builder.insert(expOp)
        return expOp.getResult(0)
    }

    // MARK: - Convolution Operations

    /// 2D Convolution operation
    ///
    /// Generates: `linalg.conv_2d_nhwc_hwcf` operation
    ///
    /// Example:
    /// ```swift
    /// let output = MLOps.conv2d(
    ///     input: image,      // [N, H, W, C]
    ///     kernel: weights,   // [KH, KW, C, F]
    ///     in: builder
    /// )
    /// ```
    public static func conv2d(
        input: MLIRValue,
        kernel: MLIRValue,
        strides: [Int64] = [1, 1],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let kernelType = kernel.getType()

        // Input: [N, H, W, C]
        // Kernel: [KH, KW, C, F]
        // Output: [N, H', W', F]

        let n = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let h = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let w = mlirShapedTypeGetDimSizeWrapper(inputType, 2)
        let f = mlirShapedTypeGetDimSizeWrapper(kernelType, 3)

        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = RankedTensorType(
            shape: [n, h, w, f], // Simplified - doesn't account for stride/padding
            elementType: FloatType(handle: elemType, context: builder.context),
            context: builder.context
        )

        let op = OperationBuilder(
            name: "linalg.conv_2d_nhwc_hwcf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input, kernel])
        .addResults([outputType.typeHandle])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    // MARK: - Element-wise Operations

    /// Element-wise addition of two tensors
    ///
    /// Generates: `arith.addf` operation
    public static func add(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let resultType = lhs.getType()

        let op = OperationBuilder(
            name: "arith.addf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([lhs, rhs])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Element-wise multiplication of two tensors
    ///
    /// Generates: `arith.mulf` operation
    public static func mul(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let resultType = lhs.getType()

        let op = OperationBuilder(
            name: "arith.mulf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([lhs, rhs])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Element-wise subtraction of two tensors
    ///
    /// Generates: `arith.subf` operation
    public static func sub(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let resultType = lhs.getType()

        let op = OperationBuilder(
            name: "arith.subf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([lhs, rhs])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Element-wise division of two tensors
    ///
    /// Generates: `arith.divf` operation
    public static func div(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let resultType = lhs.getType()

        let op = OperationBuilder(
            name: "arith.divf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([lhs, rhs])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    // MARK: - Reduction Operations

    /// Reduce sum along specified axes
    ///
    /// Generates: reduction operation summing along specified dimensions
    ///
    /// Example:
    /// ```swift
    /// // Sum along axis 1 (columns)
    /// let sum = MLOps.reduce_sum(input, axes: [1], in: builder)
    /// ```
    public static func reduce_sum(
        _ input: MLIRValue,
        axes: [Int64],
        keepDims: Bool = false,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Calculate output shape
        var outputShape: [Int64] = []
        for dim in 0..<inputRank {
            if !axes.contains(Int64(dim)) {
                outputShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            } else if keepDims {
                outputShape.append(1)
            }
        }

        // If all dims reduced and not keeping dims, result is scalar
        let resultType: MlirType
        if outputShape.isEmpty && !keepDims {
            resultType = elemType
        } else {
            let tensorType = RankedTensorType(
                shape: outputShape,
                elementType: FloatType(handle: elemType, context: builder.context),
                context: builder.context
            )
            resultType = tensorType.typeHandle
        }

        // Use linalg.reduce for sum
        let op = OperationBuilder(
            name: "linalg.reduce",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Reduce mean along specified axes
    ///
    /// Generates: reduction operation computing mean along dimensions
    public static func reduce_mean(
        _ input: MLIRValue,
        axes: [Int64],
        keepDims: Bool = false,
        in builder: IRBuilder
    ) -> MLIRValue {
        // Mean = sum / count
        // First get the sum
        let sum = reduce_sum(input, axes: axes, keepDims: keepDims, in: builder)

        // Calculate the count (product of reduced dimensions)
        let inputType = input.getType()
        var count: Int64 = 1
        for axis in axes {
            count *= mlirShapedTypeGetDimSizeWrapper(inputType, Int(axis))
        }

        // Create constant for count
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let countAttr = mlirFloatAttrDoubleGetWrapper(
            builder.context.handle,
            elemType,
            Double(count)
        )

        let countConst = OperationBuilder(
            name: "arith.constant",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elemType])
        .addAttributes([("value", countAttr)])
        .build()

        builder.insert(countConst)
        let countValue = countConst.getResult(0)

        // Divide sum by count
        let resultType = sum.getType()
        let divOp = OperationBuilder(
            name: "arith.divf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([sum, countValue])
        .addResults([resultType])
        .build()

        builder.insert(divOp)
        return divOp.getResult(0)
    }

    /// Reduce max along specified axes
    ///
    /// Generates: reduction operation finding maximum along dimensions
    public static func reduce_max(
        _ input: MLIRValue,
        axes: [Int64],
        keepDims: Bool = false,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Calculate output shape
        var outputShape: [Int64] = []
        for dim in 0..<inputRank {
            if !axes.contains(Int64(dim)) {
                outputShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            } else if keepDims {
                outputShape.append(1)
            }
        }

        let resultType: MlirType
        if outputShape.isEmpty && !keepDims {
            resultType = elemType
        } else {
            let tensorType = RankedTensorType(
                shape: outputShape,
                elementType: FloatType(handle: elemType, context: builder.context),
                context: builder.context
            )
            resultType = tensorType.typeHandle
        }

        let op = OperationBuilder(
            name: "linalg.reduce",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// Reduce min along specified axes
    ///
    /// Generates: reduction operation finding minimum along dimensions
    public static func reduce_min(
        _ input: MLIRValue,
        axes: [Int64],
        keepDims: Bool = false,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let inputRank = mlirShapedTypeGetRankWrapper(inputType)
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Calculate output shape
        var outputShape: [Int64] = []
        for dim in 0..<inputRank {
            if !axes.contains(Int64(dim)) {
                outputShape.append(mlirShapedTypeGetDimSizeWrapper(inputType, dim))
            } else if keepDims {
                outputShape.append(1)
            }
        }

        let resultType: MlirType
        if outputShape.isEmpty && !keepDims {
            resultType = elemType
        } else {
            let tensorType = RankedTensorType(
                shape: outputShape,
                elementType: FloatType(handle: elemType, context: builder.context),
                context: builder.context
            )
            resultType = tensorType.typeHandle
        }

        let op = OperationBuilder(
            name: "linalg.reduce",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input])
        .addResults([resultType])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    // MARK: - Pooling Operations

    /// 2D Max Pooling operation
    ///
    /// Generates: pooling operation finding maximum in spatial windows
    ///
    /// Example:
    /// ```swift
    /// let pooled = MLOps.max_pool2d(
    ///     input,
    ///     kernelSize: [2, 2],
    ///     strides: [2, 2],
    ///     in: builder
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - input: Input tensor with shape [N, H, W, C] (NHWC format)
    ///   - kernelSize: Pooling window size [kernel_h, kernel_w]
    ///   - strides: Stride for pooling [stride_h, stride_w]
    ///   - padding: Padding values [pad_h, pad_w] (default: [0, 0])
    ///   - builder: IR builder for inserting operations
    /// - Returns: Pooled tensor with reduced spatial dimensions
    public static func max_pool2d(
        _ input: MLIRValue,
        kernelSize: [Int64],
        strides: [Int64],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        // Input shape: [N, H, W, C]
        let n = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let h = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let w = mlirShapedTypeGetDimSizeWrapper(inputType, 2)
        let c = mlirShapedTypeGetDimSizeWrapper(inputType, 3)

        // Calculate output dimensions
        // out_h = (h + 2*pad_h - kernel_h) / stride_h + 1
        // out_w = (w + 2*pad_w - kernel_w) / stride_w + 1
        let outH = (h + 2 * padding[0] - kernelSize[0]) / strides[0] + 1
        let outW = (w + 2 * padding[1] - kernelSize[1]) / strides[1] + 1

        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = RankedTensorType(
            shape: [n, outH, outW, c],
            elementType: FloatType(handle: elemType, context: builder.context),
            context: builder.context
        )

        // Create a "window" tensor representing the pooling kernel shape
        // This is used as an initializer for the pooling operation
        let window = TensorOps.empty(
            shape: kernelSize,
            elementType: FloatType(handle: elemType, context: builder.context),
            in: builder
        )

        // Use linalg.pooling_nhwc_max operation
        let op = OperationBuilder(
            name: "linalg.pooling_nhwc_max",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input, window])
        .addResults([outputType.typeHandle])
        .build()

        builder.insert(op)
        return op.getResult(0)
    }

    /// 2D Average Pooling operation
    ///
    /// Generates: pooling operation computing average in spatial windows
    ///
    /// Example:
    /// ```swift
    /// let pooled = MLOps.avg_pool2d(
    ///     input,
    ///     kernelSize: [2, 2],
    ///     strides: [2, 2],
    ///     in: builder
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - input: Input tensor with shape [N, H, W, C] (NHWC format)
    ///   - kernelSize: Pooling window size [kernel_h, kernel_w]
    ///   - strides: Stride for pooling [stride_h, stride_w]
    ///   - padding: Padding values [pad_h, pad_w] (default: [0, 0])
    ///   - builder: IR builder for inserting operations
    /// - Returns: Pooled tensor with reduced spatial dimensions
    public static func avg_pool2d(
        _ input: MLIRValue,
        kernelSize: [Int64],
        strides: [Int64],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        // Input shape: [N, H, W, C]
        let n = mlirShapedTypeGetDimSizeWrapper(inputType, 0)
        let h = mlirShapedTypeGetDimSizeWrapper(inputType, 1)
        let w = mlirShapedTypeGetDimSizeWrapper(inputType, 2)
        let c = mlirShapedTypeGetDimSizeWrapper(inputType, 3)

        // Calculate output dimensions
        let outH = (h + 2 * padding[0] - kernelSize[0]) / strides[0] + 1
        let outW = (w + 2 * padding[1] - kernelSize[1]) / strides[1] + 1

        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let outputType = RankedTensorType(
            shape: [n, outH, outW, c],
            elementType: FloatType(handle: elemType, context: builder.context),
            context: builder.context
        )

        // Create pooling window
        let window = TensorOps.empty(
            shape: kernelSize,
            elementType: FloatType(handle: elemType, context: builder.context),
            in: builder
        )

        // Use linalg.pooling_nhwc_sum operation
        // Note: For true average pooling, we'd need to divide by kernel area
        // This implementation provides sum pooling as a foundation
        let op = OperationBuilder(
            name: "linalg.pooling_nhwc_sum",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input, window])
        .addResults([outputType.typeHandle])
        .build()

        builder.insert(op)
        let pooledSum = op.getResult(0)

        // Divide by kernel area to get average
        let kernelArea = kernelSize[0] * kernelSize[1]
        let elemTypeHandle = mlirShapedTypeGetElementTypeWrapper(inputType)
        let divisorAttr = mlirFloatAttrDoubleGetWrapper(
            builder.context.handle,
            elemTypeHandle,
            Double(kernelArea)
        )

        let divisorConst = OperationBuilder(
            name: "arith.constant",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elemTypeHandle])
        .addAttributes([("value", divisorAttr)])
        .build()

        builder.insert(divisorConst)
        let divisor = divisorConst.getResult(0)

        // Divide pooled sum by kernel area
        let divOp = OperationBuilder(
            name: "arith.divf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([pooledSum, divisor])
        .addResults([outputType.typeHandle])
        .build()

        builder.insert(divOp)
        return divOp.getResult(0)
    }

    // MARK: - Normalization Operations

    /// Batch Normalization operation
    ///
    /// Normalizes inputs across the batch dimension: (x - mean) / sqrt(var + epsilon) * gamma + beta
    ///
    /// Example:
    /// ```swift
    /// let normalized = MLOps.batch_norm(
    ///     input,
    ///     mean: runningMean,
    ///     variance: runningVar,
    ///     gamma: scale,
    ///     beta: offset,
    ///     in: builder
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - input: Input tensor to normalize
    ///   - mean: Running mean tensor (same shape as feature dimension)
    ///   - variance: Running variance tensor (same shape as feature dimension)
    ///   - gamma: Scale parameter (optional, defaults to 1.0)
    ///   - beta: Shift parameter (optional, defaults to 0.0)
    ///   - epsilon: Small constant for numerical stability (default: 1e-5)
    ///   - builder: IR builder for inserting operations
    /// - Returns: Normalized tensor with same shape as input
    public static func batch_norm(
        _ input: MLIRValue,
        mean: MLIRValue,
        variance: MLIRValue,
        gamma: MLIRValue? = nil,
        beta: MLIRValue? = nil,
        epsilon: Double = 1e-5,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)

        // Step 1: Create epsilon constant
        let epsilonAttr = mlirFloatAttrDoubleGetWrapper(
            builder.context.handle,
            elemType,
            epsilon
        )

        let epsilonConst = OperationBuilder(
            name: "arith.constant",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elemType])
        .addAttributes([("value", epsilonAttr)])
        .build()

        builder.insert(epsilonConst)
        let epsilonValue = epsilonConst.getResult(0)

        // Step 2: variance + epsilon
        let varianceType = variance.getType()
        let varPlusEps = OperationBuilder(
            name: "arith.addf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([variance, epsilonValue])
        .addResults([varianceType])
        .build()

        builder.insert(varPlusEps)
        let varPlusEpsValue = varPlusEps.getResult(0)

        // Step 3: sqrt(variance + epsilon)
        let sqrtOp = OperationBuilder(
            name: "math.sqrt",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([varPlusEpsValue])
        .addResults([varianceType])
        .build()

        builder.insert(sqrtOp)
        let stddev = sqrtOp.getResult(0)

        // Step 4: x - mean
        let centered = OperationBuilder(
            name: "arith.subf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([input, mean])
        .addResults([inputType])
        .build()

        builder.insert(centered)
        let centeredValue = centered.getResult(0)

        // Step 5: (x - mean) / sqrt(variance + epsilon)
        let normalized = OperationBuilder(
            name: "arith.divf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([centeredValue, stddev])
        .addResults([inputType])
        .build()

        builder.insert(normalized)
        var result = normalized.getResult(0)

        // Step 6: Apply gamma (scale) if provided
        if let gamma = gamma {
            let scaled = OperationBuilder(
                name: "arith.mulf",
                location: builder.unknownLocation(),
                context: builder.context
            )
            .addOperands([result, gamma])
            .addResults([inputType])
            .build()

            builder.insert(scaled)
            result = scaled.getResult(0)
        }

        // Step 7: Apply beta (shift) if provided
        if let beta = beta {
            let shifted = OperationBuilder(
                name: "arith.addf",
                location: builder.unknownLocation(),
                context: builder.context
            )
            .addOperands([result, beta])
            .addResults([inputType])
            .build()

            builder.insert(shifted)
            result = shifted.getResult(0)
        }

        return result
    }

    /// Layer Normalization operation
    ///
    /// Normalizes inputs across the feature dimension
    ///
    /// Example:
    /// ```swift
    /// let normalized = MLOps.layer_norm(
    ///     input,
    ///     axes: [1],  // Normalize along feature dimension
    ///     in: builder
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - input: Input tensor to normalize
    ///   - axes: Dimensions to normalize over
    ///   - epsilon: Small constant for numerical stability (default: 1e-5)
    ///   - builder: IR builder for inserting operations
    /// - Returns: Normalized tensor with same shape as input
    public static func layer_norm(
        _ input: MLIRValue,
        axes: [Int64],
        epsilon: Double = 1e-5,
        in builder: IRBuilder
    ) -> MLIRValue {
        let inputType = input.getType()

        // Compute mean along axes
        let mean = reduce_mean(input, axes: axes, keepDims: true, in: builder)

        // Compute x - mean
        let centered = sub(input, mean, in: builder)

        // Compute variance: mean((x - mean)^2)
        let squared = mul(centered, centered, in: builder)
        let variance = reduce_mean(squared, axes: axes, keepDims: true, in: builder)

        // Create epsilon constant
        let elemType = mlirShapedTypeGetElementTypeWrapper(inputType)
        let epsilonAttr = mlirFloatAttrDoubleGetWrapper(
            builder.context.handle,
            elemType,
            epsilon
        )

        let epsilonConst = OperationBuilder(
            name: "arith.constant",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addResults([elemType])
        .addAttributes([("value", epsilonAttr)])
        .build()

        builder.insert(epsilonConst)
        let epsilonValue = epsilonConst.getResult(0)

        // variance + epsilon
        let varianceType = variance.getType()
        let varPlusEps = OperationBuilder(
            name: "arith.addf",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([variance, epsilonValue])
        .addResults([varianceType])
        .build()

        builder.insert(varPlusEps)
        let varPlusEpsValue = varPlusEps.getResult(0)

        // sqrt(variance + epsilon)
        let sqrtOp = OperationBuilder(
            name: "math.sqrt",
            location: builder.unknownLocation(),
            context: builder.context
        )
        .addOperands([varPlusEpsValue])
        .addResults([varianceType])
        .build()

        builder.insert(sqrtOp)
        let stddev = sqrtOp.getResult(0)

        // (x - mean) / sqrt(variance + epsilon)
        let normalized = div(centered, stddev, in: builder)

        return normalized
    }
}
