//===-- ValueExtensions.swift - ML Value Extensions ------*- Swift -*-===//
//
// SwiftIR - Phase 8: ML-friendly DSL Extensions
// Provides fluent API for ML operations on MLIRValue
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import MLIRCoreWrapper

/// Extensions to MLIRValue for fluent ML operations
extension MLIRValue {

    // MARK: - Element-wise Operations

    /// Element-wise addition: self + rhs
    ///
    /// Example:
    /// ```swift
    /// let result = a.add(b, in: builder)
    /// // Equivalent to: MLOps.add(a, b, in: builder)
    /// ```
    public func add(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.add(self, rhs, in: builder)
    }

    /// Element-wise subtraction: self - rhs
    public func sub(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.sub(self, rhs, in: builder)
    }

    /// Element-wise multiplication: self * rhs
    public func mul(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.mul(self, rhs, in: builder)
    }

    /// Element-wise division: self / rhs
    public func div(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.div(self, rhs, in: builder)
    }

    // MARK: - Activation Functions

    /// Apply ReLU activation: max(0, self)
    public func relu(in builder: IRBuilder) -> MLIRValue {
        MLOps.relu(self, in: builder)
    }

    /// Apply sigmoid activation: 1 / (1 + exp(-self))
    public func sigmoid(in builder: IRBuilder) -> MLIRValue {
        MLOps.sigmoid(self, in: builder)
    }

    /// Apply tanh activation
    public func tanh(in builder: IRBuilder) -> MLIRValue {
        MLOps.tanh(self, in: builder)
    }

    /// Apply softmax normalization
    public func softmax(axis: Int64 = -1, in builder: IRBuilder) -> MLIRValue {
        MLOps.softmax(self, axis: axis, in: builder)
    }

    // MARK: - Linear Algebra

    /// Matrix multiplication: self @ rhs
    ///
    /// Example:
    /// ```swift
    /// let output = input.matmul(weights, in: builder)
    /// // Equivalent to: MLOps.matmul(lhs: input, rhs: weights, in: builder)
    /// ```
    public func matmul(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.matmul(lhs: self, rhs: rhs, in: builder)
    }

    /// Alias for matmul
    public func dot(_ rhs: MLIRValue, in builder: IRBuilder) -> MLIRValue {
        MLOps.dot(self, rhs, in: builder)
    }

    // MARK: - Reduction Operations

    /// Reduce sum along specified axes
    public func reduceSum(axes: [Int64], keepDims: Bool = false, in builder: IRBuilder) -> MLIRValue {
        MLOps.reduce_sum(self, axes: axes, keepDims: keepDims, in: builder)
    }

    /// Reduce mean along specified axes
    public func reduceMean(axes: [Int64], keepDims: Bool = false, in builder: IRBuilder) -> MLIRValue {
        MLOps.reduce_mean(self, axes: axes, keepDims: keepDims, in: builder)
    }

    /// Reduce max along specified axes
    public func reduceMax(axes: [Int64], keepDims: Bool = false, in builder: IRBuilder) -> MLIRValue {
        MLOps.reduce_max(self, axes: axes, keepDims: keepDims, in: builder)
    }

    /// Reduce min along specified axes
    public func reduceMin(axes: [Int64], keepDims: Bool = false, in builder: IRBuilder) -> MLIRValue {
        MLOps.reduce_min(self, axes: axes, keepDims: keepDims, in: builder)
    }

    // MARK: - Pooling Operations

    /// Apply 2D max pooling
    public func maxPool2d(
        kernelSize: [Int64],
        strides: [Int64],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        MLOps.max_pool2d(self, kernelSize: kernelSize, strides: strides, padding: padding, in: builder)
    }

    /// Apply 2D average pooling
    public func avgPool2d(
        kernelSize: [Int64],
        strides: [Int64],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        MLOps.avg_pool2d(self, kernelSize: kernelSize, strides: strides, padding: padding, in: builder)
    }

    // MARK: - Normalization

    /// Apply batch normalization
    public func batchNorm(
        mean: MLIRValue,
        variance: MLIRValue,
        gamma: MLIRValue? = nil,
        beta: MLIRValue? = nil,
        epsilon: Double = 1e-5,
        in builder: IRBuilder
    ) -> MLIRValue {
        MLOps.batch_norm(self, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: epsilon, in: builder)
    }

    /// Apply layer normalization
    public func layerNorm(
        axes: [Int64],
        epsilon: Double = 1e-5,
        in builder: IRBuilder
    ) -> MLIRValue {
        MLOps.layer_norm(self, axes: axes, epsilon: epsilon, in: builder)
    }

    // MARK: - Convolution

    /// Apply 2D convolution
    public func conv2d(
        kernel: MLIRValue,
        strides: [Int64] = [1, 1],
        padding: [Int64] = [0, 0],
        in builder: IRBuilder
    ) -> MLIRValue {
        MLOps.conv2d(input: self, kernel: kernel, strides: strides, padding: padding, in: builder)
    }
}

/// Operator overloads for MLIRValue (opt-in via context)
///
/// Note: These operators require an IRBuilder context to be available.
/// They are primarily useful in @MLIRFunction contexts where the builder is implicit.
extension MLIRValue {

    /// Fluent chaining helper for common patterns
    ///
    /// Example:
    /// ```swift
    /// let output = input
    ///     .matmul(weights, in: builder)
    ///     .add(bias, in: builder)
    ///     .relu(in: builder)
    /// ```
    public func pipe(_ transform: (MLIRValue, IRBuilder) -> MLIRValue, in builder: IRBuilder) -> MLIRValue {
        transform(self, builder)
    }
}

/// Shape inference helpers
extension MLIRValue {

    /// Get the shape of this tensor value
    ///
    /// Returns: Array of dimension sizes, or nil if not a shaped type
    public func shape() -> [Int64]? {
        let type = getType()
        guard mlirTypeIsARankedTensorWrapper(type) else {
            return nil
        }

        let rank = mlirShapedTypeGetRankWrapper(type)
        var shape: [Int64] = []
        for dim in 0..<rank {
            shape.append(mlirShapedTypeGetDimSizeWrapper(type, dim))
        }
        return shape
    }

    /// Get the rank (number of dimensions) of this tensor
    ///
    /// Returns: Number of dimensions, or nil if not a shaped type
    public func rank() -> Int? {
        let type = getType()
        guard mlirTypeIsARankedTensorWrapper(type) else {
            return nil
        }
        return mlirShapedTypeGetRankWrapper(type)
    }

    /// Get the element type of this tensor
    ///
    /// Returns: Element type, or nil if not a shaped type
    public func elementType() -> MlirType? {
        let type = getType()
        guard mlirTypeIsARankedTensorWrapper(type) else {
            return nil
        }
        return mlirShapedTypeGetElementTypeWrapper(type)
    }

    /// Check if this value is a tensor
    public var isTensor: Bool {
        mlirTypeIsARankedTensorWrapper(getType())
    }
}

/// Neural Network Layer Builders
///
/// High-level builders for common neural network patterns
public enum NNLayer {

    /// Linear layer: output = input @ weights + bias
    ///
    /// Example:
    /// ```swift
    /// let output = NNLayer.linear(
    ///     input: x,
    ///     weights: W,
    ///     bias: b,
    ///     in: builder
    /// )
    /// ```
    public static func linear(
        input: MLIRValue,
        weights: MLIRValue,
        bias: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        input.matmul(weights, in: builder).add(bias, in: builder)
    }

    /// Dense layer with activation: activation(input @ weights + bias)
    public static func dense(
        input: MLIRValue,
        weights: MLIRValue,
        bias: MLIRValue,
        activation: (MLIRValue, IRBuilder) -> MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let linear = input.matmul(weights, in: builder).add(bias, in: builder)
        return activation(linear, builder)
    }

    /// Convolutional layer with bias and activation
    public static func conv2dLayer(
        input: MLIRValue,
        kernel: MLIRValue,
        bias: MLIRValue,
        strides: [Int64] = [1, 1],
        padding: [Int64] = [0, 0],
        activation: ((MLIRValue, IRBuilder) -> MLIRValue)? = nil,
        in builder: IRBuilder
    ) -> MLIRValue {
        var output = input.conv2d(kernel: kernel, strides: strides, padding: padding, in: builder)
        output = output.add(bias, in: builder)

        if let activation = activation {
            output = activation(output, builder)
        }

        return output
    }

    /// Residual block: input + conv(conv(input))
    public static func residualBlock(
        input: MLIRValue,
        kernel1: MLIRValue,
        kernel2: MLIRValue,
        in builder: IRBuilder
    ) -> MLIRValue {
        let conv1 = input.conv2d(kernel: kernel1, in: builder).relu(in: builder)
        let conv2 = conv1.conv2d(kernel: kernel2, in: builder)
        return conv2.add(input, in: builder)
    }
}
