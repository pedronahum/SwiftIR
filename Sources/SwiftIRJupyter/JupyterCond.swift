// JupyterCond.swift - Differentiable Conditionals (JAX-like cond/select)
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// This file implements differentiable conditionals that allow branching in traced
// computation graphs while maintaining gradient flow.
//
// Based on JAX's lax.cond: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html

import Foundation
import _Differentiation

// MARK: - Element-wise Select

/// Element-wise conditional selection
///
/// Selects values from `onTrue` where `condition` is true, and from `onFalse` elsewhere.
/// This maps directly to `stablehlo.select` and is the foundation for differentiable masking.
///
/// **Usage Example - Attention Masking**:
/// ```swift
/// let maskedScores = jSelect(
///     mask,
///     onTrue: attentionScores,
///     onFalse: JTracer(value: -1e9, shape: attentionScores.shape, dtype: .float32)
/// )
/// let weights = jSoftmax(maskedScores, axis: -1)
/// ```
///
/// - Parameters:
///   - condition: Boolean mask tensor (will be broadcast if needed)
///   - onTrue: Values selected where condition is true
///   - onFalse: Values selected where condition is false
/// - Returns: Element-wise selection result
public func jSelect(
    _ condition: JTracer,
    onTrue: JTracer,
    onFalse: JTracer
) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    // Determine result shape through broadcasting
    let resultShape = jBroadcastShapes(condition.shape, onTrue.shape, onFalse.shape)
    let resultDtype = onTrue.dtype

    // Broadcast inputs if needed
    let broadcastedCond = jBroadcastIfNeeded(condition, to: resultShape)
    let broadcastedTrue = jBroadcastIfNeeded(onTrue, to: resultShape)
    let broadcastedFalse = jBroadcastIfNeeded(onFalse, to: resultShape)

    // Create select operation
    let resultId = builder.createSelect(
        condition: broadcastedCond.valueId,
        onTrue: broadcastedTrue.valueId,
        onFalse: broadcastedFalse.valueId,
        shape: resultShape,
        dtype: resultDtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: resultShape,
        dtype: resultDtype,
        version: JTracer.incrementVersion()
    )
}

// MARK: - Scalar Conditional (jCond)

/// Differentiable conditional execution based on scalar predicate
///
/// Unlike `jSelect` which works element-wise, `jCond` takes a scalar boolean
/// and executes one of two branches. Both branches are traced to build the computation
/// graph, but only one is executed at runtime.
///
/// **Usage Example - Dynamic Activation**:
/// ```swift
/// let output = jCond(
///     useGelu,
///     onTrue: { jGELU(x) },
///     onFalse: { jReLU(x) }
/// )
/// ```
///
/// **Note**: In the current implementation, this uses `stablehlo.select` internally.
/// Both branches must return tensors of the same shape and dtype.
///
/// - Parameters:
///   - predicate: Scalar boolean tensor
///   - onTrue: Function executed when predicate is true
///   - onFalse: Function executed when predicate is false
/// - Returns: Result from the selected branch
public func jCond(
    _ predicate: JTracer,
    onTrue: () -> JTracer,
    onFalse: () -> JTracer
) -> JTracer {
    // Validate predicate is scalar
    let dims = predicate.shape.dimensions.compactMap { $0 }
    guard dims.isEmpty || dims == [1] else {
        fatalError("jCond: Predicate must be scalar, got shape \(predicate.shape)")
    }

    // Trace both branches to determine output shape
    let trueResult = onTrue()
    let falseResult = onFalse()

    // Validate branches have matching shapes
    let trueDims = trueResult.shape.dimensions.compactMap { $0 }
    let falseDims = falseResult.shape.dimensions.compactMap { $0 }
    guard trueDims == falseDims else {
        fatalError("jCond: Branch output shapes must match. True: \(trueResult.shape), False: \(falseResult.shape)")
    }

    // For simplicity, use select with broadcast scalar predicate
    // A more advanced implementation would use stablehlo.if with regions
    return jSelect(predicate, onTrue: trueResult, onFalse: falseResult)
}

/// Differentiable conditional with operand passing
///
/// Similar to `jCond`, but passes an operand to both branches.
///
/// **Usage Example**:
/// ```swift
/// let output = jCondWith(
///     shouldTransform,
///     operand: x,
///     onTrue: { input in jReLU(jMatmul(input, W)) },
///     onFalse: { input in input }  // Identity/skip
/// )
/// ```
public func jCondWith<T>(
    _ predicate: JTracer,
    operand: T,
    onTrue: (T) -> JTracer,
    onFalse: (T) -> JTracer
) -> JTracer {
    return jCond(
        predicate,
        onTrue: { onTrue(operand) },
        onFalse: { onFalse(operand) }
    )
}

// MARK: - Where (NumPy-style)

/// NumPy-style where operation
///
/// This is an alias for `jSelect` with a more familiar name for NumPy users.
///
/// ```swift
/// let result = jWhere(x > zero, x, zeros)  // ReLU-like
/// ```
public func jWhere(
    _ condition: JTracer,
    _ x: JTracer,
    _ y: JTracer
) -> JTracer {
    return jSelect(condition, onTrue: x, onFalse: y)
}

// MARK: - Helper Functions

/// Broadcast shapes according to NumPy rules
private func jBroadcastShapes(_ shapes: JTensorShape...) -> JTensorShape {
    let allDims = shapes.map { $0.dimensions.compactMap { $0 } }
    let maxRank = allDims.map { $0.count }.max() ?? 0

    var result = [Int](repeating: 1, count: maxRank)

    for dims in allDims {
        let padded = [Int](repeating: 1, count: maxRank - dims.count) + dims
        for i in 0..<maxRank {
            if result[i] == 1 {
                result[i] = padded[i]
            } else if padded[i] != 1 && padded[i] != result[i] {
                fatalError("Shapes are not broadcast compatible: \(shapes)")
            }
        }
    }

    return JTensorShape(result)
}

/// Broadcast tensor to target shape if needed
private func jBroadcastIfNeeded(_ tensor: JTracer, to targetShape: JTensorShape) -> JTracer {
    let tensorDims = tensor.shape.dimensions.compactMap { $0 }
    let targetDims = targetShape.dimensions.compactMap { $0 }

    if tensorDims == targetDims {
        return tensor
    }

    let builder = JTracerGraphBuilder.shared

    // Calculate broadcast dimensions
    let rankDiff = targetDims.count - tensorDims.count
    var broadcastDims: [Int] = []
    for i in 0..<tensorDims.count {
        broadcastDims.append(rankDiff + i)
    }

    let resultId = builder.createBroadcastInDim(
        input: tensor.valueId,
        broadcastDimensions: broadcastDims,
        inputShape: tensor.shape,
        outputShape: targetShape,
        dtype: tensor.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: targetShape,
        dtype: tensor.dtype,
        version: JTracer.incrementVersion()
    )
}

// MARK: - Clamp Operation

/// Clamp values to a range
///
/// Equivalent to: min(max(x, minVal), maxVal)
///
/// ```swift
/// let clamped = jClamp(x, min: 0.0, max: 1.0)
/// ```
public func jClamp(
    _ x: JTracer,
    min minVal: Double,
    max maxVal: Double
) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let resultId = builder.createClamp(
        input: x.valueId,
        minVal: minVal,
        maxVal: maxVal,
        shape: x.shape,
        dtype: x.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: x.shape,
        dtype: x.dtype,
        version: JTracer.incrementVersion()
    )
}

// MARK: - Comparison Operations

/// Greater than comparison
public func jGreater(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .gt)
}

/// Greater than or equal comparison
public func jGreaterEqual(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .ge)
}

/// Less than comparison
public func jLess(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .lt)
}

/// Less than or equal comparison
public func jLessEqual(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .le)
}

/// Equal comparison
public func jEqual(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .eq)
}

/// Not equal comparison
public func jNotEqual(_ lhs: JTracer, _ rhs: JTracer) -> JTracer {
    return jCompareOp(lhs, rhs, direction: .ne)
}

/// Comparison direction enum
public enum JCompareDirection: String {
    case eq = "EQ"
    case ne = "NE"
    case lt = "LT"
    case le = "LE"
    case gt = "GT"
    case ge = "GE"
}

/// Internal comparison operation
private func jCompareOp(_ lhs: JTracer, _ rhs: JTracer, direction: JCompareDirection) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let resultShape = jBroadcastShapes(lhs.shape, rhs.shape)

    // Broadcast if needed
    let broadcastedLhs = jBroadcastIfNeeded(lhs, to: resultShape)
    let broadcastedRhs = jBroadcastIfNeeded(rhs, to: resultShape)

    let resultId = builder.createCompare(
        lhs: broadcastedLhs.valueId,
        rhs: broadcastedRhs.valueId,
        direction: direction,
        shape: resultShape,
        inputDtype: lhs.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: resultShape,
        dtype: .bool,
        version: JTracer.incrementVersion()
    )
}

// NOTE: Comparison operators (<, >, <=, >=) are defined in JupyterWhileLoop.swift
// Do not redeclare them here to avoid duplicate symbol errors.

// MARK: - Graph Builder Extensions for Cond Operations

extension JTracerGraphBuilder {
    /// Create a select operation
    public func createSelect(
        condition: UInt64,
        onTrue: UInt64,
        onFalse: UInt64,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.select(
            id: id,
            condition: condition,
            onTrue: onTrue,
            onFalse: onFalse,
            shape: shape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }

    /// Create a broadcast_in_dim operation
    public func createBroadcastInDim(
        input: UInt64,
        broadcastDimensions: [Int],
        inputShape: JTensorShape,
        outputShape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.broadcastInDim(
            id: id,
            input: input,
            broadcastDimensions: broadcastDimensions,
            inputShape: inputShape,
            outputShape: outputShape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }

    /// Create a clamp operation
    public func createClamp(
        input: UInt64,
        minVal: Double,
        maxVal: Double,
        shape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.clamp(
            id: id,
            input: input,
            minVal: minVal,
            maxVal: maxVal,
            shape: shape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }

    /// Create a compare operation
    public func createCompare(
        lhs: UInt64,
        rhs: UInt64,
        direction: JCompareDirection,
        shape: JTensorShape,
        inputDtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.compare(
            id: id,
            lhs: lhs,
            rhs: rhs,
            direction: direction,
            shape: shape,
            inputDtype: inputDtype
        )
        addOperation(op)
        return id
    }
}

// MARK: - Implementation Notes

/*
 JUPYTER COND IMPLEMENTATION STATUS:

 ✅ Implemented:
    - jSelect() - Element-wise conditional selection (stablehlo.select)
    - jCond() - Scalar conditional with two branches
    - jCondWith() - Conditional with operand passing
    - jWhere() - NumPy-style where
    - jClamp() - Value clamping
    - Comparison operators (>, >=, <, <=)
    - Comparison functions (jGreater, jGreaterEqual, jLess, jLessEqual, jEqual, jNotEqual)

 ⚠️ Current Limitations:
    - jCond uses select internally (both branches are evaluated during tracing)
    - No true stablehlo.if with lazy branch evaluation yet
    - No jSwitch for multi-way branching yet

 🔜 Future Enhancements:
    - True stablehlo.if with region-based branches
    - jSwitch for multi-way conditionals
    - VJP implementations for gradient flow
    - Support for captured variables in branches

 KEY INSIGHTS:
 - jSelect is the workhorse for element-wise masking
 - For scalar conditionals, we currently trace both branches
 - True lazy evaluation requires stablehlo.if regions
 - Gradients flow through both branches with masking
 */
