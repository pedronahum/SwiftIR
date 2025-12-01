// Cond.swift - Differentiable Conditionals (JAX-like cond/select)
// Copyright 2024 SwiftIR Project
//
// This file implements differentiable conditionals that allow branching in traced
// computation graphs while maintaining gradient flow.
//
// Based on JAX's lax.cond: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html
//
// NOTE: diffSelect and diffLessEqual are defined in ADIntegration.swift with VJP support.
// Comparison operators (<, >, <=, >=) are defined in DifferentiableWhile.swift.

import Foundation
import _Differentiation

// MARK: - Scalar Conditional (diffCond)

/// Differentiable conditional execution based on scalar predicate
///
/// Unlike `diffSelect` which works element-wise, `diffCond` takes a scalar boolean
/// and executes one of two branches. Both branches are traced to build the computation
/// graph, but only one is executed at runtime.
///
/// **Usage Example - Dynamic Activation**:
/// ```swift
/// let output = diffCond(
///     useGelu,
///     onTrue: { diffGELU(x) },
///     onFalse: { diffReLU(x) }
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
public func diffCond(
    _ predicate: DifferentiableTracer,
    onTrue: () -> DifferentiableTracer,
    onFalse: () -> DifferentiableTracer
) -> DifferentiableTracer {
    guard DifferentiableTracer.currentBuilder != nil else {
        fatalError("diffCond requires an active MLIRBuilder (must be called during tracing)")
    }

    // Validate predicate is scalar
    guard predicate.shape.isEmpty || predicate.shape == [1] else {
        fatalError("diffCond: Predicate must be scalar, got shape \(predicate.shape)")
    }

    // Trace both branches to determine output shape
    let trueResult = onTrue()
    let falseResult = onFalse()

    // Validate branches have matching shapes
    guard trueResult.shape == falseResult.shape else {
        fatalError("diffCond: Branch output shapes must match. True: \(trueResult.shape), False: \(falseResult.shape)")
    }

    // Use diffSelect from ADIntegration.swift
    return diffSelect(predicate, trueResult, falseResult)
}

/// Differentiable conditional with operand passing
///
/// Similar to `diffCond`, but passes an operand to both branches.
///
/// **Usage Example**:
/// ```swift
/// let output = diffCondWith(
///     shouldTransform,
///     operand: x,
///     onTrue: { input in diffReLU(diffMatmul(input, W)) },
///     onFalse: { input in input }  // Identity/skip
/// )
/// ```
public func diffCondWith<T>(
    _ predicate: DifferentiableTracer,
    operand: T,
    onTrue: (T) -> DifferentiableTracer,
    onFalse: (T) -> DifferentiableTracer
) -> DifferentiableTracer {
    return diffCond(
        predicate,
        onTrue: { onTrue(operand) },
        onFalse: { onFalse(operand) }
    )
}

// MARK: - Where (NumPy-style)

/// NumPy-style where operation
///
/// This is an alias for `diffSelect` with a more familiar name for NumPy users.
///
/// ```swift
/// let result = diffWhere(x > zero, x, zeros)  // ReLU-like
/// ```
public func diffWhere(
    _ condition: DifferentiableTracer,
    _ x: DifferentiableTracer,
    _ y: DifferentiableTracer
) -> DifferentiableTracer {
    return diffSelect(condition, x, y)
}

// MARK: - Clamp Operation

/// Clamp values to a range
///
/// Equivalent to: min(max(x, minVal), maxVal)
///
/// ```swift
/// let clamped = diffClamp(x, min: 0.0, max: 1.0)
/// ```
public func diffClamp(
    _ x: DifferentiableTracer,
    min minVal: Float,
    max maxVal: Float
) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffClamp requires an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()
    let typeStr = tensorType(shape: x.shape, dtype: x.dtype)

    // Create min and max constants
    let minSSA = builder.freshSSA()
    let maxSSA = builder.freshSSA()

    builder.addOperation(MLIROperation(
        result: minSSA,
        opName: "stablehlo.constant",
        operands: [],
        attributes: ["value": "dense<\(minVal)> : \(typeStr)"],
        resultType: typeStr
    ))

    builder.addOperation(MLIROperation(
        result: maxSSA,
        opName: "stablehlo.constant",
        operands: [],
        attributes: ["value": "dense<\(maxVal)> : \(typeStr)"],
        resultType: typeStr
    ))

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.clamp",
        operands: [minSSA, x.irValue, maxSSA],
        attributes: [:],
        resultType: typeStr
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: x.shape, dtype: x.dtype)
}

// MARK: - Additional Comparison Functions

/// Greater than comparison function
public func diffGreater(_ lhs: DifferentiableTracer, _ rhs: DifferentiableTracer) -> DifferentiableTracer {
    return lhs > rhs  // Uses operator from DifferentiableWhile.swift
}

/// Greater than or equal comparison function
public func diffGreaterEqual(_ lhs: DifferentiableTracer, _ rhs: DifferentiableTracer) -> DifferentiableTracer {
    return lhs >= rhs  // Uses operator from DifferentiableWhile.swift
}

/// Less than comparison function
public func diffLess(_ lhs: DifferentiableTracer, _ rhs: DifferentiableTracer) -> DifferentiableTracer {
    return lhs < rhs  // Uses operator from DifferentiableWhile.swift
}

/// Equal comparison
public func diffEqual(_ lhs: DifferentiableTracer, _ rhs: DifferentiableTracer) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("Compare operations require an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: lhs.shape, dtype: .bool)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.compare",
        operands: [lhs.irValue, rhs.irValue],
        attributes: [
            "comparison_direction": "#stablehlo<comparison_direction EQ>"
        ],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: lhs.shape, dtype: .bool)
}

/// Not equal comparison
public func diffNotEqual(_ lhs: DifferentiableTracer, _ rhs: DifferentiableTracer) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("Compare operations require an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: lhs.shape, dtype: .bool)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.compare",
        operands: [lhs.irValue, rhs.irValue],
        attributes: [
            "comparison_direction": "#stablehlo<comparison_direction NE>"
        ],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: lhs.shape, dtype: .bool)
}

// MARK: - Implementation Notes

/*
 COND/SELECT IMPLEMENTATION STATUS:

 ✅ Implemented:
    - diffCond() - Scalar conditional with two branches (this file)
    - diffCondWith() - Conditional with operand passing (this file)
    - diffWhere() - NumPy-style where (this file)
    - diffClamp() - Value clamping (this file)
    - diffEqual/diffNotEqual - Equality comparisons (this file)
    - diffSelect() - Element-wise conditional selection (ADIntegration.swift)
    - diffLessEqual() - Less than or equal comparison (ADIntegration.swift)
    - Comparison operators (<, >, <=, >=) - (DifferentiableWhile.swift)

 ⚠️ Current Limitations:
    - diffCond uses select internally (both branches are evaluated)
    - No true stablehlo.if with lazy branch evaluation yet
    - No diffSwitch for multi-way branching yet

 🔜 Future Enhancements:
    - True stablehlo.if with region-based branches
    - diffSwitch for multi-way conditionals
    - Support for captured variables in branches

 KEY INSIGHTS:
 - diffSelect is the workhorse for element-wise masking
 - For scalar conditionals, we currently evaluate both branches
 - True lazy evaluation requires stablehlo.if regions
 - Gradients flow through both branches with masking
 */
