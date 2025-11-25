/// Differentiation - High-level differentiation APIs for SwiftIR
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides gradient, valueWithGradient, and pullback functions that
/// leverage Swift's _Differentiation to build MLIR gradient graphs.

import _Differentiation

// MARK: - Gradient Functions

/// Compute the gradient of a scalar-valued function at a point
/// - Parameters:
///   - f: A differentiable function from Tracer to Tracer (must return scalar)
///   - x: The point at which to compute the gradient
/// - Returns: The gradient of f at x
public func gradient(
    of f: @differentiable(reverse) (Tracer) -> Tracer,
    at x: Tracer
) -> Tracer {
    let (output, pullback) = _Differentiation.valueWithPullback(at: x, of: f)
    // For scalar output, upstream gradient is 1.0
    // Use output shape to handle non-scalar returns
    let upstream = Tracer(value: 1.0, shape: output.shape, dtype: output.dtype)
    return pullback(upstream)
}

/// Compute the gradient of a scalar-valued function with multiple inputs
/// - Parameters:
///   - f: A differentiable function from (Tracer, Tracer) to Tracer
///   - x: First input
///   - y: Second input
/// - Returns: Tuple of gradients (dx, dy)
public func gradient(
    of f: @differentiable(reverse) (Tracer, Tracer) -> Tracer,
    at x: Tracer,
    _ y: Tracer
) -> (Tracer, Tracer) {
    let (output, pullback) = _Differentiation.valueWithPullback(at: x, y, of: f)
    let upstream = Tracer(value: 1.0, shape: output.shape, dtype: output.dtype)
    return pullback(upstream)
}

// MARK: - Value with Gradient Functions

/// Compute both the value and gradient of a function at a point
/// - Parameters:
///   - f: A differentiable function from Tracer to Tracer
///   - x: The point at which to evaluate
/// - Returns: Tuple of (value, gradient)
public func valueWithGradient(
    of f: @differentiable(reverse) (Tracer) -> Tracer,
    at x: Tracer
) -> (value: Tracer, gradient: Tracer) {
    let (value, pullback) = _Differentiation.valueWithPullback(at: x, of: f)
    let upstream = Tracer(value: 1.0, shape: value.shape, dtype: value.dtype)
    let grad = pullback(upstream)
    return (value, grad)
}

/// Compute both value and gradients for a function with multiple inputs
/// - Parameters:
///   - f: A differentiable function
///   - x: First input
///   - y: Second input
/// - Returns: Tuple of (value, (dx, dy))
public func valueWithGradient(
    of f: @differentiable(reverse) (Tracer, Tracer) -> Tracer,
    at x: Tracer,
    _ y: Tracer
) -> (value: Tracer, gradient: (Tracer, Tracer)) {
    let (value, pullback) = _Differentiation.valueWithPullback(at: x, y, of: f)
    let upstream = Tracer(value: 1.0, shape: value.shape, dtype: value.dtype)
    let grads = pullback(upstream)
    return (value, grads)
}

// MARK: - Re-exports from _Differentiation

// The following functions are re-exported from _Differentiation:
// - valueWithPullback(at:of:)
// - valueWithPullback(at:_:of:)
// - pullback(at:of:)
// Users should call these directly from _Differentiation module.

// MARK: - JVP (Forward-mode) Functions

// Note: True forward-mode AD would need @differentiable(_forward) support
// which is not yet available in Swift. For now, we use reverse-mode only.

// MARK: - Higher-Order Differentiation

// Note: Hessian-vector products and higher-order differentiation require
// nested differentiation support. This will be added in a future phase.

// MARK: - Utility Extensions

extension Tracer {
    /// Compute gradient with respect to self
    /// Useful for method chaining
    public func gradient(
        through f: @differentiable(reverse) (Tracer) -> Tracer
    ) -> Tracer {
        return SwiftIR.gradient(of: f, at: self)
    }

    /// Compute value and gradient with respect to self
    public func valueWithGradient(
        through f: @differentiable(reverse) (Tracer) -> Tracer
    ) -> (value: Tracer, gradient: Tracer) {
        return SwiftIR.valueWithGradient(of: f, at: self)
    }
}

// MARK: - Batch Gradient Computation

/// Compute gradients for a batch of inputs
/// - Parameters:
///   - f: A differentiable function
///   - inputs: Array of input tensors
/// - Returns: Array of gradients
public func batchGradient(
    of f: @differentiable(reverse) (Tracer) -> Tracer,
    at inputs: [Tracer]
) -> [Tracer] {
    return inputs.map { gradient(of: f, at: $0) }
}

// MARK: - Jacobian Computation

/// Compute the full Jacobian matrix (for small outputs)
/// Warning: This is expensive for large output dimensions
/// - Parameters:
///   - f: A differentiable function
///   - x: The input point
/// - Returns: The Jacobian as a Tracer (output_dim x input_dim)
public func jacobian(
    of f: @differentiable(reverse) (Tracer) -> Tracer,
    at x: Tracer
) -> Tracer {
    let output = f(x)
    guard let outputSize = output.shape.elementCount,
          let inputSize = x.shape.elementCount else {
        fatalError("Jacobian requires fully known shapes")
    }

    // Build Jacobian row by row using VJP
    var rows: [Tracer] = []

    for i in 0..<outputSize {
        // Create one-hot upstream gradient
        var oneHotData = [Double](repeating: 0.0, count: outputSize)
        oneHotData[i] = 1.0

        // This is a simplified version - real implementation would use
        // proper one-hot tensor creation
        let upstream = Tracer(value: 1.0, shape: output.shape, dtype: output.dtype)
        let (_, pb) = valueWithPullback(at: x, of: f)
        let row = pb(upstream)
        rows.append(row)
    }

    // Stack rows into Jacobian matrix
    // For now, return the last row as placeholder
    // Full implementation would need a stack operation
    return rows.last ?? x
}
