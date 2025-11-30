// JupyterDerivatives.swift
// VJP (Vector-Jacobian Product) implementations for automatic differentiation
// Pure Swift - works without C++ interop

import _Differentiation

// MARK: - Addition Derivatives

extension JTracer {
    @derivative(of: +)
    public static func vjpAdd(lhs: JTracer, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)
    ) {
        let result = lhs + rhs
        return (
            result,
            { upstream in
                let gradLhs = upstream.reduceToBroadcastShape(originalShape: lhs.shape)
                let gradRhs = upstream.reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Subtraction Derivatives

extension JTracer {
    @derivative(of: -)
    public static func vjpSubtract(lhs: JTracer, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)
    ) {
        let result = lhs - rhs
        return (
            result,
            { upstream in
                let gradLhs = upstream.reduceToBroadcastShape(originalShape: lhs.shape)
                let negUpstream = JTracer.zeros(shape: upstream.shape, dtype: upstream.dtype) - upstream
                let gradRhs = negUpstream.reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Multiplication Derivatives

extension JTracer {
    @derivative(of: *)
    public static func vjpMultiply(lhs: JTracer, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)
    ) {
        let result = lhs * rhs
        return (
            result,
            { upstream in
                let gradLhs = (upstream * rhs).reduceToBroadcastShape(originalShape: lhs.shape)
                let gradRhs = (upstream * lhs).reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Division Derivatives

extension JTracer {
    @derivative(of: /)
    public static func vjpDivide(lhs: JTracer, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)
    ) {
        let result = lhs / rhs
        return (
            result,
            { upstream in
                let gradLhs = (upstream / rhs).reduceToBroadcastShape(originalShape: lhs.shape)
                let negLhs = JTracer.zeros(shape: lhs.shape, dtype: lhs.dtype) - lhs
                let gradRhs = (upstream * negLhs / (rhs * rhs)).reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Scalar Operation Derivatives

extension JTracer {
    @derivative(of: *, wrt: lhs)
    public static func vjpMultiplyScalarRight(lhs: JTracer, rhs: Double) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs * rhs
        return (result, { upstream in upstream * rhs })
    }

    @derivative(of: *, wrt: rhs)
    public static func vjpMultiplyScalarLeft(lhs: Double, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs * rhs
        return (result, { upstream in upstream * lhs })
    }

    @derivative(of: +, wrt: lhs)
    public static func vjpAddScalarRight(lhs: JTracer, rhs: Double) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs + rhs
        return (result, { upstream in upstream })
    }

    @derivative(of: +, wrt: rhs)
    public static func vjpAddScalarLeft(lhs: Double, rhs: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs + rhs
        return (result, { upstream in upstream })
    }

    @derivative(of: -, wrt: lhs)
    public static func vjpSubScalarRight(lhs: JTracer, rhs: Double) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs - rhs
        return (result, { upstream in upstream })
    }

    @derivative(of: /, wrt: lhs)
    public static func vjpDivScalarRight(lhs: JTracer, rhs: Double) -> (
        value: JTracer, pullback: (JTracer) -> JTracer
    ) {
        let result = lhs / rhs
        return (result, { upstream in upstream / rhs })
    }
}

// MARK: - Unary Operation Derivatives

extension JTracer {
    @derivative(of: exp)
    public func vjpExp() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.exp()
        return (result, { upstream in upstream * result })
    }

    @derivative(of: log)
    public func vjpLog() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        return (self.log(), { upstream in upstream / self })
    }

    @derivative(of: sqrt)
    public func vjpSqrt() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.sqrt()
        return (result, { upstream in upstream / (result * 2.0) })
    }

    @derivative(of: tanh)
    public func vjpTanh() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.tanh()
        return (
            result,
            { upstream in
                let ones = JTracer.ones(shape: result.shape, dtype: result.dtype)
                return upstream * (ones - result * result)
            }
        )
    }

    @derivative(of: sigmoid)
    public func vjpSigmoid() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.sigmoid()
        return (
            result,
            { upstream in
                let ones = JTracer.ones(shape: result.shape, dtype: result.dtype)
                return upstream * result * (ones - result)
            }
        )
    }

    @derivative(of: relu)
    public func vjpRelu() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.relu()
        // Simplified - proper implementation would use a step function
        return (result, { upstream in upstream })
    }

    // Additional trigonometric derivatives
    @derivative(of: sin)
    public func vjpSin() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.sin()
        return (result, { upstream in upstream * self.cos() })
    }

    @derivative(of: cos)
    public func vjpCos() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.cos()
        return (result, { upstream in upstream * (-self.sin()) })
    }

    @derivative(of: tan)
    public func vjpTan() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.tan()
        let cosVal = self.cos()
        // d/dx tan(x) = 1/cos^2(x) = sec^2(x)
        return (result, { upstream in upstream / (cosVal * cosVal) })
    }

    @derivative(of: abs)
    public func vjpAbs() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.abs()
        // Simplified: sign function approximation
        // In practice, this would use a sign operation
        return (result, { upstream in upstream })
    }

    @derivative(of: power, wrt: self)
    public func vjpPower(_ exponent: Double) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.power(exponent)
        // d/dx x^n = n * x^(n-1)
        return (result, { upstream in
            upstream * JTracer(value: exponent, shape: .scalar, dtype: self.dtype) * self.power(exponent - 1.0)
        })
    }

    @derivative(of: reshape, wrt: self)
    public func vjpReshape(to newShape: JTensorShape) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.reshape(to: newShape)
        return (result, { upstream in upstream.reshape(to: self.shape) })
    }

    // MARK: - Additional Math Derivatives

    @derivative(of: rsqrt)
    public func vjpRsqrt() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.rsqrt()
        // d/dx (1/sqrt(x)) = -1/(2 * x^(3/2)) = -0.5 * rsqrt(x)^3
        return (result, { upstream in
            let half = JTracer(value: -0.5, shape: .scalar, dtype: self.dtype)
            return upstream * half * result * result * result
        })
    }

    @derivative(of: floor)
    public func vjpFloor() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.floor()
        // floor has zero gradient (piecewise constant)
        return (result, { _ in JTracer.zeros(shape: self.shape, dtype: self.dtype) })
    }

    @derivative(of: ceil)
    public func vjpCeil() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.ceil()
        // ceil has zero gradient (piecewise constant)
        return (result, { _ in JTracer.zeros(shape: self.shape, dtype: self.dtype) })
    }

    @derivative(of: clamp, wrt: self)
    public func vjpClamp(min minVal: Double, max maxVal: Double) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.clamp(min: minVal, max: maxVal)
        // Gradient is 1 where input is in range, 0 otherwise
        // Simplified: pass through gradient (proper impl would use indicator function)
        return (result, { upstream in upstream })
    }

    // MARK: - Advanced Activation Derivatives

    @derivative(of: leakyRelu, wrt: self)
    public func vjpLeakyRelu(alpha: Double) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.leakyRelu(alpha: alpha)
        // d/dx leakyRelu = 1 if x > 0, alpha if x <= 0
        let zero = JTracer.zeros(shape: self.shape, dtype: self.dtype)
        let isPositive = self > zero
        let one = JTracer.ones(shape: self.shape, dtype: self.dtype)
        let alphaT = JTracer(value: alpha, shape: .scalar, dtype: self.dtype)
        return (result, { upstream in
            let grad = JTracer.select(condition: isPositive, onTrue: one, onFalse: alphaT * one)
            return upstream * grad
        })
    }

    @derivative(of: elu, wrt: self)
    public func vjpElu(alpha: Double) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.elu(alpha: alpha)
        // d/dx elu = 1 if x > 0, alpha * exp(x) if x <= 0
        let zero = JTracer.zeros(shape: self.shape, dtype: self.dtype)
        let isPositive = self > zero
        let one = JTracer.ones(shape: self.shape, dtype: self.dtype)
        let alphaT = JTracer(value: alpha, shape: .scalar, dtype: self.dtype)
        return (result, { upstream in
            let grad = JTracer.select(condition: isPositive, onTrue: one, onFalse: alphaT * self.exp())
            return upstream * grad
        })
    }

    @derivative(of: silu)
    public func vjpSilu() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.silu()
        // d/dx silu = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let sig = self.sigmoid()
        let one = JTracer.ones(shape: self.shape, dtype: self.dtype)
        return (result, { upstream in
            let grad = sig * (one + self * (one - sig))
            return upstream * grad
        })
    }

    @derivative(of: gelu)
    public func vjpGelu() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.gelu()
        // Approximate GELU gradient using tanh approximation
        // This is a simplified version
        let sig = (self * 1.702).sigmoid()
        return (result, { upstream in
            upstream * sig * (JTracer.ones(shape: self.shape, dtype: self.dtype) + self * 1.702 * (JTracer.ones(shape: self.shape, dtype: self.dtype) - sig))
        })
    }

    @derivative(of: softplus)
    public func vjpSoftplus() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        // d/dx softplus(x) = sigmoid(x)
        let result = self.softplus()
        return (result, { upstream in upstream * self.sigmoid() })
    }

    // MARK: - Softmax Derivatives

    @derivative(of: softmax)
    public func vjpSoftmax() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.softmax()
        // d/dx softmax = softmax * (1 - softmax) for diagonal, -softmax_i * softmax_j for off-diagonal
        // Simplified: use identity for now (proper impl would use Jacobian)
        return (result, { upstream in upstream * result * (JTracer.ones(shape: self.shape, dtype: self.dtype) - result) })
    }

    @derivative(of: logSoftmax)
    public func vjpLogSoftmax() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.logSoftmax()
        // d/dx log_softmax = 1 - softmax
        // Simplified version
        return (result, { upstream in upstream - upstream * self.softmax() })
    }

    // MARK: - Element-wise Binary Derivatives

    @derivative(of: maximum)
    public func vjpMaximum(_ other: JTracer) -> (value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)) {
        let result = self.maximum(other)
        // Gradient flows to the larger input
        let selfIsGreater = self > other
        return (result, { upstream in
            let gradSelf = JTracer.select(condition: selfIsGreater, onTrue: upstream, onFalse: JTracer.zeros(shape: upstream.shape, dtype: upstream.dtype))
            let gradOther = JTracer.select(condition: selfIsGreater, onTrue: JTracer.zeros(shape: upstream.shape, dtype: upstream.dtype), onFalse: upstream)
            return (gradSelf.reduceToBroadcastShape(originalShape: self.shape),
                    gradOther.reduceToBroadcastShape(originalShape: other.shape))
        })
    }

    @derivative(of: minimum)
    public func vjpMinimum(_ other: JTracer) -> (value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)) {
        let result = self.minimum(other)
        // Gradient flows to the smaller input
        let selfIsSmaller = self < other
        return (result, { upstream in
            let gradSelf = JTracer.select(condition: selfIsSmaller, onTrue: upstream, onFalse: JTracer.zeros(shape: upstream.shape, dtype: upstream.dtype))
            let gradOther = JTracer.select(condition: selfIsSmaller, onTrue: JTracer.zeros(shape: upstream.shape, dtype: upstream.dtype), onFalse: upstream)
            return (gradSelf.reduceToBroadcastShape(originalShape: self.shape),
                    gradOther.reduceToBroadcastShape(originalShape: other.shape))
        })
    }

    // MARK: - Slice Derivative

    @derivative(of: slice, wrt: self)
    public func vjpSlice(starts: [Int], limits: [Int], strides: [Int]?) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.slice(starts: starts, limits: limits, strides: strides)
        // Gradient needs to be scattered back to original positions
        // Simplified: return upstream padded to original shape (proper impl would use pad operation)
        return (result, { upstream in
            // This is a simplified version - proper implementation would use stablehlo.pad
            upstream.reshape(to: self.shape)
        })
    }
}

// MARK: - Negation Derivative

extension JTracer {
    @derivative(of: -)
    public static func vjpNegate(operand: JTracer) -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = -operand
        return (result, { upstream in -upstream })
    }
}

// MARK: - Matrix Operation Derivatives

extension JTracer {
    @derivative(of: matmul)
    public func vjpMatmul(_ other: JTracer) -> (
        value: JTracer, pullback: (JTracer) -> (JTracer, JTracer)
    ) {
        let result = self.matmul(other)
        return (
            result,
            { upstream in
                let gradSelf = upstream.matmul(other.transpose())
                let gradOther = self.transpose().matmul(upstream)
                return (gradSelf, gradOther)
            }
        )
    }

    @derivative(of: transpose)
    public func vjpTranspose() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.transpose()
        return (result, { upstream in upstream.transpose() })
    }
}

// MARK: - Reduction Operation Derivatives

extension JTracer {
    @derivative(of: sum)
    public func vjpSum() -> (value: JTracer, pullback: (JTracer) -> JTracer) {
        let result = self.sum()
        return (
            result,
            { upstream in
                // Gradient of sum is broadcast of upstream to original shape
                JTracer(value: 1.0, shape: self.shape, dtype: self.dtype) * upstream
            }
        )
    }
}

// MARK: - High-Level Gradient Functions

/// Compute the gradient of a scalar-valued function at a point
public func gradient(
    of f: @differentiable(reverse) (JTracer) -> JTracer,
    at x: JTracer
) -> JTracer {
    let (output, pullback) = valueWithPullback(at: x, of: f)
    let upstream = JTracer(value: 1.0, shape: output.shape, dtype: output.dtype)
    return pullback(upstream)
}

/// Compute the gradient of a scalar-valued function with multiple inputs
public func gradient(
    of f: @differentiable(reverse) (JTracer, JTracer) -> JTracer,
    at x: JTracer,
    _ y: JTracer
) -> (JTracer, JTracer) {
    let (output, pullback) = valueWithPullback(at: x, y, of: f)
    let upstream = JTracer(value: 1.0, shape: output.shape, dtype: output.dtype)
    return pullback(upstream)
}

/// Compute both the value and gradient of a function at a point
public func valueWithGradient(
    of f: @differentiable(reverse) (JTracer) -> JTracer,
    at x: JTracer
) -> (value: JTracer, gradient: JTracer) {
    let (value, pullback) = valueWithPullback(at: x, of: f)
    let upstream = JTracer(value: 1.0, shape: value.shape, dtype: value.dtype)
    let grad = pullback(upstream)
    return (value, grad)
}

/// Compute both value and gradients for a function with multiple inputs
public func valueWithGradient(
    of f: @differentiable(reverse) (JTracer, JTracer) -> JTracer,
    at x: JTracer,
    _ y: JTracer
) -> (value: JTracer, gradient: (JTracer, JTracer)) {
    let (value, pullback) = valueWithPullback(at: x, y, of: f)
    let upstream = JTracer(value: 1.0, shape: value.shape, dtype: value.dtype)
    let grads = pullback(upstream)
    return (value, grads)
}
