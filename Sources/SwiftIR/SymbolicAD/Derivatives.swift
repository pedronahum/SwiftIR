/// Derivatives - VJP (Vector-Jacobian Product) implementations for automatic differentiation
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// These @derivative functions intercept Swift's native AD to build MLIR gradient graphs.
/// CRITICAL: Must handle broadcasting correctly in pullbacks by reducing to original shapes.

import _Differentiation

// MARK: - Helper Functions for Gradient Reduction

extension Tracer {
    /// Reduce a gradient to match the original tensor's shape after broadcasting
    /// This is CRITICAL for correct gradient computation
    public func reduceToBroadcastShape(originalShape: TensorShape) -> Tracer {
        // If shapes already match, no reduction needed
        if self.shape == originalShape {
            return self
        }

        // Find axes that were broadcast (size 1 or missing in original)
        var axesToReduce: Set<Int> = []
        let selfDims = self.shape.dimensions
        let origDims = originalShape.dimensions

        // Handle different ranks by padding original with leading 1s
        let rankDiff = selfDims.count - origDims.count
        let paddedOrigDims = Array(repeating: 1 as Int?, count: rankDiff) + origDims

        for (index, (selfDim, origDim)) in zip(selfDims, paddedOrigDims).enumerated() {
            if let od = origDim, od == 1, let sd = selfDim, sd > 1 {
                // This axis was broadcast from 1 to larger size
                axesToReduce.insert(index)
            }
        }

        // If there are leading dimensions that didn't exist in original, sum them out
        for i in 0..<rankDiff {
            axesToReduce.insert(i)
        }

        if axesToReduce.isEmpty {
            return self
        }

        // Sum along broadcast axes, keeping dims for now
        let result = self.sum(alongAxes: axesToReduce, keepDims: true)

        return result
    }
}

// MARK: - Addition Derivatives

extension Tracer {
    @derivative(of: +)
    public static func vjpAdd(lhs: Tracer, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> (Tracer, Tracer)
    ) {
        let result = lhs + rhs

        return (
            result,
            { upstream in
                // Gradient of addition: d(a+b)/da = 1, d(a+b)/db = 1
                let gradLhs = upstream.reduceToBroadcastShape(originalShape: lhs.shape)
                let gradRhs = upstream.reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Subtraction Derivatives

extension Tracer {
    @derivative(of: -)
    public static func vjpSubtract(lhs: Tracer, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> (Tracer, Tracer)
    ) {
        let result = lhs - rhs

        return (
            result,
            { upstream in
                // Gradient of subtraction: d(a-b)/da = 1, d(a-b)/db = -1
                let gradLhs = upstream.reduceToBroadcastShape(originalShape: lhs.shape)
                let negUpstream = Tracer.zeros(shape: upstream.shape, dtype: upstream.dtype) - upstream
                let gradRhs = negUpstream.reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Multiplication Derivatives

extension Tracer {
    @derivative(of: *)
    public static func vjpMultiply(lhs: Tracer, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> (Tracer, Tracer)
    ) {
        let result = lhs * rhs

        return (
            result,
            { upstream in
                // Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
                let gradLhs = (upstream * rhs).reduceToBroadcastShape(originalShape: lhs.shape)
                let gradRhs = (upstream * lhs).reduceToBroadcastShape(originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Division Derivatives

extension Tracer {
    @derivative(of: /)
    public static func vjpDivide(lhs: Tracer, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> (Tracer, Tracer)
    ) {
        let result = lhs / rhs

        return (
            result,
            { upstream in
                // Gradient of division: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                let gradLhs = (upstream / rhs).reduceToBroadcastShape(originalShape: lhs.shape)
                let negLhs = Tracer.zeros(shape: lhs.shape, dtype: lhs.dtype) - lhs
                let gradRhs = (upstream * negLhs / (rhs * rhs)).reduceToBroadcastShape(
                    originalShape: rhs.shape)
                return (gradLhs, gradRhs)
            }
        )
    }
}

// MARK: - Scalar Operation Derivatives

extension Tracer {
    @derivative(of: *, wrt: lhs)
    public static func vjpMultiplyScalarRight(lhs: Tracer, rhs: Double) -> (
        value: Tracer, pullback: (Tracer) -> Tracer
    ) {
        let result = lhs * rhs

        return (
            result,
            { upstream in
                return upstream * rhs
            }
        )
    }

    @derivative(of: *, wrt: rhs)
    public static func vjpMultiplyScalarLeft(lhs: Double, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> Tracer
    ) {
        let result = lhs * rhs

        return (
            result,
            { upstream in
                return upstream * lhs
            }
        )
    }

    @derivative(of: +, wrt: lhs)
    public static func vjpAddScalarRight(lhs: Tracer, rhs: Double) -> (
        value: Tracer, pullback: (Tracer) -> Tracer
    ) {
        let result = lhs + rhs

        return (
            result,
            { upstream in
                return upstream
            }
        )
    }

    @derivative(of: +, wrt: rhs)
    public static func vjpAddScalarLeft(lhs: Double, rhs: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> Tracer
    ) {
        let result = lhs + rhs

        return (
            result,
            { upstream in
                return upstream
            }
        )
    }
}

// MARK: - Unary Operations on Tracer

extension Tracer {
    /// Exponential
    public func exp() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .exp,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "exp")
        )
    }

    /// Natural logarithm
    public func log() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .log,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "log")
        )
    }

    /// Square root
    public func sqrt() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .sqrt,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "sqrt")
        )
    }

    /// Hyperbolic tangent
    public func tanh() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .tanh,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "tanh")
        )
    }

    /// Sigmoid activation
    public func sigmoid() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .sigmoid,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "sigmoid")
        )
    }

    /// ReLU activation
    public func relu() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .relu,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "relu")
        )
    }

    /// Absolute value
    public func abs() -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .abs,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "abs")
        )
    }

    /// Negation
    public static prefix func - (operand: Tracer) -> Tracer {
        let id = TracerGraphBuilder.shared.createUnaryOp(
            operation: .neg,
            input: operand.value,
            shape: operand.shape,
            dtype: operand.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: operand.shape,
            dtype: operand.dtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "neg")
        )
    }
}

// MARK: - Matrix Operations

extension Tracer {
    /// Matrix multiplication
    public func matmul(_ other: Tracer) -> Tracer {
        guard shape.rank >= 2, other.shape.rank >= 2 else {
            fatalError("matmul requires tensors with rank >= 2")
        }

        let m = shape.dimensions[shape.rank - 2]
        let k1 = shape.dimensions[shape.rank - 1]
        let k2 = other.shape.dimensions[other.shape.rank - 2]
        let n = other.shape.dimensions[other.shape.rank - 1]

        if let kk1 = k1, let kk2 = k2, kk1 != kk2 {
            fatalError("matmul: inner dimensions must match, got \(kk1) vs \(kk2)")
        }

        var resultDims = Array(shape.dimensions.dropLast(2))
        resultDims.append(m)
        resultDims.append(n)

        let resultShape = TensorShape(dimensions: resultDims)

        let id = TracerGraphBuilder.shared.createMatMul(
            lhs: self.value,
            rhs: other.value,
            shape: resultShape,
            dtype: self.dtype.promoted(with: other.dtype)
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: self.dtype.promoted(with: other.dtype),
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "matmul")
        )
    }

    /// Transpose the last two dimensions
    public func transpose() -> Tracer {
        guard shape.rank >= 2 else {
            return self
        }

        var newDims = shape.dimensions
        let lastIdx = newDims.count - 1
        newDims.swapAt(lastIdx - 1, lastIdx)

        let resultShape = TensorShape(dimensions: newDims)

        let id = TracerGraphBuilder.shared.createTranspose(
            input: self.value,
            shape: resultShape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "transpose")
        )
    }
}

// MARK: - Unary Operation Derivatives

extension Tracer {
    @derivative(of: exp)
    public func vjpExp() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.exp()
        return (
            result,
            { upstream in
                return upstream * result
            }
        )
    }

    @derivative(of: log)
    public func vjpLog() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        return (
            self.log(),
            { upstream in
                return upstream / self
            }
        )
    }

    @derivative(of: sqrt)
    public func vjpSqrt() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.sqrt()
        return (
            result,
            { upstream in
                return upstream / (result * 2.0)
            }
        )
    }

    @derivative(of: tanh)
    public func vjpTanh() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.tanh()
        return (
            result,
            { upstream in
                let ones = Tracer.ones(shape: result.shape, dtype: result.dtype)
                return upstream * (ones - result * result)
            }
        )
    }

    @derivative(of: sigmoid)
    public func vjpSigmoid() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.sigmoid()
        return (
            result,
            { upstream in
                let ones = Tracer.ones(shape: result.shape, dtype: result.dtype)
                return upstream * result * (ones - result)
            }
        )
    }

    @derivative(of: relu)
    public func vjpRelu() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.relu()
        return (
            result,
            { upstream in
                // Simplified - would need proper step function in production
                return upstream
            }
        )
    }
}

// MARK: - Negation Derivative

extension Tracer {
    @derivative(of: -)
    public static func vjpNegate(operand: Tracer) -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = -operand
        return (
            result,
            { upstream in
                return -upstream
            }
        )
    }
}

// MARK: - Matrix Operation Derivatives

extension Tracer {
    @derivative(of: matmul)
    public func vjpMatmul(_ other: Tracer) -> (
        value: Tracer, pullback: (Tracer) -> (Tracer, Tracer)
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
    public func vjpTranspose() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.transpose()
        return (
            result,
            { upstream in
                return upstream.transpose()
            }
        )
    }
}

// MARK: - Reduction Operation Derivatives

extension Tracer {
    @derivative(of: sum)
    public func vjpSum() -> (value: Tracer, pullback: (Tracer) -> Tracer) {
        let result = self.sum()
        return (
            result,
            { upstream in
                // Gradient of sum is broadcast of upstream to original shape
                // upstream is scalar, broadcast to self.shape
                return Tracer(value: 1.0, shape: self.shape, dtype: self.dtype) * upstream
            }
        )
    }
}
