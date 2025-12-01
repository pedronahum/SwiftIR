// Vmap.swift - Automatic Vectorization (JAX-like vmap)
// Copyright 2024 SwiftIR Project
//
// This file implements vmap (vectorized map), which automatically transforms
// a function operating on single examples into one operating on batches.
//
// Based on JAX's vmap: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html

import Foundation
import _Differentiation

// MARK: - VmapAxes Configuration

/// Specifies batch axes for vmap inputs
public struct VmapAxes: ExpressibleByIntegerLiteral, ExpressibleByNilLiteral, ExpressibleByArrayLiteral {
    public let axes: [Int?]

    /// Single axis for single input
    public init(_ axis: Int?) {
        self.axes = [axis]
    }

    /// Multiple axes for tuple inputs (variadic)
    public init(_ axes: Int?...) {
        self.axes = axes
    }

    /// Array initializer
    public init(axes: [Int?]) {
        self.axes = axes
    }

    /// Integer literal: vmap(f, inAxes: 0)
    public init(integerLiteral value: Int) {
        self.axes = [value]
    }

    /// Nil literal: vmap(f, inAxes: nil)
    public init(nilLiteral: ()) {
        self.axes = [nil]
    }

    /// Array literal: vmap(f, inAxes: [0, nil, 1])
    public init(arrayLiteral elements: Int?...) {
        self.axes = elements
    }
}

// MARK: - Batch Tracer

/// Tracks batch dimension information during vmap tracing
public struct BatchTracer {
    /// The underlying tracer
    public var tracer: DifferentiableTracer

    /// Which dimension is the batch dimension (nil if not batched/broadcast)
    public var batchAxis: Int?

    /// The batch size (nil if not batched)
    public var batchSize: Int?

    public init(tracer: DifferentiableTracer, batchAxis: Int?, batchSize: Int?) {
        self.tracer = tracer
        self.batchAxis = batchAxis
        self.batchSize = batchSize
    }
}

// MARK: - Vmap Context

/// Context for vmap transformation, supporting nested vmaps
public class VmapContext {
    /// Current batch size being traced
    public var batchSize: Int

    /// Current batch axis
    public var batchAxis: Int

    /// Stack for nested vmaps
    public nonisolated(unsafe) static var contextStack: [VmapContext] = []

    /// Get current vmap context (nil if not in vmap)
    public static var current: VmapContext? {
        contextStack.last
    }

    public init(batchSize: Int, batchAxis: Int = 0) {
        self.batchSize = batchSize
        self.batchAxis = batchAxis
    }
}

// MARK: - Primary vmap API

/// Vectorize a function over specified input axes
///
/// `diffVmap` automatically transforms a function that operates on single examples
/// into one that operates on batches. This is essential for efficient ML training.
///
/// **Usage Example - Simple Batching**:
/// ```swift
/// @differentiable(reverse)
/// func processOne(_ x: DifferentiableTracer) -> DifferentiableTracer {
///     return diffReLU(x)  // [features] -> [features]
/// }
///
/// let batchedProcess = diffVmap(processOne)
/// // Input: [batch, features] -> Output: [batch, features]
/// ```
///
/// **Usage Example - Matmul with Broadcasting**:
/// ```swift
/// @differentiable(reverse)
/// func forward(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
///     return diffMatmul(x, w)  // [in] @ [in, out] -> [out]
/// }
///
/// let batchedForward = diffVmap(forward, inAxes: VmapAxes(0, nil))
/// // x: [batch, in], w: [in, out] -> [batch, out]
/// // w is broadcast (shared) across batch
/// ```
///
/// - Parameters:
///   - fn: The function to vectorize
///   - inAxes: Which axis to batch over for each input (nil = broadcast)
///   - outAxes: Which axis the batch dimension appears in output (default: 0)
/// - Returns: A vectorized version of the function
public func diffVmap(
    _ fn: @escaping (DifferentiableTracer) -> DifferentiableTracer,
    inAxes: VmapAxes = VmapAxes(0),
    outAxes: Int = 0
) -> (DifferentiableTracer) -> DifferentiableTracer {

    return { (input: DifferentiableTracer) -> DifferentiableTracer in
        // 1. Determine batch size from input
        let batchAxis = inAxes.axes[0] ?? 0
        guard batchAxis < input.shape.count else {
            fatalError("diffVmap: Invalid batch axis \(batchAxis) for input with shape \(input.shape)")
        }
        let batchSize = input.shape[batchAxis]

        // 2. Create vmap context
        let context = VmapContext(batchSize: batchSize, batchAxis: batchAxis)
        VmapContext.contextStack.append(context)
        defer { VmapContext.contextStack.removeLast() }

        // 3. For simple element-wise operations, we can just pass through
        // The operations naturally handle the extra batch dimension
        let output = fn(input)

        // 4. Move batch axis to output position if needed
        if outAxes != batchAxis && output.shape.count > 1 {
            return diffMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

/// Vmap for two-input functions
///
/// **Usage Example**:
/// ```swift
/// @differentiable(reverse)
/// func dot(_ a: DifferentiableTracer, _ b: DifferentiableTracer) -> DifferentiableTracer {
///     return diffSum(a * b)
/// }
///
/// let batchedDot = diffVmap2(dot, inAxes: VmapAxes(0, 0))
/// // [batch, features] * [batch, features] -> [batch]
/// ```
public func diffVmap2(
    _ fn: @escaping (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer,
    inAxes: VmapAxes = VmapAxes(0, 0),
    outAxes: Int = 0
) -> (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer {

    return { (input1: DifferentiableTracer, input2: DifferentiableTracer) -> DifferentiableTracer in
        // Determine batch size
        let axis1 = inAxes.axes.count > 0 ? inAxes.axes[0] : 0
        let axis2 = inAxes.axes.count > 1 ? inAxes.axes[1] : nil

        var batchSize: Int?
        var batchAxis: Int = 0

        if let ax1 = axis1 {
            guard ax1 < input1.shape.count else {
                fatalError("diffVmap2: Invalid batch axis \(ax1) for input1 with shape \(input1.shape)")
            }
            batchSize = input1.shape[ax1]
            batchAxis = ax1
        }

        if let ax2 = axis2 {
            guard ax2 < input2.shape.count else {
                fatalError("diffVmap2: Invalid batch axis \(ax2) for input2 with shape \(input2.shape)")
            }
            if let bs = batchSize {
                guard input2.shape[ax2] == bs else {
                    fatalError("diffVmap2: Batch size mismatch. Input1 has batch size \(bs), input2 has \(input2.shape[ax2])")
                }
            } else {
                batchSize = input2.shape[ax2]
                batchAxis = ax2
            }
        }

        guard batchSize != nil else {
            fatalError("diffVmap2: At least one input must have a batch axis")
        }

        // Create context
        let context = VmapContext(batchSize: batchSize!, batchAxis: batchAxis)
        VmapContext.contextStack.append(context)
        defer { VmapContext.contextStack.removeLast() }

        // Handle broadcasting for non-batched inputs
        var processedInput1 = input1
        var processedInput2 = input2

        if axis1 == nil {
            // Broadcast input1 across batch dimension
            processedInput1 = diffBroadcastBatch(input1, batchSize: batchSize!, batchAxis: batchAxis)
        }

        if axis2 == nil {
            // Broadcast input2 across batch dimension
            processedInput2 = diffBroadcastBatch(input2, batchSize: batchSize!, batchAxis: batchAxis)
        }

        let output = fn(processedInput1, processedInput2)

        // Move batch axis if needed
        if outAxes != batchAxis && output.shape.count > 1 {
            return diffMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

/// Vmap for three-input functions (useful for MLP layers)
public func diffVmap3(
    _ fn: @escaping (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer,
    inAxes: VmapAxes = VmapAxes(0, nil, nil),
    outAxes: Int = 0
) -> (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer {

    return { (input1: DifferentiableTracer, input2: DifferentiableTracer, input3: DifferentiableTracer) -> DifferentiableTracer in
        let axis1 = inAxes.axes.count > 0 ? inAxes.axes[0] : 0
        let axis2 = inAxes.axes.count > 1 ? inAxes.axes[1] : nil
        let axis3 = inAxes.axes.count > 2 ? inAxes.axes[2] : nil

        // Find batch size from first batched input
        var batchSize: Int?
        var batchAxis: Int = 0

        if let ax1 = axis1 {
            batchSize = input1.shape[ax1]
            batchAxis = ax1
        }
        if batchSize == nil, let ax2 = axis2 {
            batchSize = input2.shape[ax2]
            batchAxis = ax2
        }
        if batchSize == nil, let ax3 = axis3 {
            batchSize = input3.shape[ax3]
            batchAxis = ax3
        }

        guard let bs = batchSize else {
            fatalError("diffVmap3: At least one input must have a batch axis")
        }

        let context = VmapContext(batchSize: bs, batchAxis: batchAxis)
        VmapContext.contextStack.append(context)
        defer { VmapContext.contextStack.removeLast() }

        // Handle broadcasting
        var p1 = input1, p2 = input2, p3 = input3
        if axis1 == nil { p1 = diffBroadcastBatch(input1, batchSize: bs, batchAxis: batchAxis) }
        if axis2 == nil { p2 = diffBroadcastBatch(input2, batchSize: bs, batchAxis: batchAxis) }
        if axis3 == nil { p3 = diffBroadcastBatch(input3, batchSize: bs, batchAxis: batchAxis) }

        let output = fn(p1, p2, p3)

        if outAxes != batchAxis && output.shape.count > 1 {
            return diffMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

// MARK: - Helper Functions

/// Move axis from one position to another
public func diffMoveAxis(_ x: DifferentiableTracer, from: Int, to: Int) -> DifferentiableTracer {
    guard from != to else { return x }
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffMoveAxis requires an active MLIRBuilder")
    }

    let rank = x.shape.count
    guard from >= 0 && from < rank && to >= 0 && to < rank else {
        fatalError("diffMoveAxis: Invalid axis. from=\(from), to=\(to), rank=\(rank)")
    }

    // Build permutation
    var perm = Array(0..<rank)
    perm.remove(at: from)
    perm.insert(from, at: to)

    // Actually we need the inverse logic for transpose
    var actualPerm = Array(0..<rank)
    let elem = actualPerm.remove(at: from)
    actualPerm.insert(elem, at: to)

    // Compute new shape
    var newShape = x.shape
    let dimVal = newShape.remove(at: from)
    newShape.insert(dimVal, at: to)

    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: newShape, dtype: x.dtype)
    let permStr = actualPerm.map(String.init).joined(separator: ", ")

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.transpose",
        operands: [x.irValue],
        attributes: ["permutation": "array<i64: \(permStr)>"],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: newShape, dtype: x.dtype)
}

/// Broadcast a tensor to add a batch dimension
public func diffBroadcastBatch(_ x: DifferentiableTracer, batchSize: Int, batchAxis: Int) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffBroadcastBatch requires an active MLIRBuilder")
    }

    // Insert batch dimension into shape
    var newShape = x.shape
    newShape.insert(batchSize, at: batchAxis)

    // Calculate broadcast dimensions (all original dims shift by 1 if batchAxis <= dim)
    var broadcastDims: [Int] = []
    for i in 0..<x.shape.count {
        if i < batchAxis {
            broadcastDims.append(i)
        } else {
            broadcastDims.append(i + 1)
        }
    }

    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: newShape, dtype: x.dtype)
    let dimsStr = broadcastDims.map(String.init).joined(separator: ", ")

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.broadcast_in_dim",
        operands: [x.irValue],
        attributes: ["broadcast_dimensions": "array<i64: \(dimsStr)>"],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: newShape, dtype: x.dtype)
}

/// Batched matrix multiplication (used internally by vmap)
///
/// Handles: [batch, m, k] @ [k, n] -> [batch, m, n]
/// Or: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
public func diffBatchedMatmul(_ a: DifferentiableTracer, _ b: DifferentiableTracer) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffBatchedMatmul requires an active MLIRBuilder")
    }

    // Determine if both have batch dim
    let aRank = a.shape.count
    let bRank = b.shape.count

    // Case: [batch, m, k] @ [k, n] -> [batch, m, n]
    if aRank == 3 && bRank == 2 {
        let batchSize = a.shape[0]
        let m = a.shape[1]
        let k = a.shape[2]
        let n = b.shape[1]

        guard b.shape[0] == k else {
            fatalError("diffBatchedMatmul: Shape mismatch. A has k=\(k), B has rows=\(b.shape[0])")
        }

        let resultShape = [batchSize, m, n]
        let resultSSA = builder.freshSSA()
        let resultType = tensorType(shape: resultShape, dtype: a.dtype)

        // Use dot_general for batched matmul
        builder.addOperation(MLIROperation(
            result: resultSSA,
            opName: "stablehlo.dot_general",
            operands: [a.irValue, b.irValue],
            attributes: [
                "dot_dimension_numbers": "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>"
            ],
            resultType: resultType
        ))

        return DifferentiableTracer(irValue: resultSSA, shape: resultShape, dtype: a.dtype)
    }

    // Case: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    if aRank == 3 && bRank == 3 {
        let batchSize = a.shape[0]
        let m = a.shape[1]
        let k = a.shape[2]
        let n = b.shape[2]

        guard a.shape[0] == b.shape[0] else {
            fatalError("diffBatchedMatmul: Batch size mismatch. A=\(a.shape[0]), B=\(b.shape[0])")
        }
        guard k == b.shape[1] else {
            fatalError("diffBatchedMatmul: Contracting dimension mismatch. A.k=\(k), B.k=\(b.shape[1])")
        }

        let resultShape = [batchSize, m, n]
        let resultSSA = builder.freshSSA()
        let resultType = tensorType(shape: resultShape, dtype: a.dtype)

        builder.addOperation(MLIROperation(
            result: resultSSA,
            opName: "stablehlo.dot_general",
            operands: [a.irValue, b.irValue],
            attributes: [
                "dot_dimension_numbers": "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>"
            ],
            resultType: resultType
        ))

        return DifferentiableTracer(irValue: resultSSA, shape: resultShape, dtype: a.dtype)
    }

    // Fallback to regular matmul for 2D case
    return diffMatmul(a, b)
}

// MARK: - Implementation Notes

/*
 VMAP IMPLEMENTATION STATUS:

 ✅ Implemented:
    - diffVmap() - Single-input vectorization
    - diffVmap2() - Two-input vectorization
    - diffVmap3() - Three-input vectorization
    - VmapAxes - Flexible axis specification
    - VmapContext - Nested vmap support
    - diffMoveAxis() - Axis transposition
    - diffBroadcastBatch() - Batch dimension broadcasting
    - diffBatchedMatmul() - Efficient batched matrix multiplication

 ⚠️ Current Limitations:
    - Element-wise ops work automatically (no explicit batching rules needed)
    - Reductions need manual axis adjustment for complex cases
    - No VJP for vmap itself yet (gradients work for vmapped functions)

 🔜 Future Enhancements:
    - Full VJP for vmap transformation
    - More batching rules for specialized ops
    - Automatic unbatching for outputs
    - Performance optimizations

 KEY INSIGHTS:
 - Most element-wise ops work naturally with extra dimensions
 - Matmul needs special handling via dot_general
 - Broadcasting handles non-batched inputs
 - Context stack enables nested vmaps
 */
