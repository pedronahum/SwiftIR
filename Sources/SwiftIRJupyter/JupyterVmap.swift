// JupyterVmap.swift - Automatic Vectorization (JAX-like vmap)
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// This file implements vmap (vectorized map), which automatically transforms
// a function operating on single examples into one operating on batches.
//
// Based on JAX's vmap: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html

import Foundation
import _Differentiation

// MARK: - JVmapAxes Configuration

/// Specifies batch axes for vmap inputs
public struct JVmapAxes: ExpressibleByIntegerLiteral, ExpressibleByNilLiteral, ExpressibleByArrayLiteral {
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

    /// Integer literal: jVmap(f, inAxes: 0)
    public init(integerLiteral value: Int) {
        self.axes = [value]
    }

    /// Nil literal: jVmap(f, inAxes: nil)
    public init(nilLiteral: ()) {
        self.axes = [nil]
    }

    /// Array literal: jVmap(f, inAxes: [0, nil, 1])
    public init(arrayLiteral elements: Int?...) {
        self.axes = elements
    }
}

// MARK: - JVmap Context

/// Context for vmap transformation, supporting nested vmaps
public class JVmapContext {
    /// Current batch size being traced
    public var batchSize: Int

    /// Current batch axis
    public var batchAxis: Int

    /// Stack for nested vmaps
    public nonisolated(unsafe) static var contextStack: [JVmapContext] = []

    /// Get current vmap context (nil if not in vmap)
    public static var current: JVmapContext? {
        contextStack.last
    }

    public init(batchSize: Int, batchAxis: Int = 0) {
        self.batchSize = batchSize
        self.batchAxis = batchAxis
    }
}

// MARK: - Primary jVmap API

/// Vectorize a function over specified input axes
///
/// `jVmap` automatically transforms a function that operates on single examples
/// into one that operates on batches. This is essential for efficient ML training.
///
/// **Usage Example - Simple Batching**:
/// ```swift
/// func processOne(_ x: JTracer) -> JTracer {
///     return jReLU(x)  // [features] -> [features]
/// }
///
/// let batchedProcess = jVmap(processOne)
/// // Input: [batch, features] -> Output: [batch, features]
/// ```
///
/// **Usage Example - Matmul with Broadcasting**:
/// ```swift
/// func forward(_ x: JTracer, _ w: JTracer) -> JTracer {
///     return jMatmul(x, w)  // [in] @ [in, out] -> [out]
/// }
///
/// let batchedForward = jVmap2(forward, inAxes: JVmapAxes(0, nil))
/// // x: [batch, in], w: [in, out] -> [batch, out]
/// // w is broadcast (shared) across batch
/// ```
///
/// - Parameters:
///   - fn: The function to vectorize
///   - inAxes: Which axis to batch over for each input (nil = broadcast)
///   - outAxes: Which axis the batch dimension appears in output (default: 0)
/// - Returns: A vectorized version of the function
public func jVmap(
    _ fn: @escaping (JTracer) -> JTracer,
    inAxes: JVmapAxes = JVmapAxes(0),
    outAxes: Int = 0
) -> (JTracer) -> JTracer {

    return { (input: JTracer) -> JTracer in
        // 1. Determine batch size from input
        let batchAxis = inAxes.axes[0] ?? 0
        let dims = input.shape.dimensions.compactMap { $0 }

        guard batchAxis < dims.count else {
            fatalError("jVmap: Invalid batch axis \(batchAxis) for input with shape \(input.shape)")
        }
        let batchSize = dims[batchAxis]

        // 2. Create vmap context
        let context = JVmapContext(batchSize: batchSize, batchAxis: batchAxis)
        JVmapContext.contextStack.append(context)
        defer { JVmapContext.contextStack.removeLast() }

        // 3. For element-wise operations, we can pass through
        // The operations naturally handle the extra batch dimension
        let output = fn(input)

        // 4. Move batch axis to output position if needed
        let outDims = output.shape.dimensions.compactMap { $0 }
        if outAxes != batchAxis && outDims.count > 1 {
            return jMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

/// Vmap for two-input functions
///
/// **Usage Example**:
/// ```swift
/// func dot(_ a: JTracer, _ b: JTracer) -> JTracer {
///     return jSum(a * b)
/// }
///
/// let batchedDot = jVmap2(dot, inAxes: JVmapAxes(0, 0))
/// // [batch, features] * [batch, features] -> [batch]
/// ```
public func jVmap2(
    _ fn: @escaping (JTracer, JTracer) -> JTracer,
    inAxes: JVmapAxes = JVmapAxes(0, 0),
    outAxes: Int = 0
) -> (JTracer, JTracer) -> JTracer {

    return { (input1: JTracer, input2: JTracer) -> JTracer in
        let dims1 = input1.shape.dimensions.compactMap { $0 }
        let dims2 = input2.shape.dimensions.compactMap { $0 }

        let axis1 = inAxes.axes.count > 0 ? inAxes.axes[0] : 0
        let axis2 = inAxes.axes.count > 1 ? inAxes.axes[1] : nil

        var batchSize: Int?
        var batchAxis: Int = 0

        if let ax1 = axis1 {
            guard ax1 < dims1.count else {
                fatalError("jVmap2: Invalid batch axis \(ax1) for input1 with shape \(input1.shape)")
            }
            batchSize = dims1[ax1]
            batchAxis = ax1
        }

        if let ax2 = axis2 {
            guard ax2 < dims2.count else {
                fatalError("jVmap2: Invalid batch axis \(ax2) for input2 with shape \(input2.shape)")
            }
            if let bs = batchSize {
                guard dims2[ax2] == bs else {
                    fatalError("jVmap2: Batch size mismatch. Input1 has batch size \(bs), input2 has \(dims2[ax2])")
                }
            } else {
                batchSize = dims2[ax2]
                batchAxis = ax2
            }
        }

        guard let bs = batchSize else {
            fatalError("jVmap2: At least one input must have a batch axis")
        }

        // Create context
        let context = JVmapContext(batchSize: bs, batchAxis: batchAxis)
        JVmapContext.contextStack.append(context)
        defer { JVmapContext.contextStack.removeLast() }

        // Handle broadcasting for non-batched inputs
        var processedInput1 = input1
        var processedInput2 = input2

        if axis1 == nil {
            processedInput1 = jBroadcastBatch(input1, batchSize: bs, batchAxis: batchAxis)
        }

        if axis2 == nil {
            processedInput2 = jBroadcastBatch(input2, batchSize: bs, batchAxis: batchAxis)
        }

        let output = fn(processedInput1, processedInput2)

        // Move batch axis if needed
        let outDims = output.shape.dimensions.compactMap { $0 }
        if outAxes != batchAxis && outDims.count > 1 {
            return jMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

/// Vmap for three-input functions (useful for MLP layers)
public func jVmap3(
    _ fn: @escaping (JTracer, JTracer, JTracer) -> JTracer,
    inAxes: JVmapAxes = JVmapAxes(0, nil, nil),
    outAxes: Int = 0
) -> (JTracer, JTracer, JTracer) -> JTracer {

    return { (input1: JTracer, input2: JTracer, input3: JTracer) -> JTracer in
        let dims1 = input1.shape.dimensions.compactMap { $0 }
        let dims2 = input2.shape.dimensions.compactMap { $0 }
        let dims3 = input3.shape.dimensions.compactMap { $0 }

        let axis1 = inAxes.axes.count > 0 ? inAxes.axes[0] : 0
        let axis2 = inAxes.axes.count > 1 ? inAxes.axes[1] : nil
        let axis3 = inAxes.axes.count > 2 ? inAxes.axes[2] : nil

        // Find batch size from first batched input
        var batchSize: Int?
        var batchAxis: Int = 0

        if let ax1 = axis1, ax1 < dims1.count {
            batchSize = dims1[ax1]
            batchAxis = ax1
        }
        if batchSize == nil, let ax2 = axis2, ax2 < dims2.count {
            batchSize = dims2[ax2]
            batchAxis = ax2
        }
        if batchSize == nil, let ax3 = axis3, ax3 < dims3.count {
            batchSize = dims3[ax3]
            batchAxis = ax3
        }

        guard let bs = batchSize else {
            fatalError("jVmap3: At least one input must have a batch axis")
        }

        let context = JVmapContext(batchSize: bs, batchAxis: batchAxis)
        JVmapContext.contextStack.append(context)
        defer { JVmapContext.contextStack.removeLast() }

        // Handle broadcasting
        var p1 = input1, p2 = input2, p3 = input3
        if axis1 == nil { p1 = jBroadcastBatch(input1, batchSize: bs, batchAxis: batchAxis) }
        if axis2 == nil { p2 = jBroadcastBatch(input2, batchSize: bs, batchAxis: batchAxis) }
        if axis3 == nil { p3 = jBroadcastBatch(input3, batchSize: bs, batchAxis: batchAxis) }

        let output = fn(p1, p2, p3)

        let outDims = output.shape.dimensions.compactMap { $0 }
        if outAxes != batchAxis && outDims.count > 1 {
            return jMoveAxis(output, from: batchAxis, to: outAxes)
        }

        return output
    }
}

// MARK: - Helper Functions

/// Move axis from one position to another
public func jMoveAxis(_ x: JTracer, from: Int, to: Int) -> JTracer {
    guard from != to else { return x }

    let builder = JTracerGraphBuilder.shared
    let dims = x.shape.dimensions.compactMap { $0 }
    let rank = dims.count

    guard from >= 0 && from < rank && to >= 0 && to < rank else {
        fatalError("jMoveAxis: Invalid axis. from=\(from), to=\(to), rank=\(rank)")
    }

    // Build permutation
    var actualPerm = Array(0..<rank)
    let elem = actualPerm.remove(at: from)
    actualPerm.insert(elem, at: to)

    // Compute new shape
    var newDims = dims
    let dimVal = newDims.remove(at: from)
    newDims.insert(dimVal, at: to)
    let newShape = JTensorShape(newDims)

    let resultId = builder.createTranspose(
        input: x.valueId,
        permutation: actualPerm,
        inputShape: x.shape,
        outputShape: newShape,
        dtype: x.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: newShape,
        dtype: x.dtype,
        version: JTracer.incrementVersion()
    )
}

/// Broadcast a tensor to add a batch dimension
public func jBroadcastBatch(_ x: JTracer, batchSize: Int, batchAxis: Int) -> JTracer {
    let builder = JTracerGraphBuilder.shared
    let dims = x.shape.dimensions.compactMap { $0 }

    // Insert batch dimension into shape
    var newDims = dims
    newDims.insert(batchSize, at: batchAxis)
    let newShape = JTensorShape(newDims)

    // Calculate broadcast dimensions
    var broadcastDims: [Int] = []
    for i in 0..<dims.count {
        if i < batchAxis {
            broadcastDims.append(i)
        } else {
            broadcastDims.append(i + 1)
        }
    }

    let resultId = builder.createBroadcastInDim(
        input: x.valueId,
        broadcastDimensions: broadcastDims,
        inputShape: x.shape,
        outputShape: newShape,
        dtype: x.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: newShape,
        dtype: x.dtype,
        version: JTracer.incrementVersion()
    )
}

/// Batched matrix multiplication
///
/// Handles: [batch, m, k] @ [k, n] -> [batch, m, n]
/// Or: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
public func jBatchedMatmul(_ a: JTracer, _ b: JTracer) -> JTracer {
    let builder = JTracerGraphBuilder.shared
    let aDims = a.shape.dimensions.compactMap { $0 }
    let bDims = b.shape.dimensions.compactMap { $0 }

    // Case: [batch, m, k] @ [k, n] -> [batch, m, n]
    if aDims.count == 3 && bDims.count == 2 {
        let batchSize = aDims[0]
        let m = aDims[1]
        let k = aDims[2]
        let n = bDims[1]

        guard bDims[0] == k else {
            fatalError("jBatchedMatmul: Shape mismatch. A has k=\(k), B has rows=\(bDims[0])")
        }

        let resultShape = JTensorShape([batchSize, m, n])

        let resultId = builder.createDotGeneral(
            lhs: a.valueId,
            rhs: b.valueId,
            lhsBatchingDims: [0],
            rhsBatchingDims: [],
            lhsContractingDims: [2],
            rhsContractingDims: [0],
            resultShape: resultShape,
            dtype: a.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: resultId),
            shape: resultShape,
            dtype: a.dtype,
            version: JTracer.incrementVersion()
        )
    }

    // Case: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    if aDims.count == 3 && bDims.count == 3 {
        let batchSize = aDims[0]
        let m = aDims[1]
        let k = aDims[2]
        let n = bDims[2]

        guard aDims[0] == bDims[0] else {
            fatalError("jBatchedMatmul: Batch size mismatch. A=\(aDims[0]), B=\(bDims[0])")
        }
        guard k == bDims[1] else {
            fatalError("jBatchedMatmul: Contracting dimension mismatch. A.k=\(k), B.k=\(bDims[1])")
        }

        let resultShape = JTensorShape([batchSize, m, n])

        let resultId = builder.createDotGeneral(
            lhs: a.valueId,
            rhs: b.valueId,
            lhsBatchingDims: [0],
            rhsBatchingDims: [0],
            lhsContractingDims: [2],
            rhsContractingDims: [1],
            resultShape: resultShape,
            dtype: a.dtype
        )

        return JTracer(
            irValue: JMLIRValue(id: resultId),
            shape: resultShape,
            dtype: a.dtype,
            version: JTracer.incrementVersion()
        )
    }

    // Fallback to regular matmul for 2D case
    return a.matmul(b)
}

// MARK: - Graph Builder Extensions for Vmap

extension JTracerGraphBuilder {
    /// Create a transpose operation
    public func createTranspose(
        input: UInt64,
        permutation: [Int],
        inputShape: JTensorShape,
        outputShape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.transpose(
            id: id,
            input: input,
            permutation: permutation,
            inputShape: inputShape,
            outputShape: outputShape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }

    /// Create a dot_general operation (batched matmul)
    public func createDotGeneral(
        lhs: UInt64,
        rhs: UInt64,
        lhsBatchingDims: [Int],
        rhsBatchingDims: [Int],
        lhsContractingDims: [Int],
        rhsContractingDims: [Int],
        resultShape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.dotGeneral(
            id: id,
            lhs: lhs,
            rhs: rhs,
            lhsBatchingDims: lhsBatchingDims,
            rhsBatchingDims: rhsBatchingDims,
            lhsContractingDims: lhsContractingDims,
            rhsContractingDims: rhsContractingDims,
            resultShape: resultShape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }
}

// MARK: - Implementation Notes

/*
 JUPYTER VMAP IMPLEMENTATION STATUS:

 ✅ Implemented:
    - jVmap() - Single-input vectorization
    - jVmap2() - Two-input vectorization
    - jVmap3() - Three-input vectorization
    - JVmapAxes - Flexible axis specification
    - JVmapContext - Nested vmap support
    - jMoveAxis() - Axis transposition
    - jBroadcastBatch() - Batch dimension broadcasting
    - jBatchedMatmul() - Efficient batched matrix multiplication

 ⚠️ Current Limitations:
    - Element-wise ops work automatically (no explicit batching rules needed)
    - Reductions may need manual axis adjustment for complex cases
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
