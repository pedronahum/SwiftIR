// Scan.swift - Efficient Sequence Processing (JAX-like scan)
// Copyright 2024 SwiftIR Project
//
// This file implements scan, a higher-level abstraction over while_loop that handles
// output accumulation automatically. Essential for RNNs, cumulative operations, and
// time series processing.
//
// Based on JAX's lax.scan: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html

import Foundation
import _Differentiation

// MARK: - Scan Configuration

/// Configuration options for scan operation
public struct ScanConfig {
    /// Explicit sequence length (nil = infer from input)
    public var length: Int?

    /// Process in reverse order
    public var reverse: Bool

    /// Unroll factor (1 = fully looped, n = unroll n steps) - future optimization
    public var unroll: Int

    public init(length: Int? = nil, reverse: Bool = false, unroll: Int = 1) {
        self.length = length
        self.reverse = reverse
        self.unroll = unroll
    }
}

// MARK: - Primary Scan API (DifferentiableTracer version)

/// Process a sequence with a carry state, accumulating outputs
///
/// `diffScan` is a higher-level abstraction over `diffWhileLoop` that automatically
/// handles output accumulation. It's ideal for:
/// - RNN/LSTM implementations
/// - Cumulative operations (cumsum, cumprod)
/// - Time series processing
/// - Any sequential computation with state
///
/// **Usage Example - Cumulative Sum**:
/// ```swift
/// func cumsumStep(carry: DifferentiableTracer, x: DifferentiableTracer)
///     -> (DifferentiableTracer, DifferentiableTracer) {
///     let newCarry = carry + x
///     return (newCarry, newCarry)
/// }
///
/// let xs = createConstant([1, 2, 3, 4, 5], shape: [5], dtype: .float32)
/// let initCarry = createConstant(0, shape: [], dtype: .float32)
/// let (final, cumsum) = diffScan(cumsumStep, initCarry: initCarry, xs: xs)
/// // cumsum = [1, 3, 6, 10, 15]
/// ```
///
/// - Parameters:
///   - fn: Step function (carry, input) -> (newCarry, output)
///   - initCarry: Initial carry state
///   - xs: Input sequence with shape [seqLen, ...features]
///   - length: Optional explicit sequence length
///   - reverse: If true, process sequence in reverse order
/// - Returns: (finalCarry, outputs) where outputs has shape [seqLen, ...outputFeatures]
public func diffScan(
    _ fn: @escaping (DifferentiableTracer, DifferentiableTracer) -> (DifferentiableTracer, DifferentiableTracer),
    initCarry: DifferentiableTracer,
    xs: DifferentiableTracer,
    length: Int? = nil,
    reverse: Bool = false
) -> (DifferentiableTracer, DifferentiableTracer) {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffScan requires an active MLIRBuilder (must be called during tracing)")
    }

    // Determine sequence length
    let seqLen = length ?? xs.shape[0]
    guard seqLen > 0 else {
        fatalError("diffScan requires non-empty sequence. Use length parameter for dynamic shapes.")
    }

    // Get feature shape (everything after sequence dimension)
    let featureShape = Array(xs.shape.dropFirst())

    // Trace one step to determine output shape
    let sampleInput = sliceAt(xs, index: 0)
    let (_, sampleOutput) = fn(initCarry, sampleInput)
    let outputShape = sampleOutput.shape

    // Create output accumulator with shape [seqLen, ...outputShape]
    let fullOutputShape = [seqLen] + outputShape
    let outputInit = scanCreateZeros(shape: fullOutputShape, dtype: sampleOutput.dtype)

    // Create iteration counter
    let iterInit = createConstant(reverse ? Float(seqLen - 1) : 0, shape: [], dtype: .float32)
    let one = createConstant(1.0, shape: [], dtype: .float32)
    let limit = createConstant(Float(seqLen), shape: [], dtype: .float32)
    let zero = createConstant(0.0, shape: [], dtype: .float32)
    let negOne = createConstant(-1.0, shape: [], dtype: .float32)

    // Build the while loop with 4-tuple state: (iter, carry, outputs, xs)
    // Note: xs is passed through to maintain reference within regions
    let result = diffWhileLoop(
        initial: (iterInit, initCarry, outputInit, xs),
        condition: { state in
            if reverse {
                return state.0 >= zero
            } else {
                return state.0 < limit
            }
        },
        body: { state in
            let (iter, carry, outputs, inputSeq) = state

            // Get input at current index using dynamic slice
            let currentInput = dynamicSliceAt(inputSeq, index: iter, featureShape: featureShape)

            // Run step function
            let (newCarry, stepOutput) = fn(carry, currentInput)

            // Update output accumulator using dynamic update slice
            let newOutputs = dynamicUpdateSliceAt(outputs, value: stepOutput, index: iter, outputShape: outputShape)

            // Update index
            let newIter = reverse ? iter + negOne : iter + one

            return (newIter, newCarry, newOutputs, inputSeq)
        }
    )

    return (result.1, result.2)
}

// MARK: - Helper Functions

/// Create a zero tensor with given shape and dtype
private func scanCreateZeros(shape: [Int], dtype: DType) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("scanCreateZeros requires an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()
    let typeStr = tensorType(shape: shape, dtype: dtype)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.constant",
        operands: [],
        attributes: ["value": "dense<0.0> : \(typeStr)"],
        resultType: typeStr
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: shape, dtype: dtype)
}

/// Create a ones tensor with given shape and dtype
private func scanCreateOnes(shape: [Int], dtype: DType) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("scanCreateOnes requires an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()
    let typeStr = tensorType(shape: shape, dtype: dtype)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.constant",
        operands: [],
        attributes: ["value": "dense<1.0> : \(typeStr)"],
        resultType: typeStr
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: shape, dtype: dtype)
}

/// Static slice at index 0 for tracing output shape
private func sliceAt(_ input: DifferentiableTracer, index: Int) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("sliceAt requires an active MLIRBuilder")
    }

    let rank = input.shape.count
    guard rank > 0 else {
        return input
    }

    // Static slice: start at [index, 0, 0, ...], size [1, dim1, dim2, ...]
    let starts = [index] + Array(repeating: 0, count: rank - 1)
    let limits = [index + 1] + Array(input.shape.dropFirst())
    let strides = Array(repeating: 1, count: rank)

    let resultSSA = builder.freshSSA()
    let slicedShape = Array(input.shape.dropFirst())  // Remove first dimension
    let resultType = tensorType(shape: slicedShape, dtype: input.dtype)

    // First slice to get [1, features...]
    let intermediateSSA = builder.freshSSA()
    let intermediateType = tensorType(shape: [1] + slicedShape, dtype: input.dtype)

    builder.addOperation(MLIROperation(
        result: intermediateSSA,
        opName: "stablehlo.slice",
        operands: [input.irValue],
        attributes: [
            "start_indices": "array<i64: \(starts.map(String.init).joined(separator: ", "))>",
            "limit_indices": "array<i64: \(limits.map(String.init).joined(separator: ", "))>",
            "strides": "array<i64: \(strides.map(String.init).joined(separator: ", "))>"
        ],
        resultType: intermediateType
    ))

    // Then reshape to remove the leading 1 dimension
    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.reshape",
        operands: [intermediateSSA],
        attributes: [:],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: slicedShape, dtype: input.dtype)
}

/// Dynamic slice at runtime index
private func dynamicSliceAt(_ input: DifferentiableTracer, index: DifferentiableTracer, featureShape: [Int]) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("dynamicSliceAt requires an active MLIRBuilder")
    }

    let rank = input.shape.count
    guard rank > 0 else {
        return input
    }

    // Convert float index to integer
    let indexInt = floatToInt(index)

    // Create start indices: [index, 0, 0, ...]
    var startIndices = [indexInt.irValue]
    for _ in 1..<rank {
        let zeroSSA = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: zeroSSA,
            opName: "stablehlo.constant",
            operands: [],
            attributes: ["value": "dense<0> : tensor<i32>"],
            resultType: "tensor<i32>"
        ))
        startIndices.append(zeroSSA)
    }

    // Slice sizes: [1, feature_dims...]
    let sliceSizes = [1] + featureShape

    let intermediateSSA = builder.freshSSA()
    let intermediateShape = sliceSizes
    let intermediateType = tensorType(shape: intermediateShape, dtype: input.dtype)

    builder.addOperation(MLIROperation(
        result: intermediateSSA,
        opName: "stablehlo.dynamic_slice",
        operands: [input.irValue] + startIndices,
        attributes: [
            "slice_sizes": "array<i64: \(sliceSizes.map(String.init).joined(separator: ", "))>"
        ],
        resultType: intermediateType
    ))

    // Reshape to remove leading 1 dimension
    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: featureShape, dtype: input.dtype)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.reshape",
        operands: [intermediateSSA],
        attributes: [:],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: featureShape, dtype: input.dtype)
}

/// Dynamic update slice at runtime index
private func dynamicUpdateSliceAt(_ operand: DifferentiableTracer, value: DifferentiableTracer, index: DifferentiableTracer, outputShape: [Int]) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("dynamicUpdateSliceAt requires an active MLIRBuilder")
    }

    let rank = operand.shape.count
    guard rank > 0 else {
        return value
    }

    // Expand value to have leading 1 dimension: [outputShape...] -> [1, outputShape...]
    let expandedSSA = builder.freshSSA()
    let expandedShape = [1] + outputShape
    let expandedType = tensorType(shape: expandedShape, dtype: value.dtype)

    builder.addOperation(MLIROperation(
        result: expandedSSA,
        opName: "stablehlo.reshape",
        operands: [value.irValue],
        attributes: [:],
        resultType: expandedType
    ))

    // Convert float index to integer
    let indexInt = floatToInt(index)

    // Create start indices: [index, 0, 0, ...]
    var startIndices = [indexInt.irValue]
    for _ in 1..<rank {
        let zeroSSA = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: zeroSSA,
            opName: "stablehlo.constant",
            operands: [],
            attributes: ["value": "dense<0> : tensor<i32>"],
            resultType: "tensor<i32>"
        ))
        startIndices.append(zeroSSA)
    }

    let resultSSA = builder.freshSSA()
    let resultType = tensorType(shape: operand.shape, dtype: operand.dtype)

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.dynamic_update_slice",
        operands: [operand.irValue, expandedSSA] + startIndices,
        attributes: [:],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: operand.shape, dtype: operand.dtype)
}

/// Convert float scalar to int32
private func floatToInt(_ x: DifferentiableTracer) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("floatToInt requires an active MLIRBuilder")
    }

    let resultSSA = builder.freshSSA()

    builder.addOperation(MLIROperation(
        result: resultSSA,
        opName: "stablehlo.convert",
        operands: [x.irValue],
        attributes: [:],
        resultType: "tensor<i32>"
    ))

    return DifferentiableTracer(irValue: resultSSA, shape: [], dtype: .int32)
}

// MARK: - Cumulative Operations (Built on Scan)

/// Cumulative sum along axis 0
///
/// Computes running sum: output[i] = sum(input[0:i+1])
///
/// **Example**:
/// ```swift
/// let x = createConstant([1, 2, 3, 4, 5], shape: [5], dtype: .float32)
/// let cumsum = diffCumsum(x)  // [1, 3, 6, 10, 15]
/// ```
public func diffCumsum(_ x: DifferentiableTracer) -> DifferentiableTracer {
    let featureShape = Array(x.shape.dropFirst())
    let initZero = scanCreateZeros(shape: featureShape, dtype: x.dtype)

    func cumsumStep(carry: DifferentiableTracer, input: DifferentiableTracer) -> (DifferentiableTracer, DifferentiableTracer) {
        let newCarry = carry + input
        return (newCarry, newCarry)
    }

    let (_, cumsum) = diffScan(cumsumStep, initCarry: initZero, xs: x)
    return cumsum
}

/// Cumulative product along axis 0
///
/// Computes running product: output[i] = product(input[0:i+1])
///
/// **Example**:
/// ```swift
/// let x = createConstant([1, 2, 3, 4], shape: [4], dtype: .float32)
/// let cumprod = diffCumprod(x)  // [1, 2, 6, 24]
/// ```
public func diffCumprod(_ x: DifferentiableTracer) -> DifferentiableTracer {
    let featureShape = Array(x.shape.dropFirst())
    let initOne = scanCreateOnes(shape: featureShape, dtype: x.dtype)

    func cumprodStep(carry: DifferentiableTracer, input: DifferentiableTracer) -> (DifferentiableTracer, DifferentiableTracer) {
        let newCarry = carry * input
        return (newCarry, newCarry)
    }

    let (_, cumprod) = diffScan(cumprodStep, initCarry: initOne, xs: x)
    return cumprod
}

// MARK: - Simplified Scan (Output = Carry)

/// Simplified scan where output equals the carry at each step
///
/// Useful when you want to record all intermediate states.
///
/// **Example**:
/// ```swift
/// func step(state: DifferentiableTracer, input: DifferentiableTracer) -> DifferentiableTracer {
///     return diffTanh(state + input)
/// }
///
/// let (finalState, allStates) = diffScanSimple(step, initState: initState, xs: inputs)
/// ```
public func diffScanSimple(
    _ fn: @escaping (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer,
    initState: DifferentiableTracer,
    xs: DifferentiableTracer,
    reverse: Bool = false
) -> (DifferentiableTracer, DifferentiableTracer) {
    func wrappedFn(carry: DifferentiableTracer, input: DifferentiableTracer) -> (DifferentiableTracer, DifferentiableTracer) {
        let newState = fn(carry, input)
        return (newState, newState)
    }

    return diffScan(wrappedFn, initCarry: initState, xs: xs, reverse: reverse)
}

// MARK: - Implementation Notes

/*
 SCAN IMPLEMENTATION STATUS:

 ✅ Core Implementation:
    - diffScan() API matching JAX lax.scan semantics
    - Uses diffWhileLoop for efficient StableHLO generation
    - Dynamic slice/update slice for sequence indexing
    - Forward and reverse scan support

 ✅ Cumulative Operations:
    - diffCumsum - cumulative sum
    - diffCumprod - cumulative product

 ⚠️ Current Limitations:
    - Single DifferentiableTracer carry (no struct carries yet)
    - No VJP implementation (gradient not differentiable through scan)
    - No unroll optimization yet

 🔜 Future Enhancements:
    - VJP for gradient support
    - Struct carries (LSTMState, etc.)
    - Unroll optimization for short sequences
    - Checkpointing for very long sequences
    - Bidirectional helpers

 KEY INSIGHTS:
 - Scan builds on top of while loop infrastructure
 - Uses dynamic_slice for reading from sequence
 - Uses dynamic_update_slice for writing to output accumulator
 - XLA optimizes the resulting StableHLO for efficient execution
 */
