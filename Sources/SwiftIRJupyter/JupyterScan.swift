// JupyterScan.swift - Efficient Sequence Processing (JAX-like scan)
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// This file implements scan, a higher-level abstraction over while_loop that handles
// output accumulation automatically. Essential for RNNs, cumulative operations, and
// time series processing.
//
// Based on JAX's lax.scan: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html

import Foundation
import _Differentiation

// MARK: - Primary Scan API

/// Process a sequence with a carry state, accumulating outputs
///
/// `jScan` is a higher-level abstraction over `jWhileLoop4` that automatically
/// handles output accumulation. It's ideal for:
/// - RNN/LSTM implementations
/// - Cumulative operations (cumsum, cumprod)
/// - Time series processing
/// - Any sequential computation with state
///
/// **Usage Example - Cumulative Sum**:
/// ```swift
/// func cumsumStep(carry: JTracer, x: JTracer) -> (JTracer, JTracer) {
///     let newCarry = carry + x
///     return (newCarry, newCarry)
/// }
///
/// let xs = JTracer(values: [1, 2, 3, 4, 5], shape: JTensorShape([5]))
/// let initCarry = JTracer(value: 0, shape: .scalar)
/// let (final, cumsum) = jScan(cumsumStep, initCarry: initCarry, xs: xs)
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
public func jScan(
    _ fn: @escaping (JTracer, JTracer) -> (JTracer, JTracer),
    initCarry: JTracer,
    xs: JTracer,
    length: Int? = nil,
    reverse: Bool = false
) -> (JTracer, JTracer) {

    let builder = JTracerGraphBuilder.shared

    // Determine sequence length
    let seqLen: Int
    let dims = xs.shape.dimensions.compactMap { $0 }
    guard dims.count > 0 else {
        fatalError("jScan: Input must have at least 1 dimension (sequence dimension)")
    }
    seqLen = length ?? dims[0]

    guard seqLen > 0 else {
        fatalError("jScan requires non-empty sequence. Use length parameter for dynamic shapes.")
    }

    // Get feature shape (everything after sequence dimension)
    let featureShape: JTensorShape
    if dims.count == 1 {
        featureShape = .scalar
    } else {
        featureShape = JTensorShape(Array(dims.dropFirst()))
    }

    // Trace one step to determine output shape
    let sampleInputId = builder.getNextId()
    let sampleInput = JTracer(irValue: JMLIRValue(id: sampleInputId), shape: featureShape, dtype: xs.dtype, version: JTracer.incrementVersion())
    let (_, sampleOutput) = fn(initCarry, sampleInput)
    let outputShape = sampleOutput.shape

    // Create output accumulator with shape [seqLen, ...outputShape]
    let fullOutputShape: JTensorShape
    let outputDims = outputShape.dimensions.compactMap { $0 }
    if outputDims.isEmpty {
        fullOutputShape = JTensorShape([seqLen])
    } else {
        fullOutputShape = JTensorShape([seqLen] + outputDims)
    }

    let outputInit = JTracer.zeros(shape: fullOutputShape, dtype: sampleOutput.dtype)

    // Create iteration counter
    let iterInitValue = reverse ? Double(seqLen - 1) : 0.0
    let iterInit = JTracer(value: iterInitValue, shape: .scalar, dtype: .float32)
    let one = JTracer(value: 1.0, shape: .scalar, dtype: .float32)
    let limit = JTracer(value: Double(seqLen), shape: .scalar, dtype: .float32)
    let zero = JTracer(value: 0.0, shape: .scalar, dtype: .float32)

    // Build the while loop with 4-tuple state: (iter, carry, outputs, xs)
    let result = jWhileLoop4(
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
            let currentInput = jDynamicSliceAt(inputSeq, index: iter, featureShape: featureShape)

            // Run step function
            let (newCarry, stepOutput) = fn(carry, currentInput)

            // Update output accumulator using dynamic update slice
            let newOutputs = jDynamicUpdateSliceAt(outputs, value: stepOutput, index: iter, outputShape: outputShape)

            // Update index
            let negOne = JTracer(value: -1.0, shape: .scalar, dtype: .float32)
            let newIter = reverse ? iter + negOne : iter + one

            return (newIter, newCarry, newOutputs, inputSeq)
        }
    )

    return (result.1, result.2)
}

// MARK: - Dynamic Slice Operations

/// Dynamic slice at runtime index
public func jDynamicSliceAt(_ input: JTracer, index: JTracer, featureShape: JTensorShape) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let inputDims = input.shape.dimensions.compactMap { $0 }
    guard inputDims.count > 0 else {
        return input
    }

    // Create dynamic slice operation
    let featureDims = featureShape.dimensions.compactMap { $0 }
    let sliceSizes: [Int]
    if featureDims.isEmpty {
        sliceSizes = [1]
    } else {
        sliceSizes = [1] + featureDims
    }

    let resultId = builder.createDynamicSlice(
        input: input.valueId,
        startIndex: index.valueId,
        sliceSizes: sliceSizes,
        inputShape: input.shape,
        dtype: input.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: featureShape,
        dtype: input.dtype,
        version: JTracer.incrementVersion()
    )
}

/// Dynamic update slice at runtime index
public func jDynamicUpdateSliceAt(_ operand: JTracer, value: JTracer, index: JTracer, outputShape: JTensorShape) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let operandDims = operand.shape.dimensions.compactMap { $0 }
    guard operandDims.count > 0 else {
        return value
    }

    let resultId = builder.createDynamicUpdateSlice(
        operand: operand.valueId,
        update: value.valueId,
        startIndex: index.valueId,
        operandShape: operand.shape,
        updateShape: outputShape,
        dtype: operand.dtype
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: operand.shape,
        dtype: operand.dtype,
        version: JTracer.incrementVersion()
    )
}

// MARK: - Graph Builder Extensions for Dynamic Slice

extension JTracerGraphBuilder {
    /// Create a dynamic slice operation
    public func createDynamicSlice(
        input: UInt64,
        startIndex: UInt64,
        sliceSizes: [Int],
        inputShape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        // Calculate output shape (same as slice sizes but without leading 1)
        let outputShape: JTensorShape
        if sliceSizes.count == 1 {
            outputShape = .scalar
        } else {
            outputShape = JTensorShape(Array(sliceSizes.dropFirst()))
        }

        let op = JTracedOperation.dynamicSlice(
            id: id,
            input: input,
            startIndex: startIndex,
            sliceSizes: sliceSizes,
            shape: outputShape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }

    /// Create a dynamic update slice operation
    public func createDynamicUpdateSlice(
        operand: UInt64,
        update: UInt64,
        startIndex: UInt64,
        operandShape: JTensorShape,
        updateShape: JTensorShape,
        dtype: JDType
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.dynamicUpdateSlice(
            id: id,
            operand: operand,
            update: update,
            startIndex: startIndex,
            shape: operandShape,
            dtype: dtype
        )
        addOperation(op)
        return id
    }
}

// MARK: - Cumulative Operations (Built on Scan)

/// Cumulative sum along axis 0
///
/// Computes running sum: output[i] = sum(input[0:i+1])
///
/// **Example**:
/// ```swift
/// let x = JTracer(values: [1, 2, 3, 4, 5], shape: JTensorShape([5]))
/// let cumsum = jCumsum(x)  // [1, 3, 6, 10, 15]
/// ```
public func jCumsum(_ x: JTracer) -> JTracer {
    let dims = x.shape.dimensions.compactMap { $0 }
    let featureShape: JTensorShape
    if dims.count == 1 {
        featureShape = .scalar
    } else {
        featureShape = JTensorShape(Array(dims.dropFirst()))
    }

    let initZero = JTracer.zeros(shape: featureShape, dtype: x.dtype)

    func cumsumStep(carry: JTracer, input: JTracer) -> (JTracer, JTracer) {
        let newCarry = carry + input
        return (newCarry, newCarry)
    }

    let (_, cumsum) = jScan(cumsumStep, initCarry: initZero, xs: x)
    return cumsum
}

/// Cumulative product along axis 0
///
/// Computes running product: output[i] = product(input[0:i+1])
///
/// **Example**:
/// ```swift
/// let x = JTracer(values: [1, 2, 3, 4], shape: JTensorShape([4]))
/// let cumprod = jCumprod(x)  // [1, 2, 6, 24]
/// ```
public func jCumprod(_ x: JTracer) -> JTracer {
    let dims = x.shape.dimensions.compactMap { $0 }
    let featureShape: JTensorShape
    if dims.count == 1 {
        featureShape = .scalar
    } else {
        featureShape = JTensorShape(Array(dims.dropFirst()))
    }

    let initOne = JTracer.ones(shape: featureShape, dtype: x.dtype)

    func cumprodStep(carry: JTracer, input: JTracer) -> (JTracer, JTracer) {
        let newCarry = carry * input
        return (newCarry, newCarry)
    }

    let (_, cumprod) = jScan(cumprodStep, initCarry: initOne, xs: x)
    return cumprod
}

// MARK: - Simplified Scan (Output = Carry)

/// Simplified scan where output equals the carry at each step
///
/// Useful when you want to record all intermediate states.
///
/// **Example**:
/// ```swift
/// func step(state: JTracer, input: JTracer) -> JTracer {
///     return state.tanh() + input
/// }
///
/// let (finalState, allStates) = jScanSimple(step, initState: initState, xs: inputs)
/// ```
public func jScanSimple(
    _ fn: @escaping (JTracer, JTracer) -> JTracer,
    initState: JTracer,
    xs: JTracer,
    reverse: Bool = false
) -> (JTracer, JTracer) {
    func wrappedFn(carry: JTracer, input: JTracer) -> (JTracer, JTracer) {
        let newState = fn(carry, input)
        return (newState, newState)
    }

    return jScan(wrappedFn, initCarry: initState, xs: xs, reverse: reverse)
}

// MARK: - Implementation Notes

/*
 JUPYTER SCAN IMPLEMENTATION STATUS:

 ✅ Core Implementation:
    - jScan() API matching JAX lax.scan semantics
    - Uses jWhileLoop4 for efficient StableHLO generation
    - Dynamic slice/update slice for sequence indexing
    - Forward and reverse scan support

 ✅ Cumulative Operations:
    - jCumsum - cumulative sum
    - jCumprod - cumulative product

 ⚠️ Current Limitations:
    - Single JTracer carry (no struct carries yet)
    - No gradient support (pure Swift version)
    - No unroll optimization yet

 🔜 Future Enhancements:
    - Struct carries (LSTMState, etc.)
    - VJP implementation for gradients
    - Unroll optimization for short sequences
    - Bidirectional helpers

 KEY INSIGHTS:
 - Scan builds on top of while loop infrastructure
 - Uses dynamic_slice for reading from sequence
 - Uses dynamic_update_slice for writing to output accumulator
 - Pure Swift implementation works in Jupyter/Colab
 */
