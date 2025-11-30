# SCAN - Efficient Sequence Processing Specification

## Overview

`scan` provides an ergonomic API for sequential operations like RNNs, time series processing, and cumulative computations. It's a higher-level abstraction over `while_loop` that handles output accumulation automatically.

## JAX Reference

```python
# JAX scan usage
def rnn_step(carry, x):
    new_carry = jnp.tanh(carry @ W_h + x @ W_x)
    output = new_carry @ W_o
    return new_carry, output

final_carry, outputs = jax.lax.scan(rnn_step, init_carry, inputs)
# inputs: [seq_len, features]
# outputs: [seq_len, output_features]
```

## SwiftIR API Specification

### Primary API

```swift
/// Process a sequence with a carry state, accumulating outputs
/// - Parameters:
///   - fn: Step function (carry, input) -> (newCarry, output)
///   - init: Initial carry state
///   - xs: Input sequence [seqLen, ...features]
///   - length: Optional explicit sequence length (for dynamic shapes)
///   - reverse: If true, process sequence in reverse order
///   - unroll: Number of steps to unroll (default: 1 = fully loop)
/// - Returns: (finalCarry, outputs) where outputs is [seqLen, ...outputFeatures]
@differentiable(reverse)
public func diffScan<Carry: Differentiable, Input: Differentiable, Output: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (Carry, Input) -> (Carry, Output),
    init: Carry,
    xs: Input,
    length: Int? = nil,
    reverse: Bool = false,
    unroll: Int = 1
) -> (Carry, Output)
```

### Simplified API (Output = Carry)

```swift
/// Simplified scan where output equals the carry at each step
@differentiable(reverse)
public func diffScanSimple<State: Differentiable, Input: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (State, Input) -> State,
    init: State,
    xs: Input,
    reverse: Bool = false
) -> (State, State)  // (finalState, allStates)
```

### Cumulative Operations

```swift
/// Cumulative sum along axis 0
@differentiable(reverse)
public func diffCumsum(_ x: DifferentiableTracer) -> DifferentiableTracer

/// Cumulative product along axis 0
@differentiable(reverse)
public func diffCumprod(_ x: DifferentiableTracer) -> DifferentiableTracer

/// Cumulative max along axis 0
@differentiable(reverse)
public func diffCummax(_ x: DifferentiableTracer) -> DifferentiableTracer
```

### Usage Examples

#### Example 1: Simple RNN
```swift
// RNN cell
@differentiable(reverse)
func rnnStep(
    carry: DifferentiableTracer,
    x: DifferentiableTracer
) -> (DifferentiableTracer, DifferentiableTracer) {
    // carry: [hiddenSize], x: [inputSize]
    let newCarry = diffTanh(
        diffMatmul(carry, Wh) + diffMatmul(x, Wx) + bh
    )
    let output = diffMatmul(newCarry, Wo) + bo
    return (newCarry, output)
}

let initHidden = DifferentiableTracer.zeros(shape: [hiddenSize])
let inputs = DifferentiableTracer.placeholder(shape: [seqLen, inputSize])

let (finalHidden, outputs) = diffScan(rnnStep, init: initHidden, xs: inputs)
// finalHidden: [hiddenSize]
// outputs: [seqLen, outputSize]
```

#### Example 2: LSTM Cell
```swift
struct LSTMCarry: Differentiable {
    var hidden: DifferentiableTracer  // [hiddenSize]
    var cell: DifferentiableTracer    // [hiddenSize]
}

@differentiable(reverse)
func lstmStep(
    carry: LSTMCarry,
    x: DifferentiableTracer
) -> (LSTMCarry, DifferentiableTracer) {
    let combined = diffConcat([carry.hidden, x], axis: 0)
    
    // Gates
    let gates = diffMatmul(combined, Wgates) + bgates  // [4 * hiddenSize]
    let (i, f, g, o) = splitGates(gates)
    
    let newCell = diffSigmoid(f) * carry.cell + diffSigmoid(i) * diffTanh(g)
    let newHidden = diffSigmoid(o) * diffTanh(newCell)
    
    return (LSTMCarry(hidden: newHidden, cell: newCell), newHidden)
}

let initCarry = LSTMCarry(
    hidden: DifferentiableTracer.zeros(shape: [hiddenSize]),
    cell: DifferentiableTracer.zeros(shape: [hiddenSize])
)

let (finalCarry, hiddenStates) = diffScan(lstmStep, init: initCarry, xs: inputs)
```

#### Example 3: Cumulative Sum (via scan)
```swift
@differentiable(reverse)
func cumsumStep(
    carry: DifferentiableTracer,
    x: DifferentiableTracer
) -> (DifferentiableTracer, DifferentiableTracer) {
    let newCarry = carry + x
    return (newCarry, newCarry)  // Output = carry at each step
}

let zeros = DifferentiableTracer.zeros(shape: x.shape.dropFirst())
let (_, cumsum) = diffScan(cumsumStep, init: zeros, xs: x)
```

#### Example 4: Bidirectional Processing
```swift
// Forward pass
let (_, forwardOutputs) = diffScan(rnnStep, init: initHidden, xs: inputs)

// Backward pass
let (_, backwardOutputs) = diffScan(rnnStep, init: initHidden, xs: inputs, reverse: true)

// Combine
let bidirectionalOutputs = diffConcat([forwardOutputs, backwardOutputs], axis: -1)
```

#### Example 5: Attention Accumulation
```swift
@differentiable(reverse)
func attentionStep(
    carry: DifferentiableTracer,  // Running attention context
    kv: (DifferentiableTracer, DifferentiableTracer)  // (key, value) pair
) -> (DifferentiableTracer, DifferentiableTracer) {
    let (k, v) = kv
    let score = diffSoftmax(diffMatmul(query, k.transposed()))
    let context = carry + diffMatmul(score, v)
    return (context, context)
}
```

## Implementation Guide

### Core Data Structures

```swift
/// Configuration for scan operation
public struct ScanConfig {
    /// Sequence length (nil = infer from input)
    public var length: Int?
    
    /// Process in reverse order
    public var reverse: Bool
    
    /// Unroll factor (1 = fully looped, n = unroll n steps)
    public var unroll: Int
    
    /// Axis of sequence dimension in input
    public var axis: Int
}

/// Internal representation of scan for tracing
struct ScanOp {
    var stepFunction: TracedFunction
    var initCarry: [DifferentiableTracer]
    var inputs: [DifferentiableTracer]
    var config: ScanConfig
}
```

### Implementation Steps

#### Step 1: Build on diffWhileLoop

scan is implemented using the existing `diffWhileLoop`:

```swift
@differentiable(reverse)
public func diffScan<Carry, Input, Output>(
    _ fn: @escaping @differentiable(reverse) (Carry, Input) -> (Carry, Output),
    init: Carry,
    xs: Input,
    length: Int? = nil,
    reverse: Bool = false,
    unroll: Int = 1
) -> (Carry, Output) {
    // 1. Extract tracers and determine sequence length
    let inputTracers = extractTracers(xs)
    let seqLen = length ?? inputTracers[0].shape[0]
    
    // 2. Prepare output accumulator
    // Trace one step to get output shape
    let sampleInput = sliceSequence(inputTracers, at: 0)
    let (_, sampleOutput) = fn(init, reconstruct(sampleInput) as! Input)
    let outputTracers = extractTracers(sampleOutput)
    
    // Create output buffer: [seqLen, ...outputShape]
    var outputAccumulator = outputTracers.map { tracer in
        DifferentiableTracer.zeros(shape: [seqLen] + tracer.shape)
    }
    
    // 3. Build while loop state: (index, carry, outputs)
    struct ScanState: Differentiable {
        var index: Int  // Not differentiable, but tracked
        var carry: Carry
        var outputs: [DifferentiableTracer]
    }
    
    let initialState = ScanState(
        index: reverse ? seqLen - 1 : 0,
        carry: init,
        outputs: outputAccumulator
    )
    
    // 4. Define loop body
    @differentiable(reverse)
    func loopBody(_ state: ScanState) -> ScanState {
        // Get input at current index
        let currentInput = sliceSequence(inputTracers, at: state.index)
        
        // Run step function
        let (newCarry, stepOutput) = fn(state.carry, reconstruct(currentInput) as! Input)
        
        // Update output accumulator
        var newOutputs = state.outputs
        let outputTracers = extractTracers(stepOutput)
        for (i, output) in outputTracers.enumerated() {
            newOutputs[i] = dynamicUpdateSlice(newOutputs[i], output, at: state.index)
        }
        
        // Update index
        let newIndex = reverse ? state.index - 1 : state.index + 1
        
        return ScanState(index: newIndex, carry: newCarry, outputs: newOutputs)
    }
    
    // 5. Run while loop
    let finalState = diffWhileLoop(
        initialState: initialState,
        condition: { state, _ in
            reverse ? state.index >= 0 : state.index < seqLen
        },
        body: loopBody
    )
    
    // 6. Extract results
    return (finalState.carry, reconstruct(finalState.outputs) as! Output)
}
```

#### Step 2: Efficient StableHLO Generation

Generate efficient StableHLO using while with dynamic-update-slice:

```swift
func generateScanStableHLO(_ scan: ScanOp) -> String {
    let seqLen = scan.config.length!
    let carryTypes = scan.initCarry.map { $0.shape.hloType }
    let outputTypes = scan.outputs.map { "tensor<\([$seqLen] + $0.shape)xf32>" }
    
    return """
    // Initialize outputs
    \(scan.outputs.enumerated().map { i, out in
        "%output_init_\(i) = stablehlo.constant dense<0.0> : tensor<\([$seqLen] + out.shape)xf32>"
    }.joined(separator: "\n"))
    
    // While loop
    %result:N = stablehlo.while(%init_idx, %init_carry..., %output_init...) : 
        (tensor<i32>, \(carryTypes.joined(separator: ", ")), \(outputTypes.joined(separator: ", "))) -> ... {
    ^condition(%idx, %carry..., %outputs...):
        %limit = stablehlo.constant dense<\(seqLen)> : tensor<i32>
        %cond = stablehlo.compare LT, %idx, %limit : tensor<i1>
        stablehlo.return %cond
    } do {
    ^body(%idx, %carry..., %outputs...):
        // Slice input at current index
        %input = stablehlo.dynamic_slice %xs, %idx, ... : tensor<...>
        
        // Run step function (inlined)
        \(scan.stepFunction.hloBody)
        
        // Update output accumulator
        %new_outputs = stablehlo.dynamic_update_slice %outputs, %step_output, %idx, ...
        
        // Increment index
        %one = stablehlo.constant dense<1> : tensor<i32>
        %new_idx = stablehlo.add %idx, %one
        
        stablehlo.return %new_idx, %new_carry..., %new_outputs...
    }
    """
}
```

#### Step 3: Gradient Implementation

The gradient of scan is a reverse scan:

```swift
@derivative(of: diffScan)
func diffScanVJP<Carry, Input, Output>(
    _ fn: @escaping @differentiable(reverse) (Carry, Input) -> (Carry, Output),
    init: Carry,
    xs: Input,
    length: Int?,
    reverse: Bool,
    unroll: Int
) -> (value: (Carry, Output),
      pullback: (Carry.TangentVector, Output.TangentVector) -> (Carry.TangentVector, Input.TangentVector)) {
    
    // Forward pass - also save intermediates for backward
    var carries: [Carry] = [init]
    var outputs: [Output] = []
    
    let seqLen = length ?? inferLength(xs)
    for i in 0..<seqLen {
        let idx = reverse ? seqLen - 1 - i : i
        let input = sliceInput(xs, at: idx)
        let (newCarry, output) = fn(carries.last!, input)
        carries.append(newCarry)
        outputs.append(output)
    }
    
    let finalCarry = carries.last!
    let stackedOutputs = stackOutputs(outputs)
    
    // Pullback: reverse scan through adjoints
    return ((finalCarry, stackedOutputs), { (dCarry, dOutputs) in
        var carryAdj = dCarry
        var inputAdjs: [Input.TangentVector] = []
        
        // Backward scan
        for i in (0..<seqLen).reversed() {
            let idx = reverse ? seqLen - 1 - i : i
            let input = sliceInput(xs, at: idx)
            let carry = carries[i]
            
            // Get output adjoint for this step
            let outputAdj = sliceOutput(dOutputs, at: i)
            
            // Pullback through step function
            let (_, pb) = valueWithPullback(at: carry, input) { c, x in fn(c, x) }
            let (dCarryStep, dInput) = pb((carryAdj, outputAdj))
            
            carryAdj = dCarryStep
            inputAdjs.append(dInput)
        }
        
        let initAdj = carryAdj
        let xsAdj = stackInputAdjs(inputAdjs.reversed())
        
        return (initAdj, xsAdj)
    })
}
```

### Optimization: Checkpointing for Long Sequences

For very long sequences, store only checkpoints:

```swift
/// Scan with gradient checkpointing for memory efficiency
@differentiable(reverse)
public func diffScanCheckpointed<Carry, Input, Output>(
    _ fn: @escaping @differentiable(reverse) (Carry, Input) -> (Carry, Output),
    init: Carry,
    xs: Input,
    checkpointEvery: Int = 100
) -> (Carry, Output)
```

## Complete Test Suite

### File: Tests/SwiftIRTests/ScanTests.swift

```swift
import XCTest
@testable import SwiftIR

final class ScanTests: XCTestCase {
    
    // MARK: - Basic Functionality
    
    func testScanCumsum() {
        // Cumulative sum via scan
        @differentiable(reverse)
        func cumsumStep(carry: DifferentiableTracer, x: DifferentiableTracer) 
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry + x
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.constant([1, 2, 3, 4, 5], shape: [5])
        let init = DifferentiableTracer.zeros(shape: [])
        
        let (final, cumsum) = diffScan(cumsumStep, init: init, xs: xs)
        
        // Compile and execute
        let result = compileAndExecute(cumsum)
        XCTAssertEqual(result, [1, 3, 6, 10, 15], accuracy: 1e-6)
        
        let finalResult = compileAndExecute(final)
        XCTAssertEqual(finalResult, [15], accuracy: 1e-6)
    }
    
    func testScanCumprod() {
        @differentiable(reverse)
        func cumprodStep(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry * x
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.constant([1, 2, 3, 4], shape: [4])
        let init = DifferentiableTracer.ones(shape: [])
        
        let (_, cumprod) = diffScan(cumprodStep, init: init, xs: xs)
        
        let result = compileAndExecute(cumprod)
        XCTAssertEqual(result, [1, 2, 6, 24], accuracy: 1e-6)
    }
    
    func testScanRNN() {
        let hiddenSize = 8
        let inputSize = 4
        let seqLen = 10
        
        // Create weight placeholders
        let Wh = DifferentiableTracer.placeholder(shape: [hiddenSize, hiddenSize])
        let Wx = DifferentiableTracer.placeholder(shape: [inputSize, hiddenSize])
        
        @differentiable(reverse)
        func rnnStep(h: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newH = diffTanh(diffMatmul(h, Wh) + diffMatmul(x, Wx))
            return (newH, newH)
        }
        
        let initH = DifferentiableTracer.zeros(shape: [hiddenSize])
        let xs = DifferentiableTracer.placeholder(shape: [seqLen, inputSize])
        
        let (finalH, outputs) = diffScan(rnnStep, init: initH, xs: xs)
        
        XCTAssertEqual(finalH.shape, [hiddenSize])
        XCTAssertEqual(outputs.shape, [seqLen, hiddenSize])
    }
    
    func testScanDifferentOutputShape() {
        // Output shape different from carry shape
        let hiddenSize = 16
        let outputSize = 4
        
        let Wo = DifferentiableTracer.placeholder(shape: [hiddenSize, outputSize])
        
        @differentiable(reverse)
        func step(h: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newH = diffReLU(h + x)  // [hidden]
            let out = diffMatmul(newH.unsqueezed(0), Wo).squeezed()  // [output]
            return (newH, out)
        }
        
        let initH = DifferentiableTracer.zeros(shape: [hiddenSize])
        let xs = DifferentiableTracer.placeholder(shape: [5, hiddenSize])
        
        let (finalH, outputs) = diffScan(step, init: initH, xs: xs)
        
        XCTAssertEqual(finalH.shape, [hiddenSize])
        XCTAssertEqual(outputs.shape, [5, outputSize])
    }
    
    // MARK: - Reverse Scan
    
    func testScanReverse() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry + x
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.constant([1, 2, 3, 4, 5], shape: [5])
        let init = DifferentiableTracer.zeros(shape: [])
        
        // Forward: [1, 3, 6, 10, 15]
        let (_, forward) = diffScan(step, init: init, xs: xs, reverse: false)
        
        // Reverse: starts from end, so [5, 9, 12, 14, 15]
        let (_, backward) = diffScan(step, init: init, xs: xs, reverse: true)
        
        let fwdResult = compileAndExecute(forward)
        let bwdResult = compileAndExecute(backward)
        
        XCTAssertEqual(fwdResult, [1, 3, 6, 10, 15], accuracy: 1e-6)
        XCTAssertEqual(bwdResult, [5, 9, 12, 14, 15], accuracy: 1e-6)
    }
    
    func testScanBidirectional() {
        let hiddenSize = 8
        
        @differentiable(reverse)
        func step(h: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newH = diffTanh(h + x)
            return (newH, newH)
        }
        
        let init = DifferentiableTracer.zeros(shape: [hiddenSize])
        let xs = DifferentiableTracer.placeholder(shape: [10, hiddenSize])
        
        let (_, fwd) = diffScan(step, init: init, xs: xs, reverse: false)
        let (_, bwd) = diffScan(step, init: init, xs: xs, reverse: true)
        
        // Concatenate for bidirectional
        let combined = diffConcat([fwd, bwd], axis: 1)
        
        XCTAssertEqual(combined.shape, [10, hiddenSize * 2])
    }
    
    // MARK: - Gradient Tests
    
    func testScanGradientSimple() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry + x
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.placeholder(shape: [5, 4])
        let init = DifferentiableTracer.zeros(shape: [4])
        
        let (gradInit, gradXs) = gradient(at: init, xs) { init, xs in
            let (final, outputs) = diffScan(step, init: init, xs: xs)
            return diffSum(final) + diffSum(outputs)
        }
        
        XCTAssertEqual(gradInit.shape, [4])
        XCTAssertEqual(gradXs.shape, [5, 4])
    }
    
    func testScanGradientRNN() {
        let hiddenSize = 8
        let inputSize = 4
        let seqLen = 5
        
        let Wh = DifferentiableTracer.placeholder(shape: [hiddenSize, hiddenSize])
        let Wx = DifferentiableTracer.placeholder(shape: [inputSize, hiddenSize])
        
        @differentiable(reverse)
        func rnnStep(h: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newH = diffTanh(diffMatmul(h, Wh) + diffMatmul(x, Wx))
            return (newH, newH)
        }
        
        let initH = DifferentiableTracer.zeros(shape: [hiddenSize])
        let xs = DifferentiableTracer.placeholder(shape: [seqLen, inputSize])
        
        let (gradWh, gradWx) = gradient(at: Wh, Wx) { Wh, Wx in
            let (_, outputs) = diffScan(rnnStep, init: initH, xs: xs)
            return diffSum(outputs)
        }
        
        XCTAssertEqual(gradWh.shape, [hiddenSize, hiddenSize])
        XCTAssertEqual(gradWx.shape, [inputSize, hiddenSize])
    }
    
    func testScanGradientNumerical() {
        // Numerical gradient check
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = diffSigmoid(carry + x)
            return (newCarry, newCarry)
        }
        
        let xsValues: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        let xs = DifferentiableTracer.constant(xsValues, shape: [4, 2])
        let init = DifferentiableTracer.zeros(shape: [2])
        
        // Analytical gradient
        let (_, analyticalGrad) = valueWithGradient(at: xs) { xs in
            let (final, _) = diffScan(step, init: init, xs: xs)
            return diffSum(final)
        }
        
        // Numerical gradient
        let numericalGrad = computeNumericalGradient(
            { xs in
                let (final, _) = diffScan(step, init: init, xs: xs)
                return diffSum(final)
            },
            at: xsValues,
            epsilon: 1e-5
        )
        
        let analytical = compileAndExecute(analyticalGrad)
        XCTAssertEqual(analytical, numericalGrad, accuracy: 1e-4)
    }
    
    // MARK: - Struct Carry Tests
    
    func testScanStructCarry() {
        struct LSTMState: Differentiable {
            var h: DifferentiableTracer
            var c: DifferentiableTracer
        }
        
        @differentiable(reverse)
        func lstmStep(state: LSTMState, x: DifferentiableTracer)
            -> (LSTMState, DifferentiableTracer) {
            // Simplified LSTM
            let newC = state.c * 0.9 + x * 0.1
            let newH = diffTanh(newC)
            return (LSTMState(h: newH, c: newC), newH)
        }
        
        let initState = LSTMState(
            h: DifferentiableTracer.zeros(shape: [8]),
            c: DifferentiableTracer.zeros(shape: [8])
        )
        let xs = DifferentiableTracer.placeholder(shape: [10, 8])
        
        let (finalState, outputs) = diffScan(lstmStep, init: initState, xs: xs)
        
        XCTAssertEqual(finalState.h.shape, [8])
        XCTAssertEqual(finalState.c.shape, [8])
        XCTAssertEqual(outputs.shape, [10, 8])
    }
    
    // MARK: - Edge Cases
    
    func testScanEmptySequence() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            return (carry + x, carry)
        }
        
        let xs = DifferentiableTracer.placeholder(shape: [0, 4])  // Empty sequence
        let init = DifferentiableTracer.ones(shape: [4])
        
        let (final, outputs) = diffScan(step, init: init, xs: xs)
        
        XCTAssertEqual(final.shape, [4])
        XCTAssertEqual(outputs.shape, [0, 4])
        
        // Final should equal init for empty sequence
    }
    
    func testScanSingleElement() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry + x
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.constant([5.0], shape: [1])
        let init = DifferentiableTracer.zeros(shape: [])
        
        let (final, outputs) = diffScan(step, init: init, xs: xs)
        
        let finalResult = compileAndExecute(final)
        let outputResult = compileAndExecute(outputs)
        
        XCTAssertEqual(finalResult, [5.0], accuracy: 1e-6)
        XCTAssertEqual(outputResult, [5.0], accuracy: 1e-6)
    }
    
    func testScanLongSequence() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry * 0.99 + x * 0.01
            return (newCarry, newCarry)
        }
        
        let xs = DifferentiableTracer.placeholder(shape: [10000, 8])  // Long sequence
        let init = DifferentiableTracer.zeros(shape: [8])
        
        let start = CFAbsoluteTimeGetCurrent()
        let (final, outputs) = diffScan(step, init: init, xs: xs)
        let compileTime = CFAbsoluteTimeGetCurrent() - start
        
        XCTAssertEqual(final.shape, [8])
        XCTAssertEqual(outputs.shape, [10000, 8])
        
        // Should compile in O(1) time (not O(n))
        XCTAssertLessThan(compileTime, 1.0)  // Less than 1 second
    }
    
    // MARK: - Composition Tests
    
    func testScanWithVmap() {
        // Batch of sequences
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = diffTanh(carry + x)
            return (newCarry, newCarry)
        }
        
        @differentiable(reverse)
        func processSequence(xs: DifferentiableTracer) -> DifferentiableTracer {
            let init = DifferentiableTracer.zeros(shape: [8])
            let (final, _) = diffScan(step, init: init, xs: xs)
            return final
        }
        
        // vmap over batch of sequences
        let batchedProcess = vmap(processSequence, inAxes: .init(0))
        
        let batch = DifferentiableTracer.placeholder(shape: [32, 10, 8])  // [batch, seq, features]
        let output = batchedProcess(batch)
        
        XCTAssertEqual(output.shape, [32, 8])
    }
    
    func testScanNestedInScan() {
        // Outer scan over time, inner scan over features
        @differentiable(reverse)
        func innerStep(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            let newCarry = carry + x
            return (newCarry, newCarry)
        }
        
        @differentiable(reverse)
        func outerStep(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            // x is a row, apply inner scan
            let (final, _) = diffScan(innerStep, init: carry, xs: x)
            return (final, final)
        }
        
        let xs = DifferentiableTracer.placeholder(shape: [10, 8])  // [time, features]
        let init = DifferentiableTracer.zeros(shape: [])
        
        let (final, outputs) = diffScan(outerStep, init: init, xs: xs)
        
        // Each outer step processes all features
        XCTAssertEqual(final.shape, [])  // Scalar
        XCTAssertEqual(outputs.shape, [10])  // One output per timestep
    }
    
    // MARK: - Performance Tests
    
    func testScanCompilationTimeScaling() {
        // Verify O(1) compilation
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            return (carry + x, carry)
        }
        
        let init = DifferentiableTracer.zeros(shape: [8])
        
        var compileTimes: [Double] = []
        
        for seqLen in [100, 1000, 10000] {
            let xs = DifferentiableTracer.placeholder(shape: [seqLen, 8])
            
            let start = CFAbsoluteTimeGetCurrent()
            let _ = diffScan(step, init: init, xs: xs)
            compileTimes.append(CFAbsoluteTimeGetCurrent() - start)
        }
        
        // Compilation time should be roughly constant (not linear in seqLen)
        let ratio = compileTimes[2] / compileTimes[0]
        XCTAssertLessThan(ratio, 5.0)  // Should not grow 100x when seqLen grows 100x
    }
    
    func testScanVsUnrolled() {
        // Compare scan performance to manually unrolled loop
        let seqLen = 100
        
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            return (diffTanh(carry + x), carry)
        }
        
        let init = DifferentiableTracer.zeros(shape: [8])
        let xs = DifferentiableTracer.placeholder(shape: [seqLen, 8])
        
        // Scan version
        let scanStart = CFAbsoluteTimeGetCurrent()
        let (scanFinal, _) = diffScan(step, init: init, xs: xs)
        let scanCompile = CFAbsoluteTimeGetCurrent() - scanStart
        
        // Scan should be much faster to compile
        XCTAssertLessThan(scanCompile, 0.5)
    }
    
    // MARK: - StableHLO Generation
    
    func testScanGeneratesWhileLoop() {
        @differentiable(reverse)
        func step(carry: DifferentiableTracer, x: DifferentiableTracer)
            -> (DifferentiableTracer, DifferentiableTracer) {
            return (carry + x, carry)
        }
        
        let xs = DifferentiableTracer.placeholder(shape: [10, 4])
        let init = DifferentiableTracer.zeros(shape: [4])
        
        let (final, outputs) = diffScan(step, init: init, xs: xs)
        
        let mlir = generateStableHLO(outputs)
        
        // Should contain while loop
        XCTAssertTrue(mlir.contains("stablehlo.while"))
        
        // Should contain dynamic slice for indexing
        XCTAssertTrue(mlir.contains("stablehlo.dynamic_slice") || 
                      mlir.contains("stablehlo.gather"))
        
        // Should contain dynamic update slice for output accumulation
        XCTAssertTrue(mlir.contains("stablehlo.dynamic_update_slice") ||
                      mlir.contains("stablehlo.scatter"))
    }
}
```

## Success Criteria

### Functional Requirements

- [ ] Basic scan with cumulative operations (sum, prod)
- [ ] scan with RNN-style step functions
- [ ] scan with struct carries (LSTM state)
- [ ] scan with different input/output shapes
- [ ] Reverse scan
- [ ] Bidirectional scan composition
- [ ] Gradient through scan
- [ ] Gradient numerical correctness

### Performance Requirements

- [ ] O(1) compilation time (not O(seqLen))
- [ ] Compilation < 100ms for typical RNNs
- [ ] Runtime comparable to manual implementation
- [ ] Memory: O(seqLen) for outputs, O(1) for carry

### Composition

- [ ] scan + vmap (batched sequences)
- [ ] scan + gradient
- [ ] scan + diffWhileLoop
- [ ] Nested scan

### Edge Cases

- [ ] Empty sequence (length 0)
- [ ] Single element sequence
- [ ] Very long sequences (10000+)
- [ ] Scalar carries
- [ ] High-dimensional carries/outputs

## Error Messages

```swift
// Shape mismatch in step function
"diffScan: Step function output carry shape [8, 8] doesn't match input carry shape [8]. Carry shape must be consistent."

// Output shape inference failed
"diffScan: Could not infer output shape. Step function must return consistent output shapes."

// Sequence axis out of bounds
"diffScan: Sequence axis 2 is out of bounds for input shape [10, 8]. Input must have at least 3 dimensions."
```

## Files to Create/Modify

### New Files
- `Sources/SwiftIR/SymbolicAD/Scan.swift` - Main implementation
- `Tests/SwiftIRTests/ScanTests.swift` - Test suite

### Modified Files
- `Sources/SwiftIR/SymbolicAD/DifferentiableWhile.swift` - May need extensions for scan
- `Sources/SwiftIR/SymbolicAD/BackendCompilation.swift` - StableHLO for dynamic slice/update
