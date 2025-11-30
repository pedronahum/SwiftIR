# COND/SWITCH - Differentiable Conditionals Specification

## Overview

Differentiable conditionals allow branching in traced computation graphs while maintaining gradient flow. This enables dynamic architectures, early stopping, and conditional computation patterns.

## JAX Reference

```python
# JAX cond usage
result = jax.lax.cond(
    pred,
    true_fun,   # Called if pred is True
    false_fun,  # Called if pred is False
    operand
)

# JAX switch for multiple branches
result = jax.lax.switch(
    index,
    [branch_0, branch_1, branch_2],
    operand
)
```

## SwiftIR API Specification

### Primary API: diffCond

```swift
/// Differentiable conditional execution
/// - Parameters:
///   - predicate: Boolean condition (scalar tensor)
///   - trueBranch: Function executed when predicate is true
///   - falseBranch: Function executed when predicate is false
/// - Returns: Result from the selected branch
@differentiable(reverse)
public func diffCond<T: Differentiable>(
    _ predicate: DifferentiableTracer,
    trueBranch: @escaping @differentiable(reverse) () -> T,
    falseBranch: @escaping @differentiable(reverse) () -> T
) -> T
```

### With Operand API

```swift
/// Differentiable conditional with shared operand
@differentiable(reverse)
public func diffCond<T: Differentiable, U: Differentiable>(
    _ predicate: DifferentiableTracer,
    _ operand: T,
    trueBranch: @escaping @differentiable(reverse) (T) -> U,
    falseBranch: @escaping @differentiable(reverse) (T) -> U
) -> U
```

### Switch API

```swift
/// Differentiable multi-way branch
/// - Parameters:
///   - index: Integer index selecting which branch to execute
///   - branches: Array of functions, one per branch
///   - operand: Input passed to the selected branch
/// - Returns: Result from the selected branch
@differentiable(reverse)
public func diffSwitch<T: Differentiable, U: Differentiable>(
    _ index: DifferentiableTracer,
    branches: [@escaping @differentiable(reverse) (T) -> U],
    operand: T
) -> U
```

### Select API (Element-wise)

```swift
/// Element-wise conditional selection
/// - Parameters:
///   - condition: Boolean mask tensor
///   - onTrue: Values selected where condition is true
///   - onFalse: Values selected where condition is false
/// - Returns: Element-wise selection result
@differentiable(reverse)
public func diffSelect(
    _ condition: DifferentiableTracer,
    onTrue: DifferentiableTracer,
    onFalse: DifferentiableTracer
) -> DifferentiableTracer
```

### Usage Examples

#### Example 1: Simple Conditional
```swift
@differentiable(reverse)
func conditionalActivation(_ x: DifferentiableTracer, useRelu: DifferentiableTracer) -> DifferentiableTracer {
    return diffCond(
        useRelu,
        trueBranch: { diffReLU(x) },
        falseBranch: { diffLeakyReLU(x, alpha: 0.1) }
    )
}
```

#### Example 2: Residual Skip
```swift
@differentiable(reverse)
func maybeResidual(_ x: DifferentiableTracer, _ shouldSkip: DifferentiableTracer) -> DifferentiableTracer {
    return diffCond(
        shouldSkip,
        x,
        trueBranch: { input in input },  // Skip: identity
        falseBranch: { input in          // Apply transformation
            let h = diffReLU(diffMatmul(input, W1))
            return diffMatmul(h, W2) + input
        }
    )
}
```

#### Example 3: Activation Switch
```swift
@differentiable(reverse)
func dynamicActivation(_ x: DifferentiableTracer, _ activationIndex: DifferentiableTracer) -> DifferentiableTracer {
    return diffSwitch(
        activationIndex,
        branches: [
            { x in diffReLU(x) },
            { x in diffGELU(x) },
            { x in diffSwish(x) },
            { x in diffTanh(x) }
        ],
        operand: x
    )
}
```

#### Example 4: Element-wise Masking
```swift
@differentiable(reverse)
func maskedAttention(_ scores: DifferentiableTracer, _ mask: DifferentiableTracer) -> DifferentiableTracer {
    let negInf = DifferentiableTracer.constant(-1e9, shape: scores.shape)
    let maskedScores = diffSelect(mask, onTrue: scores, onFalse: negInf)
    return diffSoftmax(maskedScores, axis: -1)
}
```

#### Example 5: Early Exit
```swift
@differentiable(reverse)
func maybeEarlyExit(_ x: DifferentiableTracer, _ confidence: DifferentiableTracer, threshold: Float) -> DifferentiableTracer {
    let shouldExit = confidence > DifferentiableTracer.constant(threshold)
    return diffCond(
        shouldExit,
        trueBranch: { x },  // Return current prediction
        falseBranch: {      // Continue processing
            return deeperNetwork(x)
        }
    )
}
```

#### Example 6: Mixture of Experts
```swift
@differentiable(reverse)
func mixtureOfExperts(_ x: DifferentiableTracer, _ routing: DifferentiableTracer) -> DifferentiableTracer {
    // routing: [batchSize] integer indices
    let expertIndex = diffArgmax(routing)
    
    return diffSwitch(
        expertIndex,
        branches: [
            { x in expert0(x) },
            { x in expert1(x) },
            { x in expert2(x) },
            { x in expert3(x) }
        ],
        operand: x
    )
}
```

## Implementation Guide

### Core Data Structures

```swift
/// Represents a conditional operation in the trace
struct CondOp {
    var predicate: DifferentiableTracer
    var trueBranch: TracedFunction
    var falseBranch: TracedFunction
    var resultShape: [Int]
}

/// Represents a switch operation
struct SwitchOp {
    var index: DifferentiableTracer
    var branches: [TracedFunction]
    var operand: DifferentiableTracer
    var resultShape: [Int]
}

/// A traced function (branch)
struct TracedFunction {
    var inputs: [DifferentiableTracer]
    var outputs: [DifferentiableTracer]
    var operations: [Operation]
}
```

### Implementation Steps

#### Step 1: Branch Tracing

Both branches must be traced to build the computation graph:

```swift
@differentiable(reverse)
public func diffCond<T: Differentiable>(
    _ predicate: DifferentiableTracer,
    trueBranch: @escaping @differentiable(reverse) () -> T,
    falseBranch: @escaping @differentiable(reverse) () -> T
) -> T {
    // Validate predicate is scalar boolean
    precondition(predicate.shape == [] || predicate.shape == [1], 
                 "Predicate must be scalar, got shape \(predicate.shape)")
    
    // 1. Trace both branches (abstract interpretation)
    let trueTraced = traceBranch(trueBranch)
    let falseTraced = traceBranch(falseBranch)
    
    // 2. Validate branches have matching output shapes
    precondition(trueTraced.outputShape == falseTraced.outputShape,
                 "Branch output shapes must match: \(trueTraced.outputShape) vs \(falseTraced.outputShape)")
    
    // 3. Create conditional operation
    let condOp = CondOp(
        predicate: predicate,
        trueBranch: trueTraced,
        falseBranch: falseTraced,
        resultShape: trueTraced.outputShape
    )
    
    // 4. Create result tracer
    let result = DifferentiableTracer(
        shape: condOp.resultShape,
        dtype: .float32,
        operation: .cond(condOp)
    )
    
    return reconstruct(result) as! T
}

/// Trace a branch to capture its operations
func traceBranch<T>(_ branch: () -> T) -> TracedFunction {
    // Enter branch tracing mode
    let context = BranchTracingContext()
    BranchTracingContext.stack.append(context)
    defer { BranchTracingContext.stack.removeLast() }
    
    // Execute branch (symbolically)
    let output = branch()
    
    // Collect traced operations
    return TracedFunction(
        inputs: context.capturedInputs,
        outputs: extractTracers(output),
        operations: context.operations
    )
}
```

#### Step 2: StableHLO Generation

Generate `stablehlo.if` for conditionals:

```swift
func generateCondStableHLO(_ cond: CondOp, resultId: Int) -> String {
    let resultType = "tensor<\(cond.resultShape.hloType)>"
    
    return """
    %\(resultId) = stablehlo.if %\(cond.predicate.id) -> \(resultType) {
        // True branch
        \(generateBranchHLO(cond.trueBranch))
        stablehlo.return %\(cond.trueBranch.outputId) : \(resultType)
    } else {
        // False branch
        \(generateBranchHLO(cond.falseBranch))
        stablehlo.return %\(cond.falseBranch.outputId) : \(resultType)
    }
    """
}

func generateSwitchStableHLO(_ sw: SwitchOp, resultId: Int) -> String {
    let resultType = "tensor<\(sw.resultShape.hloType)>"
    let branchBodies = sw.branches.enumerated().map { i, branch in
        """
        {
            \(generateBranchHLO(branch))
            stablehlo.return %\(branch.outputId) : \(resultType)
        }
        """
    }.joined(separator: ",\n")
    
    return """
    %\(resultId) = stablehlo.case %\(sw.index.id) : tensor<i32> -> \(resultType)
    \(branchBodies)
    """
}
```

#### Step 3: Select (Element-wise) Implementation

Select maps directly to `stablehlo.select`:

```swift
@differentiable(reverse)
public func diffSelect(
    _ condition: DifferentiableTracer,
    onTrue: DifferentiableTracer,
    onFalse: DifferentiableTracer
) -> DifferentiableTracer {
    // Validate shapes are broadcastable
    let resultShape = broadcastShapes(condition.shape, onTrue.shape, onFalse.shape)
    
    let result = DifferentiableTracer(
        shape: resultShape,
        dtype: onTrue.dtype,
        operation: .select(condition: condition, onTrue: onTrue, onFalse: onFalse)
    )
    
    return result
}

// StableHLO generation
func generateSelectStableHLO(_ select: SelectOp, resultId: Int) -> String {
    return """
    %\(resultId) = stablehlo.select %\(select.condition.id), %\(select.onTrue.id), %\(select.onFalse.id) 
        : tensor<\(select.condition.shape.hloType)>, tensor<\(select.resultShape.hloType)>
    """
}
```

#### Step 4: Gradient Implementation

Gradient flows through the taken branch only (for scalar predicates):

```swift
@derivative(of: diffCond)
func diffCondVJP<T: Differentiable>(
    _ predicate: DifferentiableTracer,
    trueBranch: @escaping @differentiable(reverse) () -> T,
    falseBranch: @escaping @differentiable(reverse) () -> T
) -> (value: T, pullback: (T.TangentVector) -> (DifferentiableTracer, (), ())) {
    
    // Forward: execute both branches symbolically, select based on predicate
    let trueResult = trueBranch()
    let falseResult = falseBranch()
    
    let value = diffCond(predicate, trueBranch: { trueResult }, falseBranch: { falseResult })
    
    return (value, { upstream in
        // Gradient of predicate is zero (not differentiable)
        let predGrad = DifferentiableTracer.zeros(shape: predicate.shape)
        
        // Gradient flows only to the taken branch
        // At trace time, we must handle both paths
        // At runtime, only the selected branch's gradient is used
        
        // This is handled by generating conditional gradient code
        return (predGrad, (), ())
    })
}

/// Gradient for select (element-wise)
@derivative(of: diffSelect)
func diffSelectVJP(
    _ condition: DifferentiableTracer,
    onTrue: DifferentiableTracer,
    onFalse: DifferentiableTracer
) -> (value: DifferentiableTracer, 
      pullback: (DifferentiableTracer) -> (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer)) {
    
    let result = diffSelect(condition, onTrue: onTrue, onFalse: onFalse)
    
    return (result, { upstream in
        // Condition gradient is zero (boolean, not differentiable)
        let condGrad = DifferentiableTracer.zeros(shape: condition.shape)
        
        // Gradient to onTrue: upstream where condition is true, else 0
        let trueGrad = diffSelect(condition, onTrue: upstream, onFalse: DifferentiableTracer.zeros(shape: upstream.shape))
        
        // Gradient to onFalse: upstream where condition is false, else 0
        let falseGrad = diffSelect(condition, onTrue: DifferentiableTracer.zeros(shape: upstream.shape), onFalse: upstream)
        
        return (condGrad, trueGrad, falseGrad)
    })
}
```

### Handling Captured Variables

When branches capture external variables, they must be properly handled:

```swift
@differentiable(reverse)
func withCaptures(_ x: DifferentiableTracer, _ y: DifferentiableTracer, pred: DifferentiableTracer) -> DifferentiableTracer {
    // Both x and y are captured in the branches
    return diffCond(
        pred,
        trueBranch: { x + y },      // Captures x, y
        falseBranch: { x * y }      // Captures x, y
    )
}
```

Implementation must track captured values:

```swift
struct TracedFunction {
    var capturedValues: [DifferentiableTracer]  // Values captured from outer scope
    var inputs: [DifferentiableTracer]          // Explicit function inputs
    var outputs: [DifferentiableTracer]
    var operations: [Operation]
}

func traceBranch<T>(_ branch: () -> T, currentScope: Set<Int>) -> TracedFunction {
    let context = BranchTracingContext()
    BranchTracingContext.stack.append(context)
    defer { BranchTracingContext.stack.removeLast() }
    
    let output = branch()
    
    // Identify captured values (tracers from outer scope used in this branch)
    let capturedValues = context.referencedTracers.filter { currentScope.contains($0.id) }
    
    return TracedFunction(
        capturedValues: capturedValues,
        inputs: context.explicitInputs,
        outputs: extractTracers(output),
        operations: context.operations
    )
}
```

## Complete Test Suite

### File: Tests/SwiftIRTests/CondTests.swift

```swift
import XCTest
@testable import SwiftIR

final class CondTests: XCTestCase {
    
    // MARK: - Basic diffCond Tests
    
    func testCondTrue() {
        let pred = DifferentiableTracer.constant(true)
        let x = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
        
        let result = diffCond(
            pred,
            trueBranch: { x * 2 },
            falseBranch: { x * 3 }
        )
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [2.0, 4.0, 6.0], accuracy: 1e-6)
    }
    
    func testCondFalse() {
        let pred = DifferentiableTracer.constant(false)
        let x = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
        
        let result = diffCond(
            pred,
            trueBranch: { x * 2 },
            falseBranch: { x * 3 }
        )
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [3.0, 6.0, 9.0], accuracy: 1e-6)
    }
    
    func testCondWithOperand() {
        let pred = DifferentiableTracer.constant(true)
        let x = DifferentiableTracer.constant([1.0, 2.0], shape: [2])
        
        let result = diffCond(
            pred,
            x,
            trueBranch: { input in diffReLU(input) },
            falseBranch: { input in diffSigmoid(input) }
        )
        
        XCTAssertEqual(result.shape, [2])
    }
    
    func testCondDynamicPredicate() {
        // Predicate computed from data
        let x = DifferentiableTracer.placeholder(shape: [4])
        let threshold = DifferentiableTracer.constant(0.5)
        
        let pred = diffMean(x) > threshold  // Dynamic predicate
        
        let result = diffCond(
            pred,
            trueBranch: { diffReLU(x) },
            falseBranch: { diffLeakyReLU(x, alpha: 0.1) }
        )
        
        XCTAssertEqual(result.shape, [4])
    }
    
    // MARK: - diffSelect Tests
    
    func testSelectBasic() {
        let cond = DifferentiableTracer.constant([true, false, true], shape: [3])
        let a = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
        let b = DifferentiableTracer.constant([10.0, 20.0, 30.0], shape: [3])
        
        let result = diffSelect(cond, onTrue: a, onFalse: b)
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [1.0, 20.0, 3.0], accuracy: 1e-6)
    }
    
    func testSelectBroadcast() {
        // Broadcast scalar condition
        let cond = DifferentiableTracer.constant(true)
        let a = DifferentiableTracer.constant([1.0, 2.0], shape: [2])
        let b = DifferentiableTracer.constant([10.0, 20.0], shape: [2])
        
        let result = diffSelect(cond, onTrue: a, onFalse: b)
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [1.0, 2.0], accuracy: 1e-6)
    }
    
    func testSelectMask() {
        // 2D mask
        let mask = DifferentiableTracer.constant(
            [true, false, true, false, true, false],
            shape: [2, 3]
        )
        let a = DifferentiableTracer.constant([1, 2, 3, 4, 5, 6], shape: [2, 3])
        let b = DifferentiableTracer.zeros(shape: [2, 3])
        
        let result = diffSelect(mask, onTrue: a, onFalse: b)
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [1, 0, 3, 0, 5, 0], accuracy: 1e-6)
    }
    
    // MARK: - diffSwitch Tests
    
    func testSwitchBasic() {
        let x = DifferentiableTracer.constant([1.0, 2.0], shape: [2])
        
        for idx in 0..<3 {
            let index = DifferentiableTracer.constant(Int32(idx))
            
            let result = diffSwitch(
                index,
                branches: [
                    { x in x * 1 },
                    { x in x * 2 },
                    { x in x * 3 }
                ],
                operand: x
            )
            
            let output = compileAndExecute(result)
            let expected = [1.0, 2.0].map { $0 * Float(idx + 1) }
            XCTAssertEqual(output, expected, accuracy: 1e-6)
        }
    }
    
    func testSwitchManyBranches() {
        let x = DifferentiableTracer.constant([1.0], shape: [1])
        
        let branches: [(DifferentiableTracer) -> DifferentiableTracer] = (0..<10).map { i in
            { x in x + DifferentiableTracer.constant(Float(i)) }
        }
        
        let index = DifferentiableTracer.constant(Int32(5))
        let result = diffSwitch(index, branches: branches, operand: x)
        
        let output = compileAndExecute(result)
        XCTAssertEqual(output, [6.0], accuracy: 1e-6)  // 1.0 + 5
    }
    
    // MARK: - Gradient Tests
    
    func testCondGradientTrueBranch() {
        let x = DifferentiableTracer.placeholder(shape: [3])
        let pred = DifferentiableTracer.constant(true)
        
        let (value, grad) = valueWithGradient(at: x) { x in
            let result = diffCond(
                pred,
                trueBranch: { x * x },     // Gradient: 2x
                falseBranch: { x * x * x } // Gradient: 3x^2
            )
            return diffSum(result)
        }
        
        // Since pred is true, gradient should be 2x
        XCTAssertEqual(grad.shape, [3])
    }
    
    func testCondGradientFalseBranch() {
        let x = DifferentiableTracer.placeholder(shape: [3])
        let pred = DifferentiableTracer.constant(false)
        
        let (_, grad) = valueWithGradient(at: x) { x in
            let result = diffCond(
                pred,
                trueBranch: { x * x },
                falseBranch: { x * 3 }  // Gradient: 3
            )
            return diffSum(result)
        }
        
        // Since pred is false, gradient should be constant 3
        XCTAssertEqual(grad.shape, [3])
    }
    
    func testSelectGradient() {
        let cond = DifferentiableTracer.constant([true, false, true], shape: [3])
        let a = DifferentiableTracer.placeholder(shape: [3])
        let b = DifferentiableTracer.placeholder(shape: [3])
        
        let (_, (gradA, gradB)) = valueWithGradient(at: a, b) { a, b in
            let result = diffSelect(cond, onTrue: a * 2, onFalse: b * 3)
            return diffSum(result)
        }
        
        // gradA should be [2, 0, 2] (only where cond is true)
        // gradB should be [0, 3, 0] (only where cond is false)
        XCTAssertEqual(gradA.shape, [3])
        XCTAssertEqual(gradB.shape, [3])
    }
    
    func testSelectGradientNumerical() {
        let cond = DifferentiableTracer.constant([true, false, true, false], shape: [4])
        let aValues: [Float] = [1, 2, 3, 4]
        let bValues: [Float] = [5, 6, 7, 8]
        
        let a = DifferentiableTracer.constant(aValues, shape: [4])
        let b = DifferentiableTracer.constant(bValues, shape: [4])
        
        let (_, (analyticalA, analyticalB)) = valueWithGradient(at: a, b) { a, b in
            let result = diffSelect(cond, onTrue: diffSigmoid(a), onFalse: diffTanh(b))
            return diffSum(result)
        }
        
        // Numerical gradient check
        let numGradA = computeNumericalGradient(
            { aVals in
                let a = DifferentiableTracer.constant(aVals, shape: [4])
                let b = DifferentiableTracer.constant(bValues, shape: [4])
                return diffSum(diffSelect(cond, onTrue: diffSigmoid(a), onFalse: diffTanh(b)))
            },
            at: aValues,
            epsilon: 1e-5
        )
        
        let analytical = compileAndExecute(analyticalA)
        XCTAssertEqual(analytical, numGradA, accuracy: 1e-4)
    }
    
    func testSwitchGradient() {
        let x = DifferentiableTracer.placeholder(shape: [2])
        let index = DifferentiableTracer.constant(Int32(1))
        
        let (_, grad) = valueWithGradient(at: x) { x in
            let result = diffSwitch(
                index,
                branches: [
                    { x in x * 2 },
                    { x in x * 3 },
                    { x in x * 4 }
                ],
                operand: x
            )
            return diffSum(result)
        }
        
        // Index is 1, so gradient should be 3
        XCTAssertEqual(grad.shape, [2])
    }
    
    // MARK: - Captured Variables Tests
    
    func testCondWithCaptures() {
        let x = DifferentiableTracer.placeholder(shape: [4])
        let y = DifferentiableTracer.placeholder(shape: [4])
        let pred = DifferentiableTracer.constant(true)
        
        // Both branches capture x and y
        let result = diffCond(
            pred,
            trueBranch: { x + y },
            falseBranch: { x * y }
        )
        
        XCTAssertEqual(result.shape, [4])
        
        // Gradient should flow to both x and y
        let (_, (gradX, gradY)) = valueWithGradient(at: x, y) { x, y in
            let result = diffCond(
                pred,
                trueBranch: { x + y },
                falseBranch: { x * y }
            )
            return diffSum(result)
        }
        
        XCTAssertEqual(gradX.shape, [4])
        XCTAssertEqual(gradY.shape, [4])
    }
    
    // MARK: - Complex Patterns
    
    func testNestedCond() {
        let x = DifferentiableTracer.placeholder(shape: [4])
        let pred1 = DifferentiableTracer.constant(true)
        let pred2 = DifferentiableTracer.constant(false)
        
        let result = diffCond(
            pred1,
            trueBranch: {
                diffCond(
                    pred2,
                    trueBranch: { x * 2 },
                    falseBranch: { x * 3 }
                )
            },
            falseBranch: { x * 4 }
        )
        
        // pred1=true, pred2=false -> x * 3
        XCTAssertEqual(result.shape, [4])
    }
    
    func testCondWithLoop() {
        let x = DifferentiableTracer.placeholder(shape: [4])
        let pred = DifferentiableTracer.constant(true)
        
        let result = diffCond(
            pred,
            trueBranch: {
                // While loop in true branch
                diffWhileLoop(
                    initialState: x,
                    condition: { _, i in i < 5 },
                    body: { state in diffReLU(state * 0.9) }
                )
            },
            falseBranch: { x }
        )
        
        XCTAssertEqual(result.shape, [4])
    }
    
    func testCondWithScan() {
        let xs = DifferentiableTracer.placeholder(shape: [10, 4])
        let pred = DifferentiableTracer.constant(true)
        
        let result = diffCond(
            pred,
            trueBranch: {
                let init = DifferentiableTracer.zeros(shape: [4])
                let (final, _) = diffScan(
                    { carry, x in (carry + x, carry) },
                    init: init,
                    xs: xs
                )
                return final
            },
            falseBranch: {
                diffMean(xs, axis: 0)
            }
        )
        
        XCTAssertEqual(result.shape, [4])
    }
    
    // MARK: - Edge Cases
    
    func testCondEmptyTensor() {
        let pred = DifferentiableTracer.constant(true)
        let x = DifferentiableTracer.placeholder(shape: [0, 4])  // Empty tensor
        
        let result = diffCond(
            pred,
            trueBranch: { x },
            falseBranch: { x * 2 }
        )
        
        XCTAssertEqual(result.shape, [0, 4])
    }
    
    func testCondScalarResult() {
        let pred = DifferentiableTracer.constant(true)
        let x = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
        
        let result = diffCond(
            pred,
            trueBranch: { diffSum(x) },
            falseBranch: { diffMean(x) }
        )
        
        XCTAssertEqual(result.shape, [])  // Scalar
    }
    
    func testSwitchOutOfBounds() {
        let x = DifferentiableTracer.constant([1.0], shape: [1])
        let index = DifferentiableTracer.constant(Int32(5))  // Only 3 branches
        
        // Should handle gracefully (clamp to valid range or error)
        // Implementation-specific behavior
    }
    
    // MARK: - Performance Tests
    
    func testCondCompilationTime() {
        let x = DifferentiableTracer.placeholder(shape: [1000, 1000])
        let pred = DifferentiableTracer.constant(true)
        
        let start = CFAbsoluteTimeGetCurrent()
        
        let result = diffCond(
            pred,
            trueBranch: {
                var y = x
                for _ in 0..<10 {
                    y = diffReLU(diffMatmul(y, DifferentiableTracer.placeholder(shape: [1000, 1000])))
                }
                return y
            },
            falseBranch: {
                var y = x
                for _ in 0..<10 {
                    y = diffGELU(diffMatmul(y, DifferentiableTracer.placeholder(shape: [1000, 1000])))
                }
                return y
            }
        )
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        XCTAssertLessThan(elapsed, 1.0)  // Should compile quickly
    }
    
    // MARK: - StableHLO Generation
    
    func testCondGeneratesIfOp() {
        let pred = DifferentiableTracer.constant(true)
        let x = DifferentiableTracer.placeholder(shape: [4])
        
        let result = diffCond(
            pred,
            trueBranch: { x * 2 },
            falseBranch: { x * 3 }
        )
        
        let mlir = generateStableHLO(result)
        
        XCTAssertTrue(mlir.contains("stablehlo.if"))
    }
    
    func testSelectGeneratesSelectOp() {
        let cond = DifferentiableTracer.placeholder(shape: [4])
        let a = DifferentiableTracer.placeholder(shape: [4])
        let b = DifferentiableTracer.placeholder(shape: [4])
        
        let result = diffSelect(cond, onTrue: a, onFalse: b)
        
        let mlir = generateStableHLO(result)
        
        XCTAssertTrue(mlir.contains("stablehlo.select"))
    }
}
```

## Success Criteria

### Functional Requirements

- [ ] diffCond with static true predicate
- [ ] diffCond with static false predicate
- [ ] diffCond with dynamic predicate
- [ ] diffCond with operand passing
- [ ] diffSelect element-wise conditional
- [ ] diffSelect with broadcasting
- [ ] diffSwitch multi-way branching
- [ ] Gradient through diffCond (both branches)
- [ ] Gradient through diffSelect
- [ ] Gradient through diffSwitch
- [ ] Captured variables in branches

### Shape/Type Requirements

- [ ] Branches must have matching output shapes
- [ ] Predicate must be scalar for diffCond
- [ ] Condition shape must broadcast with value shapes for diffSelect
- [ ] Index must be scalar integer for diffSwitch

### Composition

- [ ] Nested conditionals
- [ ] Conditional + while loop
- [ ] Conditional + scan
- [ ] Conditional + vmap

### StableHLO Generation

- [ ] diffCond generates stablehlo.if
- [ ] diffSelect generates stablehlo.select
- [ ] diffSwitch generates stablehlo.case

## Error Messages

```swift
// Shape mismatch between branches
"diffCond: Branch output shapes must match. True branch outputs shape [8, 4], but false branch outputs [8, 8]."

// Non-scalar predicate for cond
"diffCond: Predicate must be a scalar boolean tensor. Got shape [4]."

// Invalid switch index
"diffSwitch: Index must be a scalar integer. Got shape [2]."

// No branches provided
"diffSwitch: At least one branch must be provided."
```

## Files to Create/Modify

### New Files
- `Sources/SwiftIR/SymbolicAD/Cond.swift` - Main implementation
- `Tests/SwiftIRTests/CondTests.swift` - Test suite

### Modified Files
- `Sources/SwiftIR/SymbolicAD/BackendCompilation.swift` - StableHLO generation for if/select/case
- `Sources/SwiftIR/SymbolicAD/ADIntegration.swift` - Gradient rules for conditionals
