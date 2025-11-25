# StableHLO While Loop Support for SwiftIR

## Overview

This document outlines the design for native StableHLO `while` loop support with full automatic differentiation in SwiftIR.

## Current Limitation

Currently, loops must be unrolled in Swift before tracing:

```swift
@differentiable(reverse)
func runSimulation(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    var temp = initialTemp

    // UNROLLED: Each iteration becomes separate MLIR operations
    for _ in 0..<20 {
        temp = simulateTimestep(temp)
    }

    return temp
}
```

**Problems**:
- Creates 20 separate function call boundaries
- Prevents XLA loop fusion optimizations
- Requires multiple PJRT round-trips
- No loop-level vectorization

## Proposed API

### User-Facing API

```swift
@differentiable(reverse)
func runSimulation(_ dummy: DifferentiableTracer) -> DifferentiableTracer {
    struct LoopState: Differentiable {
        var iter: DifferentiableTracer  // Loop counter
        var slabTemp: DifferentiableTracer
        var quantaTemp: DifferentiableTracer
        var tankTemp: DifferentiableTracer
    }

    let initial = LoopState(
        iter: createConstant(0.0, shape: [], dtype: .float32),
        slabTemp: createConstant(20.0, shape: [], dtype: .float32),
        quantaTemp: createConstant(20.0, shape: [], dtype: .float32),
        tankTemp: createConstant(70.0, shape: [], dtype: .float32)
    )

    let maxIter = createConstant(20.0, shape: [], dtype: .float32)

    // Single stablehlo.while operation in generated MLIR
    let final = diffWhileLoop(
        initial: initial,
        condition: { state in
            // Condition region: iter < 20
            state.iter < maxIter
        },
        body: { state in
            // Body region: one timestep
            let (newSlab, newQuanta, newTank) = simulateTimestep(
                state.slabTemp,
                state.quantaTemp,
                state.tankTemp
            )
            return LoopState(
                iter: state.iter + createConstant(1.0, shape: [], dtype: .float32),
                slabTemp: newSlab,
                quantaTemp: newQuanta,
                tankTemp: newTank
            )
        }
    )

    return final.slabTemp
}
```

### Generated MLIR

```mlir
func.func @runSimulation(%dummy: tensor<f32>) -> tensor<f32> {
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %c1 = stablehlo.constant dense<1.0> : tensor<f32>
  %c20 = stablehlo.constant dense<20.0> : tensor<f32>
  %init_slab = stablehlo.constant dense<20.0> : tensor<f32>
  %init_quanta = stablehlo.constant dense<20.0> : tensor<f32>
  %init_tank = stablehlo.constant dense<70.0> : tensor<f32>

  // SINGLE stablehlo.while operation
  %iter_final, %slab_final, %quanta_final, %tank_final = stablehlo.while
    (%iter_arg = %c0, %slab_arg = %init_slab, %quanta_arg = %init_quanta, %tank_arg = %init_tank)
    : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  {
    // Condition region
    ^bb0(%iter: tensor<f32>, %slab: tensor<f32>, %quanta: tensor<f32>, %tank: tensor<f32>):
      %cond = stablehlo.compare LT, %iter, %c20 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
  } do {
    // Body region
    ^bb0(%iter: tensor<f32>, %slab: tensor<f32>, %quanta: tensor<f32>, %tank: tensor<f32>):
      // Inline all timestep operations here
      %new_slab, %new_quanta, %new_tank = call @simulateTimestep(%slab, %quanta, %tank)
      %next_iter = stablehlo.add %iter, %c1 : tensor<f32>
      stablehlo.return %next_iter, %new_slab, %new_quanta, %new_tank
        : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }

  return %slab_final : tensor<f32>
}
```

## Implementation Requirements

### 1. StableHLO While Operation Builder

File: `Sources/SwiftIRStableHLO/StablehloOps.swift`

```swift
extension Stablehlo {
    /// Creates a StableHLO while operation
    ///
    /// - Parameters:
    ///   - initialValues: Initial loop carry values
    ///   - conditionBuilder: Builds the condition region (returns i1)
    ///   - bodyBuilder: Builds the body region
    ///   - location: Source location
    ///   - context: MLIR context
    /// - Returns: The while operation
    public static func whileOp(
        initialValues: [MLIRValue],
        conditionBuilder: (MLIRBlock, [MLIRValue]) -> MLIRValue,
        bodyBuilder: (MLIRBlock, [MLIRValue]) -> [MLIRValue],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        // Create operation with two regions (cond and body)
        let resultTypes = initialValues.map { $0.getType() }
        let op = OperationBuilder(name: "stablehlo.while", location: location, context: context)
            .addOperands(initialValues)
            .addResults(resultTypes)
            .addRegions(2)  // Condition + Body
            .build()

        // Build condition region
        let condRegion = op.getRegion(0)
        let condBlock = condRegion.appendBlock()
        for type in resultTypes {
            condBlock.addArgument(type, location)
        }
        let condResult = conditionBuilder(condBlock, condBlock.getArguments())
        // Insert stablehlo.return %condResult

        // Build body region
        let bodyRegion = op.getRegion(1)
        let bodyBlock = bodyRegion.appendBlock()
        for type in resultTypes {
            bodyBlock.addArgument(type, location)
        }
        let bodyResults = bodyBuilder(bodyBlock, bodyBlock.getArguments())
        // Insert stablehlo.return %bodyResults

        return op
    }
}
```

### 2. Differentiable While Loop in SymbolicAD

File: `Sources/SwiftIR/SymbolicAD/DifferentiableWhile.swift`

```swift
/// Differentiable while loop that compiles to stablehlo.while
///
/// This function traces a while loop as a single stablehlo.while operation,
/// enabling XLA to optimize the entire loop.
///
/// - Parameters:
///   - initial: Initial loop state
///   - condition: Loop condition (returns Bool)
///   - body: Loop body (transforms state)
/// - Returns: Final loop state
@differentiable(reverse, wrt: initial)
public func diffWhileLoop<State: Differentiable>(
    initial: State,
    condition: @differentiable (State) -> Bool,
    body: @differentiable (State) -> State
) -> State {
    // During tracing, record a stablehlo.while operation
    if isTracing() {
        return traceWhileLoop(initial: initial, condition: condition, body: body)
    }

    // Fallback: regular Swift while loop (for testing)
    var state = initial
    while condition(state) {
        state = body(state)
    }
    return state
}

// VJP (Vector-Jacobian Product) for reverse-mode AD
@derivative(of: diffWhileLoop)
func _vjpWhileLoop<State: Differentiable>(
    initial: State,
    condition: @differentiable (State) -> Bool,
    body: @differentiable (State) -> State
) -> (value: State, pullback: (State.TangentVector) -> State.TangentVector) {

    // Forward pass: compute final state and save tape
    var state = initial
    var tape: [(state: State, bodyPullback: (State.TangentVector) -> State.TangentVector)] = []

    while condition(state) {
        let (newState, pb) = valueWithPullback(at: state) { s in body(s) }
        tape.append((state: state, bodyPullback: pb))
        state = newState
    }

    // Reverse pass: accumulate gradients backwards through loop
    func pullback(_ v: State.TangentVector) -> State.TangentVector {
        var grad = v
        for (_, bodyPullback) in tape.reversed() {
            grad = bodyPullback(grad)
        }
        return grad
    }

    return (state, pullback)
}
```

### 3. Symbolic Tracing Integration

```swift
func traceWhileLoop<State: Differentiable>(
    initial: State,
    condition: @differentiable (State) -> Bool,
    body: @differentiable (State) -> State
) -> State {
    guard let recorder = DifferentiableTracer.currentRecorder else {
        fatalError("No active tracer for while loop")
    }

    // Flatten State into array of DifferentiableTracers
    let initialTracers = flattenToDifferentiableTracers(initial)

    // Trace condition closure
    let conditionOp = recorder.recordOperation(
        name: "stablehlo.while.condition",
        inputs: initialTracers,
        builder: { condBlock in
            let condResult = condition(rebuildState(condBlock.arguments))
            return [condResult.tracer]
        }
    )

    // Trace body closure
    let bodyOp = recorder.recordOperation(
        name: "stablehlo.while.body",
        inputs: initialTracers,
        builder: { bodyBlock in
            let bodyResult = body(rebuildState(bodyBlock.arguments))
            return flattenToDifferentiableTracers(bodyResult)
        }
    )

    // Create stablehlo.while operation
    let whileOp = Stablehlo.whileOp(
        initialValues: initialTracers.map { $0.value },
        conditionBuilder: conditionOp,
        bodyBuilder: bodyOp,
        location: recorder.location,
        context: recorder.context
    )

    // Wrap results back into State
    let resultTracers = whileOp.getResults().map { DifferentiableTracer($0) }
    return rebuildState(resultTracers)
}
```

## Expected Performance Improvements

### Before (Unrolled Loop)
```
Forward pass:  283.2μs
Gradient:      282.2μs
Overhead:      1.00x
```

**Bottlenecks**:
- 20+ PJRT invocations
- Serialization overhead per iteration
- No loop fusion

### After (stablehlo.while)
```
Forward pass:  15-30μs  (10-20x faster)
Gradient:      15-30μs  (10-20x faster)
Overhead:      1.00x
```

**Improvements**:
- Single PJRT invocation
- One serialization round-trip
- XLA loop fusion optimizations
- Better register allocation

### Crossover Point

- **Small loops (< 10 iterations)**: Standard Swift still faster
- **Medium loops (10-100 iterations)**: SwiftIR competitive
- **Large loops (100+ iterations)**: SwiftIR significantly faster
- **GPU execution**: SwiftIR dramatically faster

## Testing Strategy

1. **Unit tests**: Test while loop tracing without execution
2. **Integration tests**: Simple counted loops (sum 1 to N)
3. **Gradient tests**: Verify VJP correctness
4. **Benchmark**: Building simulation with while loops
5. **XLA inspection**: Verify generated HLO has proper loop fusion

## Migration Path

### Phase 1: API Design (Complete)
Document the desired API and implementation strategy.

### Phase 2: Core Implementation (2-3 weeks)
- Add C++ bindings for stablehlo.while
- Implement region builders
- Basic tracing support

### Phase 3: AD Support (1-2 weeks)
- Implement VJP for while loops
- Tape-based gradient accumulation
- Integration with existing AD system

### Phase 4: Testing & Optimization (1 week)
- Comprehensive test suite
- Performance benchmarks
- XLA optimization verification

### Phase 5: Documentation & Examples (1 week)
- Update building simulation example
- API documentation
- Best practices guide

## References

- [StableHLO While Spec](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while)
- [JAX While Loop](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html)
- [TensorFlow While Loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)
- [MLIR Regions](https://mlir.llvm.org/docs/LangRef/#regions)

## Conclusion

Native StableHLO while loop support is the key to unlocking SwiftIR's full performance potential. While the implementation is non-trivial, the design is well-understood and follows established patterns from JAX and TensorFlow.

The expected 10-20x performance improvement will make SwiftIR competitive with pure Swift for medium-sized workloads and dramatically superior for large-scale simulations and GPU execution.
