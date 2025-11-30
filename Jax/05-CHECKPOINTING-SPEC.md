# Gradient Checkpointing Specification

## Overview

Gradient checkpointing trades compute for memory by not storing all intermediate activations during the forward pass. Instead, activations are recomputed during the backward pass. This enables training much larger models.

## JAX Reference

```python
@jax.checkpoint
def transformer_block(x):
    # Intermediate activations not stored
    return attention(layer_norm(x)) + x
```

## SwiftIR API Specification

### Primary API

```swift
/// Mark a function for gradient checkpointing
/// Activations are recomputed during backward pass instead of stored
@differentiable(reverse)
public func checkpoint<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    _ input: T
) -> U

/// Property wrapper for checkpointed layers
@propertyWrapper
public struct Checkpointed<Layer: Differentiable> {
    public var wrappedValue: Layer
}
```

### Usage Examples

```swift
// Checkpoint a transformer block
@differentiable(reverse)
func forward(_ x: DifferentiableTracer) -> DifferentiableTracer {
    var h = x
    for block in transformerBlocks {
        h = checkpoint(block.callAsFunction, h)  // Memory-efficient
    }
    return h
}

// Checkpoint policy: every N layers
@differentiable(reverse)
func forwardWithPolicy(_ x: DifferentiableTracer) -> DifferentiableTracer {
    var h = x
    for (i, block) in transformerBlocks.enumerated() {
        if i % 4 == 0 {  // Checkpoint every 4th layer
            h = checkpoint(block.callAsFunction, h)
        } else {
            h = block(h)
        }
    }
    return h
}
```

## Implementation Guide

```swift
@differentiable(reverse)
public func checkpoint<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    _ input: T
) -> U {
    // Forward: just compute, don't record intermediates
    return fn(input)
}

@derivative(of: checkpoint)
func checkpointVJP<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    _ input: T
) -> (value: U, pullback: (U.TangentVector) -> T.TangentVector) {
    // Forward pass - compute output
    let output = fn(input)
    
    return (output, { upstream in
        // Backward pass - recompute forward to get intermediates
        let (_, pullback) = valueWithPullback(at: input, of: fn)
        return pullback(upstream)
    })
}
```

## Test Suite

```swift
final class CheckpointingTests: XCTestCase {
    
    func testCheckpointCorrectness() {
        @differentiable(reverse)
        func expensive(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSigmoid(diffMatmul(x, w1) + diffMatmul(diffReLU(x), w2))
        }
        
        let x = DifferentiableTracer.placeholder(shape: [32, 64])
        
        // Without checkpoint
        let (val1, grad1) = valueWithGradient(at: x) { x in
            diffSum(expensive(x))
        }
        
        // With checkpoint
        let (val2, grad2) = valueWithGradient(at: x) { x in
            diffSum(checkpoint(expensive, x))
        }
        
        // Results should be identical
        XCTAssertEqual(compileAndExecute(val1), compileAndExecute(val2), accuracy: 1e-6)
        XCTAssertEqual(compileAndExecute(grad1), compileAndExecute(grad2), accuracy: 1e-6)
    }
    
    func testCheckpointMemoryReduction() {
        // Verify memory usage is reduced
        // (Implementation-specific measurement)
    }
    
    func testNestedCheckpoint() {
        @differentiable(reverse)
        func inner(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        @differentiable(reverse)
        func outer(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return checkpoint(inner, x) * 2
        }
        
        let result = checkpoint(outer, x)
        // Should work correctly with nested checkpoints
    }
}
```

## Success Criteria

- [ ] Gradient values identical with/without checkpointing
- [ ] Memory reduction of 50%+ for deep networks
- [ ] Compute overhead < 2x (one extra forward pass)
- [ ] Works with all differentiable operations
- [ ] Composable with vmap, scan, cond

## Files to Create

- `Sources/SwiftIR/SymbolicAD/Checkpointing.swift`
- `Tests/SwiftIRTests/CheckpointingTests.swift`
