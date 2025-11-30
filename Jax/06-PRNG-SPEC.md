# Functional PRNG Specification

## Overview

Functional (splittable) PRNGs enable reproducible randomness in parallel/vectorized code. Unlike stateful PRNGs, functional PRNGs are explicit and composable.

## JAX Reference

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
noise = jax.random.normal(key1, shape=(32, 64))
```

## SwiftIR API Specification

### Core Types

```swift
/// Immutable PRNG key (Threefry-based)
public struct PRNGKey: Hashable {
    internal let data: (UInt32, UInt32)
    
    public init(seed: UInt64)
    public func split() -> (PRNGKey, PRNGKey)
    public func split(into n: Int) -> [PRNGKey]
}
```

### Random Operations

```swift
/// Generate uniform random values in [0, 1)
@differentiable(reverse)
public func diffRandomUniform(
    _ key: PRNGKey,
    shape: [Int]
) -> DifferentiableTracer

/// Generate standard normal random values
@differentiable(reverse)
public func diffRandomNormal(
    _ key: PRNGKey,
    shape: [Int],
    mean: Float = 0,
    stddev: Float = 1
) -> DifferentiableTracer

/// Random integers in [minval, maxval)
public func randomInt(
    _ key: PRNGKey,
    shape: [Int],
    minval: Int,
    maxval: Int
) -> DifferentiableTracer

/// Dropout mask
@differentiable(reverse)
public func diffDropout(
    _ key: PRNGKey,
    _ x: DifferentiableTracer,
    rate: Float
) -> DifferentiableTracer

/// Shuffle along axis
public func shuffle(
    _ key: PRNGKey,
    _ x: DifferentiableTracer,
    axis: Int = 0
) -> DifferentiableTracer
```

### Usage Examples

```swift
// Basic usage
let key = PRNGKey(seed: 42)
let (key1, key2) = key.split()
let noise = diffRandomNormal(key1, shape: [32, 64])
let mask = diffRandomUniform(key2, shape: [32, 64])

// In training loop
struct TrainingState {
    var params: Model
    var key: PRNGKey
}

func trainStep(_ state: TrainingState, _ batch: Batch) -> TrainingState {
    let (dropoutKey, newKey) = state.key.split()
    
    let (loss, grad) = valueWithGradient(at: state.params) { params in
        let logits = model(batch.x, dropoutKey: dropoutKey)
        return crossEntropy(logits, batch.y)
    }
    
    let newParams = optimizer.update(state.params, grad)
    return TrainingState(params: newParams, key: newKey)
}

// With vmap - each batch element gets unique key
let keys = key.split(into: batchSize)
let batchedDropout = vmap(diffDropout, inAxes: (0, 0, nil))
let dropped = batchedDropout(keys, x, 0.1)
```

## Implementation Guide

```swift
public struct PRNGKey: Hashable {
    internal let data: (UInt32, UInt32)
    
    public init(seed: UInt64) {
        // Threefry initialization
        self.data = threefryInit(seed)
    }
    
    public func split() -> (PRNGKey, PRNGKey) {
        let (a, b) = threefryRandom2x32(data)
        return (PRNGKey(data: a), PRNGKey(data: b))
    }
    
    public func split(into n: Int) -> [PRNGKey] {
        var keys: [PRNGKey] = []
        var current = self
        for _ in 0..<n {
            let (next, key) = current.split()
            keys.append(key)
            current = next
        }
        return keys
    }
}

// Gradient of random ops: zero (randomness is not differentiable)
@derivative(of: diffRandomNormal)
func diffRandomNormalVJP(
    _ key: PRNGKey,
    shape: [Int],
    mean: Float,
    stddev: Float
) -> (value: DifferentiableTracer, pullback: (DifferentiableTracer) -> ()) {
    return (diffRandomNormal(key, shape: shape, mean: mean, stddev: stddev), { _ in })
}
```

## Test Suite

```swift
final class PRNGTests: XCTestCase {
    
    func testDeterminism() {
        let key1 = PRNGKey(seed: 42)
        let key2 = PRNGKey(seed: 42)
        
        let a = compileAndExecute(diffRandomNormal(key1, shape: [100]))
        let b = compileAndExecute(diffRandomNormal(key2, shape: [100]))
        
        XCTAssertEqual(a, b)  // Same seed = same output
    }
    
    func testSplitIndependence() {
        let key = PRNGKey(seed: 42)
        let (k1, k2) = key.split()
        
        let a = compileAndExecute(diffRandomNormal(k1, shape: [1000]))
        let b = compileAndExecute(diffRandomNormal(k2, shape: [1000]))
        
        // Should be uncorrelated
        let correlation = pearsonCorrelation(a, b)
        XCTAssertLessThan(abs(correlation), 0.1)
    }
    
    func testUniformDistribution() {
        let key = PRNGKey(seed: 42)
        let samples = compileAndExecute(diffRandomUniform(key, shape: [10000]))
        
        XCTAssertGreaterThan(samples.min()!, 0)
        XCTAssertLessThan(samples.max()!, 1)
        XCTAssertEqual(samples.mean(), 0.5, accuracy: 0.01)
    }
    
    func testNormalDistribution() {
        let key = PRNGKey(seed: 42)
        let samples = compileAndExecute(diffRandomNormal(key, shape: [10000]))
        
        XCTAssertEqual(samples.mean(), 0.0, accuracy: 0.05)
        XCTAssertEqual(samples.std(), 1.0, accuracy: 0.05)
    }
    
    func testDropoutRate() {
        let key = PRNGKey(seed: 42)
        let x = DifferentiableTracer.ones(shape: [10000])
        let dropped = compileAndExecute(diffDropout(key, x, rate: 0.3))
        
        let zeroRate = dropped.filter { $0 == 0 }.count / 10000.0
        XCTAssertEqual(zeroRate, 0.3, accuracy: 0.02)
    }
}
```

## Success Criteria

- [ ] Deterministic given same seed
- [ ] Split produces independent streams
- [ ] Correct statistical distributions
- [ ] Works with vmap (per-element keys)
- [ ] Zero gradient through random ops
- [ ] Efficient StableHLO generation

## Files to Create

- `Sources/SwiftIR/SymbolicAD/PRNG.swift`
- `Tests/SwiftIRTests/PRNGTests.swift`
