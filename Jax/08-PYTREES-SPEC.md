# Protocol-Based Pytrees Specification

## Overview

Pytrees provide a protocol for working with nested parameter structures, enabling transformations to work with complex model architectures.

## SwiftIR API

```swift
/// Protocol for nested differentiable structures
public protocol DiffTree: Differentiable {
    func flatten() -> [DifferentiableTracer]
    static func unflatten(_ leaves: [DifferentiableTracer]) -> Self
    static var treeStructure: TreeStructure { get }
}

/// Tree operations
public func treeMap<T: DiffTree>(_ tree: T, _ fn: (DifferentiableTracer) -> DifferentiableTracer) -> T
public func treeZipWith<T: DiffTree>(_ a: T, _ b: T, _ fn: (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer) -> T
public func treeReduce<T: DiffTree>(_ tree: T, _ init: DifferentiableTracer, _ fn: (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer) -> DifferentiableTracer
```

## Usage Examples

```swift
@DiffTree
struct MLP: Differentiable {
    var layer1: Linear<D784, D256>
    var layer2: Linear<D256, D10>
}

// Gradient has same structure
let grad = gradient(at: model) { m in loss(m(x), y) }

// SGD update
let newParams = treeZipWith(params, grads) { p, g in p - lr * g }

// L2 regularization
let l2 = treeReduce(params, zeros) { acc, leaf in acc + diffSum(leaf * leaf) }

// Parameter count
let count = model.flatten().map { $0.shape.reduce(1, *) }.reduce(0, +)
```

## Implementation

```swift
public func treeMap<T: DiffTree>(_ tree: T, _ fn: (DifferentiableTracer) -> DifferentiableTracer) -> T {
    return T.unflatten(tree.flatten().map(fn))
}

public func treeZipWith<T: DiffTree>(_ a: T, _ b: T, _ fn: (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer) -> T {
    let aLeaves = a.flatten()
    let bLeaves = b.flatten()
    precondition(aLeaves.count == bLeaves.count)
    return T.unflatten(zip(aLeaves, bLeaves).map(fn))
}

// Base conformances
extension DifferentiableTracer: DiffTree {
    public func flatten() -> [DifferentiableTracer] { [self] }
    public static func unflatten(_ leaves: [DifferentiableTracer]) -> Self { leaves[0] }
}

extension Array: DiffTree where Element: DiffTree {
    public func flatten() -> [DifferentiableTracer] { flatMap { $0.flatten() } }
    // unflatten chunks by Element.treeStructure.leafCount
}
```

## Test Suite

```swift
final class PytreeTests: XCTestCase {
    func testFlattenUnflatten() {
        let model = MLP()
        let leaves = model.flatten()
        let reconstructed = MLP.unflatten(leaves)
        // Should be identical
    }
    
    func testTreeMap() {
        let model = MLP()
        let doubled = treeMap(model) { $0 * 2 }
        // All parameters doubled
    }
    
    func testGradientStructure() {
        let grad = gradient(at: model) { m in loss(m(x), y) }
        XCTAssertEqual(model.flatten().count, grad.flatten().count)
    }
}
```

## Success Criteria

- [ ] flatten/unflatten roundtrip preserves values
- [ ] Gradient has identical structure to parameters
- [ ] treeMap/treeZipWith work correctly
- [ ] Works with nested structures (arrays, tuples)
- [ ] Integrates with vmap, scan, grad

## Files to Create

- `Sources/SwiftIR/Pytrees/DiffTree.swift`
- `Sources/SwiftIR/Pytrees/TreeOperations.swift`
- `Tests/SwiftIRTests/PytreeTests.swift`
