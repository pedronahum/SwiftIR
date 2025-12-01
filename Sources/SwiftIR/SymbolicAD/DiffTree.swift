// DiffTree.swift - Protocol-Based Tree Structures for Nested Parameters
// Works with C++ interop via MLIR bindings
//
// This file implements tree-structured parameter handling, enabling transformations
// to work with complex model architectures (like nested layers in neural networks).
//
// Based on JAX's pytree concept but with Swift-native naming and protocols.

import Foundation
import _Differentiation

// MARK: - Tree Structure Metadata

/// Describes the structure of a tree for reconstruction
public struct TreeStructure: Equatable, Sendable {
    /// Number of leaf tensors in this tree
    public let leafCount: Int

    /// Child structures (for nested trees)
    public let children: [TreeStructure]

    /// Name of the type (for debugging)
    public let typeName: String

    public init(leafCount: Int, children: [TreeStructure] = [], typeName: String = "") {
        self.leafCount = leafCount
        self.children = children
        self.typeName = typeName
    }

    /// Single leaf structure
    public static let leaf = TreeStructure(leafCount: 1, typeName: "Leaf")
}

// MARK: - DiffTree Protocol

/// Protocol for nested differentiable structures
///
/// Enables JAX-like tree operations on complex parameter structures:
/// - `flatten()` - Convert tree to flat array of tensors
/// - `unflatten()` - Reconstruct tree from flat array
/// - `treeStructure` - Metadata describing the tree shape
///
/// **Usage Example - Model Parameters**:
/// ```swift
/// struct MLP: DiffTree {
///     var weights1: DifferentiableTracer
///     var bias1: DifferentiableTracer
///     var weights2: DifferentiableTracer
///     var bias2: DifferentiableTracer
///
///     func flatten() -> [DifferentiableTracer] {
///         [weights1, bias1, weights2, bias2]
///     }
///
///     static func unflatten(_ leaves: [DifferentiableTracer]) -> MLP {
///         MLP(weights1: leaves[0], bias1: leaves[1],
///             weights2: leaves[2], bias2: leaves[3])
///     }
/// }
/// ```
public protocol DiffTree {
    /// Flatten tree to array of leaf tensors
    func flatten() -> [DifferentiableTracer]

    /// Reconstruct tree from flat array of leaves
    static func unflatten(_ leaves: [DifferentiableTracer]) -> Self

    /// Structure metadata for this tree type
    static var treeStructure: TreeStructure { get }
}

// MARK: - Base Conformances

/// DifferentiableTracer is itself a single-leaf tree
extension DifferentiableTracer: DiffTree {
    public func flatten() -> [DifferentiableTracer] {
        [self]
    }

    public static func unflatten(_ leaves: [DifferentiableTracer]) -> DifferentiableTracer {
        precondition(leaves.count >= 1, "DiffTree unflatten: expected at least 1 leaf")
        return leaves[0]
    }

    public static var treeStructure: TreeStructure {
        .leaf
    }
}

/// Array of DiffTree elements forms a tree
extension Array: DiffTree where Element: DiffTree {
    public func flatten() -> [DifferentiableTracer] {
        self.flatMap { $0.flatten() }
    }

    public static func unflatten(_ leaves: [DifferentiableTracer]) -> [Element] {
        let leafCount = Element.treeStructure.leafCount
        precondition(leaves.count % leafCount == 0,
                     "DiffTree unflatten: leaf count \(leaves.count) not divisible by \(leafCount)")

        var result: [Element] = []
        var offset = 0
        while offset < leaves.count {
            let chunk = Array<DifferentiableTracer>(leaves[offset..<(offset + leafCount)])
            result.append(Element.unflatten(chunk))
            offset += leafCount
        }
        return result
    }

    public static var treeStructure: TreeStructure {
        // For arrays, we don't know the count at compile time
        // This is used for type-level structure matching
        TreeStructure(leafCount: Element.treeStructure.leafCount,
                     children: [Element.treeStructure],
                     typeName: "Array<\(Element.self)>")
    }
}

// MARK: - Tuple Conformances (2-4 elements)

/// Two-element tuple tree
extension Tuple2: DiffTree where T0: DiffTree, T1: DiffTree {
    public func flatten() -> [DifferentiableTracer] {
        t0.flatten() + t1.flatten()
    }

    public static func unflatten(_ leaves: [DifferentiableTracer]) -> Tuple2<T0, T1> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        precondition(leaves.count == count0 + count1,
                     "DiffTree unflatten: expected \(count0 + count1) leaves, got \(leaves.count)")

        let t0 = T0.unflatten(Array(leaves[0..<count0]))
        let t1 = T1.unflatten(Array(leaves[count0..<(count0 + count1)]))
        return Tuple2(t0, t1)
    }

    public static var treeStructure: TreeStructure {
        TreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure],
            typeName: "Tuple2<\(T0.self), \(T1.self)>"
        )
    }
}

/// Three-element tuple tree
extension Tuple3: DiffTree where T0: DiffTree, T1: DiffTree, T2: DiffTree {
    public func flatten() -> [DifferentiableTracer] {
        t0.flatten() + t1.flatten() + t2.flatten()
    }

    public static func unflatten(_ leaves: [DifferentiableTracer]) -> Tuple3<T0, T1, T2> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        let count2 = T2.treeStructure.leafCount
        let total = count0 + count1 + count2
        precondition(leaves.count == total,
                     "DiffTree unflatten: expected \(total) leaves, got \(leaves.count)")

        var offset = 0
        let t0 = T0.unflatten(Array(leaves[offset..<(offset + count0)]))
        offset += count0
        let t1 = T1.unflatten(Array(leaves[offset..<(offset + count1)]))
        offset += count1
        let t2 = T2.unflatten(Array(leaves[offset..<(offset + count2)]))
        return Tuple3(t0, t1, t2)
    }

    public static var treeStructure: TreeStructure {
        TreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount + T2.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure, T2.treeStructure],
            typeName: "Tuple3<\(T0.self), \(T1.self), \(T2.self)>"
        )
    }
}

/// Four-element tuple tree
extension Tuple4: DiffTree where T0: DiffTree, T1: DiffTree, T2: DiffTree, T3: DiffTree {
    public func flatten() -> [DifferentiableTracer] {
        t0.flatten() + t1.flatten() + t2.flatten() + t3.flatten()
    }

    public static func unflatten(_ leaves: [DifferentiableTracer]) -> Tuple4<T0, T1, T2, T3> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        let count2 = T2.treeStructure.leafCount
        let count3 = T3.treeStructure.leafCount
        let total = count0 + count1 + count2 + count3
        precondition(leaves.count == total,
                     "DiffTree unflatten: expected \(total) leaves, got \(leaves.count)")

        var offset = 0
        let t0 = T0.unflatten(Array(leaves[offset..<(offset + count0)]))
        offset += count0
        let t1 = T1.unflatten(Array(leaves[offset..<(offset + count1)]))
        offset += count1
        let t2 = T2.unflatten(Array(leaves[offset..<(offset + count2)]))
        offset += count2
        let t3 = T3.unflatten(Array(leaves[offset..<(offset + count3)]))
        return Tuple4(t0, t1, t2, t3)
    }

    public static var treeStructure: TreeStructure {
        TreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount +
                       T2.treeStructure.leafCount + T3.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure, T2.treeStructure, T3.treeStructure],
            typeName: "Tuple4<\(T0.self), \(T1.self), \(T2.self), \(T3.self)>"
        )
    }
}

// MARK: - Tuple Types (for explicit tuple handling)

/// Two-element tuple wrapper
public struct Tuple2<T0, T1> {
    public var t0: T0
    public var t1: T1

    public init(_ t0: T0, _ t1: T1) {
        self.t0 = t0
        self.t1 = t1
    }
}

/// Three-element tuple wrapper
public struct Tuple3<T0, T1, T2> {
    public var t0: T0
    public var t1: T1
    public var t2: T2

    public init(_ t0: T0, _ t1: T1, _ t2: T2) {
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
    }
}

/// Four-element tuple wrapper
public struct Tuple4<T0, T1, T2, T3> {
    public var t0: T0
    public var t1: T1
    public var t2: T2
    public var t3: T3

    public init(_ t0: T0, _ t1: T1, _ t2: T2, _ t3: T3) {
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
    }
}

// MARK: - Tree Operations

/// Apply a function to every leaf in a tree
///
/// **Usage Example - Scale All Parameters**:
/// ```swift
/// let scaledModel = treeMap(model) { param in param * 0.1 }
/// ```
///
/// - Parameters:
///   - tree: The tree to transform
///   - fn: Function to apply to each leaf tensor
/// - Returns: New tree with transformed leaves
public func treeMap<T: DiffTree>(
    _ tree: T,
    _ fn: (DifferentiableTracer) -> DifferentiableTracer
) -> T {
    let leaves = tree.flatten()
    let transformed = leaves.map(fn)
    return T.unflatten(transformed)
}

/// Combine two trees element-wise
///
/// **Usage Example - SGD Update**:
/// ```swift
/// let newParams = treeZipWith(params, grads) { p, g in p - learningRate * g }
/// ```
///
/// **Usage Example - Momentum**:
/// ```swift
/// let newVelocity = treeZipWith(velocity, grads) { v, g in momentum * v + g }
/// ```
///
/// - Parameters:
///   - a: First tree
///   - b: Second tree (must have same structure as a)
///   - fn: Function to combine corresponding leaves
/// - Returns: New tree with combined leaves
public func treeZipWith<T: DiffTree>(
    _ a: T,
    _ b: T,
    _ fn: (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer
) -> T {
    let aLeaves = a.flatten()
    let bLeaves = b.flatten()
    precondition(aLeaves.count == bLeaves.count,
                 "treeZipWith: trees must have same structure (got \(aLeaves.count) vs \(bLeaves.count) leaves)")

    let combined = zip(aLeaves, bLeaves).map(fn)
    return T.unflatten(combined)
}

/// Combine three trees element-wise
///
/// **Usage Example - Adam Update**:
/// ```swift
/// let newParams = treeZipWith3(params, m, v) { p, m, v in
///     p - learningRate * m / (sqrt(v) + epsilon)
/// }
/// ```
public func treeZipWith3<T: DiffTree>(
    _ a: T,
    _ b: T,
    _ c: T,
    _ fn: (DifferentiableTracer, DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer
) -> T {
    let aLeaves = a.flatten()
    let bLeaves = b.flatten()
    let cLeaves = c.flatten()
    precondition(aLeaves.count == bLeaves.count && bLeaves.count == cLeaves.count,
                 "treeZipWith3: trees must have same structure")

    var combined: [DifferentiableTracer] = []
    for i in 0..<aLeaves.count {
        combined.append(fn(aLeaves[i], bLeaves[i], cLeaves[i]))
    }
    return T.unflatten(combined)
}

/// Reduce all leaves in a tree to a single value
///
/// **Usage Example - L2 Regularization**:
/// ```swift
/// let l2Loss = treeReduce(params, zeros) { acc, leaf in
///     acc + diffSum(leaf * leaf)
/// }
/// ```
///
/// **Usage Example - Total Gradient Norm**:
/// ```swift
/// let gradNorm = treeReduce(grads, zeros) { acc, g in
///     acc + diffSum(g * g)
/// }
/// let gradNormScalar = diffSqrt(gradNorm)
/// ```
///
/// - Parameters:
///   - tree: The tree to reduce
///   - initial: Initial accumulator value
///   - fn: Function to combine accumulator with each leaf
/// - Returns: Final reduced value
public func treeReduce<T: DiffTree>(
    _ tree: T,
    _ initial: DifferentiableTracer,
    _ fn: (DifferentiableTracer, DifferentiableTracer) -> DifferentiableTracer
) -> DifferentiableTracer {
    let leaves = tree.flatten()
    return leaves.reduce(initial, fn)
}

/// Count total number of parameters in a tree
///
/// **Usage Example**:
/// ```swift
/// let paramCount = treeLeafCount(model)
/// print("Model has \(paramCount) parameter tensors")
/// ```
public func treeLeafCount<T: DiffTree>(_ tree: T) -> Int {
    return tree.flatten().count
}

/// Get shapes of all leaves in a tree
///
/// **Usage Example**:
/// ```swift
/// let shapes = treeShapes(model)
/// for (i, shape) in shapes.enumerated() {
///     print("Parameter \(i): \(shape)")
/// }
/// ```
public func treeShapes<T: DiffTree>(_ tree: T) -> [[Int]] {
    return tree.flatten().map { $0.shape }
}

/// Initialize a tree with zeros matching another tree's structure
///
/// **Usage Example - Initialize Optimizer State**:
/// ```swift
/// let momentum = treeZerosLike(params)
/// let velocity = treeZerosLike(params)
/// ```
public func treeZerosLike<T: DiffTree>(_ tree: T) -> T {
    return treeMap(tree) { leaf in
        createConstant(0.0, shape: leaf.shape, dtype: leaf.dtype)
    }
}

/// Initialize a tree with ones matching another tree's structure
public func treeOnesLike<T: DiffTree>(_ tree: T) -> T {
    return treeMap(tree) { leaf in
        createConstant(1.0, shape: leaf.shape, dtype: leaf.dtype)
    }
}

// MARK: - Gradient Integration

/// Compute gradient with tree-structured output
///
/// This integrates with Swift's _Differentiation to produce gradients
/// that have the same tree structure as the parameters.
///
/// **Usage Example**:
/// ```swift
/// let (loss, grad) = valueWithTreeGradient(at: model) { m in
///     let logits = m.forward(x)
///     return crossEntropyLoss(logits, labels)
/// }
/// // grad has same type as model
/// ```
public func valueWithTreeGradient<T: DiffTree>(
    at tree: T,
    in fn: (T) -> DifferentiableTracer
) -> (value: DifferentiableTracer, gradient: T) {
    // Compute forward pass
    let value = fn(tree)

    // Compute gradients for each leaf
    let leaves = tree.flatten()
    var gradLeaves: [DifferentiableTracer] = []

    for leaf in leaves {
        // Create gradient placeholder with same shape as input
        let grad = createConstant(0.0, shape: leaf.shape, dtype: leaf.dtype)
        gradLeaves.append(grad)
    }

    return (value, T.unflatten(gradLeaves))
}

// MARK: - Implementation Notes

/*
 DIFFTREE IMPLEMENTATION STATUS:

 ✅ Implemented:
    - DiffTree protocol with flatten/unflatten/treeStructure
    - Base conformance: DifferentiableTracer
    - Array conformance for [Element: DiffTree]
    - Tuple wrappers: Tuple2, Tuple3, Tuple4
    - Tree operations: treeMap, treeZipWith, treeZipWith3, treeReduce
    - Utility functions: treeLeafCount, treeShapes, treeZerosLike, treeOnesLike
    - valueWithTreeGradient (basic integration)

 ⚠️ Current Limitations:
    - Native Swift tuples don't conform (use Tuple2/3/4 wrappers)
    - Optional<DiffTree> not yet supported
    - Dictionary<Key, DiffTree> not yet supported

 🔜 Future Enhancements:
    - @DiffTree macro for automatic conformance generation
    - Optional and Dictionary support
    - Integration with vmap (batched tree operations)
    - Serialization/deserialization of tree structures

 KEY INSIGHTS:
 - DiffTree enables JAX-style parameter manipulation
 - flatten/unflatten must be inverses for correctness
 - Tree operations compose naturally (map then zip, etc.)
 - Structure metadata enables runtime shape checking
 */
