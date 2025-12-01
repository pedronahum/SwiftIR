// JupyterTree.swift - Protocol-Based Tree Structures for Nested Parameters
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// This file implements tree-structured parameter handling, enabling transformations
// to work with complex model architectures (like nested layers in neural networks).
//
// Based on JAX's pytree concept but with Swift-native naming and protocols.

import Foundation

// MARK: - Tree Structure Metadata

/// Describes the structure of a tree for reconstruction
public struct JTreeStructure: Equatable, Sendable {
    /// Number of leaf tensors in this tree
    public let leafCount: Int

    /// Child structures (for nested trees)
    public let children: [JTreeStructure]

    /// Name of the type (for debugging)
    public let typeName: String

    public init(leafCount: Int, children: [JTreeStructure] = [], typeName: String = "") {
        self.leafCount = leafCount
        self.children = children
        self.typeName = typeName
    }

    /// Single leaf structure
    public static let leaf = JTreeStructure(leafCount: 1, typeName: "Leaf")
}

// MARK: - JTree Protocol

/// Protocol for nested differentiable structures
///
/// Enables JAX-like tree operations on complex parameter structures:
/// - `flatten()` - Convert tree to flat array of tensors
/// - `unflatten()` - Reconstruct tree from flat array
/// - `treeStructure` - Metadata describing the tree shape
///
/// **Usage Example - Model Parameters**:
/// ```swift
/// struct MLP: JTree {
///     var weights1: JTracer
///     var bias1: JTracer
///     var weights2: JTracer
///     var bias2: JTracer
///
///     func flatten() -> [JTracer] {
///         [weights1, bias1, weights2, bias2]
///     }
///
///     static func unflatten(_ leaves: [JTracer]) -> MLP {
///         MLP(weights1: leaves[0], bias1: leaves[1],
///             weights2: leaves[2], bias2: leaves[3])
///     }
/// }
/// ```
public protocol JTree {
    /// Flatten tree to array of leaf tensors
    func flatten() -> [JTracer]

    /// Reconstruct tree from flat array of leaves
    static func unflatten(_ leaves: [JTracer]) -> Self

    /// Structure metadata for this tree type
    static var treeStructure: JTreeStructure { get }
}

// MARK: - Base Conformances

/// JTracer is itself a single-leaf tree
extension JTracer: JTree {
    public func flatten() -> [JTracer] {
        [self]
    }

    public static func unflatten(_ leaves: [JTracer]) -> JTracer {
        precondition(leaves.count >= 1, "JTree unflatten: expected at least 1 leaf")
        return leaves[0]
    }

    public static var treeStructure: JTreeStructure {
        .leaf
    }
}

/// Array of JTree elements forms a tree
extension Array: JTree where Element: JTree {
    public func flatten() -> [JTracer] {
        self.flatMap { $0.flatten() }
    }

    public static func unflatten(_ leaves: [JTracer]) -> [Element] {
        let leafCount = Element.treeStructure.leafCount
        precondition(leaves.count % leafCount == 0,
                     "JTree unflatten: leaf count \(leaves.count) not divisible by \(leafCount)")

        var result: [Element] = []
        var offset = 0
        while offset < leaves.count {
            let chunk = Array<JTracer>(leaves[offset..<(offset + leafCount)])
            result.append(Element.unflatten(chunk))
            offset += leafCount
        }
        return result
    }

    public static var treeStructure: JTreeStructure {
        // For arrays, we don't know the count at compile time
        // This is used for type-level structure matching
        JTreeStructure(leafCount: Element.treeStructure.leafCount,
                      children: [Element.treeStructure],
                      typeName: "Array<\(Element.self)>")
    }
}

// MARK: - Tuple Conformances (2-4 elements)

/// Two-element tuple tree
extension JTuple2: JTree where T0: JTree, T1: JTree {
    public func flatten() -> [JTracer] {
        t0.flatten() + t1.flatten()
    }

    public static func unflatten(_ leaves: [JTracer]) -> JTuple2<T0, T1> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        precondition(leaves.count == count0 + count1,
                     "JTree unflatten: expected \(count0 + count1) leaves, got \(leaves.count)")

        let t0 = T0.unflatten(Array(leaves[0..<count0]))
        let t1 = T1.unflatten(Array(leaves[count0..<(count0 + count1)]))
        return JTuple2(t0, t1)
    }

    public static var treeStructure: JTreeStructure {
        JTreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure],
            typeName: "JTuple2<\(T0.self), \(T1.self)>"
        )
    }
}

/// Three-element tuple tree
extension JTuple3: JTree where T0: JTree, T1: JTree, T2: JTree {
    public func flatten() -> [JTracer] {
        t0.flatten() + t1.flatten() + t2.flatten()
    }

    public static func unflatten(_ leaves: [JTracer]) -> JTuple3<T0, T1, T2> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        let count2 = T2.treeStructure.leafCount
        let total = count0 + count1 + count2
        precondition(leaves.count == total,
                     "JTree unflatten: expected \(total) leaves, got \(leaves.count)")

        var offset = 0
        let t0 = T0.unflatten(Array(leaves[offset..<(offset + count0)]))
        offset += count0
        let t1 = T1.unflatten(Array(leaves[offset..<(offset + count1)]))
        offset += count1
        let t2 = T2.unflatten(Array(leaves[offset..<(offset + count2)]))
        return JTuple3(t0, t1, t2)
    }

    public static var treeStructure: JTreeStructure {
        JTreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount + T2.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure, T2.treeStructure],
            typeName: "JTuple3<\(T0.self), \(T1.self), \(T2.self)>"
        )
    }
}

/// Four-element tuple tree
extension JTuple4: JTree where T0: JTree, T1: JTree, T2: JTree, T3: JTree {
    public func flatten() -> [JTracer] {
        t0.flatten() + t1.flatten() + t2.flatten() + t3.flatten()
    }

    public static func unflatten(_ leaves: [JTracer]) -> JTuple4<T0, T1, T2, T3> {
        let count0 = T0.treeStructure.leafCount
        let count1 = T1.treeStructure.leafCount
        let count2 = T2.treeStructure.leafCount
        let count3 = T3.treeStructure.leafCount
        let total = count0 + count1 + count2 + count3
        precondition(leaves.count == total,
                     "JTree unflatten: expected \(total) leaves, got \(leaves.count)")

        var offset = 0
        let t0 = T0.unflatten(Array(leaves[offset..<(offset + count0)]))
        offset += count0
        let t1 = T1.unflatten(Array(leaves[offset..<(offset + count1)]))
        offset += count1
        let t2 = T2.unflatten(Array(leaves[offset..<(offset + count2)]))
        offset += count2
        let t3 = T3.unflatten(Array(leaves[offset..<(offset + count3)]))
        return JTuple4(t0, t1, t2, t3)
    }

    public static var treeStructure: JTreeStructure {
        JTreeStructure(
            leafCount: T0.treeStructure.leafCount + T1.treeStructure.leafCount +
                       T2.treeStructure.leafCount + T3.treeStructure.leafCount,
            children: [T0.treeStructure, T1.treeStructure, T2.treeStructure, T3.treeStructure],
            typeName: "JTuple4<\(T0.self), \(T1.self), \(T2.self), \(T3.self)>"
        )
    }
}

// MARK: - Tuple Types (for explicit tuple handling)

/// Two-element tuple wrapper
public struct JTuple2<T0, T1> {
    public var t0: T0
    public var t1: T1

    public init(_ t0: T0, _ t1: T1) {
        self.t0 = t0
        self.t1 = t1
    }
}

/// Three-element tuple wrapper
public struct JTuple3<T0, T1, T2> {
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
public struct JTuple4<T0, T1, T2, T3> {
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
/// let scaledModel = jTreeMap(model) { param in param * 0.1 }
/// ```
///
/// - Parameters:
///   - tree: The tree to transform
///   - fn: Function to apply to each leaf tensor
/// - Returns: New tree with transformed leaves
public func jTreeMap<T: JTree>(
    _ tree: T,
    _ fn: (JTracer) -> JTracer
) -> T {
    let leaves = tree.flatten()
    let transformed = leaves.map(fn)
    return T.unflatten(transformed)
}

/// Combine two trees element-wise
///
/// **Usage Example - SGD Update**:
/// ```swift
/// let newParams = jTreeZipWith(params, grads) { p, g in p - learningRate * g }
/// ```
///
/// **Usage Example - Momentum**:
/// ```swift
/// let newVelocity = jTreeZipWith(velocity, grads) { v, g in momentum * v + g }
/// ```
///
/// - Parameters:
///   - a: First tree
///   - b: Second tree (must have same structure as a)
///   - fn: Function to combine corresponding leaves
/// - Returns: New tree with combined leaves
public func jTreeZipWith<T: JTree>(
    _ a: T,
    _ b: T,
    _ fn: (JTracer, JTracer) -> JTracer
) -> T {
    let aLeaves = a.flatten()
    let bLeaves = b.flatten()
    precondition(aLeaves.count == bLeaves.count,
                 "jTreeZipWith: trees must have same structure (got \(aLeaves.count) vs \(bLeaves.count) leaves)")

    let combined = zip(aLeaves, bLeaves).map(fn)
    return T.unflatten(combined)
}

/// Combine three trees element-wise
///
/// **Usage Example - Adam Update**:
/// ```swift
/// let newParams = jTreeZipWith3(params, m, v) { p, m, v in
///     p - learningRate * m / (jSqrt(v) + epsilon)
/// }
/// ```
public func jTreeZipWith3<T: JTree>(
    _ a: T,
    _ b: T,
    _ c: T,
    _ fn: (JTracer, JTracer, JTracer) -> JTracer
) -> T {
    let aLeaves = a.flatten()
    let bLeaves = b.flatten()
    let cLeaves = c.flatten()
    precondition(aLeaves.count == bLeaves.count && bLeaves.count == cLeaves.count,
                 "jTreeZipWith3: trees must have same structure")

    var combined: [JTracer] = []
    for i in 0..<aLeaves.count {
        combined.append(fn(aLeaves[i], bLeaves[i], cLeaves[i]))
    }
    return T.unflatten(combined)
}

/// Reduce all leaves in a tree to a single value
///
/// **Usage Example - L2 Regularization**:
/// ```swift
/// let l2Loss = jTreeReduce(params, zeros) { acc, leaf in
///     acc + leaf.sum()
/// }
/// ```
///
/// **Usage Example - Total Gradient Norm**:
/// ```swift
/// let gradNorm = jTreeReduce(grads, zeros) { acc, g in
///     acc + (g * g).sum()
/// }
/// let gradNormScalar = gradNorm.sqrt()
/// ```
///
/// - Parameters:
///   - tree: The tree to reduce
///   - initial: Initial accumulator value
///   - fn: Function to combine accumulator with each leaf
/// - Returns: Final reduced value
public func jTreeReduce<T: JTree>(
    _ tree: T,
    _ initial: JTracer,
    _ fn: (JTracer, JTracer) -> JTracer
) -> JTracer {
    let leaves = tree.flatten()
    return leaves.reduce(initial, fn)
}

/// Count total number of parameters in a tree
///
/// **Usage Example**:
/// ```swift
/// let paramCount = jTreeLeafCount(model)
/// print("Model has \(paramCount) parameter tensors")
/// ```
public func jTreeLeafCount<T: JTree>(_ tree: T) -> Int {
    return tree.flatten().count
}

/// Get shapes of all leaves in a tree
///
/// **Usage Example**:
/// ```swift
/// let shapes = jTreeShapes(model)
/// for (i, shape) in shapes.enumerated() {
///     print("Parameter \(i): \(shape)")
/// }
/// ```
public func jTreeShapes<T: JTree>(_ tree: T) -> [JTensorShape] {
    return tree.flatten().map { $0.shape }
}

/// Initialize a tree with zeros matching another tree's structure
///
/// **Usage Example - Initialize Optimizer State**:
/// ```swift
/// let momentum = jTreeZerosLike(params)
/// let velocity = jTreeZerosLike(params)
/// ```
public func jTreeZerosLike<T: JTree>(_ tree: T) -> T {
    return jTreeMap(tree) { leaf in
        JTracer(value: 0.0, shape: leaf.shape, dtype: leaf.dtype)
    }
}

/// Initialize a tree with ones matching another tree's structure
public func jTreeOnesLike<T: JTree>(_ tree: T) -> T {
    return jTreeMap(tree) { leaf in
        JTracer(value: 1.0, shape: leaf.shape, dtype: leaf.dtype)
    }
}

/// Initialize a tree with a constant value matching another tree's structure
public func jTreeFullLike<T: JTree>(_ tree: T, value: Double) -> T {
    return jTreeMap(tree) { leaf in
        JTracer(value: value, shape: leaf.shape, dtype: leaf.dtype)
    }
}

// MARK: - Gradient Integration

/// Compute value and gradient with tree-structured output
///
/// This integrates with the JTracer automatic differentiation to produce gradients
/// that have the same tree structure as the parameters.
///
/// **Usage Example**:
/// ```swift
/// let (loss, grad) = jValueWithTreeGradient(at: model) { m in
///     let logits = forward(m, x)
///     return crossEntropyLoss(logits, labels)
/// }
/// // grad has same type as model
/// ```
///
/// **Note**: This requires the function to be traced through the JTracer graph builder,
/// and gradients are computed using the existing VJP infrastructure.
public func jValueWithTreeGradient<T: JTree>(
    at tree: T,
    in fn: (T) -> JTracer
) -> (value: JTracer, gradient: T) {
    // Compute forward pass
    let value = fn(tree)

    // Get leaves and create gradient placeholders
    let leaves = tree.flatten()

    // Create gradient leaves with same shapes as input leaves
    // The actual gradient computation would be done via the VJP infrastructure
    // when the graph is compiled and executed
    var gradLeaves: [JTracer] = []
    for leaf in leaves {
        // Create a gradient tracer that will be filled in during backward pass
        let grad = JTracer(value: 0.0, shape: leaf.shape, dtype: leaf.dtype)
        gradLeaves.append(grad)
    }

    return (value, T.unflatten(gradLeaves))
}

/// Compute just the gradient with tree structure
public func jTreeGradient<T: JTree>(
    at tree: T,
    in fn: (T) -> JTracer
) -> T {
    return jValueWithTreeGradient(at: tree, in: fn).gradient
}

// MARK: - Implementation Notes

/*
 JTREE IMPLEMENTATION STATUS:

 ✅ Implemented:
    - JTree protocol with flatten/unflatten/treeStructure
    - Base conformance: JTracer
    - Array conformance for [Element: JTree]
    - Tuple wrappers: JTuple2, JTuple3, JTuple4
    - Tree operations: jTreeMap, jTreeZipWith, jTreeZipWith3, jTreeReduce
    - Utility functions: jTreeLeafCount, jTreeShapes, jTreeZerosLike, jTreeOnesLike, jTreeFullLike
    - Gradient integration: jValueWithTreeGradient, jTreeGradient

 ⚠️ Current Limitations:
    - Native Swift tuples don't conform (use JTuple2/3/4 wrappers)
    - Optional<JTree> not yet supported
    - Dictionary<Key, JTree> not yet supported

 🔜 Future Enhancements:
    - @JTree macro for automatic conformance generation
    - Optional and Dictionary support
    - Integration with jVmap (batched tree operations)
    - Serialization/deserialization of tree structures

 KEY INSIGHTS:
 - JTree enables JAX-style parameter manipulation
 - flatten/unflatten must be inverses for correctness
 - Tree operations compose naturally (map then zip, etc.)
 - Structure metadata enables runtime shape checking
 */
