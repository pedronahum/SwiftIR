/// Optimization Pipeline - Graph optimization passes for SwiftIR
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides optimization passes for the traced computation graph:
/// - Constant folding
/// - Dead code elimination
/// - Common subexpression elimination
/// - Operation fusion

import Foundation

// MARK: - Optimization Pass Protocol

/// Protocol for optimization passes
public protocol OptimizationPass: Sendable {
    /// Name of the pass for debugging
    var name: String { get }

    /// Run the optimization pass on a graph
    /// - Parameter graph: The graph to optimize
    /// - Returns: The optimized graph
    func run(on graph: OptimizationGraph) -> OptimizationGraph
}

// MARK: - Optimization Graph

/// Graph representation for optimization
public final class OptimizationGraph: @unchecked Sendable {
    private let lock = NSLock()

    /// Nodes in the graph
    public private(set) var nodes: [UInt64: OptimizationNode]

    /// Output node IDs
    public var outputs: [UInt64]

    /// Input node IDs
    public var inputs: [UInt64]

    public init(nodes: [UInt64: OptimizationNode] = [:], outputs: [UInt64] = [], inputs: [UInt64] = []) {
        self.nodes = nodes
        self.outputs = outputs
        self.inputs = inputs
    }

    /// Deep copy the graph
    public func copy() -> OptimizationGraph {
        lock.lock()
        defer { lock.unlock() }

        var copiedNodes: [UInt64: OptimizationNode] = [:]
        for (id, node) in nodes {
            copiedNodes[id] = node.copy()
        }
        return OptimizationGraph(nodes: copiedNodes, outputs: outputs, inputs: inputs)
    }

    /// Add a node to the graph
    public func addNode(_ node: OptimizationNode) {
        lock.lock()
        defer { lock.unlock() }
        nodes[node.id] = node
    }

    /// Remove a node from the graph
    public func removeNode(_ id: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        nodes.removeValue(forKey: id)
    }

    /// Update a node in the graph
    public func updateNode(_ id: UInt64, with node: OptimizationNode) {
        lock.lock()
        defer { lock.unlock() }
        nodes[id] = node
    }

    /// Get users of a node (nodes that use this node as input)
    public func getUsers(of nodeId: UInt64) -> [UInt64] {
        lock.lock()
        defer { lock.unlock() }

        var users: [UInt64] = []
        for (id, node) in nodes {
            if node.inputs.contains(nodeId) {
                users.append(id)
            }
        }
        return users
    }

    /// Check if a node is used
    public func isUsed(_ nodeId: UInt64) -> Bool {
        if outputs.contains(nodeId) {
            return true
        }
        return !getUsers(of: nodeId).isEmpty
    }

    /// Replace all uses of oldId with newId
    public func replaceAllUses(of oldId: UInt64, with newId: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        for (id, node) in nodes {
            if node.inputs.contains(oldId) {
                nodes[id] = OptimizationNode(
                    id: node.id,
                    operation: node.operation,
                    inputs: node.inputs.map { $0 == oldId ? newId : $0 },
                    shape: node.shape,
                    dtype: node.dtype,
                    attributes: node.attributes
                )
            }
        }

        // Update outputs
        outputs = outputs.map { $0 == oldId ? newId : $0 }
    }

    /// Get node count
    public var nodeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return nodes.count
    }
}

// MARK: - Optimization Node

/// Node in the optimization graph
public struct OptimizationNode: Sendable {
    public let id: UInt64
    public let operation: NodeOperation
    public let inputs: [UInt64]
    public let shape: TensorShape
    public let dtype: DType
    public let attributes: [String: NodeAttribute]

    public init(
        id: UInt64,
        operation: NodeOperation,
        inputs: [UInt64],
        shape: TensorShape,
        dtype: DType,
        attributes: [String: NodeAttribute] = [:]
    ) {
        self.id = id
        self.operation = operation
        self.inputs = inputs
        self.shape = shape
        self.dtype = dtype
        self.attributes = attributes
    }

    public func copy() -> OptimizationNode {
        return OptimizationNode(
            id: id,
            operation: operation,
            inputs: inputs,
            shape: shape,
            dtype: dtype,
            attributes: attributes
        )
    }
}

/// Node operations
public enum NodeOperation: String, Sendable, Hashable {
    // Constants
    case constant
    case placeholder

    // Binary ops
    case add
    case subtract
    case multiply
    case divide
    case matmul

    // Unary ops
    case exp
    case log
    case sqrt
    case abs
    case neg
    case tanh
    case sigmoid
    case relu
    case transpose

    // Reductions
    case sum
    case mean
    case max
    case min

    // Fused ops
    case fusedMatmulAdd
    case fusedMulAdd
    case fusedReluAdd
}

/// Node attributes
public enum NodeAttribute: Sendable, Hashable {
    case double(Double)
    case int(Int)
    case bool(Bool)
    case string(String)
    case intArray([Int])
    case axes(Set<Int>)
}

// MARK: - Constant Folding Pass

/// Constant folding optimization pass
public struct ConstantFoldingPass: OptimizationPass {
    public var name: String { "ConstantFolding" }

    public init() {}

    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        let result = graph.copy()
        var changed = true

        while changed {
            changed = false

            for (id, node) in result.nodes {
                if canFold(node, in: result) {
                    if let folded = fold(node, in: result) {
                        result.updateNode(id, with: folded)
                        changed = true
                    }
                }
            }
        }

        return result
    }

    private func canFold(_ node: OptimizationNode, in graph: OptimizationGraph) -> Bool {
        // Can fold if all inputs are constants
        guard !node.inputs.isEmpty else { return false }
        guard node.operation != .constant && node.operation != .placeholder else { return false }

        return node.inputs.allSatisfy { inputId in
            graph.nodes[inputId]?.operation == .constant
        }
    }

    private func fold(_ node: OptimizationNode, in graph: OptimizationGraph) -> OptimizationNode? {
        // Get constant values
        var values: [Double] = []
        for inputId in node.inputs {
            guard let inputNode = graph.nodes[inputId],
                  case .double(let value) = inputNode.attributes["value"] else {
                return nil
            }
            values.append(value)
        }

        // Compute folded value
        let foldedValue: Double
        switch node.operation {
        case .add:
            foldedValue = values[0] + values[1]
        case .subtract:
            foldedValue = values[0] - values[1]
        case .multiply:
            foldedValue = values[0] * values[1]
        case .divide:
            guard values[1] != 0 else { return nil }
            foldedValue = values[0] / values[1]
        case .neg:
            foldedValue = -values[0]
        case .abs:
            foldedValue = Swift.abs(values[0])
        // Skip exp/log/sqrt constant folding to avoid compiler issue
        // These operations remain unfoldable in constant folding pass
        default:
            return nil
        }

        return OptimizationNode(
            id: node.id,
            operation: .constant,
            inputs: [],
            shape: node.shape,
            dtype: node.dtype,
            attributes: ["value": .double(foldedValue)]
        )
    }
}

// MARK: - Dead Code Elimination Pass

/// Dead code elimination optimization pass
public struct DeadCodeEliminationPass: OptimizationPass {
    public var name: String { "DeadCodeElimination" }

    public init() {}

    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        let result = graph.copy()

        // Find all live nodes (reachable from outputs)
        var liveNodes = Set<UInt64>()
        var worklist = result.outputs

        while !worklist.isEmpty {
            let nodeId = worklist.removeFirst()
            guard !liveNodes.contains(nodeId) else { continue }

            liveNodes.insert(nodeId)

            if let node = result.nodes[nodeId] {
                worklist.append(contentsOf: node.inputs)
            }
        }

        // Remove dead nodes
        let deadNodes = result.nodes.keys.filter { !liveNodes.contains($0) }
        for nodeId in deadNodes {
            result.removeNode(nodeId)
        }

        return result
    }
}

// MARK: - Common Subexpression Elimination Pass

/// Common subexpression elimination optimization pass
public struct CommonSubexpressionEliminationPass: OptimizationPass {
    public var name: String { "CommonSubexpressionElimination" }

    public init() {}

    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        let result = graph.copy()

        // Build expression hash -> node id mapping
        var expressionMap: [ExpressionKey: UInt64] = [:]
        var nodesToRemove: [UInt64] = []
        var replacements: [(old: UInt64, new: UInt64)] = []

        // Sort nodes by ID to process in order
        let sortedNodes = result.nodes.values.sorted { $0.id < $1.id }

        for node in sortedNodes {
            let key = ExpressionKey(operation: node.operation, inputs: node.inputs, attributes: node.attributes)

            if let existingId = expressionMap[key] {
                // Found duplicate expression
                replacements.append((old: node.id, new: existingId))
                nodesToRemove.append(node.id)
            } else {
                expressionMap[key] = node.id
            }
        }

        // Apply replacements
        for (oldId, newId) in replacements {
            result.replaceAllUses(of: oldId, with: newId)
        }

        // Remove duplicate nodes
        for nodeId in nodesToRemove {
            result.removeNode(nodeId)
        }

        return result
    }
}

/// Key for identifying common subexpressions
private struct ExpressionKey: Hashable {
    let operation: NodeOperation
    let inputs: [UInt64]
    let attributes: [String: NodeAttribute]
}

// MARK: - Operation Fusion Pass

/// Operation fusion optimization pass
public struct OperationFusionPass: OptimizationPass {
    public var name: String { "OperationFusion" }

    public init() {}

    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        let result = graph.copy()
        var changed = true

        while changed {
            changed = false

            // Try to fuse matmul + add -> fusedMatmulAdd
            for (id, node) in result.nodes {
                if node.operation == .add {
                    if let fused = tryFuseMatmulAdd(node, in: result) {
                        result.updateNode(id, with: fused)
                        changed = true
                    }
                }
            }

            // Try to fuse mul + add -> fusedMulAdd (FMA)
            for (id, node) in result.nodes {
                if node.operation == .add {
                    if let fused = tryFuseMulAdd(node, in: result) {
                        result.updateNode(id, with: fused)
                        changed = true
                    }
                }
            }
        }

        return result
    }

    private func tryFuseMatmulAdd(_ addNode: OptimizationNode, in graph: OptimizationGraph) -> OptimizationNode? {
        guard addNode.inputs.count == 2 else { return nil }

        // Check if one input is matmul
        for (i, inputId) in addNode.inputs.enumerated() {
            guard let inputNode = graph.nodes[inputId],
                  inputNode.operation == .matmul else { continue }

            // Only fuse if matmul has single use and is not an output
            let users = graph.getUsers(of: inputId)
            guard users.count == 1 && !graph.outputs.contains(inputId) else { continue }

            let biasId = addNode.inputs[1 - i]

            return OptimizationNode(
                id: addNode.id,
                operation: .fusedMatmulAdd,
                inputs: inputNode.inputs + [biasId],
                shape: addNode.shape,
                dtype: addNode.dtype,
                attributes: addNode.attributes
            )
        }

        return nil
    }

    private func tryFuseMulAdd(_ addNode: OptimizationNode, in graph: OptimizationGraph) -> OptimizationNode? {
        guard addNode.inputs.count == 2 else { return nil }

        // Check if one input is multiply
        for (i, inputId) in addNode.inputs.enumerated() {
            guard let inputNode = graph.nodes[inputId],
                  inputNode.operation == .multiply else { continue }

            // Only fuse if multiply has single use and is not an output
            let users = graph.getUsers(of: inputId)
            guard users.count == 1 && !graph.outputs.contains(inputId) else { continue }

            let addendId = addNode.inputs[1 - i]

            return OptimizationNode(
                id: addNode.id,
                operation: .fusedMulAdd,
                inputs: inputNode.inputs + [addendId],
                shape: addNode.shape,
                dtype: addNode.dtype,
                attributes: addNode.attributes
            )
        }

        return nil
    }
}

// MARK: - Algebraic Simplification Pass

/// Algebraic simplification optimization pass
public struct AlgebraicSimplificationPass: OptimizationPass {
    public var name: String { "AlgebraicSimplification" }

    public init() {}

    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        let result = graph.copy()
        var changed = true

        while changed {
            changed = false

            for (id, node) in result.nodes {
                if let simplified = simplify(node, in: result) {
                    result.updateNode(id, with: simplified)
                    changed = true
                }
            }
        }

        return result
    }

    private func simplify(_ node: OptimizationNode, in graph: OptimizationGraph) -> OptimizationNode? {
        switch node.operation {
        case .add:
            // x + 0 = x
            if let zeroIndex = findZeroConstant(in: node.inputs, graph: graph) {
                let otherId = node.inputs[1 - zeroIndex]
                return graph.nodes[otherId]?.copy()
            }

        case .multiply:
            // x * 0 = 0
            if let zeroIndex = findZeroConstant(in: node.inputs, graph: graph) {
                return OptimizationNode(
                    id: node.id,
                    operation: .constant,
                    inputs: [],
                    shape: node.shape,
                    dtype: node.dtype,
                    attributes: ["value": .double(0.0)]
                )
            }
            // x * 1 = x
            if let oneIndex = findOneConstant(in: node.inputs, graph: graph) {
                let otherId = node.inputs[1 - oneIndex]
                return graph.nodes[otherId]?.copy()
            }

        case .subtract:
            // x - 0 = x
            if node.inputs.count == 2,
               let inputNode = graph.nodes[node.inputs[1]],
               inputNode.operation == .constant,
               case .double(let value) = inputNode.attributes["value"],
               value == 0.0 {
                return graph.nodes[node.inputs[0]]?.copy()
            }

        case .divide:
            // x / 1 = x
            if node.inputs.count == 2,
               let inputNode = graph.nodes[node.inputs[1]],
               inputNode.operation == .constant,
               case .double(let value) = inputNode.attributes["value"],
               value == 1.0 {
                return graph.nodes[node.inputs[0]]?.copy()
            }

        case .neg:
            // -(-x) = x
            if let inputNode = graph.nodes[node.inputs[0]],
               inputNode.operation == .neg {
                return graph.nodes[inputNode.inputs[0]]?.copy()
            }

        case .exp:
            // exp(log(x)) = x
            if let inputNode = graph.nodes[node.inputs[0]],
               inputNode.operation == .log {
                return graph.nodes[inputNode.inputs[0]]?.copy()
            }

        case .log:
            // log(exp(x)) = x
            if let inputNode = graph.nodes[node.inputs[0]],
               inputNode.operation == .exp {
                return graph.nodes[inputNode.inputs[0]]?.copy()
            }

        default:
            break
        }

        return nil
    }

    private func findZeroConstant(in inputs: [UInt64], graph: OptimizationGraph) -> Int? {
        for (i, inputId) in inputs.enumerated() {
            if let node = graph.nodes[inputId],
               node.operation == .constant,
               case .double(let value) = node.attributes["value"],
               value == 0.0 {
                return i
            }
        }
        return nil
    }

    private func findOneConstant(in inputs: [UInt64], graph: OptimizationGraph) -> Int? {
        for (i, inputId) in inputs.enumerated() {
            if let node = graph.nodes[inputId],
               node.operation == .constant,
               case .double(let value) = node.attributes["value"],
               value == 1.0 {
                return i
            }
        }
        return nil
    }
}

// MARK: - Optimization Pipeline

/// Pipeline that runs multiple optimization passes
public final class OptimizationPipeline: @unchecked Sendable {
    private var passes: [OptimizationPass] = []
    private let lock = NSLock()

    public init() {}

    /// Add a pass to the pipeline
    public func addPass(_ pass: OptimizationPass) {
        lock.lock()
        defer { lock.unlock() }
        passes.append(pass)
    }

    /// Run all passes on a graph
    public func run(on graph: OptimizationGraph) -> OptimizationGraph {
        lock.lock()
        let currentPasses = passes
        lock.unlock()

        var result = graph
        for pass in currentPasses {
            result = pass.run(on: result)
        }
        return result
    }

    /// Get pass count
    public var passCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return passes.count
    }

    /// Create a default optimization pipeline
    public static func standard() -> OptimizationPipeline {
        let pipeline = OptimizationPipeline()
        pipeline.addPass(ConstantFoldingPass())
        pipeline.addPass(AlgebraicSimplificationPass())
        pipeline.addPass(CommonSubexpressionEliminationPass())
        pipeline.addPass(OperationFusionPass())
        pipeline.addPass(DeadCodeEliminationPass())
        return pipeline
    }

    /// Create a minimal optimization pipeline
    public static func minimal() -> OptimizationPipeline {
        let pipeline = OptimizationPipeline()
        pipeline.addPass(ConstantFoldingPass())
        pipeline.addPass(DeadCodeEliminationPass())
        return pipeline
    }
}

// MARK: - Graph Statistics

/// Statistics about an optimization graph
public struct GraphStatistics: Sendable {
    public let nodeCount: Int
    public let operationCounts: [NodeOperation: Int]
    public let inputCount: Int
    public let outputCount: Int

    public init(graph: OptimizationGraph) {
        self.nodeCount = graph.nodeCount
        self.inputCount = graph.inputs.count
        self.outputCount = graph.outputs.count

        var counts: [NodeOperation: Int] = [:]
        for node in graph.nodes.values {
            counts[node.operation, default: 0] += 1
        }
        self.operationCounts = counts
    }
}
