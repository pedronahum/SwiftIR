/// Phase 8: Production Readiness
/// Serialization, versioning, and graph validation for production deployments

import Foundation

// MARK: - Version Information

/// Version information for serialized graphs
public struct GraphVersion: Codable, Sendable, Equatable {
    public let major: Int
    public let minor: Int
    public let patch: Int

    public static let current = GraphVersion(major: 1, minor: 0, patch: 0)

    public init(major: Int, minor: Int, patch: Int) {
        self.major = major
        self.minor = minor
        self.patch = patch
    }

    public var string: String {
        "\(major).\(minor).\(patch)"
    }

    /// Check if this version is compatible with another version
    public func isCompatible(with other: GraphVersion) -> Bool {
        // Major version must match, minor can be higher
        return self.major == other.major && self.minor >= other.minor
    }
}

// MARK: - Serializable Graph

/// A serializable representation of an optimization graph
public struct SerializableGraph: Codable, Sendable {
    public let version: GraphVersion
    public let nodes: [SerializableNode]
    public let inputs: [UInt64]
    public let outputs: [UInt64]
    public let metadata: [String: String]

    public init(
        version: GraphVersion = .current,
        nodes: [SerializableNode],
        inputs: [UInt64],
        outputs: [UInt64],
        metadata: [String: String] = [:]
    ) {
        self.version = version
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.metadata = metadata
    }
}

/// A serializable representation of a graph node
public struct SerializableNode: Codable, Sendable {
    public let id: UInt64
    public let operation: String
    public let inputs: [UInt64]
    public let shape: [Int?]
    public let dtype: String
    public let attributes: [String: SerializableAttribute]

    public init(
        id: UInt64,
        operation: String,
        inputs: [UInt64],
        shape: [Int?],
        dtype: String,
        attributes: [String: SerializableAttribute] = [:]
    ) {
        self.id = id
        self.operation = operation
        self.inputs = inputs
        self.shape = shape
        self.dtype = dtype
        self.attributes = attributes
    }
}

/// Serializable attribute value
public enum SerializableAttribute: Codable, Sendable, Equatable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)
    case intArray([Int])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([Int].self) {
            self = .intArray(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode attribute value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .intArray(let value):
            try container.encode(value)
        }
    }
}

// MARK: - Graph Serializer

/// Serializes and deserializes optimization graphs
public final class GraphSerializer: @unchecked Sendable {
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init() {
        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        decoder = JSONDecoder()
    }

    /// Serialize an optimization graph to JSON data
    public func serialize(_ graph: OptimizationGraph) throws -> Data {
        let serializable = toSerializable(graph)
        return try encoder.encode(serializable)
    }

    /// Serialize an optimization graph to a JSON string
    public func serializeToString(_ graph: OptimizationGraph) throws -> String {
        let data = try serialize(graph)
        guard let string = String(data: data, encoding: .utf8) else {
            throw SerializationError.encodingFailed
        }
        return string
    }

    /// Deserialize JSON data to an optimization graph
    public func deserialize(_ data: Data) throws -> OptimizationGraph {
        let serializable = try decoder.decode(SerializableGraph.self, from: data)

        // Version compatibility check
        guard serializable.version.isCompatible(with: .current) else {
            throw SerializationError.incompatibleVersion(
                found: serializable.version,
                required: .current
            )
        }

        return try fromSerializable(serializable)
    }

    /// Deserialize a JSON string to an optimization graph
    public func deserializeFromString(_ string: String) throws -> OptimizationGraph {
        guard let data = string.data(using: .utf8) else {
            throw SerializationError.decodingFailed
        }
        return try deserialize(data)
    }

    /// Save graph to file
    public func save(_ graph: OptimizationGraph, to path: String) throws {
        let data = try serialize(graph)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
    }

    /// Load graph from file
    public func load(from path: String) throws -> OptimizationGraph {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try deserialize(data)
    }

    // MARK: - Conversion Helpers

    private func toSerializable(_ graph: OptimizationGraph) -> SerializableGraph {
        let nodes = graph.nodes.values.map { node -> SerializableNode in
            let attrs = node.attributes.mapValues { attr -> SerializableAttribute in
                switch attr {
                case .int(let v): return .int(v)
                case .double(let v): return .double(v)
                case .string(let v): return .string(v)
                case .bool(let v): return .bool(v)
                case .intArray(let v): return .intArray(v)
                case .axes(let v): return .intArray(Array(v).sorted())
                }
            }

            return SerializableNode(
                id: node.id,
                operation: node.operation.rawValue,
                inputs: node.inputs,
                shape: node.shape.dimensions,
                dtype: node.dtype.rawValue,
                attributes: attrs
            )
        }

        return SerializableGraph(
            version: .current,
            nodes: nodes,
            inputs: graph.inputs,
            outputs: graph.outputs,
            metadata: [:]
        )
    }

    private func fromSerializable(_ serializable: SerializableGraph) throws -> OptimizationGraph {
        let graph = OptimizationGraph()

        for node in serializable.nodes {
            guard let operation = NodeOperation(rawValue: node.operation) else {
                throw SerializationError.unknownOperation(node.operation)
            }

            guard let dtype = DType(rawValue: node.dtype) else {
                throw SerializationError.unknownDType(node.dtype)
            }

            let attrs = node.attributes.mapValues { attr -> NodeAttribute in
                switch attr {
                case .int(let v): return .int(v)
                case .double(let v): return .double(v)
                case .string(let v): return .string(v)
                case .bool(let v): return .bool(v)
                case .intArray(let v): return .intArray(v)
                }
            }

            let optNode = OptimizationNode(
                id: node.id,
                operation: operation,
                inputs: node.inputs,
                shape: TensorShape(dimensions: node.shape),
                dtype: dtype,
                attributes: attrs
            )

            graph.addNode(optNode)
        }

        graph.inputs = serializable.inputs
        graph.outputs = serializable.outputs

        return graph
    }
}

/// Serialization errors
public enum SerializationError: Error, CustomStringConvertible {
    case encodingFailed
    case decodingFailed
    case incompatibleVersion(found: GraphVersion, required: GraphVersion)
    case unknownOperation(String)
    case unknownDType(String)
    case invalidGraph(String)

    public var description: String {
        switch self {
        case .encodingFailed:
            return "Failed to encode graph to JSON"
        case .decodingFailed:
            return "Failed to decode JSON to graph"
        case .incompatibleVersion(let found, let required):
            return "Incompatible version: found \(found.string), required \(required.string)"
        case .unknownOperation(let op):
            return "Unknown operation: \(op)"
        case .unknownDType(let dtype):
            return "Unknown dtype: \(dtype)"
        case .invalidGraph(let reason):
            return "Invalid graph: \(reason)"
        }
    }
}

// MARK: - Graph Validator

/// Validates optimization graphs for correctness
public final class GraphValidator: @unchecked Sendable {

    public init() {}

    /// Validate a graph and return any errors found
    public func validate(_ graph: OptimizationGraph) -> [ValidationError] {
        var errors: [ValidationError] = []

        // Check for dangling inputs
        errors.append(contentsOf: checkDanglingInputs(graph))

        // Check for cycles
        if let cycleError = checkCycles(graph) {
            errors.append(cycleError)
        }

        // Check shape consistency
        errors.append(contentsOf: checkShapeConsistency(graph))

        // Check dtype consistency
        errors.append(contentsOf: checkDTypeConsistency(graph))

        // Check output validity
        errors.append(contentsOf: checkOutputValidity(graph))

        return errors
    }

    /// Check if graph is valid
    public func isValid(_ graph: OptimizationGraph) -> Bool {
        return validate(graph).isEmpty
    }

    // MARK: - Validation Checks

    private func checkDanglingInputs(_ graph: OptimizationGraph) -> [ValidationError] {
        var errors: [ValidationError] = []

        for node in graph.nodes.values {
            for inputId in node.inputs {
                if graph.nodes[inputId] == nil {
                    errors.append(.danglingInput(nodeId: node.id, inputId: inputId))
                }
            }
        }

        return errors
    }

    private func checkCycles(_ graph: OptimizationGraph) -> ValidationError? {
        var visited = Set<UInt64>()
        var recursionStack = Set<UInt64>()

        func hasCycle(nodeId: UInt64) -> Bool {
            visited.insert(nodeId)
            recursionStack.insert(nodeId)

            if let node = graph.nodes[nodeId] {
                for inputId in node.inputs {
                    if !visited.contains(inputId) {
                        if hasCycle(nodeId: inputId) {
                            return true
                        }
                    } else if recursionStack.contains(inputId) {
                        return true
                    }
                }
            }

            recursionStack.remove(nodeId)
            return false
        }

        for node in graph.nodes.values {
            if !visited.contains(node.id) {
                if hasCycle(nodeId: node.id) {
                    return .cycleDetected
                }
            }
        }

        return nil
    }

    private func checkShapeConsistency(_ graph: OptimizationGraph) -> [ValidationError] {
        var errors: [ValidationError] = []

        for node in graph.nodes.values {
            switch node.operation {
            case .add, .subtract, .multiply, .divide:
                if node.inputs.count == 2 {
                    if let input1 = graph.nodes[node.inputs[0]],
                       let input2 = graph.nodes[node.inputs[1]] {
                        if !areShapesBroadcastable(input1.shape, input2.shape) {
                            errors.append(.shapeMismatch(
                                nodeId: node.id,
                                expected: input1.shape.description,
                                found: input2.shape.description
                            ))
                        }
                    }
                }
            case .matmul:
                if node.inputs.count == 2 {
                    if let input1 = graph.nodes[node.inputs[0]],
                       let input2 = graph.nodes[node.inputs[1]] {
                        let shape1 = input1.shape.dimensions
                        let shape2 = input2.shape.dimensions
                        if shape1.count >= 2 && shape2.count >= 2 {
                            let k1 = shape1[shape1.count - 1]
                            let k2 = shape2[shape2.count - 2]
                            if let k1 = k1, let k2 = k2, k1 != k2 {
                                errors.append(.shapeMismatch(
                                    nodeId: node.id,
                                    expected: "inner dimension \(k1)",
                                    found: "inner dimension \(k2)"
                                ))
                            }
                        }
                    }
                }
            default:
                break
            }
        }

        return errors
    }

    private func checkDTypeConsistency(_ graph: OptimizationGraph) -> [ValidationError] {
        var errors: [ValidationError] = []

        for node in graph.nodes.values {
            for inputId in node.inputs {
                if let input = graph.nodes[inputId] {
                    // Most operations require matching dtypes
                    if input.dtype != node.dtype && !isDTypeMixingAllowed(node.operation) {
                        errors.append(.dtypeMismatch(
                            nodeId: node.id,
                            expected: node.dtype.rawValue,
                            found: input.dtype.rawValue
                        ))
                    }
                }
            }
        }

        return errors
    }

    private func checkOutputValidity(_ graph: OptimizationGraph) -> [ValidationError] {
        var errors: [ValidationError] = []

        for outputId in graph.outputs {
            if graph.nodes[outputId] == nil {
                errors.append(.invalidOutput(outputId: outputId))
            }
        }

        return errors
    }

    // MARK: - Helpers

    private func areShapesBroadcastable(_ shape1: TensorShape, _ shape2: TensorShape) -> Bool {
        let dims1 = shape1.dimensions.reversed()
        let dims2 = shape2.dimensions.reversed()

        for (d1, d2) in zip(dims1, dims2) {
            guard let d1 = d1, let d2 = d2 else {
                continue  // Unknown dimensions are compatible
            }
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return false
            }
        }

        return true
    }

    private func isDTypeMixingAllowed(_ operation: NodeOperation) -> Bool {
        // Most operations require matching dtypes
        // In the future, cast operations would allow mixed dtypes
        return false
    }
}

/// Validation errors
public enum ValidationError: Error, CustomStringConvertible, Equatable {
    case danglingInput(nodeId: UInt64, inputId: UInt64)
    case cycleDetected
    case shapeMismatch(nodeId: UInt64, expected: String, found: String)
    case dtypeMismatch(nodeId: UInt64, expected: String, found: String)
    case invalidOutput(outputId: UInt64)

    public var description: String {
        switch self {
        case .danglingInput(let nodeId, let inputId):
            return "Node \(nodeId) references non-existent input \(inputId)"
        case .cycleDetected:
            return "Cycle detected in graph"
        case .shapeMismatch(let nodeId, let expected, let found):
            return "Shape mismatch at node \(nodeId): expected \(expected), found \(found)"
        case .dtypeMismatch(let nodeId, let expected, let found):
            return "DType mismatch at node \(nodeId): expected \(expected), found \(found)"
        case .invalidOutput(let outputId):
            return "Output references non-existent node \(outputId)"
        }
    }
}

// MARK: - Detailed Graph Statistics

/// Detailed statistics about an optimization graph
public struct DetailedGraphStatistics: Sendable {
    public let nodeCount: Int
    public let edgeCount: Int
    public let inputCount: Int
    public let outputCount: Int
    public let operationCounts: [String: Int]
    public let maxDepth: Int
    public let memoryEstimate: Int

    public init(
        nodeCount: Int,
        edgeCount: Int,
        inputCount: Int,
        outputCount: Int,
        operationCounts: [String: Int],
        maxDepth: Int,
        memoryEstimate: Int
    ) {
        self.nodeCount = nodeCount
        self.edgeCount = edgeCount
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.operationCounts = operationCounts
        self.maxDepth = maxDepth
        self.memoryEstimate = memoryEstimate
    }

    public func summary() -> String {
        var result = "Graph Statistics\n"
        result += "================\n"
        result += "Nodes: \(nodeCount)\n"
        result += "Edges: \(edgeCount)\n"
        result += "Inputs: \(inputCount)\n"
        result += "Outputs: \(outputCount)\n"
        result += "Max Depth: \(maxDepth)\n"
        result += "Memory Estimate: \(memoryEstimate) bytes\n"
        result += "\nOperations:\n"

        for (op, count) in operationCounts.sorted(by: { $0.key < $1.key }) {
            result += "  \(op): \(count)\n"
        }

        return result
    }
}

/// Computes statistics for an optimization graph
public final class GraphAnalyzer: @unchecked Sendable {

    public init() {}

    public func analyze(_ graph: OptimizationGraph) -> DetailedGraphStatistics {
        let nodes = graph.nodes.values

        // Count edges
        var edgeCount = 0
        var operationCounts: [String: Int] = [:]

        for node in nodes {
            edgeCount += node.inputs.count
            let opName = node.operation.rawValue
            operationCounts[opName, default: 0] += 1
        }

        // Compute max depth
        let maxDepth = computeMaxDepth(graph)

        // Estimate memory
        let memoryEstimate = estimateMemory(graph)

        return DetailedGraphStatistics(
            nodeCount: nodes.count,
            edgeCount: edgeCount,
            inputCount: graph.inputs.count,
            outputCount: graph.outputs.count,
            operationCounts: operationCounts,
            maxDepth: maxDepth,
            memoryEstimate: memoryEstimate
        )
    }

    private func computeMaxDepth(_ graph: OptimizationGraph) -> Int {
        var depths: [UInt64: Int] = [:]

        func depth(of nodeId: UInt64) -> Int {
            if let cached = depths[nodeId] {
                return cached
            }

            guard let node = graph.nodes[nodeId] else {
                return 0
            }

            if node.inputs.isEmpty {
                depths[nodeId] = 0
                return 0
            }

            let maxInputDepth = node.inputs.map { depth(of: $0) }.max() ?? 0
            let result = maxInputDepth + 1
            depths[nodeId] = result
            return result
        }

        var maxDepth = 0
        for outputId in graph.outputs {
            maxDepth = max(maxDepth, depth(of: outputId))
        }

        return maxDepth
    }

    private func estimateMemory(_ graph: OptimizationGraph) -> Int {
        var total = 0

        for node in graph.nodes.values {
            let elements = node.shape.dimensions.compactMap { $0 }.reduce(1, *)
            let byteWidth: Int
            switch node.dtype {
            case .float16, .bfloat16:
                byteWidth = 2
            case .float32:
                byteWidth = 4
            case .float64:
                byteWidth = 8
            case .int8, .uint8, .bool:
                byteWidth = 1
            case .int16, .uint16:
                byteWidth = 2
            case .int32, .uint32:
                byteWidth = 4
            case .int64, .uint64:
                byteWidth = 8
            case .complex64:
                byteWidth = 8
            case .complex128:
                byteWidth = 16
            }
            total += elements * byteWidth
        }

        return total
    }
}

// MARK: - Graph Comparison

/// Result of comparing two graphs
public struct GraphComparison: Sendable {
    public let areEqual: Bool
    public let differences: [String]

    public init(areEqual: Bool, differences: [String]) {
        self.areEqual = areEqual
        self.differences = differences
    }

    public func summary() -> String {
        if areEqual {
            return "Graphs are equal"
        }

        var result = "Graph Differences:\n"
        for diff in differences {
            result += "  - \(diff)\n"
        }
        return result
    }
}

/// Compares two optimization graphs
public final class GraphComparator: @unchecked Sendable {

    public init() {}

    public func compare(_ graph1: OptimizationGraph, _ graph2: OptimizationGraph) -> GraphComparison {
        var differences: [String] = []

        let nodes1 = Array(graph1.nodes.values)
        let nodes2 = Array(graph2.nodes.values)

        // Compare node counts
        if nodes1.count != nodes2.count {
            differences.append("Node count: \(nodes1.count) vs \(nodes2.count)")
        }

        // Compare outputs
        if graph1.outputs != graph2.outputs {
            differences.append("Output nodes differ")
        }

        // Compare inputs
        if graph1.inputs != graph2.inputs {
            differences.append("Input nodes differ")
        }

        // Compare individual nodes
        let ids1 = Set(nodes1.map { $0.id })
        let ids2 = Set(nodes2.map { $0.id })

        let missingIn2 = ids1.subtracting(ids2)
        let missingIn1 = ids2.subtracting(ids1)

        if !missingIn2.isEmpty {
            differences.append("Nodes missing in second graph: \(missingIn2)")
        }

        if !missingIn1.isEmpty {
            differences.append("Nodes missing in first graph: \(missingIn1)")
        }

        // Compare common nodes
        for id in ids1.intersection(ids2) {
            if let node1 = graph1.nodes[id], let node2 = graph2.nodes[id] {
                if node1.operation != node2.operation {
                    differences.append("Node \(id) operation: \(node1.operation) vs \(node2.operation)")
                }
                if node1.shape != node2.shape {
                    differences.append("Node \(id) shape: \(node1.shape) vs \(node2.shape)")
                }
                if node1.inputs != node2.inputs {
                    differences.append("Node \(id) inputs differ")
                }
            }
        }

        return GraphComparison(areEqual: differences.isEmpty, differences: differences)
    }
}
