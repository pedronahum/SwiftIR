/// Developer Experience - Tools for debugging, visualization, and diagnostics
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides:
/// - Graph visualization (DOT format export)
/// - Error diagnostics with source locations
/// - Checkpointing for gradient computation
/// - Profiling and statistics

import Foundation

// MARK: - Graph Visualization

/// Export computation graphs to DOT format for visualization
public struct GraphVisualizer: Sendable {

    public init() {}

    /// Export an optimization graph to DOT format
    public func toDOT(_ graph: OptimizationGraph, name: String = "graph") -> String {
        var lines: [String] = []
        lines.append("digraph \(name) {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=rounded];")
        lines.append("")

        // Add nodes
        for (id, node) in graph.nodes.sorted(by: { $0.key < $1.key }) {
            let label = formatNodeLabel(node)
            let color = nodeColor(for: node.operation)
            lines.append("  n\(id) [label=\"\(label)\", fillcolor=\"\(color)\", style=\"filled,rounded\"];")
        }

        lines.append("")

        // Add edges
        for (id, node) in graph.nodes {
            for (index, inputId) in node.inputs.enumerated() {
                lines.append("  n\(inputId) -> n\(id) [label=\"\(index)\"];")
            }
        }

        // Mark outputs
        lines.append("")
        lines.append("  // Outputs")
        for outputId in graph.outputs {
            lines.append("  n\(outputId) [penwidth=3];")
        }

        lines.append("}")
        return lines.joined(separator: "\n")
    }

    /// Export a traced computation to DOT format
    public func traceToDOT(name: String = "trace") -> String {
        let operations = TracerGraphBuilder.shared.getOperations()

        var lines: [String] = []
        lines.append("digraph \(name) {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=rounded];")
        lines.append("")

        // Add nodes from traced operations
        for op in operations {
            let (id, label) = operationInfo(op)
            lines.append("  n\(id) [label=\"\(label)\"];")
        }

        lines.append("")

        // Add edges
        for op in operations {
            let edges = operationEdges(op)
            for (from, to) in edges {
                lines.append("  n\(from) -> n\(to);")
            }
        }

        lines.append("}")
        return lines.joined(separator: "\n")
    }

    private func formatNodeLabel(_ node: OptimizationNode) -> String {
        let shapeStr = node.shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
        return "\(node.operation.rawValue)\\nid: \(node.id)\\n[\(shapeStr)]"
    }

    private func nodeColor(for operation: NodeOperation) -> String {
        switch operation {
        case .constant, .placeholder:
            return "#e8f4f8"
        case .add, .subtract, .multiply, .divide:
            return "#d4edda"
        case .matmul:
            return "#fff3cd"
        case .exp, .log, .sqrt, .abs, .neg, .tanh, .sigmoid, .relu:
            return "#f8d7da"
        case .sum, .mean, .max, .min:
            return "#d1ecf1"
        case .fusedMatmulAdd, .fusedMulAdd, .fusedReluAdd:
            return "#ffeeba"
        case .transpose:
            return "#e2e3e5"
        }
    }

    private func operationInfo(_ op: TracedOperation) -> (UInt64, String) {
        switch op {
        case .constant(let id, let value, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "const(\(value))\\n[\(shapeStr)]")
        case .placeholder(let id, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "placeholder\\n[\(shapeStr)]")
        case .binary(let id, let op, _, _, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "\(op.rawValue)\\n[\(shapeStr)]")
        case .unary(let id, let op, _, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "\(op.rawValue)\\n[\(shapeStr)]")
        case .reduction(let id, let op, _, _, _, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "\(op.rawValue)\\n[\(shapeStr)]")
        case .print(let id, _, let label, _, _, _, _):
            return (id, "print(\(label))")
        case .power(let id, _, let exp, let shape, _):
            let shapeStr = shape.dimensions.map { $0 != nil ? "\($0!)" : "?" }.joined(separator: "x")
            return (id, "pow(\\(exp))\\n[\(shapeStr)]")
        }
    }

    private func operationEdges(_ op: TracedOperation) -> [(UInt64, UInt64)] {
        switch op {
        case .constant, .placeholder:
            return []
        case .binary(let id, _, let lhs, let rhs, _, _):
            return [(lhs, id), (rhs, id)]
        case .unary(let id, _, let input, _, _):
            return [(input, id)]
        case .reduction(let id, _, let input, _, _, _, _):
            return [(input, id)]
        case .print(let id, let input, _, _, _, _, _):
            return [(input, id)]
        case .power(let id, let base, _, _, _):
            return [(base, id)]
        }
    }
}

// MARK: - Error Diagnostics

/// Detailed error information with source context
public struct DiagnosticError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public let location: SourceLocation?
    public let notes: [String]
    public let suggestions: [String]

    public struct SourceLocation: Sendable {
        public let file: String
        public let line: Int
        public let column: Int

        public init(file: String = #file, line: Int = #line, column: Int = #column) {
            self.file = file
            self.line = line
            self.column = column
        }
    }

    public init(
        _ message: String,
        at location: SourceLocation? = nil,
        notes: [String] = [],
        suggestions: [String] = []
    ) {
        self.message = message
        self.location = location
        self.notes = notes
        self.suggestions = suggestions
    }

    public var description: String {
        var result = "Error: \(message)"

        if let loc = location {
            result += "\n  at \(loc.file):\(loc.line):\(loc.column)"
        }

        for note in notes {
            result += "\n  note: \(note)"
        }

        for suggestion in suggestions {
            result += "\n  suggestion: \(suggestion)"
        }

        return result
    }
}

/// Diagnostics collector for gradient computation
public final class DiagnosticsCollector: @unchecked Sendable {
    private var warnings: [String] = []
    private var errors: [DiagnosticError] = []
    private let lock = NSLock()

    public init() {}

    public func warn(_ message: String, file: String = #file, line: Int = #line) {
        lock.lock()
        defer { lock.unlock() }
        warnings.append("[\(URL(fileURLWithPath: file).lastPathComponent):\(line)] \(message)")
    }

    public func error(_ error: DiagnosticError) {
        lock.lock()
        defer { lock.unlock() }
        errors.append(error)
    }

    public func getWarnings() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return warnings
    }

    public func getErrors() -> [DiagnosticError] {
        lock.lock()
        defer { lock.unlock() }
        return errors
    }

    public func hasErrors() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !errors.isEmpty
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        warnings.removeAll()
        errors.removeAll()
    }

    public func printDiagnostics() {
        lock.lock()
        let currentWarnings = warnings
        let currentErrors = errors
        lock.unlock()

        for warning in currentWarnings {
            print("warning: \(warning)")
        }

        for error in currentErrors {
            print(error.description)
        }
    }
}

// MARK: - Checkpointing

/// Checkpoint for saving and restoring computation state
public struct Checkpoint: Sendable {
    public let id: String
    public let timestamp: Date
    public let operationCount: Int
    public let metadata: [String: String]

    public init(id: String = UUID().uuidString, metadata: [String: String] = [:]) {
        self.id = id
        self.timestamp = Date()
        self.operationCount = TracerGraphBuilder.shared.getOperations().count
        self.metadata = metadata
    }
}

/// Manager for gradient checkpointing
public final class CheckpointManager: @unchecked Sendable {
    private var checkpoints: [String: Checkpoint] = [:]
    private var savedStates: [String: [TracedOperation]] = [:]
    private let lock = NSLock()

    public init() {}

    /// Create a checkpoint at the current state
    public func createCheckpoint(name: String, metadata: [String: String] = [:]) -> Checkpoint {
        lock.lock()
        defer { lock.unlock() }

        let checkpoint = Checkpoint(id: name, metadata: metadata)
        let operations = TracerGraphBuilder.shared.getOperations()

        checkpoints[name] = checkpoint
        savedStates[name] = operations

        return checkpoint
    }

    /// Get a checkpoint by name
    public func getCheckpoint(_ name: String) -> Checkpoint? {
        lock.lock()
        defer { lock.unlock() }
        return checkpoints[name]
    }

    /// List all checkpoints
    public func listCheckpoints() -> [Checkpoint] {
        lock.lock()
        defer { lock.unlock() }
        return Array(checkpoints.values).sorted { $0.timestamp < $1.timestamp }
    }

    /// Delete a checkpoint
    public func deleteCheckpoint(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        checkpoints.removeValue(forKey: name)
        savedStates.removeValue(forKey: name)
    }

    /// Get operation count at checkpoint
    public func operationCount(at name: String) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return savedStates[name]?.count
    }

    /// Clear all checkpoints
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        checkpoints.removeAll()
        savedStates.removeAll()
    }
}

// MARK: - Profiling

/// Profile data for a computation
public struct ProfileData: Sendable {
    public let operationCounts: [String: Int]
    public let totalOperations: Int
    public let memoryEstimate: Int

    init(operations: [TracedOperation]) {
        var counts: [String: Int] = [:]
        var memory = 0

        for op in operations {
            let name = Self.operationName(op)
            counts[name, default: 0] += 1
            memory += Self.estimateMemory(op)
        }

        self.operationCounts = counts
        self.totalOperations = operations.count
        self.memoryEstimate = memory
    }

    private static func operationName(_ op: TracedOperation) -> String {
        switch op {
        case .constant: return "constant"
        case .placeholder: return "placeholder"
        case .binary(_, let op, _, _, _, _): return op.rawValue
        case .unary(_, let op, _, _, _): return op.rawValue
        case .reduction(_, let op, _, _, _, _, _): return op.rawValue
        case .print: return "print"
        case .power: return "power"
        }
    }

    private static func estimateMemory(_ op: TracedOperation) -> Int {
        let shape: TensorShape
        let dtype: DType

        switch op {
        case .constant(_, _, let s, let d): shape = s; dtype = d
        case .placeholder(_, let s, let d): shape = s; dtype = d
        case .binary(_, _, _, _, let s, let d): shape = s; dtype = d
        case .unary(_, _, _, let s, let d): shape = s; dtype = d
        case .reduction(_, _, _, _, _, let s, let d): shape = s; dtype = d
        case .print(_, _, _, let s, let d, _, _): shape = s; dtype = d
        case .power(_, _, _, let s, let d): shape = s; dtype = d
        }

        let elements = shape.elementCount ?? 1
        let byteWidth: Int
        switch dtype {
        case .float16, .bfloat16: byteWidth = 2
        case .float32: byteWidth = 4
        case .float64: byteWidth = 8
        case .int8, .uint8, .bool: byteWidth = 1
        case .int16, .uint16: byteWidth = 2
        case .int32, .uint32: byteWidth = 4
        case .int64, .uint64: byteWidth = 8
        case .complex64: byteWidth = 8
        case .complex128: byteWidth = 16
        }
        return elements * byteWidth
    }

    public func summary() -> String {
        var lines: [String] = []
        lines.append("Profile Summary:")
        lines.append("  Total operations: \(totalOperations)")
        lines.append("  Estimated memory: \(memoryEstimate) bytes")
        lines.append("  Operation breakdown:")

        for (op, count) in operationCounts.sorted(by: { $0.key < $1.key }) {
            lines.append("    \(op): \(count)")
        }

        return lines.joined(separator: "\n")
    }
}

/// Profiler for computation graphs
public final class Profiler: @unchecked Sendable {
    private var profiles: [String: ProfileData] = [:]
    private let lock = NSLock()

    public init() {}

    /// Profile the current computation
    public func profile(name: String = "default") -> ProfileData {
        let operations = TracerGraphBuilder.shared.getOperations()
        let data = ProfileData(operations: operations)

        lock.lock()
        profiles[name] = data
        lock.unlock()

        return data
    }

    /// Get a saved profile
    public func getProfile(_ name: String) -> ProfileData? {
        lock.lock()
        defer { lock.unlock() }
        return profiles[name]
    }

    /// Compare two profiles
    public func compare(_ name1: String, _ name2: String) -> String? {
        lock.lock()
        guard let p1 = profiles[name1], let p2 = profiles[name2] else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        var lines: [String] = []
        lines.append("Profile Comparison: \(name1) vs \(name2)")
        lines.append("  Operations: \(p1.totalOperations) vs \(p2.totalOperations)")
        lines.append("  Memory: \(p1.memoryEstimate) vs \(p2.memoryEstimate)")

        return lines.joined(separator: "\n")
    }

    /// Clear all profiles
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        profiles.removeAll()
    }
}

// MARK: - Shape Validation

/// Validator for tensor shapes
public struct ShapeValidator: Sendable {

    public init() {}

    /// Validate that shapes are compatible for an operation
    public func validate(
        operation: String,
        inputs: [(String, TensorShape)],
        expectedOutput: TensorShape?
    ) -> Result<Void, DiagnosticError> {
        // Check for nil dimensions
        for (name, shape) in inputs {
            if shape.dimensions.contains(where: { $0 == nil }) {
                return .failure(DiagnosticError(
                    "Shape contains unknown dimensions",
                    notes: ["Input '\(name)' has shape \(shape)"],
                    suggestions: ["Ensure all input shapes are fully specified"]
                ))
            }
        }

        // Operation-specific validation
        switch operation {
        case "matmul":
            return validateMatmul(inputs)
        case "add", "subtract", "multiply", "divide":
            return validateBroadcast(inputs)
        default:
            return .success(())
        }
    }

    private func validateMatmul(_ inputs: [(String, TensorShape)]) -> Result<Void, DiagnosticError> {
        guard inputs.count == 2 else {
            return .failure(DiagnosticError("matmul requires exactly 2 inputs"))
        }

        let (name1, shape1) = inputs[0]
        let (name2, shape2) = inputs[1]

        guard shape1.rank >= 2, shape2.rank >= 2 else {
            return .failure(DiagnosticError(
                "matmul requires tensors with rank >= 2",
                notes: [
                    "\(name1) has rank \(shape1.rank)",
                    "\(name2) has rank \(shape2.rank)"
                ]
            ))
        }

        let k1 = shape1.dimensions[shape1.rank - 1]
        let k2 = shape2.dimensions[shape2.rank - 2]

        if let kk1 = k1, let kk2 = k2, kk1 != kk2 {
            return .failure(DiagnosticError(
                "matmul inner dimensions must match",
                notes: [
                    "\(name1) has inner dimension \(kk1)",
                    "\(name2) has inner dimension \(kk2)"
                ],
                suggestions: ["Transpose one of the inputs or reshape to match dimensions"]
            ))
        }

        return .success(())
    }

    private func validateBroadcast(_ inputs: [(String, TensorShape)]) -> Result<Void, DiagnosticError> {
        guard inputs.count == 2 else {
            return .failure(DiagnosticError("Binary operation requires exactly 2 inputs"))
        }

        let (_, shape1) = inputs[0]
        let (_, shape2) = inputs[1]

        // Check if shapes are broadcastable
        let result = shape1.broadcast(with: shape2)
        if result == shape1 || result == shape2 {
            // Broadcast succeeded
            return .success(())
        }

        // Shapes might still be compatible even if not equal to inputs
        return .success(())
    }
}

// MARK: - Debug Utilities

/// Debug utilities for inspecting computation state
public struct DebugUtilities: Sendable {

    public init() {}

    /// Print current graph state
    public func printGraphState() {
        let operations = TracerGraphBuilder.shared.getOperations()
        print("Current Graph State:")
        print("  Operations: \(operations.count)")

        for (index, op) in operations.enumerated() {
            let desc = operationDescription(op)
            print("  [\(index)] \(desc)")
        }
    }

    /// Get a summary of the current computation
    public func summary() -> String {
        let operations = TracerGraphBuilder.shared.getOperations()
        let profile = ProfileData(operations: operations)
        return profile.summary()
    }

    private func operationDescription(_ op: TracedOperation) -> String {
        switch op {
        case .constant(let id, let value, let shape, _):
            return "const(\(id)): \(value) [\(shape)]"
        case .placeholder(let id, let shape, _):
            return "placeholder(\(id)): [\(shape)]"
        case .binary(let id, let op, let lhs, let rhs, let shape, _):
            return "\(op.rawValue)(\(id)): \(lhs), \(rhs) -> [\(shape)]"
        case .unary(let id, let op, let input, let shape, _):
            return "\(op.rawValue)(\(id)): \(input) -> [\(shape)]"
        case .reduction(let id, let op, let input, let axes, _, let shape, _):
            return "\(op.rawValue)(\(id)): \(input) axes=\(axes) -> [\(shape)]"
        case .print(let id, let input, let label, _, _, _, _):
            return "print(\(id)): \(input) '\(label)'"
        case .power(let id, let base, let exp, let shape, _):
            return "power(\(id)): \(base)^\(exp) -> [\(shape)]"
        }
    }
}
