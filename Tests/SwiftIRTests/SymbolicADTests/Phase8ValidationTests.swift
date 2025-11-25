/// Phase 8 Validation Tests - Production Readiness
/// Tests for serialization, versioning, validation, and analysis

import Testing
import Foundation

@testable import SwiftIR

@Suite("Phase 8: Production Readiness", .serialized)
struct Phase8ValidationTests {

    init() {
        TracerGraphBuilder.shared.reset()
    }

    // MARK: - Serialization Tests

    @Suite("Serialization")
    struct SerializationTests {

        @Test("Serialize simple graph to JSON")
        func serializeSimpleGraph() throws {
            let graph = OptimizationGraph()

            let const = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(42.0)]
            )
            graph.addNode(const)
            graph.outputs = [1]

            let serializer = GraphSerializer()
            let json = try serializer.serializeToString(graph)

            #expect(json.contains("\"version\""))
            #expect(json.contains("\"nodes\""))
            #expect(json.contains("\"outputs\""))
        }

        @Test("Deserialize graph from JSON")
        func deserializeGraph() throws {
            let graph = OptimizationGraph()

            let node1 = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let node2 = OptimizationNode(
                id: 2, operation: .relu, inputs: [1],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(node1)
            graph.addNode(node2)
            graph.inputs = [1]
            graph.outputs = [2]

            let serializer = GraphSerializer()
            let json = try serializer.serializeToString(graph)
            let restored = try serializer.deserializeFromString(json)

            #expect(restored.nodes.count == 2)
            #expect(restored.inputs == [1])
            #expect(restored.outputs == [2])
        }

        @Test("Round-trip serialization preserves structure")
        func roundTripSerialization() throws {
            let graph = OptimizationGraph()

            let input1 = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([2, 3]), dtype: .float32
            )
            let input2 = OptimizationNode(
                id: 2, operation: .placeholder, inputs: [],
                shape: TensorShape([2, 3]), dtype: .float32
            )
            let add = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 2],
                shape: TensorShape([2, 3]), dtype: .float32
            )

            graph.addNode(input1)
            graph.addNode(input2)
            graph.addNode(add)
            graph.inputs = [1, 2]
            graph.outputs = [3]

            let serializer = GraphSerializer()
            let data = try serializer.serialize(graph)
            let restored = try serializer.deserialize(data)

            #expect(restored.nodes[1]?.operation == .placeholder)
            #expect(restored.nodes[2]?.operation == .placeholder)
            #expect(restored.nodes[3]?.operation == .add)
            #expect(restored.nodes[3]?.inputs == [1, 2])
        }

        @Test("Serialize graph with attributes")
        func serializeWithAttributes() throws {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: [
                    "value": .double(3.14),
                    "name": .string("pi"),
                    "trainable": .bool(false)
                ]
            )

            graph.addNode(node)
            graph.outputs = [1]

            let serializer = GraphSerializer()
            let json = try serializer.serializeToString(graph)
            let restored = try serializer.deserializeFromString(json)

            let restoredNode = restored.nodes[1]
            #expect(restoredNode?.attributes["value"] == .double(3.14))
            #expect(restoredNode?.attributes["name"] == .string("pi"))
            #expect(restoredNode?.attributes["trainable"] == .bool(false))
        }

        @Test("Serialize graph with array attributes")
        func serializeArrayAttributes() throws {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .sum, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32,
                attributes: [
                    "axes": .axes(Set([0, 1])),
                    "keepdims": .bool(true)
                ]
            )

            graph.addNode(node)
            graph.outputs = [1]

            let serializer = GraphSerializer()
            let data = try serializer.serialize(graph)
            let restored = try serializer.deserialize(data)

            let restoredNode = restored.nodes[1]
            // axes gets serialized as intArray
            #expect(restoredNode?.attributes["axes"] == .intArray([0, 1]))
            #expect(restoredNode?.attributes["keepdims"] == .bool(true))
        }
    }

    // MARK: - Version Compatibility Tests

    @Suite("Version Compatibility")
    struct VersionCompatibilityTests {

        @Test("Version string formatting")
        func versionString() {
            let version = GraphVersion(major: 1, minor: 2, patch: 3)
            #expect(version.string == "1.2.3")
        }

        @Test("Version compatibility - same major")
        func versionCompatibilitySameMajor() {
            let v1 = GraphVersion(major: 1, minor: 0, patch: 0)
            let v2 = GraphVersion(major: 1, minor: 2, patch: 0)

            #expect(v2.isCompatible(with: v1))
        }

        @Test("Version compatibility - different major")
        func versionCompatibilityDifferentMajor() {
            let v1 = GraphVersion(major: 1, minor: 0, patch: 0)
            let v2 = GraphVersion(major: 2, minor: 0, patch: 0)

            #expect(!v2.isCompatible(with: v1))
        }

        @Test("Version compatibility - older minor")
        func versionCompatibilityOlderMinor() {
            let v1 = GraphVersion(major: 1, minor: 3, patch: 0)
            let v2 = GraphVersion(major: 1, minor: 2, patch: 0)

            #expect(!v2.isCompatible(with: v1))
        }

        @Test("Deserialize incompatible version throws error")
        func incompatibleVersionError() throws {
            // Create JSON with incompatible version
            let json = """
            {
                "version": {"major": 99, "minor": 0, "patch": 0},
                "nodes": [],
                "inputs": [],
                "outputs": [],
                "metadata": {}
            }
            """

            let serializer = GraphSerializer()

            do {
                _ = try serializer.deserializeFromString(json)
                Issue.record("Expected incompatible version error")
            } catch let error as SerializationError {
                if case .incompatibleVersion = error {
                    // Expected
                } else {
                    Issue.record("Wrong error type: \(error)")
                }
            }
        }
    }

    // MARK: - Graph Validation Tests

    @Suite("Graph Validation")
    struct GraphValidationTests {

        @Test("Valid graph passes validation")
        func validGraphPassesValidation() {
            let graph = OptimizationGraph()

            let input = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let output = OptimizationNode(
                id: 2, operation: .relu, inputs: [1],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(input)
            graph.addNode(output)
            graph.inputs = [1]
            graph.outputs = [2]

            let validator = GraphValidator()
            let errors = validator.validate(graph)

            #expect(errors.isEmpty)
            #expect(validator.isValid(graph))
        }

        @Test("Detect dangling inputs")
        func detectDanglingInputs() {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .add, inputs: [99, 100],  // Non-existent inputs
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(node)
            graph.outputs = [1]

            let validator = GraphValidator()
            let errors = validator.validate(graph)

            #expect(errors.count >= 2)
            #expect(errors.contains { error in
                if case .danglingInput(_, let inputId) = error {
                    return inputId == 99 || inputId == 100
                }
                return false
            })
        }

        @Test("Detect invalid outputs")
        func detectInvalidOutputs() {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(node)
            graph.outputs = [1, 999]  // 999 doesn't exist

            let validator = GraphValidator()
            let errors = validator.validate(graph)

            #expect(errors.contains { error in
                if case .invalidOutput(let outputId) = error {
                    return outputId == 999
                }
                return false
            })
        }

        @Test("Detect shape mismatch in matmul")
        func detectShapeMismatchMatmul() {
            let graph = OptimizationGraph()

            let input1 = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let input2 = OptimizationNode(
                id: 2, operation: .placeholder, inputs: [],
                shape: TensorShape([16, 4]),  // Wrong inner dimension
                dtype: .float32
            )
            let matmul = OptimizationNode(
                id: 3, operation: .matmul, inputs: [1, 2],
                shape: TensorShape([4, 4]), dtype: .float32
            )

            graph.addNode(input1)
            graph.addNode(input2)
            graph.addNode(matmul)
            graph.outputs = [3]

            let validator = GraphValidator()
            let errors = validator.validate(graph)

            #expect(errors.contains { error in
                if case .shapeMismatch = error {
                    return true
                }
                return false
            })
        }

        @Test("Detect dtype mismatch")
        func detectDTypeMismatch() {
            let graph = OptimizationGraph()

            let input = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let output = OptimizationNode(
                id: 2, operation: .relu, inputs: [1],
                shape: TensorShape([4, 8]), dtype: .float64  // Different dtype
            )

            graph.addNode(input)
            graph.addNode(output)
            graph.outputs = [2]

            let validator = GraphValidator()
            let errors = validator.validate(graph)

            #expect(errors.contains { error in
                if case .dtypeMismatch = error {
                    return true
                }
                return false
            })
        }
    }

    // MARK: - Graph Analysis Tests

    @Suite("Graph Analysis")
    struct GraphAnalysisTests {

        @Test("Analyze simple graph")
        func analyzeSimpleGraph() {
            let graph = OptimizationGraph()

            let input = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let relu = OptimizationNode(
                id: 2, operation: .relu, inputs: [1],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(input)
            graph.addNode(relu)
            graph.outputs = [2]

            let analyzer = GraphAnalyzer()
            let stats = analyzer.analyze(graph)

            #expect(stats.nodeCount == 2)
            #expect(stats.edgeCount == 1)
            #expect(stats.maxDepth == 1)
        }

        @Test("Compute operation counts")
        func computeOperationCounts() {
            let graph = OptimizationGraph()

            for i: UInt64 in 1...5 {
                let node = OptimizationNode(
                    id: i, operation: .constant, inputs: [],
                    shape: TensorShape.scalar, dtype: .float32
                )
                graph.addNode(node)
            }

            let add = OptimizationNode(
                id: 6, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )
            graph.addNode(add)
            graph.outputs = [6]

            let analyzer = GraphAnalyzer()
            let stats = analyzer.analyze(graph)

            #expect(stats.operationCounts["constant"] == 5)
            #expect(stats.operationCounts["add"] == 1)
        }

        @Test("Compute max depth")
        func computeMaxDepth() {
            let graph = OptimizationGraph()

            // Chain: 1 -> 2 -> 3 -> 4
            let node1 = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )
            let node2 = OptimizationNode(
                id: 2, operation: .relu, inputs: [1],
                shape: TensorShape.scalar, dtype: .float32
            )
            let node3 = OptimizationNode(
                id: 3, operation: .exp, inputs: [2],
                shape: TensorShape.scalar, dtype: .float32
            )
            let node4 = OptimizationNode(
                id: 4, operation: .log, inputs: [3],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(node1)
            graph.addNode(node2)
            graph.addNode(node3)
            graph.addNode(node4)
            graph.outputs = [4]

            let analyzer = GraphAnalyzer()
            let stats = analyzer.analyze(graph)

            #expect(stats.maxDepth == 3)
        }

        @Test("Estimate memory usage")
        func estimateMemoryUsage() {
            let graph = OptimizationGraph()

            // 4 * 8 * 4 bytes = 128 bytes for float32
            let node = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(node)
            graph.outputs = [1]

            let analyzer = GraphAnalyzer()
            let stats = analyzer.analyze(graph)

            #expect(stats.memoryEstimate == 128)
        }

        @Test("Statistics summary")
        func statisticsSummary() {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(node)
            graph.outputs = [1]

            let analyzer = GraphAnalyzer()
            let stats = analyzer.analyze(graph)
            let summary = stats.summary()

            #expect(summary.contains("Graph Statistics"))
            #expect(summary.contains("Nodes: 1"))
        }
    }

    // MARK: - Graph Comparison Tests

    @Suite("Graph Comparison")
    struct GraphComparisonTests {

        @Test("Compare equal graphs")
        func compareEqualGraphs() {
            let graph1 = OptimizationGraph()
            let graph2 = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph1.addNode(node)
            graph1.outputs = [1]

            graph2.addNode(node)
            graph2.outputs = [1]

            let comparator = GraphComparator()
            let result = comparator.compare(graph1, graph2)

            #expect(result.areEqual)
            #expect(result.differences.isEmpty)
        }

        @Test("Detect different node counts")
        func detectDifferentNodeCounts() {
            let graph1 = OptimizationGraph()
            let graph2 = OptimizationGraph()

            let node1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )
            let node2 = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph1.addNode(node1)
            graph1.outputs = [1]

            graph2.addNode(node1)
            graph2.addNode(node2)
            graph2.outputs = [1]

            let comparator = GraphComparator()
            let result = comparator.compare(graph1, graph2)

            #expect(!result.areEqual)
            #expect(result.differences.contains { $0.contains("Node count") })
        }

        @Test("Detect different operations")
        func detectDifferentOperations() {
            let graph1 = OptimizationGraph()
            let graph2 = OptimizationGraph()

            let node1 = OptimizationNode(
                id: 1, operation: .relu, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )
            let node2 = OptimizationNode(
                id: 1, operation: .sigmoid, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph1.addNode(node1)
            graph1.outputs = [1]

            graph2.addNode(node2)
            graph2.outputs = [1]

            let comparator = GraphComparator()
            let result = comparator.compare(graph1, graph2)

            #expect(!result.areEqual)
            #expect(result.differences.contains { $0.contains("operation") })
        }

        @Test("Comparison summary")
        func comparisonSummary() {
            let graph1 = OptimizationGraph()
            let graph2 = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph1.addNode(node)
            graph1.outputs = [1]
            graph2.addNode(node)
            graph2.outputs = [1]

            let comparator = GraphComparator()
            let result = comparator.compare(graph1, graph2)
            let summary = result.summary()

            #expect(summary.contains("equal"))
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 8 integration - full production workflow")
    func phase8FullWorkflow() throws {
        print("\n========================================")
        print("Phase 8: Full Production Workflow")
        print("========================================\n")

        // 1. Create a graph
        let graph = OptimizationGraph()

        let input = OptimizationNode(
            id: 1, operation: .placeholder, inputs: [],
            shape: TensorShape([32, 128]), dtype: .float32
        )
        let weight = OptimizationNode(
            id: 2, operation: .placeholder, inputs: [],
            shape: TensorShape([128, 64]), dtype: .float32
        )
        let matmul = OptimizationNode(
            id: 3, operation: .matmul, inputs: [1, 2],
            shape: TensorShape([32, 64]), dtype: .float32
        )
        let relu = OptimizationNode(
            id: 4, operation: .relu, inputs: [3],
            shape: TensorShape([32, 64]), dtype: .float32
        )

        graph.addNode(input)
        graph.addNode(weight)
        graph.addNode(matmul)
        graph.addNode(relu)
        graph.inputs = [1, 2]
        graph.outputs = [4]

        print("Created graph with 4 nodes")

        // 2. Validate the graph
        let validator = GraphValidator()
        let errors = validator.validate(graph)
        #expect(errors.isEmpty)
        print("✅ Graph validation passed")

        // 3. Analyze the graph
        let analyzer = GraphAnalyzer()
        let stats = analyzer.analyze(graph)
        print("✅ Graph analysis: \(stats.nodeCount) nodes, \(stats.maxDepth) depth")

        // 4. Serialize the graph
        let serializer = GraphSerializer()
        let json = try serializer.serializeToString(graph)
        #expect(!json.isEmpty)
        print("✅ Graph serialized to JSON (\(json.count) characters)")

        // 5. Deserialize and verify
        let restored = try serializer.deserializeFromString(json)
        #expect(restored.nodes.count == graph.nodes.count)
        print("✅ Graph deserialized successfully")

        // 6. Compare original and restored
        let comparator = GraphComparator()
        let comparison = comparator.compare(graph, restored)
        #expect(comparison.areEqual)
        print("✅ Round-trip serialization verified")

        print("\n========================================")
        print("✅ PHASE 8 FULL WORKFLOW COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 8 integration - error handling")
    func phase8ErrorHandling() {
        print("\n========================================")
        print("Phase 8: Error Handling Test")
        print("========================================\n")

        // Create invalid graph
        let graph = OptimizationGraph()

        let node = OptimizationNode(
            id: 1, operation: .add, inputs: [99, 100],  // Invalid inputs
            shape: TensorShape.scalar, dtype: .float32
        )

        graph.addNode(node)
        graph.outputs = [1, 999]  // Invalid output

        // Validate and check errors
        let validator = GraphValidator()
        let errors = validator.validate(graph)

        #expect(!errors.isEmpty)
        print("Detected \(errors.count) validation errors:")
        for error in errors {
            print("  - \(error)")
        }

        #expect(!validator.isValid(graph))

        print("\n========================================")
        print("✅ PHASE 8 ERROR HANDLING COMPLETE")
        print("========================================\n")
    }
}
