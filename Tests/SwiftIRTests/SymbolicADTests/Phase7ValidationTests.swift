/// Phase 7 Validation Tests - Developer Experience
/// Tests for graph visualization, diagnostics, checkpointing, and profiling

import Testing

@testable import SwiftIR

@Suite("Phase 7: Developer Experience", .serialized)
struct Phase7ValidationTests {

    init() {
        TracerGraphBuilder.shared.reset()
    }

    // MARK: - Graph Visualization Tests

    @Suite("Graph Visualization")
    struct GraphVisualizationTests {

        @Test("Export optimization graph to DOT")
        func exportOptimizationGraph() {
            let graph = OptimizationGraph()

            let const1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(2.0)]
            )
            let const2 = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(3.0)]
            )
            let add = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(add)
            graph.outputs = [3]

            let visualizer = GraphVisualizer()
            let dot = visualizer.toDOT(graph, name: "test")

            #expect(dot.contains("digraph test"))
            #expect(dot.contains("n1"))
            #expect(dot.contains("n2"))
            #expect(dot.contains("n3"))
            #expect(dot.contains("->"))
        }

        @Test("Export traced computation to DOT")
        func exportTracedComputation() {
            TracerGraphBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let y = Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32)
            let _ = x + y

            let visualizer = GraphVisualizer()
            let dot = visualizer.traceToDOT(name: "trace")

            #expect(dot.contains("digraph trace"))
            #expect(dot.contains("->"))
        }

        @Test("DOT output includes shape information")
        func dotIncludesShapes() {
            let graph = OptimizationGraph()

            let node = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )

            graph.addNode(node)
            graph.outputs = [1]

            let visualizer = GraphVisualizer()
            let dot = visualizer.toDOT(graph)

            #expect(dot.contains("4x8"))
        }
    }

    // MARK: - Diagnostics Tests

    @Suite("Diagnostics")
    struct DiagnosticsTests {

        @Test("Create diagnostic error")
        func createDiagnosticError() {
            let error = DiagnosticError(
                "Shape mismatch",
                notes: ["Expected [2, 3]", "Got [3, 2]"],
                suggestions: ["Transpose the input"]
            )

            let desc = error.description
            #expect(desc.contains("Shape mismatch"))
            #expect(desc.contains("Expected [2, 3]"))
            #expect(desc.contains("Transpose"))
        }

        @Test("Diagnostics collector - warnings")
        func collectWarnings() {
            let collector = DiagnosticsCollector()

            collector.warn("Unused variable")
            collector.warn("Deprecated operation")

            let warnings = collector.getWarnings()
            #expect(warnings.count == 2)
        }

        @Test("Diagnostics collector - errors")
        func collectErrors() {
            let collector = DiagnosticsCollector()

            collector.error(DiagnosticError("Error 1"))
            collector.error(DiagnosticError("Error 2"))

            #expect(collector.hasErrors())
            #expect(collector.getErrors().count == 2)
        }

        @Test("Clear diagnostics")
        func clearDiagnostics() {
            let collector = DiagnosticsCollector()

            collector.warn("Warning")
            collector.error(DiagnosticError("Error"))

            collector.clear()

            #expect(collector.getWarnings().isEmpty)
            #expect(collector.getErrors().isEmpty)
        }
    }

    // MARK: - Checkpointing Tests

    @Suite("Checkpointing")
    struct CheckpointingTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Create checkpoint")
        func createCheckpoint() {
            TracerGraphBuilder.shared.reset()

            let _ = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)

            let manager = CheckpointManager()
            let checkpoint = manager.createCheckpoint(name: "before_forward")

            #expect(checkpoint.id == "before_forward")
            #expect(checkpoint.operationCount >= 0)
        }

        @Test("Retrieve checkpoint")
        func retrieveCheckpoint() {
            let manager = CheckpointManager()
            let _ = manager.createCheckpoint(name: "test", metadata: ["stage": "forward"])

            let retrieved = manager.getCheckpoint("test")
            #expect(retrieved != nil)
            #expect(retrieved?.metadata["stage"] == "forward")
        }

        @Test("List checkpoints")
        func listCheckpoints() {
            let manager = CheckpointManager()
            let _ = manager.createCheckpoint(name: "cp1")
            let _ = manager.createCheckpoint(name: "cp2")
            let _ = manager.createCheckpoint(name: "cp3")

            let list = manager.listCheckpoints()
            #expect(list.count == 3)
        }

        @Test("Delete checkpoint")
        func deleteCheckpoint() {
            let manager = CheckpointManager()
            let _ = manager.createCheckpoint(name: "temp")

            manager.deleteCheckpoint("temp")

            #expect(manager.getCheckpoint("temp") == nil)
        }

        @Test("Get operation count at checkpoint")
        func operationCountAtCheckpoint() {
            TracerGraphBuilder.shared.reset()

            let manager = CheckpointManager()

            let _ = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let _ = manager.createCheckpoint(name: "cp1")

            let _ = Tracer(value: 2.0, shape: TensorShape([2, 3]), dtype: .float32)
            let _ = manager.createCheckpoint(name: "cp2")

            let count1 = manager.operationCount(at: "cp1")
            let count2 = manager.operationCount(at: "cp2")

            #expect(count1 != nil)
            #expect(count2 != nil)
            #expect(count2! >= count1!)
        }
    }

    // MARK: - Profiling Tests

    @Suite("Profiling")
    struct ProfilingTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Profile computation")
        func profileComputation() {
            TracerGraphBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([10, 20]), dtype: .float32)
            let y = Tracer(value: 2.0, shape: TensorShape([10, 20]), dtype: .float32)
            let _ = x + y
            let _ = x * y

            let profiler = Profiler()
            let data = profiler.profile(name: "test")

            #expect(data.totalOperations > 0)
            #expect(data.memoryEstimate > 0)
        }

        @Test("Profile summary")
        func profileSummary() {
            TracerGraphBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
            let _ = x.exp()

            let profiler = Profiler()
            let data = profiler.profile()

            let summary = data.summary()
            #expect(summary.contains("Profile Summary"))
            #expect(summary.contains("Total operations"))
        }

        @Test("Compare profiles")
        func compareProfiles() {
            let profiler = Profiler()

            TracerGraphBuilder.shared.reset()
            let _ = Tracer(value: 1.0, shape: TensorShape([4]), dtype: .float32)
            let _ = profiler.profile(name: "small")

            TracerGraphBuilder.shared.reset()
            let x = Tracer(value: 1.0, shape: TensorShape([100, 100]), dtype: .float32)
            let _ = x + x
            let _ = profiler.profile(name: "large")

            let comparison = profiler.compare("small", "large")
            #expect(comparison != nil)
            #expect(comparison!.contains("Profile Comparison"))
        }

        @Test("Retrieve saved profile")
        func retrieveSavedProfile() {
            TracerGraphBuilder.shared.reset()
            let x = Tracer(value: 1.0, shape: TensorShape([4]), dtype: .float32)
            let _ = x.exp()

            let profiler = Profiler()
            let _ = profiler.profile(name: "saved")

            let retrieved = profiler.getProfile("saved")
            #expect(retrieved != nil)
            #expect(retrieved!.totalOperations > 0)
        }
    }

    // MARK: - Shape Validation Tests

    @Suite("Shape Validation")
    struct ShapeValidationTests {

        @Test("Validate matmul shapes - valid")
        func validateMatmulValid() {
            let validator = ShapeValidator()

            let result = validator.validate(
                operation: "matmul",
                inputs: [
                    ("a", TensorShape([4, 8])),
                    ("b", TensorShape([8, 16]))
                ],
                expectedOutput: TensorShape([4, 16])
            )

            if case .failure = result {
                Issue.record("Expected validation to pass")
            }
        }

        @Test("Validate matmul shapes - invalid")
        func validateMatmulInvalid() {
            let validator = ShapeValidator()

            let result = validator.validate(
                operation: "matmul",
                inputs: [
                    ("a", TensorShape([4, 8])),
                    ("b", TensorShape([16, 4]))  // Inner dims don't match
                ],
                expectedOutput: nil
            )

            if case .success = result {
                Issue.record("Expected validation to fail")
            }
        }

        @Test("Validate broadcast shapes")
        func validateBroadcast() {
            let validator = ShapeValidator()

            let result = validator.validate(
                operation: "add",
                inputs: [
                    ("a", TensorShape([4, 1])),
                    ("b", TensorShape([1, 8]))
                ],
                expectedOutput: TensorShape([4, 8])
            )

            if case .failure = result {
                Issue.record("Expected validation to pass")
            }
        }

        @Test("Detect unknown dimensions")
        func detectUnknownDimensions() {
            let validator = ShapeValidator()

            let result = validator.validate(
                operation: "add",
                inputs: [
                    ("a", TensorShape(dimensions: [nil, 8])),
                    ("b", TensorShape([4, 8]))
                ],
                expectedOutput: nil
            )

            if case .success = result {
                Issue.record("Expected validation to fail due to unknown dimension")
            }
        }
    }

    // MARK: - Debug Utilities Tests

    @Suite("Debug Utilities")
    struct DebugUtilitiesTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Get computation summary")
        func getComputationSummary() {
            TracerGraphBuilder.shared.reset()

            let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
            let y = Tracer(value: 2.0, shape: TensorShape([4, 8]), dtype: .float32)
            let _ = (x + y).exp()

            let debug = DebugUtilities()
            let summary = debug.summary()

            #expect(summary.contains("Profile Summary"))
            #expect(summary.contains("Total operations"))
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 7 integration - full development workflow")
    func phase7FullWorkflow() {
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 7: Full Development Workflow")
        print("========================================\n")

        // 1. Create computation
        let x = Tracer(value: 1.0, shape: TensorShape([32, 128]), dtype: .float32)
        let w = Tracer(value: 0.1, shape: TensorShape([128, 64]), dtype: .float32)

        // 2. Checkpoint before forward pass
        let checkpointManager = CheckpointManager()
        let _ = checkpointManager.createCheckpoint(name: "before_forward", metadata: ["stage": "init"])
        print("✅ Created checkpoint 'before_forward'")

        // 3. Forward pass
        let h = x.matmul(w)
        let output = h.relu()

        // 4. Profile the computation
        let profiler = Profiler()
        let profile = profiler.profile(name: "forward_pass")
        print("✅ Profiled computation: \(profile.totalOperations) operations")

        // 5. Validate shapes
        let validator = ShapeValidator()
        let validationResult = validator.validate(
            operation: "matmul",
            inputs: [("x", x.shape), ("w", w.shape)],
            expectedOutput: h.shape
        )

        switch validationResult {
        case .success:
            print("✅ Shape validation passed")
        case .failure(let error):
            print("❌ Shape validation failed: \(error.message)")
        }

        // 6. Export to DOT
        let visualizer = GraphVisualizer()
        let dot = visualizer.traceToDOT(name: "neural_network")
        #expect(dot.contains("digraph"))
        print("✅ Generated DOT visualization")

        // 7. Get summary
        let debug = DebugUtilities()
        let summary = debug.summary()
        print("\n\(summary)")

        #expect(output.shape == TensorShape([32, 64]))

        print("\n========================================")
        print("✅ PHASE 7 FULL WORKFLOW COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 7 integration - diagnostics workflow")
    func phase7DiagnosticsWorkflow() {
        print("\n========================================")
        print("Phase 7: Diagnostics Workflow")
        print("========================================\n")

        let collector = DiagnosticsCollector()

        // Simulate collecting diagnostics during compilation
        collector.warn("Large tensor allocation detected")
        collector.warn("Consider using gradient checkpointing")

        // Simulate a shape error
        let validator = ShapeValidator()
        let result = validator.validate(
            operation: "matmul",
            inputs: [
                ("weights", TensorShape([64, 128])),
                ("input", TensorShape([256, 64]))  // Wrong order
            ],
            expectedOutput: nil
        )

        if case .failure(let error) = result {
            collector.error(error)
        }

        // Print diagnostics
        let warnings = collector.getWarnings()
        let errors = collector.getErrors()

        print("Warnings: \(warnings.count)")
        print("Errors: \(errors.count)")

        #expect(warnings.count == 2)
        #expect(errors.count == 1)

        print("\n========================================")
        print("✅ PHASE 7 DIAGNOSTICS WORKFLOW COMPLETE")
        print("========================================\n")
    }
}
