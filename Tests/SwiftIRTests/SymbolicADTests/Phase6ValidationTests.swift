/// Phase 6 Validation Tests - Optimization Pipeline
/// Tests for constant folding, DCE, CSE, operation fusion, and algebraic simplification

import Testing

@testable import SwiftIR

@Suite("Phase 6: Optimization Pipeline", .serialized)
struct Phase6ValidationTests {

    // MARK: - Constant Folding Tests

    @Suite("Constant Folding")
    struct ConstantFoldingTests {

        @Test("Fold addition of constants")
        func foldAddition() {
            let graph = OptimizationGraph()

            // Create: 2.0 + 3.0
            let const1 = OptimizationNode(
                id: 1,
                operation: .constant,
                inputs: [],
                shape: TensorShape.scalar,
                dtype: .float32,
                attributes: ["value": .double(2.0)]
            )
            let const2 = OptimizationNode(
                id: 2,
                operation: .constant,
                inputs: [],
                shape: TensorShape.scalar,
                dtype: .float32,
                attributes: ["value": .double(3.0)]
            )
            let add = OptimizationNode(
                id: 3,
                operation: .add,
                inputs: [1, 2],
                shape: TensorShape.scalar,
                dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(add)
            graph.outputs = [3]

            let pass = ConstantFoldingPass()
            let result = pass.run(on: graph)

            // Add node should be folded to constant
            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(5.0))
        }

        @Test("Fold multiplication of constants")
        func foldMultiplication() {
            let graph = OptimizationGraph()

            let const1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(4.0)]
            )
            let const2 = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(5.0)]
            )
            let mul = OptimizationNode(
                id: 3, operation: .multiply, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(mul)
            graph.outputs = [3]

            let pass = ConstantFoldingPass()
            let result = pass.run(on: graph)

            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(20.0))
        }

        @Test("Fold unary operations")
        func foldUnary() {
            let graph = OptimizationGraph()

            let const1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(-5.0)]
            )
            let neg = OptimizationNode(
                id: 2, operation: .neg, inputs: [1],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(neg)
            graph.outputs = [2]

            let pass = ConstantFoldingPass()
            let result = pass.run(on: graph)

            let resultNode = result.nodes[2]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(5.0))
        }

        @Test("Chain constant folding")
        func chainFolding() {
            let graph = OptimizationGraph()

            // Create: (2.0 + 3.0) * 4.0
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
            let const3 = OptimizationNode(
                id: 3, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(4.0)]
            )
            let add = OptimizationNode(
                id: 4, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )
            let mul = OptimizationNode(
                id: 5, operation: .multiply, inputs: [4, 3],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(const3)
            graph.addNode(add)
            graph.addNode(mul)
            graph.outputs = [5]

            let pass = ConstantFoldingPass()
            let result = pass.run(on: graph)

            // Final result should be folded to 20.0
            let resultNode = result.nodes[5]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(20.0))
        }
    }

    // MARK: - Dead Code Elimination Tests

    @Suite("Dead Code Elimination")
    struct DeadCodeEliminationTests {

        @Test("Remove unused nodes")
        func removeUnused() {
            let graph = OptimizationGraph()

            let const1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(1.0)]
            )
            let const2 = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(2.0)]
            )
            // const2 is not used
            let add = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 1],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(add)
            graph.outputs = [3]

            let pass = DeadCodeEliminationPass()
            let result = pass.run(on: graph)

            #expect(result.nodes[1] != nil)  // const1 is used
            #expect(result.nodes[2] == nil)  // const2 is removed
            #expect(result.nodes[3] != nil)  // add is output
        }

        @Test("Keep transitively used nodes")
        func keepTransitive() {
            let graph = OptimizationGraph()

            let const1 = OptimizationNode(
                id: 1, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(1.0)]
            )
            let neg = OptimizationNode(
                id: 2, operation: .neg, inputs: [1],
                shape: TensorShape.scalar, dtype: .float32
            )
            let exp = OptimizationNode(
                id: 3, operation: .exp, inputs: [2],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(neg)
            graph.addNode(exp)
            graph.outputs = [3]

            let pass = DeadCodeEliminationPass()
            let result = pass.run(on: graph)

            // All nodes should be kept (transitively used)
            #expect(result.nodeCount == 3)
        }
    }

    // MARK: - Common Subexpression Elimination Tests

    @Suite("Common Subexpression Elimination")
    struct CSETests {

        @Test("Eliminate duplicate expressions")
        func eliminateDuplicates() {
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
            // Two identical adds
            let add1 = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )
            let add2 = OptimizationNode(
                id: 4, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )
            // Use both
            let mul = OptimizationNode(
                id: 5, operation: .multiply, inputs: [3, 4],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(add1)
            graph.addNode(add2)
            graph.addNode(mul)
            graph.outputs = [5]

            let pass = CommonSubexpressionEliminationPass()
            let result = pass.run(on: graph)

            // add2 should be eliminated, mul should use add1 twice
            #expect(result.nodes[4] == nil)

            // mul should have both inputs as 3
            let mulNode = result.nodes[5]!
            #expect(mulNode.inputs == [3, 3])
        }

        @Test("Keep different expressions")
        func keepDifferent() {
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
            // Different operations
            let add = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )
            let mul = OptimizationNode(
                id: 4, operation: .multiply, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(add)
            graph.addNode(mul)
            graph.outputs = [3, 4]

            let pass = CommonSubexpressionEliminationPass()
            let result = pass.run(on: graph)

            // Both should be kept (different operations)
            #expect(result.nodeCount == 4)
        }
    }

    // MARK: - Operation Fusion Tests

    @Suite("Operation Fusion")
    struct OperationFusionTests {

        @Test("Fuse matmul and add")
        func fuseMatmulAdd() {
            let graph = OptimizationGraph()

            let input = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4, 8]), dtype: .float32
            )
            let weight = OptimizationNode(
                id: 2, operation: .placeholder, inputs: [],
                shape: TensorShape([8, 4]), dtype: .float32
            )
            let bias = OptimizationNode(
                id: 3, operation: .placeholder, inputs: [],
                shape: TensorShape([1, 4]), dtype: .float32
            )
            let matmul = OptimizationNode(
                id: 4, operation: .matmul, inputs: [1, 2],
                shape: TensorShape([4, 4]), dtype: .float32
            )
            let add = OptimizationNode(
                id: 5, operation: .add, inputs: [4, 3],
                shape: TensorShape([4, 4]), dtype: .float32
            )

            graph.addNode(input)
            graph.addNode(weight)
            graph.addNode(bias)
            graph.addNode(matmul)
            graph.addNode(add)
            graph.inputs = [1, 2, 3]
            graph.outputs = [5]

            let pass = OperationFusionPass()
            let result = pass.run(on: graph)

            // Add should become fusedMatmulAdd
            let fusedNode = result.nodes[5]!
            #expect(fusedNode.operation == .fusedMatmulAdd)
            #expect(fusedNode.inputs.count == 3)  // input, weight, bias
        }

        @Test("Fuse multiply and add (FMA)")
        func fuseMulAdd() {
            let graph = OptimizationGraph()

            let a = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let b = OptimizationNode(
                id: 2, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let c = OptimizationNode(
                id: 3, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let mul = OptimizationNode(
                id: 4, operation: .multiply, inputs: [1, 2],
                shape: TensorShape([4]), dtype: .float32
            )
            let add = OptimizationNode(
                id: 5, operation: .add, inputs: [4, 3],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(a)
            graph.addNode(b)
            graph.addNode(c)
            graph.addNode(mul)
            graph.addNode(add)
            graph.inputs = [1, 2, 3]
            graph.outputs = [5]

            let pass = OperationFusionPass()
            let result = pass.run(on: graph)

            let fusedNode = result.nodes[5]!
            #expect(fusedNode.operation == .fusedMulAdd)
            #expect(fusedNode.inputs.count == 3)  // a, b, c
        }

        @Test("Don't fuse when multiply has multiple uses")
        func noFuseMultipleUses() {
            let graph = OptimizationGraph()

            let a = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let b = OptimizationNode(
                id: 2, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let c = OptimizationNode(
                id: 3, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let mul = OptimizationNode(
                id: 4, operation: .multiply, inputs: [1, 2],
                shape: TensorShape([4]), dtype: .float32
            )
            let add = OptimizationNode(
                id: 5, operation: .add, inputs: [4, 3],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(a)
            graph.addNode(b)
            graph.addNode(c)
            graph.addNode(mul)
            graph.addNode(add)
            graph.inputs = [1, 2, 3]
            graph.outputs = [4, 5]  // mul is also an output

            let pass = OperationFusionPass()
            let result = pass.run(on: graph)

            // Should not fuse because mul has multiple uses
            let addNode = result.nodes[5]!
            #expect(addNode.operation == .add)
        }
    }

    // MARK: - Algebraic Simplification Tests

    @Suite("Algebraic Simplification")
    struct AlgebraicSimplificationTests {

        @Test("Simplify x + 0 = x")
        func simplifyAddZero() {
            let graph = OptimizationGraph()

            let x = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let zero = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape([4]), dtype: .float32,
                attributes: ["value": .double(0.0)]
            )
            let add = OptimizationNode(
                id: 3, operation: .add, inputs: [1, 2],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(x)
            graph.addNode(zero)
            graph.addNode(add)
            graph.outputs = [3]

            let pass = AlgebraicSimplificationPass()
            let result = pass.run(on: graph)

            // add should be replaced with x
            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .placeholder)
        }

        @Test("Simplify x * 1 = x")
        func simplifyMulOne() {
            let graph = OptimizationGraph()

            let x = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let one = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape([4]), dtype: .float32,
                attributes: ["value": .double(1.0)]
            )
            let mul = OptimizationNode(
                id: 3, operation: .multiply, inputs: [1, 2],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(x)
            graph.addNode(one)
            graph.addNode(mul)
            graph.outputs = [3]

            let pass = AlgebraicSimplificationPass()
            let result = pass.run(on: graph)

            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .placeholder)
        }

        @Test("Simplify x * 0 = 0")
        func simplifyMulZero() {
            let graph = OptimizationGraph()

            let x = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let zero = OptimizationNode(
                id: 2, operation: .constant, inputs: [],
                shape: TensorShape([4]), dtype: .float32,
                attributes: ["value": .double(0.0)]
            )
            let mul = OptimizationNode(
                id: 3, operation: .multiply, inputs: [1, 2],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(x)
            graph.addNode(zero)
            graph.addNode(mul)
            graph.outputs = [3]

            let pass = AlgebraicSimplificationPass()
            let result = pass.run(on: graph)

            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(0.0))
        }

        @Test("Simplify double negation")
        func simplifyDoubleNeg() {
            let graph = OptimizationGraph()

            let x = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let neg1 = OptimizationNode(
                id: 2, operation: .neg, inputs: [1],
                shape: TensorShape([4]), dtype: .float32
            )
            let neg2 = OptimizationNode(
                id: 3, operation: .neg, inputs: [2],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(x)
            graph.addNode(neg1)
            graph.addNode(neg2)
            graph.outputs = [3]

            let pass = AlgebraicSimplificationPass()
            let result = pass.run(on: graph)

            // -(-x) = x
            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .placeholder)
        }

        @Test("Simplify exp(log(x)) = x")
        func simplifyExpLog() {
            let graph = OptimizationGraph()

            let x = OptimizationNode(
                id: 1, operation: .placeholder, inputs: [],
                shape: TensorShape([4]), dtype: .float32
            )
            let log = OptimizationNode(
                id: 2, operation: .log, inputs: [1],
                shape: TensorShape([4]), dtype: .float32
            )
            let exp = OptimizationNode(
                id: 3, operation: .exp, inputs: [2],
                shape: TensorShape([4]), dtype: .float32
            )

            graph.addNode(x)
            graph.addNode(log)
            graph.addNode(exp)
            graph.outputs = [3]

            let pass = AlgebraicSimplificationPass()
            let result = pass.run(on: graph)

            let resultNode = result.nodes[3]!
            #expect(resultNode.operation == .placeholder)
        }
    }

    // MARK: - Pipeline Tests

    @Suite("Optimization Pipeline")
    struct PipelineTests {

        @Test("Standard pipeline creation")
        func standardPipeline() {
            let pipeline = OptimizationPipeline.standard()
            #expect(pipeline.passCount == 5)
        }

        @Test("Minimal pipeline creation")
        func minimalPipeline() {
            let pipeline = OptimizationPipeline.minimal()
            #expect(pipeline.passCount == 2)
        }

        @Test("Custom pipeline")
        func customPipeline() {
            let pipeline = OptimizationPipeline()
            pipeline.addPass(ConstantFoldingPass())
            pipeline.addPass(DeadCodeEliminationPass())

            #expect(pipeline.passCount == 2)
        }

        @Test("Run full pipeline")
        func runFullPipeline() {
            let graph = OptimizationGraph()

            // Create a graph that benefits from multiple passes
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
            let unused = OptimizationNode(
                id: 3, operation: .constant, inputs: [],
                shape: TensorShape.scalar, dtype: .float32,
                attributes: ["value": .double(100.0)]
            )
            let add = OptimizationNode(
                id: 4, operation: .add, inputs: [1, 2],
                shape: TensorShape.scalar, dtype: .float32
            )

            graph.addNode(const1)
            graph.addNode(const2)
            graph.addNode(unused)
            graph.addNode(add)
            graph.outputs = [4]

            let pipeline = OptimizationPipeline.standard()
            let result = pipeline.run(on: graph)

            // Constants should be folded, unused removed
            #expect(result.nodeCount <= 2)

            // Result should be constant 5.0
            let resultNode = result.nodes[4]!
            #expect(resultNode.operation == .constant)
            #expect(resultNode.attributes["value"] == .double(5.0))
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 6 integration - neural network optimization")
    func phase6NeuralNetworkOptimization() {
        print("\n========================================")
        print("Phase 6: Neural Network Optimization")
        print("========================================\n")

        let graph = OptimizationGraph()

        // Create: y = relu(x @ w + b) with some optimizable patterns
        let x = OptimizationNode(
            id: 1, operation: .placeholder, inputs: [],
            shape: TensorShape([32, 128]), dtype: .float32
        )
        let w = OptimizationNode(
            id: 2, operation: .placeholder, inputs: [],
            shape: TensorShape([128, 64]), dtype: .float32
        )
        let b = OptimizationNode(
            id: 3, operation: .placeholder, inputs: [],
            shape: TensorShape([1, 64]), dtype: .float32
        )
        let zero = OptimizationNode(
            id: 4, operation: .constant, inputs: [],
            shape: TensorShape([1, 64]), dtype: .float32,
            attributes: ["value": .double(0.0)]
        )
        let matmul = OptimizationNode(
            id: 5, operation: .matmul, inputs: [1, 2],
            shape: TensorShape([32, 64]), dtype: .float32
        )
        let addBias = OptimizationNode(
            id: 6, operation: .add, inputs: [5, 3],
            shape: TensorShape([32, 64]), dtype: .float32
        )
        let addZero = OptimizationNode(
            id: 7, operation: .add, inputs: [6, 4],
            shape: TensorShape([32, 64]), dtype: .float32
        )
        let relu = OptimizationNode(
            id: 8, operation: .relu, inputs: [7],
            shape: TensorShape([32, 64]), dtype: .float32
        )

        graph.addNode(x)
        graph.addNode(w)
        graph.addNode(b)
        graph.addNode(zero)
        graph.addNode(matmul)
        graph.addNode(addBias)
        graph.addNode(addZero)
        graph.addNode(relu)
        graph.inputs = [1, 2, 3]
        graph.outputs = [8]

        let beforeStats = GraphStatistics(graph: graph)
        print("Before optimization: \(beforeStats.nodeCount) nodes")

        let pipeline = OptimizationPipeline.standard()
        let optimized = pipeline.run(on: graph)

        let afterStats = GraphStatistics(graph: optimized)
        print("After optimization: \(afterStats.nodeCount) nodes")

        // Should have optimized away addZero
        #expect(afterStats.nodeCount < beforeStats.nodeCount)
        print("✅ Removed \(beforeStats.nodeCount - afterStats.nodeCount) redundant nodes")

        // Check for fused operations
        let hasFused = optimized.nodes.values.contains { $0.operation == .fusedMatmulAdd }
        if hasFused {
            print("✅ Fused matmul+add operation")
        }

        print("\n========================================")
        print("✅ PHASE 6 NEURAL NETWORK OPTIMIZATION COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 6 integration - expression simplification")
    func phase6ExpressionSimplification() {
        print("\n========================================")
        print("Phase 6: Expression Simplification")
        print("========================================\n")

        let graph = OptimizationGraph()

        // Create: exp(log(x)) * 1 + 0 - this should simplify to just x
        let x = OptimizationNode(
            id: 1, operation: .placeholder, inputs: [],
            shape: TensorShape([4]), dtype: .float32
        )
        let log = OptimizationNode(
            id: 2, operation: .log, inputs: [1],
            shape: TensorShape([4]), dtype: .float32
        )
        let exp = OptimizationNode(
            id: 3, operation: .exp, inputs: [2],
            shape: TensorShape([4]), dtype: .float32
        )
        let one = OptimizationNode(
            id: 4, operation: .constant, inputs: [],
            shape: TensorShape([4]), dtype: .float32,
            attributes: ["value": .double(1.0)]
        )
        let zero = OptimizationNode(
            id: 5, operation: .constant, inputs: [],
            shape: TensorShape([4]), dtype: .float32,
            attributes: ["value": .double(0.0)]
        )
        let mul = OptimizationNode(
            id: 6, operation: .multiply, inputs: [3, 4],
            shape: TensorShape([4]), dtype: .float32
        )
        let add = OptimizationNode(
            id: 7, operation: .add, inputs: [6, 5],
            shape: TensorShape([4]), dtype: .float32
        )

        graph.addNode(x)
        graph.addNode(log)
        graph.addNode(exp)
        graph.addNode(one)
        graph.addNode(zero)
        graph.addNode(mul)
        graph.addNode(add)
        graph.inputs = [1]
        graph.outputs = [7]

        let beforeStats = GraphStatistics(graph: graph)
        print("Before: \(beforeStats.nodeCount) nodes")

        let pipeline = OptimizationPipeline.standard()
        let optimized = pipeline.run(on: graph)

        let afterStats = GraphStatistics(graph: optimized)
        print("After: \(afterStats.nodeCount) nodes")

        #expect(afterStats.nodeCount < beforeStats.nodeCount)
        print("✅ Simplified from \(beforeStats.nodeCount) to \(afterStats.nodeCount) nodes")

        print("\n========================================")
        print("✅ PHASE 6 EXPRESSION SIMPLIFICATION COMPLETE")
        print("========================================\n")
    }
}
