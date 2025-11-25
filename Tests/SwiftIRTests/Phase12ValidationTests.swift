// Phase12ValidationTests.swift
// Tests for Phase 12: End-to-End Integration

import XCTest
@testable import SwiftIR

final class Phase12ValidationTests: XCTestCase {

    // MARK: - Pipeline Options Tests

    func testDefaultPipelineOptions() {
        let options = PipelineOptions.default

        XCTAssertEqual(options.optimizationLevel, .standard)
        XCTAssertTrue(options.enableCSE)
        XCTAssertTrue(options.enableConstantFolding)
        XCTAssertTrue(options.enableDeadCodeElimination)
        XCTAssertTrue(options.enableFusion)
        XCTAssertFalse(options.debugMode)
    }

    func testCustomPipelineOptions() {
        let options = PipelineOptions(
            target: .cuda(deviceId: 1),
            optimizationLevel: .aggressive,
            enableCSE: false,
            enableConstantFolding: false,
            enableDeadCodeElimination: false,
            enableFusion: false,
            debugMode: true
        )

        XCTAssertEqual(options.optimizationLevel, .aggressive)
        XCTAssertFalse(options.enableCSE)
        XCTAssertTrue(options.debugMode)
    }

    // MARK: - Basic Pipeline Tests

    func testPipelineSimpleFunction() throws {
        let pipeline = SwiftIRPipeline()

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return x + x
        }

        XCTAssertEqual(executable.inputSpecs.count, 1)
        XCTAssertEqual(executable.inputSpecs[0].shape, [10])
        XCTAssertEqual(executable.outputShape, [10])
    }

    func testPipelineMultipleOperations() throws {
        let pipeline = SwiftIRPipeline()

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            let y = x * x
            let z = y + x
            return z
        }

        XCTAssertGreaterThan(executable.originalGraph.nodes.count, 1)
    }

    func testPipelineMultipleInputs() throws {
        let pipeline = SwiftIRPipeline()

        let executable = try pipeline.compile(
            inputSpecs: [
                TensorSpec(shape: [10], dtype: .float32),
                TensorSpec(shape: [10], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] + inputs[1]
        }

        XCTAssertEqual(executable.inputSpecs.count, 2)
    }

    func testPipelineMatmul() throws {
        let pipeline = SwiftIRPipeline()

        let executable = try pipeline.compile(
            inputSpecs: [
                TensorSpec(shape: [10, 20], dtype: .float32),
                TensorSpec(shape: [20, 30], dtype: .float32)
            ]
        ) { inputs in
            return matmul(inputs[0], inputs[1])
        }

        XCTAssertNotNil(executable)
        XCTAssertEqual(executable.outputShape, [10, 30])
    }

    // MARK: - Optimization Tests

    func testOptimizationStatsCalculation() {
        let stats = OptimizationStats(
            originalNodeCount: 10,
            optimizedNodeCount: 7,
            nodesEliminated: 3
        )

        XCTAssertEqual(stats.reductionPercentage, 30.0)
    }

    func testOptimizationStatsZeroNodes() {
        let stats = OptimizationStats(
            originalNodeCount: 0,
            optimizedNodeCount: 0,
            nodesEliminated: 0
        )

        XCTAssertEqual(stats.reductionPercentage, 0.0)
    }

    func testPipelineWithOptimizations() throws {
        let options = PipelineOptions(
            enableCSE: true,
            enableConstantFolding: true,
            enableDeadCodeElimination: true
        )
        let pipeline = SwiftIRPipeline(options: options)

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            let a = x + x
            let b = x + x  // Duplicate - CSE should eliminate
            return a + b
        }

        XCTAssertNotNil(executable.optimizationStats)
    }

    func testPipelineWithoutOptimizations() throws {
        let options = PipelineOptions(
            enableCSE: false,
            enableConstantFolding: false,
            enableDeadCodeElimination: false
        )
        let pipeline = SwiftIRPipeline(options: options)

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return x * x
        }

        XCTAssertNotNil(executable)
    }

    // MARK: - High-Level API Tests

    func testCompileOptimizedSingleInput() throws {
        let executable = try compileOptimized(
            input: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return relu(x)
        }

        XCTAssertEqual(executable.inputSpecs.count, 1)
    }

    func testCompileOptimizedMultiInput() throws {
        let executable = try compileOptimized(
            inputs: [
                TensorSpec(shape: [10], dtype: .float32),
                TensorSpec(shape: [10], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] * inputs[1]
        }

        XCTAssertEqual(executable.inputSpecs.count, 2)
    }

    func testCompileOptimizedWithOptions() throws {
        let options = PipelineOptions(
            target: .cuda(deviceId: 0),
            optimizationLevel: .aggressive
        )

        let executable = try compileOptimized(
            input: TensorSpec(shape: [10], dtype: .float32),
            options: options
        ) { x in
            return exp(x)
        }

        XCTAssertNotNil(executable)
    }

    // MARK: - Neural Network Tests

    func testLayerSpecDense() {
        let layer = LayerSpec.dense(
            inputSize: 784,
            outputSize: 128,
            activation: .relu
        )

        XCTAssertEqual(layer.weightShape, [784, 128])
        XCTAssertEqual(layer.biasShape, [128])
        XCTAssertTrue(layer.hasBias)
        XCTAssertEqual(layer.activation, .relu)
    }

    func testLayerSpecCustom() {
        let layer = LayerSpec(
            weightShape: [100, 50],
            biasShape: [50],
            hasBias: true,
            activation: .sigmoid
        )

        XCTAssertEqual(layer.weightShape, [100, 50])
        XCTAssertEqual(layer.activation, .sigmoid)
    }

    func testLayerSpecNoBias() {
        let layer = LayerSpec(
            weightShape: [100, 50],
            hasBias: false,
            activation: .none
        )

        XCTAssertFalse(layer.hasBias)
    }

    func testCompileNeuralNetwork() throws {
        let layers = [
            LayerSpec.dense(inputSize: 784, outputSize: 128, activation: .relu),
            LayerSpec.dense(inputSize: 128, outputSize: 10, activation: .none)
        ]

        let executable = try compileNeuralNetwork(
            layers: layers,
            inputSpec: TensorSpec(shape: [32, 784], dtype: .float32)
        )

        // Input + 2 weights + 2 biases = 5 specs
        XCTAssertEqual(executable.inputSpecs.count, 5)
    }

    func testCompileNeuralNetworkSingleLayer() throws {
        let layers = [
            LayerSpec.dense(inputSize: 10, outputSize: 5, activation: .relu)
        ]

        let executable = try compileNeuralNetwork(
            layers: layers,
            inputSpec: TensorSpec(shape: [4, 10], dtype: .float32)
        )

        // Input + 1 weight + 1 bias = 3 specs
        XCTAssertEqual(executable.inputSpecs.count, 3)
    }

    // MARK: - Training Support Tests

    func testModelSpec() {
        let spec = ModelSpec(
            inputSpec: TensorSpec(shape: [32, 784], dtype: .float32),
            labelSpec: TensorSpec(shape: [32, 10], dtype: .float32),
            parameterSpecs: [
                TensorSpec(shape: [784, 128], dtype: .float32),
                TensorSpec(shape: [128, 10], dtype: .float32)
            ]
        )

        XCTAssertEqual(spec.inputSpec.shape, [32, 784])
        XCTAssertEqual(spec.labelSpec.shape, [32, 10])
        XCTAssertEqual(spec.parameterSpecs.count, 2)
    }

    func testCompileTrainingStep() throws {
        let modelSpec = ModelSpec(
            inputSpec: TensorSpec(shape: [4, 10], dtype: .float32),
            labelSpec: TensorSpec(shape: [4, 5], dtype: .float32),
            parameterSpecs: [
                TensorSpec(shape: [10, 5], dtype: .float32)
            ]
        )

        let training = try compileTrainingStep(modelSpec: modelSpec)

        XCTAssertNotNil(training.executable)
        XCTAssertFalse(training.info.isEmpty)
    }

    // MARK: - Execution Tests

    func testExecutableRun() throws {
        let executable = try compileOptimized(
            input: TensorSpec(shape: [5], dtype: .float32)
        ) { x in
            return x + x
        }

        let input = [[Float(1), 2, 3, 4, 5]]
        let output = executable.run(input)

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].count, 5)
    }

    func testExecutableInfo() throws {
        let executable = try compileOptimized(
            input: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return -x
        }

        let info = executable.info
        XCTAssertTrue(info.contains("Optimized Executable"))
        XCTAssertTrue(info.contains("Input"))
        XCTAssertTrue(info.contains("Output"))
    }

    // MARK: - Complex Pipeline Tests

    func testPipelineMLP() throws {
        let executable = try compileOptimized(
            inputs: [
                TensorSpec(shape: [32, 784], dtype: .float32),
                TensorSpec(shape: [784, 256], dtype: .float32),
                TensorSpec(shape: [256], dtype: .float32),
                TensorSpec(shape: [256, 10], dtype: .float32),
                TensorSpec(shape: [10], dtype: .float32)
            ]
        ) { inputs in
            let x = inputs[0]
            let w1 = inputs[1]
            let b1 = inputs[2]
            let w2 = inputs[3]
            let b2 = inputs[4]

            let h1 = relu(matmul(x, w1) + b1)
            let output = matmul(h1, w2) + b2

            return output
        }

        XCTAssertEqual(executable.inputSpecs.count, 5)
        XCTAssertNotNil(executable.optimizationStats)
    }

    func testPipelineChainedOperations() throws {
        let executable = try compileOptimized(
            input: TensorSpec(shape: [10, 10], dtype: .float32)
        ) { x in
            let a = x * x
            let b = a + x
            let c = relu(b)
            let d = exp(c)
            return d
        }

        XCTAssertGreaterThan(executable.originalGraph.nodes.count, 3)
    }

    func testPipelineWithTranspose() throws {
        let executable = try compileOptimized(
            input: TensorSpec(shape: [10, 20], dtype: .float32)
        ) { x in
            let t = transpose(x)
            return matmul(t, x)
        }

        XCTAssertEqual(executable.outputShape, [20, 20])
    }

    // MARK: - Different Targets Tests

    func testPipelineCPUTarget() throws {
        let options = PipelineOptions(target: .cpu)
        let pipeline = SwiftIRPipeline(options: options)

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in x + x }

        XCTAssertTrue(executable.info.contains("CPU"))
    }

    func testPipelineCUDATarget() throws {
        let options = PipelineOptions(target: .cuda(deviceId: 0))
        let pipeline = SwiftIRPipeline(options: options)

        let executable = try pipeline.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in x * x }

        XCTAssertTrue(executable.info.contains("CUDA"))
    }

    // MARK: - Integration Tests

    func testFullPipelineIntegration() throws {
        // Create pipeline
        let pipeline = SwiftIRPipeline(options: .default)

        // Compile a simple model
        let executable = try pipeline.compile(
            inputSpecs: [
                TensorSpec(shape: [4, 10], dtype: .float32),
                TensorSpec(shape: [10, 5], dtype: .float32)
            ]
        ) { inputs in
            let x = inputs[0]
            let w = inputs[1]
            return relu(matmul(x, w))
        }

        // Check all components are present
        XCTAssertNotNil(executable.compiled)
        XCTAssertNotNil(executable.originalGraph)
        XCTAssertNotNil(executable.optimizedGraph)
        XCTAssertEqual(executable.inputSpecs.count, 2)
        XCTAssertEqual(executable.outputShape, [4, 5])

        // Run the executable
        let x = [[Float]](repeating: Array(repeating: 1.0, count: 10), count: 4).flatMap { $0 }
        let w = [[Float]](repeating: Array(repeating: 0.1, count: 5), count: 10).flatMap { $0 }
        let output = executable.run([[Float](x), [Float](w)])

        XCTAssertEqual(output.count, 2)
    }

    func testEndToEndNeuralNetwork() throws {
        // Define a simple MLP
        let layers = [
            LayerSpec.dense(inputSize: 10, outputSize: 8, activation: .relu),
            LayerSpec.dense(inputSize: 8, outputSize: 4, activation: .none)
        ]

        // Compile
        let executable = try compileNeuralNetwork(
            layers: layers,
            inputSpec: TensorSpec(shape: [2, 10], dtype: .float32)
        )

        // Verify structure
        XCTAssertEqual(executable.inputSpecs.count, 5) // input + 2*(weight + bias)

        // Get stats
        let stats = executable.optimizationStats
        XCTAssertGreaterThanOrEqual(stats.originalNodeCount, 0)

        // Check info
        let info = executable.info
        XCTAssertFalse(info.isEmpty)
    }
}
