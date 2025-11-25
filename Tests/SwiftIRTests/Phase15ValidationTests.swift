// Phase15ValidationTests.swift
// Tests for Phase 15: AD Completeness - Additional Operations, Loss Functions, and Optimizers

import XCTest
@testable import SwiftIR

final class Phase15ValidationTests: XCTestCase {

    // MARK: - Differentiable Operations Tests

    func testDiffLog() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffLog(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("log"))
    }

    func testDiffSqrt() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSqrt(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("sqrt"))
    }

    func testDiffNegate() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffNegate(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("negate"))
    }

    func testDiffSigmoid() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSigmoid(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("logistic"))
    }

    func testDiffTanh() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffTanh(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("tanh"))
    }

    func testDiffAbs() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffAbs(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("abs"))
    }

    func testDiffSum() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSum(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("reduce_sum"))
    }

    func testDiffMean() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffMean(x)
        }

        // Mean uses sum and divide
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("reduce_sum"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("divide"))
    }

    func testDiffPow() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [10], dtype: .float32), (shape: [10], dtype: .float32)]
        ) { x, n in
            return diffPow(x, n)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("power"))
    }

    func testDiffSquare() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSquare(x)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testCreateConstant() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            let two = createConstant(2.0, shape: [10], dtype: .float32)
            return x * two
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("constant"))
    }

    // MARK: - Loss Function Tests

    func testMSELoss() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [10], dtype: .float32), (shape: [10], dtype: .float32)]
        ) { predictions, targets in
            return diffMSELoss(predictions, targets)
        }

        // MSE uses subtract, multiply, reduce_sum
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("subtract"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testBCELoss() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [10], dtype: .float32), (shape: [10], dtype: .float32)]
        ) { predictions, targets in
            return diffBCELoss(predictions, targets)
        }

        // BCE uses log
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("log"))
    }

    func testSoftmaxCrossEntropy() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [10], dtype: .float32), (shape: [10], dtype: .float32)]
        ) { logits, targets in
            return diffSoftmaxCrossEntropy(logits, targets)
        }

        // Softmax CE uses exp, log
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("exp"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("log"))
    }

    // MARK: - Optimizer Tests

    func testSGDOptimizer() {
        var optimizer = SGDOptimizer(learningRate: 0.1)
        var parameters: [Float] = [1.0, 2.0, 3.0]
        let gradients: [Float] = [0.1, 0.2, 0.3]

        optimizer.update(parameters: &parameters, gradients: gradients)

        // p -= lr * grad
        XCTAssertEqual(parameters[0], 0.99, accuracy: 0.001)
        XCTAssertEqual(parameters[1], 1.98, accuracy: 0.001)
        XCTAssertEqual(parameters[2], 2.97, accuracy: 0.001)
    }

    func testSGDOptimizerWithMomentum() {
        var optimizer = SGDOptimizer(learningRate: 0.1, momentum: 0.9)
        var parameters: [Float] = [1.0, 2.0, 3.0]
        let gradients: [Float] = [0.1, 0.2, 0.3]

        // First update
        optimizer.update(parameters: &parameters, gradients: gradients)

        // Second update should include momentum
        optimizer.update(parameters: &parameters, gradients: gradients)

        // Parameters should have changed more due to momentum
        XCTAssertLessThan(parameters[0], 0.98)
    }

    func testAdamOptimizer() {
        var optimizer = AdamOptimizer(learningRate: 0.1)
        var parameters: [Float] = [1.0, 2.0, 3.0]
        let gradients: [Float] = [0.1, 0.2, 0.3]

        optimizer.update(parameters: &parameters, gradients: gradients)

        // Parameters should have changed
        XCTAssertNotEqual(parameters[0], 1.0)
        XCTAssertNotEqual(parameters[1], 2.0)
        XCTAssertNotEqual(parameters[2], 3.0)
    }

    func testAdamOptimizerMultipleSteps() {
        var optimizer = AdamOptimizer(learningRate: 0.1)
        var parameters: [Float] = [1.0, 2.0, 3.0]
        let gradients: [Float] = [0.1, 0.2, 0.3]

        for _ in 0..<10 {
            optimizer.update(parameters: &parameters, gradients: gradients)
        }

        // Parameters should have decreased with positive gradients
        XCTAssertLessThan(parameters[0], 1.0)
        XCTAssertLessThan(parameters[1], 2.0)
        XCTAssertLessThan(parameters[2], 3.0)
    }

    func testRMSpropOptimizer() {
        var optimizer = RMSpropOptimizer(learningRate: 0.1)
        var parameters: [Float] = [1.0, 2.0, 3.0]
        let gradients: [Float] = [0.1, 0.2, 0.3]

        optimizer.update(parameters: &parameters, gradients: gradients)

        // Parameters should have changed
        XCTAssertNotEqual(parameters[0], 1.0)
        XCTAssertNotEqual(parameters[1], 2.0)
        XCTAssertNotEqual(parameters[2], 3.0)
    }

    // MARK: - Training Config Tests

    func testTrainingConfigDefaults() {
        let config = TrainingConfig()

        XCTAssertEqual(config.epochs, 100)
        XCTAssertEqual(config.batchSize, 32)
        XCTAssertEqual(config.learningRate, 0.01)
        XCTAssertEqual(config.printEvery, 10)
    }

    func testTrainingConfigCustom() {
        let config = TrainingConfig(
            epochs: 50,
            batchSize: 16,
            learningRate: 0.001,
            printEvery: 5
        )

        XCTAssertEqual(config.epochs, 50)
        XCTAssertEqual(config.batchSize, 16)
        XCTAssertEqual(config.learningRate, 0.001)
        XCTAssertEqual(config.printEvery, 5)
    }

    func testTrainingResultInfo() {
        let result = TrainingResult(
            finalLoss: 0.001,
            losses: [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001],
            epochs: 100
        )

        XCTAssertTrue(result.info.contains("100"))
        XCTAssertTrue(result.info.contains("0.001"))
    }

    // MARK: - Higher-Order AD Tests

    func testJacobianCompilation() throws {
        let gradFunc = try jacobian(
            of: { x in x * x },
            inputShape: [5],
            outputShape: [5]
        )

        XCTAssertNotNil(gradFunc)
        XCTAssertEqual(gradFunc.inputShape, [5])
    }

    func testHessianCompilation() throws {
        let gradFunc = try hessian(
            of: { x in diffSum(x * x) },
            inputShape: [5]
        )

        XCTAssertNotNil(gradFunc)
        XCTAssertEqual(gradFunc.inputShape, [5])
    }

    // MARK: - Complex Expression Tests

    func testSigmoidBackward() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSigmoid(x)
        }

        // Sigmoid backward uses multiply and constant
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("logistic"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testTanhBackward() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffTanh(x)
        }

        // Tanh backward uses multiply
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("tanh"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testLogSqrtComposition() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffLog(diffSqrt(x))
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("sqrt"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("log"))
    }

    func testComplexLossFunction() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [10], dtype: .float32), (shape: [10], dtype: .float32)]
        ) { x, y in
            // L2 loss with sigmoid activation
            let pred = diffSigmoid(x)
            return diffMSELoss(pred, y)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("logistic"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("subtract"))
    }

    // MARK: - Neural Network Building Blocks

    func testDenseLayerWithActivation() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [
                (shape: [32, 10], dtype: .float32),  // input
                (shape: [10, 20], dtype: .float32),  // weights
            ]
        ) { x, w in
            let linear = diffMatmul(x, w)
            return diffRelu(linear)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("dot"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("maximum"))
    }

    func testTwoLayerNetwork() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [
                (shape: [32, 10], dtype: .float32),  // input
                (shape: [10, 20], dtype: .float32),  // w1
            ]
        ) { x, w1 in
            let h1 = diffRelu(diffMatmul(x, w1))
            return diffSum(h1)  // Simple aggregation
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("dot"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("maximum"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("reduce_sum"))
    }

    func testSigmoidActivatedNetwork() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [32, 10],
            dtype: .float32
        ) { x in
            let activated = diffSigmoid(x)
            return diffMean(activated)
        }

        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("logistic"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("reduce_sum"))
    }

    // MARK: - Gradient Correctness Tests

    func testSigmoidGradientStructure() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSigmoid(x)
        }

        // Sigmoid gradient: y * (1 - y) - should have constant and multiply
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("constant"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testTanhGradientStructure() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffTanh(x)
        }

        // Tanh gradient: 1 - y^2 - should have constant and multiply
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("constant"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("multiply"))
    }

    func testSqrtGradientStructure() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSqrt(x)
        }

        // Sqrt gradient: 1/(2*sqrt(x)) - should have constant and divide
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("constant"))
        XCTAssertTrue(gradFunc.compiled.mlirSource.contains("divide"))
    }

    // MARK: - CompiledTrainingStep Tests

    func testCompiledTrainingStepCreation() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return diffSum(x * x)
        }

        let trainingStep = CompiledTrainingStep(
            gradientFunc: gradFunc,
            learningRate: 0.01
        )

        XCTAssertEqual(trainingStep.learningRate, 0.01)
    }

    // MARK: - Integration Tests

    func testFullPipelineWithSigmoid() throws {
        // Test full pipeline: trace -> compile -> gradient info
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [5],
            dtype: .float32
        ) { x in
            let activated = diffSigmoid(x)
            return diffSum(activated)
        }

        XCTAssertEqual(gradFunc.inputShape, [5])
        XCTAssertTrue(gradFunc.info.contains("Gradient"))
    }

    func testFullPipelineWithTanh() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputShape: [5],
            dtype: .float32
        ) { x in
            let activated = diffTanh(x)
            return diffMean(activated)
        }

        XCTAssertEqual(gradFunc.inputShape, [5])
    }

    func testFullPipelineWithLoss() throws {
        let compiler = GradientCompiler()
        let gradFunc = try compiler.compileWithGradients(
            inputSpecs: [(shape: [5], dtype: .float32), (shape: [5], dtype: .float32)]
        ) { pred, target in
            return diffMSELoss(pred, target)
        }

        XCTAssertEqual(gradFunc.inputShape, [5])
    }
}
