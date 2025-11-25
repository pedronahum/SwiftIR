// Phase9ValidationTests.swift
// Tests for Phase 9: Backend Compilation Infrastructure

import XCTest
@testable import SwiftIR

final class Phase9ValidationTests: XCTestCase {

    // MARK: - MLIR Builder Tests

    func testMLIRBuilderBasicOperation() {
        let builder = MLIRBuilder()

        let result = builder.freshSSA()
        builder.addArgument(name: "%arg0", type: "tensor<10xf32>")

        let op = MLIROperation(
            result: result,
            opName: "add",
            operands: ["%arg0", "%arg0"],
            resultType: "tensor<10xf32>"
        )
        builder.addOperation(op)
        builder.setResults([result])

        let module = builder.build(functionName: "test_add")

        XCTAssertEqual(module.functionName, "test_add")
        XCTAssertEqual(module.operationCount, 1)
        XCTAssertTrue(module.mlirText.contains("func.func @test_add"))
        XCTAssertTrue(module.mlirText.contains("add"))
    }

    func testMLIRBuilderMultipleOperations() {
        let builder = MLIRBuilder()

        builder.addArgument(name: "%x", type: "tensor<5x5xf32>")
        builder.addArgument(name: "%y", type: "tensor<5x5xf32>")

        let v0 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v0,
            opName: "add",
            operands: ["%x", "%y"],
            resultType: "tensor<5x5xf32>"
        ))

        let v1 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v1,
            opName: "multiply",
            operands: [v0, "%x"],
            resultType: "tensor<5x5xf32>"
        ))

        builder.setResults([v1])

        let module = builder.build()

        XCTAssertEqual(module.operationCount, 2)
        XCTAssertEqual(module.arguments.count, 2)
    }

    func testMLIRBuilderReset() {
        let builder = MLIRBuilder()

        builder.addArgument(name: "%x", type: "tensor<10xf32>")
        builder.addOperation(MLIROperation(
            result: "%v0",
            opName: "negate",
            operands: ["%x"],
            resultType: "tensor<10xf32>"
        ))

        builder.reset()

        let module = builder.build()
        XCTAssertEqual(module.operationCount, 0)
        XCTAssertEqual(module.arguments.count, 0)
    }

    func testMLIRBuilderSSAGeneration() {
        let builder = MLIRBuilder()

        let v0 = builder.freshSSA()
        let v1 = builder.freshSSA()
        let v2 = builder.freshSSA()

        XCTAssertEqual(v0, "%v0")
        XCTAssertEqual(v1, "%v1")
        XCTAssertEqual(v2, "%v2")
    }

    // MARK: - MLIR Operation Tests

    func testMLIROperationText() {
        let op = MLIROperation(
            result: "%0",
            opName: "add",
            operands: ["%arg0", "%arg1"],
            resultType: "tensor<10xf32>"
        )

        let text = op.mlirText
        XCTAssertTrue(text.contains("%0 = add"))
        XCTAssertTrue(text.contains("%arg0, %arg1"))
        XCTAssertTrue(text.contains("tensor<10xf32>"))
    }

    func testMLIROperationWithAttributes() {
        let op = MLIROperation(
            result: "%0",
            opName: "transpose",
            operands: ["%arg0"],
            attributes: ["permutation" : "[1, 0]"],
            resultType: "tensor<20x10xf32>"
        )

        let text = op.mlirText
        // Transpose uses dims = [...] format now
        XCTAssertTrue(text.contains("dims"))
        XCTAssertTrue(text.contains("[1, 0]"))
    }

    // MARK: - MLIR Module Tests

    func testMLIRModuleGeneration() {
        let module = MLIRModule(
            functionName: "simple",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "negate",
                    operands: ["%x"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        let text = module.mlirText
        XCTAssertTrue(text.contains("module {"))
        XCTAssertTrue(text.contains("func.func @simple"))
        XCTAssertTrue(text.contains("return %0"))
        XCTAssertTrue(text.contains("}"))
    }

    // MARK: - StableHLO Emitter Tests

    func testStableHLOConversionBasicOps() throws {
        let module = MLIRModule(
            functionName: "test",
            arguments: ["%x: tensor<10xf32>", "%y: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "add",
                    operands: ["%x", "%y"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        let emitter = StableHLOEmitter()
        let stablehlo = try emitter.convert(module)

        XCTAssertEqual(stablehlo.operationCount, 1)
        XCTAssertTrue(stablehlo.mlirText.contains("stablehlo.add"))
    }

    func testStableHLOConversionMultipleOps() throws {
        let module = MLIRModule(
            functionName: "chain",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "exp",
                    operands: ["%x"],
                    resultType: "tensor<10xf32>"
                ),
                MLIROperation(
                    result: "%1",
                    opName: "log",
                    operands: ["%0"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%1"]
        )

        let emitter = StableHLOEmitter()
        let stablehlo = try emitter.convert(module)

        XCTAssertEqual(stablehlo.operationCount, 2)
        XCTAssertTrue(stablehlo.mlirText.contains("stablehlo.exponential"))
        XCTAssertTrue(stablehlo.mlirText.contains("stablehlo.log"))
    }

    func testStableHLOConversionMatmul() throws {
        let module = MLIRModule(
            functionName: "matmul_test",
            arguments: ["%a: tensor<10x20xf32>", "%b: tensor<20x30xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "dot",
                    operands: ["%a", "%b"],
                    resultType: "tensor<10x30xf32>"
                )
            ],
            results: ["%0"]
        )

        let emitter = StableHLOEmitter()
        let stablehlo = try emitter.convert(module)

        XCTAssertTrue(stablehlo.mlirText.contains("stablehlo.dot_general"))
    }

    // MARK: - Compilation Options Tests

    func testDefaultCompilationOptions() {
        let options = CompilationOptions.default

        XCTAssertEqual(options.optimizationLevel, .standard)
        XCTAssertTrue(options.enableFusion)
        XCTAssertTrue(options.enableLayoutOptimization)
        XCTAssertFalse(options.debugMode)
        XCTAssertTrue(options.cacheEnabled)
    }

    func testCustomCompilationOptions() {
        let options = CompilationOptions(
            target: .cuda(deviceId: 0),
            optimizationLevel: .aggressive,
            enableFusion: false,
            enableLayoutOptimization: false,
            debugMode: true,
            cacheEnabled: false
        )

        XCTAssertEqual(options.optimizationLevel, .aggressive)
        XCTAssertFalse(options.enableFusion)
        XCTAssertTrue(options.debugMode)
        XCTAssertFalse(options.cacheEnabled)
    }

    // MARK: - XLA Interface Tests

    func testXLAInterfaceCompilation() throws {
        let module = StableHLOModule(
            functionName: "test",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                StableHLOOp(
                    result: "%0",
                    opName: "stablehlo.negate",
                    operands: ["%x"],
                    attributes: [:],
                    resultType: "tensor<10xf32>",
                    regions: []
                )
            ],
            results: ["%0"]
        )

        let xla = XLAInterface(target: .cpu)
        let executable = try xla.compile(module, options: .default)

        XCTAssertEqual(executable.module.operationCount, 1)
        XCTAssertNotNil(executable.compilationTimestamp)
    }

    func testXLAExecutableInfo() throws {
        let module = StableHLOModule(
            functionName: "info_test",
            arguments: [],
            operations: [],
            results: []
        )

        let xla = XLAInterface(target: .cuda(deviceId: 1))
        let executable = try xla.compile(module, options: CompilationOptions(
            target: .cuda(deviceId: 1),
            optimizationLevel: .aggressive
        ))

        let info = executable.info
        XCTAssertTrue(info.contains("CUDA:1"))
        XCTAssertTrue(info.contains("O3"))
    }

    // MARK: - SwiftIR Compiler Tests

    func testCompilerBasicCompilation() throws {
        let module = MLIRModule(
            functionName: "basic",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "negate",
                    operands: ["%x"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        let compiler = SwiftIRCompiler()
        let compiled = try compiler.compile(module)

        XCTAssertFalse(compiled.mlirSource.isEmpty)
        XCTAssertFalse(compiled.stablehloSource.isEmpty)
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.negate"))
    }

    func testCompilerCaching() throws {
        let module = MLIRModule(
            functionName: "cached",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "exp",
                    operands: ["%x"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        let compiler = SwiftIRCompiler(options: CompilationOptions(cacheEnabled: true))

        // First compilation
        let compiled1 = try compiler.compile(module)

        // Second compilation should use cache
        let compiled2 = try compiler.compile(module)

        // Should return same compiled function
        XCTAssertEqual(compiled1.mlirSource, compiled2.mlirSource)
    }

    func testCompilerCacheClearing() throws {
        let module = MLIRModule(
            functionName: "clear_test",
            arguments: [],
            operations: [],
            results: []
        )

        let compiler = SwiftIRCompiler()
        _ = try compiler.compile(module)

        compiler.clearCache()

        // Should still compile after clearing
        let compiled = try compiler.compile(module)
        XCTAssertNotNil(compiled)
    }

    func testCompilerWithDifferentTargets() throws {
        let module = MLIRModule(
            functionName: "target_test",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "add",
                    operands: ["%x", "%x"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        // CPU target
        let cpuCompiler = SwiftIRCompiler(options: CompilationOptions(target: .cpu))
        let cpuCompiled = try cpuCompiler.compile(module)
        XCTAssertTrue(cpuCompiled.info.contains("CPU"))

        // CUDA target
        let cudaCompiler = SwiftIRCompiler(options: CompilationOptions(target: .cuda(deviceId: 0)))
        let cudaCompiled = try cudaCompiler.compile(module)
        XCTAssertTrue(cudaCompiled.info.contains("CUDA"))
    }

    // MARK: - Compiled Function Tests

    func testCompiledFunctionRun() throws {
        let module = MLIRModule(
            functionName: "run_test",
            arguments: ["%x: tensor<5xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "negate",
                    operands: ["%x"],
                    resultType: "tensor<5xf32>"
                )
            ],
            results: ["%0"]
        )

        let compiler = SwiftIRCompiler()
        let compiled = try compiler.compile(module)

        let input = [[Float(1), 2, 3, 4, 5]]
        let output = compiled.run(input)

        // Output should have same structure
        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].count, 5)
    }

    // MARK: - Convenience Function Tests

    func testConvenienceCompileFunction() throws {
        let module = MLIRModule(
            functionName: "convenience",
            arguments: [],
            operations: [],
            results: []
        )

        let compiled = try compileFunction(module)
        XCTAssertNotNil(compiled)
    }

    func testConvenienceCompileFunctionWithOptions() throws {
        let module = MLIRModule(
            functionName: "with_options",
            arguments: [],
            operations: [],
            results: []
        )

        let options = CompilationOptions(
            target: .metal(deviceId: 0),
            optimizationLevel: .basic
        )

        let compiled = try compileFunction(module, options: options)
        XCTAssertTrue(compiled.info.contains("Metal"))
    }

    // MARK: - Complex Graph Tests

    func testCompileNeuralNetworkLayer() throws {
        // Build a simple neural network layer: y = relu(x @ w + b)
        let builder = MLIRBuilder()

        builder.addArgument(name: "%x", type: "tensor<32x784xf32>")
        builder.addArgument(name: "%w", type: "tensor<784x128xf32>")
        builder.addArgument(name: "%b", type: "tensor<128xf32>")

        // xw = x @ w
        let v0 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v0,
            opName: "dot",
            operands: ["%x", "%w"],
            resultType: "tensor<32x128xf32>"
        ))

        // xw_b = xw + b
        let v1 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v1,
            opName: "add",
            operands: [v0, "%b"],
            resultType: "tensor<32x128xf32>"
        ))

        // y = relu(xw_b)
        let v2 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v2,
            opName: "maximum",
            operands: [v1, "0"],
            resultType: "tensor<32x128xf32>"
        ))

        builder.setResults([v2])

        let module = builder.build(functionName: "neural_layer")

        let compiler = SwiftIRCompiler()
        let compiled = try compiler.compile(module)

        XCTAssertEqual(module.operationCount, 3)
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.dot_general"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.add"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.maximum"))
    }

    func testCompileGradientGraph() throws {
        // Build a backward pass graph
        let builder = MLIRBuilder()

        builder.addArgument(name: "%x", type: "tensor<10x20xf32>")
        builder.addArgument(name: "%w", type: "tensor<20x10xf32>")
        builder.addArgument(name: "%dy", type: "tensor<10x10xf32>")

        // dx = dy @ w^T
        let v0 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v0,
            opName: "transpose",
            operands: ["%w"],
            attributes: ["permutation": "[1, 0]"],
            resultType: "tensor<10x20xf32>"
        ))

        let v1 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v1,
            opName: "dot",
            operands: ["%dy", v0],
            resultType: "tensor<10x20xf32>"
        ))

        // dw = x^T @ dy
        let v2 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v2,
            opName: "transpose",
            operands: ["%x"],
            attributes: ["permutation": "[1, 0]"],
            resultType: "tensor<20x10xf32>"
        ))

        let v3 = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: v3,
            opName: "dot",
            operands: [v2, "%dy"],
            resultType: "tensor<20x10xf32>"
        ))

        builder.setResults([v1, v3])

        let module = builder.build(functionName: "backward_pass")

        let compiler = SwiftIRCompiler()
        let compiled = try compiler.compile(module)

        XCTAssertEqual(module.operationCount, 4)
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.transpose"))
        XCTAssertTrue(compiled.stablehloSource.contains("stablehlo.dot_general"))
    }

    // MARK: - Optimization Level Tests

    func testDifferentOptimizationLevels() throws {
        let module = MLIRModule(
            functionName: "opt_test",
            arguments: ["%x: tensor<10xf32>"],
            operations: [
                MLIROperation(
                    result: "%0",
                    opName: "exp",
                    operands: ["%x"],
                    resultType: "tensor<10xf32>"
                )
            ],
            results: ["%0"]
        )

        for level in [OptimizationLevel.none, .basic, .standard, .aggressive] {
            let options = CompilationOptions(optimizationLevel: level)
            let compiler = SwiftIRCompiler(options: options)
            let compiled = try compiler.compile(module)

            XCTAssertTrue(compiled.info.contains("O\(level.rawValue)"))
        }
    }

    // MARK: - Target Description Tests

    func testCompilationTargetDescriptions() {
        XCTAssertEqual(CompilationTarget.cpu.description, "CPU")
        XCTAssertEqual(CompilationTarget.cuda(deviceId: 0).description, "CUDA:0")
        XCTAssertEqual(CompilationTarget.cuda(deviceId: 3).description, "CUDA:3")
        XCTAssertEqual(CompilationTarget.tpu(deviceId: 1).description, "TPU:1")
        XCTAssertEqual(CompilationTarget.metal(deviceId: 0).description, "Metal:0")
    }
}
