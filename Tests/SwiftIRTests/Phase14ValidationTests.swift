// Phase14ValidationTests.swift
// Tests for Phase 14: Connect AD Pipeline to PJRT Execution

import XCTest
@testable import SwiftIR

final class Phase14ValidationTests: XCTestCase {

    // MARK: - PJRTBackedRuntime Tests

    func testPJRTBackedRuntimeCreationCPU() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        XCTAssertTrue(runtime.info.contains("CPU"))
    }

    func testPJRTBackedRuntimePluginPath() {
        let backend = PJRTBackedRuntime.Backend.cpu
        let path = backend.pluginPath
        XCTAssertTrue(path.contains("pjrt_c_api_cpu_plugin"))
    }

    func testPJRTBackedRuntimeGPUPath() {
        let backend = PJRTBackedRuntime.Backend.gpu(deviceId: 0)
        let path = backend.pluginPath
        XCTAssertTrue(path.contains("gpu"))
    }

    func testPJRTBackedRuntimeTPUPath() {
        let backend = PJRTBackedRuntime.Backend.tpu(deviceId: 0)
        let path = backend.pluginPath
        XCTAssertTrue(path.contains("tpu"))
    }

    // MARK: - PJRTBackedExecutable Tests

    func testPJRTBackedExecutableCreation() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let executable = try runtime.compile(mlirText)
        XCTAssertFalse(executable.mlirSource.isEmpty)
        XCTAssertFalse(executable.stablehloSource.isEmpty)
    }

    func testPJRTBackedExecutableStableHLOConversion() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)

        let mlirText = """
        module {
          %0 = "add"(%arg0, %arg1) : () -> ()
        }
        """

        let executable = try runtime.compile(mlirText)
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.add"))
    }

    func testPJRTBackedExecutableInfo() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("module {}")

        let info = executable.info
        XCTAssertTrue(info.contains("PJRT"))
        XCTAssertTrue(info.contains("MLIR"))
        XCTAssertTrue(info.contains("StableHLO"))
    }

    // MARK: - RealCompiledFunction Tests

    func testRealCompiledFunctionCreation() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("module {}")
        let function = RealCompiledFunction(executable: executable)

        XCTAssertFalse(function.mlirSource.isEmpty)
        XCTAssertFalse(function.stablehloSource.isEmpty)
    }

    func testRealCompiledFunctionRun() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("module {}")
        let function = RealCompiledFunction(executable: executable)

        let inputs: [[Float]] = [[1.0, 2.0, 3.0]]
        let outputs = function.run(inputs)

        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(outputs[0], [1.0, 2.0, 3.0])
    }

    // MARK: - RealPJRTCompiler Tests

    func testRealPJRTCompilerCreation() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)
        XCTAssertNotNil(compiler)
    }

    func testRealPJRTCompilerCompileModule() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)

        let builder = MLIRBuilder()
        builder.addArgument(name: "%x", type: "tensor<10xf32>")
        let result = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: result,
            opName: "add",
            operands: ["%x", "%x"],
            resultType: "tensor<10xf32>"
        ))
        builder.setResults([result])

        let module = builder.build(functionName: "test")
        let function = try compiler.compile(module)

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.add"))
    }

    func testRealPJRTCompilerCompileFromTracer() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)

        let function = try compiler.compile(
            inputSpec: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return x + x
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.add"))
    }

    func testRealPJRTCompilerCompileMultiInput() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)

        let function = try compiler.compile(
            inputSpecs: [
                TensorSpec(shape: [10], dtype: .float32),
                TensorSpec(shape: [10], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] + inputs[1]
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.add"))
    }

    // MARK: - High-Level API Tests

    func testCompileForPJRTSingleInput() throws {
        let function = try compileForPJRT(
            input: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            return x * x
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.multiply"))
    }

    func testCompileForPJRTMultiInput() throws {
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [10, 20], dtype: .float32),
                TensorSpec(shape: [20, 30], dtype: .float32)
            ]
        ) { inputs in
            return matmul(inputs[0], inputs[1])
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.dot"))
    }

    // MARK: - Integration with Existing Types Tests

    func testCompiledFunctionToPJRTExecutable() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [10], dtype: .float32) { x in
            return x + x
        }

        let pjrtFunc = try compiled.toPJRTExecutable(backend: .cpu)
        XCTAssertTrue(pjrtFunc.stablehloSource.contains("stablehlo.add"))
    }

    func testGradientCompiledFunctionToPJRTExecutable() throws {
        let compiler = GradientCompiler()

        let gradFunc = try compiler.compileWithGradients(
            inputShape: [10],
            dtype: .float32
        ) { x in
            return x * x
        }

        let pjrtFunc = try gradFunc.toPJRTExecutable(backend: .cpu)
        XCTAssertTrue(pjrtFunc.stablehloSource.contains("stablehlo"))
    }

    // MARK: - Operation Conversion Tests

    func testStableHLOConversionAdd() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"add\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.add"))
    }

    func testStableHLOConversionSubtract() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"subtract\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.subtract"))
    }

    func testStableHLOConversionMultiply() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"multiply\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.multiply"))
    }

    func testStableHLOConversionDivide() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"divide\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.divide"))
    }

    func testStableHLOConversionDot() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"dot\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.dot"))
    }

    func testStableHLOConversionExp() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"exp\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.exponential"))
    }

    func testStableHLOConversionLog() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"log\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.log"))
    }

    func testStableHLOConversionTranspose() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"transpose\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.transpose"))
    }

    func testStableHLOConversionMaximum() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"maximum\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.maximum"))
    }

    func testStableHLOConversionNegate() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile("\"negate\"")
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.negate"))
    }

    // MARK: - Error Tests

    func testPJRTExecutionErrorDescription() {
        let error = PJRTExecutionError.notInitialized
        XCTAssertTrue(error.description.contains("not initialized"))
    }

    func testPJRTExecutionErrorPluginNotFound() {
        let error = PJRTExecutionError.pluginNotFound("/fake/path")
        XCTAssertTrue(error.description.contains("/fake/path"))
    }

    func testPJRTExecutionErrorCompilationFailed() {
        let error = PJRTExecutionError.compilationFailed("test error")
        XCTAssertTrue(error.description.contains("test error"))
    }

    func testPJRTExecutionErrorExecutionFailed() {
        let error = PJRTExecutionError.executionFailed("test error")
        XCTAssertTrue(error.description.contains("test error"))
    }

    // MARK: - Neural Network Tests

    func testNeuralNetworkLayerCompilation() throws {
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [32, 784], dtype: .float32),
                TensorSpec(shape: [784, 128], dtype: .float32),
                TensorSpec(shape: [128], dtype: .float32)
            ]
        ) { inputs in
            let x = inputs[0]
            let w = inputs[1]
            let b = inputs[2]

            let xw = matmul(x, w)
            let xwb = xw + b
            return relu(xwb)
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.dot"))
        XCTAssertTrue(function.stablehloSource.contains("stablehlo.add"))
        XCTAssertTrue(function.stablehloSource.contains("stablehlo.maximum"))
    }

    // MARK: - Complex Graph Tests

    func testComplexGraphCompilation() throws {
        let function = try compileForPJRT(
            input: TensorSpec(shape: [10], dtype: .float32)
        ) { x in
            let squared = x * x
            let expX = exp(squared)
            return expX + x
        }

        XCTAssertTrue(function.stablehloSource.contains("stablehlo.multiply"))
        XCTAssertTrue(function.stablehloSource.contains("stablehlo.exponential"))
        XCTAssertTrue(function.stablehloSource.contains("stablehlo.add"))
    }

    // MARK: - Backend Info Tests

    func testBackendInfoCPU() throws {
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        XCTAssertTrue(runtime.info.contains("CPU"))
    }

    func testBackendInfoGPU() {
        let backend = PJRTBackedRuntime.Backend.gpu(deviceId: 0)
        // Just test the path, actual GPU init might fail
        XCTAssertTrue(backend.pluginPath.contains("gpu"))
    }

    func testBackendInfoTPU() {
        let backend = PJRTBackedRuntime.Backend.tpu(deviceId: 1)
        XCTAssertTrue(backend.pluginPath.contains("tpu"))
    }
}
