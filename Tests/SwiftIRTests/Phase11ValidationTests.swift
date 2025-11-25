// Phase11ValidationTests.swift
// Tests for Phase 11: C++ Bridge Stubs

import XCTest
@testable import SwiftIR

final class Phase11ValidationTests: XCTestCase {

    // MARK: - MLIR Bridge Tests

    func testMLIRContextCreation() {
        let context = MLIRBridge.createContext()
        XCTAssertNotEqual(context.id, 0)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRRegisterDialect() {
        let context = MLIRBridge.createContext()
        MLIRBridge.registerStableHLODialect(context)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRParseModule() throws {
        let context = MLIRBridge.createContext()

        let mlirText = """
        module {
          func.func @test() {
            return
          }
        }
        """

        let module = try MLIRBridge.parseModule(context, mlirText: mlirText)
        XCTAssertNotEqual(module.id, 0)

        MLIRBridge.destroyModule(module)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRParseModuleFailure() {
        let context = MLIRBridge.createContext()

        // Missing "module" keyword should trigger parse error
        let invalidMLIR = "not a valid MLIR text"

        do {
            _ = try MLIRBridge.parseModule(context, mlirText: invalidMLIR)
            XCTFail("Expected parse error for invalid MLIR")
        } catch let error as MLIRBridgeError {
            // Expected - verify it's a parse error
            if case .parseError = error {
                // Good
            } else {
                XCTFail("Expected parseError, got \(error)")
            }
        } catch {
            XCTFail("Expected MLIRBridgeError, got \(error)")
        }

        MLIRBridge.destroyContext(context)
    }

    func testMLIRModuleToString() throws {
        let context = MLIRBridge.createContext()

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let module = try MLIRBridge.parseModule(context, mlirText: mlirText)
        let str = MLIRBridge.moduleToString(module)

        XCTAssertFalse(str.isEmpty)

        MLIRBridge.destroyModule(module)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRVerifyModule() throws {
        let context = MLIRBridge.createContext()

        let mlirText = """
        module {
          func.func @valid() { return }
        }
        """

        let module = try MLIRBridge.parseModule(context, mlirText: mlirText)

        // Should not throw
        try MLIRBridge.verifyModule(module)

        MLIRBridge.destroyModule(module)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRRunPasses() throws {
        let context = MLIRBridge.createContext()

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let module = try MLIRBridge.parseModule(context, mlirText: mlirText)

        // Should not throw
        try MLIRBridge.runPasses(module, passes: ["canonicalize", "cse"])

        MLIRBridge.destroyModule(module)
        MLIRBridge.destroyContext(context)
    }

    func testMLIRConvertToStableHLO() throws {
        let context = MLIRBridge.createContext()

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let module = try MLIRBridge.parseModule(context, mlirText: mlirText)
        let stablehlo = try MLIRBridge.convertToStableHLO(module)

        XCTAssertNotEqual(stablehlo.id, 0)

        MLIRBridge.destroyModule(stablehlo)
        MLIRBridge.destroyModule(module)
        MLIRBridge.destroyContext(context)
    }

    // MARK: - XLA Bridge Tests

    func testXLACreateCPUClient() throws {
        let client = try XLABridge.createCPUClient()
        XCTAssertNotEqual(client.id, 0)
        XLABridge.destroyClient(client)
    }

    func testXLACreateGPUClient() throws {
        let client = try XLABridge.createGPUClient(deviceId: 0)
        XCTAssertNotEqual(client.id, 0)
        XLABridge.destroyClient(client)
    }

    func testXLACreateTPUClient() throws {
        let client = try XLABridge.createTPUClient(deviceId: 0)
        XCTAssertNotEqual(client.id, 0)
        XLABridge.destroyClient(client)
    }

    func testXLABuildComputation() throws {
        let client = try XLABridge.createCPUClient()

        let hlo = "HloModule test"
        let computation = try XLABridge.buildComputation(client, hloModule: hlo)

        XCTAssertNotEqual(computation.id, 0)

        XLABridge.destroyClient(client)
    }

    func testXLACompile() throws {
        let client = try XLABridge.createCPUClient()

        let computation = try XLABridge.buildComputation(client, hloModule: "HloModule test")
        let executable = try XLABridge.compile(client, computation: computation, options: .default)

        XCTAssertNotEqual(executable.id, 0)

        XLABridge.destroyExecutable(executable)
        XLABridge.destroyClient(client)
    }

    func testXLACompileOptions() {
        let options = XLACompileOptions(
            numReplicas: 2,
            numPartitions: 4,
            optimizationLevel: 3
        )

        XCTAssertEqual(options.numReplicas, 2)
        XCTAssertEqual(options.numPartitions, 4)
        XCTAssertEqual(options.optimizationLevel, 3)
    }

    func testXLADefaultCompileOptions() {
        let options = XLACompileOptions.default

        XCTAssertEqual(options.numReplicas, 1)
        XCTAssertEqual(options.numPartitions, 1)
        XCTAssertEqual(options.optimizationLevel, 2)
    }

    func testXLADeviceCount() throws {
        let client = try XLABridge.createCPUClient()

        let count = XLABridge.deviceCount(client)
        XCTAssertGreaterThanOrEqual(count, 1)

        XLABridge.destroyClient(client)
    }

    func testXLADeviceName() throws {
        let client = try XLABridge.createCPUClient()

        let name = XLABridge.deviceName(client, deviceId: 0)
        XCTAssertFalse(name.isEmpty)

        XLABridge.destroyClient(client)
    }

    // MARK: - PJRT Bridge Tests

    func testPJRTCreateClient() throws {
        // Note: In real test, would need actual plugin path
        let client = try PJRTBridge.createClient(pluginPath: "/tmp/fake")
        XCTAssertNotEqual(client.id, 0)
        PJRTBridge.destroyClient(client)
    }

    func testPJRTCompile() throws {
        let client = try PJRTBridge.createClient(pluginPath: "/tmp/fake")

        let mlirModule = "module {}"
        let executable = try PJRTBridge.compile(client, mlirModule: mlirModule)

        XCTAssertNotEqual(executable.id, 0)

        PJRTBridge.destroyExecutable(executable)
        PJRTBridge.destroyClient(client)
    }

    func testPJRTExecute() throws {
        let client = try PJRTBridge.createClient(pluginPath: "/tmp/fake")
        let executable = try PJRTBridge.compile(client, mlirModule: "module {}")

        let inputBuffer = try PJRTBridge.bufferFromHost(
            client,
            data: UnsafeRawPointer(bitPattern: 1)!,
            size: 40,
            shape: [10],
            dtype: .float32
        )

        let outputs = try PJRTBridge.execute(executable, inputs: [inputBuffer])
        XCTAssertEqual(outputs.count, 1)

        PJRTBridge.destroyBuffer(inputBuffer)
        for output in outputs {
            PJRTBridge.destroyBuffer(output)
        }
        PJRTBridge.destroyExecutable(executable)
        PJRTBridge.destroyClient(client)
    }

    // MARK: - SwiftIR Runtime Tests

    func testRuntimeCreationCPU() throws {
        let runtime = try SwiftIRRuntime(backend: .cpu)
        XCTAssertTrue(runtime.info.contains("CPU"))
    }

    func testRuntimeCreationCUDA() throws {
        let runtime = try SwiftIRRuntime(backend: .cuda(deviceId: 0))
        XCTAssertTrue(runtime.info.contains("CUDA"))
    }

    func testRuntimeCreationTPU() throws {
        let runtime = try SwiftIRRuntime(backend: .tpu(deviceId: 0))
        XCTAssertTrue(runtime.info.contains("TPU"))
    }

    func testRuntimeCompile() throws {
        let runtime = try SwiftIRRuntime(backend: .cpu)

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let executable = try runtime.compile(mlirText)
        XCTAssertNotNil(executable)
    }

    func testRuntimeExecute() throws {
        let runtime = try SwiftIRRuntime(backend: .cpu)

        let mlirText = """
        module {
          func.func @test() { return }
        }
        """

        let executable = try runtime.compile(mlirText)
        let result = try executable.execute([[1.0, 2.0, 3.0]])

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].count, 3)
    }

    // MARK: - Device Query Tests

    func testAvailableBackends() {
        let backends = DeviceQuery.availableBackends()

        // CPU should always be available
        XCTAssertTrue(backends.contains("cpu"))
    }

    func testCUDADeviceCount() {
        let count = DeviceQuery.cudaDeviceCount()
        XCTAssertGreaterThanOrEqual(count, 0)
    }

    #if os(macOS) || os(iOS)
    func testMetalAvailable() {
        XCTAssertTrue(DeviceQuery.metalAvailable())
    }
    #endif

    // MARK: - Error Tests

    func testMLIRBridgeErrorDescription() {
        let error = MLIRBridgeError.parseError("invalid syntax")
        XCTAssertTrue(error.description.contains("parse error"))
        XCTAssertTrue(error.description.contains("invalid syntax"))
    }

    func testXLABridgeErrorDescription() {
        let error = XLABridgeError.compilationFailed("out of memory")
        XCTAssertTrue(error.description.contains("compilation failed"))
        XCTAssertTrue(error.description.contains("out of memory"))
    }

    func testPJRTBridgeErrorDescription() {
        let error = PJRTBridgeError.pluginLoadFailed("not found")
        XCTAssertTrue(error.description.contains("plugin load failed"))
        XCTAssertTrue(error.description.contains("not found"))
    }

    func testRuntimeErrorDescription() {
        let error = RuntimeError.notInitialized
        XCTAssertTrue(error.description.contains("not initialized"))
    }

    // MARK: - Integration Tests

    func testFullPipelineWithRuntime() throws {
        // Create runtime
        let runtime = try SwiftIRRuntime(backend: .cpu)

        // Build MLIR module
        let builder = MLIRBuilder()
        builder.addArgument(name: "%x", type: "tensor<10xf32>")

        let result = builder.freshSSA()
        builder.addOperation(MLIROperation(
            result: result,
            opName: "negate",
            operands: ["%x"],
            resultType: "tensor<10xf32>"
        ))
        builder.setResults([result])

        let module = builder.build(functionName: "negate_test")

        // Compile
        let executable = try runtime.compile(module.mlirText)

        // Execute
        let input: [[Float]] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        let output = try executable.execute(input)

        XCTAssertEqual(output.count, 1)
        XCTAssertEqual(output[0].count, 10)
    }

    func testMultipleContexts() {
        let context1 = MLIRBridge.createContext()
        let context2 = MLIRBridge.createContext()

        XCTAssertNotEqual(context1.id, context2.id)

        MLIRBridge.destroyContext(context1)
        MLIRBridge.destroyContext(context2)
    }

    func testMultipleClients() throws {
        let client1 = try XLABridge.createCPUClient()
        let client2 = try XLABridge.createCPUClient()

        XCTAssertNotEqual(client1.id, client2.id)

        XLABridge.destroyClient(client1)
        XLABridge.destroyClient(client2)
    }

    // MARK: - Handle Type Tests

    func testMLIRHandleTypes() {
        let context = MLIRBridge.ContextHandle(id: 1)
        let module = MLIRBridge.ModuleHandle(id: 2)
        let operation = MLIRBridge.OperationHandle(id: 3)

        XCTAssertEqual(context.id, 1)
        XCTAssertEqual(module.id, 2)
        XCTAssertEqual(operation.id, 3)
    }

    func testXLAHandleTypes() {
        let client = XLABridge.ClientHandle(id: 1)
        let computation = XLABridge.ComputationHandle(id: 2)
        let executable = XLABridge.ExecutableHandle(id: 3)
        let buffer = XLABridge.BufferHandle(id: 4)

        XCTAssertEqual(client.id, 1)
        XCTAssertEqual(computation.id, 2)
        XCTAssertEqual(executable.id, 3)
        XCTAssertEqual(buffer.id, 4)
    }

    func testPJRTHandleTypes() {
        let client = PJRTBridge.ClientHandle(id: 1)
        let executable = PJRTBridge.LoadedExecutableHandle(id: 2)
        let buffer = PJRTBridge.BufferHandle(id: 3)

        XCTAssertEqual(client.id, 1)
        XCTAssertEqual(executable.id, 2)
        XCTAssertEqual(buffer.id, 3)
    }
}
