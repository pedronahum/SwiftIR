// Phase14aRealExecutionTests.swift
// Tests for Phase 14a: Real PJRT Execution
//
// These tests verify that the AD pipeline can execute on real PJRT hardware.
// Note: These tests require the PJRT CPU plugin to be installed.

import XCTest
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase14aRealExecutionTests: XCTestCase {

    // MARK: - Basic PJRT Client Tests

    func testPJRTClientCreation() throws {
        // Test that we can create a PJRT client
        let client = try PJRTClient(backend: .cpu)
        XCTAssertEqual(client.backend, .cpu)
        XCTAssertGreaterThan(client.devices.count, 0)
        XCTAssertNotNil(client.defaultDevice)
    }

    func testPJRTBufferCreation() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = try data.withUnsafeBytes { ptr in
            try client.createBuffer(
                data: ptr.baseAddress!,
                shape: [4],
                elementType: .f32,
                device: device
            )
        }

        XCTAssertEqual(buffer.shape, [4])
        XCTAssertEqual(buffer.elementCount, 4)
        XCTAssertEqual(buffer.elementType, .f32)
    }

    func testPJRTBufferRoundTrip() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        // Create buffer with test data
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let buffer = try input.withUnsafeBytes { ptr in
            try client.createBuffer(
                data: ptr.baseAddress!,
                shape: [5],
                elementType: .f32,
                device: device
            )
        }

        // Read back
        var output = [Float](repeating: 0, count: 5)
        try output.withUnsafeMutableBytes { ptr in
            try buffer.toHost(destination: ptr.baseAddress!)
        }

        // Verify round-trip
        XCTAssertEqual(output, input)
    }

    // MARK: - Simple StableHLO Compilation Tests

    func testSimpleAddCompilation() throws {
        let client = try PJRTClient(backend: .cpu)

        // Simple add operation in StableHLO
        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)
        XCTAssertNotNil(executable)
        XCTAssertEqual(executable.executionCount, 0)
    }

    func testSimpleAddExecution() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        // Simple add operation
        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)

        // Create input buffers
        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [5.0, 6.0, 7.0, 8.0]

        let bufferA = try a.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }
        let bufferB = try b.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }

        // Execute
        let outputs = try executable.execute(arguments: [bufferA, bufferB])
        XCTAssertEqual(outputs.count, 1)

        // Read result
        var result = [Float](repeating: 0, count: 4)
        try result.withUnsafeMutableBytes { ptr in
            try outputs[0].toHost(destination: ptr.baseAddress!)
        }

        // Verify: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
        XCTAssertEqual(result[0], 6.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 8.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 10.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 12.0, accuracy: 0.001)
    }

    func testSimpleMultiplyExecution() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        // Simple multiply operation
        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)

        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [2.0, 3.0, 4.0, 5.0]

        let bufferA = try a.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }
        let bufferB = try b.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }

        let outputs = try executable.execute(arguments: [bufferA, bufferB])
        var result = [Float](repeating: 0, count: 4)
        try result.withUnsafeMutableBytes { ptr in
            try outputs[0].toHost(destination: ptr.baseAddress!)
        }

        // Verify: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
        XCTAssertEqual(result[0], 2.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 6.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 12.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 20.0, accuracy: 0.001)
    }

    // MARK: - StableHLO Operations Tests

    func testExpExecution() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.exponential %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)

        let input: [Float] = [0.0, 1.0, 2.0, -1.0]
        let buffer = try input.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }

        let outputs = try executable.execute(arguments: [buffer])
        var result = [Float](repeating: 0, count: 4)
        try result.withUnsafeMutableBytes { ptr in
            try outputs[0].toHost(destination: ptr.baseAddress!)
        }

        // Verify: exp([0, 1, 2, -1]) ≈ [1, 2.718, 7.389, 0.368]
        XCTAssertEqual(result[0], 1.0, accuracy: 0.01)
        XCTAssertEqual(result[1], 2.718, accuracy: 0.01)
        XCTAssertEqual(result[2], 7.389, accuracy: 0.01)
        XCTAssertEqual(result[3], 0.368, accuracy: 0.01)
    }

    func testNegateExecution() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.negate %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)

        let input: [Float] = [1.0, -2.0, 3.0, -4.0]
        let buffer = try input.withUnsafeBytes { ptr in
            try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
        }

        let outputs = try executable.execute(arguments: [buffer])
        var result = [Float](repeating: 0, count: 4)
        try result.withUnsafeMutableBytes { ptr in
            try outputs[0].toHost(destination: ptr.baseAddress!)
        }

        // Verify: -[1, -2, 3, -4] = [-1, 2, -3, 4]
        XCTAssertEqual(result[0], -1.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(result[2], -3.0, accuracy: 0.001)
        XCTAssertEqual(result[3], 4.0, accuracy: 0.001)
    }

    // MARK: - PJRTBackedRuntime Integration Tests

    func testPJRTBackedRuntimeExecutableCreation() throws {
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

    // MARK: - Repeated Execution Tests

    func testRepeatedExecution() throws {
        let client = try PJRTClient(backend: .cpu)
        guard let device = client.defaultDevice else {
            XCTFail("No default device")
            return
        }

        let module = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlirModule: module)

        // Execute multiple times
        for i in 0..<3 {
            let a: [Float] = [Float(i), Float(i+1), Float(i+2), Float(i+3)]
            let b: [Float] = [1.0, 1.0, 1.0, 1.0]

            let bufferA = try a.withUnsafeBytes { ptr in
                try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
            }
            let bufferB = try b.withUnsafeBytes { ptr in
                try client.createBuffer(data: ptr.baseAddress!, shape: [4], elementType: .f32, device: device)
            }

            let outputs = try executable.execute(arguments: [bufferA, bufferB])
            var result = [Float](repeating: 0, count: 4)
            try result.withUnsafeMutableBytes { ptr in
                try outputs[0].toHost(destination: ptr.baseAddress!)
            }

            // Verify: a + 1
            XCTAssertEqual(result[0], Float(i) + 1.0, accuracy: 0.001)
        }

        XCTAssertEqual(executable.executionCount, 3)
    }

    // MARK: - Error Handling Tests

    func testInvalidModuleCompilation() {
        do {
            let client = try PJRTClient(backend: .cpu)
            _ = try client.compile(mlirModule: "invalid mlir")
            XCTFail("Should have thrown an error")
        } catch {
            // Expected error
            XCTAssertTrue(true)
        }
    }
}
