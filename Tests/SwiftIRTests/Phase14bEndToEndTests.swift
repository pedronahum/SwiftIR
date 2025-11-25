// Phase14bEndToEndTests.swift
// Tests for Phase 14b: End-to-End AD + PJRT Integration
//
// These tests verify the complete pipeline:
// AD tracing → MLIR generation → StableHLO → PJRT execution → numerical results

import XCTest
@testable import SwiftIR
@testable import SwiftIRXLA

final class Phase14bEndToEndTests: XCTestCase {

    // MARK: - Basic End-to-End Tests

    func testEndToEndSimpleAdd() throws {
        // Trace the function using CompilableTracer
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [4], dtype: .float32, name: "%arg0")
        let y = context.input(shape: [4], dtype: .float32, name: "%arg1")
        let result = x + y
        context.output(result)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        // Compile to PJRT executable
        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile(module)
        let function = RealCompiledFunction(executable: executable)

        // Execute with real data
        let inputs: [[Float]] = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
        let outputs = function.run(inputs)

        // Verify numerical results
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(outputs[0].count, 4)
        XCTAssertEqual(outputs[0][0], 11.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 22.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 33.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 44.0, accuracy: 0.001)
    }

    func testEndToEndSimpleMultiply() throws {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [4], dtype: .float32, name: "%arg0")
        let y = context.input(shape: [4], dtype: .float32, name: "%arg1")
        let result = x * y
        context.output(result)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile(module)
        let function = RealCompiledFunction(executable: executable)

        let inputs: [[Float]] = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
        let outputs = function.run(inputs)

        // [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
        XCTAssertEqual(outputs[0][0], 2.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 6.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 12.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 20.0, accuracy: 0.001)
    }

    func testEndToEndSquare() throws {
        // f(x) = x * x
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [4], dtype: .float32, name: "%arg0")
        let result = x * x
        context.output(result)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile(module)
        let function = RealCompiledFunction(executable: executable)

        let inputs: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]
        let outputs = function.run(inputs)

        // [1², 2², 3², 4²] = [1, 4, 9, 16]
        XCTAssertEqual(outputs[0][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 4.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 9.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 16.0, accuracy: 0.001)
    }

    // MARK: - High-Level API End-to-End Tests

    func testCompileForPJRTSquare() throws {
        let function = try compileForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let inputs: [[Float]] = [[2.0, 3.0, 4.0, 5.0]]
        let outputs = function.run(inputs)

        // [4, 9, 16, 25]
        XCTAssertEqual(outputs[0][0], 4.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 9.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 16.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 25.0, accuracy: 0.001)
    }

    func testCompileForPJRTAddMultiply() throws {
        // f(x, y) = (x + y) * x
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            let x = inputs[0]
            let y = inputs[1]
            return (x + y) * x
        }

        let inputs: [[Float]] = [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]
        let outputs = function.run(inputs)

        // [(1+1)*1, (2+1)*2, (3+1)*3, (4+1)*4] = [2, 6, 12, 20]
        XCTAssertEqual(outputs[0][0], 2.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 6.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 12.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 20.0, accuracy: 0.001)
    }

    // MARK: - CompiledFunction to PJRT Conversion

    func testCompiledFunctionToPJRTExecution() throws {
        let compiler = FunctionCompiler()

        let compiled = try compiler.compile(inputShape: [4], dtype: .float32) { x in
            return x * x
        }

        // Convert to PJRT executable
        let pjrtFunc = try compiled.toPJRTExecutable(backend: .cpu)

        let inputs: [[Float]] = [[3.0, 4.0, 5.0, 6.0]]
        let outputs = pjrtFunc.run(inputs)

        // [9, 16, 25, 36]
        XCTAssertEqual(outputs[0][0], 9.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 16.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 25.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 36.0, accuracy: 0.001)
    }

    // MARK: - Complex Expression Tests

    func testEndToEndPolynomial() throws {
        // f(x) = x^2 + 2x + 1 = (x + 1)^2
        let function = try compileForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            let xSquared = x * x
            let twoX = x + x
            return xSquared + twoX
        }

        let inputs: [[Float]] = [[0.0, 1.0, 2.0, 3.0]]
        let outputs = function.run(inputs)

        // [0, 3, 8, 15] (x² + 2x for x = 0,1,2,3)
        XCTAssertEqual(outputs[0][0], 0.0, accuracy: 0.001)  // 0 + 0
        XCTAssertEqual(outputs[0][1], 3.0, accuracy: 0.001)  // 1 + 2
        XCTAssertEqual(outputs[0][2], 8.0, accuracy: 0.001)  // 4 + 4
        XCTAssertEqual(outputs[0][3], 15.0, accuracy: 0.001) // 9 + 6
    }

    func testEndToEndChainedOperations() throws {
        // f(x, y) = x * y + x - y
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            let x = inputs[0]
            let y = inputs[1]
            let xy = x * y
            let sum = xy + x
            return sum - y
        }

        let inputs: [[Float]] = [[2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]]
        let outputs = function.run(inputs)

        // [2*1+2-1, 3*2+3-2, 4*3+4-3, 5*4+5-4] = [3, 7, 13, 21]
        XCTAssertEqual(outputs[0][0], 3.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 7.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 13.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 21.0, accuracy: 0.001)
    }

    // MARK: - Division Tests

    func testEndToEndDivision() throws {
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] / inputs[1]
        }

        let inputs: [[Float]] = [[10.0, 20.0, 30.0, 40.0], [2.0, 4.0, 5.0, 8.0]]
        let outputs = function.run(inputs)

        // [5, 5, 6, 5]
        XCTAssertEqual(outputs[0][0], 5.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 5.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 6.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 5.0, accuracy: 0.001)
    }

    // MARK: - Repeated Execution Tests

    func testRepeatedPJRTExecution() throws {
        let function = try compileForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        // Run multiple times to test caching
        for i in 0..<5 {
            let scale = Float(i + 1)
            let inputs: [[Float]] = [[scale, scale * 2, scale * 3, scale * 4]]
            let outputs = function.run(inputs)

            XCTAssertEqual(outputs[0][0], scale * scale, accuracy: 0.001)
            XCTAssertEqual(outputs[0][1], (scale * 2) * (scale * 2), accuracy: 0.001)
        }
    }

    // MARK: - Larger Tensor Tests

    func testEndToEndLargerTensor() throws {
        let size = 100

        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [size], dtype: .float32),
                TensorSpec(shape: [size], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] + inputs[1]
        }

        let a = (0..<size).map { Float($0) }
        let b = (0..<size).map { Float($0 * 2) }

        let outputs = function.run([a, b])

        XCTAssertEqual(outputs[0].count, size)
        for i in 0..<size {
            XCTAssertEqual(outputs[0][i], Float(i * 3), accuracy: 0.001)
        }
    }

    // MARK: - StableHLO Pipeline Verification

    func testStableHLOPipelineVerification() throws {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let x = context.input(shape: [4], dtype: .float32, name: "%arg0")
        let result = x * x
        context.output(result)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        let runtime = try PJRTBackedRuntime(backend: .cpu)
        let executable = try runtime.compile(module)

        // Verify StableHLO was generated correctly
        XCTAssertTrue(executable.stablehloSource.contains("stablehlo.multiply"))

        // Verify execution works
        let function = RealCompiledFunction(executable: executable)
        let outputs = function.run([[2.0, 3.0, 4.0, 5.0]])
        XCTAssertEqual(outputs[0][0], 4.0, accuracy: 0.001)
    }

    // MARK: - Multi-Operation Pipeline Tests

    func testMultiOperationPipeline() throws {
        // Test a more complex pipeline with multiple operation types
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            let a = inputs[0]
            let b = inputs[1]
            let c = inputs[2]
            // (a + b) * c
            return (a + b) * c
        }

        let inputs: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ]
        let outputs = function.run(inputs)

        // [(1+2)*3, (2+3)*4, (3+4)*5, (4+5)*6] = [9, 20, 35, 54]
        XCTAssertEqual(outputs[0][0], 9.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 20.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 35.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 54.0, accuracy: 0.001)
    }

    // MARK: - Edge Case Tests

    func testZeroValues() throws {
        let function = try compileForPJRT(
            input: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x * x
        }

        let inputs: [[Float]] = [[0.0, 0.0, 0.0, 0.0]]
        let outputs = function.run(inputs)

        for i in 0..<4 {
            XCTAssertEqual(outputs[0][i], 0.0, accuracy: 0.001)
        }
    }

    func testNegativeValues() throws {
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] * inputs[1]
        }

        let inputs: [[Float]] = [[-1.0, -2.0, 3.0, -4.0], [2.0, -3.0, -4.0, -5.0]]
        let outputs = function.run(inputs)

        // [-2, 6, -12, 20]
        XCTAssertEqual(outputs[0][0], -2.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 6.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], -12.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 20.0, accuracy: 0.001)
    }

    func testSmallValues() throws {
        let function = try compileForPJRT(
            inputs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] * inputs[1]
        }

        let inputs: [[Float]] = [[0.001, 0.002, 0.003, 0.004], [0.1, 0.1, 0.1, 0.1]]
        let outputs = function.run(inputs)

        XCTAssertEqual(outputs[0][0], 0.0001, accuracy: 0.00001)
        XCTAssertEqual(outputs[0][1], 0.0002, accuracy: 0.00001)
        XCTAssertEqual(outputs[0][2], 0.0003, accuracy: 0.00001)
        XCTAssertEqual(outputs[0][3], 0.0004, accuracy: 0.00001)
    }

    // MARK: - RealPJRTCompiler Integration

    func testRealPJRTCompilerEndToEnd() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)

        let function = try compiler.compile(
            inputSpec: TensorSpec(shape: [4], dtype: .float32)
        ) { x in
            return x + x
        }

        let outputs = function.run([[5.0, 10.0, 15.0, 20.0]])

        // [10, 20, 30, 40]
        XCTAssertEqual(outputs[0][0], 10.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 20.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 30.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 40.0, accuracy: 0.001)
    }

    func testRealPJRTCompilerMultiInput() throws {
        let compiler = try RealPJRTCompiler(backend: .cpu)

        let function = try compiler.compile(
            inputSpecs: [
                TensorSpec(shape: [4], dtype: .float32),
                TensorSpec(shape: [4], dtype: .float32)
            ]
        ) { inputs in
            return inputs[0] - inputs[1]
        }

        let outputs = function.run([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])

        // [9, 18, 27, 36]
        XCTAssertEqual(outputs[0][0], 9.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][1], 18.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][2], 27.0, accuracy: 0.001)
        XCTAssertEqual(outputs[0][3], 36.0, accuracy: 0.001)
    }
}
