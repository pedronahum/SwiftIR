// TracerIntegration.swift
// Phase 10: Integration between Tracers and Backend Compilation
//
// This connects the symbolic tracing system with the compilation pipeline,
// implementing the "Trojan Horse" mechanism where Tracers build MLIR graphs.

import Foundation

// MARK: - Compilable Tracer

/// A Tracer that integrates with the compilation pipeline
public struct CompilableTracer: AdditiveArithmetic, Sendable {
    /// The MLIR value this tracer represents
    public let irValue: String

    /// Shape of the tensor
    public let shape: [Int]

    /// Data type
    public let dtype: DType

    /// The builder accumulating operations
    /// Note: This is intentionally mutable for tracing context management
    nonisolated(unsafe) public static var currentBuilder: MLIRBuilder?

    public init(irValue: String, shape: [Int], dtype: DType = .float32) {
        self.irValue = irValue
        self.shape = shape
        self.dtype = dtype
    }

    // MARK: - AdditiveArithmetic

    public static var zero: CompilableTracer {
        CompilableTracer(irValue: "zeros", shape: [], dtype: .float32)
    }

    public static func + (lhs: CompilableTracer, rhs: CompilableTracer) -> CompilableTracer {
        guard let builder = currentBuilder else {
            fatalError("No active MLIRBuilder. Call within a tracing context.")
        }

        let result = builder.freshSSA()
        let resultType = "tensor<\(lhs.shape.map(String.init).joined(separator: "x"))x\(lhs.dtype.rawValue)>"

        builder.addOperation(MLIROperation(
            result: result,
            opName: "add",
            operands: [lhs.irValue, rhs.irValue],
            resultType: resultType
        ))

        return CompilableTracer(irValue: result, shape: lhs.shape, dtype: lhs.dtype)
    }

    public static func - (lhs: CompilableTracer, rhs: CompilableTracer) -> CompilableTracer {
        guard let builder = currentBuilder else {
            fatalError("No active MLIRBuilder. Call within a tracing context.")
        }

        let result = builder.freshSSA()
        let resultType = "tensor<\(lhs.shape.map(String.init).joined(separator: "x"))x\(lhs.dtype.rawValue)>"

        builder.addOperation(MLIROperation(
            result: result,
            opName: "subtract",
            operands: [lhs.irValue, rhs.irValue],
            resultType: resultType
        ))

        return CompilableTracer(irValue: result, shape: lhs.shape, dtype: lhs.dtype)
    }
}

// MARK: - Tracer Operations

extension CompilableTracer {
    /// Element-wise multiplication
    public static func * (lhs: CompilableTracer, rhs: CompilableTracer) -> CompilableTracer {
        guard let builder = currentBuilder else {
            fatalError("No active MLIRBuilder. Call within a tracing context.")
        }

        let result = builder.freshSSA()
        let resultType = "tensor<\(lhs.shape.map(String.init).joined(separator: "x"))x\(lhs.dtype.rawValue)>"

        builder.addOperation(MLIROperation(
            result: result,
            opName: "multiply",
            operands: [lhs.irValue, rhs.irValue],
            resultType: resultType
        ))

        return CompilableTracer(irValue: result, shape: lhs.shape, dtype: lhs.dtype)
    }

    /// Element-wise division
    public static func / (lhs: CompilableTracer, rhs: CompilableTracer) -> CompilableTracer {
        guard let builder = currentBuilder else {
            fatalError("No active MLIRBuilder. Call within a tracing context.")
        }

        let result = builder.freshSSA()
        let resultType = "tensor<\(lhs.shape.map(String.init).joined(separator: "x"))x\(lhs.dtype.rawValue)>"

        builder.addOperation(MLIROperation(
            result: result,
            opName: "divide",
            operands: [lhs.irValue, rhs.irValue],
            resultType: resultType
        ))

        return CompilableTracer(irValue: result, shape: lhs.shape, dtype: lhs.dtype)
    }

    /// Negation
    public static prefix func - (operand: CompilableTracer) -> CompilableTracer {
        guard let builder = currentBuilder else {
            fatalError("No active MLIRBuilder. Call within a tracing context.")
        }

        let result = builder.freshSSA()
        let resultType = "tensor<\(operand.shape.map(String.init).joined(separator: "x"))x\(operand.dtype.rawValue)>"

        builder.addOperation(MLIROperation(
            result: result,
            opName: "negate",
            operands: [operand.irValue],
            resultType: resultType
        ))

        return CompilableTracer(irValue: result, shape: operand.shape, dtype: operand.dtype)
    }
}

// MARK: - Math Operations

/// Matrix multiplication
public func matmul(_ a: CompilableTracer, _ b: CompilableTracer) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    // Compute output shape for matmul
    let m = a.shape.count >= 2 ? a.shape[a.shape.count - 2] : 1
    let n = b.shape.count >= 2 ? b.shape[b.shape.count - 1] : 1
    let outputShape = [m, n]

    let result = builder.freshSSA()
    let resultType = "tensor<\(outputShape.map(String.init).joined(separator: "x"))x\(a.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "dot",
        operands: [a.irValue, b.irValue],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: outputShape, dtype: a.dtype)
}

/// Transpose
public func transpose(_ a: CompilableTracer) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    let outputShape = a.shape.reversed().map { $0 }
    let result = builder.freshSSA()
    let resultType = "tensor<\(outputShape.map(String.init).joined(separator: "x"))x\(a.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "transpose",
        operands: [a.irValue],
        attributes: ["permutation": "[\(Array(0..<a.shape.count).reversed().map(String.init).joined(separator: ", "))]"],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: outputShape, dtype: a.dtype)
}

/// ReLU activation
public func relu(_ x: CompilableTracer) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    let result = builder.freshSSA()
    let resultType = "tensor<\(x.shape.map(String.init).joined(separator: "x"))x\(x.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "maximum",
        operands: [x.irValue, "0"],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: x.shape, dtype: x.dtype)
}

/// Exponential
public func exp(_ x: CompilableTracer) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    let result = builder.freshSSA()
    let resultType = "tensor<\(x.shape.map(String.init).joined(separator: "x"))x\(x.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "exp",
        operands: [x.irValue],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: x.shape, dtype: x.dtype)
}

/// Natural logarithm
public func log(_ x: CompilableTracer) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    let result = builder.freshSSA()
    let resultType = "tensor<\(x.shape.map(String.init).joined(separator: "x"))x\(x.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "log",
        operands: [x.irValue],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: x.shape, dtype: x.dtype)
}

/// Sum reduction
public func sum(_ x: CompilableTracer, axes: [Int]? = nil) -> CompilableTracer {
    guard let builder = CompilableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder. Call within a tracing context.")
    }

    let reduceAxes = axes ?? Array(0..<x.shape.count)
    var outputShape = x.shape
    for axis in reduceAxes.sorted().reversed() {
        outputShape.remove(at: axis)
    }
    if outputShape.isEmpty {
        outputShape = [1]
    }

    let result = builder.freshSSA()
    let resultType = "tensor<\(outputShape.map(String.init).joined(separator: "x"))x\(x.dtype.rawValue)>"

    builder.addOperation(MLIROperation(
        result: result,
        opName: "reduce_sum",
        operands: [x.irValue],
        attributes: ["dimensions": "[\(reduceAxes.map(String.init).joined(separator: ", "))]"],
        resultType: resultType
    ))

    return CompilableTracer(irValue: result, shape: outputShape, dtype: x.dtype)
}

// MARK: - Tracing Context

/// Context for tracing operations
public class TracingContext {
    public let builder: MLIRBuilder
    private var argumentCount: Int = 0

    public init() {
        self.builder = MLIRBuilder()
    }

    /// Create a symbolic input
    public func input(shape: [Int], dtype: DType = .float32, name: String? = nil) -> CompilableTracer {
        let argName = name ?? "%arg\(argumentCount)"
        argumentCount += 1

        let typeStr = "tensor<\(shape.map(String.init).joined(separator: "x"))x\(dtype.rawValue)>"
        builder.addArgument(name: argName, type: typeStr)

        return CompilableTracer(irValue: argName, shape: shape, dtype: dtype)
    }

    /// Set the output of the traced function
    public func output(_ tracers: CompilableTracer...) {
        builder.setResults(tracers.map { $0.irValue })
    }

    /// Build the MLIR module
    public func buildModule(name: String = "main") -> MLIRModule {
        builder.build(functionName: name)
    }
}

// MARK: - Function Compilation

/// Compiles a Swift function to an executable
public class FunctionCompiler {
    private let compilerOptions: CompilationOptions

    public init(options: CompilationOptions = .default) {
        self.compilerOptions = options
    }

    /// Compile a function that takes a single tracer input
    public func compile(
        inputShape: [Int],
        dtype: DType = .float32,
        _ function: (CompilableTracer) -> CompilableTracer
    ) throws -> CompiledFunction {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        // Create symbolic input
        let input = context.input(shape: inputShape, dtype: dtype)

        // Trace the function
        let output = function(input)

        // Set output
        context.output(output)

        // Build and compile
        let module = context.buildModule()
        let compiler = SwiftIRCompiler(options: compilerOptions)

        CompilableTracer.currentBuilder = nil

        return try compiler.compile(module)
    }

    /// Compile a function with multiple inputs
    public func compile(
        inputSpecs: [(shape: [Int], dtype: DType)],
        _ function: ([CompilableTracer]) -> CompilableTracer
    ) throws -> CompiledFunction {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        // Create symbolic inputs
        let inputs = inputSpecs.map { spec in
            context.input(shape: spec.shape, dtype: spec.dtype)
        }

        // Trace the function
        let output = function(inputs)

        // Set output
        context.output(output)

        // Build and compile
        let module = context.buildModule()
        let compiler = SwiftIRCompiler(options: compilerOptions)

        CompilableTracer.currentBuilder = nil

        return try compiler.compile(module)
    }

    /// Compile a function with gradient computation
    public func compileWithGradients(
        inputShape: [Int],
        dtype: DType = .float32,
        _ function: @escaping (CompilableTracer) -> CompilableTracer
    ) throws -> CompiledGradientFunction {
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        // Create symbolic input
        let input = context.input(shape: inputShape, dtype: dtype, name: "%x")

        // Forward pass
        let output = function(input)

        // Create seed gradient for backward pass
        let seedGrad = context.input(shape: output.shape, dtype: dtype, name: "%seed")

        // For now, we'll need to manually specify the backward pass
        // In the full implementation, this would use Swift's AD

        context.output(output)

        let module = context.buildModule(name: "forward_with_grad")
        let compiler = SwiftIRCompiler(options: compilerOptions)

        CompilableTracer.currentBuilder = nil

        let compiled = try compiler.compile(module)

        return CompiledGradientFunction(
            forward: compiled,
            inputShape: inputShape,
            outputShape: output.shape
        )
    }
}

// MARK: - Compiled Gradient Function

/// A compiled function that can compute both forward and backward passes
public class CompiledGradientFunction {
    public let forward: CompiledFunction
    public let inputShape: [Int]
    public let outputShape: [Int]

    public init(forward: CompiledFunction, inputShape: [Int], outputShape: [Int]) {
        self.forward = forward
        self.inputShape = inputShape
        self.outputShape = outputShape
    }

    /// Run forward pass
    public func run(_ inputs: [[Float]]) -> [[Float]] {
        forward.run(inputs)
    }

    /// Get info about the compiled function
    public var info: String {
        """
        Gradient Function:
          Input shape: \(inputShape)
          Output shape: \(outputShape)
        \(forward.info)
        """
    }
}

// MARK: - Convenience API

/// Trace and compile a function
public func trace(
    input: (shape: [Int], dtype: DType),
    options: CompilationOptions = .default,
    _ function: (CompilableTracer) -> CompilableTracer
) throws -> CompiledFunction {
    let compiler = FunctionCompiler(options: options)
    return try compiler.compile(inputShape: input.shape, dtype: input.dtype, function)
}

/// Trace and compile a multi-input function
public func trace(
    inputs: [(shape: [Int], dtype: DType)],
    options: CompilationOptions = .default,
    _ function: ([CompilableTracer]) -> CompilableTracer
) throws -> CompiledFunction {
    let compiler = FunctionCompiler(options: options)
    return try compiler.compile(inputSpecs: inputs, function)
}

// MARK: - Input Specification

/// Specification for a tensor input
public struct TensorSpec {
    public let shape: [Int]
    public let dtype: DType

    public init(shape: [Int], dtype: DType = .float32) {
        self.shape = shape
        self.dtype = dtype
    }

    /// Create a tensor spec from dimensions
    public static func tensor(_ dimensions: Int..., dtype: DType = .float32) -> TensorSpec {
        TensorSpec(shape: dimensions, dtype: dtype)
    }
}
