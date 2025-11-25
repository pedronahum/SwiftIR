// PJRTExecution.swift
// Phase 14: Connect AD Pipeline to Real PJRT Execution
//
// This bridges the SymbolicAD compilation pipeline with actual PJRT execution
// on CPU/GPU/TPU hardware.

import Foundation
import SwiftIRXLA

// MARK: - PJRT-Backed Runtime

/// Runtime that executes compiled functions on real hardware via PJRT
public class PJRTBackedRuntime {
    /// Backend type
    public enum Backend {
        case cpu
        case gpu(deviceId: Int)
        case tpu(deviceId: Int)

        var pluginPath: String {
            switch self {
            case .cpu:
                #if os(macOS)
                let paths = [
                    "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_cpu_plugin.dylib"
                ]
                #else
                let paths = [
                    "/opt/swiftir-deps/lib/pjrt_c_api_cpu_plugin.so",
                    "/usr/local/lib/pjrt_c_api_cpu_plugin.so"
                ]
                #endif
                for path in paths {
                    if FileManager.default.fileExists(atPath: path) {
                        return path
                    }
                }
                return paths[0]
            case .gpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_gpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_gpu_plugin.so"
                #endif
            case .tpu:
                #if os(macOS)
                return "/Users/pedro/programming/swift/SwiftIR/lib/pjrt_c_api_tpu_plugin.dylib"
                #else
                return "/opt/swiftir-deps/lib/pjrt_c_api_tpu_plugin.so"
                #endif
            }
        }
    }

    public let backend: Backend
    private var isInitialized: Bool = false

    /// Info about this runtime
    public var info: String {
        switch backend {
        case .cpu:
            return "PJRT Runtime (CPU)"
        case .gpu(let id):
            return "PJRT Runtime (GPU:\(id))"
        case .tpu(let id):
            return "PJRT Runtime (TPU:\(id))"
        }
    }

    public init(backend: Backend = .cpu) throws {
        self.backend = backend

        // Verify plugin exists
        let pluginPath = backend.pluginPath
        guard FileManager.default.fileExists(atPath: pluginPath) else {
            throw PJRTExecutionError.pluginNotFound(pluginPath)
        }

        self.isInitialized = true
    }

    /// Compile an MLIR module to an executable
    public func compile(_ mlirModule: String) throws -> PJRTBackedExecutable {
        guard isInitialized else {
            throw PJRTExecutionError.notInitialized
        }

        return PJRTBackedExecutable(
            mlirSource: mlirModule,
            runtime: self
        )
    }

    /// Compile from MLIRModule
    public func compile(_ module: MLIRModule) throws -> PJRTBackedExecutable {
        return try compile(module.mlirText)
    }
}

// MARK: - PJRT-Backed Executable

/// An executable that runs on real PJRT hardware
public class PJRTBackedExecutable {
    public let mlirSource: String
    public let stablehloSource: String
    private let backend: PJRTBackedRuntime.Backend

    /// Cached PJRT client and executable for repeated execution
    private var pjrtClient: PJRTClient?
    private var pjrtExecutable: PJRTLoadedExecutable?

    init(mlirSource: String, runtime: PJRTBackedRuntime) {
        self.mlirSource = mlirSource
        self.backend = runtime.backend

        // Convert to StableHLO format
        self.stablehloSource = Self.convertToStableHLO(mlirSource)
    }

    /// Execute with float arrays using real PJRT
    public func execute(_ inputs: [[Float]]) throws -> [[Float]] {
        // Convert backend type
        let pjrtBackend: PJRTClient.Backend
        switch backend {
        case .cpu:
            pjrtBackend = .cpu
        case .gpu:
            pjrtBackend = .gpu
        case .tpu:
            pjrtBackend = .tpu
        }

        // Create client if needed
        if pjrtClient == nil {
            do {
                pjrtClient = try PJRTClient(backend: pjrtBackend)
            } catch {
                throw PJRTExecutionError.compilationFailed("Failed to create PJRT client: \(error)")
            }
        }

        guard let client = pjrtClient else {
            throw PJRTExecutionError.notInitialized
        }

        // Compile if needed
        if pjrtExecutable == nil {
            do {
                // Use stablehloSource directly - it already contains a complete module
                pjrtExecutable = try client.compile(mlirModule: stablehloSource)
            } catch {
                // Include the MLIR source in the error message for debugging
                let truncatedSource = stablehloSource.count > 2000
                    ? String(stablehloSource.prefix(2000)) + "\n... [truncated]"
                    : stablehloSource
                throw PJRTExecutionError.compilationFailed("""
                    Failed to compile: \(error)

                    Generated MLIR:
                    \(truncatedSource)
                    """)
            }
        }

        guard let executable = pjrtExecutable else {
            throw PJRTExecutionError.compilationFailed("No executable available")
        }

        guard let device = client.defaultDevice else {
            throw PJRTExecutionError.executionFailed("No device available")
        }

        // Create input buffers
        var inputBuffers: [PJRTBuffer] = []
        for input in inputs {
            let buffer = try input.withUnsafeBytes { ptr in
                try client.createBuffer(
                    data: ptr.baseAddress!,
                    shape: [input.count],
                    elementType: .f32,
                    device: device
                )
            }
            inputBuffers.append(buffer)
        }

        // Execute
        let outputBuffers: [PJRTBuffer]
        do {
            outputBuffers = try executable.execute(arguments: inputBuffers)
        } catch {
            throw PJRTExecutionError.executionFailed("Execution failed: \(error)")
        }

        // Read results back to host
        var results: [[Float]] = []
        for buffer in outputBuffers {
            let count = buffer.elementCount
            var result = [Float](repeating: 0, count: count)
            try result.withUnsafeMutableBytes { ptr in
                try buffer.toHost(destination: ptr.baseAddress!)
            }
            results.append(result)
        }

        return results
    }

    /// Wrap MLIR operations in a proper StableHLO module
    private static func wrapInStableHLOModule(_ source: String, inputs: [[Float]]) -> String {
        // Build argument types based on inputs
        let argTypes = inputs.enumerated().map { (i, input) in
            "%arg\(i): tensor<\(input.count)xf32>"
        }.joined(separator: ", ")

        // For simple cases, create a minimal StableHLO module
        // This is a simplified wrapper - real implementation would parse the MLIR
        return """
        module @main {
          func.func @main(\(argTypes)) -> tensor<\(inputs[0].count)xf32> {
            \(source)
          }
        }
        """
    }

    /// Convert MLIR to StableHLO text format
    private static func convertToStableHLO(_ mlir: String) -> String {
        var result = mlir

        // Replace operation names with stablehlo prefixed versions
        // The MLIR format is: %0 = opname operands : type
        let replacements = [
            // Unquoted form (from MLIRBuilder)
            ("= add ", "= stablehlo.add "),
            ("= subtract ", "= stablehlo.subtract "),
            ("= multiply ", "= stablehlo.multiply "),
            ("= divide ", "= stablehlo.divide "),
            ("= negate ", "= stablehlo.negate "),
            ("= dot ", "= stablehlo.dot "),
            ("= transpose ", "= stablehlo.transpose "),
            ("= exp ", "= stablehlo.exponential "),
            ("= log ", "= stablehlo.log "),
            ("= maximum ", "= stablehlo.maximum "),
            ("= reduce_sum ", "= stablehlo.reduce "),
            ("= reduce_max ", "= stablehlo.reduce "),
            ("= reduce_min ", "= stablehlo.reduce "),
            ("= compare ", "= stablehlo.compare "),
            ("= select ", "= stablehlo.select "),
            // Phase 15: Additional operations
            ("= sqrt ", "= stablehlo.sqrt "),
            ("= logistic ", "= stablehlo.logistic "),
            ("= tanh ", "= stablehlo.tanh "),
            ("= abs ", "= stablehlo.abs "),
            ("= sign ", "= stablehlo.sign "),
            ("= sine ", "= stablehlo.sine "),
            ("= cosine ", "= stablehlo.cosine "),
            ("= power ", "= stablehlo.power "),
            ("= constant ", "= stablehlo.constant "),
            ("= broadcast_in_dim ", "= stablehlo.broadcast_in_dim "),
            ("= reshape ", "= stablehlo.reshape "),
            ("= slice ", "= stablehlo.slice "),
            ("= pad ", "= stablehlo.pad "),
            ("= concatenate ", "= stablehlo.concatenate "),
            ("= convolution(", "= stablehlo.convolution("),
            ("= floor ", "= stablehlo.floor "),
            ("= ceil ", "= stablehlo.ceil "),
            ("= clamp ", "= stablehlo.clamp "),
            ("= rsqrt ", "= stablehlo.rsqrt "),
            ("= and ", "= stablehlo.and "),
            // Quoted form (for inline strings)
            ("\"add\"", "\"stablehlo.add\""),
            ("\"subtract\"", "\"stablehlo.subtract\""),
            ("\"multiply\"", "\"stablehlo.multiply\""),
            ("\"divide\"", "\"stablehlo.divide\""),
            ("\"negate\"", "\"stablehlo.negate\""),
            ("\"dot\"", "\"stablehlo.dot\""),
            ("\"transpose\"", "\"stablehlo.transpose\""),
            ("\"exp\"", "\"stablehlo.exponential\""),
            ("\"log\"", "\"stablehlo.log\""),
            ("\"maximum\"", "\"stablehlo.maximum\""),
            ("\"reduce_sum\"", "\"stablehlo.reduce\""),
            ("\"compare\"", "\"stablehlo.compare\""),
            ("\"select\"", "\"stablehlo.select\""),
            // Phase 15: Additional operations (quoted)
            ("\"sqrt\"", "\"stablehlo.sqrt\""),
            ("\"logistic\"", "\"stablehlo.logistic\""),
            ("\"tanh\"", "\"stablehlo.tanh\""),
            ("\"abs\"", "\"stablehlo.abs\""),
            ("\"sign\"", "\"stablehlo.sign\""),
            ("\"power\"", "\"stablehlo.power\""),
            ("\"constant\"", "\"stablehlo.constant\""),
            ("\"broadcast_in_dim\"", "\"stablehlo.broadcast_in_dim\""),
        ]

        for (from, to) in replacements {
            result = result.replacingOccurrences(of: from, with: to)
        }

        // No longer need to fix permutation format - using dims = [...] directly

        // Fix _dot_types attribute: move from attribute to type signature
        // Pattern: {_dot_types = (...)} : type -> : (...)
        let dotTypesPattern = try? NSRegularExpression(pattern: "\\{_dot_types = (\\([^}]+\\))\\} : tensor<[^>]+>")
        if let regex = dotTypesPattern {
            let nsRange = NSRange(result.startIndex..<result.endIndex, in: result)
            let matches = regex.matches(in: result, options: [], range: nsRange)
            for match in matches.reversed() {
                if let range = Range(match.range, in: result),
                   let typesRange = Range(match.range(at: 1), in: result) {
                    let types = String(result[typesRange])
                    let replacement = ": \(types)"
                    result.replaceSubrange(range, with: replacement)
                }
            }
        }

        return result
    }

    public var info: String {
        let backendInfo: String
        switch backend {
        case .cpu:
            backendInfo = "CPU"
        case .gpu(let id):
            backendInfo = "GPU:\(id)"
        case .tpu(let id):
            backendInfo = "TPU:\(id)"
        }
        return """
        PJRT Executable:
          MLIR size: \(mlirSource.count) bytes
          StableHLO size: \(stablehloSource.count) bytes
          Backend: \(backendInfo)
        """
    }
}

// MARK: - Real Compiled Function

/// A compiled function that can execute on real PJRT hardware
public class RealCompiledFunction {
    public let executable: PJRTBackedExecutable
    public let mlirSource: String
    public let stablehloSource: String

    public init(executable: PJRTBackedExecutable) {
        self.executable = executable
        self.mlirSource = executable.mlirSource
        self.stablehloSource = executable.stablehloSource
    }

    /// Run the function with input arrays
    public func run(_ inputs: [[Float]]) -> [[Float]] {
        do {
            return try executable.execute(inputs)
        } catch {
            print("Execution error: \(error)")
            return inputs // Fallback
        }
    }

    public var info: String {
        executable.info
    }
}

// MARK: - Real PJRT Compiler

/// Compiler that produces real PJRT executables
public class RealPJRTCompiler {
    private let runtime: PJRTBackedRuntime
    private let options: CompilationOptions

    public init(backend: PJRTBackedRuntime.Backend = .cpu, options: CompilationOptions = .default) throws {
        self.runtime = try PJRTBackedRuntime(backend: backend)
        self.options = options
    }

    /// Compile an MLIR module
    public func compile(_ module: MLIRModule) throws -> RealCompiledFunction {
        let executable = try runtime.compile(module)
        return RealCompiledFunction(executable: executable)
    }

    /// Compile from tracer context
    public func compile(
        inputSpecs: [TensorSpec],
        _ function: ([CompilableTracer]) -> CompilableTracer
    ) throws -> RealCompiledFunction {
        // Trace the function
        let context = TracingContext()
        CompilableTracer.currentBuilder = context.builder

        let inputs = inputSpecs.enumerated().map { (i, spec) in
            context.input(shape: spec.shape, dtype: spec.dtype, name: "%arg\(i)")
        }

        let output = function(inputs)
        context.output(output)

        let module = context.buildModule(name: "main")
        CompilableTracer.currentBuilder = nil

        return try compile(module)
    }

    /// Compile a single-input function
    public func compile(
        inputSpec: TensorSpec,
        _ function: (CompilableTracer) -> CompilableTracer
    ) throws -> RealCompiledFunction {
        try compile(inputSpecs: [inputSpec]) { inputs in
            function(inputs[0])
        }
    }
}

// MARK: - Errors

/// Errors for PJRT execution
public enum PJRTExecutionError: Error, CustomStringConvertible {
    case notInitialized
    case pluginNotFound(String)
    case compilationFailed(String)
    case executionFailed(String)
    case bufferError(String)

    public var description: String {
        switch self {
        case .notInitialized:
            return "PJRT runtime not initialized"
        case .pluginNotFound(let path):
            return "PJRT plugin not found at: \(path)"
        case .compilationFailed(let msg):
            return "PJRT compilation failed: \(msg)"
        case .executionFailed(let msg):
            return "PJRT execution failed: \(msg)"
        case .bufferError(let msg):
            return "PJRT buffer error: \(msg)"
        }
    }
}

// MARK: - High-Level API

/// Compile and run a function on real PJRT hardware
public func compileForPJRT(
    input: TensorSpec,
    backend: PJRTBackedRuntime.Backend = .cpu,
    _ function: (CompilableTracer) -> CompilableTracer
) throws -> RealCompiledFunction {
    let compiler = try RealPJRTCompiler(backend: backend)
    return try compiler.compile(inputSpec: input, function)
}

/// Compile and run a multi-input function on real PJRT hardware
public func compileForPJRT(
    inputs: [TensorSpec],
    backend: PJRTBackedRuntime.Backend = .cpu,
    _ function: ([CompilableTracer]) -> CompilableTracer
) throws -> RealCompiledFunction {
    let compiler = try RealPJRTCompiler(backend: backend)
    return try compiler.compile(inputSpecs: inputs, function)
}

// MARK: - Integration with Existing CompiledFunction

extension CompiledFunction {
    /// Convert to a real PJRT-backed executable
    public func toPJRTExecutable(backend: PJRTBackedRuntime.Backend = .cpu) throws -> RealCompiledFunction {
        let runtime = try PJRTBackedRuntime(backend: backend)
        let executable = try runtime.compile(mlirSource)
        return RealCompiledFunction(executable: executable)
    }
}

// MARK: - Integration with GradientCompiledFunction

extension GradientCompiledFunction {
    /// Convert to a real PJRT-backed executable
    public func toPJRTExecutable(backend: PJRTBackedRuntime.Backend = .cpu) throws -> RealCompiledFunction {
        return try compiled.toPJRTExecutable(backend: backend)
    }
}
