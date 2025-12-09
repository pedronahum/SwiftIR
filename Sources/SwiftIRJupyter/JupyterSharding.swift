// JupyterSharding.swift - SDY Sharding Integration for Jupyter/Colab
// Pure Swift - works without C++ interop
//
// Integrates Shardy sharding annotations with the Jupyter tracing API,
// enabling distributed tensor computation across device meshes.

import Foundation

// MARK: - SDY Device Mesh (Native Jupyter Version)

/// Device mesh for SDY sharding in Jupyter.
///
/// This is a native Jupyter implementation that mirrors SwiftIRShardingLite.DeviceMesh
/// but is specifically designed for the Jupyter tracing API.
public struct JSDYMesh: Sendable, Hashable {
    /// Name of this mesh (used as MLIR symbol)
    public let name: String

    /// Axes and their sizes
    public let axes: [(name: String, size: Int)]

    /// Total number of devices
    public var deviceCount: Int {
        axes.reduce(1) { $0 * $1.size }
    }

    /// Creates a device mesh.
    public init(name: String, axes: [(name: String, size: Int)]) {
        self.name = name
        self.axes = axes
    }

    /// Creates a 1D linear mesh for data parallelism.
    public static func linear(name: String = "devices", axis: String = "x", size: Int) -> JSDYMesh {
        JSDYMesh(name: name, axes: [(axis, size)])
    }

    /// Creates a 2D mesh for hybrid data/model parallelism.
    public static func grid(
        name: String = "mesh",
        dataParallel: Int,
        modelParallel: Int,
        dataAxis: String = "data",
        modelAxis: String = "model"
    ) -> JSDYMesh {
        JSDYMesh(name: name, axes: [(dataAxis, dataParallel), (modelAxis, modelParallel)])
    }

    /// Creates a TPU-style 2D mesh.
    public static func tpuMesh(name: String = "tpu_mesh", x: Int, y: Int) -> JSDYMesh {
        JSDYMesh(name: name, axes: [("x", x), ("y", y)])
    }

    /// MLIR text for mesh definition
    public var mlirText: String {
        let axesStr = axes.map { "\"\($0.name)\"=\($0.size)" }.joined(separator: ", ")
        return "sdy.mesh @\(name) = <[\(axesStr)]>"
    }

    // Hashable conformance (tuples aren't Hashable by default)
    public static func == (lhs: JSDYMesh, rhs: JSDYMesh) -> Bool {
        lhs.name == rhs.name &&
        lhs.axes.count == rhs.axes.count &&
        zip(lhs.axes, rhs.axes).allSatisfy { $0.name == $1.name && $0.size == $1.size }
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        for axis in axes {
            hasher.combine(axis.name)
            hasher.combine(axis.size)
        }
    }
}

// MARK: - SDY Tensor Sharding

/// Specifies how a tensor is sharded across a mesh.
public struct JSDYSharding: Sendable, Hashable, CustomStringConvertible {
    /// The mesh name this sharding refers to
    public let meshName: String

    /// Axis assignments for each dimension (nil = replicated)
    public let dimAxes: [String?]

    /// Creates a tensor sharding.
    public init(mesh: String, axes: [String?]) {
        self.meshName = mesh
        self.dimAxes = axes
    }

    /// Creates a tensor sharding from a mesh.
    public init(mesh: JSDYMesh, axes: [String?]) {
        self.meshName = mesh.name
        self.dimAxes = axes
    }

    /// Fully replicated sharding (no partitioning).
    public static func replicated(mesh: String, rank: Int) -> JSDYSharding {
        JSDYSharding(mesh: mesh, axes: Array(repeating: nil, count: rank))
    }

    /// Fully replicated sharding from mesh.
    public static func replicated(mesh: JSDYMesh, rank: Int) -> JSDYSharding {
        JSDYSharding(mesh: mesh.name, axes: Array(repeating: nil, count: rank))
    }

    /// Data parallel sharding (batch dimension on first axis).
    public static func dataParallel(mesh: JSDYMesh, rank: Int, batchAxis: String = "data") -> JSDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if rank > 0 {
            axes[0] = batchAxis
        }
        return JSDYSharding(mesh: mesh.name, axes: axes)
    }

    /// Model parallel sharding (features on specified axis).
    public static func modelParallel(
        mesh: JSDYMesh,
        rank: Int,
        featureDim: Int,
        modelAxis: String = "model"
    ) -> JSDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if featureDim < rank {
            axes[featureDim] = modelAxis
        }
        return JSDYSharding(mesh: mesh.name, axes: axes)
    }

    /// MLIR attribute text
    public var mlirAttributeText: String {
        let dimStr = dimAxes.map { axis in
            if let axis = axis {
                return "{\"\(axis)\"}"
            } else {
                return "{}"
            }
        }.joined(separator: ", ")
        return "#sdy.sharding<@\(meshName), [\(dimStr)]>"
    }

    public var description: String {
        mlirAttributeText
    }
}

// MARK: - Sharded Tracing Context

/// A tracing context that supports SDY sharding annotations.
///
/// Example usage:
/// ```swift
/// let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
/// let ctx = JShardedTracingContext(mesh: mesh)
///
/// let x = ctx.input(shape: [16, 64], dtype: .float32,
///                   sharding: .dataParallel(mesh: mesh, rank: 2))
/// let w = ctx.input(shape: [64, 128], dtype: .float32,
///                   sharding: .modelParallel(mesh: mesh, rank: 2, featureDim: 1))
///
/// let y = x.matmul(w)
/// ctx.output(y, sharding: JSDYSharding(mesh: mesh, axes: ["data", "model"]))
///
/// let mlir = ctx.buildShardedModule()
/// ```
public class JShardedTracingContext {
    /// The base tracing context
    public let baseContext: JTracingContext

    /// Device mesh for this context
    public let mesh: JSDYMesh

    /// Input shardings by argument name
    private var inputShardings: [String: JSDYSharding] = [:]

    /// Output shardings
    private var outputShardings: [JSDYSharding] = []

    /// Argument counter
    private var argumentCount: Int = 0

    /// Creates a sharded tracing context.
    public init(mesh: JSDYMesh) {
        self.mesh = mesh
        self.baseContext = JTracingContext()
    }

    /// Create a sharded symbolic input.
    public func input(
        shape: JTensorShape,
        dtype: JDType = .float32,
        name: String? = nil,
        sharding: JSDYSharding? = nil
    ) -> JTracer {
        let argName = name ?? "%arg\(argumentCount)"
        argumentCount += 1

        // Store sharding for this input
        if let sharding = sharding {
            inputShardings[argName] = sharding
        }

        // Create the symbolic input through base context
        return baseContext.input(shape: shape, dtype: dtype, name: argName)
    }

    /// Set sharded outputs.
    public func output(_ tracers: JTracer..., sharding: JSDYSharding? = nil) {
        if let sharding = sharding {
            outputShardings = Array(repeating: sharding, count: tracers.count)
        }
        baseContext.output(tracers.first!)
    }

    /// Build the MLIR module with SDY sharding annotations.
    public func buildShardedModule(name: String = "sharded") -> String {
        // Get the base MLIR
        let baseMLIR = baseContext.buildModule(name: name)

        // Insert mesh definition and sharding annotations
        return insertShardingAnnotations(into: baseMLIR)
    }

    /// Insert sharding annotations into the MLIR module.
    private func insertShardingAnnotations(into mlir: String) -> String {
        var result = ""

        // Add mesh definition at the start of the module
        result += "module @sharded {\n"
        result += "  \(mesh.mlirText)\n\n"

        // Process the original MLIR to add sharding attributes
        let lines = mlir.components(separatedBy: "\n")
        var inFunction = false

        for line in lines {
            // Skip the original module wrapper
            if line.contains("module @") {
                continue
            }
            if line == "}" && !inFunction {
                continue
            }

            // Detect function signature
            if line.contains("func.func @") {
                inFunction = true
                result += insertArgumentShardings(in: line)
                result += "\n"
                continue
            }

            // Detect function end
            if line.trimmed == "}" && inFunction {
                inFunction = false
            }

            result += "\(line)\n"
        }

        result += "}\n"
        return result
    }

    /// Insert sharding attributes into function arguments.
    private func insertArgumentShardings(in funcLine: String) -> String {
        var result = funcLine

        // Sort by arg index to process in order (otherwise string offsets shift)
        let sortedShardings = inputShardings.sorted { a, b in
            // Extract arg number from name like "%arg0"
            let numA = Int(a.key.replacingOccurrences(of: "%arg", with: "")) ?? 0
            let numB = Int(b.key.replacingOccurrences(of: "%arg", with: "")) ?? 0
            return numA > numB  // Process in reverse order to maintain correct indices
        }

        for (argName, sharding) in sortedShardings {
            // Find the argument in the function signature
            let pattern = "\(argName): tensor<"
            if let patternRange = result.range(of: pattern) {
                // Find the matching '>' for this tensor type by counting brackets
                var depth = 1
                var idx = patternRange.upperBound
                while idx < result.endIndex && depth > 0 {
                    let char = result[idx]
                    if char == "<" {
                        depth += 1
                    } else if char == ">" {
                        depth -= 1
                    }
                    idx = result.index(after: idx)
                }

                if depth == 0 {
                    // idx is now just after the final '>'
                    let shardingAttr = " {sdy.sharding = \(sharding.mlirAttributeText)}"
                    result.insert(contentsOf: shardingAttr, at: idx)
                }
            }
        }

        return result
    }
}

// MARK: - Sharded Function Compiler

/// Compiles Swift functions with sharding annotations.
public class JShardedFunctionCompiler {
    public init() {}

    /// Compile a sharded function.
    public func compile(
        mesh: JSDYMesh,
        inputSpecs: [(shape: [Int], dtype: JDType, sharding: JSDYSharding?)],
        outputSharding: JSDYSharding? = nil,
        _ function: ([JTracer]) -> JTracer
    ) throws -> JupyterCompiledFunction {
        let context = JShardedTracingContext(mesh: mesh)

        // Create symbolic inputs with shardings
        let inputs = inputSpecs.enumerated().map { (i, spec) in
            context.input(
                shape: JTensorShape(spec.shape),
                dtype: spec.dtype,
                sharding: spec.sharding
            )
        }

        // Trace the function
        let output = function(inputs)

        // Set output with sharding
        context.output(output, sharding: outputSharding)

        // Build sharded MLIR
        let mlir = context.buildShardedModule()

        return JupyterCompiledFunction(mlirSource: mlir)
    }
}

// MARK: - Convenience Functions

/// Trace and compile a sharded function.
public func jitCompileSharded(
    mesh: JSDYMesh,
    inputSpecs: [(shape: [Int], dtype: JDType, sharding: JSDYSharding?)],
    outputSharding: JSDYSharding? = nil,
    _ function: ([JTracer]) -> JTracer
) throws -> JupyterCompiledFunction {
    let compiler = JShardedFunctionCompiler()
    return try compiler.compile(
        mesh: mesh,
        inputSpecs: inputSpecs,
        outputSharding: outputSharding,
        function
    )
}

/// Simple data-parallel compilation helper.
public func jitCompileDataParallel(
    numDevices: Int,
    inputShapes: [[Int]],
    dtype: JDType = .float32,
    _ function: ([JTracer]) -> JTracer
) throws -> JupyterCompiledFunction {
    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: numDevices)

    let inputSpecs = inputShapes.map { shape in
        (
            shape: shape,
            dtype: dtype,
            sharding: JSDYSharding.dataParallel(mesh: mesh, rank: shape.count, batchAxis: "batch")
        )
    }

    return try jitCompileSharded(
        mesh: mesh,
        inputSpecs: inputSpecs,
        outputSharding: inputSpecs.first?.sharding,
        function
    )
}

// MARK: - Bridge to JDeviceMesh

extension JDeviceMesh {
    /// Convert to SDY mesh.
    public func toSDYMesh() -> JSDYMesh {
        let axes = zip(axisNames, shape).map { (name: $0, size: $1) }
        return JSDYMesh(name: name, axes: axes)
    }
}

extension JShardingSpec {
    /// Convert to SDY sharding.
    public func toSDYSharding() -> JSDYSharding {
        JSDYSharding(mesh: mesh.name, axes: dimMapping)
    }
}

// MARK: - MLIR Module with Sharding

/// Extension to JMLIRBuilder for sharded module generation.
extension JMLIRBuilder {
    /// Build a sharded MLIR module with mesh definition.
    public func buildSharded(
        functionName: String = "main",
        mesh: JSDYMesh,
        argumentShardings: [JSDYSharding] = [],
        resultShardings: [JSDYSharding] = []
    ) -> String {
        var text = "module @\(functionName) {\n"

        // Add mesh definition
        text += "  \(mesh.mlirText)\n\n"

        // Build function signature with sharding attributes
        text += "  func.func @main("

        // Add arguments with shardings
        for (i, arg) in arguments.enumerated() {
            if i > 0 { text += ", " }
            text += arg

            // Add sharding attribute if available
            if i < argumentShardings.count {
                text += " {sdy.sharding = \(argumentShardings[i].mlirAttributeText)}"
            }
        }
        text += ")"

        // Add return type
        if !results.isEmpty && !operations.isEmpty {
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                }
            }

            if resultTypes.count == 1 {
                text += " -> \(resultTypes[0])"
            } else if resultTypes.count > 1 {
                text += " -> (\(resultTypes.joined(separator: ", ")))"
            }
        }

        text += " {\n"

        // Add operations
        for op in operations {
            text += "    \(op.mlirText)\n"
        }

        // Add return
        if !results.isEmpty {
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                }
            }
            text += "    return \(results.joined(separator: ", ")) : \(resultTypes.joined(separator: ", "))\n"
        }

        text += "  }\n"
        text += "}\n"
        return text
    }
}

// MARK: - String Extension

fileprivate extension String {
    var trimmed: String {
        trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Sharding Visualization

extension JSDYMesh {
    /// Print mesh information.
    public func printInfo() {
        print("SDY Device Mesh: \(name)")
        print("  Total devices: \(deviceCount)")
        print("  Axes:")
        for axis in axes {
            print("    \(axis.name): \(axis.size)")
        }
        print("  MLIR: \(mlirText)")
    }
}

extension JSDYSharding {
    /// Print sharding information.
    public func printInfo(shape: [Int]? = nil) {
        print("SDY Tensor Sharding")
        print("  Mesh: \(meshName)")
        print("  Dimensions:")
        for (i, axis) in dimAxes.enumerated() {
            let dimSize = shape?[i].description ?? "?"
            let axisStr = axis ?? "replicated"
            print("    dim[\(i)] (size \(dimSize)): \(axisStr)")
        }
        print("  MLIR: \(mlirAttributeText)")
    }
}
