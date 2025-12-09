/// SDYSharding.swift - Shardy Dialect Integration for SwiftIR
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Provides SDY (Shardy) sharding annotations for distributed tensor computation:
/// - SDYMesh: Device mesh topology definitions
/// - SDYSharding: Tensor sharding specifications with MLIR attribute generation
/// - ShardedTracer: Tracer with sharding annotations
/// - ShardedTracingContext: Context for tracing with sharding

import Foundation

// MARK: - SDY Device Mesh

/// Device mesh for SDY sharding dialect.
///
/// Defines the topology of devices for distributed computation.
/// Generates MLIR mesh definitions like: `sdy.mesh @mesh_name = <["axis"=size, ...]>`
///
/// Example:
/// ```swift
/// let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: 4)
/// let mesh2D = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
/// ```
public struct SDYMesh: Sendable, Hashable {
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
    public static func linear(name: String = "devices", axis: String = "x", size: Int) -> SDYMesh {
        SDYMesh(name: name, axes: [(axis, size)])
    }

    /// Creates a 2D mesh for hybrid data/model parallelism.
    public static func grid(
        name: String = "mesh",
        dataParallel: Int,
        modelParallel: Int,
        dataAxis: String = "data",
        modelAxis: String = "model"
    ) -> SDYMesh {
        SDYMesh(name: name, axes: [(dataAxis, dataParallel), (modelAxis, modelParallel)])
    }

    /// Creates a TPU-style 2D mesh.
    public static func tpuMesh(name: String = "tpu_mesh", x: Int, y: Int) -> SDYMesh {
        SDYMesh(name: name, axes: [("x", x), ("y", y)])
    }

    /// Creates a 3D mesh for advanced parallelism strategies.
    public static func cube(
        name: String = "cube",
        data: Int,
        tensor: Int,
        pipeline: Int
    ) -> SDYMesh {
        SDYMesh(name: name, axes: [("data", data), ("tensor", tensor), ("pipeline", pipeline)])
    }

    /// MLIR text for mesh definition
    public var mlirText: String {
        let axesStr = axes.map { "\"\($0.name)\"=\($0.size)" }.joined(separator: ", ")
        return "sdy.mesh @\(name) = <[\(axesStr)]>"
    }

    /// Get the size of a named axis
    public func axisSize(_ axisName: String) -> Int? {
        axes.first { $0.name == axisName }?.size
    }

    /// Get axis names
    public var axisNames: [String] {
        axes.map { $0.name }
    }

    // Hashable conformance (tuples aren't Hashable by default)
    public static func == (lhs: SDYMesh, rhs: SDYMesh) -> Bool {
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
///
/// Generates MLIR sharding attributes like:
/// `#sdy.sharding<@mesh_name, [{"axis"}, {}, {"axis2"}]>`
///
/// Example:
/// ```swift
/// let dataParallel = SDYSharding.dataParallel(mesh: mesh, rank: 2)
/// let custom = SDYSharding(mesh: mesh, axes: ["data", nil, "model"])
/// ```
public struct SDYSharding: Sendable, Hashable, CustomStringConvertible {
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
    public init(mesh: SDYMesh, axes: [String?]) {
        self.meshName = mesh.name
        self.dimAxes = axes
    }

    /// Fully replicated sharding (no partitioning).
    public static func replicated(mesh: String, rank: Int) -> SDYSharding {
        SDYSharding(mesh: mesh, axes: Array(repeating: nil, count: rank))
    }

    /// Fully replicated sharding from mesh.
    public static func replicated(mesh: SDYMesh, rank: Int) -> SDYSharding {
        SDYSharding(mesh: mesh.name, axes: Array(repeating: nil, count: rank))
    }

    /// Data parallel sharding (batch dimension on specified axis).
    public static func dataParallel(mesh: SDYMesh, rank: Int, batchAxis: String? = nil) -> SDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if rank > 0 {
            // Use first axis of mesh if no explicit batchAxis provided
            axes[0] = batchAxis ?? mesh.axes.first?.name
        }
        return SDYSharding(mesh: mesh.name, axes: axes)
    }

    /// Model parallel sharding (features on specified axis).
    public static func modelParallel(
        mesh: SDYMesh,
        rank: Int,
        featureDim: Int,
        modelAxis: String? = nil
    ) -> SDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if featureDim < rank {
            // Use second axis of mesh if no explicit modelAxis provided, otherwise first
            axes[featureDim] = modelAxis ?? (mesh.axes.count > 1 ? mesh.axes[1].name : mesh.axes.first?.name)
        }
        return SDYSharding(mesh: mesh.name, axes: axes)
    }

    /// Column-parallel sharding for tensor parallelism (shard output dimension).
    public static func columnParallel(mesh: SDYMesh, rank: Int, axis: String? = nil) -> SDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if rank > 1 {
            axes[rank - 1] = axis ?? mesh.axes.first?.name
        }
        return SDYSharding(mesh: mesh.name, axes: axes)
    }

    /// Row-parallel sharding for tensor parallelism (shard input dimension).
    public static func rowParallel(mesh: SDYMesh, rank: Int, axis: String? = nil) -> SDYSharding {
        var axes: [String?] = Array(repeating: nil, count: rank)
        if rank > 1 {
            axes[rank - 2] = axis ?? mesh.axes.first?.name
        }
        return SDYSharding(mesh: mesh.name, axes: axes)
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

    /// Check if this sharding is fully replicated
    public var isReplicated: Bool {
        dimAxes.allSatisfy { $0 == nil }
    }

    /// Get the sharding for a specific dimension
    public func shardingForDim(_ dim: Int) -> String? {
        guard dim < dimAxes.count else { return nil }
        return dimAxes[dim]
    }
}

// MARK: - Sharded Tracer

/// A Tracer with associated sharding information.
///
/// This wraps a regular Tracer with sharding metadata for distributed computation.
public struct ShardedTracer {
    /// The underlying tracer
    public let tracer: Tracer

    /// The sharding specification (nil = infer from context)
    public let sharding: SDYSharding?

    /// Creates a sharded tracer.
    public init(tracer: Tracer, sharding: SDYSharding? = nil) {
        self.tracer = tracer
        self.sharding = sharding
    }

    /// Forward common Tracer properties
    public var shape: TensorShape { tracer.shape }
    public var dtype: DType { tracer.dtype }
    public var valueId: UInt64 { tracer.valueId }

    /// Create with explicit shape and sharding
    public init(
        shape: TensorShape,
        dtype: DType = .float32,
        sharding: SDYSharding? = nil
    ) {
        self.tracer = Tracer(shape: shape, dtype: dtype)
        self.sharding = sharding
    }
}

// MARK: - Sharded Tracing Context

/// Context for tracing computations with sharding annotations.
///
/// Example:
/// ```swift
/// let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
/// let ctx = ShardedTracingContext(mesh: mesh)
///
/// let x = ctx.input(shape: TensorShape([16, 64]),
///                   sharding: .dataParallel(mesh: mesh, rank: 2))
/// let w = ctx.input(shape: TensorShape([64, 32]),
///                   sharding: .replicated(mesh: mesh, rank: 2))
/// let y = x.tracer.matmul(w.tracer)
/// ctx.output(ShardedTracer(tracer: y, sharding: xSharding))
///
/// let mlir = ctx.buildShardedModule()
/// ```
public class ShardedTracingContext {
    /// The device mesh for this context
    public let mesh: SDYMesh

    /// Input shardings by argument name
    private var inputShardings: [String: SDYSharding] = [:]

    /// Input shapes by argument name
    private var inputShapes: [String: TensorShape] = [:]

    /// Input dtypes by argument name
    private var inputDtypes: [String: DType] = [:]

    /// Output shardings
    private var outputShardings: [SDYSharding] = []

    /// Output tracers
    private var outputs: [Tracer] = []

    /// Argument counter
    private var argumentCount: Int = 0

    /// Creates a sharded tracing context.
    public init(mesh: SDYMesh) {
        self.mesh = mesh
        // Reset the graph builder for fresh tracing
        TracerGraphBuilder.shared.reset()
    }

    /// Create a sharded symbolic input.
    public func input(
        shape: TensorShape,
        dtype: DType = .float32,
        name: String? = nil,
        sharding: SDYSharding? = nil
    ) -> ShardedTracer {
        let argName = name ?? "%arg\(argumentCount)"
        argumentCount += 1

        // Store input metadata
        inputShapes[argName] = shape
        inputDtypes[argName] = dtype
        if let sharding = sharding {
            inputShardings[argName] = sharding
        }

        // Create the symbolic tracer
        let tracer = Tracer(shape: shape, dtype: dtype)

        return ShardedTracer(tracer: tracer, sharding: sharding)
    }

    /// Set sharded outputs.
    public func output(_ shardedTracers: ShardedTracer...) {
        for st in shardedTracers {
            outputs.append(st.tracer)
            if let sharding = st.sharding {
                outputShardings.append(sharding)
            }
        }
    }

    /// Set output with explicit sharding.
    public func output(_ tracer: Tracer, sharding: SDYSharding? = nil) {
        outputs.append(tracer)
        if let sharding = sharding {
            outputShardings.append(sharding)
        }
    }

    /// Build the MLIR module with SDY sharding annotations.
    ///
    /// Generates complete MLIR including mesh definition, function signature
    /// with sharding annotations, and all traced operations.
    public func buildShardedModule(name: String = "sharded") -> String {
        var text = "module @\(name) {\n"

        // Add mesh definition
        text += "  \(mesh.mlirText)\n\n"

        // Build function signature with sharding attributes
        text += "  func.func @main("
        text += buildFunctionArguments()
        text += ")"

        // Add return type if there are outputs
        if !outputs.isEmpty {
            let resultTypes = outputs.map { tensorType(for: $0.shape, dtype: $0.dtype) }
            if resultTypes.count == 1 {
                text += " -> \(resultTypes[0])"
            } else {
                text += " -> (\(resultTypes.joined(separator: ", ")))"
            }
        }

        text += " {\n"

        // Add operations from TracerGraphBuilder
        text += buildOperations()

        // Add return statement
        if !outputs.isEmpty {
            let resultNames = outputs.map { "%v\($0.valueId)" }
            let resultTypes = outputs.map { tensorType(for: $0.shape, dtype: $0.dtype) }
            text += "    return \(resultNames.joined(separator: ", ")) : \(resultTypes.joined(separator: ", "))\n"
        }

        text += "  }\n"
        text += "}\n"

        return text
    }

    /// Build function arguments with sharding annotations
    private func buildFunctionArguments() -> String {
        var args: [String] = []

        // Sort by argument number
        let sortedArgs = inputShapes.keys.sorted { a, b in
            let numA = Int(a.replacingOccurrences(of: "%arg", with: "")) ?? 0
            let numB = Int(b.replacingOccurrences(of: "%arg", with: "")) ?? 0
            return numA < numB
        }

        for argName in sortedArgs {
            guard let shape = inputShapes[argName],
                  let dtype = inputDtypes[argName] else { continue }

            let tensorTypeStr = tensorType(for: shape, dtype: dtype)
            var argStr = "\(argName): \(tensorTypeStr)"

            // Add sharding attribute if present
            if let sharding = inputShardings[argName] {
                argStr += " {sdy.sharding = \(sharding.mlirAttributeText)}"
            }

            args.append(argStr)
        }

        return args.joined(separator: ", ")
    }

    /// Build operations from TracerGraphBuilder
    private func buildOperations() -> String {
        var text = ""

        let operations = TracerGraphBuilder.shared.getOperations()
        for op in operations {
            text += emitOperation(op)
        }

        return text
    }

    /// Emit a single traced operation as StableHLO MLIR
    private func emitOperation(_ op: TracedOperation) -> String {
        switch op {
        case .constant(let id, let value, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            return "    %v\(id) = stablehlo.constant dense<\(formatValue(value, dtype: dtype))> : \(type)\n"

        case .placeholder:
            // Placeholders are function arguments, handled separately
            return ""

        case .binary(let id, let opType, let lhs, let rhs, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let opName = stablehloOpName(for: opType)
            if opType == .matmul {
                // Matmul uses stablehlo.dot
                return "    %v\(id) = stablehlo.dot %v\(lhs), %v\(rhs) : \(type)\n"
            }
            return "    %v\(id) = \(opName) %v\(lhs), %v\(rhs) : \(type)\n"

        case .unary(let id, let opType, let input, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let opName = stablehloUnaryOpName(for: opType)
            if opType == .relu {
                // ReLU is max(x, 0)
                return "    %zero\(id) = stablehlo.constant dense<0.0> : \(type)\n" +
                       "    %v\(id) = stablehlo.maximum %v\(input), %zero\(id) : \(type)\n"
            }
            return "    %v\(id) = \(opName) %v\(input) : \(type)\n"

        case .reduction(let id, _, let input, let axes, _, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let axesStr = "[\(axes.sorted().map { String($0) }.joined(separator: ", "))]"
            return "    %v\(id) = stablehlo.reduce(%v\(input)) across dimensions = \(axesStr) : \(type)\n"

        case .print(let id, let input, let label, _, _, _, _):
            return "    // print(\"\(label)\")\n    %v\(id) = %v\(input)\n"

        case .power(let id, let base, let exponent, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            return "    %exp\(id) = stablehlo.constant dense<\(formatValue(exponent, dtype: dtype))> : \(type)\n" +
                   "    %v\(id) = stablehlo.power %v\(base), %exp\(id) : \(type)\n"
        }
    }

    /// Format value for MLIR constant
    private func formatValue(_ value: Double, dtype: DType) -> String {
        if dtype.isFloatingPoint {
            if value == 0.0 {
                return "0.000000e+00"
            } else if value == 1.0 {
                return "1.000000e+00"
            } else {
                return String(format: "%e", value)
            }
        } else {
            return String(Int(value))
        }
    }

    /// Get StableHLO operation name for binary ops
    private func stablehloOpName(for op: BinaryOperation) -> String {
        switch op {
        case .add: return "stablehlo.add"
        case .subtract: return "stablehlo.subtract"
        case .multiply: return "stablehlo.multiply"
        case .divide: return "stablehlo.divide"
        case .matmul: return "stablehlo.dot"
        }
    }

    /// Get StableHLO operation name for unary ops
    private func stablehloUnaryOpName(for op: UnaryOperation) -> String {
        switch op {
        case .exp: return "stablehlo.exponential"
        case .log: return "stablehlo.log"
        case .sqrt: return "stablehlo.sqrt"
        case .abs: return "stablehlo.abs"
        case .neg: return "stablehlo.negate"
        case .sin: return "stablehlo.sine"
        case .cos: return "stablehlo.cosine"
        case .tan: return "stablehlo.tan"
        case .tanh: return "stablehlo.tanh"
        case .sigmoid: return "stablehlo.logistic"
        case .relu: return "stablehlo.maximum"  // handled specially
        case .reshape: return "stablehlo.reshape"
        case .transpose: return "stablehlo.transpose"
        }
    }

    /// Format tensor type string
    private func tensorType(for shape: TensorShape, dtype: DType) -> String {
        // DType rawValue gives us the MLIR type string directly (f32, f64, i32, etc.)
        let dtypeStr = dtype.rawValue

        if shape.rank == 0 {
            // Scalar tensor
            return "tensor<\(dtypeStr)>"
        } else {
            let dims = shape.dimensions.compactMap { $0 }.map { String($0) }.joined(separator: "x")
            return "tensor<\(dims)x\(dtypeStr)>"
        }
    }
}

// MARK: - Sharded Function Compiler

/// Compiles functions with sharding annotations.
public class ShardedFunctionCompiler {
    public init() {}

    /// Compile a sharded function.
    public func compile(
        mesh: SDYMesh,
        inputSpecs: [(shape: TensorShape, dtype: DType, sharding: SDYSharding?)],
        outputSharding: SDYSharding? = nil,
        _ function: ([Tracer]) -> Tracer
    ) -> String {
        let context = ShardedTracingContext(mesh: mesh)

        // Create symbolic inputs with shardings
        let inputs = inputSpecs.map { spec in
            context.input(
                shape: spec.shape,
                dtype: spec.dtype,
                sharding: spec.sharding
            )
        }

        // Trace the function
        let output = function(inputs.map { $0.tracer })

        // Set output with sharding
        context.output(output, sharding: outputSharding)

        // Build sharded MLIR
        return context.buildShardedModule()
    }
}

// MARK: - Convenience Functions

/// Trace and compile a sharded function.
public func traceSharded(
    mesh: SDYMesh,
    inputSpecs: [(shape: TensorShape, dtype: DType, sharding: SDYSharding?)],
    outputSharding: SDYSharding? = nil,
    _ function: ([Tracer]) -> Tracer
) -> String {
    let compiler = ShardedFunctionCompiler()
    return compiler.compile(
        mesh: mesh,
        inputSpecs: inputSpecs,
        outputSharding: outputSharding,
        function
    )
}

/// Simple data-parallel compilation helper.
public func traceDataParallel(
    numDevices: Int,
    inputShapes: [TensorShape],
    dtype: DType = .float32,
    _ function: ([Tracer]) -> Tracer
) -> String {
    let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: numDevices)

    let inputSpecs = inputShapes.map { shape in
        (
            shape: shape,
            dtype: dtype,
            sharding: SDYSharding.dataParallel(mesh: mesh, rank: shape.rank, batchAxis: "batch")
        )
    }

    return traceSharded(
        mesh: mesh,
        inputSpecs: inputSpecs,
        outputSharding: inputSpecs.first?.sharding,
        function
    )
}

// MARK: - Bridge from DeviceMesh/ShardingSpec

extension DeviceMesh {
    /// Convert to SDY mesh.
    public func toSDYMesh() -> SDYMesh {
        let axes = zip(axisNames, shape).map { (name: $0, size: $1) }
        return SDYMesh(name: name, axes: axes)
    }
}

extension ShardingSpec {
    /// Convert to SDY sharding.
    public func toSDYSharding() -> SDYSharding {
        SDYSharding(mesh: mesh.name, axes: dimMapping)
    }
}

// MARK: - Mesh Visualization

extension SDYMesh {
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

extension SDYSharding {
    /// Print sharding information.
    public func printInfo(shape: TensorShape? = nil) {
        print("SDY Tensor Sharding")
        print("  Mesh: \(meshName)")
        print("  Dimensions:")
        for (i, axis) in dimAxes.enumerated() {
            let dimSize = shape?.dimensions[i].map { String($0) } ?? "?"
            let axisStr = axis ?? "replicated"
            print("    dim[\(i)] (size \(dimSize)): \(axisStr)")
        }
        print("  MLIR: \(mlirAttributeText)")
    }
}
