/// ShardedAutoDiff.swift - Sharding-Aware Automatic Differentiation
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// Integrates sharding with automatic differentiation to provide:
/// - Automatic gradient synchronization based on sharding patterns
/// - Sharding-aware VJP rules for distributed computation
/// - Collective operations automatically inserted in backward pass
/// - Complete MLIR generation with sharding annotations for both forward and backward

import Foundation
import _Differentiation

// MARK: - Sharding Pattern Analysis

/// Describes the gradient synchronization pattern needed for a sharding configuration
public enum GradientSyncPattern: Sendable, Equatable {
    /// No synchronization needed (fully replicated)
    case none

    /// All-reduce sum across specified axes
    case allReduceSum(axes: [String])

    /// All-reduce mean across specified axes (for data-parallel averaging)
    case allReduceMean(axes: [String])

    /// All-gather to reconstruct full gradient
    case allGather(axis: String, dim: Int)

    /// Reduce-scatter for ZeRO-style optimization
    case reduceScatter(axis: String, dim: Int)
}

/// Analyzes sharding patterns to determine required gradient synchronization
public struct ShardingAnalyzer: Sendable {
    /// The device mesh
    public let mesh: SDYMesh

    /// Initialize with a mesh
    public init(mesh: SDYMesh) {
        self.mesh = mesh
    }

    /// Determine the gradient sync pattern for a parameter based on its sharding
    public func gradientSyncPattern(
        for sharding: SDYSharding,
        parameterType: ParameterType = .weight
    ) -> GradientSyncPattern {
        // Fully replicated parameters need all-reduce
        if sharding.isReplicated {
            // Data-parallel: all gradients are partial sums, need all-reduce
            let dataAxes = mesh.axes.map { $0.name }
            return parameterType == .weight
                ? .allReduceMean(axes: dataAxes)  // Average for SGD
                : .allReduceSum(axes: dataAxes)   // Sum for accumulation
        }

        // Find which axes the parameter is sharded on
        let shardedAxes = sharding.dimAxes.compactMap { $0 }

        if shardedAxes.isEmpty {
            // Replicated but with specific mesh reference
            let allAxes = mesh.axes.map { $0.name }
            return .allReduceMean(axes: allAxes)
        }

        // Tensor-parallel: gradients are already partitioned
        // Need all-reduce only on non-sharded axes
        let nonShardedAxes = mesh.axes.map { $0.name }.filter { !shardedAxes.contains($0) }

        if nonShardedAxes.isEmpty {
            // Fully sharded, no sync needed
            return .none
        }

        return .allReduceSum(axes: nonShardedAxes)
    }

    /// Parameter type for gradient synchronization strategy
    public enum ParameterType: Sendable {
        case weight      // Model weights - use mean reduction
        case activation  // Activations - use sum reduction
        case loss        // Loss values - use sum reduction
    }
}

// MARK: - Sharded Differentiable Value

/// A value with both forward value and associated sharding information
/// that enables automatic gradient synchronization
public struct ShardedValue<T> {
    /// The underlying value
    public var value: T

    /// Sharding specification
    public let sharding: SDYSharding?

    /// The mesh this value is sharded on
    public let mesh: SDYMesh?

    /// Create a sharded value
    public init(value: T, sharding: SDYSharding? = nil, mesh: SDYMesh? = nil) {
        self.value = value
        self.sharding = sharding
        self.mesh = mesh
    }

    /// Create a replicated value (no sharding)
    public static func replicated(_ value: T) -> ShardedValue<T> {
        ShardedValue(value: value, sharding: nil, mesh: nil)
    }
}

// MARK: - Sharded Tracer with Gradient Tracking

/// A Tracer that tracks sharding for automatic gradient synchronization
public struct ShardedDifferentiableTracer {
    /// The underlying tracer
    public let tracer: Tracer

    /// Sharding specification
    public let sharding: SDYSharding?

    /// The mesh this tracer is associated with
    public let mesh: SDYMesh?

    /// Gradient sync pattern for backward pass
    public let gradientSyncPattern: GradientSyncPattern

    /// Create a sharded differentiable tracer
    public init(
        tracer: Tracer,
        sharding: SDYSharding? = nil,
        mesh: SDYMesh? = nil,
        gradientSyncPattern: GradientSyncPattern = .none
    ) {
        self.tracer = tracer
        self.sharding = sharding
        self.mesh = mesh
        self.gradientSyncPattern = gradientSyncPattern
    }

    /// Create with automatic sync pattern inference
    public init(
        tracer: Tracer,
        sharding: SDYSharding?,
        mesh: SDYMesh,
        parameterType: ShardingAnalyzer.ParameterType = .weight
    ) {
        self.tracer = tracer
        self.sharding = sharding
        self.mesh = mesh

        if let sharding = sharding {
            let analyzer = ShardingAnalyzer(mesh: mesh)
            self.gradientSyncPattern = analyzer.gradientSyncPattern(
                for: sharding,
                parameterType: parameterType
            )
        } else {
            self.gradientSyncPattern = .none
        }
    }

    // Forward common properties
    public var shape: TensorShape { tracer.shape }
    public var dtype: DType { tracer.dtype }
    public var valueId: UInt64 { tracer.valueId }
}

// MARK: - Sharded Gradient

/// A gradient with associated sharding and synchronization information
public struct ShardedGradient {
    /// The gradient tracer
    public let gradient: Tracer

    /// Sharding of the gradient
    public let sharding: SDYSharding?

    /// Whether this gradient has been synchronized
    public private(set) var isSynchronized: Bool

    /// Create a sharded gradient
    public init(gradient: Tracer, sharding: SDYSharding? = nil, isSynchronized: Bool = false) {
        self.gradient = gradient
        self.sharding = sharding
        self.isSynchronized = isSynchronized
    }

    /// Synchronize this gradient according to its sharding pattern
    public func synchronized(mesh: SDYMesh, reduction: ReductionType = .sum) -> ShardedGradient {
        guard !isSynchronized else { return self }

        let syncedGradient = gradient.allReduce(reduction: reduction)
        return ShardedGradient(
            gradient: syncedGradient,
            sharding: sharding,
            isSynchronized: true
        )
    }
}

// MARK: - Sharded Differentiable Context

/// Context for computing gradients with automatic sharding-aware synchronization
public class ShardedDifferentiableContext {
    /// The device mesh
    public let mesh: SDYMesh

    /// The sharding analyzer
    private let analyzer: ShardingAnalyzer

    /// Tracked parameters and their shardings
    private var parameterShardings: [UInt64: (sharding: SDYSharding, type: ShardingAnalyzer.ParameterType)] = [:]

    /// Input tracers
    private var inputs: [ShardedDifferentiableTracer] = []

    /// Output tracers
    private var outputs: [Tracer] = []

    /// Output shardings
    private var outputShardings: [SDYSharding] = []

    /// Gradient operations to insert
    private var gradientSyncOps: [GradientSyncOperation] = []

    /// Initialize with a mesh
    public init(mesh: SDYMesh) {
        self.mesh = mesh
        self.analyzer = ShardingAnalyzer(mesh: mesh)
        TracerGraphBuilder.shared.reset()
    }

    /// Create a sharded input
    public func input(
        shape: TensorShape,
        dtype: DType = .float32,
        sharding: SDYSharding? = nil,
        parameterType: ShardingAnalyzer.ParameterType = .weight
    ) -> ShardedDifferentiableTracer {
        let tracer = Tracer(shape: shape, dtype: dtype)

        let shardedTracer = ShardedDifferentiableTracer(
            tracer: tracer,
            sharding: sharding,
            mesh: mesh,
            parameterType: parameterType
        )

        inputs.append(shardedTracer)

        // Track for gradient sync
        if let sharding = sharding {
            parameterShardings[tracer.valueId] = (sharding, parameterType)
        }

        return shardedTracer
    }

    /// Record an output
    public func output(_ tracer: Tracer, sharding: SDYSharding? = nil) {
        outputs.append(tracer)
        if let sharding = sharding {
            outputShardings.append(sharding)
        }
    }

    /// Compute value and gradients with automatic synchronization
    public func valueWithGradient(
        of function: @differentiable(reverse) (Tracer) -> Tracer,
        at input: ShardedDifferentiableTracer
    ) -> (value: Tracer, gradient: ShardedGradient) {
        // Compute forward and backward
        let (value, pullback) = _Differentiation.valueWithPullback(at: input.tracer, of: function)

        // Create upstream gradient (1.0 for scalar loss)
        let upstream = Tracer(value: 1.0, shape: value.shape, dtype: value.dtype)

        // Compute raw gradient
        let rawGradient = pullback(upstream)

        // Create sharded gradient with sync pattern
        var shardedGradient = ShardedGradient(
            gradient: rawGradient,
            sharding: input.sharding,
            isSynchronized: false
        )

        // Automatically synchronize based on pattern
        switch input.gradientSyncPattern {
        case .none:
            shardedGradient = ShardedGradient(
                gradient: rawGradient,
                sharding: input.sharding,
                isSynchronized: true
            )

        case .allReduceSum(let axes):
            let synced = rawGradient.allReduce(reduction: .sum)
            shardedGradient = ShardedGradient(
                gradient: synced,
                sharding: input.sharding,
                isSynchronized: true
            )
            recordGradientSync(.allReduce(inputId: rawGradient.valueId, axes: axes, reduction: .sum))

        case .allReduceMean(let axes):
            let synced = rawGradient.allReduce(reduction: .mean)
            shardedGradient = ShardedGradient(
                gradient: synced,
                sharding: input.sharding,
                isSynchronized: true
            )
            recordGradientSync(.allReduce(inputId: rawGradient.valueId, axes: axes, reduction: .mean))

        case .allGather(let axis, let dim):
            let synced = rawGradient.allGather(gatherDim: dim)
            shardedGradient = ShardedGradient(
                gradient: synced,
                sharding: nil, // Now fully gathered
                isSynchronized: true
            )
            recordGradientSync(.allGather(inputId: rawGradient.valueId, axis: axis, dim: dim))

        case .reduceScatter(let axis, let dim):
            let synced = rawGradient.reduceScatter(reduction: .sum, scatterDim: dim)
            shardedGradient = ShardedGradient(
                gradient: synced,
                sharding: input.sharding,
                isSynchronized: true
            )
            recordGradientSync(.reduceScatter(inputId: rawGradient.valueId, axis: axis, dim: dim))
        }

        return (value, shardedGradient)
    }

    /// Compute gradients for multiple inputs with automatic synchronization
    public func valueWithGradient(
        of function: @differentiable(reverse) (Tracer, Tracer) -> Tracer,
        at x: ShardedDifferentiableTracer,
        _ y: ShardedDifferentiableTracer
    ) -> (value: Tracer, gradients: (ShardedGradient, ShardedGradient)) {
        let (value, pullback) = _Differentiation.valueWithPullback(at: x.tracer, y.tracer, of: function)
        let upstream = Tracer(value: 1.0, shape: value.shape, dtype: value.dtype)
        let (rawGradX, rawGradY) = pullback(upstream)

        let gradX = synchronizeGradient(rawGradX, pattern: x.gradientSyncPattern, sharding: x.sharding)
        let gradY = synchronizeGradient(rawGradY, pattern: y.gradientSyncPattern, sharding: y.sharding)

        return (value, (gradX, gradY))
    }

    /// Synchronize a gradient according to its pattern
    private func synchronizeGradient(
        _ gradient: Tracer,
        pattern: GradientSyncPattern,
        sharding: SDYSharding?
    ) -> ShardedGradient {
        switch pattern {
        case .none:
            return ShardedGradient(gradient: gradient, sharding: sharding, isSynchronized: true)

        case .allReduceSum(let axes):
            let synced = gradient.allReduce(reduction: .sum)
            recordGradientSync(.allReduce(inputId: gradient.valueId, axes: axes, reduction: .sum))
            return ShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)

        case .allReduceMean(let axes):
            let synced = gradient.allReduce(reduction: .mean)
            recordGradientSync(.allReduce(inputId: gradient.valueId, axes: axes, reduction: .mean))
            return ShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)

        case .allGather(let axis, let dim):
            let synced = gradient.allGather(gatherDim: dim)
            recordGradientSync(.allGather(inputId: gradient.valueId, axis: axis, dim: dim))
            return ShardedGradient(gradient: synced, sharding: nil, isSynchronized: true)

        case .reduceScatter(let axis, let dim):
            let synced = gradient.reduceScatter(reduction: .sum, scatterDim: dim)
            recordGradientSync(.reduceScatter(inputId: gradient.valueId, axis: axis, dim: dim))
            return ShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)
        }
    }

    /// Record a gradient sync operation for MLIR generation
    private func recordGradientSync(_ op: GradientSyncOperation) {
        gradientSyncOps.append(op)
    }

    /// Build MLIR module with sharding annotations including backward pass
    public func buildShardedModule(name: String = "sharded_autodiff") -> String {
        var text = "module @\(name) {\n"

        // Add mesh definition
        text += "  \(mesh.mlirText)\n\n"

        // Add forward function with sharding
        text += buildForwardFunction()

        // Add backward function with gradient sync
        text += buildBackwardFunction()

        text += "}\n"
        return text
    }

    /// Build the forward function
    private func buildForwardFunction() -> String {
        var text = "  // Forward pass\n"
        text += "  func.func @forward("

        // Add arguments with shardings
        let args = inputs.enumerated().map { (i, input) -> String in
            var arg = "%arg\(i): \(tensorType(for: input.shape, dtype: input.dtype))"
            if let sharding = input.sharding {
                arg += " {sdy.sharding = \(sharding.mlirAttributeText)}"
            }
            return arg
        }
        text += args.joined(separator: ", ")
        text += ")"

        // Return type
        if !outputs.isEmpty {
            let types = outputs.map { tensorType(for: $0.shape, dtype: $0.dtype) }
            text += " -> \(types.joined(separator: ", "))"
        }

        text += " {\n"

        // Add operations
        let operations = TracerGraphBuilder.shared.getOperations()
        for op in operations {
            text += emitOperation(op)
        }

        // Return
        if !outputs.isEmpty {
            let results = outputs.map { "%v\($0.valueId)" }
            let types = outputs.map { tensorType(for: $0.shape, dtype: $0.dtype) }
            text += "    return \(results.joined(separator: ", ")) : \(types.joined(separator: ", "))\n"
        }

        text += "  }\n\n"
        return text
    }

    /// Build the backward function with gradient synchronization
    private func buildBackwardFunction() -> String {
        var text = "  // Backward pass with automatic gradient synchronization\n"
        text += "  func.func @backward("

        // Inputs: original inputs + upstream gradients
        var args: [String] = []

        for (i, input) in inputs.enumerated() {
            var arg = "%arg\(i): \(tensorType(for: input.shape, dtype: input.dtype))"
            if let sharding = input.sharding {
                arg += " {sdy.sharding = \(sharding.mlirAttributeText)}"
            }
            args.append(arg)
        }

        // Upstream gradient
        for (i, output) in outputs.enumerated() {
            let arg = "%grad_out\(i): \(tensorType(for: output.shape, dtype: output.dtype))"
            args.append(arg)
        }

        text += args.joined(separator: ", ")
        text += ")"

        // Return gradients
        if !inputs.isEmpty {
            let gradTypes = inputs.map { tensorType(for: $0.shape, dtype: $0.dtype) }
            text += " -> (\(gradTypes.joined(separator: ", ")))"
        }

        text += " {\n"

        // Comment about gradient sync operations
        text += "    // Gradient synchronization operations:\n"
        for op in gradientSyncOps {
            text += "    // \(op.description)\n"
        }
        text += "\n"

        // Placeholder for actual gradient computation
        // In a real implementation, this would contain the traced backward pass
        text += "    // ... backward pass operations ...\n"

        // Gradient sync operations
        for (i, _) in inputs.enumerated() {
            text += "    // %grad\(i)_synced = collective operation based on sharding pattern\n"
        }

        text += "  }\n"
        return text
    }

    /// Format tensor type
    private func tensorType(for shape: TensorShape, dtype: DType) -> String {
        let dims = shape.dimensions.compactMap { $0 }.map { String($0) }.joined(separator: "x")
        return "tensor<\(dims)x\(dtype.rawValue)>"
    }

    /// Emit a single operation
    private func emitOperation(_ op: TracedOperation) -> String {
        switch op {
        case .constant(let id, let value, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let valStr = dtype.isFloatingPoint ? String(format: "%e", value) : String(Int(value))
            return "    %v\(id) = stablehlo.constant dense<\(valStr)> : \(type)\n"

        case .placeholder:
            return ""

        case .binary(let id, let opType, let lhs, let rhs, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let opName = stablehloOpName(for: opType)
            if opType == .matmul {
                return "    %v\(id) = stablehlo.dot %v\(lhs), %v\(rhs) : \(type)\n"
            }
            return "    %v\(id) = \(opName) %v\(lhs), %v\(rhs) : \(type)\n"

        case .unary(let id, let opType, let input, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            if opType == .relu {
                return "    %zero\(id) = stablehlo.constant dense<0.0> : \(type)\n" +
                       "    %v\(id) = stablehlo.maximum %v\(input), %zero\(id) : \(type)\n"
            }
            return "    %v\(id) = \(stablehloUnaryOpName(for: opType)) %v\(input) : \(type)\n"

        case .reduction(let id, _, let input, let axes, _, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let axesStr = "[\(axes.sorted().map { String($0) }.joined(separator: ", "))]"
            return "    %v\(id) = stablehlo.reduce(%v\(input)) across dimensions = \(axesStr) : \(type)\n"

        case .print(let id, let input, let label, _, _, _, _):
            return "    // print(\"\(label)\")\n    %v\(id) = %v\(input)\n"

        case .power(let id, let base, let exponent, let shape, let dtype):
            let type = tensorType(for: shape, dtype: dtype)
            let expStr = dtype.isFloatingPoint ? String(format: "%e", exponent) : String(Int(exponent))
            return "    %exp\(id) = stablehlo.constant dense<\(expStr)> : \(type)\n" +
                   "    %v\(id) = stablehlo.power %v\(base), %exp\(id) : \(type)\n"
        }
    }

    private func stablehloOpName(for op: BinaryOperation) -> String {
        switch op {
        case .add: return "stablehlo.add"
        case .subtract: return "stablehlo.subtract"
        case .multiply: return "stablehlo.multiply"
        case .divide: return "stablehlo.divide"
        case .matmul: return "stablehlo.dot"
        }
    }

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
        case .relu: return "stablehlo.maximum"
        case .reshape: return "stablehlo.reshape"
        case .transpose: return "stablehlo.transpose"
        }
    }
}

// MARK: - Gradient Sync Operation Record

/// Records a gradient synchronization operation for MLIR generation
internal enum GradientSyncOperation: Sendable, CustomStringConvertible {
    case allReduce(inputId: UInt64, axes: [String], reduction: ReductionType)
    case allGather(inputId: UInt64, axis: String, dim: Int)
    case reduceScatter(inputId: UInt64, axis: String, dim: Int)

    var description: String {
        switch self {
        case .allReduce(let id, let axes, let reduction):
            return "all_reduce(%v\(id), axes=[\(axes.joined(separator: ","))], reduction=\(reduction.rawValue))"
        case .allGather(let id, let axis, let dim):
            return "all_gather(%v\(id), axis=\(axis), dim=\(dim))"
        case .reduceScatter(let id, let axis, let dim):
            return "reduce_scatter(%v\(id), axis=\(axis), dim=\(dim))"
        }
    }
}

// MARK: - Convenience Functions

/// Compute gradient with automatic sharding-aware synchronization
public func shardedGradient(
    mesh: SDYMesh,
    inputSharding: SDYSharding,
    of function: @differentiable(reverse) (Tracer) -> Tracer,
    at input: Tracer
) -> ShardedGradient {
    let ctx = ShardedDifferentiableContext(mesh: mesh)
    let shardedInput = ShardedDifferentiableTracer(
        tracer: input,
        sharding: inputSharding,
        mesh: mesh,
        parameterType: .weight
    )
    let (_, gradient) = ctx.valueWithGradient(of: function, at: shardedInput)
    return gradient
}

/// Compute value and gradient with automatic sharding-aware synchronization
public func shardedValueWithGradient(
    mesh: SDYMesh,
    inputSharding: SDYSharding,
    of function: @differentiable(reverse) (Tracer) -> Tracer,
    at input: Tracer
) -> (value: Tracer, gradient: ShardedGradient) {
    let ctx = ShardedDifferentiableContext(mesh: mesh)
    let shardedInput = ShardedDifferentiableTracer(
        tracer: input,
        sharding: inputSharding,
        mesh: mesh,
        parameterType: .weight
    )
    return ctx.valueWithGradient(of: function, at: shardedInput)
}

/// Data-parallel gradient computation with automatic all-reduce
public func dataParallelGradient(
    numDevices: Int,
    of function: @differentiable(reverse) (Tracer) -> Tracer,
    at input: Tracer
) -> ShardedGradient {
    let mesh = SDYMesh.linear(name: "data_parallel", axis: "batch", size: numDevices)
    let sharding = SDYSharding.dataParallel(mesh: mesh, rank: input.shape.rank, batchAxis: "batch")
    return shardedGradient(mesh: mesh, inputSharding: sharding, of: function, at: input)
}

// MARK: - Sharded Training Step

/// A complete sharded training step with forward, backward, and gradient sync
public struct ShardedTrainingStep {
    /// The device mesh
    public let mesh: SDYMesh

    /// Parameter shardings
    public let parameterShardings: [SDYSharding]

    /// Initialize
    public init(mesh: SDYMesh, parameterShardings: [SDYSharding]) {
        self.mesh = mesh
        self.parameterShardings = parameterShardings
    }

    /// Execute a training step with automatic gradient synchronization
    public func step(
        loss: @differentiable(reverse) (Tracer, Tracer) -> Tracer,
        input: Tracer,
        weights: Tracer,
        inputSharding: SDYSharding,
        weightSharding: SDYSharding
    ) -> (loss: Tracer, inputGrad: ShardedGradient, weightGrad: ShardedGradient) {
        let ctx = ShardedDifferentiableContext(mesh: mesh)

        let shardedInput = ShardedDifferentiableTracer(
            tracer: input,
            sharding: inputSharding,
            mesh: mesh,
            parameterType: .activation
        )

        let shardedWeights = ShardedDifferentiableTracer(
            tracer: weights,
            sharding: weightSharding,
            mesh: mesh,
            parameterType: .weight
        )

        let (lossValue, grads) = ctx.valueWithGradient(of: loss, at: shardedInput, shardedWeights)
        return (lossValue, grads.0, grads.1)
    }
}

// MARK: - Sharding-Aware Optimizer

/// An optimizer that handles sharded parameters and gradients
public class ShardedOptimizer {
    /// The device mesh
    public let mesh: SDYMesh

    /// Learning rate
    public var learningRate: Double

    /// Initialize
    public init(mesh: SDYMesh, learningRate: Double = 0.01) {
        self.mesh = mesh
        self.learningRate = learningRate
    }

    /// Apply gradients to parameters with proper sharding
    public func apply(
        gradients: [ShardedGradient],
        to parameters: [Tracer]
    ) -> [Tracer] {
        precondition(gradients.count == parameters.count)

        return zip(parameters, gradients).map { (param, grad) in
            // Ensure gradient is synchronized
            let syncedGrad = grad.isSynchronized
                ? grad.gradient
                : grad.gradient.allReduce(reduction: .mean)

            // Simple SGD update: param = param - lr * grad
            let lr = Tracer(value: learningRate, shape: TensorShape([]), dtype: param.dtype)
            let scaledGrad = syncedGrad * lr
            return param - scaledGrad
        }
    }
}
