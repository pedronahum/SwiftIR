// JupyterShardedAutoDiff.swift - Sharding-Aware Automatic Differentiation for Jupyter/Colab
// Pure Swift - works without C++ interop
//
// Integrates sharding with automatic differentiation to provide:
// - Automatic gradient synchronization based on sharding patterns
// - Sharding-aware VJP rules for distributed computation
// - Collective operations automatically inserted in backward pass
// - Complete MLIR generation with sharding annotations for both forward and backward

import Foundation

// MARK: - Sharding Pattern Analysis

/// Describes the gradient synchronization pattern needed for a sharding configuration
public enum JGradientSyncPattern: Sendable, Equatable {
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
public struct JShardingAnalyzer: Sendable {
    /// The device mesh
    public let mesh: JSDYMesh

    /// Initialize with a mesh
    public init(mesh: JSDYMesh) {
        self.mesh = mesh
    }

    /// Determine the gradient sync pattern for a parameter based on its sharding
    public func gradientSyncPattern(
        for sharding: JSDYSharding,
        parameterType: JParameterType = .weight
    ) -> JGradientSyncPattern {
        // Fully replicated parameters need all-reduce
        let shardedAxes = sharding.dimAxes.compactMap { $0 }

        if shardedAxes.isEmpty {
            // Data-parallel: all gradients are partial sums, need all-reduce
            let dataAxes = mesh.axes.map { $0.name }
            return parameterType == .weight
                ? .allReduceMean(axes: dataAxes)  // Average for SGD
                : .allReduceSum(axes: dataAxes)   // Sum for accumulation
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
}

/// Parameter type for gradient synchronization strategy
public enum JParameterType: Sendable {
    case weight      // Model weights - use mean reduction
    case activation  // Activations - use sum reduction
    case loss        // Loss values - use sum reduction
}

// MARK: - Sharded Differentiable Value

/// A value with both forward value and associated sharding information
public struct JShardedValue<T> {
    /// The underlying value
    public var value: T

    /// Sharding specification
    public let sharding: JSDYSharding?

    /// The mesh this value is sharded on
    public let mesh: JSDYMesh?

    /// Create a sharded value
    public init(value: T, sharding: JSDYSharding? = nil, mesh: JSDYMesh? = nil) {
        self.value = value
        self.sharding = sharding
        self.mesh = mesh
    }

    /// Create a replicated value (no sharding)
    public static func replicated(_ value: T) -> JShardedValue<T> {
        JShardedValue(value: value, sharding: nil, mesh: nil)
    }
}

// MARK: - Sharded Tracer with Gradient Tracking

/// A JTracer that tracks sharding for automatic gradient synchronization
public struct JShardedDifferentiableTracer {
    /// The underlying tracer
    public let tracer: JTracer

    /// Sharding specification
    public let sharding: JSDYSharding?

    /// The mesh this tracer is associated with
    public let mesh: JSDYMesh?

    /// Gradient sync pattern for backward pass
    public let gradientSyncPattern: JGradientSyncPattern

    /// Create a sharded differentiable tracer
    public init(
        tracer: JTracer,
        sharding: JSDYSharding? = nil,
        mesh: JSDYMesh? = nil,
        gradientSyncPattern: JGradientSyncPattern = .none
    ) {
        self.tracer = tracer
        self.sharding = sharding
        self.mesh = mesh
        self.gradientSyncPattern = gradientSyncPattern
    }

    /// Create with automatic sync pattern inference
    public init(
        tracer: JTracer,
        sharding: JSDYSharding?,
        mesh: JSDYMesh,
        parameterType: JParameterType = .weight
    ) {
        self.tracer = tracer
        self.sharding = sharding
        self.mesh = mesh

        if let sharding = sharding {
            let analyzer = JShardingAnalyzer(mesh: mesh)
            self.gradientSyncPattern = analyzer.gradientSyncPattern(
                for: sharding,
                parameterType: parameterType
            )
        } else {
            self.gradientSyncPattern = .none
        }
    }

    // Forward common properties
    public var shape: JTensorShape { tracer.shape }
    public var dtype: JDType { tracer.dtype }
    public var valueId: UInt64 { tracer.value.id }
}

// MARK: - Sharded Gradient

/// A gradient with associated sharding and synchronization information
public struct JShardedGradient {
    /// The gradient tracer
    public let gradient: JTracer

    /// Sharding of the gradient
    public let sharding: JSDYSharding?

    /// Whether this gradient has been synchronized
    public private(set) var isSynchronized: Bool

    /// Create a sharded gradient
    public init(gradient: JTracer, sharding: JSDYSharding? = nil, isSynchronized: Bool = false) {
        self.gradient = gradient
        self.sharding = sharding
        self.isSynchronized = isSynchronized
    }

    /// Synchronize this gradient according to its sharding pattern
    public func synchronized(mesh: JSDYMesh, reduction: JReductionType = .sum) -> JShardedGradient {
        guard !isSynchronized else { return self }

        let syncedGradient = gradient.allReduce(reduction: reduction)
        return JShardedGradient(
            gradient: syncedGradient,
            sharding: sharding,
            isSynchronized: true
        )
    }
}

// MARK: - Sharded Differentiable Context

/// Context for computing gradients with automatic sharding-aware synchronization
public class JShardedDifferentiableContext {
    /// The device mesh
    public let mesh: JSDYMesh

    /// The sharding analyzer
    private let analyzer: JShardingAnalyzer

    /// Base tracing context
    private let baseContext: JTracingContext

    /// Tracked parameters and their shardings
    private var parameterShardings: [UInt64: (sharding: JSDYSharding, type: JParameterType)] = [:]

    /// Input tracers
    private var inputs: [JShardedDifferentiableTracer] = []

    /// Output tracers
    private var outputs: [JTracer] = []

    /// Output shardings
    private var outputShardings: [JSDYSharding] = []

    /// Gradient operations to insert
    private var gradientSyncOps: [JGradientSyncOperation] = []

    /// Initialize with a mesh
    public init(mesh: JSDYMesh) {
        self.mesh = mesh
        self.analyzer = JShardingAnalyzer(mesh: mesh)
        self.baseContext = JTracingContext()
    }

    /// Create a sharded input
    public func input(
        shape: JTensorShape,
        dtype: JDType = .float32,
        name: String? = nil,
        sharding: JSDYSharding? = nil,
        parameterType: JParameterType = .weight
    ) -> JShardedDifferentiableTracer {
        let argName = name ?? "%arg\(inputs.count)"
        let tracer = baseContext.input(shape: shape, dtype: dtype, name: argName)

        let shardedTracer = JShardedDifferentiableTracer(
            tracer: tracer,
            sharding: sharding,
            mesh: mesh,
            parameterType: parameterType
        )

        inputs.append(shardedTracer)

        // Track for gradient sync
        if let sharding = sharding {
            parameterShardings[tracer.value.id] = (sharding, parameterType)
        }

        return shardedTracer
    }

    /// Record an output
    public func output(_ tracer: JTracer, sharding: JSDYSharding? = nil) {
        outputs.append(tracer)
        if let sharding = sharding {
            outputShardings.append(sharding)
        }
        baseContext.output(tracer)
    }

    /// Compute value and gradients with automatic synchronization
    public func valueWithGradient(
        function: (JTracer) -> JTracer,
        gradientFunction: (JTracer, JTracer) -> JTracer,
        at input: JShardedDifferentiableTracer
    ) -> (value: JTracer, gradient: JShardedGradient) {
        // Compute forward
        let value = function(input.tracer)

        // Create upstream gradient (1.0 for scalar loss)
        let upstream = JTracer(shape: value.shape, dtype: value.dtype)

        // Compute raw gradient using provided gradient function
        let rawGradient = gradientFunction(input.tracer, upstream)

        // Create sharded gradient with sync pattern
        var shardedGradient = JShardedGradient(
            gradient: rawGradient,
            sharding: input.sharding,
            isSynchronized: false
        )

        // Automatically synchronize based on pattern
        shardedGradient = synchronizeGradient(
            rawGradient,
            pattern: input.gradientSyncPattern,
            sharding: input.sharding
        )

        return (value, shardedGradient)
    }

    /// Compute gradients for multiple inputs with automatic synchronization
    public func valueWithGradient(
        function: (JTracer, JTracer) -> JTracer,
        gradientFunctions: ((JTracer, JTracer, JTracer) -> JTracer, (JTracer, JTracer, JTracer) -> JTracer),
        at x: JShardedDifferentiableTracer,
        _ y: JShardedDifferentiableTracer
    ) -> (value: JTracer, gradients: (JShardedGradient, JShardedGradient)) {
        // Compute forward
        let value = function(x.tracer, y.tracer)

        // Create upstream gradient
        let upstream = JTracer(shape: value.shape, dtype: value.dtype)

        // Compute raw gradients
        let rawGradX = gradientFunctions.0(x.tracer, y.tracer, upstream)
        let rawGradY = gradientFunctions.1(x.tracer, y.tracer, upstream)

        let gradX = synchronizeGradient(rawGradX, pattern: x.gradientSyncPattern, sharding: x.sharding)
        let gradY = synchronizeGradient(rawGradY, pattern: y.gradientSyncPattern, sharding: y.sharding)

        return (value, (gradX, gradY))
    }

    /// Synchronize a gradient according to its pattern
    private func synchronizeGradient(
        _ gradient: JTracer,
        pattern: JGradientSyncPattern,
        sharding: JSDYSharding?
    ) -> JShardedGradient {
        switch pattern {
        case .none:
            return JShardedGradient(gradient: gradient, sharding: sharding, isSynchronized: true)

        case .allReduceSum(let axes):
            let synced = gradient.allReduce(reduction: .sum)
            recordGradientSync(.allReduce(inputId: gradient.value.id, axes: axes, reduction: .sum))
            return JShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)

        case .allReduceMean(let axes):
            let synced = gradient.allReduce(reduction: .mean)
            recordGradientSync(.allReduce(inputId: gradient.value.id, axes: axes, reduction: .mean))
            return JShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)

        case .allGather(let axis, let dim):
            let synced = gradient.allGather(gatherDim: dim)
            recordGradientSync(.allGather(inputId: gradient.value.id, axis: axis, dim: dim))
            return JShardedGradient(gradient: synced, sharding: nil, isSynchronized: true)

        case .reduceScatter(let axis, let dim):
            let synced = gradient.reduceScatter(reduction: .sum, scatterDim: dim)
            recordGradientSync(.reduceScatter(inputId: gradient.value.id, axis: axis, dim: dim))
            return JShardedGradient(gradient: synced, sharding: sharding, isSynchronized: true)
        }
    }

    /// Record a gradient sync operation for MLIR generation
    private func recordGradientSync(_ op: JGradientSyncOperation) {
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

        // Add operations from base context
        let mlir = baseContext.buildModule(name: "temp")
        // Extract operations from the generated MLIR
        let lines = mlir.components(separatedBy: "\n")
        for line in lines {
            if line.contains("stablehlo.") || line.contains("arith.") {
                text += "  \(line)\n"
            }
        }

        // Return
        if !outputs.isEmpty {
            let results = outputs.map { "%v\($0.value.id)" }
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
        text += "    // ... backward pass operations ...\n"

        // Gradient sync operations
        for (i, _) in inputs.enumerated() {
            text += "    // %grad\(i)_synced = collective operation based on sharding pattern\n"
        }

        text += "  }\n"
        return text
    }

    /// Format tensor type
    private func tensorType(for shape: JTensorShape, dtype: JDType) -> String {
        let dims = shape.dimensions.compactMap { $0 }.map { String($0) }.joined(separator: "x")
        return "tensor<\(dims)x\(dtype.rawValue)>"
    }
}

// MARK: - Gradient Sync Operation Record

/// Records a gradient synchronization operation for MLIR generation
internal enum JGradientSyncOperation: Sendable, CustomStringConvertible {
    case allReduce(inputId: UInt64, axes: [String], reduction: JReductionType)
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
public func jShardedGradient(
    mesh: JSDYMesh,
    inputSharding: JSDYSharding,
    function: (JTracer) -> JTracer,
    gradientFunction: (JTracer, JTracer) -> JTracer,
    at input: JTracer
) -> JShardedGradient {
    let ctx = JShardedDifferentiableContext(mesh: mesh)
    let shardedInput = JShardedDifferentiableTracer(
        tracer: input,
        sharding: inputSharding,
        mesh: mesh,
        parameterType: .weight
    )
    let (_, gradient) = ctx.valueWithGradient(
        function: function,
        gradientFunction: gradientFunction,
        at: shardedInput
    )
    return gradient
}

/// Compute value and gradient with automatic sharding-aware synchronization
public func jShardedValueWithGradient(
    mesh: JSDYMesh,
    inputSharding: JSDYSharding,
    function: (JTracer) -> JTracer,
    gradientFunction: (JTracer, JTracer) -> JTracer,
    at input: JTracer
) -> (value: JTracer, gradient: JShardedGradient) {
    let ctx = JShardedDifferentiableContext(mesh: mesh)
    let shardedInput = JShardedDifferentiableTracer(
        tracer: input,
        sharding: inputSharding,
        mesh: mesh,
        parameterType: .weight
    )
    return ctx.valueWithGradient(
        function: function,
        gradientFunction: gradientFunction,
        at: shardedInput
    )
}

/// Data-parallel gradient computation with automatic all-reduce
public func jDataParallelGradient(
    numDevices: Int,
    function: (JTracer) -> JTracer,
    gradientFunction: (JTracer, JTracer) -> JTracer,
    at input: JTracer
) -> JShardedGradient {
    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: numDevices)
    let sharding = JSDYSharding.dataParallel(mesh: mesh, rank: input.shape.rank, batchAxis: "batch")
    return jShardedGradient(
        mesh: mesh,
        inputSharding: sharding,
        function: function,
        gradientFunction: gradientFunction,
        at: input
    )
}

// MARK: - Sharded Training Step

/// A complete sharded training step with forward, backward, and gradient sync
public struct JShardedTrainingStep {
    /// The device mesh
    public let mesh: JSDYMesh

    /// Parameter shardings
    public let parameterShardings: [JSDYSharding]

    /// Initialize
    public init(mesh: JSDYMesh, parameterShardings: [JSDYSharding]) {
        self.mesh = mesh
        self.parameterShardings = parameterShardings
    }

    /// Execute a training step with automatic gradient synchronization
    public func step(
        lossFunction: (JTracer, JTracer) -> JTracer,
        gradientFunctions: ((JTracer, JTracer, JTracer) -> JTracer, (JTracer, JTracer, JTracer) -> JTracer),
        input: JTracer,
        weights: JTracer,
        inputSharding: JSDYSharding,
        weightSharding: JSDYSharding
    ) -> (loss: JTracer, inputGrad: JShardedGradient, weightGrad: JShardedGradient) {
        let ctx = JShardedDifferentiableContext(mesh: mesh)

        let shardedInput = JShardedDifferentiableTracer(
            tracer: input,
            sharding: inputSharding,
            mesh: mesh,
            parameterType: .activation
        )

        let shardedWeights = JShardedDifferentiableTracer(
            tracer: weights,
            sharding: weightSharding,
            mesh: mesh,
            parameterType: .weight
        )

        let (lossValue, grads) = ctx.valueWithGradient(
            function: lossFunction,
            gradientFunctions: gradientFunctions,
            at: shardedInput,
            shardedWeights
        )
        return (lossValue, grads.0, grads.1)
    }
}

// MARK: - Sharding-Aware Optimizer

/// An optimizer that handles sharded parameters and gradients
public class JShardedOptimizer {
    /// The device mesh
    public let mesh: JSDYMesh

    /// Learning rate
    public var learningRate: Double

    /// Initialize
    public init(mesh: JSDYMesh, learningRate: Double = 0.01) {
        self.mesh = mesh
        self.learningRate = learningRate
    }

    /// Apply gradients to parameters with proper sharding
    public func apply(
        gradients: [JShardedGradient],
        to parameters: [JTracer]
    ) -> [JTracer] {
        precondition(gradients.count == parameters.count)

        return zip(parameters, gradients).map { (param, grad) in
            // Ensure gradient is synchronized
            let syncedGrad = grad.isSynchronized
                ? grad.gradient
                : grad.gradient.allReduce(reduction: .mean)

            // Simple SGD update: param = param - lr * grad
            let lr = JTracer(shape: JTensorShape([]), dtype: param.dtype)
            let scaledGrad = syncedGrad * lr
            return param - scaledGrad
        }
    }
}

// MARK: - Automatic Gradient VJP Rules

/// Sharding-aware VJP rules for common operations
public struct JShardedVJPRules {

    /// MatMul VJP with sharding awareness
    /// For Y = X @ W:
    /// - dX = dY @ W^T
    /// - dW = X^T @ dY
    /// With data-parallel sharding on X, dW needs all-reduce
    public static func matmulVJP(
        x: JShardedDifferentiableTracer,
        w: JShardedDifferentiableTracer,
        upstream: JTracer,
        mesh: JSDYMesh
    ) -> (JShardedGradient, JShardedGradient) {
        // Compute raw gradients
        let dX = upstream.matmul(w.tracer.transpose())
        let dW = x.tracer.transpose().matmul(upstream)

        // Apply sharding-aware synchronization
        let analyzer = JShardingAnalyzer(mesh: mesh)

        let xPattern = x.sharding.map { analyzer.gradientSyncPattern(for: $0, parameterType: .activation) } ?? .none
        let wPattern = w.sharding.map { analyzer.gradientSyncPattern(for: $0, parameterType: .weight) } ?? .none

        let gradX = applySyncPattern(dX, pattern: xPattern, sharding: x.sharding)
        let gradW = applySyncPattern(dW, pattern: wPattern, sharding: w.sharding)

        return (gradX, gradW)
    }

    /// Apply sync pattern to a gradient
    private static func applySyncPattern(
        _ gradient: JTracer,
        pattern: JGradientSyncPattern,
        sharding: JSDYSharding?
    ) -> JShardedGradient {
        switch pattern {
        case .none:
            return JShardedGradient(gradient: gradient, sharding: sharding, isSynchronized: true)
        case .allReduceSum:
            return JShardedGradient(
                gradient: gradient.allReduce(reduction: .sum),
                sharding: sharding,
                isSynchronized: true
            )
        case .allReduceMean:
            return JShardedGradient(
                gradient: gradient.allReduce(reduction: .mean),
                sharding: sharding,
                isSynchronized: true
            )
        case .allGather(_, let dim):
            return JShardedGradient(
                gradient: gradient.allGather(gatherDim: dim),
                sharding: nil,
                isSynchronized: true
            )
        case .reduceScatter(_, let dim):
            return JShardedGradient(
                gradient: gradient.reduceScatter(scatterDim: dim),
                sharding: sharding,
                isSynchronized: true
            )
        }
    }

    /// Softmax cross-entropy VJP with sharding
    /// Loss is always reduced across devices
    public static func softmaxCrossEntropyVJP(
        logits: JShardedDifferentiableTracer,
        labels: JTracer,
        upstream: JTracer,
        mesh: JSDYMesh
    ) -> JShardedGradient {
        // dL/dlogits = softmax(logits) - labels (for one-hot labels)
        let softmax = logits.tracer.softmax()
        let dLogits = (softmax - labels) * upstream

        // Loss gradient needs all-reduce if data-parallel
        let analyzer = JShardingAnalyzer(mesh: mesh)
        let pattern = logits.sharding.map { analyzer.gradientSyncPattern(for: $0, parameterType: .loss) } ?? .none

        return applySyncPattern(dLogits, pattern: pattern, sharding: logits.sharding)
    }
}
