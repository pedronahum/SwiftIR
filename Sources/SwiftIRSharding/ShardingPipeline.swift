/// Sharding Pipeline for SwiftIR
/// Integrates Shardy (SDY) dialect with SwiftIR's MLIR pipeline.

import SdyCAPIWrapper
import MLIRCoreWrapper

// MARK: - SDY Dialect Loading

/// Loads the SDY dialect into an MLIR context.
///
/// This function registers and loads the Shardy dialect which provides
/// tensor sharding and partitioning capabilities.
///
/// - Parameter context: The MLIR context handle
/// - Returns: true if the dialect was loaded successfully
@discardableResult
public func loadSdyDialect(_ context: MlirContext) -> Bool {
    // Register SDY passes first
    SdyDialect.registerPasses()

    // Load the SDY dialect
    SdyDialect.loadDialect(into: context)

    return true
}

/// Loads all sharding-related dialects into an MLIR context.
///
/// This includes the SDY dialect and ensures StableHLO is compatible
/// with sharding annotations.
///
/// - Parameter context: The MLIR context handle
/// - Returns: true if all dialects were loaded successfully
@discardableResult
public func loadAllShardingDialects(_ context: MlirContext) -> Bool {
    return loadSdyDialect(context)
}

// MARK: - Sharding Pipeline

/// A pipeline for adding sharding annotations and running propagation.
///
/// ShardingPipeline manages the complete workflow for sharding:
/// 1. Load SDY dialect
/// 2. Add mesh definitions
/// 3. Annotate tensors with shardings
/// 4. Run propagation to infer shardings
/// 5. Export for execution
///
/// Example:
/// ```swift
/// let pipeline = ShardingPipeline(context: mlirContext)
/// pipeline.addMesh(DeviceMesh.grid(name: "mesh", rows: 2, cols: 2))
///
/// // After building your StableHLO module...
/// try pipeline.propagate(module: module)
/// ```
public class ShardingPipeline {
    /// The MLIR context (optional - only needed for C API operations)
    private let context: MlirContext?

    /// The sharding context for mesh and annotation management
    public let shardingContext: ShardingContext

    /// Whether SDY passes have been registered (static initialization)
    private static let passesRegistered: Bool = {
        SdyDialect.registerPasses()
        return true
    }()

    /// Creates a new sharding pipeline for text-based MLIR generation.
    ///
    /// This initializer is suitable for generating MLIR text with sharding
    /// annotations. It does not require an MLIR context.
    ///
    /// Example:
    /// ```swift
    /// let pipeline = ShardingPipeline()
    /// pipeline.addMesh(DeviceMesh.linear(name: "mesh", axisName: "x", size: 4))
    /// let module = pipeline.generateShardedModule(...)
    /// ```
    public init() {
        self.context = nil
        self.shardingContext = ShardingContext()

        // Ensure passes are registered for later use
        _ = ShardingPipeline.passesRegistered
    }

    /// Creates a new sharding pipeline with an MLIR context.
    ///
    /// Use this initializer when you need to work with the C API
    /// for actual pass execution and IR manipulation.
    ///
    /// - Parameter context: The MLIR context handle
    public init(context: MlirContext) {
        self.context = context
        self.shardingContext = ShardingContext()

        // Ensure SDY dialect is loaded
        _ = loadSdyDialect(context)

        // Ensure passes are registered (via static initialization)
        _ = ShardingPipeline.passesRegistered
    }

    /// Returns the MLIR context if available.
    public var mlirContext: MlirContext? {
        return context
    }

    /// Adds a device mesh to the pipeline.
    ///
    /// - Parameter mesh: The device mesh to add
    public func addMesh(_ mesh: DeviceMesh) {
        shardingContext.addMesh(mesh)
    }

    /// Gets a mesh by name.
    ///
    /// - Parameter name: The mesh name
    /// - Returns: The mesh, or nil if not found
    public func mesh(named name: String) -> DeviceMesh? {
        shardingContext.mesh(named: name)
    }

    /// Generates MLIR text for all mesh definitions.
    ///
    /// This can be prepended to a module to define the meshes.
    ///
    /// - Returns: MLIR text defining all meshes
    public func meshDefinitions() -> String {
        shardingContext.meshDefinitions()
    }

    /// Creates a sharding specification for a tensor.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to shard on
    ///   - axisNames: Axis names for each dimension (nil for replicated)
    /// - Returns: A TensorSharding, or nil if mesh not found
    public func sharding(mesh meshName: String, axes axisNames: [String?]) -> TensorSharding? {
        shardingContext.sharding(mesh: meshName, axes: axisNames)
    }

    /// Generates a sharding attribute string for use in MLIR.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to reference
    ///   - dimShardings: Sharding specs for each dimension
    /// - Returns: The sharding attribute text
    public func shardingAttribute(
        mesh meshName: String,
        dimShardings: [DimensionSharding]
    ) -> String {
        shardingContext.shardingAttribute(mesh: meshName, dimShardings: dimShardings)
    }
}

// MARK: - Pass Pipeline Execution

extension ShardingPipeline {
    /// The result of running a sharding pipeline.
    public struct PropagationResult {
        /// Whether propagation succeeded
        public let success: Bool

        /// Any diagnostic messages
        public let diagnostics: [String]

        /// Number of operations with shardings after propagation
        public let shardedOperations: Int
    }

    /// Runs the SDY import pipeline on a module.
    ///
    /// The import pipeline prepares the module for sharding propagation.
    ///
    /// - Parameter moduleText: The MLIR module text
    /// - Returns: The prepared module text, or nil on failure
    public func runImportPipeline(moduleText: String) -> String? {
        // For now, return the module with mesh definitions prepended
        // The actual C API pass execution will be integrated when linking is complete
        let meshDefs = meshDefinitions()
        if meshDefs.isEmpty {
            return moduleText
        }
        return meshDefs + "\n\n" + moduleText
    }

    /// Runs the SDY propagation pipeline on a module.
    ///
    /// This is the main entry point for sharding propagation. It will:
    /// 1. Parse the module
    /// 2. Run the import pipeline
    /// 3. Run propagation to infer shardings
    /// 4. Return the fully sharded module
    ///
    /// - Parameter moduleText: The MLIR module text with initial shardings
    /// - Returns: The module text with propagated shardings
    public func runPropagationPipeline(moduleText: String) -> String? {
        // Prepend mesh definitions
        guard let prepared = runImportPipeline(moduleText: moduleText) else {
            return nil
        }

        // In the future, this will use the pass manager to run:
        // - sdy-import-pipeline
        // - sdy-propagation-pipeline
        // - sdy-export-pipeline
        //
        // For now, return the prepared module
        return prepared
    }
}

// MARK: - Sharding Annotation Helpers

extension ShardingPipeline {
    /// Creates an MLIR string attribute for a sharding.
    ///
    /// - Parameter sharding: The tensor sharding
    /// - Returns: The attribute text (e.g., `{sdy.sharding = #sdy.sharding<@mesh, [...]>}`)
    public func shardingAttributeDict(_ sharding: TensorSharding) -> String {
        return "{sdy.sharding = \(sharding.mlirAttributeText)}"
    }

    /// Creates a sharded tensor type string.
    ///
    /// - Parameters:
    ///   - baseType: The base tensor type (e.g., "tensor<8x8xf32>")
    ///   - sharding: The tensor sharding
    /// - Returns: The annotated type string
    public func shardedType(_ baseType: String, sharding: TensorSharding) -> String {
        return "\(baseType) \(shardingAttributeDict(sharding))"
    }

    /// Annotates a function argument with sharding.
    ///
    /// - Parameters:
    ///   - argName: The argument name (e.g., "%arg0")
    ///   - argType: The argument type (e.g., "tensor<8x8xf32>")
    ///   - sharding: The tensor sharding
    /// - Returns: The annotated argument string
    public func shardedArg(
        name argName: String,
        type argType: String,
        sharding: TensorSharding
    ) -> String {
        return "\(argName): \(argType) \(shardingAttributeDict(sharding))"
    }
}

// MARK: - Common Sharding Patterns

extension ShardingPipeline {
    /// Creates a data-parallel sharding for a batched tensor.
    ///
    /// Shards the first dimension (batch) across devices.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to shard on
    ///   - batchAxis: The axis to shard the batch dimension on
    ///   - rank: Total tensor rank
    /// - Returns: A TensorSharding for data parallelism
    public func dataParallelSharding(
        mesh meshName: String,
        batchAxis: String,
        rank: Int
    ) -> TensorSharding {
        var axisNames: [String?] = Array(repeating: nil, count: rank)
        axisNames[0] = batchAxis
        return TensorSharding(meshName: meshName, axisNames: axisNames)
    }

    /// Creates a model-parallel sharding for a weight tensor.
    ///
    /// Shards a specified dimension (typically output features) across devices.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to shard on
    ///   - axis: The axis to shard the model dimension on
    ///   - shardedDim: Which dimension to shard
    ///   - rank: Total tensor rank
    /// - Returns: A TensorSharding for model parallelism
    public func modelParallelSharding(
        mesh meshName: String,
        axis: String,
        shardedDim: Int,
        rank: Int
    ) -> TensorSharding {
        var axisNames: [String?] = Array(repeating: nil, count: rank)
        if shardedDim >= 0 && shardedDim < rank {
            axisNames[shardedDim] = axis
        }
        return TensorSharding(meshName: meshName, axisNames: axisNames)
    }

    /// Creates a 2D sharding for matrix operations.
    ///
    /// Shards both dimensions of a 2D tensor.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to shard on
    ///   - rowAxis: The axis for the row dimension
    ///   - colAxis: The axis for the column dimension
    /// - Returns: A TensorSharding for 2D parallelism
    public func matrixSharding(
        mesh meshName: String,
        rowAxis: String,
        colAxis: String
    ) -> TensorSharding {
        TensorSharding(meshName: meshName, axisNames: [rowAxis, colAxis])
    }
}

// MARK: - MLIR Text Generation

extension ShardingPipeline {
    /// Generates a complete MLIR module with sharding annotations.
    ///
    /// This helper creates a module with:
    /// - Mesh definitions
    /// - A main function with sharded arguments
    /// - The function body
    ///
    /// - Parameters:
    ///   - funcName: The function name
    ///   - args: List of (name, type, sharding) tuples for arguments
    ///   - results: Result types
    ///   - body: The function body MLIR text
    /// - Returns: Complete MLIR module text
    public func generateShardedModule(
        funcName: String,
        args: [(name: String, type: String, sharding: TensorSharding?)],
        results: [String],
        body: String
    ) -> String {
        var lines: [String] = []

        // Mesh definitions
        let meshDefs = meshDefinitions()
        if !meshDefs.isEmpty {
            lines.append(meshDefs)
            lines.append("")
        }

        // Function signature
        let argStrings = args.map { arg in
            if let sharding = arg.sharding {
                return shardedArg(name: arg.name, type: arg.type, sharding: sharding)
            } else {
                return "\(arg.name): \(arg.type)"
            }
        }

        let argsStr = argStrings.joined(separator: ", ")
        let resultsStr = results.isEmpty ? "" : " -> (\(results.joined(separator: ", ")))"

        lines.append("func.func @\(funcName)(\(argsStr))\(resultsStr) {")

        // Body (indent each line)
        for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
            lines.append("  \(line)")
        }

        lines.append("}")

        return lines.joined(separator: "\n")
    }
}
