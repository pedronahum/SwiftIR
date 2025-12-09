/// ShardingPipeline for SwiftIRShardingLite
/// Pure Swift implementation - no C dependencies.
///
/// Provides MLIR text generation for sharded StableHLO programs.

import Foundation

// MARK: - Sharding Pipeline

/// A pipeline for generating sharded MLIR modules.
///
/// ShardingPipeline manages the workflow for sharding:
/// 1. Define device meshes
/// 2. Create tensor shardings
/// 3. Generate MLIR text with annotations
/// 4. Optionally run sdy_opt for propagation
///
/// Example:
/// ```swift
/// let pipeline = ShardingPipeline()
/// pipeline.addMesh(DeviceMesh.grid(name: "mesh", rows: 2, cols: 2))
///
/// let sharding = pipeline.dataParallelSharding(mesh: "mesh", batchAxis: "x", rank: 2)
/// let module = pipeline.generateShardedModule(...)
/// ```
public class ShardingPipeline {
    /// The sharding context for mesh and annotation management
    public let shardingContext: ShardingContext

    /// Creates a new sharding pipeline.
    public init() {
        self.shardingContext = ShardingContext()
    }

    /// Adds a device mesh to the pipeline.
    public func addMesh(_ mesh: DeviceMesh) {
        shardingContext.addMesh(mesh)
    }

    /// Gets a mesh by name.
    public func mesh(named name: String) -> DeviceMesh? {
        shardingContext.mesh(named: name)
    }

    /// Generates MLIR text for all mesh definitions.
    public func meshDefinitions() -> String {
        shardingContext.meshDefinitions()
    }

    /// Creates a sharding specification for a tensor.
    public func sharding(mesh meshName: String, axes axisNames: [String?]) -> TensorSharding? {
        shardingContext.sharding(mesh: meshName, axes: axisNames)
    }
}

// MARK: - Common Sharding Patterns

extension ShardingPipeline {
    /// Creates a data-parallel sharding for a batched tensor.
    ///
    /// Shards the first dimension (batch) across devices.
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
    public func matrixSharding(
        mesh meshName: String,
        rowAxis: String,
        colAxis: String
    ) -> TensorSharding {
        TensorSharding(meshName: meshName, axisNames: [rowAxis, colAxis])
    }
}

// MARK: - Sharding Annotation Helpers

extension ShardingPipeline {
    /// Creates an MLIR attribute dict for a sharding.
    public func shardingAttributeDict(_ sharding: TensorSharding) -> String {
        return "{sdy.sharding = \(sharding.mlirAttributeText)}"
    }

    /// Creates a sharded tensor type string.
    public func shardedType(_ baseType: String, sharding: TensorSharding) -> String {
        return "\(baseType) \(shardingAttributeDict(sharding))"
    }

    /// Annotates a function argument with sharding.
    public func shardedArg(
        name argName: String,
        type argType: String,
        sharding: TensorSharding
    ) -> String {
        return "\(argName): \(argType) \(shardingAttributeDict(sharding))"
    }
}

// MARK: - MLIR Text Generation

extension ShardingPipeline {
    /// Generates a complete MLIR module with sharding annotations.
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

    /// Runs the import pipeline (prepends mesh definitions).
    public func runImportPipeline(moduleText: String) -> String? {
        let meshDefs = meshDefinitions()
        if meshDefs.isEmpty {
            return moduleText
        }
        return meshDefs + "\n\n" + moduleText
    }
}

// MARK: - SDY Opt Runner

extension ShardingPipeline {
    /// Checks if sdy_opt is available for propagation.
    public static func isSdyOptAvailable(at path: String? = nil) -> Bool {
        let runner = SdyOptRunner(path: path)
        return runner.isAvailable
    }

    /// Runs propagation using sdy_opt external process.
    public func runPropagationWithSdyOpt(
        moduleText: String,
        sdyOptPath: String? = nil
    ) -> String? {
        guard let prepared = runImportPipeline(moduleText: moduleText) else {
            return nil
        }

        let fullModule: String
        if prepared.contains("module {") {
            fullModule = prepared
        } else {
            fullModule = "module {\n\(prepared)\n}"
        }

        let runner = SdyOptRunner(path: sdyOptPath)
        return runner.runPropagation(on: fullModule)
    }
}
