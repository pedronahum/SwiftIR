/// SDY Dialect Registration and Pass Pipeline Management for SwiftIR
/// This file provides functions for integrating Shardy into SwiftIR's MLIR pipeline.

import SdyCAPIWrapper
import MLIRCoreWrapper

// MARK: - SDY Dialect Registration

/// Manages the SDY dialect and its passes within SwiftIR.
///
/// The SDY dialect provides tensor sharding and partitioning capabilities
/// from the OpenXLA Shardy project.
public enum SdyDialect {
    /// Whether SDY passes have been registered
    private nonisolated(unsafe) static var passesRegistered = false

    /// Registers all SDY passes and pipelines with MLIR.
    ///
    /// This should be called once at startup before using any SDY passes.
    /// It's safe to call multiple times - subsequent calls are no-ops.
    public static func registerPasses() {
        guard !passesRegistered else { return }
        mlirRegisterAllSdyPassesAndPipelines()
        passesRegistered = true
    }

    /// Gets the SDY dialect handle for loading into a context.
    ///
    /// - Returns: The MlirDialectHandle for the SDY dialect
    public static var dialectHandle: MlirDialectHandle {
        mlirGetDialectHandle__sdy__()
    }

    /// Loads the SDY dialect into an MLIR context.
    ///
    /// - Parameter context: The MLIR context to load the dialect into
    public static func loadDialect(into context: MlirContext) {
        let handle = dialectHandle
        mlirDialectHandleLoadDialect(handle, context)
    }

    /// Registers the SDY dialect with a context (creates but doesn't load).
    ///
    /// - Parameter context: The MLIR context to register with
    public static func registerDialect(with context: MlirContext) {
        let handle = dialectHandle
        mlirDialectHandleRegisterDialect(handle, context)
    }
}

// MARK: - Sharding Context

/// A context for managing tensor sharding annotations.
///
/// ShardingContext provides a high-level interface for adding sharding
/// annotations to tensors and running the sharding propagation pipeline.
///
/// Example:
/// ```swift
/// let ctx = ShardingContext()
/// let mesh = DeviceMesh.grid(name: "mesh", rows: 2, cols: 4)
/// ctx.addMesh(mesh)
///
/// // Add shardings to operations...
/// ctx.propagateShardings(module)
/// ```
public class ShardingContext {
    /// The device meshes available in this context
    public private(set) var meshes: [String: DeviceMesh] = [:]

    /// Creates a new sharding context.
    public init() {
        // Ensure SDY passes are registered
        SdyDialect.registerPasses()
    }

    /// Adds a device mesh to this context.
    ///
    /// - Parameter mesh: The mesh to add
    public func addMesh(_ mesh: DeviceMesh) {
        meshes[mesh.name] = mesh
    }

    /// Gets a mesh by name.
    ///
    /// - Parameter name: The mesh name
    /// - Returns: The mesh, or nil if not found
    public func mesh(named name: String) -> DeviceMesh? {
        meshes[name]
    }

    /// Generates MLIR text for all mesh definitions.
    ///
    /// - Returns: MLIR text defining all meshes
    public func meshDefinitions() -> String {
        meshes.values.map { $0.mlirText }.joined(separator: "\n")
    }

    /// Creates a TensorSharding for the given mesh and dimension specs.
    ///
    /// - Parameters:
    ///   - meshName: The mesh to shard on
    ///   - axisNames: Axis names for each dimension (nil for replicated)
    /// - Returns: A TensorSharding, or nil if mesh not found
    public func sharding(mesh meshName: String, axes axisNames: [String?]) -> TensorSharding? {
        guard meshes[meshName] != nil else { return nil }
        return TensorSharding(meshName: meshName, axisNames: axisNames)
    }
}

// MARK: - Pass Pipeline Names

/// Names of the SDY pass pipelines available.
public enum SdyPassPipeline {
    /// The full propagation pipeline including import, propagation, and export.
    public static let propagation = "sdy-propagation-pipeline"

    /// Import pipeline for preparing IR for propagation.
    public static let `import` = "sdy-import-pipeline"

    /// Export pipeline for post-propagation processing.
    public static let export = "sdy-export-pipeline"

    /// Basic propagation algorithm.
    public static let basicPropagate = "sdy-basic-propagate"

    /// Aggressive propagation algorithm.
    public static let aggressivePropagate = "sdy-aggressive-propagate"

    /// User-priority propagation algorithm.
    public static let userPriorityPropagate = "sdy-user-priority-propagate"

    /// Op-priority propagation algorithm.
    public static let opPriorityPropagate = "sdy-op-priority-propagate"
}

// MARK: - MLIR Text Generation Helpers

extension ShardingContext {
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
        let sharding = TensorSharding(meshName: meshName, dimShardings: dimShardings)
        return sharding.mlirAttributeText
    }

    /// Generates a per-value sharding attribute string.
    ///
    /// - Parameter shardings: List of shardings for each value
    /// - Returns: The sharding_per_value attribute text
    public func perValueShardingAttribute(shardings: [TensorSharding]) -> String {
        let inner = shardings.map { s in
            let dims = s.dimShardings.map { $0.description }.joined(separator: ", ")
            return "<@\(s.meshName), [\(dims)]>"
        }.joined(separator: ", ")
        return "#sdy.sharding_per_value<[\(inner)]>"
    }
}
