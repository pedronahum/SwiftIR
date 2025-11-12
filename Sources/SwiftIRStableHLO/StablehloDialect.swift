// StablehloDialect.swift - Swift wrapper for StableHLO Dialect
// Copyright 2024 SwiftIR Project
// Following the pattern established by SwiftIRCore for MLIR dialects

import SwiftIRCore
import MLIRCoreWrapper

// MARK: - Pass Registration

/// Registers all StableHLO compiler passes
///
/// This must be called before creating a PassManager that uses StableHLO passes.
/// The passes include:
/// - stablehlo-canonicalize-dynamism: Convert dynamic ops to static where possible
/// - stablehlo-refine-shapes: Propagate shape information
/// - stablehlo-legalize-to-linalg: Lower StableHLO to Linalg
/// - And many more optimization and transformation passes
public func registerAllStablehloPasses() {
    mlirRegisterAllStablehloPassesWrapper()
}

// MARK: - StableHLO Dialect

/// Loads the StableHLO dialect for the given context if not already loaded.
///
/// StableHLO is a portability layer for ML models. It provides versioned,
/// stable operations that serve as an interface between different ML frameworks
/// (JAX, PyTorch, TensorFlow) and ML compilers (XLA).
///
/// - Parameter context: The MLIR context
/// - Returns: true if the dialect was loaded successfully
@discardableResult
public func loadStablehloDialect(_ context: MLIRContext) -> Bool {
    // Register and load the StableHLO dialect handle
    context.registerStablehloDialect()
    return context.loadDialect("stablehlo")
}

// MARK: - CHLO Dialect

/// Loads the CHLO (Client HLO) dialect for the given context if not already loaded.
///
/// CHLO provides client-level high-level operations that are later lowered
/// to StableHLO. These operations are easier to use from frameworks but
/// are not part of the stable StableHLO specification.
///
/// - Parameter context: The MLIR context
/// - Returns: true if the dialect was loaded successfully
@discardableResult
public func loadChloDialect(_ context: MLIRContext) -> Bool {
    // Register and load the CHLO dialect handle
    context.registerChloDialect()
    return context.loadDialect("chlo")
}

// MARK: - Helper Functions

/// Loads all StableHLO-related dialects (StableHLO, CHLO)
///
/// - Parameter context: The MLIR context
@discardableResult
public func loadAllStablehloDialects(_ context: MLIRContext) -> Bool {
    let stablehloLoaded = loadStablehloDialect(context)
    let chloLoaded = loadChloDialect(context)
    return stablehloLoaded && chloLoaded
}
