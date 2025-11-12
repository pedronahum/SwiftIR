//===-- SwiftIRXLA.swift - Tensor Operations for ML ------*- Swift -*-===//
//
// SwiftIR - Phase 7: Tensor Operations for ML
// Foundation for XLA-compatible ML operations
//
//===------------------------------------------------------------------===//

@_exported import SwiftIRCore
@_exported import SwiftIRTypes
@_exported import SwiftIRDialects

/// SwiftIRXLA - Tensor operations for machine learning
///
/// Phase 7 Implementation Strategy:
/// - Uses MLIR's built-in Tensor dialect as foundation
/// - Provides tensor creation and manipulation operations
/// - Element-wise operations via Arith dialect
/// - Foundation for future StableHLO/XLA integration (Phase 8-9)
///
/// Key Operations:
/// - `TensorOps.empty()` - Create empty tensors
/// - `TensorOps.fromElements()` - Create tensors from values
/// - `TensorOps.extract/insert()` - Element access
/// - Arith ops (add, mul, etc.) work on tensor<f32> types
///
/// Example:
/// ```swift
/// let ctx = Context()
/// TensorDialect.registerAndLoad(into: ctx)
///
/// let module = Module(context: ctx) { builder in
///     let tensor = TensorOps.empty(
///         shape: [2, 3],
///         elementType: FloatType.f32(context: ctx),
///         in: builder
///     )
/// }
/// ```
public struct SwiftIRXLA {
    /// Current version of SwiftIRXLA
    public static let version = "0.1.0-phase7"

    /// Phase 7 status
    public static let phase = "Phase 7: Tensor Dialect Foundation"

    /// Future phases
    public static let roadmap = """
    Phase 7 (Current): MLIR Tensor dialect ✓
    Phase 8 (Next): StableHLO operations
    Phase 9 (Future): XLA runtime integration
    """
}
