//===-- TensorDialect.swift - MLIR Tensor Dialect --------*- Swift -*-===//
//
// SwiftIR - Phase 7: Tensor Operations for ML
// Provides Swift wrapper for MLIR Tensor dialect
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import MLIRCoreWrapper

/// The MLIR Tensor dialect - foundation for ML operations
///
/// Phase 7 Strategy:
/// - Use MLIR's built-in Tensor dialect as foundation
/// - Provides tensor types and operations
/// - Compatible with XLA, ONNX, and other ML backends
/// - Will migrate to StableHLO in future phases when compatibility is resolved
public struct TensorDialect {
    /// Register the Tensor dialect with an MLIR context
    ///
    /// Note: This properly registers the tensor dialect handle before loading.
    /// With StableHLO's LLVM build, dialects need explicit handle registration
    /// via mlirDialectHandleRegisterDialect() before they can be used.
    public static func register(with context: MLIRContext) {
        // Properly register and load the tensor dialect
        context.registerTensorDialect()
    }

    /// Register and load in one step (convenience method)
    public static func registerAndLoad(into context: MLIRContext) {
        register(with: context)
    }
}
