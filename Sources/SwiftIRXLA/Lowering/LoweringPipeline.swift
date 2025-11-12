//===-- LoweringPipeline.swift - Lowering Pipeline -------*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime
// Orchestrates the lowering pipeline from Tensor/Linalg to LLVM
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import MLIRCoreWrapper

/// Lowering pipeline that converts high-level IR to executable LLVM
///
/// Pipeline stages:
/// 1. Linalg → Loops (SCF dialect)
/// 2. SCF → ControlFlow
/// 3. MemRef → LLVM
/// 4. Arith → LLVM
/// 5. Func → LLVM
public class LoweringPipeline {
    /// The MLIR context
    private let context: MLIRContext

    /// Create a new lowering pipeline
    public init(context: MLIRContext) {
        self.context = context

        // Register all required passes
        registerAllPasses()
    }

    /// Lower a module from Tensor/Linalg to LLVM dialect
    ///
    /// This prepares the module for execution with ExecutionEngine
    public func lower(module: MLIRModule) throws {
        // IMPORTANT: Add llvm.emit_c_interface BEFORE lowering
        // The func-to-llvm pass needs to see this attribute to generate C wrappers
        addEmitCInterfaceToFuncOps(in: module)

        let pm = PassManager(context: context)

        // Enable verification to catch errors early
        pm.enableVerifier(true)

        // Get the OpPassManager for adding passes
        let opPM = pm.asOpPassManager()

        // Stage 1: Lower Linalg operations to loops
        // Converts linalg.matmul, linalg.pooling, etc. to nested SCF loops
        let linalgToLoops = mlirCreateLinalgConvertLinalgToLoopsPass()
        mlirOpPassManagerAddOwnedPass(opPM, linalgToLoops)

        // Stage 2: Lower SCF (Structured Control Flow) to ControlFlow
        // Converts scf.for, scf.if to cf.br, cf.cond_br
        let scfToControlFlow = mlirCreateConversionSCFToControlFlowPass()
        mlirOpPassManagerAddOwnedPass(opPM, scfToControlFlow)

        // Stage 3: Lower Arith operations to LLVM
        // Converts arith.addf, arith.mulf, etc. to llvm.fadd, llvm.fmul
        let arithToLLVM = mlirCreateConversionArithToLLVMConversionPass()
        mlirOpPassManagerAddOwnedPass(opPM, arithToLLVM)

        // Stage 4: Lower MemRef operations to LLVM
        // Converts memref.alloc, memref.load, memref.store to LLVM ops
        let memrefToLLVM = mlirCreateConversionFinalizeMemRefToLLVMConversionPass()
        mlirOpPassManagerAddOwnedPass(opPM, memrefToLLVM)

        // Stage 5: Lower Func operations to LLVM
        // Converts func.func, func.call, func.return to LLVM equivalents
        // This pass will generate C interface wrappers for functions with the attribute
        let funcToLLVM = mlirCreateConversionConvertFuncToLLVMPass()
        mlirOpPassManagerAddOwnedPass(opPM, funcToLLVM)

        // Run the complete pipeline
        try pm.run(on: module)
    }

    /// Adds llvm.emit_c_interface attribute to all func.func operations
    ///
    /// This must be called BEFORE lowering so the func-to-llvm pass can see it
    /// and generate the appropriate C interface wrappers
    private func addEmitCInterfaceToFuncOps(in module: MLIRModule) {
        let unitAttr = context.createUnitAttribute()
        let body = module.getBody()

        body.forEach { op in
            // Add the attribute to func.func operations
            if op.name == "func.func" {
                op.setDiscardableAttribute(name: "llvm.emit_c_interface", value: unitAttr)
            }
        }
    }

    /// Lower with IR printing enabled (for debugging)
    public func lowerWithDebugOutput(module: MLIRModule) throws {
        // Add emit_c_interface before lowering
        addEmitCInterfaceToFuncOps(in: module)

        let pm = PassManager(context: context)

        pm.enableVerifier(true)
        pm.enableIRPrinting(
            printBeforeAll: false,
            printAfterAll: true,
            printModuleScope: true,
            printAfterOnlyOnChange: true,
            printAfterOnlyOnFailure: true
        )

        let opPM = pm.asOpPassManager()

        // Add all the passes
        mlirOpPassManagerAddOwnedPass(opPM, mlirCreateLinalgConvertLinalgToLoopsPass())
        mlirOpPassManagerAddOwnedPass(opPM, mlirCreateConversionSCFToControlFlowPass())
        mlirOpPassManagerAddOwnedPass(opPM, mlirCreateConversionArithToLLVMConversionPass())
        mlirOpPassManagerAddOwnedPass(opPM, mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
        mlirOpPassManagerAddOwnedPass(opPM, mlirCreateConversionConvertFuncToLLVMPass())

        try pm.run(on: module)
    }

    /// Check if a module is ready for execution
    ///
    /// A module is ready if it contains only LLVM dialect operations
    public static func isExecutable(module: MLIRModule) -> Bool {
        // For now, we'll assume the module is executable if lowering succeeded
        // In the future, we could walk the IR and verify all ops are LLVM dialect
        return true
    }
}

// MARK: - Convenience Functions

/// Lower a module using the standard pipeline
public func lowerToLLVM(module: MLIRModule, context: MLIRContext) throws {
    let pipeline = LoweringPipeline(context: context)
    try pipeline.lower(module: module)
}

/// Lower a module with debug output
public func lowerToLLVMWithDebug(module: MLIRModule, context: MLIRContext) throws {
    let pipeline = LoweringPipeline(context: context)
    try pipeline.lowerWithDebugOutput(module: module)
}
