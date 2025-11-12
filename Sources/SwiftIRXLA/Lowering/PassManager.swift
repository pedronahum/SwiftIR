//===-- PassManager.swift - MLIR Pass Manager Wrapper -----*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime
// Swift wrapper for MLIR's pass management infrastructure
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import MLIRCoreWrapper

/// Swift wrapper for MLIR PassManager
///
/// Manages transformation passes that lower high-level IR to executable form
public class PassManager {
    let handle: MlirPassManager
    private let context: MLIRContext

    /// Create a new pass manager
    public init(context: MLIRContext) {
        self.context = context
        self.handle = mlirPassManagerCreate(context.handle)
    }

    deinit {
        mlirPassManagerDestroy(handle)
    }

    /// Run the pass manager on a module
    public func run(on module: MLIRModule) throws {
        // Get the operation from the module
        let moduleOp = module.operation
        let result = mlirPassManagerRunOnOp(handle, moduleOp.handle)
        if !mlirLogicalResultIsSuccess(result) {
            throw PassError.passManagerFailed("Pass pipeline failed to execute")
        }
    }

    /// Get the underlying OpPassManager for adding passes
    public func asOpPassManager() -> MlirOpPassManager {
        return mlirPassManagerGetAsOpPassManager(handle)
    }

    /// Enable IR printing for debugging
    public func enableIRPrinting(
        printBeforeAll: Bool = false,
        printAfterAll: Bool = true,
        printModuleScope: Bool = true,
        printAfterOnlyOnChange: Bool = false,
        printAfterOnlyOnFailure: Bool = true
    ) {
        let flags = mlirOpPrintingFlagsCreate()
        let emptyPath = mlirStringRefCreateFromCString("")

        mlirPassManagerEnableIRPrinting(
            handle,
            printBeforeAll,
            printAfterAll,
            printModuleScope,
            printAfterOnlyOnChange,
            printAfterOnlyOnFailure,
            flags,
            emptyPath
        )

        mlirOpPrintingFlagsDestroy(flags)
    }

    /// Enable verification after each pass
    public func enableVerifier(_ enable: Bool = true) {
        mlirPassManagerEnableVerifier(handle, enable)
    }

    /// Parse and add a pass pipeline string
    public func parsePipeline(_ pipelineStr: String) throws {
        let pipelineRef = pipelineStr.withCString { ptr in
            mlirStringRefCreateFromCString(ptr)
        }

        let parseResult = mlirParsePassPipeline(asOpPassManager(), pipelineRef, { _, _ in }, nil)
        guard mlirLogicalResultIsSuccess(parseResult) else {
            throw PassError.pipelineParseFailed("Failed to parse pipeline: \(pipelineStr)")
        }
    }
}

/// Errors that can occur during pass execution
public enum PassError: Error, CustomStringConvertible {
    case passManagerFailed(String)
    case passRegistrationFailed(String)
    case pipelineParseFailed(String)

    public var description: String {
        switch self {
        case .passManagerFailed(let msg):
            return "Pass manager failed: \(msg)"
        case .passRegistrationFailed(let msg):
            return "Pass registration failed: \(msg)"
        case .pipelineParseFailed(let msg):
            return "Pipeline parse failed: \(msg)"
        }
    }
}

// MARK: - Built-in Pass Registration

/// Register all built-in MLIR passes
public func registerAllPasses() {
    // Register conversion passes
    mlirRegisterConversionPasses()

    // Register linalg passes
    mlirRegisterLinalgPasses()
}

// MARK: - C API Imports

// Pass.h
@_silgen_name("mlirPassManagerCreate")
func mlirPassManagerCreate(_ ctx: MlirContext) -> MlirPassManager

@_silgen_name("mlirPassManagerDestroy")
func mlirPassManagerDestroy(_ pm: MlirPassManager)

@_silgen_name("mlirPassManagerGetAsOpPassManager")
func mlirPassManagerGetAsOpPassManager(_ pm: MlirPassManager) -> MlirOpPassManager

@_silgen_name("mlirPassManagerRunOnOp")
func mlirPassManagerRunOnOp(_ pm: MlirPassManager, _ op: MlirOperation) -> MlirLogicalResult

@_silgen_name("mlirPassManagerEnableIRPrinting")
func mlirPassManagerEnableIRPrinting(
    _ pm: MlirPassManager,
    _ printBeforeAll: Bool,
    _ printAfterAll: Bool,
    _ printModuleScope: Bool,
    _ printAfterOnlyOnChange: Bool,
    _ printAfterOnlyOnFailure: Bool,
    _ flags: MlirOpPrintingFlags,
    _ treePrintingPath: MlirStringRef
)

@_silgen_name("mlirPassManagerEnableVerifier")
func mlirPassManagerEnableVerifier(_ pm: MlirPassManager, _ enable: Bool)

@_silgen_name("mlirOpPassManagerAddOwnedPass")
func mlirOpPassManagerAddOwnedPass(_ pm: MlirOpPassManager, _ pass: MlirPass)

// Conversion.h
@_silgen_name("mlirRegisterConversionPasses")
func mlirRegisterConversionPasses()

@_silgen_name("mlirCreateConversionConvertLinalgToLoopsPass")
func mlirCreateConversionConvertLinalgToLoopsPass() -> MlirPass

@_silgen_name("mlirCreateConversionConvertFuncToLLVMPass")
func mlirCreateConversionConvertFuncToLLVMPass() -> MlirPass

@_silgen_name("mlirCreateConversionSCFToControlFlowPass")
func mlirCreateConversionSCFToControlFlowPass() -> MlirPass

@_silgen_name("mlirCreateConversionFinalizeMemRefToLLVMConversionPass")
func mlirCreateConversionFinalizeMemRefToLLVMConversionPass() -> MlirPass

@_silgen_name("mlirCreateConversionArithToLLVMConversionPass")
func mlirCreateConversionArithToLLVMConversionPass() -> MlirPass

// Linalg passes
@_silgen_name("mlirRegisterLinalgPasses")
func mlirRegisterLinalgPasses()

@_silgen_name("mlirCreateLinalgConvertLinalgToLoopsPass")
func mlirCreateLinalgConvertLinalgToLoopsPass() -> MlirPass

// Pass pipeline parsing
@_silgen_name("mlirParsePassPipeline")
func mlirParsePassPipeline(
    _ pm: MlirOpPassManager,
    _ pipelineText: MlirStringRef,
    _ callback: (@convention(c) (MlirStringRef, Int, UnsafeMutableRawPointer?) -> Void)?,
    _ userData: UnsafeMutableRawPointer?
) -> MlirLogicalResult

@_silgen_name("mlirStringRefCreateFromCString")
func mlirStringRefCreateFromCString(_ str: UnsafePointer<CChar>) -> MlirStringRef
