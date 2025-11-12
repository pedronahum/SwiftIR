//===-- ExecutionEngine.swift - MLIR Execution Engine ----*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime
// Swift wrapper for MLIR's JIT execution engine
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import MLIRCoreWrapper
import Foundation

/// JIT execution engine for running MLIR code
///
/// The ExecutionEngine compiles LLVM dialect to native code and executes it
public class ExecutionEngine {
    let handle: MlirExecutionEngine
    private var isDestroyed = false

    /// Create an execution engine from a module
    ///
    /// The module must be in LLVM dialect (i.e., already lowered)
    ///
    /// - Parameters:
    ///   - module: The MLIR module (must contain only LLVM dialect ops)
    ///   - optLevel: Optimization level (0-3), default is 2
    ///   - enableObjectDump: Dump object files for debugging, default false
    public init(module: MLIRModule, optLevel: Int = 2, enableObjectDump: Bool = false) throws {
        // Create the execution engine
        let engine = mlirExecutionEngineCreate(
            module.handle,
            Int32(optLevel),
            0,  // numPaths (no shared libraries)
            nil,  // sharedLibPaths
            enableObjectDump
        )

        // Check if creation failed
        // mlirExecutionEngineIsNull is an inline function, so we check the ptr directly
        if engine.ptr == nil {
            throw ExecutionEngineError.creationFailed(
                "Failed to create execution engine. Ensure the module is in LLVM dialect."
            )
        }

        self.handle = engine
    }

    deinit {
        if !isDestroyed {
            mlirExecutionEngineDestroy(handle)
        }
    }

    /// Invoke a function by name
    ///
    /// The function must have been tagged with llvm.emit_c_interface attribute
    ///
    /// - Parameters:
    ///   - name: The function name
    ///   - arguments: Array of argument pointers (inputs and outputs)
    public func invokePacked(name: String, arguments: inout [UnsafeMutableRawPointer?]) throws {
        let nameRef = mlirStringRefCreateFromCStringWrapper(name)
        let result = mlirExecutionEngineInvokePacked(handle, nameRef, &arguments)

        if !mlirLogicalResultIsSuccess(result) {
            throw ExecutionEngineError.invocationFailed(
                "Failed to invoke function '\(name)'. Check that the function exists and has llvm.emit_c_interface attribute."
            )
        }
    }

    /// Lookup a symbol in the execution engine
    ///
    /// Returns a pointer to the function/symbol
    public func lookup(name: String) -> UnsafeMutableRawPointer? {
        let nameRef = mlirStringRefCreateFromCStringWrapper(name)
        return mlirExecutionEngineLookup(handle, nameRef)
    }

    /// Lookup a packed wrapper for a function
    public func lookupPacked(name: String) -> UnsafeMutableRawPointer? {
        let nameRef = mlirStringRefCreateFromCStringWrapper(name)
        return mlirExecutionEngineLookupPacked(handle, nameRef)
    }

    /// Register a symbol with the JIT
    ///
    /// Makes a host symbol accessible to JIT-compiled code
    public func registerSymbol(name: String, pointer: UnsafeMutableRawPointer) {
        let nameRef = mlirStringRefCreateFromCStringWrapper(name)
        mlirExecutionEngineRegisterSymbol(handle, nameRef, pointer)
    }

    /// Dump the compiled object to a file (for debugging)
    public func dumpToObjectFile(fileName: String) {
        let fileNameRef = mlirStringRefCreateFromCStringWrapper(fileName)
        mlirExecutionEngineDumpToObjectFile(handle, fileNameRef)
    }
}

/// Errors that can occur during execution
public enum ExecutionEngineError: Error, CustomStringConvertible {
    case creationFailed(String)
    case invocationFailed(String)
    case symbolNotFound(String)

    public var description: String {
        switch self {
        case .creationFailed(let msg):
            return "Execution engine creation failed: \(msg)"
        case .invocationFailed(let msg):
            return "Function invocation failed: \(msg)"
        case .symbolNotFound(let msg):
            return "Symbol not found: \(msg)"
        }
    }
}

// MARK: - C API Imports

@_silgen_name("mlirExecutionEngineCreate")
func mlirExecutionEngineCreate(
    _ module: MlirModule,
    _ optLevel: Int32,
    _ numPaths: Int32,
    _ sharedLibPaths: UnsafePointer<MlirStringRef>?,
    _ enableObjectDump: Bool
) -> MlirExecutionEngine

@_silgen_name("mlirExecutionEngineDestroy")
func mlirExecutionEngineDestroy(_ engine: MlirExecutionEngine)

@_silgen_name("mlirExecutionEngineInvokePacked")
func mlirExecutionEngineInvokePacked(
    _ engine: MlirExecutionEngine,
    _ name: MlirStringRef,
    _ arguments: UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> MlirLogicalResult

@_silgen_name("mlirExecutionEngineLookup")
func mlirExecutionEngineLookup(
    _ engine: MlirExecutionEngine,
    _ name: MlirStringRef
) -> UnsafeMutableRawPointer?

@_silgen_name("mlirExecutionEngineLookupPacked")
func mlirExecutionEngineLookupPacked(
    _ engine: MlirExecutionEngine,
    _ name: MlirStringRef
) -> UnsafeMutableRawPointer?

@_silgen_name("mlirExecutionEngineRegisterSymbol")
func mlirExecutionEngineRegisterSymbol(
    _ engine: MlirExecutionEngine,
    _ name: MlirStringRef,
    _ sym: UnsafeMutableRawPointer
)

@_silgen_name("mlirExecutionEngineDumpToObjectFile")
func mlirExecutionEngineDumpToObjectFile(
    _ engine: MlirExecutionEngine,
    _ fileName: MlirStringRef
)
