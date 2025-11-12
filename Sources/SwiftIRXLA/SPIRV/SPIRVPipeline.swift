//===-- SPIRVPipeline.swift - SPIR-V Lowering Pipeline ----*- Swift -*-===//
//
// SwiftIR - Phase 11C: SPIR-V Integration
// Multi-stage lowering pipeline: Linalg → GPU → SPIR-V → Binary
//
//===------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import MLIRCoreWrapper

/// SPIR-V lowering and serialization pipeline
///
/// This class orchestrates the multi-stage lowering process:
/// 1. Linalg → GPU dialect (parallelization + mapping)
/// 2. GPU → SPIR-V dialect (conversion to shader operations)
/// 3. SPIR-V serialization to binary (for Vulkan/Metal consumption)
public class SPIRVPipeline {
    /// MLIR context for all operations
    private let context: MLIRContext

    /// Configuration options
    public struct Options {
        /// Target SPIR-V version (e.g., "1.3", "1.5")
        public var spirvVersion: String = "1.5"

        /// Target execution environment
        public enum ExecutionEnvironment {
            case vulkan        // Vulkan API
            case openCL        // OpenCL
            case openGL        // OpenGL/GLSL
        }
        public var environment: ExecutionEnvironment = .vulkan

        /// Workgroup size for GPU kernels (x, y, z)
        public var workgroupSize: (Int, Int, Int) = (32, 1, 1)

        /// Enable verification after each pass
        public var verifyAfterEachPass: Bool = true

        /// Print IR after each stage (for debugging)
        public var dumpIR: Bool = false

        public init() {}
    }

    public let options: Options

    /// Initialize a SPIR-V pipeline
    ///
    /// - Parameters:
    ///   - context: MLIR context (must have GPU and SPIR-V dialects registered)
    ///   - options: Pipeline configuration options
    public init(context: MLIRContext, options: Options = Options()) {
        self.context = context
        self.options = options

        // Ensure required dialects are registered
        context.registerGPUDialect()
        context.registerSPIRVDialect()
    }

    /// Lower Linalg operations to SPIR-V dialect
    ///
    /// This performs the complete modern lowering pipeline (2024 approach):
    /// - Stage 1: Linalg → Affine loops (enables optimizations)
    /// - Stage 2: Affine loops → GPU dialect (maps to GPU execution model)
    /// - Stage 3: GPU dialect → SPIR-V dialect (target code generation)
    ///
    /// - Parameter module: MLIR module containing Linalg operations
    /// - Throws: SPIRVError if lowering fails
    public func lowerToSPIRV(module: MLIRModule) throws {
        print("🔄 Starting SPIR-V lowering pipeline...")

        // Stage 1: Linalg → Affine Loops
        print("   Stage 1: Linalg → Affine Loops")
        try runStage1_LinalgToAffineLoops(module: module)

        if options.dumpIR {
            print("\n   IR after Stage 1:")
            print(module.dump())
        }

        // Stage 2: Affine Loops → GPU Dialect
        print("   Stage 2: Affine Loops → GPU Dialect")
        try runStage2_AffineLoopsToGPU(module: module)

        if options.dumpIR {
            print("\n   IR after Stage 2:")
            print(module.dump())
        }

        // Stage 3: GPU → SPIR-V Dialect
        print("   Stage 3: GPU → SPIR-V Dialect")

        // Add SPIR-V ABI attributes before conversion
        let moduleWithABI = try addSPIRVABIAttributes(module: module)

        try runStage3_GPUToSPIRV(module: moduleWithABI)

        if options.dumpIR {
            print("\n   IR after Stage 3 (Final SPIR-V):")
            print(moduleWithABI.dump())
        }

        // Verify final module
        if !moduleWithABI.verify() {
            throw SPIRVError.verificationFailed("Final SPIR-V module verification failed")
        }

        print("✅ SPIR-V lowering complete!")
    }

    /// Serialize SPIR-V module to binary format
    ///
    /// The binary can be loaded into Vulkan as a shader module.
    ///
    /// - Parameter module: MLIR module containing SPIR-V dialect operations
    /// - Returns: SPIR-V binary as array of 32-bit words
    /// - Throws: SPIRVError if serialization fails
    public func serializeSPIRV(module: MLIRModule) throws -> [UInt32] {
        print("🔨 Serializing SPIR-V binary...")

        // TODO: Once SPIR-V serialization APIs are available:
        // 1. Find spirv.module operations
        // 2. Call mlirTranslateSPIRVModuleToBinary()
        // 3. Return binary data

        // For now, stub implementation
        print("⚠️  SPIR-V serialization stub")
        print("   Note: Full implementation requires MLIR SPIR-V translation APIs")

        // Return mock binary (SPIR-V magic number + minimal header)
        let spirvMagicNumber: UInt32 = 0x07230203  // SPIR-V magic number
        return [spirvMagicNumber, 0x00010500, 0, 0, 0]  // Version 1.5, minimal header
    }

    // MARK: - Stage Implementations

    /// Stage 1: Convert Linalg operations to parallel loops
    ///
    /// GPU-oriented approach: Use SCF parallel loops instead of affine
    /// This is easier for GPU lowering as SCF→GPU is well-supported
    ///
    /// Applies passes:
    /// - convert-linalg-to-parallel-loops
    private func runStage1_LinalgToAffineLoops(module: MLIRModule) throws {
        let pm = PassManager(context: context)

        // Enable verification if requested
        if options.verifyAfterEachPass {
            pm.enableVerifier(true)
        }

        // Add linalg-to-parallel-loops pass
        // This converts linalg.generic operations to scf.parallel loop nests
        let pipelineStr = "builtin.module(convert-linalg-to-parallel-loops)"
        let pipelineRef = pipelineStr.withCString { ptr in
            mlirStringRefCreateFromCString(ptr)
        }

        let parseResult = mlirParsePassPipeline(pm.asOpPassManager(), pipelineRef, { _, _ in }, nil)
        guard mlirLogicalResultIsSuccess(parseResult) else {
            throw SPIRVError.passFailed("convert-linalg-to-parallel-loops")
        }

        // Run the pass pipeline
        try pm.run(on: module)
    }

    /// Stage 2: Convert parallel loops to GPU dialect
    ///
    /// Modern MLIR approach: Three-step process
    /// 1. gpu-map-parallel-loops: Add mapping attributes to scf.parallel
    /// 2. convert-parallel-loops-to-gpu: Convert annotated loops to gpu.launch
    /// 3. gpu-kernel-outlining: Extract GPU kernels into gpu.func
    /// 4. lower-affine: Lower any affine.apply operations created during conversion
    ///
    /// Applies passes:
    /// - gpu-map-parallel-loops (annotate loops with GPU dimension mapping)
    /// - convert-parallel-loops-to-gpu (convert to gpu.launch using annotations)
    /// - gpu-kernel-outlining (extract kernels into separate functions)
    /// - gpu.module(lower-affine) (lower affine ops inside GPU kernels)
    private func runStage2_AffineLoopsToGPU(module: MLIRModule) throws {
        let pm = PassManager(context: context)

        if options.verifyAfterEachPass {
            pm.enableVerifier(true)
        }

        // Modern pipeline based on MLIR 2024 examples + affine lowering:
        // 1. gpu-map-parallel-loops: Adds mapping attributes (processor, map, bound)
        // 2. convert-parallel-loops-to-gpu: Uses attributes to create gpu.launch
        // 3. gpu-kernel-outlining: Extracts GPU kernels into gpu.func
        // 4. lower-affine: Lower ALL affine.apply (host + kernel) to arith ops
        let pipelineStr = "builtin.module(func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu),gpu-kernel-outlining,lower-affine)"
        let pipelineRef = pipelineStr.withCString { ptr in
            mlirStringRefCreateFromCString(ptr)
        }

        let parseResult = mlirParsePassPipeline(pm.asOpPassManager(), pipelineRef, { _, _ in }, nil)
        guard mlirLogicalResultIsSuccess(parseResult) else {
            throw SPIRVError.passFailed("parallel-loops-to-gpu")
        }

        try pm.run(on: module)
    }

    /// Add SPIR-V entry point ABI attributes to GPU kernel functions
    ///
    /// The convert-gpu-to-spirv pass requires gpu.func operations to have
    /// a spirv.entry_point_abi attribute. This method manually adds it via
    /// string manipulation and re-parsing (interim solution).
    ///
    /// - Returns: A new module with ABI attributes added
    /// - Throws: MLIRError if re-parsing fails
    private func addSPIRVABIAttributes(module: MLIRModule) throws -> MLIRModule {
        print("      Adding SPIR-V entry point ABI attributes to GPU kernels...")

        // Get the module's IR as string
        var ir = module.dump()

        // Build the ABI attribute string
        let abiAttr = "spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [\(options.workgroupSize.0), \(options.workgroupSize.1), \(options.workgroupSize.2)]>"

        // Pattern to match: } kernel attributes {known_block_size = ...}
        // We need to add the spirv.entry_point_abi to the attributes dict

        // Find all gpu.func operations with kernel attribute
        // Pattern in IR: "kernel attributes {known_block_size = array<i32: 1, 1, 1>} {"
        // We need to insert the ABI attribute inside the attributes dict

        // Use regex to find and replace the pattern
        let pattern = "(kernel attributes \\{[^}]+)(\\})"
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let nsString = ir as NSString
            let matches = regex.matches(in: ir, options: [], range: NSRange(location: 0, length: nsString.length))

            print("         Found \(matches.count) kernel function(s)")

            var modifiedIR = ir
            // Process matches in reverse order to maintain string indices
            for match in matches.reversed() {
                if match.numberOfRanges == 3 {
                    let beforeClosing = nsString.substring(with: match.range(at: 1))
                    let closingBrace = nsString.substring(with: match.range(at: 2))

                    let replacement = beforeClosing + ", " + abiAttr + closingBrace
                    let matchRange = match.range
                    modifiedIR = (modifiedIR as NSString).replacingCharacters(in: matchRange, with: replacement)
                    print("         ✓ Added ABI attribute to kernel")
                }
            }

            ir = modifiedIR
        } else {
            print("         ⚠️  Regex compilation failed")
        }

        // Debug: print a section of the modified IR to verify
        print("      Debug: Checking modified IR...")
        let debugLines = ir.split(separator: "\n")
        for (idx, line) in debugLines.enumerated() {
            if line.contains("gpu.func") || line.contains("spirv.entry_point_abi") {
                print("         Line \(idx): \(line)")
            }
        }

        // Re-parse the modified IR
        print("      Re-parsing module with ABI attributes...")
        let newModule = try MLIRModule.parse(ir, context: context)

        if !newModule.verify() {
            throw MLIRError.verificationFailed("Module with ABI attributes failed verification")
        }

        print("      ✓ ABI attributes added successfully")
        return newModule
    }

    /// Stage 3: Convert GPU dialect to SPIR-V dialect
    ///
    /// Uses MLIR's recommended "progressive lowering" approach for SPIR-V.
    /// Each conversion pass handles one dialect at a time, with reconciliation between.
    ///
    /// Pipeline (LLVM 19+ approach):
    /// 1. Attach SPIR-V target to GPU modules
    /// 2. Progressive lowering (index → arith → func → reconcile → gpu)
    /// 3. Finalize SPIR-V modules
    ///
    /// Applies passes:
    /// - spirv-attach-target - attach SPIR-V environment
    /// - convert-index-to-spirv - convert index dialect ops
    /// - convert-arith-to-spirv - convert arithmetic ops
    /// - convert-func-to-spirv - convert function signatures
    /// - reconcile-unrealized-casts - clean up type conversion casts
    /// - convert-gpu-to-spirv - convert GPU kernel operations
    /// - gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)) - finalize
    private func runStage3_GPUToSPIRV(module: MLIRModule) throws {
        // Attach diagnostic handler to capture detailed MLIR error messages
        print("      Installing diagnostic handler for detailed error reporting...")

        var diagnosticUserData: UnsafeMutableRawPointer? = nil
        let diagnosticHandler: @convention(c) (MlirDiagnostic, UnsafeMutableRawPointer?) -> MlirLogicalResult = { diagnostic, _ in
            // Get severity - compare raw values
            let severity = mlirDiagnosticGetSeverity(diagnostic)
            let severityStr: String
            // MlirDiagnosticSeverity enum values: 0=Error, 1=Warning, 2=Note, 3=Remark
            let rawValue = unsafeBitCast(severity, to: UInt32.self)
            switch rawValue {
            case 0:
                severityStr = "ERROR"
            case 1:
                severityStr = "WARNING"
            case 2:
                severityStr = "NOTE"
            case 3:
                severityStr = "REMARK"
            default:
                severityStr = "UNKNOWN(\(rawValue))"
            }

            // Get diagnostic message using mlirDiagnosticPrint
            var message = ""
            mlirDiagnosticPrint(diagnostic, { strPtr, length, userData in
                if let userData = userData?.assumingMemoryBound(to: String.self),
                   let strPtr = strPtr {
                    // Reinterpret CChar (Int8) as UInt8 for UTF8 decoding
                    let uint8Ptr = UnsafeRawPointer(strPtr).bindMemory(to: UInt8.self, capacity: length)
                    let buffer = UnsafeBufferPointer(start: uint8Ptr, count: length)
                    userData.pointee += String(decoding: buffer, as: UTF8.self)
                }
            }, &message)

            // Print formatted diagnostic
            print("      ⚠️  MLIR Diagnostic [\(severityStr)]")
            print("          Message: \(message)")

            // Print any notes attached to this diagnostic
            let numNotes = mlirDiagnosticGetNumNotes(diagnostic)
            if numNotes > 0 {
                print("          Notes:")
                for i in 0..<numNotes {
                    let note = mlirDiagnosticGetNote(diagnostic, i)
                    var noteMessage = ""
                    mlirDiagnosticPrint(note, { strPtr, length, userData in
                        if let userData = userData?.assumingMemoryBound(to: String.self),
                           let strPtr = strPtr {
                            // Reinterpret CChar (Int8) as UInt8 for UTF8 decoding
                            let uint8Ptr = UnsafeRawPointer(strPtr).bindMemory(to: UInt8.self, capacity: length)
                            let buffer = UnsafeBufferPointer(start: uint8Ptr, count: length)
                            userData.pointee += String(decoding: buffer, as: UTF8.self)
                        }
                    }, &noteMessage)
                    print("            - \(noteMessage)")
                }
            }

            // Return success (MlirLogicalResult is a struct with value 1 for success)
            var result = MlirLogicalResult()
            result.value = 1
            return result
        }

        let diagnosticHandlerID = mlirContextAttachDiagnosticHandler(
            context.handle,
            diagnosticHandler,
            diagnosticUserData,
            nil
        )

        defer {
            mlirContextDetachDiagnosticHandler(context.handle, diagnosticHandlerID)
        }

        let pm = PassManager(context: context)

        // Enable verification to get error messages
        pm.enableVerifier(true)

        // SPIR-V lowering pipeline
        // According to MLIR docs, convert-gpu-to-spirv should handle all internal conversions
        // (arith, memref, scf, index) automatically when they are inside GPU kernels.
        // The key is using the correct nesting and options.
        //
        // Pipeline:
        // 1. convert-gpu-to-spirv with use-64bit-index=false option
        //    This automatically handles index→i32, arith→spirv, memref→spirv inside kernels
        print("      Running GPU→SPIR-V conversion...")

        // According to MLIR GPU→SPIR-V documentation, the convert-gpu-to-spirv pass
        // should automatically lower index, arith, memref, etc. when use-64bit-index is set
        print("      Pipeline: gpu-to-spirv with auto-conversion of internal ops")
        let pipelineStr = "builtin.module(convert-gpu-to-spirv{use-64bit-index=false})"
        let pipelineRef = pipelineStr.withCString { ptr in
            mlirStringRefCreateFromCString(ptr)
        }

        print("      Parsing pass pipeline: \(pipelineStr)")
        let parseResult = mlirParsePassPipeline(pm.asOpPassManager(), pipelineRef, { _, _ in }, nil)
        guard mlirLogicalResultIsSuccess(parseResult) else {
            print("      ❌ Failed to parse pass pipeline!")
            throw SPIRVError.passFailed("gpu-to-spirv pipeline parse")
        }

        print("      ✓ Pipeline parsed successfully, running passes...")

        try pm.run(on: module)
    }
}

/// SPIR-V pipeline errors
public enum SPIRVError: Error {
    case dialectNotRegistered(String)
    case passFailed(String)
    case verificationFailed(String)
    case serializationFailed(String)

    public var localizedDescription: String {
        switch self {
        case .dialectNotRegistered(let dialect):
            return "SPIR-V dialect not registered: \(dialect)"
        case .passFailed(let pass):
            return "SPIR-V pass failed: \(pass)"
        case .verificationFailed(let msg):
            return "SPIR-V verification failed: \(msg)"
        case .serializationFailed(let msg):
            return "SPIR-V serialization failed: \(msg)"
        }
    }
}

// MARK: - C API Imports for Diagnostics

/// Diagnostic handler callback type
typealias MlirDiagnosticHandlerCallback = @convention(c) (MlirDiagnostic, UnsafeMutableRawPointer?) -> MlirLogicalResult

/// String callback type for mlirDiagnosticPrint
typealias MlirStringCallback = @convention(c) (UnsafePointer<CChar>?, Int, UnsafeMutableRawPointer?) -> Void

@_silgen_name("mlirContextAttachDiagnosticHandler")
func mlirContextAttachDiagnosticHandler(
    _ context: MlirContext,
    _ handler: @escaping MlirDiagnosticHandlerCallback,
    _ userData: UnsafeMutableRawPointer?,
    _ deleteUserData: (@convention(c) (UnsafeMutableRawPointer?) -> Void)?
) -> MlirDiagnosticHandlerID

@_silgen_name("mlirContextDetachDiagnosticHandler")
func mlirContextDetachDiagnosticHandler(_ context: MlirContext, _ id: MlirDiagnosticHandlerID)

@_silgen_name("mlirDiagnosticPrint")
func mlirDiagnosticPrint(
    _ diagnostic: MlirDiagnostic,
    _ callback: @escaping MlirStringCallback,
    _ userData: UnsafeMutableRawPointer?
)

@_silgen_name("mlirDiagnosticGetSeverity")
func mlirDiagnosticGetSeverity(_ diagnostic: MlirDiagnostic) -> MlirDiagnosticSeverity

@_silgen_name("mlirDiagnosticGetLocation")
func mlirDiagnosticGetLocation(_ diagnostic: MlirDiagnostic) -> MlirLocation

@_silgen_name("mlirDiagnosticGetNumNotes")
func mlirDiagnosticGetNumNotes(_ diagnostic: MlirDiagnostic) -> Int

@_silgen_name("mlirDiagnosticGetNote")
func mlirDiagnosticGetNote(_ diagnostic: MlirDiagnostic, _ index: Int) -> MlirDiagnostic
