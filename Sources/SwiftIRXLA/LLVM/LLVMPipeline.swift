//===-- LLVMPipeline.swift - GPU to LLVM/PTX Lowering Pipeline -*- Swift -*-===//
//
// SwiftIR - LLVM Backend Pipeline
// Lowers MLIR GPU dialect to LLVM IR and PTX for NVIDIA GPUs
//
//===----------------------------------------------------------------------===//

import SwiftIRCore
import Foundation

/// LLVM/PTX lowering pipeline for GPU kernels
///
/// This pipeline takes high-level Linalg operations and lowers them through
/// multiple MLIR dialects to LLVM IR suitable for NVIDIA PTX compilation:
///
/// Pipeline stages:
/// 1. Linalg → SCF parallel loops (structured control flow)
/// 2. SCF → GPU dialect (kernel outlining and launch operations)
/// 3. GPU → LLVM/NVVM dialect → PTX-compatible LLVM IR
///
/// This is the recommended backend for GPU code generation as it's more
/// mature and better supported than SPIR-V in the current MLIR build.
public class LLVMPipeline {
    private let context: MLIRContext
    private let options: LLVMPipelineOptions

    public init(context: MLIRContext, options: LLVMPipelineOptions = LLVMPipelineOptions()) {
        self.context = context
        self.options = options
    }

    /// Run the complete Linalg → GPU → LLVM/PTX lowering pipeline
    public func run(on module: MLIRModule) throws {
        print("\n🚀 Starting LLVM/PTX Lowering Pipeline")

        // Stage 1: Linalg → SCF Parallel Loops
        print("   Stage 1: Linalg → SCF Parallel Loops")
        try runStage1_LinalgToSCF(module: module)

        if options.dumpIR {
            print("\n   IR after Stage 1:")
            print(module.dump())
        }

        // Stage 2: SCF → GPU Dialect
        print("   Stage 2: SCF → GPU Dialect (kernel outlining)")
        try runStage2_SCFToGPU(module: module)

        if options.dumpIR {
            print("\n   IR after Stage 2:")
            print(module.dump())
        }

        // Stage 3: GPU → LLVM/NVVM Dialect
        print("   Stage 3: GPU → LLVM/NVVM → PTX")
        try runStage3_GPUToLLVM(module: module)

        if options.dumpIR {
            print("\n   IR after Stage 3 (Final LLVM IR):")
            print(module.dump())
        }

        // Verify final module
        if !module.verify() {
            throw LLVMPipelineError.verificationFailed("Final LLVM module verification failed")
        }

        print("✅ LLVM/PTX lowering complete!")
        print("   💡 Next step: Generate PTX with LLVM's llc tool")
        print("      llc -march=nvptx64 -mcpu=sm_80 <module.ll> -o kernel.ptx")
    }

    /// Stage 1: Lower Linalg operations to SCF parallel loops
    ///
    /// Converts high-level tensor operations (linalg.generic, linalg.matmul, etc.)
    /// into structured control flow with parallel loops.
    ///
    /// Applies passes:
    /// - convert-linalg-to-parallel-loops
    private func runStage1_LinalgToSCF(module: MLIRModule) throws {
        let pm = PassManager(context: context)
        pm.enableVerifier(true)

        try pm.parsePipeline("builtin.module(convert-linalg-to-parallel-loops)")
        try pm.run(on: module)
    }

    /// Stage 2: Lower SCF parallel loops to GPU dialect
    ///
    /// Converts parallel loops into GPU kernel launches with proper mapping
    /// to GPU thread hierarchy (blocks and threads).
    ///
    /// Applies passes:
    /// - gpu-map-parallel-loops (map iterations to GPU threads)
    /// - convert-parallel-loops-to-gpu (outline kernels)
    /// - lower-affine (eliminate affine operations)
    /// - convert-scf-to-cf (lower remaining SCF to control flow)
    private func runStage2_SCFToGPU(module: MLIRModule) throws {
        let pm = PassManager(context: context)
        pm.enableVerifier(true)

        // Map parallel loops to GPU thread hierarchy
        let pipelineStr = """
        builtin.module(
            gpu-map-parallel-loops,
            convert-parallel-loops-to-gpu,
            lower-affine,
            convert-scf-to-cf
        )
        """

        try pm.parsePipeline(pipelineStr)
        try pm.run(on: module)
    }

    /// Stage 3: Lower GPU dialect to LLVM/NVVM for PTX generation
    ///
    /// Converts GPU kernels to LLVM IR with NVVM (NVIDIA LLVM) intrinsics.
    /// This IR can then be compiled to PTX by LLVM's PTX backend.
    ///
    /// Pipeline:
    /// 1. convert-gpu-to-nvvm - Convert GPU operations to NVVM dialect
    /// 2. gpu-to-llvm - Lower GPU launch operations to LLVM
    /// 3. convert-index-to-llvm - Convert index type to i64
    /// 4. convert-func-to-llvm - Convert function signatures to LLVM
    /// 5. reconcile-unrealized-casts - Clean up type conversion artifacts
    /// 6. canonicalize - Apply canonical patterns
    /// 7. cse - Common subexpression elimination
    ///
    /// The resulting LLVM IR contains:
    /// - GPU kernels as LLVM functions with NVVM metadata
    /// - Host-side kernel launch logic
    /// - Proper linkage for device/host split
    private func runStage3_GPUToLLVM(module: MLIRModule) throws {
        let pm = PassManager(context: context)
        pm.enableVerifier(true)

        // GPU → LLVM/NVVM lowering pipeline
        // This is much more mature and well-tested than GPU → SPIR-V
        let pipelineStr = """
        builtin.module(
            gpu.module(
                strip-debuginfo,
                convert-gpu-to-nvvm{index-bitwidth=\(options.indexBitwidth)}
            ),
            gpu-to-llvm,
            convert-index-to-llvm{index-bitwidth=\(options.indexBitwidth)},
            convert-func-to-llvm,
            reconcile-unrealized-casts,
            canonicalize,
            cse
        )
        """

        print("      Parsing LLVM lowering pipeline...")
        try pm.parsePipeline(pipelineStr)

        print("      Running GPU→LLVM conversion...")
        try pm.run(on: module)

        print("      ✅ Successfully generated LLVM IR with NVVM intrinsics!")
    }
}

/// Configuration options for the LLVM lowering pipeline
public struct LLVMPipelineOptions {
    /// Index type bitwidth (32 or 64)
    public var indexBitwidth: Int = 64

    /// Target GPU compute capability (e.g., sm_80 for A100)
    public var computeCapability: String = "sm_80"

    /// Whether to dump IR after each stage
    public var dumpIR: Bool = false

    public init(
        indexBitwidth: Int = 64,
        computeCapability: String = "sm_80",
        dumpIR: Bool = false
    ) {
        self.indexBitwidth = indexBitwidth
        self.computeCapability = computeCapability
        self.dumpIR = dumpIR
    }
}

/// LLVM pipeline errors
public enum LLVMPipelineError: Error {
    case passFailed(String)
    case verificationFailed(String)

    public var localizedDescription: String {
        switch self {
        case .passFailed(let pass):
            return "LLVM pipeline pass failed: \(pass)"
        case .verificationFailed(let msg):
            return "LLVM pipeline verification failed: \(msg)"
        }
    }
}
