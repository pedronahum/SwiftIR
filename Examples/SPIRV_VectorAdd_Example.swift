//===-- SPIRV_VectorAdd_Example.swift - Real SPIR-V Lowering --*- Swift -*-===//
//
// SwiftIR - SPIR-V Integration with Real Linalg IR
// Demonstrates complete lowering pipeline:
// - Linalg operations (element-wise vector addition)
// - Lowering through GPU dialect
// - SPIR-V dialect generation
// - Binary serialization
//
//===--------------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import SwiftIRTypes
import SwiftIRXLA

print("========================================")
print("SwiftIR: Real SPIR-V Lowering Example")
print("Vector Addition: C = A + B")
print("========================================")

do {
    // Step 1: Initialize MLIR context with all required dialects
    print("\n[1/5] Initializing MLIR context with dialects...")
    let context = MLIRContext()
    context.loadAllDialects()
    context.registerFuncDialect()  // Required for parsing func.func operations
    context.registerLinalgDialect()
    context.registerGPUDialect()
    context.registerSPIRVDialect()
    context.registerAllLinalgPasses()  // Required for Linalg transformation passes
    context.registerAllGPUPasses()     // Required for GPU transformation passes (includes spirv-attach-target)
    context.registerAllConversionPasses()  // Required for all conversion passes
    print("      ✓ All dialects registered (func, linalg, GPU, SPIR-V)")
    print("      ✓ All passes registered (Linalg, GPU, Conversion)")

    // Step 2: Create Linalg IR for vector addition
    print("\n[2/5] Creating Linalg IR for vector addition...")

    // This MLIR represents: C[i] = A[i] + B[i] for a 256-element vector
    let linalgIR = """
    module {
      func.func @vector_add(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) {
        linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        }
        ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>)
        outs(%arg2 : memref<256xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
          %sum = arith.addf %in0, %in1 : f32
          linalg.yield %sum : f32
        }
        return
      }
    }
    """

    print("      Linalg IR:")
    for line in linalgIR.split(separator: "\n").prefix(10) {
        print("        \(line)")
    }
    print("        ...")
    print("      ✓ Linalg IR created (element-wise addition)")

    // Step 3: Parse the MLIR module
    print("\n[3/5] Parsing MLIR module...")
    let module = try MLIRModule.parse(linalgIR, context: context)
    guard module.verify() else {
        print("      ERROR: Module verification failed")
        exit(1)
    }
    print("      ✓ Module parsed and verified")

    // Step 4: Run SPIR-V lowering pipeline
    print("\n[4/5] Running SPIR-V lowering pipeline...")

    // Configure pipeline options
    var options = SPIRVPipeline.Options()
    options.spirvVersion = "1.5"
    options.environment = .vulkan
    options.workgroupSize = (256, 1, 1)  // 256 threads, 1 workgroup for 256 elements
    options.verifyAfterEachPass = true
    options.dumpIR = true  // Show IR after each stage

    let pipeline = SPIRVPipeline(context: context, options: options)

    print("")
    print("   Pipeline Configuration:")
    print("      • SPIR-V version: \(options.spirvVersion)")
    print("      • Environment: vulkan")
    print("      • Workgroup size: \(options.workgroupSize.0) × \(options.workgroupSize.1) × \(options.workgroupSize.2)")
    print("")

    // Run the lowering
    try pipeline.lowerToSPIRV(module: module)

    // Step 5: Serialize to SPIR-V binary
    print("\n[5/5] Serializing SPIR-V binary...")
    let spirvBinary = try pipeline.serializeSPIRV(module: module)
    print("      ✓ SPIR-V binary generated")
    print("      Size: \(spirvBinary.count * 4) bytes (\(spirvBinary.count) words)")
    print("")
    print("      Binary header (first 5 words):")
    for (i, word) in spirvBinary.prefix(5).enumerated() {
        print("        [\(i)]: 0x\(String(word, radix: 16, uppercase: true).padding(toLength: 8, withPad: "0", startingAt: 0))")
    }

    // Step 6: Display summary
    print("\n========================================")
    print("LOWERING PIPELINE COMPLETE")
    print("========================================")

    print("\nStages executed:")
    print("  ✓ Stage 1: Linalg → Parallel Loops")
    print("     - Converted linalg.generic to scf.parallel")
    print("     - Extracted parallel iteration structure")
    print("")
    print("  ✓ Stage 2: Parallel Loops → GPU Dialect")
    print("     - Mapped loops to GPU grid/blocks")
    print("     - Created gpu.launch operations")
    print("     - Outlined GPU kernels")
    print("")
    print("  ✓ Stage 3: GPU → SPIR-V Dialect")
    print("     - Converted GPU ops to SPIR-V")
    print("     - Lowered functions and control flow")
    print("     - Generated spirv.module")
    print("")
    print("  ✓ Serialization: SPIR-V Binary")
    print("     - Ready for Vulkan/Metal consumption")

    print("\n========================================")
    print("WHAT THIS DEMONSTRATES")
    print("========================================")

    print("\n1. Real Linalg IR Input:")
    print("   • linalg.generic operation")
    print("   • Element-wise vector addition")
    print("   • Parallel iteration semantics")

    print("\n2. Multi-Stage Lowering:")
    print("   • Automatic parallelization detection")
    print("   • GPU kernel generation")
    print("   • Target-specific code generation")

    print("\n3. SPIR-V Output:")
    print("   • Portable GPU shader binary")
    print("   • Executable on Vulkan/Metal")
    print("   • Cross-platform compatibility")

    print("\n========================================")
    print("NEXT STEPS")
    print("========================================")

    print("\nTo enable full GPU execution:")
    print("  1. Install Vulkan SDK:")
    print("     macOS: brew install vulkan-sdk molten-vk")
    print("     Linux: sudo apt-get install vulkan-tools libvulkan-dev")
    print("")
    print("  2. Implement Vulkan bindings:")
    print("     • Device enumeration")
    print("     • Buffer creation")
    print("     • Shader module loading")
    print("     • Compute pipeline dispatch")
    print("")
    print("  3. Test end-to-end execution:")
    print("     • Upload data to GPU")
    print("     • Execute SPIR-V shader")
    print("     • Download results")
    print("     • Verify correctness")

    print("\n✨ SPIR-V lowering pipeline working!")
    print("========================================\n")

} catch {
    print("\n❌ ERROR: \(error)")
    exit(1)
}
