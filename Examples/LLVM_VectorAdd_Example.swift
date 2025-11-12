import SwiftIRCore
import SwiftIRXLA

/// Example: Vector Addition with LLVM/PTX Backend
///
/// This example demonstrates the complete Linalg → GPU → LLVM/PTX pipeline:
/// 1. Create Linalg operations for vector addition
/// 2. Lower through SCF parallel loops
/// 3. Convert to GPU dialect
/// 4. Generate LLVM IR with NVVM intrinsics
///
/// The resulting LLVM IR can be compiled to PTX using LLVM's PTX backend.

print("========================================")
print("SwiftIR: GPU Vector Addition (LLVM/PTX Backend)")
print("========================================")

do {
    // Step 1: Initialize MLIR context with all required dialects
    print("\n[1/4] Initializing MLIR context with dialects...")
    let context = MLIRContext()
    context.loadAllDialects()
    context.registerFuncDialect()
    context.registerLinalgDialect()
    context.registerGPUDialect()
    context.registerAllGPUPasses()
    context.registerAllConversionPasses()
    print("   ✓ Context initialized with Linalg, SCF, GPU, LLVM dialects")

    // Step 2: Create Linalg IR for vector addition: C[i] = A[i] + B[i]
    print("\n[2/4] Creating Linalg IR for vector addition...")

    let vectorSize = 256
    let linalgIR = """
    func.func @vector_add(%arg0: tensor<\(vectorSize)xf32>, %arg1: tensor<\(vectorSize)xf32>) -> tensor<\(vectorSize)xf32> {
      %0 = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<\(vectorSize)xf32>, tensor<\(vectorSize)xf32>)
        outs(%arg0 : tensor<\(vectorSize)xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %sum = arith.addf %a, %b : f32
          linalg.yield %sum : f32
      } -> tensor<\(vectorSize)xf32>
      return %0 : tensor<\(vectorSize)xf32>
    }
    """

    let module = try MLIRModule.parse(linalgIR, context: context)

    if !module.verify() {
        print("❌ Module verification failed")
        throw MLIRError.verificationFailed("Input Linalg module")
    }
    print("   ✓ Linalg module created and verified")
    print("   📝 Operation: C[i] = A[i] + B[i] for \(vectorSize) elements")

    // Step 3: Run LLVM/PTX lowering pipeline
    print("\n[3/4] Running Linalg → GPU → LLVM/PTX lowering pipeline...")

    let options = LLVMPipelineOptions(
        indexBitwidth: 64,
        computeCapability: "sm_80",  // NVIDIA A100
        dumpIR: false  // Set to true to see IR at each stage
    )

    let pipeline = LLVMPipeline(context: context, options: options)
    try pipeline.run(on: module)

    // Step 4: Display results
    print("\n[4/4] Lowering complete! Generated LLVM IR:")
    print("─────────────────────────────────────────")

    let finalIR = module.dump()

    // Show a snippet of the LLVM IR (first 50 lines)
    let lines = finalIR.split(separator: "\n")
    let previewLines = min(50, lines.count)
    print("\nShowing first \(previewLines) lines of LLVM IR:")
    print("─────────────────────────────────────────")
    for (i, line) in lines.prefix(previewLines).enumerated() {
        print(String(format: "%3d: %@", i + 1, String(line)))
    }

    if lines.count > 50 {
        print("... (\(lines.count - 50) more lines)")
    }

    print("\n─────────────────────────────────────────")
    print("\n✅ SUCCESS! GPU code generated via LLVM backend")
    print("\n📊 Pipeline Summary:")
    print("   • Input: High-level Linalg operations")
    print("   • Stage 1: Linalg → SCF parallel loops")
    print("   • Stage 2: SCF → GPU dialect (kernel outlining)")
    print("   • Stage 3: GPU → LLVM IR with NVVM intrinsics")
    print("   • Output: PTX-ready LLVM IR")

    print("\n🎯 Next Steps:")
    print("   1. Save LLVM IR to file: module.ll")
    print("   2. Compile to PTX: llc -march=nvptx64 -mcpu=sm_80 module.ll -o kernel.ptx")
    print("   3. Load PTX in CUDA runtime for execution")

    print("\n💡 Why LLVM Backend:")
    print("   • More mature than SPIR-V in current MLIR build")
    print("   • Index type conversion works automatically")
    print("   • Better documented and tested")
    print("   • Direct path to NVIDIA GPUs via PTX")

} catch {
    print("\n❌ ERROR: \(error)")
    if let mlirError = error as? MLIRError {
        print("   MLIR Error: \(mlirError)")
    } else if let pipelineError = error as? LLVMPipelineError {
        print("   Pipeline Error: \(pipelineError)")
    }
}
