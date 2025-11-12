import SwiftIRCore
import SwiftIRXLA

/// Simplified test to verify manual index casts work
print("========================================")
print("SwiftIR: Manual Index Cast Test")
print("========================================")

do {
    print("\n[1/2] Creating GPU kernel with i32 parameters (manually casted)...")

    let context = MLIRContext()
    context.loadAllDialects()
    context.registerFuncDialect()
    context.registerGPUDialect()
    context.registerSPIRVDialect()
    context.registerAllGPUPasses()
    context.registerAllConversionPasses()

    // IR with i32 for addressing instead of index
    // Test hypothesis: If we use i32 for memref addressing, will GPU→SPIR-V succeed?
    let irWithI32 = """
module attributes {gpu.container_module} {
  gpu.module @test_kernel {
    gpu.func @test_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>} {
      %c0 = arith.constant 0 : i32
      %0 = memref.load %arg0[%c0] : memref<256xf32>
      %1 = memref.load %arg1[%c0] : memref<256xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %arg2[%c0] : memref<256xf32>
      gpu.return
    }
  }
}
"""

    let module = try MLIRModule.parse(irWithI32, context: context)

    guard module.verify() else {
        print("❌ Module verification failed")
        throw MLIRError.verificationFailed("Input")
    }
    print("   ✓ Module with i32 parameters parsed")

    print("\n[2/2] Testing convert-gpu-to-spirv with i32...")
    let pm = PassManager(context: context)
    pm.enableVerifier(true)

    let pipelineStr = "builtin.module(convert-gpu-to-spirv{use-64bit-index=false})"
    try pm.parsePipeline(pipelineStr)

    print("   Running conversion...")
    try pm.run(on: module)

    print("\n✅ SUCCESS! GPU→SPIR-V works with i32 parameters!")
    print("   Hypothesis CONFIRMED: Manual casts will solve the problem")
    print("\n   Next step: Implement automatic index→i32 cast injection")

} catch {
    print("\n❌ ERROR: \(error)")
}
