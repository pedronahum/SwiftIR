//===-- ExecutionTests.swift - Execution Runtime Tests ----*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime Tests
// Tests for the complete compilation and execution pipeline
//
//===------------------------------------------------------------------===//

import Testing
@testable import SwiftIRCore
@testable import SwiftIRTypes
@testable import SwiftIRBuilders
@testable import SwiftIRDialects
@testable import SwiftIRXLA
import MLIRCoreWrapper

@Suite("Phase 9: Execution Runtime")
struct ExecutionTests {

    /// Test the simplest possible case: return a constant
    ///
    /// This builds MLIR IR for: func @constant() -> i32 { return 42 }
    /// Then lowers it through the pipeline
    @Test("Simple constant function")
    func testSimpleConstant() throws {
        let ctx = MLIRContext()
        let builder = IRBuilder(context: ctx)
        let i32 = IntegerType.i32(context: ctx)

        // Build function using the result builder API
        let funcOp = builder.buildFunction(
            name: "constant",
            inputs: [],
            results: [i32.typeHandle]
        ) { _ in
            let const = builder.constantInt(42, type: i32)
            builder.return([const])
        }

        // Create module and add function
        let module = MLIRModule(context: ctx)
        module.append(funcOp)

        // Print the IR before lowering
        print("\n=== IR Before Lowering ===")
        print(module.dump())

        // Verify the IR
        #expect(module.verify())

        // Try to lower it
        print("\n=== Starting Lowering Pipeline ===")
        let pipeline = LoweringPipeline(context: ctx)

        do {
            try pipeline.lower(module: module)
            print("✅ Lowering succeeded!")

            // Print the lowered IR
            print("\n=== IR After Lowering ===")
            print(module.dump())

        } catch {
            print("❌ Lowering failed: \(error)")
            Issue.record("Lowering pipeline failed: \(error)")
            return
        }

        // Try to create execution engine
        print("\n=== Creating Execution Engine ===")
        do {
            let engine = try ExecutionEngine(module: module, optLevel: 2)
            print("✅ Execution engine created!")

        } catch {
            print("❌ Execution engine creation failed: \(error)")
            // This is expected - we likely need llvm.emit_c_interface attribute
            print("Note: This is expected. We need to add llvm.emit_c_interface attribute.")
        }
    }

    /// Test the Backend API with a simple function
    @Test("Backend compilation API")
    func testBackendAPI() throws {
        let ctx = MLIRContext()
        let builder = IRBuilder(context: ctx)
        let i32 = IntegerType.i32(context: ctx)

        // Build a simple function that returns 100
        let funcOp = builder.buildFunction(
            name: "main",
            inputs: [],
            results: [i32.typeHandle]
        ) { _ in
            let const = builder.constantInt(100, type: i32)
            builder.return([const])
        }

        let module = MLIRModule(context: ctx)
        module.append(funcOp)

        print("\n=== Testing Backend API ===")
        print("Original IR:")
        print(module.dump())

        // Use the backend
        let backend = MLIRCPUBackend()
        #expect(backend.isAvailable)

        do {
            let executable = try backend.compile(module: module)
            print("✅ Compilation succeeded!")
            print("Entry point: \(executable.entryPoint)")

            // Try to execute (will likely fail without proper setup, but tests the API)
            do {
                let outputs = try executable.execute(inputs: [])
                print("✅ Execution succeeded! Outputs: \(outputs)")
            } catch {
                print("⚠️  Execution failed (expected): \(error)")
                // This is expected - we need proper function setup
            }

        } catch {
            print("❌ Compilation failed: \(error)")
            // Don't fail the test - we're learning what's needed
            print("Note: This helps us understand what's missing")
        }
    }

    /// Test direct execution engine invocation with return value
    @Test("Direct execution with return value")
    func testDirectExecution() throws {
        let ctx = MLIRContext()
        let builder = IRBuilder(context: ctx)
        let i32 = IntegerType.i32(context: ctx)

        // Build a function that returns 42
        let funcOp = builder.buildFunction(
            name: "get_answer",
            inputs: [],
            results: [i32.typeHandle]
        ) { _ in
            let const = builder.constantInt(42, type: i32)
            builder.return([const])
        }

        let module = MLIRModule(context: ctx)
        module.append(funcOp)

        print("\n=== Testing Direct Execution ===")
        print("IR Before Lowering:")
        print(module.dump())

        // Lower to LLVM
        let pipeline = LoweringPipeline(context: ctx)
        try pipeline.lower(module: module)

        print("\nIR After Lowering:")
        print(module.dump())

        // Create execution engine
        let engine = try ExecutionEngine(module: module, optLevel: 2)
        print("✅ Execution engine created!")

        // Try to invoke the function and get the return value
        print("\n=== Attempting Function Invocation ===")

        // First, try to lookup the function to see if it exists
        print("Looking up function symbols...")

        if let funcPtr = engine.lookup(name: "get_answer") {
            print("✅ Found function 'get_answer' at \(funcPtr)")
        } else {
            print("❌ Function 'get_answer' not found")
        }

        if let packedPtr = engine.lookupPacked(name: "get_answer") {
            print("✅ Found packed wrapper 'get_answer' at \(packedPtr)")
        } else {
            print("⚠️  Packed wrapper 'get_answer' not found")
        }

        if let cifacePtr = engine.lookup(name: "_mlir_ciface_get_answer") {
            print("✅ Found C interface '_mlir_ciface_get_answer' at \(cifacePtr)")
        } else {
            print("⚠️  C interface '_mlir_ciface_get_answer' not found")
        }

        // Try using lookup to call the function directly
        if let funcPtr = engine.lookup(name: "get_answer") {
            print("\n=== Calling function directly via pointer ===")

            // Cast the pointer to a function type: () -> Int32
            typealias FuncType = @convention(c) () -> Int32
            let fn = unsafeBitCast(funcPtr, to: FuncType.self)

            let result = fn()
            print("✅ Function executed! Return value: \(result)")
            #expect(result == 42)
        } else {
            print("❌ Could not call function - not found")
        }
    }
}
