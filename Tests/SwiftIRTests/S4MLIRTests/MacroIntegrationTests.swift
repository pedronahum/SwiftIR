import Testing
@testable import SwiftIR
@testable import SwiftIRCore

// Helper struct to contain functions (peer macros work within types)
struct MLIRTestFunctions {
    // Create a simple test function with the macro
    @MLIRFunction
    func addIntegers(a: Int, b: Int) -> Int {
        return a + b
    }

    @MLIRFunction
    func noReturn() {
    }
}

@Suite("Macro Integration Tests")
struct MacroIntegrationTests {
    let testFuncs = MLIRTestFunctions()
    @Test("Macro generates MLIR function")
    func testMacroGeneratesMLIR() {
        let mlir = testFuncs.addIntegers_mlir()
        #expect(mlir.contains("func.func"))
        #expect(mlir.contains("@addIntegers"))
        #expect(mlir.contains("i64"))
    }

    @Test("Macro preserves function name")
    func testMacroPreservesName() {
        let mlir = testFuncs.addIntegers_mlir()
        #expect(mlir.contains("addIntegers"))
    }

    @Test("Macro handles void return")
    func testMacroVoidReturn() {
        let mlir = testFuncs.noReturn_mlir()
        #expect(mlir.contains("func.func"))
        #expect(mlir.contains("@noReturn"))
        #expect(mlir.contains("()"))
    }

    @Test("Generated MLIR is valid structure")
    func testGeneratedMLIRStructure() {
        let mlir = testFuncs.addIntegers_mlir()
        #expect(mlir.contains("{"))
        #expect(mlir.contains("}"))
        #expect(mlir.contains("return"))
    }
}
