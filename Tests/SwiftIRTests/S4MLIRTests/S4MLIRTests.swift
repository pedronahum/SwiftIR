import Testing
@testable import SwiftIR
@testable import SwiftIRCore

@Suite("S4MLIR Tests")
struct S4MLIRTests {
    @Test("Basic setup and version")
    func testBasicSetup() {
        _ = S4MLIR()
        #expect(S4MLIR.version == "0.1.0")
    }

    @Test("MLIR context creation")
    func testMLIRContext() {
        let context = MLIRContext()
        #expect(!context.isNull)
    }

    @Test("MLIR module creation")
    func testMLIRModule() {
        let context = MLIRContext()
        let module = MLIRModule(context: context)
        #expect(!module.isNull)
    }

    @Test("MLIR module has operation")
    func testMLIRModuleOperation() {
        let context = MLIRContext()
        let module = MLIRModule(context: context)
        let operation = module.operation
        // If we got here without crashing, the operation exists
        _ = operation
    }

    @Test("MLIR location creation - unknown")
    func testMLIRLocationUnknown() {
        let context = MLIRContext()
        let location = MLIRLocation.unknown(in: context)
        _ = location
    }

    @Test("MLIR location creation - file")
    func testMLIRLocationFile() {
        let context = MLIRContext()
        let location = MLIRLocation.file("test.swift", line: 10, column: 5, in: context)
        _ = location
    }

    @Test("Multiple contexts are independent")
    func testMultipleContexts() {
        let context1 = MLIRContext()
        let context2 = MLIRContext()
        #expect(!context1.isNull)
        #expect(!context2.isNull)
    }

    @Test("Multiple modules in same context")
    func testMultipleModules() {
        let context = MLIRContext()
        let module1 = MLIRModule(context: context)
        let module2 = MLIRModule(context: context)
        #expect(!module1.isNull)
        #expect(!module2.isNull)
    }

    @Test("MLIR module can be printed")
    func testModulePrint() {
        let context = MLIRContext()
        let module = MLIRModule(context: context)
        let mlirText = module.dump()
        // An empty module should print something like: module { }
        #expect(mlirText.contains("module"))
    }

    @Test("MLIR module verification")
    func testModuleVerification() {
        let context = MLIRContext()
        let module = MLIRModule(context: context)
        // An empty module should verify successfully
        #expect(module.verify())
    }

    @Test("MLIR operation verification")
    func testOperationVerification() {
        let context = MLIRContext()
        let module = MLIRModule(context: context)
        let operation = module.operation
        #expect(operation.verify())
    }
}
