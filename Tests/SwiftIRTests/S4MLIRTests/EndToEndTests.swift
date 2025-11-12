import Testing
@testable import SwiftIRDialects
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct EndToEndTests {

    // MARK: - IR Builder Tests

    @Test("Create IR builder")
    func createIRBuilder() {
        let builder = IRBuilder(context: MLIRContext())
        #expect(builder.module.verify())
    }

    @Test("Build simple function with return")
    func buildSimpleFunctionWithReturn() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "simple",
            inputs: [],
            results: [i32.typeHandle]
        ) { block in
            let const = builder.constantInt(42, type: i32)
            builder.return([const])
        }

        #expect(function.verify())
        #expect(function.numResults == 0) // Functions don't have results, they have return operations
    }

    @Test("Build function with arithmetic")
    func buildFunctionWithArithmetic() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "add_two_numbers",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let arg0 = block.getArgument(0)
            let arg1 = block.getArgument(1)
            let sum = builder.addi(arg0, arg1)
            builder.return([sum])
        }

        #expect(function.verify())
    }

    @Test("Build function with multiple operations")
    func buildFunctionWithMultipleOperations() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "compute",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            // Compute: (a + b) * (a - b)
            let a = block.getArgument(0)
            let b = block.getArgument(1)

            let sum = builder.addi(a, b)
            let diff = builder.subi(a, b)
            let result = builder.muli(sum, diff)

            builder.return([result])
        }

        #expect(function.verify())
    }

    @Test("Build function with constants")
    func buildFunctionWithConstants() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "add_constant",
            inputs: [i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let arg = block.getArgument(0)
            let const = builder.constantInt(10, type: i32)
            let sum = builder.addi(arg, const)
            builder.return([sum])
        }

        #expect(function.verify())
    }

    @Test("Build float function")
    func buildFloatFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)

        let function = builder.buildFunction(
            name: "float_compute",
            inputs: [f32.typeHandle, f32.typeHandle],
            results: [f32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)

            // Manually create float operations
            let loc = builder.unknownLocation()
            let sum = Arith.addf(a, b, location: loc, context: builder.context)
            _ = builder.insert(sum)
            let result = sum.getResult(0)

            builder.return([result])
        }

        #expect(function.verify())
    }

    @Test("Build void function")
    func buildVoidFunction() {
        let builder = IRBuilder(context: MLIRContext())

        let function = builder.buildFunction(
            name: "void_func",
            inputs: [],
            results: []
        ) { _ in
            builder.return()
        }

        #expect(function.verify())
    }

    @Test("Build function with file location")
    func buildFunctionWithFileLocation() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let location = builder.fileLocation("test.swift", line: 10, column: 5)

        let function = builder.buildFunction(
            name: "located",
            inputs: [],
            results: [i32.typeHandle],
            location: location
        ) { _ in
            let const = builder.constantInt(100, type: i32, location: location)
            builder.return([const])
        }

        #expect(function.verify())
    }

    // MARK: - Module Finalization Tests

    @Test("Finalize empty module")
    func finalizeEmptyModule() {
        let builder = IRBuilder(context: MLIRContext())
        let mlirText = builder.finalize()

        #expect(mlirText.contains("module"))
    }

    @Test("Finalize module with function")
    func finalizeModuleWithFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_func",
            inputs: [],
            results: [i32.typeHandle]
        ) { _ in
            let const = builder.constantInt(42, type: i32)
            builder.return([const])
        }

        // Must append the function to the module
        builder.module.append(function)

        let mlirText = builder.finalize()

        #expect(mlirText.contains("module"))
        #expect(mlirText.contains("func"))
    }

    // MARK: - Complex Examples

    @Test("Build factorial-style computation")
    func buildFactorialStyleComputation() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // Build a function that computes: n * (n-1) * (n-2)
        let function = builder.buildFunction(
            name: "triple_product",
            inputs: [i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let n = block.getArgument(0)

            let one = builder.constantInt(1, type: i32)
            let two = builder.constantInt(2, type: i32)

            let n_minus_1 = builder.subi(n, one)
            let n_minus_2 = builder.subi(n, two)

            let temp = builder.muli(n, n_minus_1)
            let result = builder.muli(temp, n_minus_2)

            builder.return([result])
        }

        #expect(function.verify())
    }

    @Test("Build multi-input function")
    func buildMultiInputFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // Build: (a + b + c) * d
        let function = builder.buildFunction(
            name: "multi_input",
            inputs: [i32.typeHandle, i32.typeHandle, i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let c = block.getArgument(2)
            let d = block.getArgument(3)

            let ab = builder.addi(a, b)
            let abc = builder.addi(ab, c)
            let result = builder.muli(abc, d)

            builder.return([result])
        }

        #expect(function.verify())
    }

    @Test("Build multiple functions in module")
    func buildMultipleFunctionsInModule() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // First function
        let func1 = builder.buildFunction(
            name: "add",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let sum = builder.addi(a, b)
            builder.return([sum])
        }

        // Second function
        let func2 = builder.buildFunction(
            name: "multiply",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let product = builder.muli(a, b)
            builder.return([product])
        }

        #expect(func1.verify())
        #expect(func2.verify())
        #expect(builder.module.verify())
    }
}
