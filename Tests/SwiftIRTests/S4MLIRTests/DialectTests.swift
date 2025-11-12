import Testing
@testable import SwiftIRDialects
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct DialectTests {

    // MARK: - Arith Dialect Setup

    @Test("Load arith dialect")
    func loadArithDialect() {
        let context = MLIRContext()

        // Register and load the dialect explicitly
        Arith.load(in: context)

        // Verify it was loaded by creating an arith operation
        let i32 = IntegerType.i32(context: context)
        let loc = MLIRLocation.unknown(in: context)
        let op = Arith.constant(42, type: i32, location: loc, context: context)

        // If the dialect is loaded, the operation should verify
        #expect(op.verify())
    }

    // MARK: - Constant Operations in Function Context

    @Test("Create integer constant in function")
    func createIntegerConstantInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_const",
            inputs: [],
            results: [i32.typeHandle]
        ) { block in
            let const = builder.constantInt(42, type: i32)
            builder.return([const])
        }

        builder.module.append(function)

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create float constant in function")
    func createFloatConstantInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_float_const",
            inputs: [],
            results: [f32.typeHandle]
        ) { block in
            let const = builder.constantFloat(3.14, type: f32)
            builder.return([const])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create index constant in function")
    func createIndexConstantInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let indexType = IndexType(context: builder.context)
        let i64 = IntegerType.i64(context: builder.context)

        let function = builder.buildFunction(
            name: "test_index_const",
            inputs: [],
            results: [i64.typeHandle]
        ) { block in
            // Create index constant and use it
            let location = builder.unknownLocation()
            let constOp = Arith.constant(10, indexType: indexType, location: location, context: builder.context)
            _ = builder.insert(constOp)
            let indexVal = constOp.getResult(0)

            // Convert to i64 for return (simplified - in real code would use proper cast)
            let i64Const = builder.constantInt(10, type: i64)
            builder.return([i64Const])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    // MARK: - Integer Arithmetic Operations

    @Test("Create integer addition in function")
    func createIntegerAdditionInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_add",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let sum = builder.addi(a, b)
            builder.return([sum])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create integer subtraction in function")
    func createIntegerSubtractionInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_sub",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let diff = builder.subi(a, b)
            builder.return([diff])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create integer multiplication in function")
    func createIntegerMultiplicationInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_mul",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let product = builder.muli(a, b)
            builder.return([product])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create signed integer division in function")
    func createSignedIntegerDivisionInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_div",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let location = builder.unknownLocation()
            let divOp = Arith.divsi(a, b, location: location, context: builder.context)
            _ = builder.insert(divOp)
            let result = divOp.getResult(0)
            builder.return([result])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    // MARK: - Float Arithmetic Operations

    @Test("Create float addition in function")
    func createFloatAdditionInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_addf",
            inputs: [f32.typeHandle, f32.typeHandle],
            results: [f32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let location = builder.unknownLocation()
            let addOp = Arith.addf(a, b, location: location, context: builder.context)
            _ = builder.insert(addOp)
            let result = addOp.getResult(0)
            builder.return([result])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create float multiplication in function")
    func createFloatMultiplicationInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)

        let function = builder.buildFunction(
            name: "test_mulf",
            inputs: [f32.typeHandle, f32.typeHandle],
            results: [f32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let location = builder.unknownLocation()
            let mulOp = Arith.mulf(a, b, location: location, context: builder.context)
            _ = builder.insert(mulOp)
            let result = mulOp.getResult(0)
            builder.return([result])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    // MARK: - Comparison Operations

    @Test("Create integer comparison - equal in function")
    func createIntegerComparisonEqualInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let i1 = IntegerType.i1(context: builder.context)

        let function = builder.buildFunction(
            name: "test_cmpi_eq",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i1.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let cmp = builder.compare(.eq, a, b)
            builder.return([cmp])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create integer comparison - less than in function")
    func createIntegerComparisonLessThanInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let i1 = IntegerType.i1(context: builder.context)

        let function = builder.buildFunction(
            name: "test_cmpi_slt",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i1.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let cmp = builder.compare(.slt, a, b)
            builder.return([cmp])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    @Test("Create float comparison in function")
    func createFloatComparisonInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)
        let i1 = IntegerType.i1(context: builder.context)

        let function = builder.buildFunction(
            name: "test_cmpf",
            inputs: [f32.typeHandle, f32.typeHandle],
            results: [i1.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let cmp = builder.compareFloat(.olt, a, b)
            builder.return([cmp])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    // MARK: - Complex Expression Tests

    @Test("Build complex arithmetic expression in function")
    func buildComplexArithmeticExpressionInFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // Build: (a + b) * (c - d)
        let function = builder.buildFunction(
            name: "complex_expr",
            inputs: [i32.typeHandle, i32.typeHandle, i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let c = block.getArgument(2)
            let d = block.getArgument(3)

            let sum = builder.addi(a, b)
            let diff = builder.subi(c, d)
            let product = builder.muli(sum, diff)

            builder.return([product])
        }

        #expect(function.verify())
        #expect(builder.module.verify())
    }

    // MARK: - Module Integration

    @Test("Create module with multiple functions")
    func createModuleWithMultipleFunctions() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // Create first function
        let func1 = builder.buildFunction(
            name: "add_func",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let sum = builder.addi(a, b)
            builder.return([sum])
        }

        // Create second function
        let func2 = builder.buildFunction(
            name: "mul_func",
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
