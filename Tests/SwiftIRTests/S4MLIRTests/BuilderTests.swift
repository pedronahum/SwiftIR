import Testing
@testable import SwiftIRBuilders
@testable import SwiftIRDialects
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct BuilderTests {

    // MARK: - Result Builder Tests

    @Test("ModuleBuilder - single function")
    func moduleBuilderSingleFunction() {
        let context = MLIRContext()
        let builder = IRBuilder(context: context)
        let i32 = IntegerType.i32(context: context)

        let operations = ModuleBuilder.buildBlock(
            builder.buildFunction(
                name: "test",
                inputs: [],
                results: [i32.typeHandle]
            ) { _ in
                let const = builder.constantInt(42, type: i32)
                builder.return([const])
            }
        )

        #expect(operations.count == 1)
        #expect(operations[0].verify())
    }

    @Test("FunctionBodyBuilder - multiple operations")
    func functionBodyBuilderMultipleOps() {
        let context = MLIRContext()
        let builder = IRBuilder(context: context)
        let i32 = IntegerType.i32(context: context)
        let loc = builder.unknownLocation()

        let a = Arith.constant(5, type: i32, location: loc, context: context).getResult(0)
        let b = Arith.constant(3, type: i32, location: loc, context: context).getResult(0)

        let operations = FunctionBodyBuilder.buildBlock(
            Arith.addi(a, b, location: loc, context: context),
            Arith.muli(a, b, location: loc, context: context)
        )

        #expect(operations.count == 2)
        #expect(operations[0].verify())
        #expect(operations[1].verify())
    }

    @Test("RegionBuilder - single block")
    func regionBuilderSingleBlock() {
        let context = MLIRContext()
        let i32 = IntegerType.i32(context: context)

        let block = MLIRBlock(arguments: [i32.typeHandle], context: context)
        let blocks = RegionBuilder.buildBlock(block)

        #expect(blocks.count == 1)
    }

    @Test("BlockBuilder - operation sequence")
    func blockBuilderOperationSequence() {
        let context = MLIRContext()
        let builder = IRBuilder(context: context)
        let i32 = IntegerType.i32(context: context)
        let loc = builder.unknownLocation()

        let const1 = Arith.constant(10, type: i32, location: loc, context: context)
        let const2 = Arith.constant(20, type: i32, location: loc, context: context)

        let operations = BlockBuilder.buildBlock(const1, const2)

        #expect(operations.count == 2)
        #expect(operations[0].verify())
        #expect(operations[1].verify())
    }

    // MARK: - Declarative Operation Tests

    @Test("ConstantInt declarative operation")
    func constantIntDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let constOp = ConstantInt(42, type: i32)
        let op = constOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("ConstantFloat declarative operation")
    func constantFloatDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)

        let constOp = ConstantFloat(3.14, type: f32)
        let op = constOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("AddInt declarative operation")
    func addIntDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let loc = builder.unknownLocation()

        let lhs = Arith.constant(5, type: i32, location: loc, context: builder.context).getResult(0)
        let rhs = Arith.constant(3, type: i32, location: loc, context: builder.context).getResult(0)

        let addOp = AddInt(lhs, rhs)
        let op = addOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("SubInt declarative operation")
    func subIntDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let loc = builder.unknownLocation()

        let lhs = Arith.constant(10, type: i32, location: loc, context: builder.context).getResult(0)
        let rhs = Arith.constant(4, type: i32, location: loc, context: builder.context).getResult(0)

        let subOp = SubInt(lhs, rhs)
        let op = subOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("MulInt declarative operation")
    func mulIntDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let loc = builder.unknownLocation()

        let lhs = Arith.constant(6, type: i32, location: loc, context: builder.context).getResult(0)
        let rhs = Arith.constant(7, type: i32, location: loc, context: builder.context).getResult(0)

        let mulOp = MulInt(lhs, rhs)
        let op = mulOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("Return declarative operation")
    func returnDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        // func.return must be inside a func.func, so create a function
        let function = builder.buildFunction(
            name: "test_return",
            inputs: [],
            results: [i32.typeHandle]
        ) { block in
            let value = builder.constantInt(42, type: i32)
            builder.return([value])
        }

        // Verify the function (which contains the return)
        #expect(function.verify())
    }

    // MARK: - IRBuilder DSL Extension Tests

    @Test("IRBuilder function with result builder")
    func irBuilderFunctionWithResultBuilder() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.function(
            name: "test_func",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { block in
            let a = block.getArgument(0)
            let b = block.getArgument(1)
            let sum = Arith.addi(a, b, location: builder.unknownLocation(), context: builder.context)
            let ret = Func.return([sum.getResult(0)], location: builder.unknownLocation(), context: builder.context)
            return [sum, ret]
        }

        #expect(function.verify())
    }

    @Test("IRBuilder compare operation")
    func irBuilderCompare() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let a = builder.constantInt(5, type: i32)
        let b = builder.constantInt(10, type: i32)
        let cmp = builder.compare(.slt, a, b)

        #expect(!cmp.mlirValue.isNull)
    }

    @Test("IRBuilder compareFloat operation")
    func irBuilderCompareFloat() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let a = builder.constantFloat(2.5, type: f32)
        let b = builder.constantFloat(3.5, type: f32)
        let cmp = builder.compareFloat(.olt, a, b)

        #expect(!cmp.mlirValue.isNull)
    }

    // MARK: - High-Level DSL Tests

    @Test("Function declarative wrapper")
    func functionDeclarativeWrapper() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = Function(
            "add",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { builder, args in
            let sum = builder.addi(args[0], args[1])
            builder.return([sum])
        }

        let op = function.build(in: builder)
        #expect(op.verify())
    }

    @Test("Function with private visibility")
    func functionWithPrivateVisibility() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = Function(
            "private_func",
            inputs: [],
            results: [i32.typeHandle],
            isPrivate: true
        ) { builder, _ in
            let const = builder.constantInt(100, type: i32)
            builder.return([const])
        }

        let op = function.build(in: builder)
        #expect(op.verify())
    }

    @Test("Module declarative wrapper")
    func moduleDeclarativeWrapper() {
        let i32 = IntegerType.i32(context: MLIRContext())

        let module = Module { builder in
            let func1 = Function("add", inputs: [i32.typeHandle, i32.typeHandle], results: [i32.typeHandle]) { builder, args in
                let sum = builder.addi(args[0], args[1])
                builder.return([sum])
            }
            func1.build(in: builder)
        }

        #expect(module.verify())
    }

    @Test("CompareInt declarative")
    func compareIntDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let loc = builder.unknownLocation()

        let lhs = Arith.constant(5, type: i32, location: loc, context: builder.context).getResult(0)
        let rhs = Arith.constant(5, type: i32, location: loc, context: builder.context).getResult(0)

        let cmpOp = CompareInt(.eq, lhs, rhs)
        let op = cmpOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("CompareFloat declarative")
    func compareFloatDeclarative() {
        let builder = IRBuilder(context: MLIRContext())
        let f32 = FloatType.f32(context: builder.context)
        let loc = builder.unknownLocation()

        let lhs = Arith.constant(2.5, type: f32, location: loc, context: builder.context).getResult(0)
        let rhs = Arith.constant(3.5, type: f32, location: loc, context: builder.context).getResult(0)

        let cmpOp = CompareFloat(.olt, lhs, rhs)
        let op = cmpOp.build(in: builder)

        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("Value wrapper arithmetic operations")
    func valueWrapperArithmetic() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let a = Value(builder.constantInt(5, type: i32), builder: builder)
        let b = Value(builder.constantInt(3, type: i32), builder: builder)

        let sum = a + b
        let diff = a - b
        let prod = a * b

        #expect(!sum.mlirValue.isNull)
        #expect(!diff.mlirValue.isNull)
        #expect(!prod.mlirValue.isNull)
    }

    @Test("Value wrapper comparison operations")
    func valueWrapperComparison() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let a = Value(builder.constantInt(5, type: i32), builder: builder)
        let b = Value(builder.constantInt(10, type: i32), builder: builder)

        let eq = a == b
        let lt = a < b
        let gt = a > b
        let le = a <= b
        let ge = a >= b

        #expect(!eq.mlirValue.isNull)
        #expect(!lt.mlirValue.isNull)
        #expect(!gt.mlirValue.isNull)
        #expect(!le.mlirValue.isNull)
        #expect(!ge.mlirValue.isNull)
    }

    @Test("Range helper")
    func rangeHelper() {
        let builder = IRBuilder(context: MLIRContext())
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let range = Range(0, 10, step: 2)
        let (start, end, step) = range.bounds(in: builder)

        #expect(!start.mlirValue.isNull)
        #expect(!end.mlirValue.isNull)
        #expect(!step.mlirValue.isNull)
    }

    @Test("Range helper - simple form")
    func rangeHelperSimple() {
        let builder = IRBuilder(context: MLIRContext())
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let range = Range(10)
        let (start, end, step) = range.bounds(in: builder)

        #expect(!start.mlirValue.isNull)
        #expect(!end.mlirValue.isNull)
        #expect(!step.mlirValue.isNull)
    }

    @Test("IRBuilder defineFunction convenience")
    func irBuilderDefineFunction() {
        let builder = IRBuilder(context: MLIRContext())
        let i32 = IntegerType.i32(context: builder.context)

        let function = builder.defineFunction(
            "compute",
            inputs: [i32.typeHandle, i32.typeHandle],
            results: [i32.typeHandle]
        ) { builder, args in
            let sum = builder.addi(args[0], args[1])
            builder.return([sum])
        }

        #expect(function.verify())
    }
}
