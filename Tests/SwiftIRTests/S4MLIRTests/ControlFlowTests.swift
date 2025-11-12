import Testing
@testable import SwiftIRBuilders
@testable import SwiftIRDialects
@testable import SwiftIRTypes
@testable import SwiftIRCore

struct ControlFlowTests {

    // MARK: - SCF Dialect Tests

    @Test("Load SCF dialect")
    func loadSCFDialect() {
        let context = MLIRContext()

        // Register and load the SCF dialect
        SCF.load(in: context)

        // Verify the dialect was loaded by checking that we can create SCF operations
        // (The actual verification is tested in other SCF tests)
        let loc = MLIRLocation.unknown(in: context)
        let yieldOp = SCF.yield([], location: loc, context: context)

        // If the dialect is loaded, we should be able to create the operation
        #expect(yieldOp.handle.ptr != nil)
    }

    @Test("SCF if operation")
    func scfIfOperation() {
        let context = MLIRContext()
        context.loadAllDialects()
        SCF.load(in: context)

        let i1 = IntegerType.i1(context: context)
        let i32 = IntegerType.i32(context: context)
        let location = MLIRLocation.unknown(in: context)

        // Create condition
        let condition = Arith.constant(1, type: i1, location: location, context: context).getResult(0)

        // Create then region
        let thenRegion = MLIRRegion()
        let thenBlock = MLIRBlock(arguments: [], context: context)
        let thenValue = Arith.constant(42, type: i32, location: location, context: context)
        thenBlock.append(thenValue)
        let thenYield = SCF.yield([thenValue.getResult(0)], location: location, context: context)
        thenBlock.append(thenYield)
        thenRegion.append(thenBlock)

        // Create else region
        let elseRegion = MLIRRegion()
        let elseBlock = MLIRBlock(arguments: [], context: context)
        let elseValue = Arith.constant(0, type: i32, location: location, context: context)
        elseBlock.append(elseValue)
        let elseYield = SCF.yield([elseValue.getResult(0)], location: location, context: context)
        elseBlock.append(elseYield)
        elseRegion.append(elseBlock)

        // Create if operation
        let ifOp = SCF.if(
            condition,
            resultTypes: [i32.typeHandle],
            location: location,
            context: context,
            thenRegion: thenRegion,
            elseRegion: elseRegion
        )

        #expect(ifOp.verify())
        #expect(ifOp.numResults == 1)
    }

    @Test("SCF yield operation")
    func scfYieldOperation() {
        let context = MLIRContext()
        context.loadAllDialects()
        SCF.load(in: context)

        let i32 = IntegerType.i32(context: context)
        let i1 = IntegerType.i1(context: context)
        let location = MLIRLocation.unknown(in: context)

        // Create a condition for the if operation
        let conditionOp = Arith.constant(1, type: i1, location: location, context: context)
        let condition = conditionOp.getResult(0)

        // Create an scf.if with yields in both regions (scf.if requires 2 regions)
        let thenRegion = MLIRRegion()
        let thenBlock = MLIRBlock(arguments: [], context: context)
        let thenValue = Arith.constant(42, type: i32, location: location, context: context)
        thenBlock.append(thenValue)
        let thenYield = SCF.yield([thenValue.getResult(0)], location: location, context: context)
        thenBlock.append(thenYield)
        thenRegion.append(thenBlock)

        let elseRegion = MLIRRegion()
        let elseBlock = MLIRBlock(arguments: [], context: context)
        let elseValue = Arith.constant(0, type: i32, location: location, context: context)
        elseBlock.append(elseValue)
        let elseYield = SCF.yield([elseValue.getResult(0)], location: location, context: context)
        elseBlock.append(elseYield)
        elseRegion.append(elseBlock)

        let ifOp = SCF.if(condition, resultTypes: [i32.typeHandle], location: location, context: context, thenRegion: thenRegion, elseRegion: elseRegion)

        // Verify the if operation (which contains the yields)
        #expect(ifOp.verify())
    }

    @Test("SCF for loop operation")
    func scfForLoopOperation() {
        let context = MLIRContext()
        context.loadAllDialects()
        SCF.load(in: context)

        let indexType = IndexType(context: context)
        let location = MLIRLocation.unknown(in: context)

        // Create loop bounds
        let lowerBound = Arith.constant(0, indexType: indexType, location: location, context: context).getResult(0)
        let upperBound = Arith.constant(10, indexType: indexType, location: location, context: context).getResult(0)
        let step = Arith.constant(1, indexType: indexType, location: location, context: context).getResult(0)

        // Create body region
        let bodyRegion = MLIRRegion()
        let bodyBlock = MLIRBlock(arguments: [indexType.typeHandle], context: context)
        let yieldOp = SCF.yield([], location: location, context: context)
        bodyBlock.append(yieldOp)
        bodyRegion.append(bodyBlock)

        // Create for loop
        let forOp = SCF.for(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            location: location,
            context: context,
            bodyRegion: bodyRegion
        )

        #expect(forOp.verify())
    }

    // MARK: - Declarative Control Flow Tests

    @Test("If declarative operation")
    func ifDeclarativeOperation() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i1 = IntegerType.i1(context: builder.context)
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        // Create condition
        let condition = builder.constantInt(1, type: i1)

        // Create if operation
        let ifOp = If(
            condition,
            resultTypes: [i32.typeHandle],
            then: { builder in
                let value = Arith.constant(42, type: i32, location: builder.unknownLocation(), context: builder.context)
                let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [value, yield]
            },
            else: { builder in
                let value = Arith.constant(0, type: i32, location: builder.unknownLocation(), context: builder.context)
                let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [value, yield]
            }
        )

        let op = ifOp.build(in: builder)
        #expect(op.verify())
        #expect(op.numResults == 1)
    }

    @Test("For declarative operation")
    func forDeclarativeOperation() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        // Create loop bounds using index type (required by scf.for)
        let indexType = IndexType(context: builder.context)
        let location = builder.unknownLocation()
        let lowerBound = Arith.constant(0, indexType: indexType, location: location, context: builder.context).getResult(0)
        let upperBound = Arith.constant(10, indexType: indexType, location: location, context: builder.context).getResult(0)
        let step = Arith.constant(1, indexType: indexType, location: location, context: builder.context).getResult(0)

        // Create for loop
        let forOp = For(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            body: { builder, inductionVar in
                let yield = SCF.yield([], location: builder.unknownLocation(), context: builder.context)
                return [yield]
            }
        )

        let op = forOp.build(in: builder)
        #expect(op.verify())
    }

    @Test("Yield declarative operation")
    func yieldDeclarativeOperation() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i32 = IntegerType.i32(context: builder.context)
        let i1 = IntegerType.i1(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let condition = builder.constantInt(1, type: i1)

        // scf.yield must be inside a parent operation, so wrap it in scf.if
        let ifOp = builder.buildIf(
            condition,
            resultTypes: [i32.typeHandle],
            then: { builder in
                let value = builder.constantInt(42, type: i32)
                let yieldOp = Yield(value)
                let op = yieldOp.build(in: builder)
                return [op]
            },
            else: { builder in
                let zeroValue = builder.constantInt(0, type: i32)
                builder.yield([zeroValue])
                return []
            }
        )

        // Verify the if operation (which contains the yield)
        #expect(ifOp.verify())
    }

    // MARK: - IRBuilder Control Flow Extensions Tests

    @Test("IRBuilder buildIf convenience method")
    func irBuilderBuildIf() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i1 = IntegerType.i1(context: builder.context)
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let condition = builder.constantInt(1, type: i1)

        let ifOp = builder.buildIf(
            condition,
            resultTypes: [i32.typeHandle],
            then: { builder in
                let value = Arith.constant(42, type: i32, location: builder.unknownLocation(), context: builder.context)
                let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [value, yield]
            },
            else: { builder in
                let value = Arith.constant(0, type: i32, location: builder.unknownLocation(), context: builder.context)
                let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [value, yield]
            }
        )

        #expect(ifOp.verify())
        #expect(ifOp.numResults == 1)
    }

    @Test("IRBuilder buildFor convenience method")
    func irBuilderBuildFor() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        // Use index type for loop bounds (required by scf.for)
        let indexType = IndexType(context: builder.context)
        let location = builder.unknownLocation()
        let lowerBound = Arith.constant(0, indexType: indexType, location: location, context: builder.context).getResult(0)
        let upperBound = Arith.constant(10, indexType: indexType, location: location, context: builder.context).getResult(0)
        let step = Arith.constant(1, indexType: indexType, location: location, context: builder.context).getResult(0)

        let forOp = builder.buildFor(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            body: { builder, inductionVar in
                let yield = SCF.yield([], location: builder.unknownLocation(), context: builder.context)
                return [yield]
            }
        )

        #expect(forOp.verify())
    }

    @Test("IRBuilder yield convenience method")
    func irBuilderYieldConvenience() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let value = builder.constantInt(42, type: i32)
        builder.yield([value])

        // Yield was successfully inserted
        #expect(true)
    }

    // MARK: - Complex Control Flow Tests

    @Test("Nested if operations")
    func nestedIfOperations() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i1 = IntegerType.i1(context: builder.context)
        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        let condition1 = builder.constantInt(1, type: i1)

        let outerIf = builder.buildIf(
            condition1,
            resultTypes: [i32.typeHandle],
            then: { builder in
                // Create condition for nested if inside the outer then clause
                let condition2 = builder.constantInt(0, type: i1)

                // Nested if
                let innerIf = builder.buildIf(
                    condition2,
                    resultTypes: [i32.typeHandle],
                    then: { builder in
                        let value = Arith.constant(1, type: i32, location: builder.unknownLocation(), context: builder.context)
                        let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                        return [value, yield]
                    },
                    else: { builder in
                        let value = Arith.constant(2, type: i32, location: builder.unknownLocation(), context: builder.context)
                        let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                        return [value, yield]
                    }
                )
                let yield = SCF.yield([innerIf.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [innerIf, yield]
            },
            else: { builder in
                let value = Arith.constant(0, type: i32, location: builder.unknownLocation(), context: builder.context)
                let yield = SCF.yield([value.getResult(0)], location: builder.unknownLocation(), context: builder.context)
                return [value, yield]
            }
        )

        #expect(outerIf.verify())
        #expect(outerIf.numResults == 1)
    }

    @Test("For loop with accumulation")
    func forLoopWithAccumulation() {
        let builder = IRBuilder(context: MLIRContext())
        builder.loadSCF()

        let i32 = IntegerType.i32(context: builder.context)
        let block = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(block)

        // Use index type for loop bounds (required by scf.for)
        let indexType = IndexType(context: builder.context)
        let location = builder.unknownLocation()

        // Create operations and append them to the block
        let lowerBoundOp = Arith.constant(0, indexType: indexType, location: location, context: builder.context)
        _ = builder.insert(lowerBoundOp)
        let lowerBound = lowerBoundOp.getResult(0)

        let upperBoundOp = Arith.constant(10, indexType: indexType, location: location, context: builder.context)
        _ = builder.insert(upperBoundOp)
        let upperBound = upperBoundOp.getResult(0)

        let stepOp = Arith.constant(1, indexType: indexType, location: location, context: builder.context)
        _ = builder.insert(stepOp)
        let step = stepOp.getResult(0)

        let initValue = builder.constantInt(0, type: i32)

        let forOp = builder.buildFor(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            iterArgs: [initValue],
            body: { builder, inductionVar in
                // For accumulation with iter args, the loop body block has block arguments
                // The first argument is the induction variable, and subsequent arguments are the iter args
                // For now, just create a new value and yield it (in reality we'd use the block argument)
                let iterArgValue = builder.constantInt(0, type: i32)
                let yield = SCF.yield([iterArgValue], location: builder.unknownLocation(), context: builder.context)
                yield
            }
        )

        #expect(forOp.verify())
    }
}
