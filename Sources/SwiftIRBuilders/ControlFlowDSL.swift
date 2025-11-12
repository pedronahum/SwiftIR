/// Control flow DSL support for S4MLIR
/// Provides high-level control flow constructs (if/else, loops, etc.)
import SwiftIRCore
import MLIRCoreWrapper
import SwiftIRTypes
import SwiftIRDialects

// MARK: - SCF Dialect Support

/// The SCF (Structured Control Flow) dialect namespace
public enum SCF {
    /// Dialect name
    public static let dialectName = "scf"

    /// Ensures the scf dialect is loaded in the context
    public static func load(in context: MLIRContext) {
        mlirRegisterSCFDialectWrapper(context.handle)
    }
}

// MARK: - If/Else Support

extension SCF {
    /// Creates an scf.if operation
    public static func `if`(
        _ condition: MLIRValue,
        resultTypes: [MlirType] = [],
        location: MLIRLocation,
        context: MLIRContext,
        thenRegion: MLIRRegion,
        elseRegion: MLIRRegion? = nil
    ) -> MLIROperation {
        var regions = [thenRegion]
        if let elseRegion = elseRegion {
            regions.append(elseRegion)
        }

        return OperationBuilder(name: "scf.if", location: location, context: context)
            .addOperands([condition])
            .addResults(resultTypes)
            .addRegions(regions)
            .build()
    }

    /// Creates an scf.yield operation (for returning values from control flow)
    public static func yield(
        _ operands: [MLIRValue] = [],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "scf.yield", location: location, context: context)
            .addOperands(operands)
            .build()
    }
}

// MARK: - Loop Support

extension SCF {
    /// Creates an scf.for operation (counted loop)
    public static func `for`(
        lowerBound: MLIRValue,
        upperBound: MLIRValue,
        step: MLIRValue,
        iterArgs: [MLIRValue] = [],
        location: MLIRLocation,
        context: MLIRContext,
        bodyRegion: MLIRRegion
    ) -> MLIROperation {
        let operands = [lowerBound, upperBound, step] + iterArgs
        let resultTypes = iterArgs.map { $0.getType() }

        return OperationBuilder(name: "scf.for", location: location, context: context)
            .addOperands(operands)
            .addResults(resultTypes)
            .addRegions([bodyRegion])
            .build()
    }

    /// Creates an scf.while operation (condition-based loop)
    public static func `while`(
        operands: [MLIRValue],
        resultTypes: [MlirType],
        location: MLIRLocation,
        context: MLIRContext,
        beforeRegion: MLIRRegion,
        afterRegion: MLIRRegion
    ) -> MLIROperation {
        return OperationBuilder(name: "scf.while", location: location, context: context)
            .addOperands(operands)
            .addResults(resultTypes)
            .addRegions([beforeRegion, afterRegion])
            .build()
    }

    /// Creates an scf.condition operation (for while loops)
    public static func condition(
        _ condition: MLIRValue,
        _ args: [MLIRValue],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "scf.condition", location: location, context: context)
            .addOperands([condition] + args)
            .build()
    }
}

// MARK: - High-Level DSL Wrappers

/// Declarative if operation
public struct If: DeclarativeOperation {
    public let condition: MLIRValue
    public let resultTypes: [MlirType]
    public let thenBuilder: (IRBuilder) -> [MLIROperation]
    public let elseBuilder: ((IRBuilder) -> [MLIROperation])?
    public let location: MLIRLocation?

    public init(
        _ condition: MLIRValue,
        resultTypes: [MlirType] = [],
        location: MLIRLocation? = nil,
        then thenBuilder: @escaping (IRBuilder) -> [MLIROperation],
        else elseBuilder: ((IRBuilder) -> [MLIROperation])? = nil
    ) {
        self.condition = condition
        self.resultTypes = resultTypes
        self.thenBuilder = thenBuilder
        self.elseBuilder = elseBuilder
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()

        // Build then region
        let thenRegion = MLIRRegion()
        let thenBlock = MLIRBlock(arguments: [], context: builder.context)
        builder.setInsertionPoint(thenBlock)
        let thenOps = thenBuilder(builder)
        for op in thenOps {
            thenBlock.append(op)
        }
        thenRegion.append(thenBlock)

        // Build else region if present
        var elseRegion: MLIRRegion? = nil
        if let elseBuilder = elseBuilder {
            let region = MLIRRegion()
            let block = MLIRBlock(arguments: [], context: builder.context)
            builder.setInsertionPoint(block)
            let elseOps = elseBuilder(builder)
            for op in elseOps {
                block.append(op)
            }
            region.append(block)
            elseRegion = region
        }

        return SCF.if(
            condition,
            resultTypes: resultTypes,
            location: loc,
            context: builder.context,
            thenRegion: thenRegion,
            elseRegion: elseRegion
        )
    }
}

/// Declarative for loop operation
public struct For: DeclarativeOperation {
    public let lowerBound: MLIRValue
    public let upperBound: MLIRValue
    public let step: MLIRValue
    public let iterArgs: [MLIRValue]
    public let bodyBuilder: (IRBuilder, MLIRValue) -> [MLIROperation]
    public let location: MLIRLocation?

    public init(
        lowerBound: MLIRValue,
        upperBound: MLIRValue,
        step: MLIRValue,
        iterArgs: [MLIRValue] = [],
        location: MLIRLocation? = nil,
        body bodyBuilder: @escaping (IRBuilder, MLIRValue) -> [MLIROperation]
    ) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.step = step
        self.iterArgs = iterArgs
        self.bodyBuilder = bodyBuilder
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()

        // Build body region
        let bodyRegion = MLIRRegion()
        let indexType = IndexType(context: builder.context)
        let argTypes = [indexType.typeHandle] + iterArgs.map { $0.getType() }
        let bodyBlock = MLIRBlock(arguments: argTypes, context: builder.context)
        builder.setInsertionPoint(bodyBlock)

        let inductionVar = bodyBlock.getArgument(0)
        let bodyOps = bodyBuilder(builder, inductionVar)
        for op in bodyOps {
            bodyBlock.append(op)
        }
        bodyRegion.append(bodyBlock)

        return SCF.for(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            iterArgs: iterArgs,
            location: loc,
            context: builder.context,
            bodyRegion: bodyRegion
        )
    }
}

/// Declarative yield operation
public struct Yield: DeclarativeOperation {
    public let operands: [MLIRValue]
    public let location: MLIRLocation?

    public init(_ operands: [MLIRValue] = [], location: MLIRLocation? = nil) {
        self.operands = operands
        self.location = location
    }

    public init(_ operand: MLIRValue, location: MLIRLocation? = nil) {
        self.operands = [operand]
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return SCF.yield(operands, location: loc, context: builder.context)
    }
}

// MARK: - IRBuilder Extensions for Control Flow

extension IRBuilder {
    /// Load SCF dialect
    public func loadSCF() {
        SCF.load(in: context)
    }

    /// Create an if operation with DSL
    public func buildIf(
        _ condition: MLIRValue,
        resultTypes: [MlirType] = [],
        location: MLIRLocation? = nil,
        @BlockBuilder then thenBuilder: @escaping (IRBuilder) -> [MLIROperation],
        else elseBuilder: ((IRBuilder) -> [MLIROperation])? = nil
    ) -> MLIROperation {
        let ifOp = If(
            condition,
            resultTypes: resultTypes,
            location: location,
            then: thenBuilder,
            else: elseBuilder
        )
        return ifOp.build(in: self)
    }

    /// Create a for loop with DSL
    public func buildFor(
        lowerBound: MLIRValue,
        upperBound: MLIRValue,
        step: MLIRValue,
        iterArgs: [MLIRValue] = [],
        location: MLIRLocation? = nil,
        @BlockBuilder body: @escaping (IRBuilder, MLIRValue) -> [MLIROperation]
    ) -> MLIROperation {
        let forOp = For(
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            iterArgs: iterArgs,
            location: location,
            body: body
        )
        return forOp.build(in: self)
    }

    /// Create a yield operation
    public func yield(_ operands: [MLIRValue] = [], location: MLIRLocation? = nil) {
        let loc = location ?? unknownLocation()
        let op = SCF.yield(operands, location: loc, context: context)
        insert(op)
    }
}
