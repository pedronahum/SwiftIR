/// High-level DSL constructs for S4MLIR
/// Provides convenient abstractions for common MLIR patterns
import SwiftIRCore
import MLIRCoreWrapper
import SwiftIRTypes
import SwiftIRDialects

// MARK: - Function DSL

/// Declarative function definition
public struct Function {
    public let name: String
    public let inputs: [MlirType]
    public let results: [MlirType]
    public let isPrivate: Bool
    public let location: MLIRLocation?
    public let bodyBuilder: (IRBuilder, [MLIRValue]) -> Void

    public init(
        _ name: String,
        inputs: [MlirType],
        results: [MlirType],
        isPrivate: Bool = false,
        location: MLIRLocation? = nil,
        body: @escaping (IRBuilder, [MLIRValue]) -> Void
    ) {
        self.name = name
        self.inputs = inputs
        self.results = results
        self.isPrivate = isPrivate
        self.location = location
        self.bodyBuilder = body
    }

    /// Build the function operation
    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        let functionType = FunctionType(inputs: inputs, results: results, context: builder.context)

        // Create the entry block
        let entryBlock = MLIRBlock(arguments: inputs, context: builder.context)

        // Set insertion point and build body
        builder.setInsertionPoint(entryBlock)
        let args = (0..<inputs.count).map { entryBlock.getArgument($0) }
        bodyBuilder(builder, args)

        // Build the function
        var funcBuilder = Func.function(name: name, type: functionType, location: loc, context: builder.context)
            .setBody(entryBlock)

        if isPrivate {
            funcBuilder = funcBuilder.setPrivate()
        }

        return funcBuilder.build()
    }
}

// MARK: - Module DSL

/// Declarative module builder
public struct Module {
    public let context: MLIRContext
    public let operations: [MLIROperation]

    public init(
        context: MLIRContext = MLIRContext(),
        @ModuleBuilder content: (IRBuilder) -> [MLIROperation]
    ) {
        self.context = context
        let builder = IRBuilder(context: context)
        self.operations = content(builder)
    }

    /// Get the MLIR module
    public func getModule() -> MLIRModule {
        let module = MLIRModule(context: context)
        for op in operations {
            module.append(op)
        }
        return module
    }

    /// Dump the module as a string
    public func dump() -> String {
        return getModule().dump()
    }

    /// Verify the module
    public func verify() -> Bool {
        return getModule().verify()
    }
}

// MARK: - Computation DSL

/// Declarative computation block
public struct Compute {
    public let builder: IRBuilder
    public let operations: [MLIROperation]

    public init(
        builder: IRBuilder,
        @BlockBuilder body: (IRBuilder) -> [MLIROperation]
    ) {
        self.builder = builder
        self.operations = body(builder)
    }

    /// Execute the computation and return the last result
    public func result() -> MLIRValue? {
        for op in operations {
            builder.insert(op)
        }
        return operations.last?.getResult(0)
    }

    /// Execute the computation and return all results
    public func results() -> [MLIRValue] {
        for op in operations {
            builder.insert(op)
        }
        return operations.compactMap { op in
            guard op.numResults > 0 else { return nil }
            return op.getResult(0)
        }
    }
}

// MARK: - Comparison DSL

/// Integer comparison builder
public struct CompareInt {
    public let predicate: Arith.CmpIPredicate
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(
        _ predicate: Arith.CmpIPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation? = nil
    ) {
        self.predicate = predicate
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.cmpi(predicate, lhs, rhs, location: loc, context: builder.context)
    }
}

/// Float comparison builder
public struct CompareFloat {
    public let predicate: Arith.CmpFPredicate
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(
        _ predicate: Arith.CmpFPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation? = nil
    ) {
        self.predicate = predicate
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.cmpf(predicate, lhs, rhs, location: loc, context: builder.context)
    }
}

// MARK: - Convenience Extensions

extension IRBuilder {
    /// Create a comparison operation
    public func compare(
        _ predicate: Arith.CmpIPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation? = nil
    ) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.cmpi(predicate, lhs, rhs, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Create a float comparison operation
    public func compareFloat(
        _ predicate: Arith.CmpFPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation? = nil
    ) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.cmpf(predicate, lhs, rhs, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Create a function with convenient syntax
    public func defineFunction(
        _ name: String,
        inputs: [MlirType],
        results: [MlirType],
        isPrivate: Bool = false,
        location: MLIRLocation? = nil,
        body: @escaping (IRBuilder, [MLIRValue]) -> Void
    ) -> MLIROperation {
        let function = Function(
            name,
            inputs: inputs,
            results: results,
            isPrivate: isPrivate,
            location: location,
            body: body
        )
        return function.build(in: self)
    }
}

// MARK: - Operator Overloads for Convenience

/// Protocol for values that support arithmetic operations
public protocol ArithmeticValue {
    var mlirValue: MLIRValue { get }
}

extension MLIRValue: ArithmeticValue {
    public var mlirValue: MLIRValue { self }
}

/// Value wrapper for DSL operations
public struct Value: ArithmeticValue {
    public let mlirValue: MLIRValue
    public let builder: IRBuilder

    public init(_ value: MLIRValue, builder: IRBuilder) {
        self.mlirValue = value
        self.builder = builder
    }

    /// Addition
    public static func + (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.addi(lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Subtraction
    public static func - (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.subi(lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Multiplication
    public static func * (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.muli(lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Equality comparison
    public static func == (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.compare(.eq, lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Less than comparison
    public static func < (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.compare(.slt, lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Greater than comparison
    public static func > (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.compare(.sgt, lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Less than or equal comparison
    public static func <= (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.compare(.sle, lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }

    /// Greater than or equal comparison
    public static func >= (lhs: Value, rhs: Value) -> Value {
        let result = lhs.builder.compare(.sge, lhs.mlirValue, rhs.mlirValue)
        return Value(result, builder: lhs.builder)
    }
}

// MARK: - Pattern Matching DSL

/// Switch-like pattern matching for MLIR values
public struct Match {
    public let value: MLIRValue
    public let cases: [(MLIRValue, () -> Void)]
    public let defaultCase: (() -> Void)?

    public init(
        _ value: MLIRValue,
        cases: [(MLIRValue, () -> Void)],
        default defaultCase: (() -> Void)? = nil
    ) {
        self.value = value
        self.cases = cases
        self.defaultCase = defaultCase
    }

    /// Execute the match
    public func execute(in builder: IRBuilder) {
        // This would need more sophisticated IR construction
        // For now, it's a placeholder for future implementation
        // Real implementation would use scf.switch or nested scf.if operations
    }
}

// MARK: - Loop Range DSL

/// Range-based loop helper
public struct Range {
    public let start: Int64
    public let end: Int64
    public let step: Int64

    public init(_ start: Int64, _ end: Int64, step: Int64 = 1) {
        self.start = start
        self.end = end
        self.step = step
    }

    public init(_ end: Int64) {
        self.start = 0
        self.end = end
        self.step = 1
    }

    /// Create loop bounds as MLIR values
    public func bounds(in builder: IRBuilder) -> (MLIRValue, MLIRValue, MLIRValue) {
        let startVal = builder.constantInt(start, type: IntegerType.i64(context: builder.context))
        let endVal = builder.constantInt(end, type: IntegerType.i64(context: builder.context))
        let stepVal = builder.constantInt(step, type: IntegerType.i64(context: builder.context))
        return (startVal, endVal, stepVal)
    }
}
