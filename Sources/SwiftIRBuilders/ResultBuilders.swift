/// Result builders for declarative MLIR construction
/// Provides SwiftUI-style DSL for building MLIR IR
import SwiftIRCore
import MLIRCoreWrapper
import SwiftIRTypes
import SwiftIRDialects

// MARK: - Module Builder

/// Result builder for constructing MLIR modules declaratively
@resultBuilder
public struct ModuleBuilder {
    /// Build a single module element (function, global, etc.)
    public static func buildBlock(_ components: MLIROperation...) -> [MLIROperation] {
        components
    }

    /// Build multiple module elements
    public static func buildArray(_ components: [[MLIROperation]]) -> [MLIROperation] {
        components.flatMap { $0 }
    }

    /// Support for optional elements
    public static func buildOptional(_ component: [MLIROperation]?) -> [MLIROperation] {
        component ?? []
    }

    /// Support for if-else conditions
    public static func buildEither(first component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    public static func buildEither(second component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    /// Support for limited availability
    public static func buildLimitedAvailability(_ component: [MLIROperation]) -> [MLIROperation] {
        component
    }
}

// MARK: - Function Body Builder

/// Result builder for function bodies
@resultBuilder
public struct FunctionBodyBuilder {
    /// Build a sequence of operations
    public static func buildBlock(_ components: MLIROperation...) -> [MLIROperation] {
        components
    }

    /// Build multiple operations
    public static func buildArray(_ components: [[MLIROperation]]) -> [MLIROperation] {
        components.flatMap { $0 }
    }

    /// Support for optional operations
    public static func buildOptional(_ component: [MLIROperation]?) -> [MLIROperation] {
        component ?? []
    }

    /// Support for if-else conditions
    public static func buildEither(first component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    public static func buildEither(second component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    /// Support for limited availability
    public static func buildLimitedAvailability(_ component: [MLIROperation]) -> [MLIROperation] {
        component
    }
}

// MARK: - Region Builder

/// Result builder for constructing MLIR regions
@resultBuilder
public struct RegionBuilder {
    /// Build a single block
    public static func buildBlock(_ components: MLIRBlock...) -> [MLIRBlock] {
        components
    }

    /// Build multiple blocks
    public static func buildArray(_ components: [[MLIRBlock]]) -> [MLIRBlock] {
        components.flatMap { $0 }
    }

    /// Support for optional blocks
    public static func buildOptional(_ component: [MLIRBlock]?) -> [MLIRBlock] {
        component ?? []
    }

    /// Support for if-else conditions
    public static func buildEither(first component: [MLIRBlock]) -> [MLIRBlock] {
        component
    }

    public static func buildEither(second component: [MLIRBlock]) -> [MLIRBlock] {
        component
    }

    /// Support for limited availability
    public static func buildLimitedAvailability(_ component: [MLIRBlock]) -> [MLIRBlock] {
        component
    }
}

// MARK: - Block Builder

/// Result builder for constructing MLIR blocks
@resultBuilder
public struct BlockBuilder {
    /// Build a sequence of operations
    public static func buildBlock(_ components: MLIROperation...) -> [MLIROperation] {
        components
    }

    /// Build multiple operations
    public static func buildArray(_ components: [[MLIROperation]]) -> [MLIROperation] {
        components.flatMap { $0 }
    }

    /// Support for optional operations
    public static func buildOptional(_ component: [MLIROperation]?) -> [MLIROperation] {
        component ?? []
    }

    /// Support for if-else conditions
    public static func buildEither(first component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    public static func buildEither(second component: [MLIROperation]) -> [MLIROperation] {
        component
    }

    /// Support for limited availability
    public static func buildLimitedAvailability(_ component: [MLIROperation]) -> [MLIROperation] {
        component
    }
}

// MARK: - DSL Support Extensions

extension IRBuilder {
    /// Build a function using result builder syntax
    public func function(
        name: String,
        inputs: [MlirType],
        results: [MlirType],
        location: MLIRLocation? = nil,
        @BlockBuilder body: (MLIRBlock) -> [MLIROperation]
    ) -> MLIROperation {
        let loc = location ?? unknownLocation()
        let functionType = FunctionType(inputs: inputs, results: results, context: context)

        // Create the entry block
        let entryBlock = MLIRBlock(arguments: inputs, context: context)

        // Set insertion point and build body
        setInsertionPoint(entryBlock)
        let operations = body(entryBlock)

        // Append all operations to the block
        for op in operations {
            entryBlock.append(op)
        }

        // Build the function
        let funcOp = Func.function(name: name, type: functionType, location: loc, context: context)
            .setBody(entryBlock)
            .build()

        return funcOp
    }

    /// Build a module using result builder syntax
    public func buildModule(
        @ModuleBuilder content: () -> [MLIROperation]
    ) -> MLIRModule {
        let operations = content()

        // Append all operations to the module
        for op in operations {
            module.append(op)
        }

        return module
    }
}

// MARK: - Declarative Operation Wrappers

/// Declarative wrapper for operations
public protocol DeclarativeOperation {
    func build(in builder: IRBuilder) -> MLIROperation
}

/// Declarative constant integer
public struct ConstantInt: DeclarativeOperation {
    public let value: Int64
    public let type: IntegerType
    public let location: MLIRLocation?

    public init(_ value: Int64, type: IntegerType, location: MLIRLocation? = nil) {
        self.value = value
        self.type = type
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.constant(value, type: type, location: loc, context: builder.context)
    }
}

/// Declarative constant float
public struct ConstantFloat: DeclarativeOperation {
    public let value: Double
    public let type: FloatType
    public let location: MLIRLocation?

    public init(_ value: Double, type: FloatType, location: MLIRLocation? = nil) {
        self.value = value
        self.type = type
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.constant(value, type: type, location: loc, context: builder.context)
    }
}

/// Declarative integer addition
public struct AddInt: DeclarativeOperation {
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) {
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.addi(lhs, rhs, location: loc, context: builder.context)
    }
}

/// Declarative integer subtraction
public struct SubInt: DeclarativeOperation {
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) {
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.subi(lhs, rhs, location: loc, context: builder.context)
    }
}

/// Declarative integer multiplication
public struct MulInt: DeclarativeOperation {
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) {
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.muli(lhs, rhs, location: loc, context: builder.context)
    }
}

/// Declarative float addition
public struct AddFloat: DeclarativeOperation {
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) {
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.addf(lhs, rhs, location: loc, context: builder.context)
    }
}

/// Declarative float multiplication
public struct MulFloat: DeclarativeOperation {
    public let lhs: MLIRValue
    public let rhs: MLIRValue
    public let location: MLIRLocation?

    public init(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) {
        self.lhs = lhs
        self.rhs = rhs
        self.location = location
    }

    public func build(in builder: IRBuilder) -> MLIROperation {
        let loc = location ?? builder.unknownLocation()
        return Arith.mulf(lhs, rhs, location: loc, context: builder.context)
    }
}

/// Declarative return operation
public struct Return: DeclarativeOperation {
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
        return Func.return(operands, location: loc, context: builder.context)
    }
}
