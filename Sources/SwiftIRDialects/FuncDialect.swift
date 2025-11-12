/// Func dialect support for S4MLIR
/// Provides function definition and call operations
import SwiftIRCore
import MLIRCoreWrapper
import SwiftIRTypes

// MARK: - Func Dialect

/// The Func dialect namespace
public enum Func {
    /// Dialect name
    public static let dialectName = "func"

    /// Ensures the func dialect is loaded in the context
    public static func load(in context: MLIRContext) {
        mlirRegisterFuncDialectWrapper(context.handle)
    }
}

// MARK: - Function Type

/// Represents a function type in MLIR
public struct FunctionType {
    public let inputs: [MlirType]
    public let results: [MlirType]
    public let context: MLIRContext

    public init(inputs: [MlirType], results: [MlirType], context: MLIRContext) {
        self.inputs = inputs
        self.results = results
        self.context = context
    }

    /// Creates a function type from Swift types
    public init<T: MLIRType, R: MLIRType>(
        inputs: [T],
        results: [R],
        context: MLIRContext
    ) {
        self.inputs = inputs.map { $0.typeHandle }
        self.results = results.map { $0.typeHandle }
        self.context = context
    }
}

// MARK: - Function Operations

extension Func {

    /// Function builder for creating func.func operations
    public final class FunctionBuilder {
        private let name: String
        private let functionType: FunctionType
        private let location: MLIRLocation
        private let context: MLIRContext
        private var isPrivate: Bool = false
        private var bodyBlock: MLIRBlock?

        public init(
            name: String,
            type: FunctionType,
            location: MLIRLocation,
            context: MLIRContext
        ) {
            self.name = name
            self.functionType = type
            self.location = location
            self.context = context
        }

        /// Sets the function visibility
        public func setPrivate(_ value: Bool = true) -> FunctionBuilder {
            self.isPrivate = value
            return self
        }

        /// Sets the function body
        public func setBody(_ block: MLIRBlock) -> FunctionBuilder {
            self.bodyBlock = block
            return self
        }

        /// Builds the function operation
        public func build() -> MLIROperation {
            // Create the function type attribute
            let typeAttr = createFunctionTypeAttribute()

            // Create the function name attribute
            let nameAttr = name.withCString { ptr in
                let strRef = mlirStringRefCreateWrapper(ptr, name.utf8.count)
                return mlirStringAttrGetWrapper(context.handle, strRef)
            }

            var attributes: [(String, MlirAttribute)] = [
                ("sym_name", nameAttr),
                ("function_type", typeAttr)
            ]

            // Add visibility if private
            if isPrivate {
                let privateAttr = "private".withCString { ptr in
                    let strRef = mlirStringRefCreateWrapper(ptr, "private".utf8.count)
                    return mlirStringAttrGetWrapper(context.handle, strRef)
                }
                attributes.append(("sym_visibility", privateAttr))
            }

            // Create the region for the function body
            let region = MLIRRegion()
            if let body = bodyBlock {
                region.append(body)
            }

            // Build the operation
            return OperationBuilder(name: "func.func", location: location, context: context)
                .addAttributes(attributes)
                .addRegions([region])
                .build()
        }

        private func createFunctionTypeAttribute() -> MlirAttribute {
            // Create a proper MLIR function type
            var inputs = functionType.inputs
            var results = functionType.results
            let inputCount = inputs.count
            let resultCount = results.count

            let funcType = inputs.withUnsafeMutableBufferPointer { inputsPtr in
                results.withUnsafeMutableBufferPointer { resultsPtr in
                    mlirFunctionTypeGetWrapper(
                        context.handle,
                        inputCount,
                        inputsPtr.baseAddress,
                        resultCount,
                        resultsPtr.baseAddress
                    )
                }
            }

            // Create a type attribute from the function type
            return mlirTypeAttrGetWrapper(funcType)
        }
    }

    /// Creates a function operation
    public static func function(
        name: String,
        type: FunctionType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> FunctionBuilder {
        return FunctionBuilder(name: name, type: type, location: location, context: context)
    }

    /// Creates a function call operation
    public static func call(
        _ callee: String,
        _ operands: [MLIRValue],
        resultTypes: [MlirType],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let calleeAttr = callee.withCString { ptr in
            let strRef = mlirStringRefCreateWrapper(ptr, callee.utf8.count)
            return mlirStringAttrGetWrapper(context.handle, strRef)
        }

        return OperationBuilder(name: "func.call", location: location, context: context)
            .addOperands(operands)
            .addResults(resultTypes)
            .addAttributes([("callee", calleeAttr)])
            .build()
    }

    /// Creates a return operation
    public static func `return`(
        _ operands: [MLIRValue] = [],
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        return OperationBuilder(name: "func.return", location: location, context: context)
            .addOperands(operands)
            .build()
    }
}

// MARK: - IR Builder Utility

/// Helper for building well-formed MLIR IR
public final class IRBuilder {
    public let context: MLIRContext
    public let module: MLIRModule
    public private(set) var currentBlock: MLIRBlock?

    public init(context: MLIRContext) {
        self.context = context
        self.module = MLIRModule(context: context)

        // Load commonly used dialects
        context.loadAllDialects()
        Arith.load(in: context)
        Func.load(in: context)
        // Register tensor and linalg dialects needed for ML operations
        context.registerTensorDialect()
        context.registerLinalgDialect()
    }

    /// Creates a location for generated operations
    public func unknownLocation() -> MLIRLocation {
        MLIRLocation.unknown(in: context)
    }

    /// Creates a file location
    public func fileLocation(_ filename: String, line: UInt, column: UInt) -> MLIRLocation {
        MLIRLocation.file(filename, line: line, column: column, in: context)
    }

    /// Sets the current insertion block
    public func setInsertionPoint(_ block: MLIRBlock) {
        self.currentBlock = block
    }

    /// Inserts an operation at the current insertion point
    @discardableResult
    public func insert(_ operation: MLIROperation) -> MLIROperation {
        if let block = currentBlock {
            block.append(operation)
        }
        return operation
    }

    /// Creates and inserts a constant integer
    public func constantInt(_ value: Int64, type: IntegerType, location: MLIRLocation? = nil) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.constant(value, type: type, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Creates and inserts a constant float
    public func constantFloat(_ value: Double, type: FloatType, location: MLIRLocation? = nil) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.constant(value, type: type, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Creates and inserts an integer addition
    public func addi(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.addi(lhs, rhs, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Creates and inserts an integer subtraction
    public func subi(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.subi(lhs, rhs, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Creates and inserts an integer multiplication
    public func muli(_ lhs: MLIRValue, _ rhs: MLIRValue, location: MLIRLocation? = nil) -> MLIRValue {
        let loc = location ?? unknownLocation()
        let op = Arith.muli(lhs, rhs, location: loc, context: context)
        insert(op)
        return op.getResult(0)
    }

    /// Creates and inserts a return operation
    public func `return`(_ operands: [MLIRValue] = [], location: MLIRLocation? = nil) {
        let loc = location ?? unknownLocation()
        let op = Func.return(operands, location: loc, context: context)
        insert(op)
    }

    /// Builds a function with a body builder closure
    public func buildFunction(
        name: String,
        inputs: [MlirType],
        results: [MlirType],
        location: MLIRLocation? = nil,
        @FunctionBodyBuilder body: (MLIRBlock) -> Void
    ) -> MLIROperation {
        let loc = location ?? unknownLocation()
        let functionType = FunctionType(inputs: inputs, results: results, context: context)

        // Create the entry block
        let entryBlock = MLIRBlock(arguments: inputs, context: context)

        // Set insertion point and build body
        setInsertionPoint(entryBlock)
        body(entryBlock)

        // Build the function
        return Func.function(name: name, type: functionType, location: loc, context: context)
            .setBody(entryBlock)
            .build()
    }

    /// Returns the built module as a string
    public func finalize() -> String {
        return module.dump()
    }
}

/// Result builder for function bodies
@resultBuilder
public struct FunctionBodyBuilder {
    public static func buildBlock(_ components: Void...) -> Void {
        ()
    }
}
