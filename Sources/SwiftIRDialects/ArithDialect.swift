/// Arith dialect support for S4MLIR
/// Provides arithmetic operations on integer, index, and floating point types
import SwiftIRCore
import MLIRCoreWrapper
import SwiftIRTypes

// MARK: - Arith Dialect

/// The Arith dialect namespace
public enum Arith {
    /// Dialect name
    public static let dialectName = "arith"

    /// Ensures the arith dialect is loaded in the context
    public static func load(in context: MLIRContext) {
        mlirRegisterArithDialectWrapper(context.handle)
    }
}

// MARK: - Binary Integer Operations

extension Arith {

    /// Creates an integer addition operation: result = lhs + rhs
    public static func addi(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.addi", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates an integer subtraction operation: result = lhs - rhs
    public static func subi(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.subi", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates an integer multiplication operation: result = lhs * rhs
    public static func muli(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.muli", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a signed integer division operation: result = lhs / rhs
    public static func divsi(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.divsi", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates an unsigned integer division operation: result = lhs / rhs
    public static func divui(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.divui", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a signed integer remainder operation: result = lhs % rhs
    public static func remsi(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.remsi", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Binary Floating Point Operations

extension Arith {

    /// Creates a floating point addition operation: result = lhs + rhs
    public static func addf(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.addf", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a floating point subtraction operation: result = lhs - rhs
    public static func subf(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.subf", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a floating point multiplication operation: result = lhs * rhs
    public static func mulf(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.mulf", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }

    /// Creates a floating point division operation: result = lhs / rhs
    public static func divf(
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let resultType = lhs.getType()
        return OperationBuilder(name: "arith.divf", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([resultType])
            .build()
    }
}

// MARK: - Constant Operations

extension Arith {

    /// Creates an integer constant operation
    public static func constant(
        _ value: Int64,
        type: IntegerType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let attr = mlirIntegerAttrGetWrapper(type.typeHandle, value)
        return OperationBuilder(name: "arith.constant", location: location, context: context)
            .addResults([type.typeHandle])
            .addAttributes([("value", attr)])
            .build()
    }

    /// Creates a floating point constant operation
    public static func constant(
        _ value: Double,
        type: FloatType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let attr = mlirFloatAttrDoubleGetWrapper(context.handle, type.typeHandle, value)
        return OperationBuilder(name: "arith.constant", location: location, context: context)
            .addResults([type.typeHandle])
            .addAttributes([("value", attr)])
            .build()
    }

    /// Creates an index constant operation
    public static func constant(
        _ value: Int64,
        indexType: IndexType,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let attr = mlirIntegerAttrGetWrapper(indexType.typeHandle, value)
        return OperationBuilder(name: "arith.constant", location: location, context: context)
            .addResults([indexType.typeHandle])
            .addAttributes([("value", attr)])
            .build()
    }
}

// MARK: - Comparison Operations

extension Arith {

    /// Integer comparison predicate
    public enum CmpIPredicate: String {
        case eq = "eq"   // Equal
        case ne = "ne"   // Not equal
        case slt = "slt" // Signed less than
        case sle = "sle" // Signed less than or equal
        case sgt = "sgt" // Signed greater than
        case sge = "sge" // Signed greater than or equal
        case ult = "ult" // Unsigned less than
        case ule = "ule" // Unsigned less than or equal
        case ugt = "ugt" // Unsigned greater than
        case uge = "uge" // Unsigned greater than or equal
    }

    /// Creates an integer comparison operation
    public static func cmpi(
        _ predicate: CmpIPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let i1Type = IntegerType.i1(context: context)

        // The predicate must be an integer attribute with i64 type
        // Map the predicate enum to its integer value
        let predicateValue: Int64 = switch predicate {
        case .eq: 0
        case .ne: 1
        case .slt: 2
        case .sle: 3
        case .sgt: 4
        case .sge: 5
        case .ult: 6
        case .ule: 7
        case .ugt: 8
        case .uge: 9
        }

        let i64Type = IntegerType.i64(context: context)
        let predicateAttr = mlirIntegerAttrGetWrapper(i64Type.typeHandle, predicateValue)

        return OperationBuilder(name: "arith.cmpi", location: location, context: context)
            .addAttributes([("predicate", predicateAttr)])
            .addOperands([lhs, rhs])
            .addResults([i1Type.typeHandle])
            .build()
    }

    /// Float comparison predicate
    public enum CmpFPredicate: String {
        case oeq = "oeq" // Ordered equal
        case one = "one" // Ordered not equal
        case olt = "olt" // Ordered less than
        case ole = "ole" // Ordered less than or equal
        case ogt = "ogt" // Ordered greater than
        case oge = "oge" // Ordered greater than or equal
    }

    /// Creates a floating point comparison operation
    public static func cmpf(
        _ predicate: CmpFPredicate,
        _ lhs: MLIRValue,
        _ rhs: MLIRValue,
        location: MLIRLocation,
        context: MLIRContext
    ) -> MLIROperation {
        let i1Type = IntegerType.i1(context: context)

        // The predicate must be an integer attribute with i64 type
        // Map the predicate enum to its integer value (based on MLIR's CmpFPredicate enum)
        let predicateValue: Int64 = switch predicate {
        case .oeq: 1  // OEQ
        case .ogt: 2  // OGT
        case .oge: 3  // OGE
        case .olt: 4  // OLT
        case .ole: 5  // OLE
        case .one: 6  // ONE
        }

        let i64Type = IntegerType.i64(context: context)
        let predicateAttr = mlirIntegerAttrGetWrapper(i64Type.typeHandle, predicateValue)

        return OperationBuilder(name: "arith.cmpf", location: location, context: context)
            .addOperands([lhs, rhs])
            .addResults([i1Type.typeHandle])
            .addAttributes([("predicate", predicateAttr)])
            .build()
    }
}
