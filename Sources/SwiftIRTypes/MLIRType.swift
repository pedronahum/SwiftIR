/// Core protocols and types for type-safe MLIR construction in Swift
import SwiftIRCore
import MLIRCoreWrapper

/// Protocol that all MLIR types must conform to
///
/// This protocol provides type safety and ensures that only valid MLIR types
/// can be used in operations and transformations.
public protocol MLIRType {
    /// The underlying MLIR type handle
    var typeHandle: MlirType { get }
    
    /// The context in which this type was created
    var context: MLIRContext { get }
    
    /// Creates a type from an existing MLIR type handle
    init(handle: MlirType, context: MLIRContext)
    
    /// Returns true if this type is valid (not null)
    var isValid: Bool { get }
}

/// Protocol for types that have a known bit width
public protocol BitWidthType: MLIRType {
    /// The bit width of this type
    var bitWidth: UInt { get }
}

/// Protocol for shaped types (tensors, memrefs, vectors)
public protocol ShapedType: MLIRType {
    /// The shape of this type (dimensions)
    var shape: [Int64] { get }
    
    /// The element type of this shaped type
    var elementType: any MLIRType { get }
    
    /// The rank (number of dimensions)
    var rank: Int { get }
    
    /// Whether this type has a static shape (all dimensions known)
    var hasStaticShape: Bool { get }
}

/// Integer types in MLIR
public struct IntegerType: BitWidthType {
    public let typeHandle: MlirType
    public let context: MLIRContext
    public let bitWidth: UInt
    public let isSigned: Bool
    
    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
        self.bitWidth = UInt(mlirIntegerTypeGetWidthWrapper(handle))
        self.isSigned = mlirIntegerTypeIsSignedWrapper(handle)
    }

    /// Creates a new integer type with the specified bit width
    public init(bitWidth: UInt, signed: Bool = false, context: MLIRContext = MLIRContext()) {
        self.context = context
        self.bitWidth = bitWidth
        self.isSigned = signed

        if signed {
            self.typeHandle = mlirIntegerTypeSignedGetWrapper(context.handle, UInt32(bitWidth))
        } else {
            self.typeHandle = mlirIntegerTypeGetWrapper(context.handle, UInt32(bitWidth))
        }
    }
}

/// Floating-point types in MLIR
public struct FloatType: BitWidthType {
    public let typeHandle: MlirType
    public let context: MLIRContext
    public let bitWidth: UInt
    
    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context

        // Determine bit width based on float type
        if mlirTypeIsAF16Wrapper(handle) {
            self.bitWidth = 16
        } else if mlirTypeIsAF32Wrapper(handle) {
            self.bitWidth = 32
        } else if mlirTypeIsAF64Wrapper(handle) {
            self.bitWidth = 64
        } else {
            self.bitWidth = 0
        }
    }

    /// Creates a new float type with the specified bit width
    public init(bitWidth: UInt, context: MLIRContext = MLIRContext()) {
        self.context = context
        self.bitWidth = bitWidth

        switch bitWidth {
        case 16:
            self.typeHandle = mlirF16TypeGetWrapper(context.handle)
        case 32:
            self.typeHandle = mlirF32TypeGetWrapper(context.handle)
        case 64:
            self.typeHandle = mlirF64TypeGetWrapper(context.handle)
        default:
            fatalError("Unsupported float bit width: \(bitWidth)")
        }
    }
}

/// Index type in MLIR (for indexing and dimension sizes)
public struct IndexType: MLIRType {
    public let typeHandle: MlirType
    public let context: MLIRContext
    
    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
    }

    public init(context: MLIRContext = MLIRContext()) {
        self.context = context
        self.typeHandle = mlirIndexTypeGetWrapper(context.handle)
    }
}

/// Convenience type aliases
extension IntegerType {
    /// Creates an i1 (boolean) type
    public static func i1(context: MLIRContext = MLIRContext()) -> IntegerType {
        IntegerType(bitWidth: 1, context: context)
    }
    
    /// Creates an i8 type
    public static func i8(context: MLIRContext = MLIRContext()) -> IntegerType {
        IntegerType(bitWidth: 8, context: context)
    }
    
    /// Creates an i16 type
    public static func i16(context: MLIRContext = MLIRContext()) -> IntegerType {
        IntegerType(bitWidth: 16, context: context)
    }
    
    /// Creates an i32 type
    public static func i32(context: MLIRContext = MLIRContext()) -> IntegerType {
        IntegerType(bitWidth: 32, context: context)
    }
    
    /// Creates an i64 type
    public static func i64(context: MLIRContext = MLIRContext()) -> IntegerType {
        IntegerType(bitWidth: 64, context: context)
    }
}

extension FloatType {
    /// Creates an f16 type
    public static func f16(context: MLIRContext = MLIRContext()) -> FloatType {
        FloatType(bitWidth: 16, context: context)
    }

    /// Creates an f32 type
    public static func f32(context: MLIRContext = MLIRContext()) -> FloatType {
        FloatType(bitWidth: 32, context: context)
    }

    /// Creates an f64 type
    public static func f64(context: MLIRContext = MLIRContext()) -> FloatType {
        FloatType(bitWidth: 64, context: context)
    }
}

// MARK: - Shaped Types

/// Ranked tensor type in MLIR
public struct RankedTensorType: ShapedType {
    public let typeHandle: MlirType
    public let context: MLIRContext

    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public var shape: [Int64] {
        guard mlirTypeIsARankedTensorWrapper(typeHandle) else { return [] }
        let rank = Int(mlirShapedTypeGetRankWrapper(typeHandle))
        return (0..<rank).map { mlirShapedTypeGetDimSizeWrapper(typeHandle, $0) }
    }

    public var elementType: any MLIRType {
        let elemHandle = mlirShapedTypeGetElementTypeWrapper(typeHandle)

        // Try to identify the element type
        if mlirTypeIsAF32Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF64Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF16Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else {
            return IntegerType(handle: elemHandle, context: context)
        }
    }

    public var rank: Int {
        guard mlirTypeIsARankedTensorWrapper(typeHandle) else { return 0 }
        return Int(mlirShapedTypeGetRankWrapper(typeHandle))
    }

    public var hasStaticShape: Bool {
        guard mlirTypeIsARankedTensorWrapper(typeHandle) else { return false }
        return mlirShapedTypeHasStaticShapeWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
    }

    /// Creates a ranked tensor type with the specified shape and element type
    public init<T: MLIRType>(shape: [Int64], elementType: T, context: MLIRContext = MLIRContext()) {
        self.context = context

        var shapeArray = shape
        let nullAttr = mlirAttributeGetNullWrapper()

        self.typeHandle = shapeArray.withUnsafeMutableBufferPointer { shapePtr in
            mlirRankedTensorTypeGetWrapper(
                context.handle,
                shape.count,
                shapePtr.baseAddress!,
                elementType.typeHandle,
                nullAttr
            )
        }
    }
}

/// Unranked tensor type in MLIR
public struct UnrankedTensorType: MLIRType {
    public let typeHandle: MlirType
    public let context: MLIRContext

    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
    }

    /// Creates an unranked tensor type with the specified element type
    public init<T: MLIRType>(elementType: T, context: MLIRContext = MLIRContext()) {
        self.context = context
        self.typeHandle = mlirUnrankedTensorTypeGetWrapper(context.handle, elementType.typeHandle)
    }
}

/// MemRef type in MLIR (for buffers and memory regions)
public struct MemRefType: ShapedType {
    public let typeHandle: MlirType
    public let context: MLIRContext

    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public var shape: [Int64] {
        guard mlirTypeIsAMemRefWrapper(typeHandle) else { return [] }
        let rank = Int(mlirShapedTypeGetRankWrapper(typeHandle))
        return (0..<rank).map { mlirShapedTypeGetDimSizeWrapper(typeHandle, $0) }
    }

    public var elementType: any MLIRType {
        let elemHandle = mlirShapedTypeGetElementTypeWrapper(typeHandle)

        // Try to identify the element type
        if mlirTypeIsAF32Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF64Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF16Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else {
            return IntegerType(handle: elemHandle, context: context)
        }
    }

    public var rank: Int {
        guard mlirTypeIsAMemRefWrapper(typeHandle) else { return 0 }
        return Int(mlirShapedTypeGetRankWrapper(typeHandle))
    }

    public var hasStaticShape: Bool {
        guard mlirTypeIsAMemRefWrapper(typeHandle) else { return false }
        return mlirShapedTypeHasStaticShapeWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
    }

    /// Creates a memref type with the specified shape and element type
    public init<T: MLIRType>(shape: [Int64], elementType: T, context: MLIRContext = MLIRContext()) {
        self.context = context

        var shapeArray = shape
        let nullAttr = mlirAttributeGetNullWrapper()

        self.typeHandle = shapeArray.withUnsafeMutableBufferPointer { shapePtr in
            mlirMemRefTypeGetWrapper(
                context.handle,
                elementType.typeHandle,
                shape.count,
                shapePtr.baseAddress!,
                nullAttr,
                nullAttr
            )
        }
    }
}

/// Vector type in MLIR (for SIMD vectors)
public struct VectorType: ShapedType {
    public let typeHandle: MlirType
    public let context: MLIRContext

    public var isValid: Bool {
        !mlirTypeIsNullWrapper(typeHandle)
    }

    public var shape: [Int64] {
        guard mlirTypeIsAVectorWrapper(typeHandle) else { return [] }
        let rank = Int(mlirShapedTypeGetRankWrapper(typeHandle))
        return (0..<rank).map { mlirShapedTypeGetDimSizeWrapper(typeHandle, $0) }
    }

    public var elementType: any MLIRType {
        let elemHandle = mlirShapedTypeGetElementTypeWrapper(typeHandle)

        // Try to identify the element type
        if mlirTypeIsAF32Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF64Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else if mlirTypeIsAF16Wrapper(elemHandle) {
            return FloatType(handle: elemHandle, context: context)
        } else {
            return IntegerType(handle: elemHandle, context: context)
        }
    }

    public var rank: Int {
        guard mlirTypeIsAVectorWrapper(typeHandle) else { return 0 }
        return Int(mlirShapedTypeGetRankWrapper(typeHandle))
    }

    public var hasStaticShape: Bool {
        guard mlirTypeIsAVectorWrapper(typeHandle) else { return false }
        return mlirShapedTypeHasStaticShapeWrapper(typeHandle)
    }

    public init(handle: MlirType, context: MLIRContext) {
        self.typeHandle = handle
        self.context = context
    }

    /// Creates a vector type with the specified shape and element type
    public init<T: MLIRType>(shape: [Int64], elementType: T, context: MLIRContext = MLIRContext()) {
        self.context = context

        var shapeArray = shape
        self.typeHandle = shapeArray.withUnsafeMutableBufferPointer { shapePtr in
            mlirVectorTypeGetWrapper(
                shape.count,
                shapePtr.baseAddress!,
                elementType.typeHandle
            )
        }
    }
}
