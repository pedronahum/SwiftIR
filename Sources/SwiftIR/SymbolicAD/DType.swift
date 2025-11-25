/// DType - Data type representation for tensors
/// Part of SwiftIR Symbolic Pullback Tracing system

/// Data types supported by the tensor system
/// Maps to MLIR/StableHLO element types
public enum DType: String, Hashable, CaseIterable, Sendable {
    // Floating point types
    case float16 = "f16"
    case bfloat16 = "bf16"
    case float32 = "f32"
    case float64 = "f64"

    // Integer types
    case int8 = "i8"
    case int16 = "i16"
    case int32 = "i32"
    case int64 = "i64"

    // Unsigned integer types
    case uint8 = "ui8"
    case uint16 = "ui16"
    case uint32 = "ui32"
    case uint64 = "ui64"

    // Boolean type
    case bool = "i1"

    // Complex types
    case complex64 = "complex<f32>"
    case complex128 = "complex<f64>"

    /// Size in bytes of each element
    public var sizeInBytes: Int {
        switch self {
        case .bool: return 1
        case .int8, .uint8: return 1
        case .int16, .uint16, .float16, .bfloat16: return 2
        case .int32, .uint32, .float32: return 4
        case .int64, .uint64, .float64, .complex64: return 8
        case .complex128: return 16
        }
    }

    /// Whether this is a floating point type
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .bfloat16, .float32, .float64: return true
        default: return false
        }
    }

    /// Whether this is an integer type
    public var isInteger: Bool {
        switch self {
        case .int8, .int16, .int32, .int64,
            .uint8, .uint16, .uint32, .uint64: return true
        default: return false
        }
    }

    /// Whether this is an unsigned type
    public var isUnsigned: Bool {
        switch self {
        case .uint8, .uint16, .uint32, .uint64, .bool: return true
        default: return false
        }
    }

    /// Whether this is a complex type
    public var isComplex: Bool {
        switch self {
        case .complex64, .complex128: return true
        default: return false
        }
    }

    /// The MLIR type string representation
    public var mlirTypeString: String {
        return rawValue
    }

    /// Get the promoted type for arithmetic operations
    /// Follows standard type promotion rules
    public func promoted(with other: DType) -> DType {
        // Complex promotion
        if self.isComplex || other.isComplex {
            if self == .complex128 || other == .complex128 { return .complex128 }
            return .complex64
        }

        // Float promotion
        if self.isFloatingPoint || other.isFloatingPoint {
            let maxSize = max(self.sizeInBytes, other.sizeInBytes)
            if maxSize >= 8 { return .float64 }
            if maxSize >= 4 { return .float32 }
            // Both are float16/bfloat16, prefer float16
            if self == .bfloat16 || other == .bfloat16 { return .bfloat16 }
            return .float16
        }

        // Integer promotion
        let selfSize = self.sizeInBytes
        let otherSize = other.sizeInBytes
        let maxSize = max(selfSize, otherSize)

        // If either is unsigned, result is unsigned (unless signed is larger)
        let resultUnsigned = (self.isUnsigned && selfSize >= otherSize)
            || (other.isUnsigned && otherSize >= selfSize)

        switch maxSize {
        case 1: return resultUnsigned ? .uint8 : .int8
        case 2: return resultUnsigned ? .uint16 : .int16
        case 4: return resultUnsigned ? .uint32 : .int32
        default: return resultUnsigned ? .uint64 : .int64
        }
    }
}

// MARK: - Swift Type Mapping

public extension DType {
    /// Get the DType corresponding to a Swift type
    static func from<T: BinaryFloatingPoint>(_ type: T.Type) -> DType {
        switch MemoryLayout<T>.size {
        case 2: return .float16
        case 4: return .float32
        case 8: return .float64
        default: return .float32
        }
    }

    /// Get the DType corresponding to a Swift integer type
    static func from<T: BinaryInteger>(_ type: T.Type) -> DType {
        let isSigned = T.isSigned
        switch MemoryLayout<T>.size {
        case 1: return isSigned ? .int8 : .uint8
        case 2: return isSigned ? .int16 : .uint16
        case 4: return isSigned ? .int32 : .uint32
        default: return isSigned ? .int64 : .uint64
        }
    }
}

// MARK: - Type Errors

/// Errors related to data type operations
public enum DTypeError: Error, CustomStringConvertible {
    case incompatibleTypes(lhs: DType, rhs: DType, operation: String)
    case unsupportedType(dtype: DType, operation: String)
    case implicitConversionNotAllowed(from: DType, to: DType)

    public var description: String {
        switch self {
        case .incompatibleTypes(let lhs, let rhs, let op):
            return """
            Incompatible types in \(op):
              LHS type: \(lhs)
              RHS type: \(rhs)

            Hint: Explicitly cast one operand to match the other.
            """

        case .unsupportedType(let dtype, let op):
            return """
            Type \(dtype) is not supported for \(op).
            """

        case .implicitConversionNotAllowed(let from, let to):
            return """
            Implicit conversion from \(from) to \(to) is not allowed.
            Use explicit `cast(_:to:)` instead.

            Note: SwiftIR does not allow implicit type conversions to prevent precision loss.
            """
        }
    }
}
