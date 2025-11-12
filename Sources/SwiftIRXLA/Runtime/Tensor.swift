//===-- Tensor.swift - Runtime Tensor Data Structure -----*- Swift -*-===//
//
// SwiftIR - Phase 9: Execution Runtime
// Runtime tensor with actual data storage
//
//===------------------------------------------------------------------===//

import Foundation

/// Runtime tensor with actual data
///
/// This is different from MLIRValue which represents symbolic tensors
/// in the IR. Tensor holds actual numerical data for execution.
public struct Tensor {
    /// The raw data buffer
    let data: Data

    /// Shape of the tensor
    public let shape: [Int64]

    /// Element data type
    public let dtype: DataType

    /// Total number of elements
    public var elementCount: Int {
        Int(shape.reduce(1, *))
    }

    /// Size in bytes
    public var byteSize: Int {
        data.count
    }

    /// Initialize tensor with raw data
    public init(data: Data, shape: [Int64], dtype: DataType) {
        self.data = data
        self.shape = shape
        self.dtype = dtype
    }

    /// Create tensor from Float array
    public static func from(_ values: [Float], shape: [Int64]) -> Tensor {
        let data = values.withUnsafeBytes { Data($0) }
        return Tensor(data: data, shape: shape, dtype: .float32)
    }

    /// Create tensor from Double array
    public static func from(_ values: [Double], shape: [Int64]) -> Tensor {
        let data = values.withUnsafeBytes { Data($0) }
        return Tensor(data: data, shape: shape, dtype: .float64)
    }

    /// Create tensor from Int32 array
    public static func from(_ values: [Int32], shape: [Int64]) -> Tensor {
        let data = values.withUnsafeBytes { Data($0) }
        return Tensor(data: data, shape: shape, dtype: .int32)
    }

    /// Create tensor filled with zeros
    public static func zeros(shape: [Int64], dtype: DataType = .float32) -> Tensor {
        let elementCount = shape.reduce(1, *)
        let byteSize = Int(elementCount) * dtype.byteSize
        let data = Data(count: byteSize)
        return Tensor(data: data, shape: shape, dtype: dtype)
    }

    /// Create tensor filled with ones
    public static func ones(shape: [Int64], dtype: DataType = .float32) -> Tensor {
        let elementCount = shape.reduce(1, *)
        switch dtype {
        case .float32:
            let values = [Float](repeating: 1.0, count: Int(elementCount))
            return Tensor.from(values, shape: shape)
        case .float64:
            let values = [Double](repeating: 1.0, count: Int(elementCount))
            return Tensor.from(values, shape: shape)
        case .int32:
            let values = [Int32](repeating: 1, count: Int(elementCount))
            return Tensor.from(values, shape: shape)
        case .int64:
            let values = [Int64](repeating: 1, count: Int(elementCount))
            return Tensor(data: values.withUnsafeBytes { Data($0) }, shape: shape, dtype: dtype)
        }
    }

    /// Convert tensor data to Float array
    public func toFloat() -> [Float]? {
        guard dtype == .float32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }

    /// Convert tensor data to Double array
    public func toDouble() -> [Double]? {
        guard dtype == .float64 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Double.self))
        }
    }

    /// Convert tensor data to Int32 array
    public func toInt32() -> [Int32]? {
        guard dtype == .int32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int32.self))
        }
    }

    /// Access raw data pointer (for passing to C APIs)
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try data.withUnsafeBytes(body)
    }

    /// Create mutable copy for in-place operations
    public func mutableCopy() -> MutableTensor {
        MutableTensor(data: data, shape: shape, dtype: dtype)
    }
}

/// Mutable tensor for in-place operations
public struct MutableTensor {
    /// The raw data buffer (mutable)
    var data: Data

    /// Shape of the tensor
    public let shape: [Int64]

    /// Element data type
    public let dtype: DataType

    /// Initialize with data
    init(data: Data, shape: [Int64], dtype: DataType) {
        self.data = data
        self.shape = shape
        self.dtype = dtype
    }

    /// Convert back to immutable tensor
    public func freeze() -> Tensor {
        Tensor(data: data, shape: shape, dtype: dtype)
    }

    /// Access mutable raw data pointer
    public mutating func withUnsafeMutableBytes<R>(
        _ body: (UnsafeMutableRawBufferPointer) throws -> R
    ) rethrows -> R {
        try data.withUnsafeMutableBytes(body)
    }
}

/// Supported data types for tensors
public enum DataType {
    case float32
    case float64
    case int32
    case int64

    /// Size in bytes of one element
    public var byteSize: Int {
        switch self {
        case .float32, .int32:
            return 4
        case .float64, .int64:
            return 8
        }
    }

    /// String representation matching MLIR types
    public var mlirTypeName: String {
        switch self {
        case .float32:
            return "f32"
        case .float64:
            return "f64"
        case .int32:
            return "i32"
        case .int64:
            return "i64"
        }
    }
}

// MARK: - CustomStringConvertible

extension Tensor: CustomStringConvertible {
    public var description: String {
        let shapeStr = shape.map(String.init).joined(separator: "x")
        return "Tensor<\(shapeStr)x\(dtype.mlirTypeName)>"
    }
}

extension DataType: CustomStringConvertible {
    public var description: String {
        mlirTypeName
    }
}
