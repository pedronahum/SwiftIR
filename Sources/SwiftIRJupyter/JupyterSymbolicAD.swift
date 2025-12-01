// JupyterSymbolicAD.swift
// High-level symbolic automatic differentiation for Jupyter/Colab
// Pure Swift - works without C++ interop via dlopen/dlsym

import Foundation
import _Differentiation

// MARK: - TensorShape

/// Type-safe tensor shape representation with broadcasting support
public struct JTensorShape: Hashable, Equatable, CustomStringConvertible, Sendable {
    /// Dimensions of the tensor. nil represents a dynamic dimension
    public let dimensions: [Int?]

    /// Rank of the tensor (number of dimensions)
    public var rank: Int { dimensions.count }

    /// Returns true if all dimensions are statically known
    public var isFullyKnown: Bool { dimensions.allSatisfy { $0 != nil } }

    /// Returns the total number of elements (nil if any dimension is dynamic)
    public var elementCount: Int? {
        guard isFullyKnown else { return nil }
        return dimensions.compactMap { $0 }.reduce(1, *)
    }

    /// String representation for debugging
    public var description: String {
        let dims = dimensions.map { d -> String in
            if let d = d { return "\(d)" }
            return "?"
        }
        return "[\(dims.joined(separator: ", "))]"
    }

    // MARK: - Initializers

    /// Create a shape from an array of dimensions
    public init(dimensions: [Int?]) {
        self.dimensions = dimensions
    }

    /// Create a shape from known dimensions
    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions.map { $0 as Int? }
    }

    /// Create a scalar (0-dimensional) shape
    public static let scalar = JTensorShape(dimensions: [])

    // MARK: - Broadcasting

    /// Check if this shape is broadcast-compatible with another shape
    public func isBroadcastCompatible(with other: JTensorShape) -> Bool {
        let maxRank = max(self.rank, other.rank)

        let selfPadded = Array(repeating: 1 as Int?, count: maxRank - rank) + dimensions
        let otherPadded = Array(repeating: 1 as Int?, count: maxRank - other.rank) + other.dimensions

        for (d1, d2) in zip(selfPadded, otherPadded) {
            if d1 == nil || d2 == nil { continue }
            if d1 == d2 { continue }
            if d1 == 1 || d2 == 1 { continue }
            return false
        }

        return true
    }

    /// Compute the result shape after broadcasting with another shape
    public func broadcast(with other: JTensorShape) -> JTensorShape {
        let maxRank = max(self.rank, other.rank)
        var resultDims: [Int?] = []

        let selfPadded = Array(repeating: 1 as Int?, count: maxRank - rank) + dimensions
        let otherPadded = Array(repeating: 1 as Int?, count: maxRank - other.rank) + other.dimensions

        for (d1, d2) in zip(selfPadded, otherPadded) {
            if let dim1 = d1, let dim2 = d2 {
                resultDims.append(max(dim1, dim2))
            } else if let dim1 = d1 {
                resultDims.append(dim1 == 1 ? d2 : dim1)
            } else if let dim2 = d2 {
                resultDims.append(dim2 == 1 ? d1 : dim2)
            } else {
                resultDims.append(nil)
            }
        }

        return JTensorShape(dimensions: resultDims)
    }

    /// Get the shape after reducing along specified axes
    public func reduced(alongAxes axes: Set<Int>, keepDims: Bool = false) -> JTensorShape {
        var resultDims: [Int?] = []

        for (index, dim) in dimensions.enumerated() {
            if axes.contains(index) {
                if keepDims {
                    resultDims.append(1)
                }
            } else {
                resultDims.append(dim)
            }
        }

        return JTensorShape(dimensions: resultDims)
    }

    // Convenience factory methods
    public static func vector(_ size: Int) -> JTensorShape { JTensorShape([size]) }
    public static func matrix(_ rows: Int, _ cols: Int) -> JTensorShape { JTensorShape([rows, cols]) }
}

// MARK: - DType

/// Data types supported by the tensor system
public enum JDType: String, Hashable, CaseIterable, Sendable {
    case float16 = "f16"
    case bfloat16 = "bf16"
    case float32 = "f32"
    case float64 = "f64"
    case int8 = "i8"
    case int16 = "i16"
    case int32 = "i32"
    case int64 = "i64"
    case bool = "i1"

    /// Size in bytes of each element
    public var sizeInBytes: Int {
        switch self {
        case .bool, .int8: return 1
        case .int16, .float16, .bfloat16: return 2
        case .int32, .float32: return 4
        case .int64, .float64: return 8
        }
    }

    /// Whether this is a floating point type
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .bfloat16, .float32, .float64: return true
        default: return false
        }
    }

    /// Get the promoted type for arithmetic operations
    public func promoted(with other: JDType) -> JDType {
        if self.isFloatingPoint || other.isFloatingPoint {
            let maxSize = max(self.sizeInBytes, other.sizeInBytes)
            if maxSize >= 8 { return .float64 }
            return .float32
        }
        return .int32
    }
}

// MARK: - MLIRValue

/// Represents an SSA value in the MLIR graph
public struct JMLIRValue: Hashable {
    internal let id: UInt64

    internal init(id: UInt64) {
        self.id = id
    }
}

// MARK: - Token (for side-effect ordering)

/// Represents a sequencing token for side-effecting operations
public struct JToken: Hashable {
    internal let id: UInt64

    internal init(id: UInt64) {
        self.id = id
    }

    /// The global token that sequences all side effects
    public static var global: JToken {
        JTokenManager.shared.globalToken
    }

    /// Create a new token
    internal static func create() -> JToken {
        JTokenManager.shared.createToken()
    }
}

/// Manages token creation
internal final class JTokenManager: @unchecked Sendable {
    nonisolated(unsafe) static let shared = JTokenManager()

    private var nextId: UInt64 = 1
    private let lock = NSLock()
    let globalToken: JToken

    private init() {
        self.globalToken = JToken(id: 0)
    }

    func createToken() -> JToken {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        return JToken(id: id)
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 1
    }
}

// MARK: - Traced Operation Types

/// Types of operations that can be traced
public enum JTracedOperation {
    case constant(id: UInt64, value: Double, shape: JTensorShape, dtype: JDType)
    case placeholder(id: UInt64, shape: JTensorShape, dtype: JDType)
    case binary(id: UInt64, op: JBinaryOperation, lhs: UInt64, rhs: UInt64, shape: JTensorShape, dtype: JDType)
    case unary(id: UInt64, op: JUnaryOperation, input: UInt64, shape: JTensorShape, dtype: JDType)
    case reduction(id: UInt64, op: JReductionOperation, input: UInt64, axes: Set<Int>, keepDims: Bool, shape: JTensorShape, dtype: JDType)
    case power(id: UInt64, base: UInt64, exponent: Double, shape: JTensorShape, dtype: JDType)
    case comparison(id: UInt64, lhs: UInt64, rhs: UInt64, direction: String, inputDtype: JDType)
    case reshape(id: UInt64, input: UInt64, newShape: JTensorShape, dtype: JDType)
    case clamp(id: UInt64, input: UInt64, minVal: Double, maxVal: Double, shape: JTensorShape, dtype: JDType)
    case slice(id: UInt64, input: UInt64, starts: [Int], limits: [Int], strides: [Int], shape: JTensorShape, dtype: JDType)
    case concatenate(id: UInt64, inputs: [UInt64], axis: Int, shape: JTensorShape, dtype: JDType)
    case binaryElementwise(id: UInt64, op: JBinaryElementwiseOp, lhs: UInt64, rhs: UInt64, shape: JTensorShape, dtype: JDType)
    case select(id: UInt64, condition: UInt64, onTrue: UInt64, onFalse: UInt64, shape: JTensorShape, dtype: JDType)
    case unaryWithAlpha(id: UInt64, op: JUnaryOperation, input: UInt64, alpha: Double, shape: JTensorShape, dtype: JDType)
    case dynamicSlice(id: UInt64, input: UInt64, startIndex: UInt64, sliceSizes: [Int], shape: JTensorShape, dtype: JDType)
    case dynamicUpdateSlice(id: UInt64, operand: UInt64, update: UInt64, startIndex: UInt64, shape: JTensorShape, dtype: JDType)
    case broadcastInDim(id: UInt64, input: UInt64, broadcastDimensions: [Int], inputShape: JTensorShape, outputShape: JTensorShape, dtype: JDType)
    case compare(id: UInt64, lhs: UInt64, rhs: UInt64, direction: JCompareDirection, shape: JTensorShape, inputDtype: JDType)
    case transpose(id: UInt64, input: UInt64, permutation: [Int], inputShape: JTensorShape, outputShape: JTensorShape, dtype: JDType)
    case dotGeneral(id: UInt64, lhs: UInt64, rhs: UInt64, lhsBatchingDims: [Int], rhsBatchingDims: [Int], lhsContractingDims: [Int], rhsContractingDims: [Int], resultShape: JTensorShape, dtype: JDType)
    case rng(id: UInt64, keyData: (UInt32, UInt32), shape: JTensorShape, dtype: JDType, distribution: JRngDistribution)
    case convert(id: UInt64, input: UInt64, inputDtype: JDType, outputDtype: JDType, shape: JTensorShape)
}

/// Binary operations
public enum JBinaryOperation: String {
    case add, subtract, multiply, divide, matmul
}

/// Unary operations
public enum JUnaryOperation: String {
    case exp, log, sqrt, abs, neg, sin, cos, tan, tanh, sigmoid, relu, reshape, transpose
    case rsqrt, floor, ceil  // Additional math operations
    case leakyRelu, elu, silu, gelu, softplus  // Advanced activations
    case softmax, logSoftmax  // Softmax variants
}

/// Reduction operations
public enum JReductionOperation: String {
    case sum, mean, max, min
}

/// Binary max/min operations
public enum JBinaryElementwiseOp: String {
    case maximum, minimum
}

// MARK: - TracerGraphBuilder

/// Graph builder for tracing operations (exposed for advanced usage)
public final class JTracerGraphBuilder: @unchecked Sendable {
    public nonisolated(unsafe) static let shared = JTracerGraphBuilder()

    private var nextId: UInt64 = 1
    private let lock = NSLock()
    private var operations: [JTracedOperation] = []

    private init() {}

    func createConstant(value: Double, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.constant(id: id, value: value, shape: shape, dtype: dtype))
        return id
    }

    func createPlaceholder(shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.placeholder(id: id, shape: shape, dtype: dtype))
        return id
    }

    func createBinaryOp(operation: JBinaryOperation, lhs: JMLIRValue, rhs: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.binary(id: id, op: operation, lhs: lhs.id, rhs: rhs.id, shape: shape, dtype: dtype))
        return id
    }

    func createUnaryOp(operation: JUnaryOperation, input: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.unary(id: id, op: operation, input: input.id, shape: shape, dtype: dtype))
        return id
    }

    func createReduction(operation: JReductionOperation, input: JMLIRValue, axes: Set<Int>, keepDims: Bool, resultShape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.reduction(id: id, op: operation, input: input.id, axes: axes, keepDims: keepDims, shape: resultShape, dtype: dtype))
        return id
    }

    func createMatMul(lhs: JMLIRValue, rhs: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.binary(id: id, op: .matmul, lhs: lhs.id, rhs: rhs.id, shape: shape, dtype: dtype))
        return id
    }

    func createTranspose(input: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.unary(id: id, op: .transpose, input: input.id, shape: shape, dtype: dtype))
        return id
    }

    func createPower(base: JMLIRValue, exponent: Double, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.power(id: id, base: base.id, exponent: exponent, shape: shape, dtype: dtype))
        return id
    }

    func createComparison(lhs: JMLIRValue, rhs: JMLIRValue, direction: JComparisonDirection, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.comparison(id: id, lhs: lhs.id, rhs: rhs.id, direction: direction.rawValue, inputDtype: dtype))
        return id
    }

    func createReshape(input: JMLIRValue, newShape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.reshape(id: id, input: input.id, newShape: newShape, dtype: dtype))
        return id
    }

    func createClamp(input: JMLIRValue, minVal: Double, maxVal: Double, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.clamp(id: id, input: input.id, minVal: minVal, maxVal: maxVal, shape: shape, dtype: dtype))
        return id
    }

    func createSlice(input: JMLIRValue, starts: [Int], limits: [Int], strides: [Int], resultShape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.slice(id: id, input: input.id, starts: starts, limits: limits, strides: strides, shape: resultShape, dtype: dtype))
        return id
    }

    func createConcatenate(inputs: [JMLIRValue], axis: Int, resultShape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.concatenate(id: id, inputs: inputs.map { $0.id }, axis: axis, shape: resultShape, dtype: dtype))
        return id
    }

    func createBinaryElementwise(op: JBinaryElementwiseOp, lhs: JMLIRValue, rhs: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.binaryElementwise(id: id, op: op, lhs: lhs.id, rhs: rhs.id, shape: shape, dtype: dtype))
        return id
    }

    func createSelect(condition: JMLIRValue, onTrue: JMLIRValue, onFalse: JMLIRValue, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.select(id: id, condition: condition.id, onTrue: onTrue.id, onFalse: onFalse.id, shape: shape, dtype: dtype))
        return id
    }

    func createUnaryWithAlpha(op: JUnaryOperation, input: JMLIRValue, alpha: Double, shape: JTensorShape, dtype: JDType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        operations.append(.unaryWithAlpha(id: id, op: op, input: input.id, alpha: alpha, shape: shape, dtype: dtype))
        return id
    }

    func getOperations() -> [JTracedOperation] {
        lock.lock()
        defer { lock.unlock() }
        return operations
    }

    /// Get count of operations (for region tracing)
    func operationCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return operations.count
    }

    /// Get operations added since a given count (for region tracing)
    func operationsSince(_ count: Int) -> [JTracedOperation] {
        lock.lock()
        defer { lock.unlock() }
        guard count < operations.count else { return [] }
        return Array(operations[count...])
    }

    /// Remove operations from a given index (for region tracing)
    func removeOperationsSince(_ count: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard count < operations.count else { return }
        operations.removeSubrange(count...)
    }

    /// Get next ID (for creating region arguments)
    public func getNextId() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        return id
    }

    /// Add an operation to the trace
    public func addOperation(_ op: JTracedOperation) {
        lock.lock()
        defer { lock.unlock() }
        operations.append(op)
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 1
        operations.removeAll()
    }
}

// MARK: - JTracer (The "Trojan Horse" Type)

/// The core symbolic tensor type that traces operations into MLIR
///
/// JTracer looks like a number to Swift's type system but actually emits
/// MLIR operations when manipulated. This is the "Trojan Horse" mechanism
/// that allows Swift's native `@differentiable` to build gradient graphs.
///
/// Example:
/// ```swift
/// @differentiable(reverse)
/// func model(_ x: JTracer) -> JTracer {
///     return x * 2.0 + 1.0  // Emits MLIR multiply and add operations
/// }
///
/// let grad = gradient(at: x, in: model)  // Builds forward + backward graph
/// ```
public struct JTracer: Differentiable {
    // MARK: - Core Properties

    private let irValue: JMLIRValue
    public let shape: JTensorShape
    public let dtype: JDType
    private let version: UInt64

    // MARK: - Static Version Counter

    private nonisolated(unsafe) static var _globalVersion: UInt64 = 0
    private static let versionLock = NSLock()

    internal static func incrementVersion() -> UInt64 {
        versionLock.lock()
        defer { versionLock.unlock() }
        _globalVersion += 1
        return _globalVersion
    }

    // MARK: - Initializers

    internal init(irValue: JMLIRValue, shape: JTensorShape, dtype: JDType, version: UInt64) {
        self.irValue = irValue
        self.shape = shape
        self.dtype = dtype
        self.version = version
    }

    /// Create a tracer from a scalar value (for constants)
    public init(value: Double, shape: JTensorShape = .scalar, dtype: JDType = .float32) {
        let id = JTracerGraphBuilder.shared.createConstant(value: value, shape: shape, dtype: dtype)
        self.irValue = JMLIRValue(id: id)
        self.shape = shape
        self.dtype = dtype
        self.version = Self.incrementVersion()
    }

    /// Create a tracer with a specific shape and type
    public init(shape: JTensorShape, dtype: JDType = .float32) {
        let id = JTracerGraphBuilder.shared.createPlaceholder(shape: shape, dtype: dtype)
        self.irValue = JMLIRValue(id: id)
        self.shape = shape
        self.dtype = dtype
        self.version = Self.incrementVersion()
    }

    // MARK: - Safe Value Accessor

    public var value: JMLIRValue { irValue }
    public var valueId: UInt64 { irValue.id }

    // MARK: - Differentiable Conformance

    public typealias TangentVector = JTracer

    public mutating func move(by offset: TangentVector) {
        let newId = JTracerGraphBuilder.shared.createBinaryOp(
            operation: .add,
            lhs: self.irValue,
            rhs: offset.irValue,
            shape: self.shape,
            dtype: self.dtype
        )
        self = JTracer(
            irValue: JMLIRValue(id: newId),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion()
        )
    }
}

// MARK: - AdditiveArithmetic Conformance

extension JTracer: AdditiveArithmetic {
    public static var zero: JTracer {
        return JTracer(value: 0.0, shape: .scalar, dtype: .float32)
    }

    public static func + (lhs: JTracer, rhs: JTracer) -> JTracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = JTracerGraphBuilder.shared.createBinaryOp(
            operation: .add,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion()
        )
    }

    public static func - (lhs: JTracer, rhs: JTracer) -> JTracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = JTracerGraphBuilder.shared.createBinaryOp(
            operation: .subtract,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion()
        )
    }
}

// MARK: - Numeric Operations

extension JTracer {
    public static func * (lhs: JTracer, rhs: JTracer) -> JTracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = JTracerGraphBuilder.shared.createBinaryOp(
            operation: .multiply,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion()
        )
    }

    public static func / (lhs: JTracer, rhs: JTracer) -> JTracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = JTracerGraphBuilder.shared.createBinaryOp(
            operation: .divide,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return JTracer(
            irValue: JMLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion()
        )
    }

    // Scalar operations
    public static func * (lhs: JTracer, rhs: Double) -> JTracer {
        let scalar = JTracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs * scalar
    }

    public static func * (lhs: Double, rhs: JTracer) -> JTracer {
        let scalar = JTracer(value: lhs, shape: .scalar, dtype: rhs.dtype)
        return scalar * rhs
    }

    public static func + (lhs: JTracer, rhs: Double) -> JTracer {
        let scalar = JTracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs + scalar
    }

    public static func + (lhs: Double, rhs: JTracer) -> JTracer {
        let scalar = JTracer(value: lhs, shape: .scalar, dtype: rhs.dtype)
        return scalar + rhs
    }

    public static func - (lhs: JTracer, rhs: Double) -> JTracer {
        let scalar = JTracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs - scalar
    }

    public static func / (lhs: JTracer, rhs: Double) -> JTracer {
        let scalar = JTracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs / scalar
    }
}

// MARK: - ExpressibleByFloatLiteral

extension JTracer: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(value: value, shape: .scalar, dtype: .float32)
    }
}

// MARK: - Unary Operations

extension JTracer {
    public func exp() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .exp,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func log() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .log,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func sqrt() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .sqrt,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func tanh() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .tanh,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func sigmoid() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .sigmoid,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func relu() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .relu,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public static prefix func - (operand: JTracer) -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .neg,
            input: operand.value,
            shape: operand.shape,
            dtype: operand.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: operand.shape, dtype: operand.dtype, version: incrementVersion())
    }

    // Additional unary operations
    public func sin() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .sin,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func cos() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .cos,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func tan() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .tan,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func abs() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .abs,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Power operation: self^exponent
    public func power(_ exponent: Double) -> JTracer {
        let id = JTracerGraphBuilder.shared.createPower(
            base: self.value,
            exponent: exponent,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Reshape to a new shape
    public func reshape(to newShape: JTensorShape) -> JTracer {
        let id = JTracerGraphBuilder.shared.createReshape(
            input: self.value,
            newShape: newShape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: newShape, dtype: self.dtype, version: Self.incrementVersion())
    }

    // MARK: - Additional Math Operations

    /// Reciprocal square root: 1/sqrt(x)
    public func rsqrt() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .rsqrt,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Floor: largest integer <= x
    public func floor() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .floor,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Ceil: smallest integer >= x
    public func ceil() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .ceil,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Clamp values to range [minVal, maxVal]
    public func clamp(min minVal: Double, max maxVal: Double) -> JTracer {
        let id = JTracerGraphBuilder.shared.createClamp(
            input: self.value,
            minVal: minVal,
            maxVal: maxVal,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    // MARK: - Advanced Activation Functions

    /// Leaky ReLU: max(alpha * x, x)
    public func leakyRelu(alpha: Double = 0.01) -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryWithAlpha(
            op: .leakyRelu,
            input: self.value,
            alpha: alpha,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// ELU: x if x > 0, else alpha * (exp(x) - 1)
    public func elu(alpha: Double = 1.0) -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryWithAlpha(
            op: .elu,
            input: self.value,
            alpha: alpha,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// SiLU (Swish): x * sigmoid(x)
    public func silu() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .silu,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// GELU: Gaussian Error Linear Unit
    public func gelu() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .gelu,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Softplus: log(1 + exp(x))
    public func softplus() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .softplus,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    // MARK: - Softmax Operations

    /// Softmax along last axis
    public func softmax() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .softmax,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Log softmax along last axis
    public func logSoftmax() -> JTracer {
        let id = JTracerGraphBuilder.shared.createUnaryOp(
            operation: .logSoftmax,
            input: self.value,
            shape: self.shape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: self.shape, dtype: self.dtype, version: Self.incrementVersion())
    }

    // MARK: - Element-wise Binary Operations

    /// Element-wise maximum
    public func maximum(_ other: JTracer) -> JTracer {
        let resultShape = self.shape.broadcast(with: other.shape)
        let id = JTracerGraphBuilder.shared.createBinaryElementwise(
            op: .maximum,
            lhs: self.value,
            rhs: other.value,
            shape: resultShape,
            dtype: self.dtype.promoted(with: other.dtype)
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype.promoted(with: other.dtype), version: Self.incrementVersion())
    }

    /// Element-wise minimum
    public func minimum(_ other: JTracer) -> JTracer {
        let resultShape = self.shape.broadcast(with: other.shape)
        let id = JTracerGraphBuilder.shared.createBinaryElementwise(
            op: .minimum,
            lhs: self.value,
            rhs: other.value,
            shape: resultShape,
            dtype: self.dtype.promoted(with: other.dtype)
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype.promoted(with: other.dtype), version: Self.incrementVersion())
    }

    // MARK: - Conditional Selection

    /// Select values based on condition: condition ? onTrue : onFalse
    public static func select(condition: JTracer, onTrue: JTracer, onFalse: JTracer) -> JTracer {
        let resultShape = onTrue.shape.broadcast(with: onFalse.shape)
        let id = JTracerGraphBuilder.shared.createSelect(
            condition: condition.value,
            onTrue: onTrue.value,
            onFalse: onFalse.value,
            shape: resultShape,
            dtype: onTrue.dtype.promoted(with: onFalse.dtype)
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: onTrue.dtype.promoted(with: onFalse.dtype), version: incrementVersion())
    }

    // MARK: - Tensor Slicing

    /// Slice tensor with start indices, limit indices, and strides
    public func slice(starts: [Int], limits: [Int], strides: [Int]? = nil) -> JTracer {
        let actualStrides = strides ?? Array(repeating: 1, count: starts.count)

        // Calculate result shape
        var resultDims: [Int?] = []
        for i in 0..<starts.count {
            let start = starts[i]
            let limit = limits[i]
            let stride = actualStrides[i]
            resultDims.append((limit - start + stride - 1) / stride)
        }
        let resultShape = JTensorShape(dimensions: resultDims)

        let id = JTracerGraphBuilder.shared.createSlice(
            input: self.value,
            starts: starts,
            limits: limits,
            strides: actualStrides,
            resultShape: resultShape,
            dtype: self.dtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }
}

// MARK: - Concatenation

extension JTracer {
    /// Concatenate tensors along an axis
    public static func concatenate(_ tensors: [JTracer], axis: Int) -> JTracer {
        guard !tensors.isEmpty else {
            fatalError("Cannot concatenate empty array of tensors")
        }

        let first = tensors[0]
        var resultDims = first.shape.dimensions

        // Calculate concatenated dimension size
        var totalSize = 0
        for tensor in tensors {
            if let size = tensor.shape.dimensions[axis] {
                totalSize += size
            }
        }
        resultDims[axis] = totalSize

        let resultShape = JTensorShape(dimensions: resultDims)
        let resultDtype = tensors.reduce(first.dtype) { $0.promoted(with: $1.dtype) }

        let id = JTracerGraphBuilder.shared.createConcatenate(
            inputs: tensors.map { $0.value },
            axis: axis,
            resultShape: resultShape,
            dtype: resultDtype
        )
        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: resultDtype, version: incrementVersion())
    }
}

// MARK: - Reduction Operations

extension JTracer {
    public func sum() -> JTracer {
        let allAxes = Set(0..<shape.rank)
        return sum(alongAxes: allAxes, keepDims: false)
    }

    public func sum(alongAxes axes: Set<Int>, keepDims: Bool = false) -> JTracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = JTracerGraphBuilder.shared.createReduction(
            operation: .sum,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }

    public func mean(alongAxes axes: Set<Int>, keepDims: Bool = false) -> JTracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = JTracerGraphBuilder.shared.createReduction(
            operation: .mean,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Max reduction along specified axes
    public func max(alongAxes axes: Set<Int>, keepDims: Bool = false) -> JTracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = JTracerGraphBuilder.shared.createReduction(
            operation: .max,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }

    /// Min reduction along specified axes
    public func min(alongAxes axes: Set<Int>, keepDims: Bool = false) -> JTracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = JTracerGraphBuilder.shared.createReduction(
            operation: .min,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }
}

// MARK: - Matrix Operations

extension JTracer {
    public func matmul(_ other: JTracer) -> JTracer {
        guard shape.rank >= 2, other.shape.rank >= 2 else {
            fatalError("matmul requires tensors with rank >= 2")
        }

        let m = shape.dimensions[shape.rank - 2]
        let n = other.shape.dimensions[other.shape.rank - 1]

        var resultDims = Array(shape.dimensions.dropLast(2))
        resultDims.append(m)
        resultDims.append(n)

        let resultShape = JTensorShape(dimensions: resultDims)

        let id = JTracerGraphBuilder.shared.createMatMul(
            lhs: self.value,
            rhs: other.value,
            shape: resultShape,
            dtype: self.dtype.promoted(with: other.dtype)
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype.promoted(with: other.dtype), version: Self.incrementVersion())
    }

    public func transpose() -> JTracer {
        guard shape.rank >= 2 else { return self }

        var newDims = shape.dimensions
        let lastIdx = newDims.count - 1
        newDims.swapAt(lastIdx - 1, lastIdx)

        let resultShape = JTensorShape(dimensions: newDims)

        let id = JTracerGraphBuilder.shared.createTranspose(
            input: self.value,
            shape: resultShape,
            dtype: self.dtype
        )

        return JTracer(irValue: JMLIRValue(id: id), shape: resultShape, dtype: self.dtype, version: Self.incrementVersion())
    }
}

// MARK: - Factory Methods

extension JTracer {
    public static func zeros(shape: JTensorShape, dtype: JDType = .float32) -> JTracer {
        let id = JTracerGraphBuilder.shared.createConstant(value: 0.0, shape: shape, dtype: dtype)
        return JTracer(irValue: JMLIRValue(id: id), shape: shape, dtype: dtype, version: incrementVersion())
    }

    public static func ones(shape: JTensorShape, dtype: JDType = .float32) -> JTracer {
        let id = JTracerGraphBuilder.shared.createConstant(value: 1.0, shape: shape, dtype: dtype)
        return JTracer(irValue: JMLIRValue(id: id), shape: shape, dtype: dtype, version: incrementVersion())
    }
}

// MARK: - Helper for Gradient Reduction

extension JTracer {
    /// Reduce a gradient to match the original tensor's shape after broadcasting
    public func reduceToBroadcastShape(originalShape: JTensorShape) -> JTracer {
        if self.shape == originalShape { return self }

        var axesToReduce: Set<Int> = []
        let selfDims = self.shape.dimensions
        let origDims = originalShape.dimensions

        let rankDiff = selfDims.count - origDims.count
        let paddedOrigDims = Array(repeating: 1 as Int?, count: rankDiff) + origDims

        for (index, (selfDim, origDim)) in zip(selfDims, paddedOrigDims).enumerated() {
            if let od = origDim, od == 1, let sd = selfDim, sd > 1 {
                axesToReduce.insert(index)
            }
        }

        for i in 0..<rankDiff {
            axesToReduce.insert(i)
        }

        if axesToReduce.isEmpty { return self }

        return self.sum(alongAxes: axesToReduce, keepDims: true)
    }
}

// MARK: - Loss Functions

extension JTracer {
    /// Mean Squared Error loss: mean((predictions - targets)^2)
    public static func mseLoss(predictions: JTracer, targets: JTracer) -> JTracer {
        let diff = predictions - targets
        let squared = diff * diff
        let allAxes = Set(0..<squared.shape.rank)
        return squared.mean(alongAxes: allAxes, keepDims: false)
    }

    /// Binary Cross Entropy loss (element-wise, then reduced)
    public static func bceLoss(predictions: JTracer, targets: JTracer, epsilon: Double = 1e-7) -> JTracer {
        // BCE = -[y * log(p + eps) + (1 - y) * log(1 - p + eps)]
        let eps = JTracer(value: epsilon, shape: .scalar, dtype: predictions.dtype)
        let one = JTracer(value: 1.0, shape: .scalar, dtype: predictions.dtype)

        let logP = (predictions + eps).log()
        let log1MinusP = (one - predictions + eps).log()

        let loss = -(targets * logP + (one - targets) * log1MinusP)
        let allAxes = Set(0..<loss.shape.rank)
        return loss.mean(alongAxes: allAxes, keepDims: false)
    }

    /// Cross Entropy loss with logits (numerically stable)
    /// predictions: raw logits (before softmax)
    /// targets: one-hot encoded labels
    public static func crossEntropyLoss(logits: JTracer, targets: JTracer) -> JTracer {
        // Cross entropy = -sum(targets * log_softmax(logits))
        let logProbs = logits.logSoftmax()
        let loss = -(targets * logProbs)
        let allAxes = Set(0..<loss.shape.rank)
        return loss.sum(alongAxes: allAxes, keepDims: false)
    }

    /// Sparse Cross Entropy loss (targets are class indices, not one-hot)
    /// This is a simplified version that works with soft targets
    public static func softmaxCrossEntropyLoss(logits: JTracer, targets: JTracer) -> JTracer {
        // softmax cross entropy = -sum(targets * log_softmax(logits)) / batch_size
        let logProbs = logits.logSoftmax()
        let loss = -(targets * logProbs)

        // Sum over classes (last axis), then mean over batch
        if loss.shape.rank > 1 {
            let classAxis = Set([loss.shape.rank - 1])
            let perSample = loss.sum(alongAxes: classAxis, keepDims: false)
            let batchAxes = Set(0..<perSample.shape.rank)
            return perSample.mean(alongAxes: batchAxes, keepDims: false)
        } else {
            let allAxes = Set(0..<loss.shape.rank)
            return loss.mean(alongAxes: allAxes, keepDims: false)
        }
    }

    /// Huber loss (smooth L1): quadratic for small errors, linear for large
    public static func huberLoss(predictions: JTracer, targets: JTracer, delta: Double = 1.0) -> JTracer {
        let diff = predictions - targets
        let absDiff = diff.abs()
        let deltaT = JTracer(value: delta, shape: .scalar, dtype: predictions.dtype)
        let half = JTracer(value: 0.5, shape: .scalar, dtype: predictions.dtype)

        // quadratic part: 0.5 * diff^2
        let quadratic = half * diff * diff

        // linear part: delta * (|diff| - 0.5 * delta)
        let linear = deltaT * (absDiff - half * deltaT)

        // Use select to choose between quadratic (small) and linear (large)
        let isSmall = absDiff <= deltaT
        let loss = JTracer.select(condition: isSmall, onTrue: quadratic, onFalse: linear)

        let allAxes = Set(0..<loss.shape.rank)
        return loss.mean(alongAxes: allAxes, keepDims: false)
    }
}
