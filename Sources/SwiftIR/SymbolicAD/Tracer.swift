/// Tracer - The core symbolic tensor type for automatic differentiation
/// Part of SwiftIR Symbolic Pullback Tracing system
///
/// CRITICAL DESIGN DECISIONS:
/// 1. Tracer MUST be a struct (value type) to maintain SSA correctness
/// 2. Never wrap irValue in a class or use reference semantics
/// 3. Mutations create NEW SSA values - old copies retain original values

import Foundation
import _Differentiation

/// Debug information for tracing operations back to source code
public struct DebugInfo: Hashable {
    public let file: String
    public let line: Int
    public let column: Int
    public let operationName: String?

    public init(
        file: String = #file, line: Int = #line, column: Int = #column,
        operationName: String? = nil
    ) {
        self.file = file
        self.line = line
        self.column = column
        self.operationName = operationName
    }
}

/// The core symbolic tensor type that traces operations into MLIR
///
/// Tracer looks like a number to Swift's type system but actually emits
/// MLIR operations when manipulated. This is the "Trojan Horse" mechanism
/// that allows Swift's native `@differentiable` to build gradient graphs.
///
/// Example:
/// ```swift
/// @differentiable(reverse)
/// func model(_ x: Tracer) -> Tracer {
///     return x * 2.0 + 1.0  // Emits MLIR multiply and add operations
/// }
///
/// let grad = gradient(at: x, in: model)  // Builds forward + backward graph
/// ```
public struct Tracer: Differentiable {
    // MARK: - Core Properties

    /// The underlying MLIR SSA value (immutable)
    /// This is private to prevent misuse
    private let irValue: MLIRValue

    /// Shape of the tensor
    public let shape: TensorShape

    /// Data type of the tensor
    public let dtype: DType

    /// Version counter for detecting stale references in debug builds
    private let version: UInt64

    /// Optional debug information for error messages
    public let debugInfo: DebugInfo?

    // MARK: - Static Version Counter

    /// Thread-safe global version counter
    private nonisolated(unsafe) static var _globalVersion: UInt64 = 0
    private static let versionLock = NSLock()

    internal static func incrementVersion() -> UInt64 {
        versionLock.lock()
        defer { versionLock.unlock() }
        _globalVersion += 1
        return _globalVersion
    }

    // MARK: - Initializers

    /// Internal initializer with all fields
    internal init(
        irValue: MLIRValue, shape: TensorShape, dtype: DType, version: UInt64,
        debugInfo: DebugInfo? = nil
    ) {
        self.irValue = irValue
        self.shape = shape
        self.dtype = dtype
        self.version = version
        self.debugInfo = debugInfo
    }

    /// Create a tracer from a scalar value (for constants)
    public init(
        value: Double, shape: TensorShape = .scalar, dtype: DType = .float32,
        file: String = #file, line: Int = #line
    ) {
        // Create a constant operation in the graph
        let id = TracerGraphBuilder.shared.createConstant(value: value, shape: shape, dtype: dtype)
        self.irValue = MLIRValue(id: id)
        self.shape = shape
        self.dtype = dtype
        self.version = Self.incrementVersion()
        self.debugInfo = DebugInfo(file: file, line: line, operationName: "constant")
    }

    /// Create a tracer with a specific shape and type
    public init(
        shape: TensorShape, dtype: DType = .float32, file: String = #file, line: Int = #line
    ) {
        let id = TracerGraphBuilder.shared.createPlaceholder(shape: shape, dtype: dtype)
        self.irValue = MLIRValue(id: id)
        self.shape = shape
        self.dtype = dtype
        self.version = Self.incrementVersion()
        self.debugInfo = DebugInfo(file: file, line: line, operationName: "placeholder")
    }

    // MARK: - Safe Value Accessor

    /// Get the underlying IR value with optional debug validation
    public var value: MLIRValue {
        #if DEBUG
            TracerGraphBuilder.shared.validateSSAValue(irValue, version: version)
        #endif
        return irValue
    }

    /// Get the IR value ID (for debugging)
    public var valueId: UInt64 {
        return irValue.id
    }

    // MARK: - Differentiable Conformance

    public typealias TangentVector = Tracer

    public mutating func move(by offset: TangentVector) {
        // CRITICAL: Create NEW SSA value for the addition
        let newId = TracerGraphBuilder.shared.createBinaryOp(
            operation: .add,
            lhs: self.irValue,
            rhs: offset.irValue,
            shape: self.shape,
            dtype: self.dtype
        )

        // REPLACE self with entirely new struct
        // This ensures value semantics: old copies aren't affected
        self = Tracer(
            irValue: MLIRValue(id: newId),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "move(by:)")
        )
    }
}

// MARK: - Compile-time Safety Enforcement

extension Tracer {
    /// FORBIDDEN: This will cause a compile error if attempted
    /// Tracer must never use reference semantics
    @available(*, unavailable, message: "Tracer must not use reference semantics. Never wrap in a class.")
    public init(referencing _: AnyObject) {
        fatalError("Unreachable")
    }
}

// MARK: - AdditiveArithmetic Conformance

extension Tracer: AdditiveArithmetic {
    public static var zero: Tracer {
        return Tracer(value: 0.0, shape: .scalar, dtype: .float32)
    }

    public static func + (lhs: Tracer, rhs: Tracer) -> Tracer {
        // Compute broadcast shape
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = TracerGraphBuilder.shared.createBinaryOp(
            operation: .add,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "add")
        )
    }

    public static func - (lhs: Tracer, rhs: Tracer) -> Tracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = TracerGraphBuilder.shared.createBinaryOp(
            operation: .subtract,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "subtract")
        )
    }
}

// MARK: - Numeric Operations

extension Tracer {
    public static func * (lhs: Tracer, rhs: Tracer) -> Tracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = TracerGraphBuilder.shared.createBinaryOp(
            operation: .multiply,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "multiply")
        )
    }

    public static func / (lhs: Tracer, rhs: Tracer) -> Tracer {
        let resultShape = lhs.shape.broadcast(with: rhs.shape)
        let resultDtype = lhs.dtype.promoted(with: rhs.dtype)

        let id = TracerGraphBuilder.shared.createBinaryOp(
            operation: .divide,
            lhs: lhs.irValue,
            rhs: rhs.irValue,
            shape: resultShape,
            dtype: resultDtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: resultDtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "divide")
        )
    }

    /// Scalar operations
    public static func * (lhs: Tracer, rhs: Double) -> Tracer {
        let scalar = Tracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs * scalar
    }

    public static func * (lhs: Double, rhs: Tracer) -> Tracer {
        let scalar = Tracer(value: lhs, shape: .scalar, dtype: rhs.dtype)
        return scalar * rhs
    }

    public static func + (lhs: Tracer, rhs: Double) -> Tracer {
        let scalar = Tracer(value: rhs, shape: .scalar, dtype: lhs.dtype)
        return lhs + scalar
    }

    public static func + (lhs: Double, rhs: Tracer) -> Tracer {
        let scalar = Tracer(value: lhs, shape: .scalar, dtype: rhs.dtype)
        return scalar + rhs
    }
}

// MARK: - ExpressibleByFloatLiteral

extension Tracer: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(value: value, shape: .scalar, dtype: .float32)
    }
}

// MARK: - Reduction Operations

extension Tracer {
    /// Sum all elements to a scalar
    public func sum() -> Tracer {
        let allAxes = Set(0..<shape.rank)
        return sum(alongAxes: allAxes, keepDims: false)
    }

    /// Sum along specified axes
    public func sum(alongAxes axes: Set<Int>, keepDims: Bool = false) -> Tracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = TracerGraphBuilder.shared.createReduction(
            operation: .sum,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "sum")
        )
    }

    /// Mean along specified axes
    public func mean(alongAxes axes: Set<Int>, keepDims: Bool = false) -> Tracer {
        let resultShape = shape.reduced(alongAxes: axes, keepDims: keepDims)

        let id = TracerGraphBuilder.shared.createReduction(
            operation: .mean,
            input: self.irValue,
            axes: axes,
            keepDims: keepDims,
            resultShape: resultShape,
            dtype: self.dtype
        )

        return Tracer(
            irValue: MLIRValue(id: id),
            shape: resultShape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "mean")
        )
    }
}

// MARK: - Factory Methods

extension Tracer {
    /// Create a tensor of zeros
    public static func zeros(shape: TensorShape, dtype: DType = .float32) -> Tracer {
        let id = TracerGraphBuilder.shared.createConstant(value: 0.0, shape: shape, dtype: dtype)
        return Tracer(
            irValue: MLIRValue(id: id),
            shape: shape,
            dtype: dtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "zeros")
        )
    }

    /// Create a tensor of ones
    public static func ones(shape: TensorShape, dtype: DType = .float32) -> Tracer {
        let id = TracerGraphBuilder.shared.createConstant(value: 1.0, shape: shape, dtype: dtype)
        return Tracer(
            irValue: MLIRValue(id: id),
            shape: shape,
            dtype: dtype,
            version: incrementVersion(),
            debugInfo: DebugInfo(operationName: "ones")
        )
    }
}

// MARK: - Print/Debug Operations with Token Chaining

extension Tracer {
    /// Print operation that respects execution order
    public func print(label: String, after token: Token = .global) -> (printed: Tracer, token: Token) {
        let (resultId, nextToken) = TracerGraphBuilder.shared.createPrint(
            input: self.irValue,
            label: label,
            shape: self.shape,
            dtype: self.dtype,
            token: token
        )

        let printedTracer = Tracer(
            irValue: MLIRValue(id: resultId),
            shape: self.shape,
            dtype: self.dtype,
            version: Self.incrementVersion(),
            debugInfo: DebugInfo(operationName: "print(\(label))")
        )

        return (printedTracer, nextToken)
    }
}

// MARK: - Graph Builder

/// Internal graph builder for tracing operations
/// This collects operations as they are executed
internal final class TracerGraphBuilder: @unchecked Sendable {
    nonisolated(unsafe) static let shared = TracerGraphBuilder()

    private var nextId: UInt64 = 1
    private let lock = NSLock()

    /// Operations recorded during tracing
    private var operations: [TracedOperation] = []

    /// Map of superseded values for stale reference detection
    private var supersessionMap: [UInt64: UInt64] = [:]

    private init() {}

    /// Create a constant value
    func createConstant(value: Double, shape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(.constant(id: id, value: value, shape: shape, dtype: dtype))
        return id
    }

    /// Create a placeholder for function input
    func createPlaceholder(shape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(.placeholder(id: id, shape: shape, dtype: dtype))
        return id
    }

    /// Create a binary operation
    func createBinaryOp(
        operation: BinaryOperation, lhs: MLIRValue, rhs: MLIRValue, shape: TensorShape, dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(
            .binary(id: id, op: operation, lhs: lhs.id, rhs: rhs.id, shape: shape, dtype: dtype))
        return id
    }

    /// Create a reduction operation
    func createReduction(
        operation: ReductionOperation, input: MLIRValue, axes: Set<Int>, keepDims: Bool,
        resultShape: TensorShape, dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(
            .reduction(
                id: id, op: operation, input: input.id, axes: axes, keepDims: keepDims,
                shape: resultShape, dtype: dtype))
        return id
    }

    /// Create a print operation with token
    func createPrint(
        input: MLIRValue, label: String, shape: TensorShape, dtype: DType, token: Token
    ) -> (UInt64, Token) {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        let nextToken = Token.create()

        operations.append(
            .print(
                id: id, input: input.id, label: label, shape: shape, dtype: dtype,
                tokenIn: token.id, tokenOut: nextToken.id))
        return (id, nextToken)
    }

    /// Validate SSA value is not stale (debug only)
    func validateSSAValue(_ value: MLIRValue, version: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        if let newerVersion = supersessionMap[value.id] {
            if newerVersion > version {
                Swift.print(
                    """
                    ⚠️ WARNING: Using stale SSA value!
                    This Tracer was captured before a mutation.
                    SSA Value ID: \(value.id)
                    Your version: \(version)
                    Current version: \(newerVersion)
                    """
                )
            }
        }
    }

    /// Reset the graph builder (for testing)
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 1
        operations.removeAll()
        supersessionMap.removeAll()
    }

    /// Get all recorded operations
    func getOperations() -> [TracedOperation] {
        lock.lock()
        defer { lock.unlock() }
        return operations
    }

    /// Create a unary operation
    func createUnaryOp(
        operation: UnaryOperation, input: MLIRValue, shape: TensorShape, dtype: DType
    ) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(.unary(id: id, op: operation, input: input.id, shape: shape, dtype: dtype))
        return id
    }

    /// Create a reshape operation
    func createReshape(input: MLIRValue, newShape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(
            .unary(id: id, op: .reshape, input: input.id, shape: newShape, dtype: dtype))
        return id
    }

    /// Create a power operation
    func createPower(base: MLIRValue, exponent: Double, shape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(.power(id: id, base: base.id, exponent: exponent, shape: shape, dtype: dtype))
        return id
    }

    /// Create a matrix multiplication operation
    func createMatMul(lhs: MLIRValue, rhs: MLIRValue, shape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(
            .binary(id: id, op: .matmul, lhs: lhs.id, rhs: rhs.id, shape: shape, dtype: dtype))
        return id
    }

    /// Create a transpose operation
    func createTranspose(input: MLIRValue, shape: TensorShape, dtype: DType) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1

        operations.append(.unary(id: id, op: .transpose, input: input.id, shape: shape, dtype: dtype))
        return id
    }
}

// MARK: - Operation Types

/// Types of operations that can be traced
internal enum TracedOperation {
    case constant(id: UInt64, value: Double, shape: TensorShape, dtype: DType)
    case placeholder(id: UInt64, shape: TensorShape, dtype: DType)
    case binary(
        id: UInt64, op: BinaryOperation, lhs: UInt64, rhs: UInt64, shape: TensorShape, dtype: DType)
    case unary(id: UInt64, op: UnaryOperation, input: UInt64, shape: TensorShape, dtype: DType)
    case reduction(
        id: UInt64, op: ReductionOperation, input: UInt64, axes: Set<Int>, keepDims: Bool,
        shape: TensorShape, dtype: DType)
    case print(
        id: UInt64, input: UInt64, label: String, shape: TensorShape, dtype: DType, tokenIn: UInt64,
        tokenOut: UInt64)
    case power(id: UInt64, base: UInt64, exponent: Double, shape: TensorShape, dtype: DType)
}

/// Binary operations
internal enum BinaryOperation: String {
    case add
    case subtract
    case multiply
    case divide
    case matmul
}

/// Unary operations
internal enum UnaryOperation: String {
    case exp
    case log
    case sqrt
    case abs
    case neg
    case sin
    case cos
    case tan
    case tanh
    case sigmoid
    case relu
    case reshape
    case transpose
}

/// Reduction operations
internal enum ReductionOperation: String {
    case sum
    case mean
    case max
    case min
}
