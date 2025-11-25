/// TensorShape - Type-safe tensor shape representation with broadcasting support
/// Part of SwiftIR Symbolic Pullback Tracing system

/// Compile-time shape representation for tensors
/// Supports both static (known at compile time) and dynamic (nil) dimensions
public struct TensorShape: Hashable, Equatable, CustomStringConvertible, Sendable {
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
    /// - Parameter dimensions: Array of dimension sizes (nil for dynamic)
    public init(dimensions: [Int?]) {
        self.dimensions = dimensions
    }

    /// Create a shape from known dimensions
    /// - Parameter dimensions: Array of known dimension sizes
    public init(_ dimensions: [Int]) {
        self.dimensions = dimensions.map { $0 as Int? }
    }

    /// Create a scalar (0-dimensional) shape
    public static let scalar = TensorShape(dimensions: [])

    // MARK: - Broadcasting

    /// Check if this shape is broadcast-compatible with another shape
    /// Uses NumPy-style broadcasting rules
    public func isBroadcastCompatible(with other: TensorShape) -> Bool {
        let maxRank = max(self.rank, other.rank)

        // Pad shapes with leading 1s
        let selfPadded = Array(repeating: 1 as Int?, count: maxRank - rank) + dimensions
        let otherPadded = Array(repeating: 1 as Int?, count: maxRank - other.rank) + other.dimensions

        // Check each dimension
        for (d1, d2) in zip(selfPadded, otherPadded) {
            // Both dynamic - always compatible
            if d1 == nil || d2 == nil { continue }

            // Same size - compatible
            if d1 == d2 { continue }

            // One is 1 - compatible (will be broadcast)
            if d1 == 1 || d2 == 1 { continue }

            // Different non-1 sizes - incompatible
            return false
        }

        return true
    }

    /// Compute the result shape after broadcasting with another shape
    /// - Parameter other: The other shape to broadcast with
    /// - Returns: The resulting broadcast shape
    public func broadcast(with other: TensorShape) -> TensorShape {
        let maxRank = max(self.rank, other.rank)
        var resultDims: [Int?] = []

        // Pad shapes with leading 1s
        let selfPadded = Array(repeating: 1 as Int?, count: maxRank - rank) + dimensions
        let otherPadded = Array(repeating: 1 as Int?, count: maxRank - other.rank) + other.dimensions

        for (d1, d2) in zip(selfPadded, otherPadded) {
            if let dim1 = d1, let dim2 = d2 {
                // Both known - take the larger (broadcast semantics)
                resultDims.append(max(dim1, dim2))
            } else if let dim1 = d1 {
                // Only first known
                resultDims.append(dim1 == 1 ? d2 : dim1)
            } else if let dim2 = d2 {
                // Only second known
                resultDims.append(dim2 == 1 ? d1 : dim2)
            } else {
                // Both dynamic
                resultDims.append(nil)
            }
        }

        return TensorShape(dimensions: resultDims)
    }

    /// Validate that this shape is valid for an operation
    /// - Parameter operation: Name of the operation (for error messages)
    /// - Throws: ShapeError if validation fails
    public func validate(for operation: String = "operation") throws {
        // Check for negative dimensions
        for (index, dim) in dimensions.enumerated() {
            if let d = dim, d < 0 {
                throw ShapeError.negativeDimension(dimension: index, size: d, operation: operation)
            }
        }
    }

    /// Get the shape after reducing along specified axes
    /// - Parameters:
    ///   - axes: Set of axes to reduce
    ///   - keepDims: If true, reduced dimensions become 1; if false, they are removed
    /// - Returns: The resulting shape after reduction
    public func reduced(alongAxes axes: Set<Int>, keepDims: Bool = false) -> TensorShape {
        var resultDims: [Int?] = []

        for (index, dim) in dimensions.enumerated() {
            if axes.contains(index) {
                if keepDims {
                    resultDims.append(1)
                }
                // If not keepDims, skip this dimension
            } else {
                resultDims.append(dim)
            }
        }

        return TensorShape(dimensions: resultDims)
    }

    /// Check if two shapes are equal (considering dynamic dimensions)
    public static func == (lhs: TensorShape, rhs: TensorShape) -> Bool {
        guard lhs.rank == rhs.rank else { return false }

        for (d1, d2) in zip(lhs.dimensions, rhs.dimensions) {
            // If both are known, they must be equal
            if let dim1 = d1, let dim2 = d2 {
                if dim1 != dim2 { return false }
            }
            // If either is dynamic, we consider them potentially equal
        }

        return true
    }
}

// MARK: - Shape Errors

/// Errors related to tensor shape operations
public enum ShapeError: Error, CustomStringConvertible {
    case shapeMismatch(expected: TensorShape, got: TensorShape, operation: String)
    case incompatibleBroadcast(lhs: TensorShape, rhs: TensorShape, operation: String)
    case invalidAxis(axis: Int, rank: Int, operation: String)
    case negativeDimension(dimension: Int, size: Int, operation: String)
    case dynamicShapeInStaticContext(shape: TensorShape, operation: String)

    public var description: String {
        switch self {
        case .shapeMismatch(let expected, let got, let op):
            return """
            Shape mismatch in \(op):
              Expected: \(expected)
              Got: \(got)

            Hint: Use `reshape` or `broadcast` to make shapes compatible.
            """

        case .incompatibleBroadcast(let lhs, let rhs, let op):
            return """
            Incompatible shapes for broadcasting in \(op):
              LHS shape: \(lhs)
              RHS shape: \(rhs)

            Hint: Shapes must be broadcast-compatible (dimensions must be equal or one must be 1).
            """

        case .invalidAxis(let axis, let rank, let op):
            return """
            Invalid axis \(axis) for \(op) on tensor with rank \(rank).
            Valid axes are 0..<\(rank).
            """

        case .negativeDimension(let dimension, let size, let op):
            return """
            Negative dimension size \(size) at dimension \(dimension) in \(op).
            Dimension sizes must be non-negative.
            """

        case .dynamicShapeInStaticContext(let shape, let op):
            return """
            Dynamic shape \(shape) used in \(op) which requires static shapes.

            Hint: Use runtime shape assertion or switch to dynamic API.
            """
        }
    }
}

// MARK: - Convenience Extensions

public extension TensorShape {
    /// Create a shape for a vector
    static func vector(_ size: Int) -> TensorShape {
        TensorShape([size])
    }

    /// Create a shape for a matrix
    static func matrix(_ rows: Int, _ cols: Int) -> TensorShape {
        TensorShape([rows, cols])
    }

    /// Create a shape for a 3D tensor
    static func tensor3D(_ d0: Int, _ d1: Int, _ d2: Int) -> TensorShape {
        TensorShape([d0, d1, d2])
    }

    /// Create a shape for a 4D tensor (common for images: NHWC or NCHW)
    static func tensor4D(_ d0: Int, _ d1: Int, _ d2: Int, _ d3: Int) -> TensorShape {
        TensorShape([d0, d1, d2, d3])
    }
}
