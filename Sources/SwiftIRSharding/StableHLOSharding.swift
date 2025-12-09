/// StableHLO Sharding Integration
/// Helpers for adding sharding annotations to StableHLO operations.

import SdyCAPIWrapper
import MLIRCoreWrapper

// MARK: - StableHLO Operation Sharding

/// Helpers for creating sharded StableHLO operations.
///
/// These functions generate MLIR text with sharding annotations that can be
/// parsed to create sharded StableHLO programs.
public enum StableHLOSharding {

    // MARK: - Tensor Operations

    /// Creates a sharded constant operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name (e.g., "%0")
    ///   - value: The constant value as dense attribute (e.g., "dense<1.0>")
    ///   - type: The tensor type (e.g., "tensor<8x8xf32>")
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the constant operation
    public static func constant(
        result resultName: String,
        value: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.constant \(value) : \(type)\(shardingAttr)"
    }

    /// Creates a sharded add operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - lhs: Left operand SSA value
    ///   - rhs: Right operand SSA value
    ///   - type: The tensor type
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the add operation
    public static func add(
        result resultName: String,
        lhs: String,
        rhs: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.add \(lhs), \(rhs) : \(type)\(shardingAttr)"
    }

    /// Creates a sharded multiply operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - lhs: Left operand SSA value
    ///   - rhs: Right operand SSA value
    ///   - type: The tensor type
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the multiply operation
    public static func multiply(
        result resultName: String,
        lhs: String,
        rhs: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.multiply \(lhs), \(rhs) : \(type)\(shardingAttr)"
    }

    /// Creates a sharded dot_general (matrix multiply) operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - lhs: Left operand SSA value
    ///   - rhs: Right operand SSA value
    ///   - lhsType: Left operand type
    ///   - rhsType: Right operand type
    ///   - resultType: Result type
    ///   - contractingDims: Contracting dimensions (lhs, rhs)
    ///   - batchingDims: Batching dimensions (lhs, rhs)
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the dot_general operation
    public static func dotGeneral(
        result resultName: String,
        lhs: String,
        rhs: String,
        lhsType: String,
        rhsType: String,
        resultType: String,
        contractingDims: (lhs: [Int], rhs: [Int]),
        batchingDims: (lhs: [Int], rhs: [Int]) = ([], []),
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""

        let lhsContracting = contractingDims.lhs.map(String.init).joined(separator: ", ")
        let rhsContracting = contractingDims.rhs.map(String.init).joined(separator: ", ")
        let lhsBatching = batchingDims.lhs.map(String.init).joined(separator: ", ")
        let rhsBatching = batchingDims.rhs.map(String.init).joined(separator: ", ")

        let dimNumbers = "dot_dimension_numbers = <batching_dims = [[\(lhsBatching)], [\(rhsBatching)]], contracting_dims = [[\(lhsContracting)], [\(rhsContracting)]]>"

        return "\(resultName) = stablehlo.dot_general \(lhs), \(rhs), \(dimNumbers) : (\(lhsType), \(rhsType)) -> \(resultType)\(shardingAttr)"
    }

    /// Creates a sharded convolution operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input tensor SSA value
    ///   - kernel: Kernel tensor SSA value
    ///   - inputType: Input tensor type
    ///   - kernelType: Kernel tensor type
    ///   - resultType: Result type
    ///   - strides: Convolution strides
    ///   - padding: Convolution padding
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the convolution operation
    public static func convolution(
        result resultName: String,
        input: String,
        kernel: String,
        inputType: String,
        kernelType: String,
        resultType: String,
        strides: [Int] = [1, 1],
        padding: [[Int]] = [[0, 0], [0, 0]],
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""

        let stridesStr = strides.map(String.init).joined(separator: ", ")
        let paddingStr = padding.map { "[\($0[0]), \($0[1])]" }.joined(separator: ", ")

        // Using NHWC format (default for StableHLO)
        let dimNumbers = "dimension_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]"

        return "\(resultName) = stablehlo.convolution(\(input), \(kernel)) \(dimNumbers), window = {stride = [\(stridesStr)], pad = [[\(paddingStr)]]} : (\(inputType), \(kernelType)) -> \(resultType)\(shardingAttr)"
    }

    // MARK: - Activation Functions

    /// Creates a sharded ReLU operation (maximum with zero).
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input SSA value
    ///   - zero: Zero constant SSA value
    ///   - type: The tensor type
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the ReLU operation
    public static func relu(
        result resultName: String,
        input: String,
        zero: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.maximum \(input), \(zero) : \(type)\(shardingAttr)"
    }

    /// Creates a sharded tanh operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input SSA value
    ///   - type: The tensor type
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the tanh operation
    public static func tanh(
        result resultName: String,
        input: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.tanh \(input) : \(type)\(shardingAttr)"
    }

    // MARK: - Reduction Operations

    /// Creates a sharded reduce operation.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input SSA value
    ///   - init: Initial value SSA value
    ///   - inputType: Input tensor type
    ///   - initType: Initial value type (scalar)
    ///   - resultType: Result type
    ///   - dimensions: Dimensions to reduce
    ///   - body: The reduction body (e.g., add for sum)
    ///   - sharding: Optional sharding annotation
    /// - Returns: The MLIR text for the reduce operation
    public static func reduce(
        result resultName: String,
        input: String,
        `init`: String,
        inputType: String,
        initType: String,
        resultType: String,
        dimensions: [Int],
        body: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        let dimsStr = dimensions.map(String.init).joined(separator: ", ")

        return """
            \(resultName) = stablehlo.reduce(\(input) init: \(`init`)) across dimensions = [\(dimsStr)] : (\(inputType), \(initType)) -> \(resultType)
              reducer(%arg0: \(initType), %arg1: \(initType)) {
                \(body)
              }\(shardingAttr)
            """
    }

    // MARK: - Sharding Constraint

    /// Creates a sharding_constraint operation.
    ///
    /// This explicitly constrains the sharding of a tensor.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input SSA value
    ///   - type: The tensor type
    ///   - sharding: The sharding constraint
    /// - Returns: The MLIR text for the sharding constraint
    public static func shardingConstraint(
        result resultName: String,
        input: String,
        type: String,
        sharding: TensorSharding
    ) -> String {
        return "\(resultName) = sdy.sharding_constraint \(input) \(sharding.mlirAttributeText) : \(type)"
    }

    /// Creates a reshard operation.
    ///
    /// Explicitly reshards a tensor to a new sharding.
    ///
    /// - Parameters:
    ///   - resultName: The SSA value name
    ///   - input: Input SSA value
    ///   - type: The tensor type
    ///   - sharding: The target sharding
    /// - Returns: The MLIR text for the reshard operation
    public static func reshard(
        result resultName: String,
        input: String,
        type: String,
        sharding: TensorSharding
    ) -> String {
        return "\(resultName) = sdy.reshard \(input) \(sharding.mlirAttributeText) : \(type)"
    }
}

// MARK: - Sharded Module Builder

/// A builder for creating complete sharded StableHLO modules.
public class ShardedModuleBuilder {
    private var operations: [String] = []
    private var ssaCounter = 0
    private let pipeline: ShardingPipeline

    /// Creates a new sharded module builder.
    ///
    /// - Parameter pipeline: The sharding pipeline with mesh definitions
    public init(pipeline: ShardingPipeline) {
        self.pipeline = pipeline
    }

    /// Generates the next SSA value name.
    public func nextSSA() -> String {
        let name = "%\(ssaCounter)"
        ssaCounter += 1
        return name
    }

    /// Adds an operation to the module.
    public func add(_ operation: String) {
        operations.append(operation)
    }

    /// Gets the current operations as a body string.
    public func body() -> String {
        operations.joined(separator: "\n")
    }

    /// Resets the builder for a new module.
    public func reset() {
        operations = []
        ssaCounter = 0
    }

    /// Creates a sharded constant.
    @discardableResult
    public func constant(
        value: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let result = nextSSA()
        add(StableHLOSharding.constant(result: result, value: value, type: type, sharding: sharding))
        return result
    }

    /// Creates a sharded add.
    @discardableResult
    public func add(
        lhs: String,
        rhs: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let result = nextSSA()
        add(StableHLOSharding.add(result: result, lhs: lhs, rhs: rhs, type: type, sharding: sharding))
        return result
    }

    /// Creates a sharded multiply.
    @discardableResult
    public func multiply(
        lhs: String,
        rhs: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let result = nextSSA()
        add(StableHLOSharding.multiply(result: result, lhs: lhs, rhs: rhs, type: type, sharding: sharding))
        return result
    }

    /// Creates a sharded matrix multiply.
    @discardableResult
    public func matmul(
        lhs: String,
        rhs: String,
        lhsType: String,
        rhsType: String,
        resultType: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let result = nextSSA()
        add(StableHLOSharding.dotGeneral(
            result: result,
            lhs: lhs,
            rhs: rhs,
            lhsType: lhsType,
            rhsType: rhsType,
            resultType: resultType,
            contractingDims: (lhs: [1], rhs: [0]),
            sharding: sharding
        ))
        return result
    }

    /// Creates a sharding constraint.
    @discardableResult
    public func shardingConstraint(
        input: String,
        type: String,
        sharding: TensorSharding
    ) -> String {
        let result = nextSSA()
        add(StableHLOSharding.shardingConstraint(
            result: result,
            input: input,
            type: type,
            sharding: sharding
        ))
        return result
    }
}
