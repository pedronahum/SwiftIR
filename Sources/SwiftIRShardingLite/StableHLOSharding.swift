/// StableHLOSharding for SwiftIRShardingLite
/// Pure Swift implementation - no C dependencies.
///
/// Helpers for creating sharded StableHLO operations.

import Foundation

// MARK: - StableHLO Operation Sharding

/// Helpers for creating sharded StableHLO operations.
public enum StableHLOSharding {

    // MARK: - Tensor Operations

    /// Creates a sharded constant operation.
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

    // MARK: - Activation Functions

    /// Creates a sharded ReLU operation.
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
    public static func tanh(
        result resultName: String,
        input: String,
        type: String,
        sharding: TensorSharding? = nil
    ) -> String {
        let shardingAttr = sharding.map { " {sdy.sharding = \($0.mlirAttributeText)}" } ?? ""
        return "\(resultName) = stablehlo.tanh \(input) : \(type)\(shardingAttr)"
    }

    // MARK: - Sharding Operations

    /// Creates a sharding_constraint operation.
    public static func shardingConstraint(
        result resultName: String,
        input: String,
        type: String,
        sharding: TensorSharding
    ) -> String {
        return "\(resultName) = sdy.sharding_constraint \(input) \(sharding.mlirAttributeText) : \(type)"
    }

    /// Creates a reshard operation.
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
