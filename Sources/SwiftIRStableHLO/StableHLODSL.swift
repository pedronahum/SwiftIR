// StableHLODSL.swift - Swift result builder DSL for StableHLO
// Copyright 2024 SwiftIR Project
//
// This file provides a SwiftUI-like declarative DSL for building StableHLO programs

import Foundation
import SwiftIRCore
import SwiftIRTypes
import MLIRCoreWrapper

// MARK: - Result Builders

/// Result builder for constructing StableHLO modules declaratively
@resultBuilder
public struct StableHLOModuleBuilder {
    public static func buildBlock(_ components: MLIROperation...) -> [MLIROperation] {
        components
    }

    public static func buildBlock(_ components: [MLIROperation]...) -> [MLIROperation] {
        components.flatMap { $0 }
    }
}

/// Result builder for constructing function bodies with sequential operations
@resultBuilder
public struct StableHLOFunctionBuilder {
    public static func buildBlock(_ components: StableHLOStatement...) -> [StableHLOStatement] {
        components
    }
}

// MARK: - Sequential Composition

/// Represents a statement in a StableHLO function body
/// Can be either a Let binding (named intermediate result) or a Return operation
public enum StableHLOStatement {
    case let_(name: String, operation: StableHLOOperation)
    case return_(operation: StableHLOOperation)
}

/// Creates a let binding for an intermediate result
/// Usage: Let("hidden1") { DotGeneral(...) }
public func Let(_ name: String, _ operation: StableHLOOperation) -> StableHLOStatement {
    .let_(name: name, operation: operation)
}

/// Creates a return statement
/// Usage: Return { Add(...) }
public func Return(_ operation: StableHLOOperation) -> StableHLOStatement {
    .return_(operation: operation)
}

// MARK: - DSL Types

/// Represents a tensor type with shape and element type
public struct TensorType {
    public let shape: [Int]
    public let elementType: String

    public init(shape: [Int], elementType: String = "f32") {
        self.shape = shape
        self.elementType = elementType
    }

    /// Creates the MLIR type string representation
    public var mlirType: String {
        let shapeStr = shape.map(String.init).joined(separator: "x")
        return "tensor<\(shapeStr)x\(elementType)>"
    }
}

/// Represents a function parameter
public struct Parameter {
    public let name: String
    public let type: TensorType

    public init(name: String, type: TensorType) {
        self.name = name
        self.type = type
    }
}

// MARK: - StableHLO Module DSL

/// Main entry point for building StableHLO modules
public struct StableHLOModule {
    public let name: String
    private let functions: [StableHLOFunction]

    public init(name: String, @StableHLOModuleBuilder _ content: () -> [StableHLOFunction]) {
        self.name = name
        self.functions = content()
    }

    /// Builds the MLIR string representation
    public func build() -> String {
        var result = "module @\(name) {\n"
        for function in functions {
            result += function.buildMLIR().split(separator: "\n").map { "  " + $0 }.joined(separator: "\n")
            result += "\n"
        }
        result += "}\n"
        return result
    }
}

/// Represents a StableHLO function
public struct StableHLOFunction {
    public let name: String
    public let parameters: [Parameter]
    public let returnType: TensorType
    private let body: [StableHLOStatement]

    public init(
        name: String,
        parameters: [Parameter],
        returnType: TensorType,
        @StableHLOFunctionBuilder body: () -> [StableHLOStatement]
    ) {
        self.name = name
        self.parameters = parameters
        self.returnType = returnType
        self.body = body()
    }

    func buildMLIR() -> String {
        let paramDecls = parameters.enumerated().map { (i, p) in
            "%\(p.name): \(p.type.mlirType)"
        }.joined(separator: ",\n    ")

        var result = "func.func public @\(name)(\n    \(paramDecls)\n  ) -> \(returnType.mlirType) {\n"

        var resultIndex = 0
        var returnValue: String?

        for statement in body {
            switch statement {
            case .let_(let variableName, let operation):
                // Generate MLIR for the operation and bind it to a named variable
                let mlirCode = operation.buildMLIR(resultIndex: resultIndex)
                // Replace %resultIndex with %variableName
                let withNamedResult = mlirCode.replacingOccurrences(of: "%\(resultIndex)", with: "%\(variableName)")
                result += "    \(withNamedResult)\n"
                resultIndex += 1

            case .return_(let operation):
                // Generate MLIR for the return operation
                let mlirCode = operation.buildMLIR(resultIndex: resultIndex)
                result += "    \(mlirCode)\n"
                returnValue = "%\(resultIndex)"
                resultIndex += 1
            }
        }

        // Return statement
        if let returnValue = returnValue {
            result += "    return \(returnValue) : \(returnType.mlirType)\n"
        }

        result += "  }"
        return result
    }
}

// MARK: - StableHLO Operations

/// Base protocol for StableHLO operations
public protocol StableHLOOperation {
    func buildMLIR(resultIndex: Int) -> String
}

/// Dot general operation (matrix multiplication)
public struct DotGeneral: StableHLOOperation {
    let lhs: String
    let rhs: String
    let lhsType: TensorType
    let rhsType: TensorType
    let resultType: TensorType
    let contractingDims: (Int, Int)

    public init(
        _ lhs: String,
        _ rhs: String,
        lhsType: TensorType,
        rhsType: TensorType,
        resultType: TensorType,
        contractingDims: (Int, Int) = (1, 0)
    ) {
        self.lhs = lhs
        self.rhs = rhs
        self.lhsType = lhsType
        self.rhsType = rhsType
        self.resultType = resultType
        self.contractingDims = contractingDims
    }

    public func buildMLIR(resultIndex: Int) -> String {
        """
        %\(resultIndex) = stablehlo.dot_general %\(lhs), %\(rhs),
          contracting_dims = [\(contractingDims.0)] x [\(contractingDims.1)]
          : (\(lhsType.mlirType), \(rhsType.mlirType)) -> \(resultType.mlirType)
        """
    }
}

/// Add operation
public struct Add: StableHLOOperation {
    let lhs: String
    let rhs: String
    let type: TensorType

    public init(_ lhs: String, _ rhs: String, type: TensorType) {
        self.lhs = lhs
        self.rhs = rhs
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.add %\(lhs), %\(rhs) : \(type.mlirType)"
    }
}

/// Multiply operation
public struct Multiply: StableHLOOperation {
    let lhs: String
    let rhs: String
    let type: TensorType

    public init(_ lhs: String, _ rhs: String, type: TensorType) {
        self.lhs = lhs
        self.rhs = rhs
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.multiply %\(lhs), %\(rhs) : \(type.mlirType)"
    }
}

/// Subtract operation
public struct Subtract: StableHLOOperation {
    let lhs: String
    let rhs: String
    let type: TensorType

    public init(_ lhs: String, _ rhs: String, type: TensorType) {
        self.lhs = lhs
        self.rhs = rhs
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.subtract %\(lhs), %\(rhs) : \(type.mlirType)"
    }
}

// MARK: - Activation Functions

/// Maximum operation (element-wise max of tensor and scalar)
/// Used to implement ReLU: max(x, 0)
public struct Maximum: StableHLOOperation {
    let input: String
    let scalar: String
    let type: TensorType

    public init(_ input: String, _ scalar: String, type: TensorType) {
        self.input = input
        self.scalar = scalar
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.maximum %\(input), %\(scalar) : \(type.mlirType)"
    }
}

/// Exponential operation (e^x)
public struct Exponential: StableHLOOperation {
    let operand: String
    let type: TensorType

    public init(_ operand: String, type: TensorType) {
        self.operand = operand
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.exponential %\(operand) : \(type.mlirType)"
    }
}

/// Logarithm operation (ln(x))
public struct Logarithm: StableHLOOperation {
    let operand: String
    let type: TensorType

    public init(_ operand: String, type: TensorType) {
        self.operand = operand
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.log %\(operand) : \(type.mlirType)"
    }
}

/// Tanh operation (hyperbolic tangent)
public struct Tanh: StableHLOOperation {
    let operand: String
    let type: TensorType

    public init(_ operand: String, type: TensorType) {
        self.operand = operand
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.tanh %\(operand) : \(type.mlirType)"
    }
}

/// Logistic/Sigmoid operation (1 / (1 + e^(-x)))
public struct Logistic: StableHLOOperation {
    let operand: String
    let type: TensorType

    public init(_ operand: String, type: TensorType) {
        self.operand = operand
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.logistic %\(operand) : \(type.mlirType)"
    }
}

/// Negate operation (unary minus: -x)
public struct Negate: StableHLOOperation {
    let operand: String
    let type: TensorType

    public init(_ operand: String, type: TensorType) {
        self.operand = operand
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        "%\(resultIndex) = stablehlo.negate %\(operand) : \(type.mlirType)"
    }
}

/// Constant operation (creates a constant tensor)
public struct Constant: StableHLOOperation {
    let value: Float
    let type: TensorType

    public init(value: Float, type: TensorType) {
        self.value = value
        self.type = type
    }

    public func buildMLIR(resultIndex: Int) -> String {
        // For scalar constants, use dense attribute
        if type.shape.isEmpty || (type.shape.count == 1 && type.shape[0] == 1) {
            return "%\(resultIndex) = stablehlo.constant dense<\(value)> : \(type.mlirType)"
        }
        // For tensors, broadcast the scalar value
        return "%\(resultIndex) = stablehlo.constant dense<\(value)> : \(type.mlirType)"
    }
}

// MARK: - Convenience Extensions

extension StableHLOModuleBuilder {
    public static func buildBlock(_ component: StableHLOFunction) -> [StableHLOFunction] {
        [component]
    }
}

extension StableHLOFunctionBuilder {
    // Support for single statement
    public static func buildBlock(_ component: StableHLOStatement) -> [StableHLOStatement] {
        [component]
    }

    // Support for 2 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2]
    }

    // Support for 3 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3]
    }

    // Support for 4 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4]
    }

    // Support for 5 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5]
    }

    // Support for 6 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement, _ s6: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5, s6]
    }

    // Support for 7 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement, _ s6: StableHLOStatement, _ s7: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5, s6, s7]
    }

    // Support for 8 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement, _ s6: StableHLOStatement, _ s7: StableHLOStatement, _ s8: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5, s6, s7, s8]
    }

    // Support for 9 statements
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement, _ s6: StableHLOStatement, _ s7: StableHLOStatement, _ s8: StableHLOStatement, _ s9: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    }

    // Support for 10 statements (should be enough for most use cases)
    public static func buildBlock(_ s1: StableHLOStatement, _ s2: StableHLOStatement, _ s3: StableHLOStatement, _ s4: StableHLOStatement, _ s5: StableHLOStatement, _ s6: StableHLOStatement, _ s7: StableHLOStatement, _ s8: StableHLOStatement, _ s9: StableHLOStatement, _ s10: StableHLOStatement) -> [StableHLOStatement] {
        [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    }
}
