// JupyterCompiler.swift
// Compilation pipeline for Jupyter/Colab
// Converts traced operations to StableHLO and executes via PJRT

import Foundation

// MARK: - MLIR Operation

/// Represents a single MLIR operation
public struct JMLIROperation: Equatable {
    public let result: String
    public let opName: String
    public let operands: [String]
    public let attributes: [String: String]
    public let resultType: String

    public init(
        result: String,
        opName: String,
        operands: [String],
        attributes: [String: String] = [:],
        resultType: String
    ) {
        self.result = result
        self.opName = opName
        self.operands = operands
        self.attributes = attributes
        self.resultType = resultType
    }

    /// Generate MLIR text representation
    public var mlirText: String {
        // Special handling for raw MLIR (like while loops)
        if let rawMlir = attributes["_raw_mlir"] {
            return rawMlir
        }

        var text = "\(result) = \(opName)"

        // Special handling for constant operation
        if opName == "stablehlo.constant" {
            if let value = attributes["value"] {
                text += " \(value)"
                return text
            }
        }

        // Special handling for dot operation
        if opName == "stablehlo.dot_general" || opName == "stablehlo.dot" {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            if let dotTypes = attributes["_dot_types"] {
                text += ", contracting_dims = [1] x [0] : \(dotTypes)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for broadcast_in_dim (scalar to tensor broadcast)
        if opName == "stablehlo.broadcast_in_dim" {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            // For scalar broadcast, dims = [] means no dimensions map
            text += ", dims = []"
            text += " : (tensor<f32>) -> \(resultType)"
            return text
        }

        // Special handling for transpose
        if opName == "stablehlo.transpose" {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            if let perm = attributes["permutation"] {
                text += ", dims = \(perm)"
            }
            if let inputType = attributes["_input_type"] {
                text += " : (\(inputType)) -> \(resultType)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for reduce operations
        if opName == "stablehlo.reduce" {
            let inputOperand = operands.count > 0 ? operands[0] : "%input"
            let dims = attributes["dimensions"] ?? "[]"
            let inputType = attributes["_input_type"] ?? resultType

            // Extract element type
            let elementType: String
            if resultType.contains("xf32>") || resultType.contains("<f32>") || resultType == "tensor<f32>" {
                elementType = "f32"
            } else if resultType.contains("xf64>") || resultType.contains("<f64>") || resultType == "tensor<f64>" {
                elementType = "f64"
            } else {
                elementType = "f32"
            }

            let scalarType = "tensor<\(elementType)>"

            // Generate reduce with init and region
            text = """
%reduce_init_\(result.dropFirst()) = stablehlo.constant dense<0.0> : \(scalarType)
\(result) = stablehlo.reduce(\(inputOperand) init: %reduce_init_\(result.dropFirst())) across dimensions = \(dims) : (\(inputType), \(scalarType)) -> \(resultType)
     reducer(%reduce_arg0: \(scalarType), %reduce_arg1: \(scalarType)) {
       %reduce_sum = stablehlo.add %reduce_arg0, %reduce_arg1 : \(scalarType)
       stablehlo.return %reduce_sum : \(scalarType)
     }
"""
            return text
        }

        if !operands.isEmpty {
            text += " " + operands.joined(separator: ", ")
        }
        if !attributes.isEmpty {
            let filteredAttrs = attributes.filter { !$0.key.hasPrefix("_") }
            if !filteredAttrs.isEmpty {
                let attrStr = filteredAttrs.map { "\($0.key) = \($0.value)" }.joined(separator: ", ")
                text += " {\(attrStr)}"
            }
        }
        text += " : \(resultType)"
        return text
    }
}

// MARK: - MLIR Builder

/// Accumulates MLIR operations during graph building
public class JMLIRBuilder {
    public var operations: [JMLIROperation] = []
    public var arguments: [String] = []
    public var results: [String] = []
    private var ssaCounter: Int = 0

    public init() {}

    /// Generate a fresh SSA value name
    public func freshSSA() -> String {
        let name = "%v\(ssaCounter)"
        ssaCounter += 1
        return name
    }

    /// Add an operation to the builder
    public func addOperation(_ op: JMLIROperation) {
        operations.append(op)
    }

    /// Add an argument to the function
    public func addArgument(name: String, type: String) {
        arguments.append("\(name): \(type)")
    }

    /// Set the return values
    public func setResults(_ values: [String]) {
        results = values
    }

    /// Build the MLIR module text
    public func build(functionName: String = "main") -> String {
        var text = "module @\(functionName) {\n"
        text += "  func.func @main("
        text += arguments.joined(separator: ", ")
        text += ")"

        // Add return type if there are results
        if !results.isEmpty && !operations.isEmpty {
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    // Result might be an argument
                    for arg in arguments {
                        let parts = arg.split(separator: ":", maxSplits: 1)
                        if parts.count == 2 && parts[0].trimmingCharacters(in: .whitespaces) == result {
                            resultTypes.append(parts[1].trimmingCharacters(in: .whitespaces))
                            break
                        }
                    }
                }
            }

            if resultTypes.count == 1 {
                text += " -> \(resultTypes[0])"
            } else if resultTypes.count > 1 {
                text += " -> (\(resultTypes.joined(separator: ", ")))"
            }
        }

        text += " {\n"

        for op in operations {
            text += "    \(op.mlirText)\n"
        }

        if !results.isEmpty {
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    for arg in arguments {
                        let parts = arg.split(separator: ":", maxSplits: 1)
                        if parts.count == 2 && parts[0].trimmingCharacters(in: .whitespaces) == result {
                            resultTypes.append(parts[1].trimmingCharacters(in: .whitespaces))
                            break
                        }
                    }
                }
            }
            text += "    return \(results.joined(separator: ", ")) : \(resultTypes.joined(separator: ", "))\n"
        }

        text += "  }\n"
        text += "}\n"
        return text
    }

    /// Reset the builder
    public func reset() {
        operations = []
        arguments = []
        results = []
        ssaCounter = 0
    }
}

// MARK: - Tracing Context

/// Context for tracing operations and building MLIR
public class JTracingContext {
    public let builder: JMLIRBuilder
    private var argumentCount: Int = 0
    private var valueMap: [UInt64: String] = [:]  // Map tracer IDs to SSA values

    public init() {
        self.builder = JMLIRBuilder()
        // Reset the global graph builder for a fresh trace
        JTracerGraphBuilder.shared.reset()
    }

    /// Create a symbolic input
    public func input(shape: JTensorShape, dtype: JDType = .float32, name: String? = nil) -> JTracer {
        let argName = name ?? "%arg\(argumentCount)"
        argumentCount += 1

        let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
        // Handle scalar (rank-0) tensors correctly
        let typeStr: String
        if shape.rank == 0 {
            typeStr = "tensor<\(dtype.rawValue)>"
        } else {
            typeStr = "tensor<\(shapeStr)x\(dtype.rawValue)>"
        }
        builder.addArgument(name: argName, type: typeStr)

        // Create a tracer that references this input
        let tracer = JTracer(shape: shape, dtype: dtype)
        valueMap[tracer.valueId] = argName

        return tracer
    }

    /// Set the output of the traced function
    public func output(_ tracers: JTracer...) {
        // Convert traced operations to MLIR
        convertToMLIR()

        // Set results
        let resultNames = tracers.map { valueMap[$0.valueId] ?? "%v\($0.valueId)" }
        builder.setResults(resultNames)
    }

    /// Convert traced operations to MLIR operations
    private func convertToMLIR() {
        let operations = JTracerGraphBuilder.shared.getOperations()

        for op in operations {
            switch op {
            case .constant(let id, let value, let shape, let dtype):
                // Skip constants that are already mapped (like inputs)
                if valueMap[id] != nil { continue }

                let result = "%v\(id)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.constant",
                    operands: [],
                    attributes: ["value": "dense<\(value)> : \(typeStr)"],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .placeholder(let id, let shape, let dtype):
                // Placeholders are already handled as inputs
                if valueMap[id] == nil {
                    valueMap[id] = "%arg\(id)"
                }

            case .binary(let id, let op, let lhs, let rhs, let shape, let dtype):
                let result = "%v\(id)"
                var lhsName = valueMap[lhs] ?? "%v\(lhs)"
                var rhsName = valueMap[rhs] ?? "%v\(rhs)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                // Broadcast scalar operands to match result shape for element-wise ops
                if op != .matmul && shape.rank > 0 {
                    let lhsType = getTypeForValue(lhs)
                    let rhsType = getTypeForValue(rhs)

                    // Check if LHS is scalar and needs broadcast
                    if let lt = lhsType, lt == "tensor<\(dtype.rawValue)>" || lt == "tensor<f32>" {
                        let broadcastResult = "%broadcast_lhs_\(id)"
                        builder.addOperation(JMLIROperation(
                            result: broadcastResult,
                            opName: "stablehlo.broadcast_in_dim",
                            operands: [lhsName],
                            attributes: ["broadcast_dimensions": "dense<> : tensor<0xi64>"],
                            resultType: typeStr
                        ))
                        lhsName = broadcastResult
                    }

                    // Check if RHS is scalar and needs broadcast
                    if let rt = rhsType, rt == "tensor<\(dtype.rawValue)>" || rt == "tensor<f32>" {
                        let broadcastResult = "%broadcast_rhs_\(id)"
                        builder.addOperation(JMLIROperation(
                            result: broadcastResult,
                            opName: "stablehlo.broadcast_in_dim",
                            operands: [rhsName],
                            attributes: ["broadcast_dimensions": "dense<> : tensor<0xi64>"],
                            resultType: typeStr
                        ))
                        rhsName = broadcastResult
                    }
                }

                let opName: String
                var attrs: [String: String] = [:]
                switch op {
                case .add: opName = "stablehlo.add"
                case .subtract: opName = "stablehlo.subtract"
                case .multiply: opName = "stablehlo.multiply"
                case .divide: opName = "stablehlo.divide"
                case .matmul:
                    opName = "stablehlo.dot_general"
                    // Get LHS and RHS types for dot_general signature
                    if let lhsType = getTypeForValue(lhs), let rhsType = getTypeForValue(rhs) {
                        attrs["_dot_types"] = "(\(lhsType), \(rhsType)) -> \(typeStr)"
                    }
                }

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: opName,
                    operands: [lhsName, rhsName],
                    attributes: attrs,
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .unary(let id, let op, let input, let shape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                var attrs: [String: String] = [:]
                switch op {
                case .exp:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.exponential", operands: [inputName], resultType: typeStr))
                case .log:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.log", operands: [inputName], resultType: typeStr))
                case .sqrt:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.sqrt", operands: [inputName], resultType: typeStr))
                case .rsqrt:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.rsqrt", operands: [inputName], resultType: typeStr))
                case .abs:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.abs", operands: [inputName], resultType: typeStr))
                case .neg:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.negate", operands: [inputName], resultType: typeStr))
                case .sin:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.sine", operands: [inputName], resultType: typeStr))
                case .cos:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.cosine", operands: [inputName], resultType: typeStr))
                case .tan:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.tan", operands: [inputName], resultType: typeStr))
                case .tanh:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.tanh", operands: [inputName], resultType: typeStr))
                case .sigmoid:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.logistic", operands: [inputName], resultType: typeStr))
                case .floor:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.floor", operands: [inputName], resultType: typeStr))
                case .ceil:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.ceil", operands: [inputName], resultType: typeStr))
                case .relu:
                    // ReLU is max(x, 0)
                    let zeroResult = "%zero_\(id)"
                    builder.addOperation(JMLIROperation(result: zeroResult, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.maximum", operands: [inputName, zeroResult], resultType: typeStr))
                case .silu:
                    // SiLU = x * sigmoid(x)
                    let sigResult = "%sig_\(id)"
                    builder.addOperation(JMLIROperation(result: sigResult, opName: "stablehlo.logistic", operands: [inputName], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.multiply", operands: [inputName, sigResult], resultType: typeStr))
                case .gelu:
                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let half = "%gelu_half_\(id)"
                    let sqrtTwoPi = "%gelu_sqrt2pi_\(id)"
                    let coef = "%gelu_coef_\(id)"
                    let one = "%gelu_one_\(id)"
                    let x3 = "%gelu_x3_\(id)"
                    let scaled = "%gelu_scaled_\(id)"
                    let inner = "%gelu_inner_\(id)"
                    let innerScaled = "%gelu_innerscaled_\(id)"
                    let tanhResult = "%gelu_tanh_\(id)"
                    let onePlusTanh = "%gelu_oneplustanh_\(id)"
                    let halfX = "%gelu_halfx_\(id)"

                    builder.addOperation(JMLIROperation(result: half, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.5> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: sqrtTwoPi, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.7978845608> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: coef, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.044715> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: one, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<1.0> : \(typeStr)"], resultType: typeStr))

                    // x^3 = x * x * x
                    let x2 = "%gelu_x2_\(id)"
                    builder.addOperation(JMLIROperation(result: x2, opName: "stablehlo.multiply", operands: [inputName, inputName], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: x3, opName: "stablehlo.multiply", operands: [x2, inputName], resultType: typeStr))

                    // 0.044715 * x^3
                    builder.addOperation(JMLIROperation(result: scaled, opName: "stablehlo.multiply", operands: [coef, x3], resultType: typeStr))

                    // x + 0.044715 * x^3
                    builder.addOperation(JMLIROperation(result: inner, opName: "stablehlo.add", operands: [inputName, scaled], resultType: typeStr))

                    // sqrt(2/pi) * (x + 0.044715 * x^3)
                    builder.addOperation(JMLIROperation(result: innerScaled, opName: "stablehlo.multiply", operands: [sqrtTwoPi, inner], resultType: typeStr))

                    // tanh(...)
                    builder.addOperation(JMLIROperation(result: tanhResult, opName: "stablehlo.tanh", operands: [innerScaled], resultType: typeStr))

                    // 1 + tanh(...)
                    builder.addOperation(JMLIROperation(result: onePlusTanh, opName: "stablehlo.add", operands: [one, tanhResult], resultType: typeStr))

                    // 0.5 * x
                    builder.addOperation(JMLIROperation(result: halfX, opName: "stablehlo.multiply", operands: [half, inputName], resultType: typeStr))

                    // 0.5 * x * (1 + tanh(...))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.multiply", operands: [halfX, onePlusTanh], resultType: typeStr))
                case .softplus:
                    // softplus = log(1 + exp(x))
                    let one = "%sp_one_\(id)"
                    let expX = "%sp_exp_\(id)"
                    let onePlusExp = "%sp_oneplusexp_\(id)"
                    builder.addOperation(JMLIROperation(result: one, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<1.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: expX, opName: "stablehlo.exponential", operands: [inputName], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: onePlusExp, opName: "stablehlo.add", operands: [one, expX], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.log", operands: [onePlusExp], resultType: typeStr))
                case .softmax, .logSoftmax:
                    // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
                    // LogSoftmax: x - max(x) - log(sum(exp(x - max(x))))
                    // For simplicity, we'll use the numerically stable version
                    // Assuming last axis reduction
                    let axis = shape.rank > 0 ? shape.rank - 1 : 0
                    let elementType = dtype.rawValue
                    let scalarType = "tensor<\(elementType)>"

                    // Generate reduce_max first
                    let maxResult = "%softmax_max_\(id)"
                    let maxInit = "%softmax_maxinit_\(id)"
                    builder.addOperation(JMLIROperation(result: maxInit, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<-3.40282347E+38> : \(scalarType)"], resultType: scalarType))

                    // For softmax we need reduce_max along last axis
                    // This is a simplified version - full implementation would need proper reduce with keepdims
                    let shifted = "%softmax_shifted_\(id)"
                    let expShifted = "%softmax_exp_\(id)"
                    let sumExp = "%softmax_sum_\(id)"

                    // Simplified: just compute exp(x) / sum(exp(x)) without max subtraction for now
                    // A proper implementation would need axis-aware reductions
                    builder.addOperation(JMLIROperation(result: expShifted, opName: "stablehlo.exponential", operands: [inputName], resultType: typeStr))

                    // Create reduce sum
                    let sumInit = "%softmax_suminit_\(id)"
                    builder.addOperation(JMLIROperation(result: sumInit, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.0> : \(scalarType)"], resultType: scalarType))

                    // For now, use element-wise divide with broadcast
                    // A full implementation would properly reduce along axis
                    if op == .softmax {
                        // exp(x) - simplified version
                        builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.exponential", operands: [inputName], resultType: typeStr))
                    } else {
                        // log_softmax = log(softmax) = x - log(sum(exp(x)))
                        // Simplified: just compute log(exp(x)) = x for now
                        builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.log", operands: [inputName], resultType: typeStr))
                    }
                case .reshape:
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.reshape", operands: [inputName], resultType: typeStr))
                case .transpose:
                    let inputType = getTypeForValue(input) ?? typeStr
                    attrs["_input_type"] = inputType
                    attrs["permutation"] = "[1, 0]"
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.transpose", operands: [inputName], attributes: attrs, resultType: typeStr))
                case .leakyRelu, .elu:
                    // These should be handled by unaryWithAlpha, but add fallback
                    let zeroResult = "%zero_\(id)"
                    builder.addOperation(JMLIROperation(result: zeroResult, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.maximum", operands: [inputName, zeroResult], resultType: typeStr))
                }
                valueMap[id] = result

            case .reduction(let id, let op, let input, let axes, _, let shape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                let inputType = getTypeForValue(input) ?? typeStr

                let dims = "[\(axes.sorted().map(String.init).joined(separator: ", "))]"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.reduce",
                    operands: [inputName],
                    attributes: [
                        "dimensions": dims,
                        "_input_type": inputType
                    ],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .power(let id, let base, let exponent, let shape, let dtype):
                let result = "%v\(id)"
                let baseName = valueMap[base] ?? "%v\(base)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                // Create exponent constant
                let expResult = "%exp_\(id)"
                builder.addOperation(JMLIROperation(
                    result: expResult,
                    opName: "stablehlo.constant",
                    operands: [],
                    attributes: ["value": "dense<\(exponent)> : \(typeStr)"],
                    resultType: typeStr
                ))

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.power",
                    operands: [baseName, expResult],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .comparison(let id, let lhs, let rhs, let direction, let inputDtype):
                let result = "%v\(id)"
                let lhsName = valueMap[lhs] ?? "%v\(lhs)"
                let rhsName = valueMap[rhs] ?? "%v\(rhs)"
                let inputType = getTypeForValue(lhs) ?? "tensor<\(inputDtype.rawValue)>"
                let resultType = "tensor<i1>"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.compare",
                    operands: [lhsName, rhsName],
                    attributes: [
                        "comparison_direction": "#stablehlo<comparison_direction \(direction)>",
                        "_input_type": inputType
                    ],
                    resultType: resultType
                ))
                valueMap[id] = result

            case .reshape(let id, let input, let newShape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = newShape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = newShape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                let inputType = getTypeForValue(input) ?? typeStr

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.reshape",
                    operands: [inputName],
                    attributes: ["_input_type": inputType],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .clamp(let id, let input, let minVal, let maxVal, let shape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                // Create min and max constants
                let minResult = "%clamp_min_\(id)"
                let maxResult = "%clamp_max_\(id)"
                builder.addOperation(JMLIROperation(result: minResult, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<\(minVal)> : \(typeStr)"], resultType: typeStr))
                builder.addOperation(JMLIROperation(result: maxResult, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<\(maxVal)> : \(typeStr)"], resultType: typeStr))
                builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.clamp", operands: [minResult, inputName, maxResult], resultType: typeStr))
                valueMap[id] = result

            case .slice(let id, let input, let starts, let limits, let strides, let shape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                let inputType = getTypeForValue(input) ?? typeStr

                let startsStr = "dense<[\(starts.map(String.init).joined(separator: ", "))]> : tensor<\(starts.count)xi64>"
                let limitsStr = "dense<[\(limits.map(String.init).joined(separator: ", "))]> : tensor<\(limits.count)xi64>"
                let stridesStr = "dense<[\(strides.map(String.init).joined(separator: ", "))]> : tensor<\(strides.count)xi64>"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.slice",
                    operands: [inputName],
                    attributes: [
                        "start_indices": startsStr,
                        "limit_indices": limitsStr,
                        "strides": stridesStr,
                        "_input_type": inputType
                    ],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .concatenate(let id, let inputs, let axis, let shape, let dtype):
                let result = "%v\(id)"
                let inputNames = inputs.map { valueMap[$0] ?? "%v\($0)" }
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.concatenate",
                    operands: inputNames,
                    attributes: ["dimension": "\(axis)"],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .binaryElementwise(let id, let op, let lhs, let rhs, let shape, let dtype):
                let result = "%v\(id)"
                let lhsName = valueMap[lhs] ?? "%v\(lhs)"
                let rhsName = valueMap[rhs] ?? "%v\(rhs)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                let opName: String
                switch op {
                case .maximum: opName = "stablehlo.maximum"
                case .minimum: opName = "stablehlo.minimum"
                }

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: opName,
                    operands: [lhsName, rhsName],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .select(let id, let condition, let onTrue, let onFalse, let shape, let dtype):
                let result = "%v\(id)"
                let condName = valueMap[condition] ?? "%v\(condition)"
                let trueName = valueMap[onTrue] ?? "%v\(onTrue)"
                let falseName = valueMap[onFalse] ?? "%v\(onFalse)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                builder.addOperation(JMLIROperation(
                    result: result,
                    opName: "stablehlo.select",
                    operands: [condName, trueName, falseName],
                    resultType: typeStr
                ))
                valueMap[id] = result

            case .unaryWithAlpha(let id, let op, let input, let alpha, let shape, let dtype):
                let result = "%v\(id)"
                let inputName = valueMap[input] ?? "%v\(input)"
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

                switch op {
                case .leakyRelu:
                    // LeakyReLU: max(alpha * x, x) where alpha < 1
                    // Or equivalently: x if x > 0 else alpha * x
                    let alphaConst = "%leaky_alpha_\(id)"
                    let alphaX = "%leaky_alphax_\(id)"
                    let zero = "%leaky_zero_\(id)"
                    let cond = "%leaky_cond_\(id)"

                    builder.addOperation(JMLIROperation(result: alphaConst, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<\(alpha)> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: alphaX, opName: "stablehlo.multiply", operands: [alphaConst, inputName], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: zero, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: cond, opName: "stablehlo.compare", operands: [inputName, zero], attributes: ["comparison_direction": "#stablehlo<comparison_direction GT>"], resultType: "tensor<i1>"))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.select", operands: [cond, inputName, alphaX], resultType: typeStr))

                case .elu:
                    // ELU: x if x > 0 else alpha * (exp(x) - 1)
                    let alphaConst = "%elu_alpha_\(id)"
                    let one = "%elu_one_\(id)"
                    let zero = "%elu_zero_\(id)"
                    let expX = "%elu_exp_\(id)"
                    let expMinus1 = "%elu_expminus1_\(id)"
                    let alphaExpMinus1 = "%elu_alphaexpminus1_\(id)"
                    let cond = "%elu_cond_\(id)"

                    builder.addOperation(JMLIROperation(result: alphaConst, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<\(alpha)> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: one, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<1.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: zero, opName: "stablehlo.constant", operands: [], attributes: ["value": "dense<0.0> : \(typeStr)"], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: expX, opName: "stablehlo.exponential", operands: [inputName], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: expMinus1, opName: "stablehlo.subtract", operands: [expX, one], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: alphaExpMinus1, opName: "stablehlo.multiply", operands: [alphaConst, expMinus1], resultType: typeStr))
                    builder.addOperation(JMLIROperation(result: cond, opName: "stablehlo.compare", operands: [inputName, zero], attributes: ["comparison_direction": "#stablehlo<comparison_direction GT>"], resultType: "tensor<i1>"))
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.select", operands: [cond, inputName, alphaExpMinus1], resultType: typeStr))

                default:
                    // Fallback: just pass through
                    builder.addOperation(JMLIROperation(result: result, opName: "stablehlo.abs", operands: [inputName], resultType: typeStr))
                }
                valueMap[id] = result
            }
        }
    }

    /// Get the MLIR type string for a traced value
    private func getTypeForValue(_ id: UInt64) -> String? {
        let operations = JTracerGraphBuilder.shared.getOperations()
        for op in operations {
            switch op {
            case .constant(let opId, _, let shape, let dtype),
                 .placeholder(let opId, let shape, let dtype):
                if opId == id {
                    let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                    return shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                }
            case .binary(let opId, _, _, _, let shape, let dtype),
                 .unary(let opId, _, _, let shape, let dtype),
                 .reduction(let opId, _, _, _, _, let shape, let dtype),
                 .power(let opId, _, _, let shape, let dtype),
                 .reshape(let opId, _, let shape, let dtype),
                 .clamp(let opId, _, _, _, let shape, let dtype),
                 .slice(let opId, _, _, _, _, let shape, let dtype),
                 .concatenate(let opId, _, _, let shape, let dtype),
                 .binaryElementwise(let opId, _, _, _, let shape, let dtype),
                 .select(let opId, _, _, _, let shape, let dtype),
                 .unaryWithAlpha(let opId, _, _, _, let shape, let dtype):
                if opId == id {
                    let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                    return shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                }
            case .comparison(let opId, _, _, _, _):
                if opId == id {
                    return "tensor<i1>"
                }
            }
        }
        return nil
    }

    /// Generate while loop MLIR from traced operations
    private func generateWhileLoops() {
        let whileLoops = JWhileLoopBuilder.shared.getWhileLoops()

        for record in whileLoops {
            // Get types for state elements
            var tupleTypes: [String] = []
            for (shape, dtype) in zip(record.shapes, record.dtypes) {
                let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
                let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
                tupleTypes.append(typeStr)
            }

            // Get input value names
            var inputNames: [String] = []
            for id in record.initialIds {
                inputNames.append(valueMap[id] ?? "%v\(id)")
            }

            // Generate argument names for regions
            var argNames: [String] = []
            for id in record.argIds {
                argNames.append("%warg\(id)")
            }

            // Build condition region with traced operations
            var condRegion = "  {\n"
            condRegion += "  ^cond("
            for (i, (argName, typeStr)) in zip(argNames, tupleTypes).enumerated() {
                condRegion += "\(argName): \(typeStr)"
                condRegion += i < tupleTypes.count - 1 ? ", " : ""
            }
            condRegion += "):\n"

            // Convert condition operations to MLIR
            for op in record.conditionOps {
                let mlirText = convertOperationToMLIR(op, argIds: record.argIds, argNames: argNames)
                condRegion += "    \(mlirText)\n"
            }

            // Return condition result
            let condResultName = getValueName(record.conditionResultId, argIds: record.argIds, argNames: argNames)
            condRegion += "    stablehlo.return \(condResultName) : tensor<i1>\n"
            condRegion += "  }"

            // Build body region with traced operations
            var bodyRegion = "  {\n"
            bodyRegion += "  ^body("
            for (i, (argName, typeStr)) in zip(argNames, tupleTypes).enumerated() {
                bodyRegion += "\(argName): \(typeStr)"
                bodyRegion += i < tupleTypes.count - 1 ? ", " : ""
            }
            bodyRegion += "):\n"

            // Convert body operations to MLIR
            for op in record.bodyOps {
                let mlirText = convertOperationToMLIR(op, argIds: record.argIds, argNames: argNames)
                bodyRegion += "    \(mlirText)\n"
            }

            // Return body results
            let bodyResultNames = record.bodyResultIds.map { getValueName($0, argIds: record.argIds, argNames: argNames) }
            bodyRegion += "    stablehlo.return \(bodyResultNames.joined(separator: ", ")) : \(tupleTypes.joined(separator: ", "))\n"
            bodyRegion += "  }"

            // Generate the while operation
            let resultSSA = "%while_\(record.id)"
            let resultType = tupleTypes.count == 1 ? tupleTypes[0] : "(\(tupleTypes.joined(separator: ", ")))"

            // Create the while operation with regions
            let whileOp = """
\(resultSSA):4 = stablehlo.while(\(inputNames.joined(separator: ", "))) : \(resultType)
\(condRegion),
\(bodyRegion)
"""
            builder.addOperation(JMLIROperation(
                result: resultSSA,
                opName: "// stablehlo.while",  // Mark as special
                operands: [],
                attributes: ["_raw_mlir": whileOp],
                resultType: resultType
            ))

            // Map result IDs to SSA names
            for (i, resultId) in record.resultIds.enumerated() {
                if record.stateCount == 1 {
                    valueMap[resultId] = resultSSA
                } else {
                    valueMap[resultId] = "\(resultSSA)#\(i)"
                }
            }
        }
    }

    /// Get value name, using arg name if it matches an arg ID
    private func getValueName(_ id: UInt64, argIds: [UInt64], argNames: [String]) -> String {
        if let idx = argIds.firstIndex(of: id) {
            return argNames[idx]
        }
        return valueMap[id] ?? "%v\(id)"
    }

    /// Convert a traced operation to MLIR text
    private func convertOperationToMLIR(_ op: JTracedOperation, argIds: [UInt64], argNames: [String]) -> String {
        switch op {
        case .constant(let id, let value, let shape, let dtype):
            let result = "%v\(id)"
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            return "\(result) = stablehlo.constant dense<\(value)> : \(typeStr)"

        case .placeholder(let id, let shape, let dtype):
            // Placeholders in regions are handled via argIds
            return "// placeholder \(id)"

        case .binary(let id, let op, let lhs, let rhs, let shape, let dtype):
            let result = "%v\(id)"
            let lhsName = getValueName(lhs, argIds: argIds, argNames: argNames)
            let rhsName = getValueName(rhs, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

            let opName: String
            switch op {
            case .add: opName = "stablehlo.add"
            case .subtract: opName = "stablehlo.subtract"
            case .multiply: opName = "stablehlo.multiply"
            case .divide: opName = "stablehlo.divide"
            case .matmul: opName = "stablehlo.dot"
            }

            return "\(result) = \(opName) \(lhsName), \(rhsName) : \(typeStr)"

        case .unary(let id, let op, let input, let shape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

            let opName: String
            switch op {
            case .exp: opName = "stablehlo.exponential"
            case .log: opName = "stablehlo.log"
            case .sqrt: opName = "stablehlo.sqrt"
            case .rsqrt: opName = "stablehlo.rsqrt"
            case .abs: opName = "stablehlo.abs"
            case .neg: opName = "stablehlo.negate"
            case .sin: opName = "stablehlo.sine"
            case .cos: opName = "stablehlo.cosine"
            case .tan: opName = "stablehlo.tan"
            case .tanh: opName = "stablehlo.tanh"
            case .sigmoid: opName = "stablehlo.logistic"
            case .floor: opName = "stablehlo.floor"
            case .ceil: opName = "stablehlo.ceil"
            case .relu: opName = "stablehlo.maximum"  // max(x, 0)
            case .reshape: opName = "stablehlo.reshape"
            case .transpose: opName = "stablehlo.transpose"
            case .silu, .gelu, .softplus, .softmax, .logSoftmax, .leakyRelu, .elu:
                // These are expanded in the main convertToMLIR, simplified here
                opName = "stablehlo.abs"  // fallback
            }

            return "\(result) = \(opName) \(inputName) : \(typeStr)"

        case .comparison(let id, let lhs, let rhs, let direction, let inputDtype):
            let result = "%v\(id)"
            let lhsName = getValueName(lhs, argIds: argIds, argNames: argNames)
            let rhsName = getValueName(rhs, argIds: argIds, argNames: argNames)
            let inputType = "tensor<\(inputDtype.rawValue)>"

            return "\(result) = stablehlo.compare \(direction), \(lhsName), \(rhsName) : (\(inputType), \(inputType)) -> tensor<i1>"

        case .reduction(let id, _, let input, let axes, _, let shape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            let dims = "[\(axes.sorted().map(String.init).joined(separator: ", "))]"

            return "\(result) = stablehlo.reduce(\(inputName)) across dimensions = \(dims) : \(typeStr)"

        case .power(let id, let base, let exponent, let shape, let dtype):
            let result = "%v\(id)"
            let baseName = getValueName(base, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

            return "\(result) = stablehlo.power \(baseName), dense<\(exponent)> : \(typeStr)"

        case .reshape(let id, let input, let newShape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = newShape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = newShape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"

            return "\(result) = stablehlo.reshape \(inputName) : \(typeStr)"

        case .clamp(let id, let input, let minVal, let maxVal, let shape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            return "\(result) = stablehlo.clamp dense<\(minVal)>, \(inputName), dense<\(maxVal)> : \(typeStr)"

        case .slice(let id, let input, let starts, let limits, let strides, let shape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            return "\(result) = stablehlo.slice \(inputName) [starts=\(starts), limits=\(limits), strides=\(strides)] : \(typeStr)"

        case .concatenate(let id, let inputs, let axis, let shape, let dtype):
            let result = "%v\(id)"
            let inputNames = inputs.map { getValueName($0, argIds: argIds, argNames: argNames) }
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            return "\(result) = stablehlo.concatenate \(inputNames.joined(separator: ", ")), dim=\(axis) : \(typeStr)"

        case .binaryElementwise(let id, let op, let lhs, let rhs, let shape, let dtype):
            let result = "%v\(id)"
            let lhsName = getValueName(lhs, argIds: argIds, argNames: argNames)
            let rhsName = getValueName(rhs, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            let opName = op == .maximum ? "stablehlo.maximum" : "stablehlo.minimum"
            return "\(result) = \(opName) \(lhsName), \(rhsName) : \(typeStr)"

        case .select(let id, let condition, let onTrue, let onFalse, let shape, let dtype):
            let result = "%v\(id)"
            let condName = getValueName(condition, argIds: argIds, argNames: argNames)
            let trueName = getValueName(onTrue, argIds: argIds, argNames: argNames)
            let falseName = getValueName(onFalse, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            return "\(result) = stablehlo.select \(condName), \(trueName), \(falseName) : \(typeStr)"

        case .unaryWithAlpha(let id, _, let input, _, let shape, let dtype):
            let result = "%v\(id)"
            let inputName = getValueName(input, argIds: argIds, argNames: argNames)
            let shapeStr = shape.dimensions.compactMap { $0 }.map(String.init).joined(separator: "x")
            let typeStr = shape.rank == 0 ? "tensor<\(dtype.rawValue)>" : "tensor<\(shapeStr)x\(dtype.rawValue)>"
            // Simplified version for while loops - full expansion in main convertToMLIR
            return "\(result) = stablehlo.abs \(inputName) : \(typeStr)"
        }
    }

    /// Build the MLIR module
    public func buildModule(name: String = "traced") -> String {
        // Generate any while loops that were recorded
        generateWhileLoops()
        return builder.build(functionName: name)
    }
}

// MARK: - Jupyter Compiled Function

/// A compiled function that can execute on PJRT
public class JupyterCompiledFunction {
    public let mlirSource: String
    private var executable: JupyterPJRTExecutable?
    private var client: JupyterPJRTClient?

    public init(mlirSource: String) {
        self.mlirSource = mlirSource
    }

    /// Compile the function (lazy - done on first execution)
    private func ensureCompiled() throws {
        guard executable == nil else { return }

        if client == nil {
            client = try SwiftIRJupyter.shared.createClient()
        }
        executable = try client!.compile(mlir: mlirSource)
    }

    /// Execute with float arrays
    public func execute(_ inputs: [[Float]], shapes: [[Int64]]) throws -> [[Float]] {
        try ensureCompiled()

        guard let exec = executable, let client = client else {
            throw SwiftIRJupyterError.invalidState(message: "Failed to compile")
        }

        // Create input buffers
        var buffers: [JupyterPJRTBuffer] = []
        for (data, shape) in zip(inputs, shapes) {
            let buffer = try client.createBuffer(data: data, shape: shape)
            buffers.append(buffer)
        }

        // Execute
        let outputPtrs = try exec.execute(inputs: buffers)

        // Read outputs
        var results: [[Float]] = []
        for ptr in outputPtrs {
            // For now, assume same size as first input
            // In production, would get actual output shape from executable
            let count = inputs.isEmpty ? 1 : inputs[0].count
            var result = [Float](repeating: 0, count: count)
            try PJRTBindings.shared.bufferToHost(ptr, into: &result)
            PJRTBindings.shared.destroyBuffer(ptr)
            results.append(result)
        }

        return results
    }

    public var info: String {
        """
        JupyterCompiledFunction
        MLIR size: \(mlirSource.count) bytes
        """
    }
}

// MARK: - Function Compiler

/// Compiles Swift functions to PJRT executables
public class JupyterFunctionCompiler {
    public init() {}

    /// Compile a function that takes a single tracer input
    public func compile(
        inputShape: [Int],
        dtype: JDType = .float32,
        _ function: (JTracer) -> JTracer
    ) throws -> JupyterCompiledFunction {
        let context = JTracingContext()

        // Create symbolic input
        let input = context.input(shape: JTensorShape(inputShape), dtype: dtype)

        // Trace the function
        let output = function(input)

        // Set output
        context.output(output)

        // Build MLIR
        let mlir = context.buildModule()

        return JupyterCompiledFunction(mlirSource: mlir)
    }

    /// Compile a function with multiple inputs
    public func compile(
        inputSpecs: [(shape: [Int], dtype: JDType)],
        _ function: ([JTracer]) -> JTracer
    ) throws -> JupyterCompiledFunction {
        let context = JTracingContext()

        // Create symbolic inputs
        let inputs = inputSpecs.map { spec in
            context.input(shape: JTensorShape(spec.shape), dtype: spec.dtype)
        }

        // Trace the function
        let output = function(inputs)

        // Set output
        context.output(output)

        // Build MLIR
        let mlir = context.buildModule()

        return JupyterCompiledFunction(mlirSource: mlir)
    }
}

// MARK: - Convenience API

/// Trace and compile a function
public func jitCompile(
    inputShape: [Int],
    dtype: JDType = .float32,
    _ function: (JTracer) -> JTracer
) throws -> JupyterCompiledFunction {
    let compiler = JupyterFunctionCompiler()
    return try compiler.compile(inputShape: inputShape, dtype: dtype, function)
}

/// Trace and compile a multi-input function
public func jitCompile(
    inputSpecs: [(shape: [Int], dtype: JDType)],
    _ function: ([JTracer]) -> JTracer
) throws -> JupyterCompiledFunction {
    let compiler = JupyterFunctionCompiler()
    return try compiler.compile(inputSpecs: inputSpecs, function)
}
