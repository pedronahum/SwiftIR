// BackendCompilation.swift
// Phase 9: Backend Compilation Infrastructure
//
// This implements the compilation pipeline that transforms
// MLIR graphs into executable code via XLA.

import Foundation

// MARK: - Compilation Target

/// Target platform for compilation
public enum CompilationTarget {
    case cpu
    case cuda(deviceId: Int)
    case tpu(deviceId: Int)
    case metal(deviceId: Int)

    public static var defaultCPU: CompilationTarget { .cpu }

    public var description: String {
        switch self {
        case .cpu: return "CPU"
        case .cuda(let id): return "CUDA:\(id)"
        case .tpu(let id): return "TPU:\(id)"
        case .metal(let id): return "Metal:\(id)"
        }
    }
}

// MARK: - Optimization Level

/// Optimization level for compilation
public enum OptimizationLevel: Int {
    case none = 0      // O0: No optimization
    case basic = 1     // O1: Basic optimizations
    case standard = 2  // O2: Standard optimizations (default)
    case aggressive = 3 // O3: Aggressive optimizations

    public static var `default`: OptimizationLevel { .standard }
}

// MARK: - Compilation Options

/// Options for the compilation pipeline
public struct CompilationOptions {
    public var target: CompilationTarget
    public var optimizationLevel: OptimizationLevel
    public var enableFusion: Bool
    public var enableLayoutOptimization: Bool
    public var debugMode: Bool
    public var cacheEnabled: Bool

    public static var `default`: CompilationOptions {
        CompilationOptions(
            target: .cpu,
            optimizationLevel: .standard,
            enableFusion: true,
            enableLayoutOptimization: true,
            debugMode: false,
            cacheEnabled: true
        )
    }

    public init(
        target: CompilationTarget = .cpu,
        optimizationLevel: OptimizationLevel = .standard,
        enableFusion: Bool = true,
        enableLayoutOptimization: Bool = true,
        debugMode: Bool = false,
        cacheEnabled: Bool = true
    ) {
        self.target = target
        self.optimizationLevel = optimizationLevel
        self.enableFusion = enableFusion
        self.enableLayoutOptimization = enableLayoutOptimization
        self.debugMode = debugMode
        self.cacheEnabled = cacheEnabled
    }
}

// MARK: - MLIR Builder

/// Accumulates MLIR operations during graph building
public class MLIRBuilder {
    public var operations: [MLIROperation] = []
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
    public func addOperation(_ op: MLIROperation) {
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

    /// Build the MLIR module
    public func build(functionName: String = "main") -> MLIRModule {
        MLIRModule(
            functionName: functionName,
            arguments: arguments,
            operations: operations,
            results: results
        )
    }

    /// Reset the builder
    public func reset() {
        operations = []
        arguments = []
        results = []
        ssaCounter = 0
    }
}

// MARK: - MLIR Operation

/// Represents a single MLIR operation
public struct MLIROperation: Equatable {
    public let result: String
    public let opName: String
    public let operands: [String]
    public let attributes: [String: String]
    public let resultType: String
    public let regions: [String]  // MLIR regions (code blocks)

    public init(
        result: String,
        opName: String,
        operands: [String],
        attributes: [String: String] = [:],
        resultType: String,
        regions: [String] = []
    ) {
        self.result = result
        self.opName = opName
        self.operands = operands
        self.attributes = attributes
        self.resultType = resultType
        self.regions = regions
    }

    /// Generate MLIR text representation
    public var mlirText: String {
        var text = "\(result) = \(opName)"

        // Special handling for constant operation - value comes before type
        if opName == "constant" || opName == "stablehlo.constant" {
            if let value = attributes["value"] {
                text += " \(value)"
                return text
            }
        }

        // Special handling for dot operation - needs full signature type
        if opName == "dot" || opName == "stablehlo.dot" {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            // Get the special _dot_types attribute if present
            if let dotTypes = attributes["_dot_types"] {
                text += " : \(dotTypes)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for transpose - use dims = [...] format with functional type
        if opName == "transpose" || opName == "stablehlo.transpose" {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            if let perm = attributes["permutation"] {
                // Use dims = [...] format for StableHLO transpose
                text += ", dims = \(perm)"
            }
            // Get input type from _input_type attribute if present
            if let inputType = attributes["_input_type"] {
                text += " : (\(inputType)) -> \(resultType)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for compare operation
        if opName == "compare" || opName == "stablehlo.compare" {
            if let direction = attributes["comparison_direction"] {
                // Format: %result = stablehlo.compare GT, %a, %b : (type, type) -> result_type
                text += " \(direction), " + operands.joined(separator: ", ")
                // Get input type - for scalar tensor<i1>, we need tensor<f32>
                // For tensor<Nxi1>, we get tensor<Nxf32>
                var inputType: String
                if resultType == "tensor<i1>" {
                    // Scalar comparison - use _input_type if available, otherwise default to tensor<f32>
                    inputType = attributes["_input_type"] ?? "tensor<f32>"
                } else {
                    // Tensor comparison - extract shape and convert i1 to f32
                    inputType = resultType.replacingOccurrences(of: "xi1>", with: "xf32>")
                }
                text += " : (\(inputType), \(inputType)) -> \(resultType)"
                return text
            }
        }

        // Special handling for select operation
        if opName == "select" || opName == "stablehlo.select" {
            // Format: %result = stablehlo.select %pred, %on_true, %on_false : (pred_type, val_type, val_type) -> result_type
            text += " " + operands.joined(separator: ", ")
            // pred_type is i1 version of result_type
            let predType = resultType.replacingOccurrences(of: "xf32>", with: "xi1>")
            text += " : (\(predType), \(resultType), \(resultType)) -> \(resultType)"
            return text
        }

        // Special handling for broadcast_in_dim operation
        if opName == "broadcast_in_dim" || opName == "stablehlo.broadcast_in_dim" {
            // Format: %result = stablehlo.broadcast_in_dim %input, dims = [0, 1] : (input_type) -> result_type
            text += " " + operands.joined(separator: ", ")
            if let dims = attributes["broadcast_dimensions"] {
                text += ", dims = \(dims)"
            }
            // Use the _input_type attribute if available
            if let inputType = attributes["_input_type"] {
                text += " : (\(inputType)) -> \(resultType)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for reshape operation
        if opName == "reshape" || opName == "stablehlo.reshape" {
            // Format: %result = stablehlo.reshape %input : (input_type) -> result_type
            text += " " + operands.joined(separator: ", ")
            // Use the _input_type attribute if available
            if let inputType = attributes["_input_type"] {
                text += " : (\(inputType)) -> \(resultType)"
            } else {
                // Infer input type - for scalar to tensor reshape
                let inputType = "tensor<\(resultType.contains("xf32") ? "f32" : "f64")>"
                text += " : (\(inputType)) -> \(resultType)"
            }
            return text
        }

        // Special handling for slice operation
        if opName == "slice" || opName == "stablehlo.slice" {
            // Format: %result = stablehlo.slice %input [s0:l0, s1:l1:stride] : (input_type) -> result_type
            text += " " + operands.joined(separator: ", ")

            // Use pretty-print syntax with bracket notation
            if let starts = attributes["start_indices"],
               let limits = attributes["limit_indices"],
               let strides = attributes["strides"] {
                // Parse arrays - remove brackets and split
                let startsArr = starts.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                let limitsArr = limits.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                let stridesArr = strides.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }

                // Build slice indices like [0:2, 1:3] or [0:2:1, 1:3:2]
                var sliceIndices: [String] = []
                for i in 0..<startsArr.count {
                    let s = startsArr[i]
                    let l = limitsArr[i]
                    let st = stridesArr[i]
                    if st == "1" {
                        sliceIndices.append("\(s):\(l)")
                    } else {
                        sliceIndices.append("\(s):\(l):\(st)")
                    }
                }
                text += " [\(sliceIndices.joined(separator: ", "))]"

                // Get input type from _input_type attribute
                let inputType = attributes["_input_type"] ?? resultType
                text += " : (\(inputType)) -> \(resultType)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for pad operation
        if opName == "pad" || opName == "stablehlo.pad" {
            // Format: %result = stablehlo.pad %input, %pad_value, low = [...], high = [...], interior = [...] : (input_type, scalar_type) -> result_type
            text += " " + operands.joined(separator: ", ")

            if let low = attributes["edge_padding_low"],
               let high = attributes["edge_padding_high"],
               let interior = attributes["interior_padding"] {
                // Get input type from _input_type attribute
                let inputType = attributes["_input_type"] ?? resultType
                // Pad value type is scalar
                let scalarType = resultType.contains("f32") ? "tensor<f32>" : "tensor<f64>"
                text += ", low = \(low)"
                text += ", high = \(high)"
                text += ", interior = \(interior)"
                text += " : (\(inputType), \(scalarType)) -> \(resultType)"
            } else {
                text += " : \(resultType)"
            }
            return text
        }

        // Special handling for convolution operation
        if opName == "convolution" || opName == "stablehlo.convolution" {
            // Format: stablehlo.convolution(%lhs, %rhs) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {...} {...} : types
            let lhs = operands.count > 0 ? operands[0] : "%lhs"
            let rhs = operands.count > 1 ? operands[1] : "%rhs"
            text += "(\(lhs), \(rhs))"

            // Add dimension numbers (required for convolution)
            // Format: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
            if let dimNumbers = attributes["dimension_numbers"] {
                // Remove the #stablehlo.conv<...> wrapper if present
                var dims = dimNumbers
                if dims.hasPrefix("#stablehlo.conv<") && dims.hasSuffix(">") {
                    dims = String(dims.dropFirst(16).dropLast())
                }
                text += " dim_numbers = \(dims)"
            }

            // Add window attributes
            if let strides = attributes["window_strides"] {
                text += ", window = {stride = \(strides)"
                if let padding = attributes["padding"] {
                    text += ", pad = \(padding)"
                }
                if let lhsDilation = attributes["lhs_dilation"] {
                    text += ", lhs_dilate = \(lhsDilation)"
                }
                text += "}"
            }

            // Add feature group count and batch group count as attribute dict
            let fgc = attributes["feature_group_count"] ?? "1"
            let bgc = attributes["batch_group_count"] ?? "1"
            text += " {batch_group_count = \(bgc) : i64, feature_group_count = \(fgc) : i64}"

            // Get types
            let inputType = attributes["_input_type"] ?? resultType
            let filterType = attributes["_filter_type"] ?? resultType
            text += " : (\(inputType), \(filterType)) -> \(resultType)"

            return text
        }

        // Special handling for reduce operations (sum, max, min)
        if opName == "reduce_sum" || opName == "reduce_max" || opName == "reduce_min" || opName == "stablehlo.reduce" {
            // Format requires a region body with reducer function
            guard operands.count >= 2 else {
                // For single operand reduce (reduce_max, reduce_min without explicit init)
                // We need to generate init value
                if operands.count == 1 {
                    let inputOperand = operands[0]
                    let dims = attributes["dimensions"] ?? "[]"

                    // Extract the element type from result type
                    let elementType: String
                    if resultType.contains("xf32>") || resultType.contains("<f32>") || resultType == "tensor<f32>" {
                        elementType = "f32"
                    } else if resultType.contains("xf64>") || resultType.contains("<f64>") || resultType == "tensor<f64>" {
                        elementType = "f64"
                    } else {
                        elementType = "f32"
                    }

                    let scalarType = "tensor<\(elementType)>"
                    let inputType = attributes["_input_type"] ?? resultType

                    // Determine reducer operation and init value
                    let reducerOp: String
                    let initValue: String
                    let reducerResultName: String
                    if opName == "reduce_max" {
                        reducerOp = "stablehlo.maximum"
                        initValue = elementType == "f32" ? "-3.40282347E+38" : "-1.7976931348623157E+308"
                        reducerResultName = "%reduce_max"
                    } else if opName == "reduce_min" {
                        reducerOp = "stablehlo.minimum"
                        initValue = elementType == "f32" ? "3.40282347E+38" : "1.7976931348623157E+308"
                        reducerResultName = "%reduce_min"
                    } else {
                        reducerOp = "stablehlo.add"
                        initValue = "0.000000e+00"
                        reducerResultName = "%reduce_sum"
                    }

                    // Generate inline init constant
                    text = """
%reduce_init_\(result.dropFirst()) = stablehlo.constant dense<\(initValue)> : \(scalarType)
\(result) = stablehlo.reduce(\(inputOperand) init: %reduce_init_\(result.dropFirst())) across dimensions = \(dims) : (\(inputType), \(scalarType)) -> \(resultType)
     reducer(%reduce_arg0: \(scalarType), %reduce_arg1: \(scalarType)) {
       \(reducerResultName) = \(reducerOp) %reduce_arg0, %reduce_arg1 : \(scalarType)
       stablehlo.return \(reducerResultName) : \(scalarType)
     }
"""
                    return text
                }
                return text + " : \(resultType)"
            }

            let inputOperand = operands[0]
            let initOperand = operands[1]
            let dims = attributes["dimensions"] ?? "[]"
            let inputType = attributes["_input_type"] ?? resultType

            // Extract the element type from result type (e.g., tensor<4xf32> -> f32)
            let elementType: String
            if resultType.contains("xf32>") {
                elementType = "f32"
            } else if resultType.contains("xf64>") {
                elementType = "f64"
            } else if resultType.contains("<f32>") || resultType == "tensor<f32>" {
                elementType = "f32"
            } else {
                elementType = "f32"
            }

            // Scalar tensor type for reducer arguments
            let scalarType = "tensor<\(elementType)>"

            // Determine reducer operation based on opName
            let reducerOp: String
            let reducerResultName: String
            if opName == "reduce_max" {
                reducerOp = "stablehlo.maximum"
                reducerResultName = "%reduce_max"
            } else if opName == "reduce_min" {
                reducerOp = "stablehlo.minimum"
                reducerResultName = "%reduce_min"
            } else {
                reducerOp = "stablehlo.add"
                reducerResultName = "%reduce_sum"
            }

            // Generate the reduce operation with region body
            text = """
\(result) = stablehlo.reduce(\(inputOperand) init: \(initOperand)) across dimensions = \(dims) : (\(inputType), \(scalarType)) -> \(resultType)
     reducer(%reduce_arg0: \(scalarType), %reduce_arg1: \(scalarType)) {
       \(reducerResultName) = \(reducerOp) %reduce_arg0, %reduce_arg1 : \(scalarType)
       stablehlo.return \(reducerResultName) : \(scalarType)
     }
"""
            return text
        }

        // Special handling for reduce_window operations (pooling)
        if opName == "reduce_window_max" || opName == "reduce_window_sum" {
            guard operands.count >= 1 else {
                return "// Error: reduce_window requires at least 1 operand"
            }

            let inputOperand = operands[0]
            let windowDims = attributes["window_dimensions"] ?? "[2, 2]"
            let windowStrides = attributes["window_strides"] ?? "[1, 1]"
            let padding = attributes["padding"] ?? "[[0, 0], [0, 0]]"

            // Extract element type
            let elementType: String
            if resultType.contains("xf32>") || resultType.contains("<f32>") {
                elementType = "f32"
            } else if resultType.contains("xf64>") || resultType.contains("<f64>") {
                elementType = "f64"
            } else {
                elementType = "f32"
            }

            let scalarType = "tensor<\(elementType)>"
            let inputType = attributes["_input_type"] ?? resultType

            // Determine reducer and init value
            let reducerOp: String
            let initValue: String
            let reducerResultName: String
            if opName == "reduce_window_max" {
                reducerOp = "stablehlo.maximum"
                initValue = elementType == "f32" ? "-3.40282347E+38" : "-1.7976931348623157E+308"
                reducerResultName = "%reduce_max"
            } else {
                reducerOp = "stablehlo.add"
                initValue = "0.000000e+00"
                reducerResultName = "%reduce_sum"
            }

            // Generate reduce_window operation
            // For 2D pooling on NHWC, window is [1, H, W, 1]
            let fullWindowDims = "[1, \(windowDims.dropFirst().dropLast()), 1]"
            let fullWindowStrides = "[1, \(windowStrides.dropFirst().dropLast()), 1]"
            let fullPadding = "[[0, 0], \(padding.dropFirst().dropLast()), [0, 0]]"

            // Construct input type from input shape attribute
            let inputShapeStr = attributes["_input_shape"] ?? "[1, 4, 4, 1]"
            let actualInputType = "tensor<\(inputShapeStr.dropFirst().dropLast().replacingOccurrences(of: ", ", with: "x"))x\(elementType)>"

            // Use generic form for reduce_window
            text = """
%\(result.dropFirst())_init = stablehlo.constant dense<\(initValue)> : \(scalarType)
\(result) = "stablehlo.reduce_window"(\(inputOperand), %\(result.dropFirst())_init) ({
^bb0(%arg_lhs: \(scalarType), %arg_rhs: \(scalarType)):
  \(reducerResultName) = \(reducerOp) %arg_lhs, %arg_rhs : \(scalarType)
  "stablehlo.return"(\(reducerResultName)) : (\(scalarType)) -> ()
}) {window_dimensions = array<i64: \(fullWindowDims.dropFirst().dropLast())>, window_strides = array<i64: \(fullWindowStrides.dropFirst().dropLast())>, padding = dense<\(fullPadding)> : tensor<4x2xi64>} : (\(actualInputType), \(scalarType)) -> \(resultType)
"""
            return text
        }

        // Special handling for select_and_scatter (maxpool gradient)
        if opName == "select_and_scatter_max" {
            guard operands.count >= 2 else {
                return "// Error: select_and_scatter requires 2 operands"
            }

            let inputOperand = operands[0]
            let sourceOperand = operands[1]
            let windowDims = attributes["window_dimensions"] ?? "[2, 2]"
            let windowStrides = attributes["window_strides"] ?? "[1, 1]"
            let padding = attributes["padding"] ?? "[[0, 0], [0, 0]]"

            // Extract element type
            let elementType: String
            if resultType.contains("xf32>") || resultType.contains("<f32>") {
                elementType = "f32"
            } else {
                elementType = "f32"
            }

            let scalarType = "tensor<\(elementType)>"

            // Full window dimensions for NHWC
            let fullWindowDims = "[1, \(windowDims.dropFirst().dropLast()), 1]"
            let fullWindowStrides = "[1, \(windowStrides.dropFirst().dropLast()), 1]"
            let fullPadding = "[[0, 0], \(padding.dropFirst().dropLast()), [0, 0]]"

            // Extract source type from resultType - it's the output shape (smaller)
            // The source is the gradient from the output, which is smaller than input
            // We need to pass source shape through attributes
            let sourceType = attributes["_source_type"] ?? resultType

            // Use generic form for select_and_scatter
            text = """
%\(result.dropFirst())_init = stablehlo.constant dense<0.000000e+00> : \(scalarType)
\(result) = "stablehlo.select_and_scatter"(\(inputOperand), \(sourceOperand), %\(result.dropFirst())_init) ({
^bb0(%arg_lhs: \(scalarType), %arg_rhs: \(scalarType)):
  %cmp = stablehlo.compare GE, %arg_lhs, %arg_rhs : (\(scalarType), \(scalarType)) -> tensor<i1>
  "stablehlo.return"(%cmp) : (tensor<i1>) -> ()
}, {
^bb0(%arg_lhs: \(scalarType), %arg_rhs: \(scalarType)):
  %add = stablehlo.add %arg_lhs, %arg_rhs : \(scalarType)
  "stablehlo.return"(%add) : (\(scalarType)) -> ()
}) {window_dimensions = array<i64: \(fullWindowDims.dropFirst().dropLast())>, window_strides = array<i64: \(fullWindowStrides.dropFirst().dropLast())>, padding = dense<\(fullPadding)> : tensor<4x2xi64>} : (\(resultType), \(sourceType), \(scalarType)) -> \(resultType)
"""
            return text
        }

        // Special handling for reduce_window_sum_grad (avgpool gradient)
        if opName == "reduce_window_sum_grad" {
            // This broadcasts the gradient back to the input shape
            // For simplicity, we use a reverse of reduce_window which is essentially padding + broadcasting
            guard operands.count >= 1 else {
                return "// Error: reduce_window_sum_grad requires 1 operand"
            }

            // For now, use a simple approach: broadcast dy to output shape
            // This is a simplified version - full implementation would need proper scatter
            let dyOperand = operands[0]
            let dyType = attributes["_dy_type"] ?? "tensor<1x2x2x1xf32>"

            // Use broadcast_in_dim as a placeholder
            // Full implementation would require custom lowering
            text = "\(result) = stablehlo.broadcast_in_dim \(dyOperand), dims = [0, 1, 2, 3] : (\(dyType)) -> \(resultType)"
            return text
        }

        // Special handling for scatter operation with region
        if (opName == "stablehlo.scatter" || opName == "scatter") && !regions.isEmpty {
            if !operands.isEmpty {
                text += " " + operands.joined(separator: ", ")
            }
            // Add attributes except scatter_computation
            if !attributes.isEmpty {
                let attrStr = attributes.filter { $0.key != "scatter_computation" }.map { "\($0.key) = \($0.value)" }.joined(separator: ", ")
                if !attrStr.isEmpty {
                    text += " {\(attrStr)}"
                }
            }
            text += " : \(resultType) {\n"
            for region in regions {
                let lines = region.split(separator: "\n")
                for line in lines {
                    text += "      \(line)\n"
                }
            }
            text += "    }"
            return text
        }

        // Special handling for scatter operation - use generic syntax with region
        if (opName == "stablehlo.scatter" || opName == "scatter") && regions.isEmpty {
            if let computation = attributes["scatter_computation"], computation == "add" {
                // Extract element type from result type
                let elementType: String
                if resultType.contains("f32") {
                    elementType = "f32"
                } else if resultType.contains("f64") {
                    elementType = "f64"
                } else if resultType.contains("i32") {
                    elementType = "i32"
                } else {
                    elementType = "f32"  // default
                }

                // Get the full type signature from _scatter_types attribute
                let typeSignature = attributes["_scatter_types"] ?? "(tensor<*x\(elementType)>, tensor<*xi32>, tensor<*x\(elementType)>) -> \(resultType)"

                // Use generic syntax: "stablehlo.scatter"(...) ({region}) {attrs} : signature
                text = "    \(result) = \"stablehlo.scatter\"("
                if !operands.isEmpty {
                    text += operands.joined(separator: ", ")
                }
                text += ") ({\n"
                text += "      ^bb0(%lhs: tensor<\(elementType)>, %rhs: tensor<\(elementType)>):\n"
                text += "        %0 = stablehlo.add %lhs, %rhs : tensor<\(elementType)>\n"
                text += "        stablehlo.return %0 : tensor<\(elementType)>\n"
                text += "    }) {\n"

                // Add attributes except scatter_computation and _scatter_types
                let filteredAttrs = attributes.filter { $0.key != "scatter_computation" && $0.key != "_scatter_types" }
                for (i, attr) in filteredAttrs.enumerated() {
                    text += "      \(attr.key) = \(attr.value)"
                    if i < filteredAttrs.count - 1 {
                        text += ",\n"
                    } else {
                        text += "\n"
                    }
                }
                text += "    } : \(typeSignature)"
                return text
            }
        }

        // Special handling for gather operation - use generic syntax with type signature
        if (opName == "stablehlo.gather" || opName == "gather") {
            if let typeSignature = attributes["_gather_types"] {
                // Use generic syntax: "stablehlo.gather"(...) {attrs} : signature
                text = "    \(result) = \"stablehlo.gather\"("
                if !operands.isEmpty {
                    text += operands.joined(separator: ", ")
                }
                text += ") {\n"

                // Add attributes except _gather_types
                let filteredAttrs = attributes.filter { $0.key != "_gather_types" }
                for (i, attr) in filteredAttrs.enumerated() {
                    text += "      \(attr.key) = \(attr.value)"
                    if i < filteredAttrs.count - 1 {
                        text += ",\n"
                    } else {
                        text += "\n"
                    }
                }
                text += "    } : \(typeSignature)"
                return text
            }
        }

        // Special handling for broadcast_in_dim operation - use generic syntax with type signature
        if (opName == "stablehlo.broadcast_in_dim" || opName == "broadcast_in_dim") {
            if let inputType = attributes["_input_type"] {
                // Build type signature: (input_type) -> result_type
                let typeSignature = "(\(inputType)) -> \(resultType)"

                // Use generic syntax: "stablehlo.broadcast_in_dim"(...) {attrs} : signature
                text = "    \(result) = \"stablehlo.broadcast_in_dim\"("
                if !operands.isEmpty {
                    text += operands.joined(separator: ", ")
                }
                text += ") {\n"

                // Add attributes except _input_type
                let filteredAttrs = attributes.filter { $0.key != "_input_type" }
                for (i, attr) in filteredAttrs.enumerated() {
                    text += "      \(attr.key) = \(attr.value)"
                    if i < filteredAttrs.count - 1 {
                        text += ",\n"
                    } else {
                        text += "\n"
                    }
                }
                text += "    } : \(typeSignature)"
                return text
            }
        }

        // Special handling for stablehlo.while operation with condition and body regions
        if opName == "stablehlo.while" && regions.count == 2 {
            // StableHLO while syntax:
            // %results0, %results1 = stablehlo.while(%arg0 = %init0, %arg1 = %init1) : tensor<T>, tensor<T>
            //   cond { ... }
            //   do { ... }

            // Parse the result type to get individual types (e.g., "(tensor<f32>, tensor<f32>)")
            let typeList = resultType.dropFirst().dropLast()  // Remove outer parentheses
            let types = typeList.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }

            // Extract region argument names from regions[0] (condition region)
            // The region starts with ^bb0(%argN: type, ...):
            var condArgNames: [String] = []
            let condFirstLine = regions[0].split(separator: "\n").first ?? ""
            if let openParen = condFirstLine.firstIndex(of: "("),
               let closeParen = condFirstLine.firstIndex(of: ")") {
                let argsStr = condFirstLine[condFirstLine.index(after: openParen)..<closeParen]
                let argParts = argsStr.split(separator: ",")
                for part in argParts {
                    if let colonIdx = part.firstIndex(of: ":") {
                        let argName = part[..<colonIdx].trimmingCharacters(in: .whitespaces)
                        condArgNames.append(argName)
                    }
                }
            }

            // Build the while op with proper binding syntax
            // Format: stablehlo.while(%arg0 = %init0, %arg1 = %init1, ...) : type0, type1, ...
            var bindings: [String] = []
            for i in 0..<min(operands.count, condArgNames.count) {
                bindings.append("\(condArgNames[i]) = \(operands[i])")
            }
            text += "(" + bindings.joined(separator: ", ") + ") : " + types.joined(separator: ", ")

            // Helper function to strip ^bb0(...) header from region (not needed with named arguments)
            func stripBlockHeader(_ region: String) -> String {
                var lines = region.split(separator: "\n", omittingEmptySubsequences: false).map { String($0) }
                if !lines.isEmpty && lines[0].hasPrefix("^bb") {
                    lines.removeFirst()
                }
                return lines.joined(separator: "\n")
            }

            // Add condition region (without ^bb0 header)
            text += "\n  cond {\n"
            let condBody = stripBlockHeader(regions[0])
            let condLines = condBody.split(separator: "\n", omittingEmptySubsequences: false)
            for line in condLines {
                if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                    text += "    \(line.trimmingCharacters(in: .whitespaces))\n"
                }
            }

            // Add body region (without ^bb0 header)
            text += "  } do {\n"
            let bodyBody = stripBlockHeader(regions[1])
            let bodyLines = bodyBody.split(separator: "\n", omittingEmptySubsequences: false)
            for line in bodyLines {
                if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                    text += "    \(line.trimmingCharacters(in: .whitespaces))\n"
                }
            }

            text += "  }"
            return text
        }

        if !operands.isEmpty {
            text += " " + operands.joined(separator: ", ")
        }
        if !attributes.isEmpty {
            let attrStr = attributes.map { "\($0.key) = \($0.value)" }.joined(separator: ", ")
            text += " {\(attrStr)}"
        }
        text += " : \(resultType)"
        return text
    }
}

// MARK: - MLIR Module

/// Represents a complete MLIR module
public struct MLIRModule {
    public let functionName: String
    public let arguments: [String]
    public let operations: [MLIROperation]
    public let results: [String]

    /// Module attributes for XLA optimization hints (matches JAX output)
    public var moduleAttributes: [String: String] = [
        "mhlo.num_partitions": "1 : i32",
        "mhlo.num_replicas": "1 : i32"
    ]

    /// Input-output aliases for buffer donation
    /// When specified, XLA can reuse input buffers for outputs with matching shapes
    public var inputOutputAliases: [InputOutputAlias] = []

    public init(
        functionName: String,
        arguments: [String],
        operations: [MLIROperation],
        results: [String]
    ) {
        self.functionName = functionName
        self.arguments = arguments
        self.operations = operations
        self.results = results
    }

    /// Generate MLIR text representation with optimization attributes
    public var mlirText: String {
        // Build module attributes string like JAX does
        let attrString: String
        if !moduleAttributes.isEmpty {
            let attrs = moduleAttributes.map { "\($0.key) = \($0.value)" }.joined(separator: ", ")
            attrString = " attributes {\(attrs)}"
        } else {
            attrString = ""
        }

        // Build a map of input index to output alias for argument attribute generation
        var aliasMap: [Int: InputOutputAlias] = [:]
        for alias in inputOutputAliases {
            aliasMap[alias.inputIndex] = alias
        }

        var text = "module @\(functionName)_module\(attrString) {\n"
        text += "  func.func @\(functionName)("

        // Generate arguments with optional alias attributes
        var argStrings: [String] = []
        for (i, arg) in arguments.enumerated() {
            if let alias = aliasMap[i] {
                // Add mhlo.result_alias attribute to this argument
                // Format: %arg0: tensor<N> {mhlo.result_alias = mhlo.result_alias<result_index = 0, must_alias = true>}
                let parts = arg.split(separator: ":", maxSplits: 1)
                if parts.count == 2 {
                    let argName = String(parts[0]).trimmingCharacters(in: .whitespaces)
                    let argType = String(parts[1]).trimmingCharacters(in: .whitespaces)
                    let mustAliasStr = alias.mustAlias ? "true" : "false"
                    argStrings.append("\(argName): \(argType) {mhlo.result_alias = mhlo.result_alias<result_index = \(alias.outputIndex), must_alias = \(mustAliasStr)>}")
                } else {
                    argStrings.append(arg)
                }
            } else {
                argStrings.append(arg)
            }
        }
        text += argStrings.joined(separator: ", ")
        text += ")"

        // Add return type if there are results
        if !results.isEmpty && !operations.isEmpty {
            // Get return types for all results
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    // Result might be an argument - extract type from arguments
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
            // Get result types for return statement
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    // Result might be an argument - extract type from arguments
                    for arg in arguments {
                        let parts = arg.split(separator: ":", maxSplits: 1)
                        if parts.count == 2 && parts[0].trimmingCharacters(in: .whitespaces) == result {
                            resultTypes.append(parts[1].trimmingCharacters(in: .whitespaces))
                            break
                        }
                    }
                }
            }

            text += "    return \(results.joined(separator: ", ")) : "
            text += resultTypes.joined(separator: ", ")
            text += "\n"
        }

        text += "  }\n"
        text += "}\n"
        return text
    }

    /// Get operation count
    public var operationCount: Int {
        operations.count
    }
}

// MARK: - StableHLO Emitter

/// Converts generic MLIR to StableHLO dialect
public class StableHLOEmitter {
    public init() {}

    /// Convert an MLIR module to StableHLO
    public func convert(_ module: MLIRModule) throws -> StableHLOModule {
        var stablehloOps: [StableHLOOp] = []

        for op in module.operations {
            let stablehloOp = try convertOperation(op)
            stablehloOps.append(stablehloOp)
        }

        return StableHLOModule(
            functionName: module.functionName,
            arguments: module.arguments,
            operations: stablehloOps,
            results: module.results
        )
    }

    private func convertOperation(_ op: MLIROperation) throws -> StableHLOOp {
        // Map operation names to StableHLO equivalents
        let stablehloName: String
        switch op.opName {
        case "add", "arith.addf":
            stablehloName = "stablehlo.add"
        case "multiply", "arith.mulf":
            stablehloName = "stablehlo.multiply"
        case "subtract", "arith.subf":
            stablehloName = "stablehlo.subtract"
        case "divide", "arith.divf":
            stablehloName = "stablehlo.divide"
        case "negate", "arith.negf":
            stablehloName = "stablehlo.negate"
        case "dot", "linalg.matmul":
            stablehloName = "stablehlo.dot_general"
        case "transpose":
            stablehloName = "stablehlo.transpose"
        case "reshape":
            stablehloName = "stablehlo.reshape"
        case "broadcast":
            stablehloName = "stablehlo.broadcast_in_dim"
        case "reduce_sum":
            stablehloName = "stablehlo.reduce"
        case "maximum":
            stablehloName = "stablehlo.maximum"
        case "minimum":
            stablehloName = "stablehlo.minimum"
        case "exp":
            stablehloName = "stablehlo.exponential"
        case "log":
            stablehloName = "stablehlo.log"
        case "compare":
            stablehloName = "stablehlo.compare"
        case "select":
            stablehloName = "stablehlo.select"
        case "constant":
            stablehloName = "stablehlo.constant"
        case "stablehlo.gather":
            stablehloName = "stablehlo.gather"
        case "stablehlo.scatter":
            stablehloName = "stablehlo.scatter"
        default:
            // Check if opName already has stablehlo prefix
            if op.opName.hasPrefix("stablehlo.") {
                stablehloName = op.opName
            } else {
                stablehloName = "stablehlo.\(op.opName)"
            }
        }

        // Special handling for scatter: generate region from scatter_computation attribute
        var regions = op.regions
        if stablehloName == "stablehlo.scatter" && regions.isEmpty {
            if let computation = op.attributes["scatter_computation"], computation == "add" {
                // Generate add reduction region for scatter
                // Extract element type from result type
                let elementType: String
                if op.resultType.contains("f32") {
                    elementType = "f32"
                } else if op.resultType.contains("f64") {
                    elementType = "f64"
                } else if op.resultType.contains("i32") {
                    elementType = "i32"
                } else {
                    elementType = "f32"  // default
                }

                let region = """
^bb0(%arg0: tensor<\(elementType)>, %arg1: tensor<\(elementType)>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<\(elementType)>
    stablehlo.return %sum : tensor<\(elementType)>
"""
                regions = [region]
            }
        }

        return StableHLOOp(
            result: op.result,
            opName: stablehloName,
            operands: op.operands,
            attributes: op.attributes,
            resultType: op.resultType,
            regions: regions
        )
    }
}

// MARK: - StableHLO Types

/// Represents a StableHLO operation
public struct StableHLOOp {
    public let result: String
    public let opName: String
    public let operands: [String]
    public let attributes: [String: String]
    public let resultType: String
    public let regions: [String]  // MLIR regions (code blocks)

    public var mlirText: String {
        var text = "\(result) = \(opName)"

        // Special handling for constant operation - value comes before type
        if opName == "stablehlo.constant" {
            if let value = attributes["value"] {
                text += " \(value)"
                return text
            }
        }

        // Special handling for compare operation
        if opName == "stablehlo.compare", let direction = attributes["comparison_direction"] {
            // Format: %result = stablehlo.compare GT, %a, %b : (type, type) -> result_type
            text += " \(direction), " + operands.joined(separator: ", ")
            // Extract input type from result type (e.g., tensor<4xi1> -> tensor<4xf32>)
            let inputType = resultType.replacingOccurrences(of: "xi1>", with: "xf32>")
            text += " : (\(inputType), \(inputType)) -> \(resultType)"
            return text
        }

        if !operands.isEmpty {
            text += " " + operands.joined(separator: ", ")
        }
        if !attributes.isEmpty {
            let attrStr = attributes.map { key, value -> String in
                // Special handling for comparison_direction attribute
                if key == "comparison_direction" {
                    return "\(key) = #stablehlo<comparison_direction \(value)>"
                }
                // Skip scatter_computation attribute as it's handled by region
                if key == "scatter_computation" {
                    return ""
                }
                return "\(key) = \(value)"
            }.filter { !$0.isEmpty }.joined(separator: ", ")
            if !attrStr.isEmpty {
                text += " {\(attrStr)}"
            }
        }
        text += " : \(resultType)"

        // Add regions if present (for operations like scatter)
        if !regions.isEmpty {
            text += " {\n"
            for region in regions {
                // Each region line should be indented
                let lines = region.split(separator: "\n")
                for line in lines {
                    text += "    \(line)\n"
                }
            }
            text += "  }"
        }

        return text
    }
}

/// Represents an input-to-output alias for buffer donation
/// This tells XLA that an input buffer can be reused for an output with matching shape
public struct InputOutputAlias {
    /// Index of the input argument to donate
    public let inputIndex: Int
    /// Index of the output result that will reuse the input buffer
    public let outputIndex: Int
    /// Whether this is a must-alias (true) or may-alias (false)
    public let mustAlias: Bool

    public init(inputIndex: Int, outputIndex: Int, mustAlias: Bool = true) {
        self.inputIndex = inputIndex
        self.outputIndex = outputIndex
        self.mustAlias = mustAlias
    }
}

/// Represents a StableHLO module
public struct StableHLOModule {
    public let functionName: String
    public let arguments: [String]
    public let operations: [StableHLOOp]
    public let results: [String]

    /// Input-output aliases for buffer donation
    /// When specified, XLA can reuse input buffers for outputs with matching shapes
    public var inputOutputAliases: [InputOutputAlias] = []

    public var mlirText: String {
        // Build a map of input index to output alias for argument attribute generation
        var aliasMap: [Int: InputOutputAlias] = [:]
        for alias in inputOutputAliases {
            aliasMap[alias.inputIndex] = alias
        }

        var text = "module {\n"
        text += "  func.func @\(functionName)("

        // Generate arguments with optional alias attributes
        var argStrings: [String] = []
        for (i, arg) in arguments.enumerated() {
            if let alias = aliasMap[i] {
                // Add mhlo.result_alias attribute to this argument
                // Format: %arg0: tensor<N> {mhlo.result_alias = mhlo.result_alias<result_index = 0, must_alias = true>}
                let parts = arg.split(separator: ":", maxSplits: 1)
                if parts.count == 2 {
                    let argName = String(parts[0]).trimmingCharacters(in: .whitespaces)
                    let argType = String(parts[1]).trimmingCharacters(in: .whitespaces)
                    let mustAliasStr = alias.mustAlias ? "true" : "false"
                    argStrings.append("\(argName): \(argType) {mhlo.result_alias = mhlo.result_alias<result_index = \(alias.outputIndex), must_alias = \(mustAliasStr)>}")
                } else {
                    argStrings.append(arg)
                }
            } else {
                argStrings.append(arg)
            }
        }
        text += argStrings.joined(separator: ", ")
        text += ")"

        // Add return type if there are results
        if !results.isEmpty {
            // Get return types for all results
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    // Result might be an argument - extract type from arguments
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
            let opText = op.mlirText
            text += "    \(opText)\n"
        }

        if !results.isEmpty {
            // Get result types for return statement
            var resultTypes: [String] = []
            for result in results {
                if let op = operations.first(where: { $0.result == result }) {
                    resultTypes.append(op.resultType)
                } else {
                    // Result might be an argument - extract type from arguments
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

    public var operationCount: Int {
        operations.count
    }
}

// MARK: - XLA Interface

/// Interface to XLA compilation
public class XLAInterface {
    public let target: CompilationTarget

    public init(target: CompilationTarget) {
        self.target = target
    }

    /// Compile StableHLO to XLA executable
    public func compile(_ module: StableHLOModule, options: CompilationOptions) throws -> XLAExecutable {
        // In a real implementation, this would:
        // 1. Serialize StableHLO to protobuf
        // 2. Call XLA's C++ compiler
        // 3. Return the compiled binary

        // For now, we create a mock executable that captures the compilation
        return XLAExecutable(
            module: module,
            target: target,
            optimizationLevel: options.optimizationLevel
        )
    }
}

// MARK: - XLA Executable

/// Represents a compiled XLA executable
public class XLAExecutable {
    public let module: StableHLOModule
    public let target: CompilationTarget
    public let optimizationLevel: OptimizationLevel
    public let compilationTimestamp: Date

    public init(
        module: StableHLOModule,
        target: CompilationTarget,
        optimizationLevel: OptimizationLevel
    ) {
        self.module = module
        self.target = target
        self.optimizationLevel = optimizationLevel
        self.compilationTimestamp = Date()
    }

    /// Execute with input data
    public func execute(_ inputs: [[Float]]) -> [[Float]] {
        // In a real implementation, this would:
        // 1. Transfer data to device
        // 2. Launch compiled kernel
        // 3. Transfer results back

        // For now, return placeholder results
        return inputs.map { $0.map { _ in Float(0) } }
    }

    /// Get compiled code info
    public var info: String {
        """
        XLA Executable:
          Target: \(target.description)
          Optimization: O\(optimizationLevel.rawValue)
          Operations: \(module.operationCount)
          Compiled: \(compilationTimestamp)
        """
    }
}

// MARK: - SwiftIR Compiler

/// Main compiler class that orchestrates the compilation pipeline
public class SwiftIRCompiler {
    private let options: CompilationOptions
    private var cache: [String: CompiledFunction] = [:]

    public init(options: CompilationOptions = .default) {
        self.options = options
    }

    /// Compile an MLIR module to an executable
    public func compile(_ module: MLIRModule) throws -> CompiledFunction {
        // Check cache
        let cacheKey = module.mlirText
        if options.cacheEnabled, let cached = cache[cacheKey] {
            return cached
        }

        // Step 1: Convert to StableHLO
        let stablehloEmitter = StableHLOEmitter()
        let stablehloModule = try stablehloEmitter.convert(module)

        // Step 2: Optimize (future: run optimization passes)
        let optimizedModule = try optimize(stablehloModule)

        // Step 3: Compile with XLA
        let xlaInterface = XLAInterface(target: options.target)
        let executable = try xlaInterface.compile(optimizedModule, options: options)

        // Create compiled function
        let compiled = CompiledFunction(
            executable: executable,
            mlirSource: module.mlirText,
            stablehloSource: stablehloModule.mlirText
        )

        // Cache it
        if options.cacheEnabled {
            cache[cacheKey] = compiled
        }

        return compiled
    }

    /// Compile a traced function
    public func compileTraced<T>(
        _ function: (T) -> TracedValue,
        input: T
    ) throws -> CompiledFunction where T: TracerConvertible {
        // Build MLIR by tracing
        let builder = MLIRBuilder()

        // Create symbolic input
        let inputName = builder.freshSSA()
        builder.addArgument(name: inputName, type: "tensor<*xf32>")

        // Trace the function
        let traced = function(input)

        // Set result
        builder.setResults([traced.irValue])

        // Build and compile
        let module = builder.build()
        return try compile(module)
    }

    private func optimize(_ module: StableHLOModule) throws -> StableHLOModule {
        // Future: implement optimization passes like:
        // - Common subexpression elimination
        // - Dead code elimination
        // - Operation fusion
        // - Layout optimization

        // For now, return as-is
        return module
    }

    /// Clear the compilation cache
    public func clearCache() {
        cache.removeAll()
    }
}

// MARK: - Compiled Function

/// Represents a compiled and executable function
public class CompiledFunction {
    public let executable: XLAExecutable
    public let mlirSource: String
    public let stablehloSource: String

    public init(
        executable: XLAExecutable,
        mlirSource: String,
        stablehloSource: String
    ) {
        self.executable = executable
        self.mlirSource = mlirSource
        self.stablehloSource = stablehloSource
    }

    /// Run the compiled function with inputs
    public func run(_ inputs: [[Float]]) -> [[Float]] {
        executable.execute(inputs)
    }

    /// Get compilation info
    public var info: String {
        executable.info
    }
}

// MARK: - Supporting Protocols

/// Protocol for types that can be converted to/from Tracers
public protocol TracerConvertible {
    var tracerRepresentation: String { get }
}

/// Represents a traced value in the computation graph
public struct TracedValue {
    public let irValue: String
    public let shape: [Int]

    public init(irValue: String, shape: [Int] = []) {
        self.irValue = irValue
        self.shape = shape
    }
}

// MARK: - Convenience Functions

/// Compile a function with default options
public func compileFunction(_ module: MLIRModule) throws -> CompiledFunction {
    let compiler = SwiftIRCompiler()
    return try compiler.compile(module)
}

/// Compile with specific options
public func compileFunction(
    _ module: MLIRModule,
    options: CompilationOptions
) throws -> CompiledFunction {
    let compiler = SwiftIRCompiler(options: options)
    return try compiler.compile(module)
}

// MARK: - Compilation Errors

/// Errors that can occur during compilation
public enum CompilationError: Error, CustomStringConvertible {
    case invalidMLIR(String)
    case unsupportedOperation(String)
    case targetNotAvailable(CompilationTarget)
    case optimizationFailed(String)
    case xlaCompilationFailed(String)

    public var description: String {
        switch self {
        case .invalidMLIR(let msg):
            return "Invalid MLIR: \(msg)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .targetNotAvailable(let target):
            return "Target not available: \(target.description)"
        case .optimizationFailed(let msg):
            return "Optimization failed: \(msg)"
        case .xlaCompilationFailed(let msg):
            return "XLA compilation failed: \(msg)"
        }
    }
}
