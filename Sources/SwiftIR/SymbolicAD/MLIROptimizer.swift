/// MLIROptimizer.swift
/// MLIR-level optimization passes for gradient code
///
/// Implements Tangent-style 7-pass optimization pipeline:
/// 1. Constant Folding: Evaluate constant expressions at compile time
/// 2. Constant Deduplication: Merge identical constant definitions
/// 3. Assignment Propagation: Inline single-use variables
/// 4. Strength Reduction: Replace expensive ops with cheaper equivalents
/// 5. Common Subexpression Elimination (CSE): Reuse identical computations
/// 6. Algebraic Simplification: Apply mathematical identities (x*1=x, x+0=x)
/// 7. Fixed-Point Iteration: Repeat until no more optimizations possible
///
/// Dead Code Elimination runs at the end to clean up.
///
/// Reference: Tangent autodiff achieves 2.35x speedup with this pipeline.

import Foundation

// MARK: - Optimization Level

/// Controls how much MLIR-level optimization to apply
/// JAX applies minimal MLIR optimization, letting XLA handle it
public enum MLIROptimizationLevel: Sendable {
    /// No MLIR optimization - pass through to XLA directly (JAX-style)
    case none

    /// Minimal optimization - only symbol DCE (matches JAX exactly)
    case minimal

    /// Reduced optimization - CSE and DCE only (recommended for most cases)
    case reduced

    /// Standard optimization - all 7 passes (current default)
    case standard

    /// Maximum optimization - all passes with extra iterations
    case maximum
}

// MARK: - MLIR Optimizer

/// Optimizer that applies passes to MLIR operations
public class MLIROptimizer {

    /// The optimization level to use
    public var level: MLIROptimizationLevel = .standard

    public init(level: MLIROptimizationLevel = .standard) {
        self.level = level
    }

    /// Apply all optimization passes to an array of MLIR operations
    /// Uses fixed-point iteration to maximize optimization
    /// - Parameters:
    ///   - operations: The operations to optimize
    ///   - results: The result SSA values (used for DCE liveness analysis)
    /// - Returns: Optimized operations with SSA renaming map
    public func optimize(
        operations: [MLIROperation],
        results: [String]
    ) -> (operations: [MLIROperation], ssaMap: [String: String]) {
        var ops = operations
        var ssaMap: [String: String] = [:]

        // Handle different optimization levels
        switch level {
        case .none:
            // No optimization - pass through directly (like JAX with XLA)
            return (ops, ssaMap)

        case .minimal:
            // Only DCE (matches JAX's symbol-dce)
            ops = eliminateDeadCode(ops, results: results, ssaMap: ssaMap)
            return (ops, ssaMap)

        case .reduced:
            // CSE + DCE only - let XLA handle most optimizations
            (ops, ssaMap) = eliminateCommonSubexpressions(ops, ssaMap: ssaMap)
            ops = eliminateDeadCode(ops, results: results, ssaMap: ssaMap)
            return (ops, ssaMap)

        case .standard, .maximum:
            // Full 7-pass pipeline
            break
        }

        // Fixed-point iteration: repeat passes until no changes
        var changed = true
        var iterations = 0
        let maxIterations = level == .maximum ? 20 : 10  // More iterations for maximum

        while changed && iterations < maxIterations {
            changed = false
            let countBefore = ops.count
            let mapSizeBefore = ssaMap.count

            // Pass 1: Constant Folding (evaluate constant expressions)
            (ops, ssaMap) = foldConstants(ops, ssaMap: ssaMap)

            // Pass 2: Constant Deduplication
            (ops, ssaMap) = deduplicateConstants(ops, ssaMap: ssaMap)

            // Pass 3: Assignment Propagation (inline single-use values)
            (ops, ssaMap) = propagateAssignments(ops, ssaMap: ssaMap, results: results)

            // Pass 4: Strength Reduction (x**2 -> x*x, etc.)
            (ops, ssaMap) = reduceStrength(ops, ssaMap: ssaMap)

            // Pass 5: Common Subexpression Elimination
            (ops, ssaMap) = eliminateCommonSubexpressions(ops, ssaMap: ssaMap)

            // Pass 6: Algebraic Simplification (x*1=x, x+0=x, x-0=x, x/1=x)
            (ops, ssaMap) = simplifyAlgebraically(ops, ssaMap: ssaMap)

            // Check if anything changed
            if ops.count != countBefore || ssaMap.count != mapSizeBefore {
                changed = true
            }

            iterations += 1
        }

        // Final pass: Dead Code Elimination
        ops = eliminateDeadCode(ops, results: results, ssaMap: ssaMap)

        return (ops, ssaMap)
    }

    /// Apply SSA renaming to an operation
    private func renameOperands(_ op: MLIROperation, ssaMap: [String: String]) -> MLIROperation {
        let renamedOperands = op.operands.map { ssaMap[$0] ?? $0 }
        return MLIROperation(
            result: op.result,
            opName: op.opName,
            operands: renamedOperands,
            attributes: op.attributes,
            resultType: op.resultType,
            regions: op.regions
        )
    }

    // MARK: - Constant Deduplication

    /// Deduplicate identical constant definitions
    /// Multiple `constant dense<1.0>` with same type -> single constant
    private func deduplicateConstants(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap

        // Key: (value attribute, resultType) -> first SSA name
        var constantMap: [String: String] = [:]

        for op in operations {
            // Apply existing renames
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            if renamedOp.opName == "constant" || renamedOp.opName == "stablehlo.constant" {
                // Create a key from value and type
                let valueAttr = renamedOp.attributes["value"] ?? ""
                let key = "\(valueAttr)|\(renamedOp.resultType)"

                if let existingSSA = constantMap[key] {
                    // This constant already exists - map this SSA to existing
                    newSSAMap[renamedOp.result] = existingSSA
                    // Don't add to result - it's a duplicate
                } else {
                    // First time seeing this constant
                    constantMap[key] = renamedOp.result
                    result.append(renamedOp)
                }
            } else {
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    // MARK: - Common Subexpression Elimination

    /// Eliminate common subexpressions
    /// If two operations have same (opName, operands, attributes), reuse first result
    private func eliminateCommonSubexpressions(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap

        // Key: canonical expression string -> first SSA name
        var expressionMap: [String: String] = [:]

        for op in operations {
            // Apply existing renames to get canonical operands
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            // Skip operations that have side effects or regions
            if !renamedOp.regions.isEmpty || isSideEffecting(renamedOp.opName) {
                result.append(renamedOp)
                continue
            }

            // Create canonical key
            let key = canonicalKey(for: renamedOp)

            if let existingSSA = expressionMap[key] {
                // This expression already exists - map this SSA to existing
                newSSAMap[renamedOp.result] = existingSSA
                // Don't add to result - it's a duplicate
            } else {
                // First time seeing this expression
                expressionMap[key] = renamedOp.result
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    /// Create a canonical key for an operation
    private func canonicalKey(for op: MLIROperation) -> String {
        // Sort attributes for consistent ordering
        let sortedAttrs = op.attributes
            .filter { !$0.key.hasPrefix("_") } // Skip internal attributes
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ",")

        return "\(op.opName)|\(op.operands.joined(separator: ","))|\(sortedAttrs)|\(op.resultType)"
    }

    /// Check if an operation has side effects
    private func isSideEffecting(_ opName: String) -> Bool {
        // Most StableHLO ops are pure (no side effects)
        // Side-effecting ops include: infeed, outfeed, send, recv, etc.
        let sideEffectingOps = Set([
            "infeed", "outfeed", "send", "recv", "all_reduce", "all_gather",
            "stablehlo.infeed", "stablehlo.outfeed", "stablehlo.send", "stablehlo.recv"
        ])
        return sideEffectingOps.contains(opName)
    }

    // MARK: - Dead Code Elimination

    /// Eliminate operations whose results are never used
    private func eliminateDeadCode(
        _ operations: [MLIROperation],
        results: [String],
        ssaMap: [String: String]
    ) -> [MLIROperation] {
        // Apply SSA renaming to results to get canonical result names
        let canonicalResults = results.map { ssaMap[$0] ?? $0 }

        // Build set of live SSA values by walking backwards from results
        var liveSSAs = Set(canonicalResults)
        var changed = true

        // Apply SSA renaming to all operations first
        let renamedOps = operations.map { renameOperands($0, ssaMap: ssaMap) }

        // Iterate until fixed point
        while changed {
            changed = false
            for op in renamedOps.reversed() {
                if liveSSAs.contains(op.result) {
                    // This operation is live - mark its inputs as live
                    for operand in op.operands {
                        if !liveSSAs.contains(operand) {
                            liveSSAs.insert(operand)
                            changed = true
                        }
                    }
                }
            }
        }

        // Filter to only live operations
        return renamedOps.filter { liveSSAs.contains($0.result) }
    }

    // MARK: - Constant Folding

    /// Fold constant expressions at compile time
    /// e.g., constant(1.0) * constant(2.0) -> constant(2.0)
    private func foldConstants(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap
        var constantValues: [String: Double] = [:]  // SSA -> constant value

        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            // Track constant values
            if renamedOp.opName == "constant" || renamedOp.opName == "stablehlo.constant" {
                if let value = extractConstantValue(renamedOp.attributes["value"] ?? "") {
                    constantValues[renamedOp.result] = value
                }
                result.append(renamedOp)
                continue
            }

            // Try to fold binary operations on constants
            if renamedOp.operands.count == 2,
               let lhs = constantValues[renamedOp.operands[0]],
               let rhs = constantValues[renamedOp.operands[1]],
               let foldedValue = foldBinaryOp(renamedOp.opName, lhs: lhs, rhs: rhs) {
                // Create a new constant operation
                let newConstOp = MLIROperation(
                    result: renamedOp.result,
                    opName: "constant",
                    operands: [],
                    attributes: ["value": "dense<\(formatFloat(foldedValue))>"],
                    resultType: renamedOp.resultType,
                    regions: []
                )
                result.append(newConstOp)
                constantValues[renamedOp.result] = foldedValue
            } else {
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    /// Extract numeric value from constant attribute like "dense<1.000000e+00>"
    private func extractConstantValue(_ attr: String) -> Double? {
        // Handle patterns like:
        //   "dense<1.000000e+00>" (simple)
        //   "dense<1.000000e+00> : tensor<f32>" (with type annotation)
        guard attr.hasPrefix("dense<") else { return nil }

        // Find the closing > for dense<...>
        guard let closeIndex = attr.firstIndex(of: ">") else { return nil }

        // Extract value between dense< and >
        let startIndex = attr.index(attr.startIndex, offsetBy: 6) // after "dense<"
        let valueStr = String(attr[startIndex..<closeIndex])

        // Check if it's a scalar (not a tensor with brackets)
        if valueStr.contains("[") {
            return nil
        }

        return Double(valueStr)
    }

    /// Fold a binary operation on two constants
    private func foldBinaryOp(_ opName: String, lhs: Double, rhs: Double) -> Double? {
        switch opName {
        case "add", "stablehlo.add":
            return lhs + rhs
        case "subtract", "stablehlo.subtract":
            return lhs - rhs
        case "multiply", "stablehlo.multiply":
            return lhs * rhs
        case "divide", "stablehlo.divide":
            return rhs != 0 ? lhs / rhs : nil
        default:
            return nil
        }
    }

    /// Format a double for MLIR output
    private func formatFloat(_ value: Double) -> String {
        // Use scientific notation for consistency with MLIR
        return String(format: "%.6e", value)
    }

    // MARK: - Assignment Propagation

    /// Inline single-use variables to reduce overhead
    /// If %v5 is only used once, replace its use with its definition
    private func propagateAssignments(
        _ operations: [MLIROperation],
        ssaMap: [String: String],
        results: [String]
    ) -> ([MLIROperation], [String: String]) {
        // Count uses of each SSA value
        var useCounts: [String: Int] = [:]
        let renamedOps = operations.map { renameOperands($0, ssaMap: ssaMap) }

        // Count operand uses
        for op in renamedOps {
            for operand in op.operands {
                useCounts[operand, default: 0] += 1
            }
        }

        // Count result uses
        let canonicalResults = results.map { ssaMap[$0] ?? $0 }
        for r in canonicalResults {
            useCounts[r, default: 0] += 1
        }

        // Find single-use values that are simple (broadcasts, converts)
        var newSSAMap = ssaMap
        var result: [MLIROperation] = []

        // Operations that can be inlined (they just rename/reshape values)
        let inlinableOps = Set(["convert", "stablehlo.convert", "reshape", "stablehlo.reshape"])

        for op in renamedOps {
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            // Check if this op produces a value used only once and is inlinable
            if useCounts[renamedOp.result] == 1 &&
               inlinableOps.contains(renamedOp.opName) &&
               renamedOp.operands.count == 1 {
                // Map this result to its input (inline the operation)
                newSSAMap[renamedOp.result] = renamedOp.operands[0]
                // Don't add to result - it's inlined
            } else {
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    // MARK: - Strength Reduction

    /// Replace expensive operations with cheaper equivalents
    /// Currently a placeholder - MLIR/XLA already handles most of these
    private func reduceStrength(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        let newSSAMap = ssaMap

        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)
            // For now, just pass through - XLA handles strength reduction
            // Future: x**2 -> x*x, etc.
            result.append(renamedOp)
        }

        return (result, newSSAMap)
    }

    // MARK: - Algebraic Simplification

    /// Apply mathematical identities: x*1=x, x+0=x, x-0=x, x/1=x, x*0=0
    /// Handles broadcast(constant) patterns - crucial for gradient code
    private func simplifyAlgebraically(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap
        var constantValues: [String: Double] = [:]

        // First pass: collect constant values AND propagate through broadcasts
        // We need to apply existing ssaMap to operands but NOT to results
        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: ssaMap)  // Use original ssaMap, not newSSAMap

            if renamedOp.opName == "constant" || renamedOp.opName == "stablehlo.constant" {
                let attr = renamedOp.attributes["value"] ?? ""
                if let value = extractConstantValue(attr) {
                    constantValues[renamedOp.result] = value
                }
            }
            // Track broadcast_in_dim: propagate constant value through broadcast
            // This is crucial: %v5 = broadcast(%v4) where %v4 = 1.0 means %v5 is also 1.0
            if (renamedOp.opName == "broadcast_in_dim" || renamedOp.opName == "stablehlo.broadcast_in_dim"),
               renamedOp.operands.count == 1 {
                let source = renamedOp.operands[0]
                if let val = constantValues[source] {
                    constantValues[renamedOp.result] = val
                }
            }
        }

        // Second pass: apply simplifications
        // Use the SAME ssaMap as first pass to ensure operand names match constantValues keys
        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: ssaMap)

            // Skip constants
            if renamedOp.opName == "constant" || renamedOp.opName == "stablehlo.constant" {
                result.append(renamedOp)
                continue
            }

            // Check for algebraic identities with 2 operands
            if renamedOp.operands.count == 2 {
                let op0 = renamedOp.operands[0]
                let op1 = renamedOp.operands[1]
                let val0 = constantValues[op0]
                let val1 = constantValues[op1]

                switch renamedOp.opName {
                case "multiply", "stablehlo.multiply":
                    // x * 1 = x, 1 * x = x
                    if val1 == 1.0 {
                        newSSAMap[renamedOp.result] = op0
                        continue
                    }
                    if val0 == 1.0 {
                        newSSAMap[renamedOp.result] = op1
                        continue
                    }
                    // x * 0 = 0, 0 * x = 0
                    if val1 == 0.0 {
                        newSSAMap[renamedOp.result] = op1
                        continue
                    }
                    if val0 == 0.0 {
                        newSSAMap[renamedOp.result] = op0
                        continue
                    }

                case "add", "stablehlo.add":
                    // x + 0 = x, 0 + x = x
                    if val1 == 0.0 {
                        newSSAMap[renamedOp.result] = op0
                        continue
                    }
                    if val0 == 0.0 {
                        newSSAMap[renamedOp.result] = op1
                        continue
                    }

                case "subtract", "stablehlo.subtract":
                    // x - 0 = x
                    if val1 == 0.0 {
                        newSSAMap[renamedOp.result] = op0
                        continue
                    }

                case "divide", "stablehlo.divide":
                    // x / 1 = x
                    if val1 == 1.0 {
                        newSSAMap[renamedOp.result] = op0
                        continue
                    }

                default:
                    break
                }
            }

            // Check for unary identities
            if renamedOp.operands.count == 1 {
                let op0 = renamedOp.operands[0]

                switch renamedOp.opName {
                case "negate", "stablehlo.negate":
                    // -(-x) = x - check if operand is also a negate
                    // This would require tracking operation types, skip for now
                    break
                default:
                    break
                }
            }

            result.append(renamedOp)
        }

        return (result, newSSAMap)
    }
}

// MARK: - MLIRBuilder Extension

extension MLIRBuilder {
    /// Optimize operations before building the module
    /// This applies optimization passes based on the specified level
    /// - Parameters:
    ///   - functionName: Name of the MLIR function
    ///   - level: Optimization level (default: .standard)
    ///     - .none: No optimization (JAX-style, let XLA handle it)
    ///     - .minimal: Only DCE
    ///     - .reduced: CSE + DCE only
    ///     - .standard: Full 7-pass pipeline
    ///     - .maximum: Full pipeline with more iterations
    public func buildOptimized(
        functionName: String = "main",
        level: MLIROptimizationLevel = .standard
    ) -> MLIRModule {
        let optimizer = MLIROptimizer(level: level)

        // Get optimized operations
        let (optimizedOps, ssaMap) = optimizer.optimize(
            operations: operations,
            results: results
        )

        // Apply SSA renaming to results
        let optimizedResults = results.map { ssaMap[$0] ?? $0 }

        return MLIRModule(
            functionName: functionName,
            arguments: arguments,
            operations: optimizedOps,
            results: optimizedResults
        )
    }
}

// MARK: - Statistics

/// Statistics about MLIR optimization
public struct MLIROptimizationStats {
    public let originalOpCount: Int
    public let optimizedOpCount: Int
    public let constantsRemoved: Int
    public let cseDuplicatesRemoved: Int
    public let deadCodeRemoved: Int

    public var reductionPercentage: Double {
        guard originalOpCount > 0 else { return 0 }
        return Double(originalOpCount - optimizedOpCount) / Double(originalOpCount) * 100
    }

    public var description: String {
        """
        MLIR Optimization Stats:
          Original ops:     \(originalOpCount)
          Optimized ops:    \(optimizedOpCount)
          Reduction:        \(String(format: "%.1f", reductionPercentage))%
          Constants dedup:  \(constantsRemoved)
          CSE duplicates:   \(cseDuplicatesRemoved)
          Dead code:        \(deadCodeRemoved)
        """
    }
}

/// Optimizer with statistics tracking
public class MLIROptimizerWithStats {
    private var constantsRemoved = 0
    private var cseDuplicatesRemoved = 0
    private var deadCodeRemoved = 0

    public init() {}

    /// Optimize and return statistics
    public func optimizeWithStats(
        operations: [MLIROperation],
        results: [String]
    ) -> (operations: [MLIROperation], stats: MLIROptimizationStats) {
        let originalCount = operations.count
        var ops = operations
        var ssaMap: [String: String] = [:]

        // Pass 1: Constant deduplication
        let beforeConst = ops.count
        (ops, ssaMap) = deduplicateConstants(ops, ssaMap: ssaMap)
        constantsRemoved = beforeConst - ops.count

        // Pass 2: CSE
        let beforeCSE = ops.count
        (ops, ssaMap) = eliminateCommonSubexpressions(ops, ssaMap: ssaMap)
        cseDuplicatesRemoved = beforeCSE - ops.count

        // Pass 3: DCE
        let beforeDCE = ops.count
        ops = eliminateDeadCode(ops, results: results, ssaMap: ssaMap)
        deadCodeRemoved = beforeDCE - ops.count

        let stats = MLIROptimizationStats(
            originalOpCount: originalCount,
            optimizedOpCount: ops.count,
            constantsRemoved: constantsRemoved,
            cseDuplicatesRemoved: cseDuplicatesRemoved,
            deadCodeRemoved: deadCodeRemoved
        )

        return (ops, stats)
    }

    // Same implementation as MLIROptimizer but with stats tracking
    private func renameOperands(_ op: MLIROperation, ssaMap: [String: String]) -> MLIROperation {
        let renamedOperands = op.operands.map { ssaMap[$0] ?? $0 }
        return MLIROperation(
            result: op.result,
            opName: op.opName,
            operands: renamedOperands,
            attributes: op.attributes,
            resultType: op.resultType,
            regions: op.regions
        )
    }

    private func deduplicateConstants(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap
        var constantMap: [String: String] = [:]

        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            if renamedOp.opName == "constant" || renamedOp.opName == "stablehlo.constant" {
                let valueAttr = renamedOp.attributes["value"] ?? ""
                let key = "\(valueAttr)|\(renamedOp.resultType)"

                if let existingSSA = constantMap[key] {
                    newSSAMap[renamedOp.result] = existingSSA
                } else {
                    constantMap[key] = renamedOp.result
                    result.append(renamedOp)
                }
            } else {
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    private func eliminateCommonSubexpressions(
        _ operations: [MLIROperation],
        ssaMap: [String: String]
    ) -> ([MLIROperation], [String: String]) {
        var result: [MLIROperation] = []
        var newSSAMap = ssaMap
        var expressionMap: [String: String] = [:]

        for op in operations {
            let renamedOp = renameOperands(op, ssaMap: newSSAMap)

            if !renamedOp.regions.isEmpty || isSideEffecting(renamedOp.opName) {
                result.append(renamedOp)
                continue
            }

            let key = canonicalKey(for: renamedOp)

            if let existingSSA = expressionMap[key] {
                newSSAMap[renamedOp.result] = existingSSA
            } else {
                expressionMap[key] = renamedOp.result
                result.append(renamedOp)
            }
        }

        return (result, newSSAMap)
    }

    private func canonicalKey(for op: MLIROperation) -> String {
        let sortedAttrs = op.attributes
            .filter { !$0.key.hasPrefix("_") }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ",")

        return "\(op.opName)|\(op.operands.joined(separator: ","))|\(sortedAttrs)|\(op.resultType)"
    }

    private func isSideEffecting(_ opName: String) -> Bool {
        let sideEffectingOps = Set([
            "infeed", "outfeed", "send", "recv", "all_reduce", "all_gather",
            "stablehlo.infeed", "stablehlo.outfeed", "stablehlo.send", "stablehlo.recv"
        ])
        return sideEffectingOps.contains(opName)
    }

    private func eliminateDeadCode(
        _ operations: [MLIROperation],
        results: [String],
        ssaMap: [String: String]
    ) -> [MLIROperation] {
        let canonicalResults = results.map { ssaMap[$0] ?? $0 }
        var liveSSAs = Set(canonicalResults)
        var changed = true

        let renamedOps = operations.map { renameOperands($0, ssaMap: ssaMap) }

        while changed {
            changed = false
            for op in renamedOps.reversed() {
                if liveSSAs.contains(op.result) {
                    for operand in op.operands {
                        if !liveSSAs.contains(operand) {
                            liveSSAs.insert(operand)
                            changed = true
                        }
                    }
                }
            }
        }

        return renamedOps.filter { liveSSAs.contains($0.result) }
    }
}
