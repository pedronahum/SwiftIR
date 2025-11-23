import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros
import SwiftDiagnostics
import Foundation

/// The @MLIRFunction macro transforms Swift functions into MLIR representations.
///
/// This macro acts as a peer macro that generates a companion function returning the MLIR IR
public struct MLIRFunctionMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        // Only works on function declarations
        guard let funcDecl = declaration.as(FunctionDeclSyntax.self) else {
            context.diagnose(
                Diagnostic(
                    node: node,
                    message: MLIRDiagnostic.notAFunction
                )
            )
            return []
        }

        // Extract function information
        let funcName = funcDecl.name.text
        let mlirFuncName = "\(funcName)_mlir"

        // Get source location for MLIR
        let sourceFile = context.location(of: funcDecl)?.file.description ?? "unknown"
        let lineExpr = context.location(of: funcDecl)?.line
        let line = lineExpr?.description ?? "0"

        // Generate MLIR representation
        let mlirIR = try generateMLIRForFunction(funcDecl, sourceFile: sourceFile, line: line)

        // Create a peer function that returns the MLIR representation
        let peerFunction = try FunctionDeclSyntax(
            "func \(raw: mlirFuncName)() -> String"
        ) {
            """
            return \"\"\"
            \(raw: mlirIR)
            \"\"\"
            """
        }

        return [DeclSyntax(peerFunction)]
    }

    /// Generates MLIR IR for a Swift function declaration.
    private static func generateMLIRForFunction(
        _ funcDecl: FunctionDeclSyntax,
        sourceFile: String,
        line: String
    ) throws -> String {
        let funcName = funcDecl.name.text

        // Extract parameters
        let params = funcDecl.signature.parameterClause.parameters
        let mlirParams = params.map { param -> String in
            let paramName = param.secondName?.text ?? param.firstName.text
            let paramType = param.type.description.trimmingCharacters(in: .whitespaces)
            // Map Swift types to MLIR types (simplified for now)
            let mlirType = swiftTypeToMLIRType(paramType)
            return "%\(paramName): \(mlirType)"
        }.joined(separator: ", ")

        // Determine return type
        let returnType: String
        if let returnClause = funcDecl.signature.returnClause {
            let swiftReturnType = returnClause.type.description.trimmingCharacters(in: .whitespaces)
            returnType = swiftTypeToMLIRType(swiftReturnType)
        } else {
            returnType = "()"  // Void/Unit type in MLIR
        }

        // Generate MLIR function
        var mlir = """
        // Generated from Swift function '\(funcName)' at \(sourceFile):\(line)
        func.func @\(funcName)(\(mlirParams)) -> \(returnType) {
        """

        // Process function body (simplified - just add a placeholder for now)
        if let body = funcDecl.body {
            mlir += "\n  // Function body with \(body.statements.count) statement(s)"
            // For now, just return a placeholder
            if returnType != "()" {
                mlir += "\n  // TODO: Process function body"
                mlir += "\n  %0 = arith.constant 0 : \(returnType)"
                mlir += "\n  return %0 : \(returnType)"
            } else {
                mlir += "\n  return"
            }
        } else {
            mlir += "\n  // External function (no body)"
            if returnType != "()" {
                mlir += "\n  %0 = arith.constant 0 : \(returnType)"
                mlir += "\n  return %0 : \(returnType)"
            } else {
                mlir += "\n  return"
            }
        }

        mlir += "\n}"

        return mlir
    }

    /// Maps Swift types to MLIR types (simplified mapping).
    private static func swiftTypeToMLIRType(_ swiftType: String) -> String {
        switch swiftType {
        case "Int", "Int64":
            return "i64"
        case "Int32":
            return "i32"
        case "Int16":
            return "i16"
        case "Int8":
            return "i8"
        case "UInt", "UInt64":
            return "ui64"
        case "UInt32":
            return "ui32"
        case "Float":
            return "f32"
        case "Double":
            return "f64"
        case "Bool":
            return "i1"
        case "()":
            return "()"
        default:
            // For unknown types, use a generic memref
            return "memref<?xi8>"
        }
    }
}

/// Diagnostic messages for the MLIR function macro.
enum MLIRDiagnostic: String, DiagnosticMessage {
    case notAFunction

    var message: String {
        switch self {
        case .notAFunction:
            return "@MLIRFunction can only be applied to function declarations"
        }
    }

    var diagnosticID: MessageID {
        MessageID(domain: "S4MLIRMacros", id: rawValue)
    }

    var severity: DiagnosticSeverity {
        .error
    }
}

@main
struct S4MLIRPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        MLIRFunctionMacro.self,
    ]
}
