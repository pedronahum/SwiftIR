@_exported import SwiftIRCore

/// S4MLIR - Swift for MLIR
/// A modern metaprogramming system for MLIR using Swift
public struct S4MLIR {
    public static let version = "0.1.0"

    public init() {}
}

/// Transform a Swift function into MLIR
@attached(peer, names: arbitrary)
public macro MLIRFunction() = #externalMacro(
    module: "SwiftIRMacros",
    type: "MLIRFunctionMacro"
)
