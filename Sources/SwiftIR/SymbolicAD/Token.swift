/// Token - Side-effect ordering for deterministic execution
/// Part of SwiftIR Symbolic Pullback Tracing system

import Foundation
import SwiftIRCore

/// Represents a sequencing token for side-effecting operations
/// Tokens enforce execution ordering in the compiled graph
///
/// XLA/StableHLO uses tokens to ensure deterministic ordering of side effects.
/// Operations like print, RNG, and collective communications must be sequenced
/// to ensure reproducible results.
///
/// Example:
/// ```swift
/// var token = Token.global
/// let (x1, token1) = x.print(label: "first", after: token)
/// let (x2, token2) = y.print(label: "second", after: token1)
/// // "first" is guaranteed to print before "second"
/// ```
public struct Token: Hashable {
    /// Internal identifier for the token
    internal let id: UInt64

    /// The IR value representing this token (for MLIR generation)
    internal var irValue: MLIRValue?

    /// Create a token with a specific ID
    internal init(id: UInt64, irValue: MLIRValue? = nil) {
        self.id = id
        self.irValue = irValue
    }

    /// The global token that sequences all side effects
    /// Use this as the starting token for any chain of side-effecting operations
    public static var global: Token {
        return TokenManager.shared.globalToken
    }

    /// Create a new token (used internally by tokenized operations)
    internal static func create() -> Token {
        return TokenManager.shared.createToken()
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    public static func == (lhs: Token, rhs: Token) -> Bool {
        return lhs.id == rhs.id
    }
}

/// Placeholder for MLIR Value - will be connected to actual MLIR infrastructure
/// This represents an SSA value in the MLIR graph
public struct MLIRValue: Hashable {
    internal let id: UInt64

    internal init(id: UInt64) {
        self.id = id
    }
}

// MARK: - Token Manager

/// Manages token creation and tracking
/// Thread-safe singleton for generating unique token IDs
internal final class TokenManager: @unchecked Sendable {
    nonisolated(unsafe) static let shared = TokenManager()

    private var nextId: UInt64 = 1
    private let lock = NSLock()

    /// The global initial token
    let globalToken: Token

    private init() {
        // Token 0 is reserved for the global token
        self.globalToken = Token(id: 0)
    }

    /// Create a new unique token
    func createToken() -> Token {
        lock.lock()
        defer { lock.unlock() }

        let id = nextId
        nextId += 1
        return Token(id: id)
    }

    /// Reset the token manager (for testing)
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        nextId = 1
    }
}

// MARK: - Token Chain Builder

/// A builder for constructing chains of side-effecting operations
/// Maintains the current token and threads it through operations automatically
public class TokenizedGraphBuilder {
    private var currentToken: Token

    /// Create a new builder starting from the global token
    public init() {
        self.currentToken = Token.global
    }

    /// Create a builder starting from a specific token
    public init(startingFrom token: Token) {
        self.currentToken = token
    }

    /// Get the current token in the chain
    public var token: Token { currentToken }

    /// Execute a side-effecting operation and update the token
    /// - Parameter operation: A closure that takes the current token and returns result + next token
    /// - Returns: The result of the operation
    public func execute<T>(_ operation: (Token) -> (T, Token)) -> T {
        let (result, nextToken) = operation(currentToken)
        currentToken = nextToken
        return result
    }

    /// Build a graph with guaranteed side-effect ordering
    /// - Parameter closure: A closure that builds the graph using this builder
    /// - Returns: The result of the closure and the final token
    public func build<T>(_ closure: (TokenizedGraphBuilder) -> T) -> (result: T, finalToken: Token) {
        let result = closure(self)
        return (result, currentToken)
    }
}

// MARK: - Protocol for Side-Effecting Operations

/// Protocol for operations that have side effects and require token chaining
public protocol SideEffectingOperation {
    associatedtype Result

    /// Execute the operation after a token, producing a result and next token
    func execute(after token: Token) -> (result: Result, nextToken: Token)
}

// MARK: - Debug Extensions

public extension Token {
    /// A string description of the token for debugging
    var debugDescription: String {
        if id == 0 {
            return "Token(global)"
        }
        return "Token(\(id))"
    }
}
