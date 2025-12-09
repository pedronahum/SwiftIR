/// SDY Opt Runner for SwiftIR
/// Provides external process-based sharding propagation using sdy_opt.

import Foundation

// MARK: - SDY Opt Runner

/// Runs sdy_opt as an external process for sharding propagation.
///
/// This class provides a way to run Shardy's propagation passes without
/// requiring full C API integration. It works by:
/// 1. Writing MLIR text to a temporary file
/// 2. Running sdy_opt with appropriate passes
/// 3. Capturing the output
///
/// Example:
/// ```swift
/// let runner = SdyOptRunner()
/// if let result = runner.runPropagation(on: moduleText) {
///     print(result)
/// }
/// ```
public class SdyOptRunner {
    /// Path to the sdy_opt binary
    public let sdyOptPath: String

    /// Whether sdy_opt is available
    public var isAvailable: Bool {
        FileManager.default.fileExists(atPath: sdyOptPath)
    }

    /// Creates a runner with the default sdy_opt path from Shardy build.
    ///
    /// - Parameter path: Optional custom path to sdy_opt binary
    public init(path: String? = nil) {
        if let customPath = path {
            self.sdyOptPath = customPath
        } else {
            // Try common locations
            let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
            let commonPaths = [
                "\(homeDir)/programming/swift/shardy/bazel-bin/shardy/tools/sdy_opt",
                "/opt/swiftir-deps/bin/sdy_opt",
                "/usr/local/bin/sdy_opt",
            ]

            self.sdyOptPath = commonPaths.first { FileManager.default.fileExists(atPath: $0) }
                ?? commonPaths[0]
        }
    }

    /// Result of running sdy_opt
    public struct RunResult {
        /// Whether the run succeeded
        public let success: Bool

        /// The output MLIR text (on success)
        public let output: String?

        /// Error message (on failure)
        public let error: String?

        /// Exit code from the process
        public let exitCode: Int32
    }

    /// Runs sdy_opt with the given passes on the input MLIR.
    ///
    /// - Parameters:
    ///   - input: The MLIR module text
    ///   - passes: The passes to run (default: import and propagation)
    /// - Returns: The result of running sdy_opt
    public func run(
        input: String,
        passes: [String] = ["sdy-add-data-flow-edges", "sdy-propagation"]
    ) -> RunResult {
        guard isAvailable else {
            return RunResult(
                success: false,
                output: nil,
                error: "sdy_opt not found at \(sdyOptPath)",
                exitCode: -1
            )
        }

        // Create temporary file for input
        let tempDir = FileManager.default.temporaryDirectory
        let inputFile = tempDir.appendingPathComponent("sdy_input_\(UUID().uuidString).mlir")

        do {
            try input.write(to: inputFile, atomically: true, encoding: .utf8)
        } catch {
            return RunResult(
                success: false,
                output: nil,
                error: "Failed to write input file: \(error)",
                exitCode: -1
            )
        }

        defer {
            try? FileManager.default.removeItem(at: inputFile)
        }

        // Run sdy_opt
        let process = Process()
        process.executableURL = URL(fileURLWithPath: sdyOptPath)
        process.arguments = passes.map { "--\($0)" } + [inputFile.path]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return RunResult(
                success: false,
                output: nil,
                error: "Failed to run sdy_opt: \(error)",
                exitCode: -1
            )
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""

        if process.terminationStatus == 0 {
            return RunResult(
                success: true,
                output: stdout,
                error: nil,
                exitCode: process.terminationStatus
            )
        } else {
            return RunResult(
                success: false,
                output: nil,
                error: stderr.isEmpty ? "sdy_opt failed with exit code \(process.terminationStatus)" : stderr,
                exitCode: process.terminationStatus
            )
        }
    }

    /// Runs the full SDY propagation pipeline on a module.
    ///
    /// This runs the complete import → propagation → export pipeline.
    ///
    /// - Parameter moduleText: The MLIR module text with initial shardings
    /// - Returns: The module text with propagated shardings, or nil on failure
    public func runPropagation(on moduleText: String) -> String? {
        let result = run(
            input: moduleText,
            passes: ["sdy-add-data-flow-edges", "sdy-propagation"]
        )
        return result.success ? result.output : nil
    }

    /// Runs only the import pipeline on a module.
    ///
    /// This prepares the module for propagation without running it.
    ///
    /// - Parameter moduleText: The MLIR module text
    /// - Returns: The prepared module text, or nil on failure
    public func runImport(on moduleText: String) -> String? {
        let result = run(
            input: moduleText,
            passes: ["sdy-add-data-flow-edges"]
        )
        return result.success ? result.output : nil
    }
}

// MARK: - ShardingPipeline Extension

extension ShardingPipeline {
    /// Runs propagation using sdy_opt external process.
    ///
    /// This method provides full sharding propagation by invoking sdy_opt
    /// as an external process. Use this when the C API is not available.
    ///
    /// - Parameters:
    ///   - moduleText: The MLIR module text with initial shardings
    ///   - sdyOptPath: Optional custom path to sdy_opt binary
    /// - Returns: The module text with propagated shardings, or nil on failure
    public func runPropagationWithSdyOpt(
        moduleText: String,
        sdyOptPath: String? = nil
    ) -> String? {
        // First, prepare the module with mesh definitions
        guard let prepared = runImportPipeline(moduleText: moduleText) else {
            return nil
        }

        // Wrap in a module if not already
        let fullModule: String
        if prepared.contains("module {") {
            fullModule = prepared
        } else {
            fullModule = "module {\n\(prepared)\n}"
        }

        // Run sdy_opt
        let runner = SdyOptRunner(path: sdyOptPath)
        return runner.runPropagation(on: fullModule)
    }

    /// Checks if sdy_opt is available for propagation.
    ///
    /// - Parameter path: Optional custom path to check
    /// - Returns: true if sdy_opt is available
    public static func isSdyOptAvailable(at path: String? = nil) -> Bool {
        let runner = SdyOptRunner(path: path)
        return runner.isAvailable
    }
}
