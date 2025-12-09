/// SDY Opt Runner for SwiftIRShardingLite
/// Pure Swift implementation - runs sdy_opt as external process.

import Foundation

// MARK: - SDY Opt Runner

/// Runs sdy_opt as an external process for sharding propagation.
///
/// This class provides a way to run Shardy's propagation passes without
/// requiring full C API integration. It works by:
/// 1. Writing MLIR text to a temporary file
/// 2. Running sdy_opt with appropriate passes
/// 3. Capturing the output
public class SdyOptRunner {
    /// Path to the sdy_opt binary
    public let sdyOptPath: String

    /// Whether sdy_opt is available
    public var isAvailable: Bool {
        FileManager.default.fileExists(atPath: sdyOptPath)
    }

    /// Creates a runner with the default sdy_opt path from Shardy build.
    public init(path: String? = nil) {
        if let customPath = path {
            self.sdyOptPath = customPath
        } else {
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
        public let success: Bool
        public let output: String?
        public let error: String?
        public let exitCode: Int32
    }

    /// Runs sdy_opt with the given passes on the input MLIR.
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
    public func runPropagation(on moduleText: String) -> String? {
        let result = run(
            input: moduleText,
            passes: ["sdy-add-data-flow-edges", "sdy-propagation"]
        )
        return result.success ? result.output : nil
    }

    /// Runs only the import pipeline on a module.
    public func runImport(on moduleText: String) -> String? {
        let result = run(
            input: moduleText,
            passes: ["sdy-add-data-flow-edges"]
        )
        return result.success ? result.output : nil
    }
}
