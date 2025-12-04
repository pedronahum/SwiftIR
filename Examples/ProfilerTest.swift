// ProfilerTest.swift - Test of SwiftIR TSL profiling functionality
//
// This example demonstrates the TSL-based profiler API:
// 1. Named scopes (TraceMes)
// 2. XSpace data collection
// 3. TensorBoard-compatible export
//
// Run with:
//   SWIFTIR_DEPS=/opt/swiftir-deps LD_LIBRARY_PATH=/opt/swiftir-deps/lib swift run ProfilerTest

import SwiftIRProfiler
import Foundation

// MARK: - Helper Functions

/// Simulate some computation
func heavyComputation(iterations: Int) -> Double {
    var sum = 0.0
    for i in 0..<iterations {
        sum += sin(Double(i)) * cos(Double(i))
    }
    return sum
}

/// Simulate matrix-like operation
func matrixOperation(size: Int) -> [[Double]] {
    var result = [[Double]](repeating: [Double](repeating: 0.0, count: size), count: size)
    for i in 0..<size {
        for j in 0..<size {
            result[i][j] = Double(i * j) / Double(size * size)
        }
    }
    return result
}

// MARK: - Main Test

@main
struct ProfilerTest {
    static func main() throws {
        print("SwiftIR TSL Profiler Test")
        print("=========================\n")

        // Initialize the profiler library
        print("Loading profiler library...")
        try Profiler.initialize()

        print("Profiler version: \(Profiler.version)")
        print("TSL available: \(Profiler.isTSLAvailable)\n")

        // MARK: - Test 1: Named Scopes (TraceMe)

        print("Test 1: Named Scopes (TraceMe)")
        print("------------------------------")

        // Use named scopes for hierarchical profiling
        Profiler.scope("outer_scope") {
            print("In outer scope...")

            Profiler.scope("inner_computation") {
                _ = heavyComputation(iterations: 50_000)
            }

            Profiler.scope("inner_matrix") {
                _ = matrixOperation(size: 50)
            }
        }

        print("Named scopes completed.\n")

        // MARK: - Test 2: Using traced() Helper

        print("Test 2: traced() Helper Function")
        print("--------------------------------")

        let result = traced("traced_computation") {
            heavyComputation(iterations: 25_000)
        }
        print("Traced computation result: \(result)")

        // MARK: - Test 3: ScopedTrace class

        print("\nTest 3: ScopedTrace class")
        print("-------------------------")

        do {
            let trace = ScopedTrace("scoped_trace_test")
            _ = heavyComputation(iterations: 10_000)
            // trace ends when it goes out of scope
            _ = trace  // silence unused warning
        }
        print("ScopedTrace completed.\n")

        // MARK: - Test 4: Full Profiler Session with XSpace Export

        print("Test 4: Full Profiler Session")
        print("-----------------------------")

        let session = try ProfilerSession()
        try session.start()

        // Do some work that will be captured
        traced("profiled_heavy_computation") {
            _ = heavyComputation(iterations: 100_000)
        }

        traced("profiled_matrix") {
            _ = matrixOperation(size: 100)
        }

        // Collect data
        let data = try session.collectData()
        print("Collected \(data.count) bytes of XSpace data")

        // Export to file
        let logDir = "/tmp/swiftir_profiler_test"
        try FileManager.default.createDirectory(
            atPath: logDir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let timestamp = Int(Date().timeIntervalSince1970 * 1000)
        let filepath = "\(logDir)/swiftir.\(timestamp).xspace.pb"

        let bindings = ProfilerBindings.shared
        try bindings.exportToFile(data, filepath: filepath)
        print("Exported XSpace to: \(filepath)")
        print("You can view this in TensorBoard with: tensorboard --logdir=\(logDir)")

        // MARK: - Done

        print("\n=========================")
        print("All tests completed!")
        print("=========================")
    }
}
