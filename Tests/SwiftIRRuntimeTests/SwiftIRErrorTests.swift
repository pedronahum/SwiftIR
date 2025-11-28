// SwiftIRErrorTests.swift - Tests for SwiftIRError enum
// Copyright 2024 SwiftIR Project

import XCTest
@testable import SwiftIRRuntime

final class SwiftIRErrorTests: XCTestCase {

    // MARK: - Error Description Tests

    func testPluginNotFoundDescription() {
        let error = SwiftIRError.pluginNotFound(
            accelerator: .gpu,
            searchedPaths: ["/path/1", "/path/2"]
        )

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("GPU"))
        XCTAssertTrue(description.contains("/path/1"))
        XCTAssertTrue(description.contains("/path/2"))
    }

    func testPluginLoadFailedDescription() {
        let error = SwiftIRError.pluginLoadFailed(
            path: "/some/path.so",
            reason: "symbol not found"
        )

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("/some/path.so"))
        XCTAssertTrue(description.contains("symbol not found"))
    }

    func testSymbolNotFoundDescription() {
        let error = SwiftIRError.symbolNotFound(
            symbol: "GetPjrtApi",
            path: "/some/plugin.so"
        )

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("GetPjrtApi"))
        XCTAssertTrue(description.contains("/some/plugin.so"))
    }

    func testClientCreationFailedDescription() {
        let error = SwiftIRError.clientCreationFailed(
            accelerator: .tpu,
            reason: "device not initialized"
        )

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("TPU"))
        XCTAssertTrue(description.contains("device not initialized"))
    }

    func testAcceleratorNotAvailableDescription() {
        let error = SwiftIRError.acceleratorNotAvailable(.gpu)

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("GPU"))
        XCTAssertTrue(description.contains("not available"))
    }

    func testCompilationFailedDescription() {
        let error = SwiftIRError.compilationFailed(reason: "invalid MLIR")

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("compilation"))
        XCTAssertTrue(description.contains("invalid MLIR"))
    }

    func testExecutionFailedDescription() {
        let error = SwiftIRError.executionFailed(reason: "out of memory")

        let description = error.errorDescription!
        XCTAssertTrue(description.contains("execution"))
        XCTAssertTrue(description.contains("out of memory"))
    }

    // MARK: - Equatable Tests

    func testEquatablePluginNotFound() {
        let error1 = SwiftIRError.pluginNotFound(accelerator: .cpu, searchedPaths: ["/a"])
        let error2 = SwiftIRError.pluginNotFound(accelerator: .cpu, searchedPaths: ["/a"])
        let error3 = SwiftIRError.pluginNotFound(accelerator: .gpu, searchedPaths: ["/a"])

        XCTAssertEqual(error1, error2)
        XCTAssertNotEqual(error1, error3)
    }

    func testEquatablePluginLoadFailed() {
        let error1 = SwiftIRError.pluginLoadFailed(path: "/a", reason: "x")
        let error2 = SwiftIRError.pluginLoadFailed(path: "/a", reason: "x")
        let error3 = SwiftIRError.pluginLoadFailed(path: "/b", reason: "x")

        XCTAssertEqual(error1, error2)
        XCTAssertNotEqual(error1, error3)
    }

    func testEquatableAcceleratorNotAvailable() {
        let error1 = SwiftIRError.acceleratorNotAvailable(.tpu)
        let error2 = SwiftIRError.acceleratorNotAvailable(.tpu)
        let error3 = SwiftIRError.acceleratorNotAvailable(.gpu)

        XCTAssertEqual(error1, error2)
        XCTAssertNotEqual(error1, error3)
    }

    func testEquatableDifferentCases() {
        let error1 = SwiftIRError.acceleratorNotAvailable(.cpu)
        let error2 = SwiftIRError.compilationFailed(reason: "test")

        XCTAssertNotEqual(error1, error2)
    }

    // MARK: - LocalizedError Conformance

    func testLocalizedErrorConformance() {
        let errors: [SwiftIRError] = [
            .pluginNotFound(accelerator: .cpu, searchedPaths: []),
            .pluginLoadFailed(path: "/test", reason: "test"),
            .symbolNotFound(symbol: "test", path: "/test"),
            .clientCreationFailed(accelerator: .gpu, reason: "test"),
            .acceleratorNotAvailable(.tpu),
            .compilationFailed(reason: "test"),
            .executionFailed(reason: "test")
        ]

        for error in errors {
            // All errors should have a non-nil error description
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}
