// RuntimeDetectorTests.swift - Tests for RuntimeDetector
// Copyright 2024 SwiftIR Project

import XCTest
@testable import SwiftIRRuntime

final class RuntimeDetectorTests: XCTestCase {

    // MARK: - Detection Tests

    func testDetectReturnsValidType() {
        let result = RuntimeDetector.detect()
        XCTAssertTrue(AcceleratorType.allCases.contains(result))
    }

    func testDetectDefaultsToCPU() {
        // On a machine without GPU/TPU, should return CPU
        // (This test will pass on most CI systems)
        let result = RuntimeDetector.detect()
        // At minimum, CPU should be a valid fallback
        XCTAssertNotNil(result)
    }

    // MARK: - TPU Detection Tests

    func testIsTPUAvailable() {
        // This is a simple availability check
        // Will return false on non-TPU machines
        let available = RuntimeDetector.isTPUAvailable()
        XCTAssertNotNil(available) // Just verify it doesn't crash
    }

    func testCountTPUCores() {
        // On non-TPU machines, should return nil
        if !RuntimeDetector.isTPUAvailable() {
            XCTAssertNil(RuntimeDetector.countTPUCores())
        }
    }

    // MARK: - GPU Detection Tests

    func testIsGPUAvailable() {
        // This is a simple availability check
        let available = RuntimeDetector.isGPUAvailable()
        XCTAssertNotNil(available) // Just verify it doesn't crash
    }

    // MARK: - Plugin Path Tests

    func testGetSearchPathsReturnsNonEmpty() {
        for accelerator in AcceleratorType.allCases {
            let paths = RuntimeDetector.getSearchPaths(for: accelerator)
            XCTAssertFalse(paths.isEmpty, "Search paths for \(accelerator) should not be empty")
        }
    }

    func testFindPluginPathForCPU() {
        // CPU plugin should be found on systems with SwiftIR installed
        let path = RuntimeDetector.findPluginPath(for: .cpu)
        // May be nil if not installed, but should not crash
        if let path = path {
            XCTAssertTrue(path.hasSuffix(".so") || path.hasSuffix(".dylib"))
        }
    }

    func testSearchPathsContainEnvironmentOverrides() {
        // Verify that environment variable paths are checked
        let cpuPaths = RuntimeDetector.getSearchPaths(for: .cpu)
        // The implementation should check PJRT_CPU_PLUGIN_PATH

        let gpuPaths = RuntimeDetector.getSearchPaths(for: .gpu)
        // The implementation should check PJRT_GPU_PLUGIN_PATH

        let tpuPaths = RuntimeDetector.getSearchPaths(for: .tpu)
        // The implementation should check TPU_LIBRARY_PATH

        // These won't contain the env values unless set, but should not be empty
        XCTAssertFalse(cpuPaths.isEmpty)
        XCTAssertFalse(gpuPaths.isEmpty)
        XCTAssertFalse(tpuPaths.isEmpty)
    }

    // MARK: - Runtime Info Tests

    func testGetRuntimeInfo() {
        let info = RuntimeDetector.getRuntimeInfo()

        // Verify all fields are populated
        XCTAssertNotNil(info.detectedAccelerator)
        // tpuAvailable and gpuAvailable are bools, always valid
        // tpuCoreCount may be nil
        // environmentVariables may be empty
        XCTAssertNotNil(info.detectedPluginPaths) // May be empty dict
    }

    func testRuntimeInfoSummary() {
        let info = RuntimeDetector.getRuntimeInfo()
        let summary = info.summary

        XCTAssertFalse(summary.isEmpty)
        XCTAssertTrue(summary.contains("SwiftIR Runtime Info"))
        XCTAssertTrue(summary.contains("Detected accelerator"))
        XCTAssertTrue(summary.contains("TPU available"))
        XCTAssertTrue(summary.contains("GPU available"))
    }

    func testRuntimeInfoDetectedAcceleratorConsistent() {
        let info = RuntimeDetector.getRuntimeInfo()
        let detected = RuntimeDetector.detect()

        // The detected accelerator in info should match direct detection
        XCTAssertEqual(info.detectedAccelerator, detected)
    }

    // MARK: - Priority Tests

    func testDetectionPriorityWithoutOverride() {
        // Without environment override, detection follows TPU > GPU > CPU
        let detected = RuntimeDetector.detect()

        if RuntimeDetector.isTPUAvailable() {
            XCTAssertEqual(detected, .tpu)
        } else if RuntimeDetector.isGPUAvailable() {
            XCTAssertEqual(detected, .gpu)
        } else {
            XCTAssertEqual(detected, .cpu)
        }
    }
}
