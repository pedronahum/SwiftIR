// PJRTClientFactoryTests.swift - Tests for PJRTClientFactory
// Copyright 2024 SwiftIR Project

import XCTest
@testable import SwiftIRRuntime

final class PJRTClientFactoryTests: XCTestCase {

    // MARK: - Detection Tests

    func testDetectedAccelerator() {
        let detected = PJRTClientFactory.detectedAccelerator
        XCTAssertTrue(AcceleratorType.allCases.contains(detected))
    }

    func testRuntimeInfo() {
        let info = PJRTClientFactory.runtimeInfo
        XCTAssertNotNil(info.detectedAccelerator)
    }

    // MARK: - Availability Tests

    func testIsAvailableCPU() {
        // CPU availability depends on plugin being installed
        let available = PJRTClientFactory.isAvailable(.cpu)
        // Just verify it doesn't crash
        XCTAssertNotNil(available)
    }

    func testIsAvailableGPU() {
        let available = PJRTClientFactory.isAvailable(.gpu)
        // On non-GPU machines, should return false
        if !RuntimeDetector.isGPUAvailable() {
            XCTAssertFalse(available)
        }
    }

    func testIsAvailableTPU() {
        let available = PJRTClientFactory.isAvailable(.tpu)
        // On non-TPU machines, should return false
        if !RuntimeDetector.isTPUAvailable() {
            XCTAssertFalse(available)
        }
    }

    func testAvailableAccelerators() {
        let available = PJRTClientFactory.availableAccelerators
        // Should be a subset of all cases
        for acc in available {
            XCTAssertTrue(AcceleratorType.allCases.contains(acc))
        }
    }

    // MARK: - Create Tests (may fail without plugins installed)

    func testCreateTPUWithoutTPU() {
        // On non-TPU machines, creating TPU client should throw
        guard !RuntimeDetector.isTPUAvailable() else {
            // Skip test on TPU machines
            return
        }

        XCTAssertThrowsError(try PJRTClientFactory.createTPU()) { error in
            guard let swiftIRError = error as? SwiftIRError else {
                XCTFail("Expected SwiftIRError")
                return
            }

            switch swiftIRError {
            case .acceleratorNotAvailable(let acc):
                XCTAssertEqual(acc, .tpu)
            case .pluginNotFound(let acc, _):
                XCTAssertEqual(acc, .tpu)
            default:
                // Other errors are also acceptable
                break
            }
        }
    }

    func testCreateGPUWithoutGPU() {
        // On non-GPU machines, creating GPU client should throw
        guard !RuntimeDetector.isGPUAvailable() else {
            // Skip test on GPU machines
            return
        }

        XCTAssertThrowsError(try PJRTClientFactory.createGPU()) { error in
            guard let swiftIRError = error as? SwiftIRError else {
                XCTFail("Expected SwiftIRError")
                return
            }

            switch swiftIRError {
            case .acceleratorNotAvailable(let acc):
                XCTAssertEqual(acc, .gpu)
            case .pluginNotFound(let acc, _):
                XCTAssertEqual(acc, .gpu)
            default:
                // Other errors are also acceptable
                break
            }
        }
    }
}
