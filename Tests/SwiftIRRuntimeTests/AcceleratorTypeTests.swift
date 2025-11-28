// AcceleratorTypeTests.swift - Tests for AcceleratorType enum
// Copyright 2024 SwiftIR Project

import XCTest
@testable import SwiftIRRuntime

final class AcceleratorTypeTests: XCTestCase {

    // MARK: - Raw Value Tests

    func testRawValues() {
        XCTAssertEqual(AcceleratorType.cpu.rawValue, "CPU")
        XCTAssertEqual(AcceleratorType.gpu.rawValue, "GPU")
        XCTAssertEqual(AcceleratorType.tpu.rawValue, "TPU")
    }

    func testInitFromRawValue() {
        XCTAssertEqual(AcceleratorType(rawValue: "CPU"), .cpu)
        XCTAssertEqual(AcceleratorType(rawValue: "GPU"), .gpu)
        XCTAssertEqual(AcceleratorType(rawValue: "TPU"), .tpu)
        XCTAssertNil(AcceleratorType(rawValue: "INVALID"))
    }

    // MARK: - Description Tests

    func testDescription() {
        XCTAssertEqual(AcceleratorType.cpu.description, "CPU")
        XCTAssertEqual(AcceleratorType.gpu.description, "GPU")
        XCTAssertEqual(AcceleratorType.tpu.description, "TPU")
    }

    // MARK: - Plugin Name Tests

    func testPluginNames() {
        XCTAssertEqual(AcceleratorType.cpu.pluginName, "pjrt_c_api_cpu_plugin.so")
        XCTAssertEqual(AcceleratorType.gpu.pluginName, "xla_cuda_plugin.so")
        XCTAssertEqual(AcceleratorType.tpu.pluginName, "libtpu.so")
    }

    // MARK: - Display Name Tests

    func testDisplayNames() {
        XCTAssertEqual(AcceleratorType.cpu.displayName, "CPU")
        XCTAssertEqual(AcceleratorType.gpu.displayName, "NVIDIA GPU (CUDA)")
        XCTAssertEqual(AcceleratorType.tpu.displayName, "Google TPU")
    }

    // MARK: - Requires Special Hardware Tests

    func testRequiresSpecialHardware() {
        XCTAssertFalse(AcceleratorType.cpu.requiresSpecialHardware)
        XCTAssertTrue(AcceleratorType.gpu.requiresSpecialHardware)
        XCTAssertTrue(AcceleratorType.tpu.requiresSpecialHardware)
    }

    // MARK: - Equatable Tests

    func testEquatable() {
        XCTAssertEqual(AcceleratorType.cpu, AcceleratorType.cpu)
        XCTAssertNotEqual(AcceleratorType.cpu, AcceleratorType.gpu)
        XCTAssertNotEqual(AcceleratorType.gpu, AcceleratorType.tpu)
    }

    // MARK: - Hashable Tests

    func testHashable() {
        var set = Set<AcceleratorType>()
        set.insert(.cpu)
        set.insert(.gpu)
        set.insert(.tpu)
        set.insert(.cpu) // Duplicate

        XCTAssertEqual(set.count, 3)
        XCTAssertTrue(set.contains(.cpu))
        XCTAssertTrue(set.contains(.gpu))
        XCTAssertTrue(set.contains(.tpu))
    }

    // MARK: - Codable Tests

    func testCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Encode
        let original = AcceleratorType.gpu
        let data = try encoder.encode(original)

        // Decode
        let decoded = try decoder.decode(AcceleratorType.self, from: data)

        XCTAssertEqual(original, decoded)
    }

    func testCodableAllCases() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for accelerator in AcceleratorType.allCases {
            let data = try encoder.encode(accelerator)
            let decoded = try decoder.decode(AcceleratorType.self, from: data)
            XCTAssertEqual(accelerator, decoded)
        }
    }

    // MARK: - CaseIterable Tests

    func testAllCases() {
        let allCases = AcceleratorType.allCases
        XCTAssertEqual(allCases.count, 3)
        XCTAssertTrue(allCases.contains(.cpu))
        XCTAssertTrue(allCases.contains(.gpu))
        XCTAssertTrue(allCases.contains(.tpu))
    }
}
