// swift-tools-version: 6.0
// SwiftIR - Swift for Intermediate Representation
// Universal heterogeneous computing: ML (XLA) + Graphics (SPIR-V) + Custom accelerators

import PackageDescription
import CompilerPluginSupport
import Foundation

// MARK: - Path Configuration

// SDK mode: explicitly enabled via SWIFTIR_USE_SDK=1 (for Colab/deployed environments)
// Local dev mode: default behavior, uses local build paths
let swiftirDeps = ProcessInfo.processInfo.environment["SWIFTIR_DEPS"] ?? "/opt/swiftir-deps"
let useSDK = ProcessInfo.processInfo.environment["SWIFTIR_USE_SDK"] == "1"

// StableHLO/LLVM/MLIR paths
let stablehloRoot: String
let llvmBuildDir: String
let llvmProjectDir: String
let stablehloBuildDir: String

// SwiftIR build paths
let swiftIRRoot: String
let cmakeBuildDir: String

// XLA/PJRT paths
let xlaRoot: String
let xlaBaseliskBuildDir: String
let pjrtHeaderPath: String

// System library paths
let systemLibDir: String

if useSDK {
    // Use pre-built SDK (Colab, Linux CI, deployed environments)
    stablehloRoot = swiftirDeps
    llvmBuildDir = swiftirDeps
    llvmProjectDir = swiftirDeps
    stablehloBuildDir = swiftirDeps
    swiftIRRoot = "."
    cmakeBuildDir = swiftirDeps
    xlaRoot = swiftirDeps
    xlaBaseliskBuildDir = swiftirDeps
    pjrtHeaderPath = "\(swiftirDeps)/include"
    systemLibDir = "\(swiftirDeps)/lib"
} else {
    // Local development paths (macOS)
    stablehloRoot = "/Users/pedro/programming/swift/stablehlo"
    llvmBuildDir = "\(stablehloRoot)/llvm-build"
    llvmProjectDir = "\(stablehloRoot)/llvm-project"
    stablehloBuildDir = "\(stablehloRoot)/build"
    swiftIRRoot = "/Users/pedro/programming/swift/SwiftIR"
    cmakeBuildDir = "\(swiftIRRoot)/cmake/build"
    xlaRoot = "/Users/pedro/programming/swift/xla"
    xlaBaseliskBuildDir = "/Users/pedro/programming/swift/xla-bazelisk-build"
    pjrtHeaderPath = "\(xlaBaseliskBuildDir)/607423a1c455c14997634f7cf9a59c45/execroot/xla/bazel-out/darwin_arm64-opt/bin/xla/pjrt/c"
    systemLibDir = "/opt/homebrew/lib"  // Homebrew on macOS
}

// MARK: - Include Paths

let mlirIncludePaths: [String]
let stablehloIncludePaths: [String]

if useSDK {
    // SDK mode: all headers are flat in include/
    mlirIncludePaths = [
        "-I", "\(swiftirDeps)/include",
    ]
    stablehloIncludePaths = [
        "-I", "\(swiftirDeps)/include/stablehlo",
    ]
} else {
    // Local dev mode: headers in build tree structure
    mlirIncludePaths = [
        "-I", "\(llvmBuildDir)/include",
        "-I", "\(llvmBuildDir)/tools/mlir/include",
        "-I", "\(llvmProjectDir)/mlir/include",
    ]
    stablehloIncludePaths = [
        "-I", "\(stablehloRoot)/stablehlo",
        "-I", "\(stablehloBuildDir)/stablehlo",
    ]
}

let swiftIRIncludePaths = [
    "-I", "Sources/SwiftIRCore/include",
]

let pjrtIncludePaths = [
    "-I", "Sources/PJRTCWrappers/include",
    "-I", pjrtHeaderPath,
]

// MARK: - Linker Settings

let mlirLinkerFlags = [
    "-L\(cmakeBuildDir)/lib",
    "-lSwiftIRMLIR",
    "-L\(systemLibDir)",
    "-lzstd",
    "-lz",
    "-lcurses",
]

let pjrtLinkerFlags: [String]
if useSDK {
    pjrtLinkerFlags = [
        "-L\(systemLibDir)",
        "-lPJRTProtoHelper",
        "-Xlinker", "-rpath", "-Xlinker", systemLibDir,
    ]
} else {
    pjrtLinkerFlags = [
        "-Lcmake/build/lib",
        "-lPJRTProtoHelper",
        "-Xlinker", "-rpath", "-Xlinker", "@loader_path/../lib",
        "-Xlinker", "-rpath", "-Xlinker", "cmake/build/lib",
    ]
}

// MARK: - Combined Settings (for targets that need multiple includes)

let allSwiftIncludePaths = swiftIRIncludePaths + mlirIncludePaths
let allSwiftIncludePathsWithStableHLO = swiftIRIncludePaths + mlirIncludePaths + stablehloIncludePaths

// MARK: - Package Definition

let package = Package(
    name: "SwiftIR",
    platforms: [
        .macOS(.v14)  // Required by StableHLO LLVM/MLIR build
    ],
    products: [
        // Main unified API
        .library(
            name: "SwiftIR",
            targets: ["SwiftIR"]
        ),
        // Individual components (for advanced users)
        .library(
            name: "SwiftIRCore",
            targets: ["SwiftIRCore"]
        ),
        .library(
            name: "SwiftIRTypes",
            targets: ["SwiftIRTypes"]
        ),
        .library(
            name: "SwiftIRDialects",
            targets: ["SwiftIRDialects"]
        ),
        .library(
            name: "SwiftIRBuilders",
            targets: ["SwiftIRBuilders"]
        ),
        // Phase 7: Tensor operations for ML
        .library(
            name: "SwiftIRXLA",
            targets: ["SwiftIRXLA"]
        ),
        // Phase 10: StableHLO portable ML operations
        .library(
            name: "SwiftIRStableHLO",
            targets: ["SwiftIRStableHLO"]
        ),
        // Runtime detection and unified client API
        .library(
            name: "SwiftIRRuntime",
            targets: ["SwiftIRRuntime"]
        ),
        // Dynamic library for Jupyter/LLDB REPL usage
        .library(
            name: "SwiftIRRuntimeDynamic",
            type: .dynamic,
            targets: ["SwiftIRRuntime"]
        ),
        // Pure Swift Jupyter/Colab integration (no C++ interop)
        .library(
            name: "SwiftIRJupyter",
            targets: ["SwiftIRJupyter"]
        ),
        // Examples
        .executable(
            name: "SimpleNN",
            targets: ["SimpleNN"]
        ),
        .executable(
            name: "CNN",
            targets: ["CNN"]
        ),
        .executable(
            name: "PJRT_Add_Example",
            targets: ["PJRT_Add_Example"]
        ),
        .executable(
            name: "PJRT_MatMul_Example",
            targets: ["PJRT_MatMul_Example"]
        ),
        .executable(
            name: "Macro_Example",
            targets: ["Macro_Example"]
        ),
        .executable(
            name: "BuildingSimulation_Simple",
            targets: ["BuildingSimulation_Simple"]
        ),
        .executable(
            name: "BuildingSimulation_SwiftIR",
            targets: ["BuildingSimulation_SwiftIR"]
        ),
        .executable(
            name: "BuildingSimulation_StandardSwift",
            targets: ["BuildingSimulation_StandardSwift"]
        ),
        .executable(
            name: "JupyterTest",
            targets: ["JupyterTest"]
        ),
        .executable(
            name: "JupyterComprehensiveTest",
            targets: ["JupyterComprehensiveTest"]
        ),
        .executable(
            name: "JupyterBuildingSimulationTest",
            targets: ["JupyterBuildingSimulationTest"]
        ),
        .executable(
            name: "JupyterHighLevelTest",
            targets: ["JupyterHighLevelTest"]
        ),
        .executable(
            name: "JupyterBuildingSimulationFull",
            targets: ["JupyterBuildingSimulationFull"]
        ),
        .executable(
            name: "JupyterFeatureParityTest",
            targets: ["JupyterFeatureParityTest"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0"),
    ],
    targets: [
        // MARK: - Core MLIR Foundation

        // C wrappers for inline MLIR functions
        .target(
            name: "MLIRCoreWrappers",
            dependencies: [],
            path: "Sources/SwiftIRCore",
            sources: ["MLIRCoreWrappers.c"],
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(mlirIncludePaths),
            ]
        ),

        // PJRT C API simplified wrapper
        // Note: PJRTProtoHelper.dylib is built separately with Bazel and linked here
        .target(
            name: "PJRTCWrappers",
            dependencies: [],
            path: "Sources/PJRTCWrappers",
            sources: ["PJRTSimpleWrapper.c"],
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(pjrtIncludePaths),
            ],
            linkerSettings: [
                .unsafeFlags(pjrtLinkerFlags),
            ]
        ),

        .target(
            name: "SwiftIRCore",
            dependencies: ["MLIRCoreWrappers", "PJRTCWrappers"],
            path: "Sources/SwiftIRCore",
            exclude: ["MLIRCoreWrappers.c", "README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ],
            linkerSettings: [
                .unsafeFlags(mlirLinkerFlags),
            ]
        ),

        // MARK: - Type System

        .target(
            name: "SwiftIRTypes",
            dependencies: ["SwiftIRCore"],
            path: "Sources/SwiftIRTypes",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // MARK: - Dialects (Arith, Func, SCF)

        .target(
            name: "SwiftIRDialects",
            dependencies: ["SwiftIRCore", "SwiftIRTypes"],
            path: "Sources/SwiftIRDialects",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // MARK: - IR Builders & DSL

        .target(
            name: "SwiftIRBuilders",
            dependencies: ["SwiftIRCore", "SwiftIRTypes", "SwiftIRDialects"],
            path: "Sources/SwiftIRBuilders",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // MARK: - Macro System

        .macro(
            name: "SwiftIRMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
            ],
            path: "Sources/SwiftIRMacros"
        ),

        // MARK: - Main SwiftIR API

        .target(
            name: "SwiftIR",
            dependencies: [
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRDialects",
                "SwiftIRBuilders",
                "SwiftIRMacros",
                "SwiftIRXLA",
                "SwiftIRStableHLO"
            ],
            path: "Sources/SwiftIR",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // MARK: - Trinity Phase 7: Tensor Dialect for ML

        // Tensor operations using MLIR Tensor dialect (StableHLO foundation)
        .target(
            name: "SwiftIRXLA",
            dependencies: ["SwiftIRCore", "SwiftIRTypes", "SwiftIRDialects", "SwiftIRStableHLO", "PJRTCWrappers"],
            path: "Sources/SwiftIRXLA",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ],
            linkerSettings: [
                .unsafeFlags(mlirLinkerFlags + [
                    // PJRT plugin is dynamically loaded at runtime via dlopen
                ]),
            ]
        ),

        // MARK: - Phase 10: StableHLO Integration

        // StableHLO dialect support for portable ML operations
        .target(
            name: "SwiftIRStableHLO",
            dependencies: ["SwiftIRCore", "SwiftIRTypes", "SwiftIRDialects"],
            path: "Sources/SwiftIRStableHLO",
            exclude: ["README.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePathsWithStableHLO),
            ],
            linkerSettings: [
                .unsafeFlags(mlirLinkerFlags),
            ]
        ),

        // MARK: - Runtime Detection & Unified API

        // Automatic CPU/GPU/TPU detection and unified client creation
        .target(
            name: "SwiftIRRuntime",
            dependencies: [],
            path: "Sources/SwiftIRRuntime"
        ),

        // MARK: - Pure Swift Jupyter/Colab Integration

        // Uses dlopen/dlsym to load native libraries - no C++ interop required
        // Works in LLDB REPL (Jupyter/Colab) where C++ interop is not supported
        .target(
            name: "SwiftIRJupyter",
            dependencies: [],
            path: "Sources/SwiftIRJupyter"
        ),

        // SPIR-V for Graphics/Compute
        // .target(
        //     name: "SwiftIRGPU",
        //     dependencies: ["SwiftIRCore", "SwiftIRTypes"],
        //     path: "Sources/SwiftIRGPU",
        //     swiftSettings: [
        //         .interoperabilityMode(.Cxx),
        //         .unsafeFlags([
        //             "-I", "Sources/SwiftIRCore/include",
        //             "-I", "/opt/homebrew/opt/llvm/include",
        //         ]),
        //     ],
        //     linkerSettings: [
        //         .unsafeFlags([
        //             "-L/opt/homebrew/opt/llvm/lib",
        //             "-lMLIRSPIRV",
        //             "-lMLIRSPIRVSerialization",
        //         ]),
        //     ]
        // ),

        // MARK: - Examples

        .executableTarget(
            name: "SimpleNN",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRBuilders",
                "SwiftIRDialects"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["SimpleNN.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "CNN",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRBuilders",
                "SwiftIRDialects"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["CNN.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "XLA_Execution",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["XLA_Execution.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "StableHLO_CNN",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["StableHLO_CNN.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "StableHLO_ResNet",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["StableHLO_ResNet.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_Add_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRDialects",
                "SwiftIRBuilders",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_Add_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_SimpleString_Example",
            dependencies: [
                "SwiftIRXLA",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_SimpleString_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_PowerfulDSL_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_PowerfulDSL_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_NeuralNet_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_NeuralNet_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_SequentialDSL_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_SequentialDSL_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_MatMul_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRDialects",
                "SwiftIRBuilders",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_MatMul_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "Macro_Example",
            dependencies: [
                "SwiftIR",
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["Macro_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "BuildingSimulation_Simple",
            dependencies: [
                "SwiftIR",
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["BuildingSimulation_Simple.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "BuildingSimulation_SwiftIR",
            dependencies: [
                "SwiftIR",
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["BuildingSimulation_SwiftIR.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "BuildingSimulation_StandardSwift",
            dependencies: [],  // No SwiftIR dependencies - pure Swift only
            path: "Examples",
            exclude: ["README.md"],
            sources: ["BuildingSimulation_StandardSwift.swift"]
        ),

        .executableTarget(
            name: "RuntimeInfo",
            dependencies: ["SwiftIRRuntime"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["RuntimeInfo.swift"]
        ),

        .executableTarget(
            name: "JupyterTest",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterTest.swift"]
        ),

        .executableTarget(
            name: "JupyterComprehensiveTest",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterComprehensiveTest.swift"]
        ),

        .executableTarget(
            name: "JupyterBuildingSimulationTest",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterBuildingSimulationTest.swift"]
        ),

        .executableTarget(
            name: "JupyterHighLevelTest",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterHighLevelTest.swift"]
        ),

        .executableTarget(
            name: "JupyterBuildingSimulationFull",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterBuildingSimulationFull.swift"]
        ),

        .executableTarget(
            name: "JupyterFeatureParityTest",
            dependencies: ["SwiftIRJupyter"],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["JupyterFeatureParityTest.swift"]
        ),

        .executableTarget(
            name: "PJRT_MultiPath_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRDialects",
                "SwiftIRBuilders",
                "SwiftIRStableHLO"
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_MultiPath_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_ThreeExecutables_Test",
            dependencies: [
                "SwiftIRXLA",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_ThreeExecutables_Test.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_TwoApproaches_Test",
            dependencies: [
                "SwiftIRXLA",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_TwoApproaches_Test.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        .executableTarget(
            name: "PJRT_Approaches1and2_Test",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["PJRT_Approaches1and2_Test.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // SPIR-V Example
        .executableTarget(
            name: "SPIRV_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRStableHLO",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["SPIRV_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // SPIR-V Manual Cast Test
        .executableTarget(
            name: "SPIRV_ManualCast_Test",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["SPIRV_ManualCast_Test.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // LLVM/PTX Backend Example
        .executableTarget(
            name: "LLVM_VectorAdd_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["LLVM_VectorAdd_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // SPIR-V Vector Addition Example (Real Linalg lowering)
        .executableTarget(
            name: "SPIRV_VectorAdd_Example",
            dependencies: [
                "SwiftIRXLA",
                "SwiftIRCore",
                "SwiftIRTypes",
            ],
            path: "Examples",
            exclude: ["README.md"],
            sources: ["SPIRV_VectorAdd_Example.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ]
        ),

        // MARK: - Tests

        .testTarget(
            name: "SwiftIRTests",
            dependencies: [
                "SwiftIR",
                "SwiftIRCore",
                "SwiftIRTypes",
                "SwiftIRDialects",
                "SwiftIRBuilders",
                "SwiftIRMacros",
                "SwiftIRXLA",
                "SwiftIRStableHLO"
            ],
            path: "Tests/SwiftIRTests",
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(allSwiftIncludePaths),
            ],
            linkerSettings: [
                .unsafeFlags(mlirLinkerFlags + [
                ]),
            ]
        ),

        .testTarget(
            name: "SwiftIRRuntimeTests",
            dependencies: ["SwiftIRRuntime"],
            path: "Tests/SwiftIRRuntimeTests"
        ),
    ]
)
