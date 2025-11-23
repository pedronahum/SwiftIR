cmake_minimum_required(VERSION 3.20)
project(SwiftIRMLIR LANGUAGES C CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific paths
if(APPLE)
    # macOS paths
    set(STABLEHLO_ROOT "/Users/pedro/programming/swift/stablehlo")
    set(MLIR_DIR "${STABLEHLO_ROOT}/llvm-build/lib/cmake/mlir")
    set(LLVM_DIR "${STABLEHLO_ROOT}/llvm-build/lib/cmake/llvm")
    set(STABLEHLO_LIB_DIR "${STABLEHLO_ROOT}/build/lib")
    set(STABLEHLO_INCLUDE_DIRS
        "${STABLEHLO_ROOT}"
        "${STABLEHLO_ROOT}/stablehlo"
        "${STABLEHLO_ROOT}/build/stablehlo"
    )
else()
    # Linux paths - use environment variables or defaults
    set(SWIFTIR_DEPS_DIR "$ENV{SWIFTIR_DEPS_DIR}")
    if(NOT SWIFTIR_DEPS_DIR)
        set(SWIFTIR_DEPS_DIR "/opt/swiftir-deps")
    endif()

    set(MLIR_DIR "${SWIFTIR_DEPS_DIR}/lib/cmake/mlir")
    set(LLVM_DIR "${SWIFTIR_DEPS_DIR}/lib/cmake/llvm")
    set(STABLEHLO_LIB_DIR "${SWIFTIR_DEPS_DIR}/lib")
    set(STABLEHLO_INCLUDE_DIRS
        "${SWIFTIR_DEPS_DIR}/include/stablehlo"
        "${SWIFTIR_DEPS_DIR}/include"
    )
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Add StableHLO library directory for direct library linking
link_directories(${STABLEHLO_LIB_DIR})

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using StableHLO libraries from: ${STABLEHLO_LIB_DIR}")

# Add MLIR/LLVM include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add StableHLO include directories
foreach(dir ${STABLEHLO_INCLUDE_DIRS})
    include_directories(${dir})
endforeach()

# Add MLIR CMake modules to path
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include MLIR/LLVM CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Create the unified shared library
add_library(SwiftIRMLIR SHARED
    SwiftIRMLIR.cpp
)

# On Linux, we need --whole-archive to include all symbols from static libraries
if(NOT APPLE)
    target_link_libraries(SwiftIRMLIR PUBLIC -Wl,--whole-archive -Wl,--allow-multiple-definition)
endif()

# Link against necessary libraries - minimal set for StableHLO v1.0.0
target_link_libraries(SwiftIRMLIR
    PUBLIC
    # StableHLO C API (available in v1.0.0)
    StablehloCAPI
    ChloCAPI

    # StableHLO Dialect Libraries (order matters - dependents before dependencies)
    ChloOps
    StablehloOps
    StablehloBase
    StablehloAssemblyFormat
    StablehloTypeInference
    StablehloBroadcastUtils  # Must come after ChloOps that depends on it

    # StableHLO Transforms (skip Passes and Serialization that need VHLO/Interpreter)
    StablehloLinalgTransforms
    StablehloTOSATransforms

    # MLIR C API - Core
    MLIRCAPIIR
    MLIRCAPIFunc
    MLIRCAPIArith
    MLIRCAPISCF

    # MLIR C API - Tensor/Linalg/Math for ML
    MLIRCAPITensor
    MLIRCAPILinalg
    MLIRCAPIMath

    # MLIR C API - Execution and Conversion
    MLIRCAPIExecutionEngine
    MLIRCAPIConversion
    MLIRCAPITransforms

    # MLIR Dialects
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRMathDialect

    # MLIR Affine Dialect
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineAnalysis
    MLIRAffineUtils
    MLIRAffineToStandard

    # MLIR GPU Dialect
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRCAPIGPU

    # MLIR SPIR-V Dialect
    MLIRSPIRVDialect
    MLIRSPIRVConversion
    MLIRSPIRVTransforms
    MLIRSPIRVUtils
    MLIRSPIRVSerialization
    MLIRCAPISPIRV

    # GPU/SPIR-V Conversion Passes
    MLIRGPUToSPIRV
    MLIRSCFToGPU
    MLIRFuncToSPIRV
    MLIRArithToSPIRV
    MLIRSCFToSPIRV
    MLIRMemRefToSPIRV
    MLIRVectorToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMathToSPIRV
    MLIRTensorToSPIRV

    # MLIR Transforms and Conversions
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRLinalgToStandard
    MLIRReconcileUnrealizedCasts
    MLIRLLVMCommonConversion

    # MLIR Core
    MLIRIR
    MLIRAnalysis
    MLIRTransforms
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRExecutionEngine

    # MLIR Interfaces
    MLIRDataLayoutInterfaces
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRInferTypeOpInterface
    MLIRInferIntRangeInterface
    MLIRCastInterfaces
    MLIRMemorySlotInterfaces
    MLIRDestinationStyleOpInterface

    # MLIR Dialect Libraries
    MLIRLLVMDialect
    MLIRLinalgUtils
    MLIRLinalgTransforms
    MLIRLinalgTransformOps

)

# On Linux with shared LLVM, link against the unified LLVM library
# On macOS, link against individual static libraries
if(NOT APPLE AND TARGET LLVM)
    # Use shared libLLVM.so when available
    target_link_libraries(SwiftIRMLIR PUBLIC LLVM)
else()
    # Link individual LLVM static libraries
    target_link_libraries(SwiftIRMLIR PUBLIC
        # LLVM Core
        LLVMCore
        LLVMSupport
        LLVMDemangle

        # LLVM ORC and JIT
        LLVMOrcJIT
        LLVMOrcShared
        LLVMRuntimeDyld
        LLVMExecutionEngine
        LLVMMCJIT

        # LLVM CodeGen
        LLVMCodeGen
        LLVMGlobalISel
        LLVMAsmPrinter
        LLVMSelectionDAG

        # LLVM Target
        LLVMTarget
        LLVMTargetParser

        # LLVM Transforms
        LLVMScalarOpts
        LLVMipo
        LLVMVectorize
        LLVMInstCombine
        LLVMTransformUtils
        LLVMAnalysis

        # LLVM MC
        LLVMMC
        LLVMMCParser

        # LLVM Bitcode
        LLVMBitReader
        LLVMBitWriter

        # LLVM Object/Binary
        LLVMObject
        LLVMBinaryFormat
    )
endif()

# Add architecture-specific targets (only when not using shared LLVM)
if(NOT (NOT APPLE AND TARGET LLVM))
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        target_link_libraries(SwiftIRMLIR PUBLIC
            LLVMAArch64CodeGen
            LLVMAArch64AsmParser
            LLVMAArch64Desc
            LLVMAArch64Info
            LLVMAArch64Utils
        )
    else()
        target_link_libraries(SwiftIRMLIR PUBLIC
            LLVMX86CodeGen
            LLVMX86AsmParser
            LLVMX86Desc
            LLVMX86Info
        )
    endif()
endif()

# Set output directory and linker flags
if(APPLE)
    set_target_properties(SwiftIRMLIR PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
        LINK_FLAGS "-Wl,-all_load"
    )
    set(LIB_EXT "dylib")
else()
    # Create a version script to export all symbols
    file(WRITE "${CMAKE_BINARY_DIR}/export_all.version" "{ global: *; };")

    set_target_properties(SwiftIRMLIR PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
        # Export all symbols via version script
        LINK_FLAGS "-Wl,--version-script=${CMAKE_BINARY_DIR}/export_all.version"
    )
    set(LIB_EXT "so")
endif()

# Install the library
install(TARGETS SwiftIRMLIR
    LIBRARY DESTINATION lib
)

message(STATUS "SwiftIRMLIR will be built at: ${CMAKE_BINARY_DIR}/lib/libSwiftIRMLIR.${LIB_EXT}")
