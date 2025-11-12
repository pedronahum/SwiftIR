cmake_minimum_required(VERSION 3.20)
project(SwiftIRMLIR LANGUAGES C CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLIR/LLVM from StableHLO build
set(MLIR_DIR "/Users/pedro/programming/swift/stablehlo/llvm-build/lib/cmake/mlir")
set(LLVM_DIR "/Users/pedro/programming/swift/stablehlo/llvm-build/lib/cmake/llvm")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Add StableHLO library directory for direct library linking
link_directories("/Users/pedro/programming/swift/stablehlo/build/lib")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using StableHLO libraries from: /Users/pedro/programming/swift/stablehlo/build/lib")

# Add MLIR/LLVM include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add StableHLO include directories
include_directories("/Users/pedro/programming/swift/stablehlo")
include_directories("/Users/pedro/programming/swift/stablehlo/stablehlo")
include_directories("/Users/pedro/programming/swift/stablehlo/build/stablehlo")

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

# Link against all necessary MLIR C API libraries
target_link_libraries(SwiftIRMLIR
    PUBLIC
    # StableHLO C API
    StablehloCAPI
    ChloCAPI
    VhloCAPI
    CheckCAPI

    # StableHLO Dialect Libraries
    StablehloOps
    StablehloBase
    StablehloAssemblyFormat
    StablehloBroadcastUtils
    StablehloTypeInference
    StablehloTypeConversion
    ChloOps
    VhloOps
    VhloTypes
    CheckOps

    # StableHLO Passes and Transforms
    StablehloPasses
    StablehloLinalgTransforms
    StablehloTOSATransforms
    StablehloOptimizationPasses
    StablehloPassUtils

    # StableHLO API
    StablehloPortableApi
    StablehloReferenceApi
    StablehloReferenceConfiguration
    StablehloSerialization

    # Interpreter
    InterpreterOps
    InterpreterPasses

    # StableHLO Reference Implementation
    StablehloReferenceElement
    StablehloReferenceTensor
    StablehloReferenceValue
    StablehloReferenceTypes
    StablehloReferenceAxes
    StablehloReferenceIndex
    StablehloReferenceScope
    StablehloReferenceProcess
    StablehloReferenceProcessGrid
    StablehloReferenceToken
    StablehloReferenceNumPy
    StablehloReferenceErrors
    StablehloReferenceOps

    # StableHLO Utilities and Version
    StablehloRegister
    Version

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

    # MLIR Affine Dialect (for GPU lowering pipeline)
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineAnalysis
    MLIRAffineUtils
    MLIRAffineToStandard

    # MLIR GPU Dialect (for SPIR-V support)
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
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64Utils

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

# Set output directory and linker flags to include all symbols
set_target_properties(SwiftIRMLIR PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    # Force inclusion of all symbols from static libraries (macOS linker)
    LINK_FLAGS "-Wl,-all_load"
)

# Install the library
install(TARGETS SwiftIRMLIR
    LIBRARY DESTINATION lib
)

message(STATUS "SwiftIRMLIR will be built at: ${CMAKE_BINARY_DIR}/lib/libSwiftIRMLIR.dylib")
