cmake_minimum_required(VERSION 3.20)
project(PJRTProtoHelper C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform-specific paths
if(APPLE)
    set(XLA_BUILD_DIR "/Users/pedro/programming/swift/xla-bazelisk-build/607423a1c455c14997634f7cf9a59c45")
    set(XLA_ARCH "darwin_arm64-opt")
    set(USE_DEPS_DIR FALSE)
else()
    # Linux paths - use swiftir-deps if available, otherwise fall back to build dir
    set(SWIFTIR_DEPS_DIR "$ENV{SWIFTIR_DEPS_DIR}")
    if(NOT SWIFTIR_DEPS_DIR)
        set(SWIFTIR_DEPS_DIR "/opt/swiftir-deps")
    endif()

    # Check if proto libraries exist in deps dir
    if(EXISTS "${SWIFTIR_DEPS_DIR}/lib/xla" AND EXISTS "${SWIFTIR_DEPS_DIR}/lib/protobuf")
        set(USE_DEPS_DIR TRUE)
        message(STATUS "Using pre-built libraries from ${SWIFTIR_DEPS_DIR}")
    else()
        set(USE_DEPS_DIR FALSE)
        # Fall back to build directory
        set(BUILD_DIR "$ENV{BUILD_DIR}")
        if(NOT BUILD_DIR)
            set(BUILD_DIR "$ENV{HOME}/swiftir-build")
        endif()

        # Find the XLA bazel build directory
        file(GLOB XLA_BUILD_DIRS "${BUILD_DIR}/xla-bazelisk-build/*/execroot/xla")
        if(XLA_BUILD_DIRS)
            list(GET XLA_BUILD_DIRS 0 XLA_EXECROOT)
            get_filename_component(XLA_BUILD_DIR "${XLA_EXECROOT}" DIRECTORY)
            get_filename_component(XLA_BUILD_DIR "${XLA_BUILD_DIR}" DIRECTORY)
        else()
            set(XLA_BUILD_DIR "${BUILD_DIR}/xla-bazelisk-build")
        endif()
        set(XLA_ARCH "k8-opt")
        message(STATUS "Using XLA Build Dir: ${XLA_BUILD_DIR}")
    endif()
endif()

# Include directories - only need our own header
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create shared library for protobuf helper
add_library(PJRTProtoHelper SHARED
    PJRTProtoHelper.cpp
)

# Set library output name
if(APPLE)
    set_target_properties(PJRTProtoHelper PROPERTIES
        OUTPUT_NAME "PJRTProtoHelper"
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
else()
    set_target_properties(PJRTProtoHelper PROPERTIES
        OUTPUT_NAME "PJRTProtoHelper"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# No external libraries needed - we manually encode protobuf format
message(STATUS "PJRTProtoHelper: Using manual protobuf encoding (no external dependencies)")

# Create shared library for PJRT Simple Wrapper (used by SwiftIRJupyter)
add_library(PJRTSimpleWrapper SHARED
    PJRTSimpleWrapper.c
)

# Set library output name and link dependencies
if(APPLE)
    set_target_properties(PJRTSimpleWrapper PROPERTIES
        OUTPUT_NAME "PJRTSimpleWrapper"
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
else()
    set_target_properties(PJRTSimpleWrapper PROPERTIES
        OUTPUT_NAME "PJRTSimpleWrapper"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# PJRTSimpleWrapper needs PJRTProtoHelper
target_link_libraries(PJRTSimpleWrapper PRIVATE PJRTProtoHelper)

# Install to SwiftIR cmake/build/lib (same location as libSwiftIRMLIR)
install(TARGETS PJRTProtoHelper PJRTSimpleWrapper
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/build/lib
)

message(STATUS "PJRTProtoHelper Configuration:")
message(STATUS "  Install destination: ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/build/lib")
message(STATUS "PJRTSimpleWrapper Configuration:")
message(STATUS "  Install destination: ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/build/lib")
