cmake_minimum_required(VERSION 3.20)
project(PJRTProtoHelper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# XLA build directory
set(XLA_BUILD_DIR "/Users/pedro/programming/swift/xla-bazelisk-build/607423a1c455c14997634f7cf9a59c45")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/xla/pjrt/proto
    ${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin
    ${XLA_BUILD_DIR}/execroot/xla
    ${XLA_BUILD_DIR}/external/com_google_protobuf/src
    ${XLA_BUILD_DIR}/external/com_google_absl
)

# Create shared library for protobuf helper
add_library(PJRTProtoHelper SHARED
    PJRTProtoHelper.cpp
)

# Set library output name
set_target_properties(PJRTProtoHelper PROPERTIES
    OUTPUT_NAME "PJRTProtoHelper"
    MACOSX_RPATH ON
    INSTALL_NAME_DIR "@rpath"
)

# Find ALL XLA proto libraries (just link all of them to be safe)
file(GLOB_RECURSE XLA_PROTO_LIBS
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/xla/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/xla/*/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/xla/*/*/*.a"
)

# Find ALL protobuf libraries
file(GLOB_RECURSE PROTOBUF_LIBS
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_protobuf/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_protobuf/*/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_protobuf/*/*/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_protobuf/*/*/*/*.a"
)

# Find ALL Abseil libraries
file(GLOB_RECURSE ABSL_LIBS
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_absl/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_absl/*/*.a"
    "${XLA_BUILD_DIR}/execroot/xla/bazel-out/darwin_arm64-opt/bin/external/com_google_absl/*/*/*.a"
)

list(LENGTH XLA_PROTO_LIBS XLA_PROTO_COUNT)
list(LENGTH PROTOBUF_LIBS PROTOBUF_COUNT)
list(LENGTH ABSL_LIBS ABSL_COUNT)

message(STATUS "Found ${XLA_PROTO_COUNT} XLA proto libraries")
message(STATUS "Found ${PROTOBUF_COUNT} Protobuf libraries")
message(STATUS "Found ${ABSL_COUNT} Abseil libraries")

# Link with -force_load for ALL libraries to ensure all symbols are pulled in
# This is the "nuclear option" - it will make the dylib large but should work
set(ALL_LIBS ${XLA_PROTO_LIBS} ${PROTOBUF_LIBS} ${ABSL_LIBS})

foreach(LIB ${ALL_LIBS})
    target_link_libraries(PJRTProtoHelper -Wl,-force_load,${LIB})
endforeach()

# Install to SwiftIR cmake/build/lib (same location as libSwiftIRMLIR.dylib)
install(TARGETS PJRTProtoHelper
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/build/lib
)

# Print summary
message(STATUS "PJRTProtoHelper Configuration:")
message(STATUS "  XLA Build Dir: ${XLA_BUILD_DIR}")
message(STATUS "  Total libraries to link: ${CMAKE_MATCH_COUNT}")
message(STATUS "  Install destination: ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/build/lib")
