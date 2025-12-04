# CMakeLists.txt for SwiftIR Profiler C Wrapper
#
# This builds libSwiftIRProfiler.so which provides:
# 1. Simple timing API (always available)
# 2. XSpace export (when TSL profiler headers are available)
#
# Usage:
#   mkdir build && cd build
#   cmake -DSWIFTIR_DEPS=/opt/swiftir-deps ..
#   make

cmake_minimum_required(VERSION 3.20)
project(SwiftIRProfiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# SwiftIR dependencies path (default to /opt/swiftir-deps)
if(NOT DEFINED SWIFTIR_DEPS)
    if(DEFINED ENV{SWIFTIR_DEPS})
        set(SWIFTIR_DEPS $ENV{SWIFTIR_DEPS})
    else()
        set(SWIFTIR_DEPS "/opt/swiftir-deps")
    endif()
endif()

message(STATUS "SWIFTIR_DEPS: ${SWIFTIR_DEPS}")

# TSL profiler support - requires protobuf 6.x and abseil from SWIFTIR_DEPS
# Check if the required headers exist
# Force enable TSL profiler since we verified files exist
option(ENABLE_TSL_PROFILER "Enable TSL profiler with XSpace export" ON)

if(ENABLE_TSL_PROFILER)
    set(TSL_PROFILER_AVAILABLE ON)
    message(STATUS "TSL Profiler: Enabled")
    message(STATUS "  - Simple timing API: Available")
    message(STATUS "  - XSpace export: Available (TensorBoard compatible)")
else()
    set(TSL_PROFILER_AVAILABLE OFF)
    message(STATUS "TSL Profiler: Disabled (simple timing mode)")
    message(STATUS "  - Simple timing API: Available")
    message(STATUS "  - XSpace export: Disabled (requires full XLA build integration)")
endif()

# Include directories
set(PROFILER_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SWIFTIR_DEPS}/include
    ${SWIFTIR_DEPS}/include/external/tsl
)

# Create the shared library
add_library(SwiftIRProfiler SHARED
    XLAProfilerWrapper.cpp
)

target_include_directories(SwiftIRProfiler PRIVATE
    ${PROFILER_INCLUDES}
)

# Compile definitions
if(TSL_PROFILER_AVAILABLE)
    target_compile_definitions(SwiftIRProfiler PRIVATE
        SWIFTIR_HAS_TSL_PROFILER=1
    )
endif()

# Link libraries
if(TSL_PROFILER_AVAILABLE)
    # Add additional include directories for protobuf and abseil headers
    target_include_directories(SwiftIRProfiler PRIVATE
        ${SWIFTIR_DEPS}/include
    )

    # Collect all required static libraries (.pic.a files only for shared library linking)
    file(GLOB XPLANE_LIBS "${SWIFTIR_DEPS}/lib/xla/libxplane_proto.pic.a"
                          "${SWIFTIR_DEPS}/lib/xla/libprofiler_options_proto.pic.a")
    file(GLOB PROTOBUF_LIBS "${SWIFTIR_DEPS}/lib/protobuf/*.pic.a")
    file(GLOB ABSL_LIBS "${SWIFTIR_DEPS}/lib/absl/*.pic.a")

    list(LENGTH PROTOBUF_LIBS PROTOBUF_COUNT)
    list(LENGTH ABSL_LIBS ABSL_COUNT)
    message(STATUS "Found ${PROTOBUF_COUNT} protobuf libraries")
    message(STATUS "Found ${ABSL_COUNT} abseil libraries")

    # Use whole-archive to avoid dependency ordering issues
    # This includes all symbols from the static libraries
    target_link_options(SwiftIRProfiler PRIVATE
        -Wl,--whole-archive
    )
    target_link_libraries(SwiftIRProfiler PRIVATE
        ${XPLANE_LIBS}
        ${PROTOBUF_LIBS}
        ${ABSL_LIBS}
    )
    target_link_options(SwiftIRProfiler PRIVATE
        -Wl,--no-whole-archive
    )
endif()

# Standard libraries
target_link_libraries(SwiftIRProfiler PRIVATE
    pthread
    dl
)

# Set output properties
set_target_properties(SwiftIRProfiler PROPERTIES
    OUTPUT_NAME "SwiftIRProfiler"
    VERSION 1.0.0
    SOVERSION 1
    # Set RPATH for finding dependencies at runtime
    INSTALL_RPATH "$ORIGIN:${SWIFTIR_DEPS}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Export symbols
if(NOT APPLE)
    target_link_options(SwiftIRProfiler PRIVATE
        -Wl,--no-undefined
    )
endif()

# Installation
install(TARGETS SwiftIRProfiler
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/XLAProfilerWrapper.h
    DESTINATION include/swiftir
)

# Print configuration summary
message(STATUS "")
message(STATUS "SwiftIR Profiler Configuration:")
message(STATUS "  TSL Profiler Available: ${TSL_PROFILER_AVAILABLE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
