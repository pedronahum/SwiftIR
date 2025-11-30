cmake_minimum_required(VERSION 3.20)
project(JupyterMLIRWrapper C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# SDK path
set(SWIFTIR_DEPS "$ENV{SWIFTIR_DEPS}")
if(NOT SWIFTIR_DEPS)
    set(SWIFTIR_DEPS "/tmp/swiftir-full-sdk")
endif()

message(STATUS "Using SWIFTIR_DEPS: ${SWIFTIR_DEPS}")

# Include paths for MLIR headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SWIFTIR_DEPS}/include
)

# Create shared library
add_library(JupyterMLIRWrapper SHARED
    JupyterMLIRWrapper.c
)

# Link against SwiftIRMLIR which contains the MLIR C API
target_link_libraries(JupyterMLIRWrapper PRIVATE
    ${SWIFTIR_DEPS}/lib/libSwiftIRMLIR.so
)

# Set install rpath to $ORIGIN so it finds libraries in the same directory
set_target_properties(JupyterMLIRWrapper PROPERTIES
    OUTPUT_NAME "JupyterMLIRWrapper"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install to the SDK lib directory
install(TARGETS JupyterMLIRWrapper
    LIBRARY DESTINATION ${SWIFTIR_DEPS}/lib
)

message(STATUS "JupyterMLIRWrapper will be installed to: ${SWIFTIR_DEPS}/lib")
