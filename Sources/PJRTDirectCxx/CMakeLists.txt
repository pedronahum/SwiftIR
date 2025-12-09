cmake_minimum_required(VERSION 3.16)
project(PJRTDirect VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pjrt_c_api.h location
set(PJRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PJRTCWrappers/include"
    CACHE PATH "Path to PJRT C API headers")

# Library sources
add_library(PJRTDirect SHARED
    PJRTDirect.cpp
)

target_include_directories(PJRTDirect PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${PJRT_INCLUDE_DIR}
)

# Link with dl for dlopen/dlsym
target_link_libraries(PJRTDirect PRIVATE dl)

# Compiler flags for performance
target_compile_options(PJRTDirect PRIVATE
    -O3
    -fno-exceptions
    -fno-rtti
    -Wall
    -Wextra
)

# Export symbols
set_target_properties(PJRTDirect PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/PJRTDirect.h
)

# Installation
install(TARGETS PJRTDirect
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/PJRTDirect.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/PJRTDirect.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/PJRTDirect.pc
    DESTINATION lib/pkgconfig
)
