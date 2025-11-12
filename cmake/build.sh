#!/bin/bash
# Build script for SwiftIRMLIR unified library

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building SwiftIRMLIR unified library...${NC}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo -e "${BLUE}Configuring CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0

# Build
echo -e "${BLUE}Building library...${NC}"
cmake --build . -j$(sysctl -n hw.ncpu)

# Check if library was created
if [ -f "${BUILD_DIR}/lib/libSwiftIRMLIR.dylib" ]; then
    echo -e "${GREEN}✓ Successfully built libSwiftIRMLIR.dylib${NC}"
    echo -e "${GREEN}  Location: ${BUILD_DIR}/lib/libSwiftIRMLIR.dylib${NC}"

    # Show library info
    echo -e "${BLUE}Library information:${NC}"
    file "${BUILD_DIR}/lib/libSwiftIRMLIR.dylib"
    echo -e "${BLUE}Library size:${NC}"
    ls -lh "${BUILD_DIR}/lib/libSwiftIRMLIR.dylib"

    # Optionally install to a common location
    echo ""
    echo -e "${BLUE}To use this library, update Package.swift with:${NC}"
    echo "  linkerSettings: ["
    echo "    .unsafeFlags(["
    echo "      \"-L${BUILD_DIR}/lib\","
    echo "      \"-lSwiftIRMLIR\","
    echo "      \"-L/opt/homebrew/lib\","
    echo "      \"-lzstd\", \"-lz\", \"-lcurses\""
    echo "    ])"
    echo "  ]"
else
    echo -e "${RED}✗ Failed to build library${NC}"
    exit 1
fi
