#!/usr/bin/env bash
# SwiftIR Installation Verification Script
# Verifies that all dependencies are correctly installed and functional

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; ((WARNINGS++)) || true; }
error() { echo -e "${RED}✗${NC} $1"; ((ERRORS++)) || true; }

echo "========================================"
echo "SwiftIR Installation Verification"
echo "========================================"
echo ""

# ============================================================
# 1. Verify Swift Installation
# ============================================================

echo "Verifying Swift installation..."

if command -v swift &> /dev/null; then
    VERSION=$(swift --version | head -1)
    success "Swift installed: $VERSION"

    # Check C++ interop support (enabled via Package.swift swiftSettings)
    if swift build --help 2>&1 | grep -q "Xcxx"; then
        success "Swift C++ compiler support available"
    else
        warning "Swift C++ compiler flag not found"
    fi
else
    error "Swift not found"
fi

echo ""

# ============================================================
# 2. Verify Dependencies Directory
# ============================================================

echo "Verifying dependencies..."

DEPS_DIR="${SWIFTIR_DEPS_DIR:-/opt/swiftir-deps}"

if [ ! -d "$DEPS_DIR" ]; then
    error "Dependencies directory not found: $DEPS_DIR"
else
    success "Dependencies directory: $DEPS_DIR"

    # Check LLVM/MLIR
    echo ""
    echo "  LLVM/MLIR:"
    MLIR_LIBS=$(ls "$DEPS_DIR/lib/libMLIR"*.a 2>/dev/null | wc -l)
    if [ "$MLIR_LIBS" -gt 0 ]; then
        success "    $MLIR_LIBS MLIR libraries found"
    elif [ -f "$DEPS_DIR/lib/libMLIR.so" ]; then
        size=$(du -h "$DEPS_DIR/lib/libMLIR.so" | cut -f1)
        success "    libMLIR.so ($size)"
    else
        error "    MLIR library not found"
    fi

    if [ -f "$DEPS_DIR/bin/mlir-opt" ]; then
        version=$("$DEPS_DIR/bin/mlir-opt" --version 2>&1 | head -1)
        success "    mlir-opt: $version"
    else
        warning "    mlir-opt not found"
    fi

    if [ -d "$DEPS_DIR/include/mlir" ]; then
        success "    MLIR headers present"
    else
        error "    MLIR headers not found"
    fi

    # Check StableHLO
    echo ""
    echo "  StableHLO:"
    STABLEHLO_LIBS=$(ls "$DEPS_DIR/lib/libStablehlo"*.a 2>/dev/null | wc -l)
    if [ "$STABLEHLO_LIBS" -gt 0 ]; then
        success "    $STABLEHLO_LIBS StableHLO libraries found"
    else
        error "    StableHLO libraries not found"
    fi

    if [ -d "$DEPS_DIR/include/stablehlo" ]; then
        success "    StableHLO headers present"
    else
        warning "    StableHLO headers not found"
    fi

    # Check PJRT Plugin
    echo ""
    echo "  XLA/PJRT:"
    if [ -f "$DEPS_DIR/lib/pjrt_c_api_cpu_plugin.so" ]; then
        size=$(du -h "$DEPS_DIR/lib/pjrt_c_api_cpu_plugin.so" | cut -f1)
        success "    pjrt_c_api_cpu_plugin.so ($size)"

        # Verify plugin symbols
        if nm -D "$DEPS_DIR/lib/pjrt_c_api_cpu_plugin.so" 2>/dev/null | grep -q "GetPjrtApi"; then
            success "    GetPjrtApi symbol present"
        else
            warning "    GetPjrtApi symbol not found (may still work)"
        fi
    else
        error "    PJRT CPU plugin not found"
    fi

    if [ -d "$DEPS_DIR/include/xla" ]; then
        success "    XLA headers present"
    else
        warning "    XLA headers not found"
    fi

    # Check Shardy (SDY Sharding Dialect)
    echo ""
    echo "  Shardy (SDY):"
    if [ -f "$DEPS_DIR/lib/libsdy_capi.so" ]; then
        size=$(du -h "$DEPS_DIR/lib/libsdy_capi.so" | cut -f1)
        success "    libsdy_capi.so ($size)"
    else
        warning "    libsdy_capi.so not found (optional - for sharded execution)"
    fi

    if [ -f "$DEPS_DIR/bin/sdy_opt" ]; then
        success "    sdy_opt available"
        # Try to get version info
        if "$DEPS_DIR/bin/sdy_opt" --help 2>&1 | head -1 | grep -q "sdy"; then
            success "    sdy_opt executable works"
        fi
    else
        warning "    sdy_opt not found (optional - for sharding transformations)"
    fi

    if [ -d "$DEPS_DIR/include/shardy" ]; then
        success "    Shardy headers present"
    else
        warning "    Shardy headers not found"
    fi
fi

echo ""

# ============================================================
# 3. Verify Environment Variables
# ============================================================

echo "Verifying environment..."

if [ -n "${SWIFTIR_DEPS_DIR:-}" ]; then
    success "SWIFTIR_DEPS_DIR=$SWIFTIR_DEPS_DIR"
else
    warning "SWIFTIR_DEPS_DIR not set"
fi

if [[ "${LD_LIBRARY_PATH:-}" == *"swiftir"* || "${LD_LIBRARY_PATH:-}" == *"$DEPS_DIR"* ]]; then
    success "LD_LIBRARY_PATH includes dependencies"
else
    warning "LD_LIBRARY_PATH may not include $DEPS_DIR/lib"
fi

if [[ "${LIBRARY_PATH:-}" == *"swiftir"* || "${LIBRARY_PATH:-}" == *"$DEPS_DIR"* ]]; then
    success "LIBRARY_PATH includes dependencies"
else
    warning "LIBRARY_PATH may not include $DEPS_DIR/lib"
fi

echo ""

# ============================================================
# 4. Verify Compiler Setup
# ============================================================

echo "Verifying compilers..."

if command -v clang-18 &> /dev/null; then
    success "clang-18 available"
else
    warning "clang-18 not found (may affect builds)"
fi

if command -v clang++-18 &> /dev/null; then
    success "clang++-18 available"
else
    warning "clang++-18 not found (may affect builds)"
fi

# Test C++ compilation
echo ""
echo "Testing C++ compilation..."

TEST_DIR=$(mktemp -d)
cat > "$TEST_DIR/test.cpp" << 'EOF'
#include <iostream>
int main() {
    std::cout << "C++ compilation OK" << std::endl;
    return 0;
}
EOF

if clang++ -std=c++17 "$TEST_DIR/test.cpp" -o "$TEST_DIR/test" 2>/dev/null; then
    if "$TEST_DIR/test" 2>/dev/null | grep -q "OK"; then
        success "C++17 compilation and execution"
    else
        warning "C++ program compiled but failed to run"
    fi
else
    error "C++17 compilation failed"
fi

rm -rf "$TEST_DIR"

echo ""

# ============================================================
# 5. Test SwiftIR Build (if in project directory)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/Package.swift" ]; then
    echo "Testing SwiftIR package resolution..."

    cd "$PROJECT_DIR"

    # Try to resolve dependencies
    if swift package resolve 2>&1 | tee /tmp/swift-resolve.log; then
        success "Swift package dependencies resolved"
    else
        warning "Swift package resolution had issues (may need path updates)"
    fi
else
    warning "Not in SwiftIR project directory, skipping package test"
fi

echo ""

# ============================================================
# Summary
# ============================================================

echo "========================================"
echo "Verification Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "SwiftIR is ready to build."
    echo "Run: swift build"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS warnings (non-critical)${NC}"
    echo ""
    echo "SwiftIR should be buildable, but review warnings above."
    exit 0
else
    echo -e "${RED}$ERRORS errors${NC}"
    echo -e "${YELLOW}$WARNINGS warnings${NC}"
    echo ""
    echo "Please fix errors before building SwiftIR."
    echo "Re-run: ./scripts/install-swiftir-ubuntu.sh"
    exit 1
fi
