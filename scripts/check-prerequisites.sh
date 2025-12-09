#!/usr/bin/env bash
# SwiftIR Prerequisites Checker
# Verifies system requirements for building SwiftIR on Ubuntu 24.04

set -uo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
MISSING=0
WARNINGS=0

success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; ((WARNINGS++)) || true; }
error() { echo -e "${RED}✗${NC} $1"; ((MISSING++)) || true; }

echo "========================================"
echo "SwiftIR Prerequisites Checker"
echo "========================================"
echo ""

# Check OS
echo "Checking system..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" && "$VERSION_ID" == "24.04" ]]; then
        success "Ubuntu 24.04 detected"
    elif [[ "$ID" == "ubuntu" ]]; then
        warning "Ubuntu $VERSION_ID detected (24.04 recommended)"
    else
        warning "$NAME $VERSION_ID detected (Ubuntu 24.04 recommended)"
    fi
else
    warning "Could not detect OS version"
fi

echo ""
echo "Checking build tools..."

# Check essential build tools
declare -A tool_packages=(
    ["cmake"]="cmake"
    ["ninja"]="ninja-build"
    ["git"]="git"
    ["wget"]="wget"
    ["curl"]="curl"
    ["python3"]="python3"
    ["pip3"]="python3-pip"
)

for tool in cmake ninja git wget curl python3 pip3; do
    if command -v $tool &> /dev/null; then
        version=$($tool --version 2>&1 | head -1)
        success "$tool: $version"
    else
        pkg="${tool_packages[$tool]}"
        error "$tool not found (install with: sudo apt install $pkg)"
    fi
done

echo ""
echo "Checking compilers..."

# Check clang/clang++
if command -v clang-18 &> /dev/null; then
    version=$(clang-18 --version | head -1)
    success "clang-18: $version"
elif command -v clang &> /dev/null; then
    version=$(clang --version | head -1)
    warning "clang found but clang-18 preferred: $version"
else
    error "clang not found (install with: sudo apt install clang-18 llvm-18-dev)"
fi

if command -v clang++-18 &> /dev/null; then
    version=$(clang++-18 --version | head -1)
    success "clang++-18: $version"
elif command -v clang++ &> /dev/null; then
    version=$(clang++ --version | head -1)
    warning "clang++ found but clang++-18 preferred: $version"
else
    error "clang++ not found (install with: sudo apt install clang-18)"
fi

echo ""
echo "Checking Swift..."

# Source swiftly environment if available
if [ -f "$HOME/.local/share/swiftly/env.sh" ]; then
    . "$HOME/.local/share/swiftly/env.sh"
elif [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check swiftly
if command -v swiftly &> /dev/null; then
    success "swiftly installed"
else
    error "swiftly not found (install with: curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash)"
fi

# Check Swift
if command -v swift &> /dev/null; then
    version=$(swift --version 2>&1 | head -1)
    success "Swift: $version"

    # Check if it's a development snapshot
    if [[ "$version" == *"DEVELOPMENT"* || "$version" == *"main"* || "$version" == *"-dev"* ]]; then
        success "Swift development snapshot detected"
    else
        warning "Swift release version detected (development snapshot recommended)"
    fi
else
    error "Swift not found (install with: swiftly install main-snapshot-2025-11-03)"
fi

echo ""
echo "Checking libraries..."

# Check for required libraries
declare -A libs=(
    ["libstdc++-13-dev"]="/usr/include/c++/13"
    ["libicu-dev"]="/usr/include/unicode"
    ["libssl-dev"]="/usr/include/openssl"
    ["libxml2-dev"]="/usr/include/libxml2"
    ["libcurl4-openssl-dev"]="/usr/include/x86_64-linux-gnu/curl"
    ["libz-dev"]="/usr/include/zlib.h"
    ["libzstd-dev"]="/usr/include/zstd.h"
    ["libncurses-dev"]="/usr/include/ncurses.h"
    ["libsqlite3-dev"]="/usr/include/sqlite3.h"
)

for lib in "${!libs[@]}"; do
    path="${libs[$lib]}"
    if [ -e "$path" ]; then
        success "$lib"
    else
        if dpkg -l | grep -q "^ii  $lib "; then
            success "$lib"
        else
            error "$lib not found (install with: sudo apt install $lib)"
        fi
    fi
done

# Check libc++
if [ -d "/usr/lib/llvm-18/include/c++/v1" ]; then
    success "libc++-18-dev"
else
    error "libc++-18-dev not found (install with: sudo apt install libc++-18-dev libc++abi-18-dev)"
fi

echo ""
echo "Checking Bazel..."

# Check bazelisk or bazel
if command -v bazelisk &> /dev/null; then
    success "bazelisk installed"
elif command -v bazel &> /dev/null; then
    version=$(bazel --version)
    success "bazel: $version"
else
    error "bazelisk/bazel not found (install with: sudo wget -O /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && sudo chmod +x /usr/local/bin/bazelisk)"
fi

echo ""
echo "Checking SwiftIR dependencies (built by install script)..."

# Check if dependencies are built
DEPS_DIR="${SWIFTIR_DEPS_DIR:-/opt/swiftir-deps}"

if [ -d "$DEPS_DIR" ]; then
    success "Dependencies directory: $DEPS_DIR"

    # Check LLVM/MLIR - warning only, install script builds this
    if [ -f "$DEPS_DIR/lib/libMLIR.a" ] || [ -d "$DEPS_DIR/lib/cmake/mlir" ]; then
        success "LLVM/MLIR installed"
    else
        warning "LLVM/MLIR not found in $DEPS_DIR (will be built by install script)"
    fi

    # Check StableHLO - warning only, install script builds this
    if ls "$DEPS_DIR/lib/libStablehlo"*.a &> /dev/null 2>&1; then
        success "StableHLO installed"
    else
        warning "StableHLO not found in $DEPS_DIR (will be built by install script)"
    fi

    # Check PJRT plugin - warning only, install script builds this
    if [ -f "$DEPS_DIR/lib/pjrt_c_api_cpu_plugin.so" ]; then
        success "PJRT CPU plugin installed"
    else
        warning "PJRT CPU plugin not found in $DEPS_DIR (will be built by install script)"
    fi

    # Check Shardy - warning only, install script builds this
    if [ -f "$DEPS_DIR/lib/libsdy_capi.so" ]; then
        success "Shardy (SDY) installed"
    else
        warning "Shardy not found in $DEPS_DIR (optional - will be built by install script)"
    fi

    if [ -f "$DEPS_DIR/bin/sdy_opt" ]; then
        success "sdy_opt tool installed"
    else
        warning "sdy_opt not found in $DEPS_DIR (optional - will be built by install script)"
    fi
else
    warning "Dependencies directory not found: $DEPS_DIR (will be created by install script)"
fi

echo ""
echo "Checking environment variables..."

# Check environment variables
if [ -n "${SWIFTIR_DEPS_DIR:-}" ]; then
    success "SWIFTIR_DEPS_DIR=$SWIFTIR_DEPS_DIR"
else
    warning "SWIFTIR_DEPS_DIR not set (will use default: /opt/swiftir-deps)"
fi

if [ -f /etc/profile.d/swiftir.sh ]; then
    success "SwiftIR environment file exists"
else
    warning "SwiftIR environment file not found at /etc/profile.d/swiftir.sh"
fi

# Check LD_LIBRARY_PATH
if [[ "${LD_LIBRARY_PATH:-}" == *"swiftir"* ]]; then
    success "LD_LIBRARY_PATH includes SwiftIR dependencies"
else
    warning "LD_LIBRARY_PATH does not include SwiftIR dependencies"
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"

if [ $MISSING -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All prerequisites satisfied!${NC}"
    exit 0
elif [ $MISSING -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS warnings (non-critical)${NC}"
    exit 0
else
    echo -e "${RED}$MISSING missing prerequisites${NC}"
    echo -e "${YELLOW}$WARNINGS warnings${NC}"
    echo ""
    echo "Run the following to install missing dependencies:"
    echo "  ./scripts/install-swiftir-ubuntu.sh"
    exit 1
fi
