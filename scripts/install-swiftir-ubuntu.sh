#!/usr/bin/env bash
# SwiftIR Installation Script for Ubuntu 24.04
# Installs Swift, LLVM/MLIR, StableHLO, and XLA PJRT dependencies

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

# Swift toolchain version (development snapshot)
SWIFT_VERSION="${SWIFT_VERSION:-main-snapshot-2025-11-03}"

# LLVM/StableHLO versions
# StableHLO requires specific LLVM commits - we use the commit from StableHLO's llvm_version.txt
STABLEHLO_VERSION="${STABLEHLO_VERSION:-v1.0.0}"
# LLVM commit that StableHLO v1.0.0 was built against
LLVM_COMMIT="${LLVM_COMMIT:-a6d7828f4c50c1ec7b0b5f61fe59d7a768175dcc}"
XLA_COMMIT="${XLA_COMMIT:-main}"

# Installation directories
DEPS_DIR="${SWIFTIR_DEPS_DIR:-/opt/swiftir-deps}"
BUILD_DIR="${BUILD_DIR:-$HOME/swiftir-build}"

# Number of parallel jobs
JOBS="${JOBS:-$(nproc)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================================
# Parse Arguments
# ============================================================

SKIP_SWIFT=false
SKIP_LLVM=false
SKIP_STABLEHLO=false
SKIP_XLA=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-swift) SKIP_SWIFT=true; shift ;;
        --skip-llvm) SKIP_LLVM=true; shift ;;
        --skip-stablehlo) SKIP_STABLEHLO=true; shift ;;
        --skip-xla) SKIP_XLA=true; shift ;;
        --skip-deps) SKIP_DEPS=true; shift ;;
        --swift-version) SWIFT_VERSION="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --help)
            echo "SwiftIR Installation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-swift      Skip Swift installation"
            echo "  --skip-llvm       Skip LLVM/MLIR build"
            echo "  --skip-stablehlo  Skip StableHLO build"
            echo "  --skip-xla        Skip XLA/PJRT build"
            echo "  --skip-deps       Skip apt dependencies"
            echo "  --swift-version   Swift version (default: $SWIFT_VERSION)"
            echo "  --jobs N          Parallel build jobs (default: $JOBS)"
            echo "  --help            Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1" ;;
    esac
done

echo "========================================"
echo "SwiftIR Installation Script"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Swift Version:    $SWIFT_VERSION"
echo "  LLVM Commit:      $LLVM_COMMIT"
echo "  StableHLO:        $STABLEHLO_VERSION"
echo "  XLA Commit:       $XLA_COMMIT"
echo "  Dependencies:     $DEPS_DIR"
echo "  Build Directory:  $BUILD_DIR"
echo "  Parallel Jobs:    $JOBS"
echo ""

# ============================================================
# 0. Run Prerequisites Check
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/check-prerequisites.sh" ]; then
    log_info "Running prerequisites check..."
    if ! "$SCRIPT_DIR/check-prerequisites.sh"; then
        echo ""
        # Only error out if we're not skipping the relevant components
        if [ "$SKIP_SWIFT" = false ] || [ "$SKIP_DEPS" = false ]; then
            log_error "Prerequisites check failed. Please install missing dependencies first."
        else
            log_warn "Prerequisites check failed, but continuing since --skip flags were provided"
        fi
    fi
    echo ""
else
    log_warn "check-prerequisites.sh not found, skipping prerequisites check"
fi

# ============================================================
# 1. Install System Dependencies
# ============================================================

if [ "$SKIP_DEPS" = false ]; then
    log_info "Installing system dependencies..."

    sudo apt-get update

    # Build essentials
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        pkg-config \
        tzdata

    # LLVM/Clang 18
    sudo apt-get install -y \
        clang-18 \
        llvm-18-dev \
        libc++-18-dev \
        libc++abi-18-dev \
        lld-18

    # Remove LLVM 17 if present to avoid conflicts
    if dpkg -l | grep -q "llvm-17"; then
        log_warn "Removing LLVM 17 to avoid conflicts..."
        sudo apt-get remove -y llvm-17* clang-17* || true
    fi

    # Swift requirements
    sudo apt-get install -y \
        libc6-dev \
        libstdc++-13-dev \
        libicu-dev \
        libssl-dev \
        libxml2-dev \
        libcurl4-openssl-dev \
        libz-dev \
        libsqlite3-dev \
        libpython3-dev

    # MLIR/StableHLO build requirements
    sudo apt-get install -y \
        libzstd-dev \
        libncurses-dev \
        zlib1g-dev

    # Python for XLA/Bazel
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-numpy

    # Set up alternatives for clang
    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100
    sudo update-alternatives --install /usr/bin/lld lld /usr/bin/lld-18 100

    log_success "System dependencies installed"
else
    log_info "Skipping system dependencies (--skip-deps)"
fi

# ============================================================
# 2. Install Bazelisk
# ============================================================

if ! command -v bazelisk &> /dev/null && [ "$SKIP_XLA" = false ]; then
    log_info "Installing Bazelisk..."

    BAZELISK_VERSION="v1.19.0"
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        BAZELISK_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        BAZELISK_ARCH="arm64"
    else
        log_error "Unsupported architecture: $ARCH"
    fi

    sudo wget -O /usr/local/bin/bazelisk \
        "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${BAZELISK_ARCH}"
    sudo chmod +x /usr/local/bin/bazelisk

    log_success "Bazelisk installed"
fi

# ============================================================
# 3. Install Swift via Swiftly
# ============================================================

if [ "$SKIP_SWIFT" = false ]; then
    log_info "Installing Swift via Swiftly..."

    # Add swiftly to PATH if installed
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Install swiftly if not present
    if ! command -v swiftly &> /dev/null; then
        log_info "Swiftly not found, installing..."
        curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash
        export PATH="$HOME/.local/bin:$PATH"
    else
        log_success "Swiftly already installed"
    fi

    # Install Swift toolchain
    swiftly install "$SWIFT_VERSION"
    swiftly use "$SWIFT_VERSION"

    # Verify installation
    swift --version

    # Get Swift toolchain directory
    SWIFT_TOOLCHAIN_DIR=$(dirname $(dirname $(which swift)))

    # Create environment file
    sudo tee /etc/profile.d/swift.sh > /dev/null << EOF
# Swift environment
export PATH="\$HOME/.local/bin:\$PATH"
export SWIFT_TOOLCHAIN_DIR="$SWIFT_TOOLCHAIN_DIR"
EOF

    log_success "Swift $SWIFT_VERSION installed"
else
    log_info "Skipping Swift installation (--skip-swift)"
fi

# ============================================================
# 4. Create Build Directories
# ============================================================

log_info "Creating build directories..."
mkdir -p "$BUILD_DIR"
sudo mkdir -p "$DEPS_DIR"/{lib,include,bin}
sudo chown -R $USER:$USER "$DEPS_DIR"

# ============================================================
# 5. Build LLVM/MLIR
# ============================================================

if [ "$SKIP_LLVM" = false ]; then
    log_info "Building LLVM/MLIR from commit $LLVM_COMMIT..."

    cd "$BUILD_DIR"

    # Clone LLVM if not present
    if [ ! -d "llvm-project" ]; then
        git clone https://github.com/llvm/llvm-project.git
    fi

    cd llvm-project
    git fetch origin
    git checkout "$LLVM_COMMIT"

    # Configure with shared library support to avoid duplicate static initializers
    # LLVM_BUILD_LLVM_DYLIB creates libLLVM.so
    # LLVM_LINK_LLVM_DYLIB makes all tools link against libLLVM.so
    # MLIR_BUILD_MLIR_C_DYLIB creates libMLIRPublicAPI.so for C API
    cmake -G Ninja llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DCMAKE_C_COMPILER=clang-18 \
        -DCMAKE_CXX_COMPILER=clang++-18 \
        -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -B build

    # Build (including FileCheck and not for StableHLO tests)
    cmake --build build -j "$JOBS"
    cmake --build build --target FileCheck -j "$JOBS"
    cmake --build build --target not -j "$JOBS"

    # Install
    cmake --install build

    # Install test utilities manually (they're not part of default install)
    cp build/bin/FileCheck "$DEPS_DIR/bin/" 2>/dev/null || true
    cp build/bin/not "$DEPS_DIR/bin/" 2>/dev/null || true

    # Patch AddLLVM.cmake to make test dependencies optional
    # This allows StableHLO to build without FileCheck/not as CMake targets
    ADDLLVM_FILE="$DEPS_DIR/lib/cmake/llvm/AddLLVM.cmake"
    if [ -f "$ADDLLVM_FILE" ]; then
        log_info "Patching AddLLVM.cmake for optional test dependencies..."
        # Replace the simple add_dependencies with a loop that checks if targets exist
        sed -i '2040,2042c\  if (ARG_DEPENDS)\n    foreach(dep IN LISTS ARG_DEPENDS)\n      if(TARGET ${dep})\n        add_dependencies(${target} ${dep})\n      endif()\n    endforeach()\n  endif()' "$ADDLLVM_FILE"
    fi

    log_success "LLVM/MLIR installed to $DEPS_DIR"
else
    log_info "Skipping LLVM/MLIR build (--skip-llvm)"
fi

# ============================================================
# 6. Build StableHLO
# ============================================================

if [ "$SKIP_STABLEHLO" = false ]; then
    log_info "Building StableHLO $STABLEHLO_VERSION..."

    cd "$BUILD_DIR"

    # Clone StableHLO if not present
    if [ ! -d "stablehlo" ]; then
        git clone --depth 1 --branch "$STABLEHLO_VERSION" \
            https://github.com/openxla/stablehlo.git
    fi

    cd stablehlo

    # Configure with external LLVM
    cmake -G Ninja . \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DIR="$DEPS_DIR/lib/cmake/llvm" \
        -DMLIR_DIR="$DEPS_DIR/lib/cmake/mlir" \
        -DCMAKE_C_COMPILER=clang-18 \
        -DCMAKE_CXX_COMPILER=clang++-18 \
        -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" \
        -B llvm-build

    # Build
    cmake --build llvm-build -j "$JOBS"

    # Install StableHLO libraries
    cp llvm-build/lib/libStablehlo*.a "$DEPS_DIR/lib/" 2>/dev/null || true
    cp llvm-build/lib/libChlo*.a "$DEPS_DIR/lib/" 2>/dev/null || true

    # Copy StableHLO headers
    mkdir -p "$DEPS_DIR/include/stablehlo"
    cp -r stablehlo "$DEPS_DIR/include/" 2>/dev/null || true

    log_success "StableHLO installed to $DEPS_DIR"
else
    log_info "Skipping StableHLO build (--skip-stablehlo)"
fi

# ============================================================
# 7. Build XLA PJRT Plugin
# ============================================================

if [ "$SKIP_XLA" = false ]; then
    log_info "Building XLA PJRT CPU Plugin..."

    cd "$BUILD_DIR"

    # Clone XLA if not present
    if [ ! -d "xla" ]; then
        git clone https://github.com/openxla/xla.git
    fi

    cd xla
    git fetch origin
    git checkout "$XLA_COMMIT"

    # Configure XLA
    python3 ./configure.py --backend=CPU

    # Add additional Bazel configuration
    cat >> xla_configure.bazelrc << 'EOF'

# Linux-specific configuration
build --cxxopt=-std=c++17
build --host_cxxopt=-std=c++17
EOF

    # Clean previous builds
    bazelisk shutdown || true
    bazelisk clean --expunge || true

    # Build PJRT plugin and proto libraries
    bazelisk --output_user_root="$BUILD_DIR/xla-bazelisk-build" \
        build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so \
               //xla/pjrt/proto:compile_options_proto

    # Find bazel output directory
    XLA_BAZEL_BIN=$(find "$BUILD_DIR/xla-bazelisk-build" -type d -name "k8-opt" -path "*/bazel-out/*" 2>/dev/null | head -1)
    if [ -z "$XLA_BAZEL_BIN" ]; then
        XLA_BAZEL_BIN="bazel-bin"
    else
        XLA_BAZEL_BIN="$XLA_BAZEL_BIN/bin"
    fi
    log_info "XLA bazel bin: $XLA_BAZEL_BIN"

    # Find and copy plugin
    PLUGIN_PATH=""
    if [ -f "bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" ]; then
        PLUGIN_PATH="bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    elif [ -f "$BUILD_DIR/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so" ]; then
        PLUGIN_PATH="$BUILD_DIR/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    else
        # Search in custom bazel output root
        PLUGIN_PATH=$(find "$BUILD_DIR/xla-bazelisk-build" -name "pjrt_c_api_cpu_plugin.so" -type f 2>/dev/null | head -1)
    fi

    if [ -n "$PLUGIN_PATH" ] && [ -f "$PLUGIN_PATH" ]; then
        cp "$PLUGIN_PATH" "$DEPS_DIR/lib/"
        log_success "PJRT CPU plugin installed to $DEPS_DIR/lib/"
    else
        log_warn "PJRT plugin not found - XLA build may have failed"
    fi

    # Copy XLA proto libraries for PJRTProtoHelper
    log_info "Copying XLA proto libraries..."
    mkdir -p "$DEPS_DIR/lib/xla"

    # Find and copy all proto .pic.a files
    find "$BUILD_DIR/xla-bazelisk-build" -name "*_proto*.pic.a" -path "*/xla/*" -exec cp {} "$DEPS_DIR/lib/xla/" \; 2>/dev/null || true
    find "$BUILD_DIR/xla-bazelisk-build" -name "*compile_options*.pic.a" -exec cp {} "$DEPS_DIR/lib/xla/" \; 2>/dev/null || true
    # Include TSL proto libraries (error_codes, etc.)
    find "$BUILD_DIR/xla-bazelisk-build" -path "*/tsl/*_proto*.pic.a" -exec cp {} "$DEPS_DIR/lib/xla/" \; 2>/dev/null || true

    # Copy protobuf libraries
    mkdir -p "$DEPS_DIR/lib/protobuf"
    find "$BUILD_DIR/xla-bazelisk-build" -path "*/com_google_protobuf/*.pic.a" -exec cp {} "$DEPS_DIR/lib/protobuf/" \; 2>/dev/null || true

    # Copy abseil libraries (need the log/libglobals.pic.a specifically)
    mkdir -p "$DEPS_DIR/lib/absl"
    find "$BUILD_DIR/xla-bazelisk-build" -path "*/com_google_absl/*.pic.a" -exec cp {} "$DEPS_DIR/lib/absl/" \; 2>/dev/null || true
    # Ensure we have the correct globals lib with logging functions
    GLOBALS_LIB=$(find "$BUILD_DIR/xla-bazelisk-build" -path "*/absl/log/libglobals.pic.a" -not -path "*/internal/*" 2>/dev/null | head -1)
    if [ -n "$GLOBALS_LIB" ]; then
        cp "$GLOBALS_LIB" "$DEPS_DIR/lib/absl/libglobals.pic.a"
    fi

    # Count copied libraries
    XLA_PROTO_COUNT=$(ls "$DEPS_DIR/lib/xla/"*.a 2>/dev/null | wc -l)
    PROTOBUF_COUNT=$(ls "$DEPS_DIR/lib/protobuf/"*.a 2>/dev/null | wc -l)
    ABSL_COUNT=$(ls "$DEPS_DIR/lib/absl/"*.a 2>/dev/null | wc -l)
    log_success "Copied $XLA_PROTO_COUNT XLA proto, $PROTOBUF_COUNT protobuf, $ABSL_COUNT absl libraries"

    # Copy PJRT headers
    mkdir -p "$DEPS_DIR/include/xla/pjrt/c"
    cp xla/pjrt/c/*.h "$DEPS_DIR/include/xla/pjrt/c/" 2>/dev/null || true

    # Copy proto headers (all XLA .pb.h files)
    log_info "Copying XLA proto headers..."
    XLA_BIN=$(find "$BUILD_DIR/xla-bazelisk-build" -type d -name "k8-opt" -path "*/bazel-out/*" 2>/dev/null | head -1)
    if [ -n "$XLA_BIN" ]; then
        XLA_BIN="$XLA_BIN/bin"
        find "$XLA_BIN" -name "*.pb.h" -path "*/xla/*" -exec sh -c '
            FILE="$1"; XLA_BIN="$2"; DEPS="$3"
            REL=$(echo "$FILE" | sed "s|$XLA_BIN/||")
            mkdir -p "$DEPS/include/$(dirname $REL)"
            cp "$FILE" "$DEPS/include/$REL"
        ' _ {} "$XLA_BIN" "$DEPS_DIR" \; 2>/dev/null || true
    fi

    # Copy protobuf headers
    log_info "Copying protobuf/absl headers..."
    PROTOBUF_SRC=$(find "$BUILD_DIR/xla-bazelisk-build" -path "*/external/com_google_protobuf/src/google/protobuf" -type d 2>/dev/null | head -1)
    if [ -n "$PROTOBUF_SRC" ]; then
        mkdir -p "$DEPS_DIR/include/google"
        cp -r "$PROTOBUF_SRC" "$DEPS_DIR/include/google/"
    fi

    # Copy abseil headers
    ABSL_SRC=$(find "$BUILD_DIR/xla-bazelisk-build" -path "*/external/com_google_absl/absl" -type d 2>/dev/null | head -1)
    if [ -n "$ABSL_SRC" ]; then
        cp -r "$ABSL_SRC" "$DEPS_DIR/include/"
    fi

    log_success "XLA PJRT plugin and proto libraries built"
else
    log_info "Skipping XLA/PJRT build (--skip-xla)"
fi

# ============================================================
# 8. Configure Environment
# ============================================================

log_info "Configuring environment..."

# Create SwiftIR environment file
sudo tee /etc/profile.d/swiftir.sh > /dev/null << EOF
# SwiftIR environment
export SWIFTIR_DEPS_DIR="$DEPS_DIR"
export LD_LIBRARY_PATH="$DEPS_DIR/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$DEPS_DIR/lib:\${LIBRARY_PATH:-}"
export CPATH="$DEPS_DIR/include:\${CPATH:-}"
export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
EOF

log_success "Environment configured in /etc/profile.d/swiftir.sh"

# ============================================================
# 9. Build SwiftIRMLIR Library
# ============================================================

log_info "Building SwiftIRMLIR shared library..."

SWIFTIR_CMAKE_DIR="$SCRIPT_DIR/../cmake"
if [ -d "$SWIFTIR_CMAKE_DIR" ]; then
    cd "$SWIFTIR_CMAKE_DIR"

    # Clean any previous build
    rm -rf build
    mkdir -p build
    cd build

    # Configure and build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake_config.log
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        cmake --build . --parallel $(nproc) 2>&1 | tee cmake_build.log
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            # Copy library to deps dir
            if [ -f "lib/libSwiftIRMLIR.so" ]; then
                cp lib/libSwiftIRMLIR.so "$DEPS_DIR/lib/"
                log_success "SwiftIRMLIR built and installed to $DEPS_DIR/lib/"
            else
                log_warn "SwiftIRMLIR library not found after build"
            fi
        else
            log_warn "SwiftIRMLIR build failed - check $SWIFTIR_CMAKE_DIR/build/cmake_build.log"
        fi
    else
        log_warn "SwiftIRMLIR cmake configure failed - check $SWIFTIR_CMAKE_DIR/build/cmake_config.log"
    fi

    cd "$SCRIPT_DIR"
else
    log_warn "SwiftIR cmake directory not found at $SWIFTIR_CMAKE_DIR"
fi

# ============================================================
# 10. Summary
# ============================================================

echo ""
echo "========================================"
echo "Installation Complete"
echo "========================================"
echo ""
echo "Dependencies installed to: $DEPS_DIR"
echo ""
echo "Contents:"
ls -la "$DEPS_DIR/lib/" | head -20
echo ""
echo "To use the environment, run:"
echo "  source /etc/profile.d/swiftir.sh"
echo ""
echo "Or log out and log back in."
echo ""
echo "To verify installation:"
echo "  ./scripts/verify-installation.sh"
echo ""
echo "To build SwiftIR:"
echo "  swift build"
echo ""
