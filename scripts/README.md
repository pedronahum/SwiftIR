# SwiftIR Installation Scripts

Scripts for setting up SwiftIR dependencies on Ubuntu 24.04.

## Quick Start

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Check prerequisites
./scripts/check-prerequisites.sh

# Install everything
./scripts/install-swiftir-ubuntu.sh

# Verify installation
./scripts/verify-installation.sh

# IMPORTANT: Set up environment before running examples
source /etc/profile.d/swiftir.sh
```

> **⚠️ Important:** After installation, you must source the environment file or set `LD_LIBRARY_PATH` before running SwiftIR examples. See [Environment Variables](#environment-variables) section below.

## Scripts

### check-prerequisites.sh

Verifies system requirements before installation:
- Ubuntu version
- Build tools (cmake, ninja, git, etc.)
- Compilers (clang-18, clang++-18)
- Swift installation
- Required libraries
- Bazelisk for XLA builds

### install-swiftir-ubuntu.sh

Main installation script that:
1. Installs system dependencies (apt packages)
2. Installs Bazelisk
3. Installs Swift via Swiftly
4. Builds LLVM/MLIR from source
5. Builds StableHLO from source
6. Builds XLA PJRT CPU plugin
7. Configures environment variables

**Options:**
```bash
--skip-swift       Skip Swift installation
--skip-llvm        Skip LLVM/MLIR build
--skip-stablehlo   Skip StableHLO build
--skip-xla         Skip XLA/PJRT build
--skip-deps        Skip apt dependencies
--swift-version    Swift version (default: main-snapshot-2025-11-03)
--jobs N           Parallel build jobs (default: nproc)
```

**Examples:**
```bash
# Full installation
./scripts/install-swiftir-ubuntu.sh

# Use specific Swift version
./scripts/install-swiftir-ubuntu.sh --swift-version swift-6.0-RELEASE

# Rebuild only XLA
./scripts/install-swiftir-ubuntu.sh --skip-deps --skip-swift --skip-llvm --skip-stablehlo

# Use fewer cores (e.g., on memory-constrained systems)
./scripts/install-swiftir-ubuntu.sh --jobs 4
```

### verify-installation.sh

Post-installation verification:
- Swift installation and C++ interop
- LLVM/MLIR libraries and tools
- StableHLO libraries
- PJRT CPU plugin
- Environment variables
- C++ compilation test

## Dependencies Installed

### System Packages (apt)

**Build Tools:**
- build-essential, cmake, ninja-build
- git, wget, curl, pkg-config

**Compilers:**
- clang-18, llvm-18-dev
- libc++-18-dev, libc++abi-18-dev, lld-18

**Swift Requirements:**
- libc6-dev, libstdc++-13-dev
- libicu-dev, libssl-dev, libxml2-dev
- libcurl4-openssl-dev, libz-dev
- libsqlite3-dev, libpython3-dev

**MLIR Build:**
- libzstd-dev, libncurses-dev, zlib1g-dev

**XLA/Bazel:**
- python3, python3-pip, python3-numpy
- bazelisk

### Built from Source

| Component | Version | Location |
|-----------|---------|----------|
| LLVM/MLIR | 19.1.6 | /opt/swiftir-deps |
| StableHLO | v1.13.0 | /opt/swiftir-deps |
| XLA PJRT | main | /opt/swiftir-deps |

## Environment Variables

After installation, `/etc/profile.d/swiftir.sh` exports:

```bash
SWIFTIR_DEPS_DIR=/opt/swiftir-deps
LD_LIBRARY_PATH=/opt/swiftir-deps/lib:$LD_LIBRARY_PATH
LIBRARY_PATH=/opt/swiftir-deps/lib:$LIBRARY_PATH
CPATH=/opt/swiftir-deps/include:$CPATH
```

Source the environment or log out/in:
```bash
source /etc/profile.d/swiftir.sh
```

## Configuration

### Custom Installation Directory

Set `SWIFTIR_DEPS_DIR` before running:
```bash
export SWIFTIR_DEPS_DIR=/home/user/swiftir-deps
./scripts/install-swiftir-ubuntu.sh
```

### Custom Swift Toolchain

```bash
# Use a different snapshot
./scripts/install-swiftir-ubuntu.sh --swift-version main-snapshot-2025-11-10

# Use release version
./scripts/install-swiftir-ubuntu.sh --swift-version swift-6.0-RELEASE
```

### Build Directory

Set `BUILD_DIR` for source builds:
```bash
export BUILD_DIR=/path/to/builds
./scripts/install-swiftir-ubuntu.sh
```

## Troubleshooting

### LLVM/MLIR Build Fails

- **Out of memory:** Reduce parallel jobs: `--jobs 4`
- **Disk space:** Need ~20GB for full build

### XLA/PJRT Build Fails

- **Bazel cache issues:** Clear with `bazelisk clean --expunge`
- **Python errors:** Ensure python3 and numpy are installed

### Swift Build Errors

Check paths in `Package.swift` match your installation:
```swift
let stablehloRoot = "/opt/swiftir-deps"  // Update to match SWIFTIR_DEPS_DIR
```

### Library Not Found at Runtime

Ensure environment is loaded:
```bash
source /etc/profile.d/swiftir.sh
echo $LD_LIBRARY_PATH  # Should include /opt/swiftir-deps/lib
```

## Platform Notes

### Differences from macOS

| Aspect | macOS | Ubuntu |
|--------|-------|--------|
| Package manager | brew | apt |
| Library path | DYLD_LIBRARY_PATH | LD_LIBRARY_PATH |
| Dynamic lib ext | .dylib | .so |
| RPATH | @loader_path | $ORIGIN |
| CPU count | sysctl -n hw.ncpu | nproc |

### Package.swift Updates

The Package.swift file needs updates for Linux:
- Change hardcoded `/Users/pedro/...` paths to environment variable lookups
- Update rpath from `@loader_path` to `$ORIGIN`
- Update plugin extension from `.dylib` to `.so`

## Build Times

Approximate times on modern hardware (16 cores):

| Component | Time |
|-----------|------|
| LLVM/MLIR | 30-60 min |
| StableHLO | 5-10 min |
| XLA PJRT | 15-30 min |
| **Total** | ~1-2 hours |

## Updating

To update dependencies:

```bash
# Update Swift
swiftly install main-snapshot-2025-11-10
swiftly use main-snapshot-2025-11-10

# Rebuild specific component
./scripts/install-swiftir-ubuntu.sh --skip-deps --skip-swift --skip-llvm --skip-stablehlo
```
