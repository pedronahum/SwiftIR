# SwiftIR Jupyter/Colab Integration

## Summary

This document describes the technical challenges and solutions for using SwiftIR in Jupyter notebooks (via swift-jupyter) and Google Colab.

## SwiftIR Library Components

SwiftIR consists of multiple interconnected targets:

| Target | Dependencies | Native Code | Jupyter Status |
|--------|--------------|-------------|----------------|
| **SwiftIRRuntime** | None (pure Swift) | No | ✅ Works (source compilation) |
| **SwiftIRCore** | MLIRCoreWrappers, PJRTCWrappers | Yes (C++ interop, links libSwiftIRMLIR.so) | ❌ Requires native libs |
| **SwiftIRTypes** | SwiftIRCore | Yes | ❌ Requires native libs |
| **SwiftIRDialects** | SwiftIRCore, SwiftIRTypes | Yes | ❌ Requires native libs |
| **SwiftIRBuilders** | SwiftIRCore, SwiftIRTypes, SwiftIRDialects | Yes | ❌ Requires native libs |
| **SwiftIRStableHLO** | SwiftIRCore, SwiftIRTypes, SwiftIRDialects | Yes | ❌ Requires native libs |
| **SwiftIRXLA** | SwiftIRCore, SwiftIRTypes, SwiftIRDialects, SwiftIRStableHLO, PJRTCWrappers | Yes | ❌ Requires native libs |
| **SwiftIR** | All above + SwiftIRMacros | Yes | ❌ Requires native libs |

## Root Cause Analysis

### Why Pre-built `.swiftmodule` Files Don't Work

Swift modules are **not ABI-stable** across compiler versions. When you build a `.swiftmodule` with Swift 6.3-dev (build A), it won't work with Swift 6.3-dev (build B) even if they're the same version string.

LLDB (which swift-jupyter uses) requires:
1. `.swiftmodule` - Type information
2. `.swiftdoc` - Documentation (required by LLDB)
3. **Matching compiler version** - The exact same Swift build

This is why `import SwiftIRRuntime` appears to succeed but `AcceleratorType` is "not found in module" - LLDB can see the module exists but can't deserialize the type information.

### What Works

1. **Source compilation** - Compiling Swift source files directly in the notebook works because it uses the same Swift compiler that the kernel is running.

2. **%install directive** - swift-jupyter's `%install` builds packages from source using SwiftPM, creating compatible `.swiftmodule` and `.so` files.

## Solution for SwiftIRRuntime (Pure Swift)

SwiftIRRuntime has no native dependencies and can be compiled on-the-fly:

```swift
%swiftir_setup /opt/swiftir-deps

// Types are now available:
print(AcceleratorType.cpu)
RuntimeDetector.printInfo()
```

The `%swiftir_setup` directive:
1. Sets library paths (LD_LIBRARY_PATH)
2. Sets module search paths
3. **Compiles `swift-sources/SwiftIRRuntime/*.swift` files** into the REPL

## Challenge for Full SwiftIR (C++ Interop)

The full SwiftIR library uses **C++ interop** with MLIR/LLVM. This creates additional challenges:

### 1. C++ Interop Requires Headers at Compile Time

SwiftIR targets use `.interoperabilityMode(.Cxx)` and include paths like:
```swift
.unsafeFlags([
    "-I", "\(llvmBuildDir)/include",
    "-I", "\(llvmProjectDir)/mlir/include",
    "-I", "\(stablehloRoot)/stablehlo",
])
```

These headers must be present when compiling, not just the `.swiftmodule`.

### 2. Linker Flags for Native Libraries

SwiftIR links against `libSwiftIRMLIR.so` (our MLIR wrapper):
```swift
linkerSettings: [
    .unsafeFlags([
        "-L\(cmakeBuildDir)/lib",
        "-lSwiftIRMLIR",
    ]),
]
```

### 3. System Libraries

The build requires system libraries:
- `libzstd`
- `libz`
- `libcurses`
- Various LLVM/MLIR libraries

## Possible Approaches for Full SwiftIR in Jupyter

### Approach 1: Pre-built Dynamic Libraries (Complex)

Build SwiftIR as dynamic libraries with all symbols exported:

```swift
.library(
    name: "SwiftIRDynamic",
    type: .dynamic,
    targets: ["SwiftIR", "SwiftIRCore", "SwiftIRTypes", ...]
),
```

**Challenges:**
- Need to ship ALL MLIR/LLVM headers (~500MB)
- Need matching Swift compiler version
- C++ interop modules are especially fragile

### Approach 2: %install from Git (Not Viable - C++ Interop Limitation)

Swift-jupyter's `%install` directive builds packages from source, which works:

```swift
%install-swiftpm-env SWIFTIR_USE_SDK=1 SWIFTIR_DEPS=/opt/swiftir-deps
%install '.package(path: "/opt/swiftir-deps/SwiftIR")' SwiftIR
```

**The package builds successfully**, but importing fails:

```
error: module 'SwiftIR' was built with C++ interoperability enabled,
but current compilation does not enable C++ interoperability
```

**Root Cause:** LLDB's REPL (which swift-jupyter uses) does not support C++ interoperability.
When a Swift module is compiled with `.interoperabilityMode(.Cxx)`, LLDB cannot import it
because there's no API to enable C++ interop in `SBExpressionOptions`.

This is a fundamental limitation of Swift's LLDB REPL, not something we can work around.

**Changes Made (for SDK mode support, if LLDB ever adds C++ interop):**
- Added `%install-swiftpm-env` directive to swift_kernel.py
- Fixed Package.swift include paths for SDK mode
- Added `SwiftIRXLA` and `SwiftIRStableHLO` as dependencies of main `SwiftIR` target

### Approach 3: Hybrid Approach (Most Practical)

Use pre-built native libraries with source compilation for Swift layer:

1. **Ship pre-built native libraries:**
   - `libSwiftIRMLIR.so` (C++ MLIR wrapper)
   - `libPJRTProtoHelper.so`
   - `pjrt_c_api_cpu_plugin.so`
   - All LLVM/MLIR `.so` files

2. **Ship Swift source files:**
   - All `Sources/SwiftIR*/*.swift` files
   - Include paths for headers

3. **Compile Swift code in notebook:**
   ```swift
   %swiftir_setup /opt/swiftir-deps  // Loads libs, sets paths
   %swiftir_compile_all              // Compiles Swift sources
   ```

**Implementation Required:**
- Add `%swiftir_compile_all` directive to swift_kernel.py
- Ship LLVM/MLIR headers in SDK (~100MB compressed)
- Ensure all include paths are set correctly

### Approach 4: WebAssembly / Remote Execution (Future)

Compile SwiftIR to WASM or use a remote execution service.

## Current Status

| Component | Jupyter Support | Notes |
|-----------|-----------------|-------|
| SwiftIRRuntime | ✅ Working | Source compilation via `%swiftir_setup` |
| SwiftIRCore | ❌ Not yet | Needs headers + native libs |
| SwiftIRTypes | ❌ Not yet | Depends on SwiftIRCore |
| SwiftIRDialects | ❌ Not yet | Depends on SwiftIRCore |
| SwiftIRBuilders | ❌ Not yet | Depends on SwiftIRDialects |
| SwiftIRStableHLO | ❌ Not yet | Needs StableHLO headers |
| SwiftIRXLA | ❌ Not yet | Needs all above |
| SwiftIR (main) | ❌ Not yet | Needs all above + macros |

## Recommended Path Forward

### Phase 1: SwiftIRRuntime Only (DONE)
- ✅ Runtime detection works
- ✅ AcceleratorType, RuntimeDetector available
- ✅ PJRTPlugin can load plugins (if libs present)

### Phase 2: Add Pre-execution Support
For users who want to **run** pre-compiled executables (not write new code):

```swift
// In notebook - execute pre-built example
!./SwiftIR-sdk/bin/PJRT_Add_Example
```

### Phase 3: Full Compilation Support (Future)
1. Ship complete headers (~100MB)
2. Implement `%swiftir_compile_all` directive
3. Handle C++ interop in REPL context

## Files Changed

### swift-jupyter/swift_kernel.py
- Added `%swiftir_setup` directive with source compilation support
- Compiles `swift-sources/SwiftIRRuntime/*.swift` files on-the-fly

### SwiftIR/Package.swift
- Added `SwiftIRRuntimeDynamic` product (dynamic library)

### CI Workflows
- `ubuntu-build-dependencies.yml`: Copies `.swiftdoc` and source files
- `build-release.yml`: Copies `.swiftdoc` and source files

## Testing

```swift
// In Jupyter/Colab
%swiftir_setup /opt/swiftir-deps

// Test runtime detection
print(AcceleratorType.cpu)           // CPU
print(RuntimeDetector.detect())      // CPU
RuntimeDetector.printInfo()          // Full info

// These will NOT work until Phase 3:
// import SwiftIR                    // Error: needs native libs
// let ctx = MLIRContext()           // Error: needs native libs
```

## SDK Structure

The current SDK (`/opt/swiftir-deps`) contains:

```
/opt/swiftir-deps/
├── lib/                          # 1.3GB - Dynamic libraries
│   ├── libSwiftIRMLIR.so         # Our C++ MLIR wrapper
│   ├── libPJRTProtoHelper.so     # PJRT helper
│   ├── pjrt_c_api_cpu_plugin.so  # CPU execution plugin
│   ├── libLLVM.so                # LLVM runtime
│   ├── libMLIR.so                # MLIR runtime
│   └── ...                       # Other LLVM/MLIR libs
│
├── include/                      # 168MB - Headers (REQUIRED for C++ interop)
│   ├── llvm/                     # LLVM headers
│   ├── llvm-c/                   # LLVM C API
│   ├── mlir/                     # MLIR headers
│   ├── mlir-c/                   # MLIR C API
│   └── stablehlo/                # StableHLO headers
│
├── swift-modules/                # Pre-built .swiftmodule/.swiftdoc
│   └── (NOT USABLE in Jupyter due to compiler version mismatch)
│
└── swift-sources/                # ✅ Source files for on-the-fly compilation
    └── SwiftIRRuntime/
        ├── AcceleratorType.swift
        ├── RuntimeDetector.swift
        ├── PJRTPlugin.swift
        ├── PJRTClientUnified.swift
        └── Errors.swift
```

## What's Needed for Full SwiftIR in Jupyter

### Required Components

| Component | Size | Status | Purpose |
|-----------|------|--------|---------|
| `lib/*.so` | 1.3GB | ✅ Shipped | Runtime libraries |
| `include/` | 168MB | ✅ Shipped | C++ headers for compilation |
| `swift-sources/SwiftIRRuntime/` | 40KB | ✅ Shipped | Pure Swift runtime |
| `swift-sources/SwiftIRCore/` | 25KB | ❌ Not shipped | Needs headers at compile |
| `swift-sources/SwiftIR*/` | ~500KB | ❌ Not shipped | Full library sources |

### The C Module Challenge

SwiftIR uses C modules to interface with MLIR. For example, `MLIRCoreWrapper.h` includes:

```c
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir-c/BuiltinTypes.h>
// ... 20+ more MLIR headers
```

For these to compile in Jupyter:
1. Headers must be at `/opt/swiftir-deps/include/mlir-c/`
2. Module map must be available
3. Swift compiler flags must include `-I /opt/swiftir-deps/include`

### Implementation Plan for Full Support

**Step 1:** Ship complete sources
```yaml
# In CI workflow
cp -r Sources/SwiftIRCore /opt/swiftir-deps/swift-sources/
cp -r Sources/SwiftIRTypes /opt/swiftir-deps/swift-sources/
# ... etc
```

**Step 2:** Create module maps for C modules
```
// swift-sources/MLIRCoreWrapper/module.modulemap
module MLIRCoreWrapper {
    header "/opt/swiftir-deps/include/SwiftIR/MLIRCoreWrapper.h"
    export *
}
```

**Step 3:** Update `%swiftir_setup` to:
1. Set include paths: `-I /opt/swiftir-deps/include`
2. Set library paths: `-L /opt/swiftir-deps/lib`
3. Load module maps
4. Compile sources in dependency order

**Step 4:** Handle linking
```swift
// In kernel, after compiling sources:
dlopen("/opt/swiftir-deps/lib/libSwiftIRMLIR.so", RTLD_NOW)
dlopen("/opt/swiftir-deps/lib/libMLIR.so", RTLD_NOW)
```

## Alternative: Use %install (NOT WORKING - LLDB Limitation)

We implemented `%install` support for SDK mode, but it **does not work** due to LLDB's
lack of C++ interoperability support:

```swift
// This builds successfully but fails at import time
%install-location /opt/swiftir-jupyter-packages
%install-swiftpm-flags -Xswiftc -I/opt/swiftir-deps/include -Xlinker -L/opt/swiftir-deps/lib -Xlinker -rpath -Xlinker /opt/swiftir-deps/lib
%install-swiftpm-env SWIFTIR_USE_SDK=1 SWIFTIR_DEPS=/opt/swiftir-deps
%install '.package(path: "/opt/swiftir-deps/SwiftIR")' SwiftIR

import SwiftIR  // ❌ ERROR: module was built with C++ interoperability enabled
```

The infrastructure is ready for when Swift/LLDB adds C++ interop support:
1. ✅ Full SwiftIR source code in SDK
2. ✅ Package.swift configured for SDK mode
3. ✅ All headers and libraries present
4. ✅ `%install-swiftpm-env` directive added to swift-jupyter

## Conclusion

**SwiftIRRuntime** (pure Swift) works in Jupyter via source compilation.

**Full SwiftIR** (C++ interop) **CANNOT work in Jupyter/Colab** due to LLDB limitations:
- LLDB's REPL does not support C++ interoperability
- There is no `SBExpressionOptions` API to enable C++ interop
- This is a fundamental Swift toolchain limitation, not a swift-jupyter limitation

### What We Tested

1. **%install approach**: Package builds successfully in SDK mode, but fails at import time
   with "module was built with C++ interoperability enabled" error.

2. **Pre-built modules**: Swift .swiftmodule files are not ABI-stable across compiler versions,
   so pre-built modules don't work with different Swift builds.

3. **Source compilation**: Works for pure Swift (SwiftIRRuntime), but C++ interop modules
   cannot be compiled in LLDB REPL context.

### Recommended Approach

The only viable approach for full SwiftIR functionality in notebooks is:

1. **Use SwiftIRRuntime** in notebooks for:
   - Runtime detection (`RuntimeDetector.detect()`)
   - Accelerator information (`AcceleratorType`)
   - Plugin discovery (`PJRTPlugin.loadPlugin()`)

2. **Run pre-built executables** for actual computation:
   ```swift
   !./SwiftIR-sdk/bin/PJRT_Add_Example
   !./SwiftIR-sdk/bin/BuildingSimulation_SwiftIR
   ```

3. **For development**, use regular Swift compilation (not notebook)

### Future Possibilities

1. **Swift REPL improvements**: If Swift adds C++ interop support to LLDB REPL,
   the `%install` approach we implemented would work immediately.

2. **WebAssembly**: Compile SwiftIR to WASM for browser-based execution.

3. **Remote execution service**: Send SwiftIR code to a backend server for compilation
   and execution, returning results to the notebook.

## Next Steps

1. ✅ **Done:** SwiftIRRuntime works via source compilation
2. ✅ **Done:** Tested `%install` approach - confirmed LLDB C++ interop limitation
3. ✅ **Done:** Package.swift SDK mode support ready for future use
4. **Future:** Monitor Swift evolution for LLDB C++ interop support
