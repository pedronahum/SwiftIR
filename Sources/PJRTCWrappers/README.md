# PJRTCWrappers

**C wrappers for PJRT (Plugin JAX Runtime) C API**

## Purpose

PJRTCWrappers provides Swift-friendly C wrappers around the PJRT C API, simplifying interop between Swift and the XLA runtime. It bridges the gap between Swift's type system and PJRT's C interface.

## What's Inside

### C Headers (`include/`)
- `pjrt_c_api.h` - Complete PJRT C API specification
  - Client creation and management
  - Device enumeration
  - Buffer management
  - Compilation and execution
  - Error handling

- `pjrt_c_api_cpu.h` - CPU-specific PJRT plugin entry point
  - CPU plugin initialization
  - Platform-specific configuration

- `PJRTSimpleWrapper.h` - Simplified C wrapper for Swift interop
  - Enum-to-integer conversions
  - Simplified error handling
  - Opaque pointer wrappers
  - Swift-friendly function signatures

- `PJRTProtoHelper.h` - Protocol buffer helpers
  - Serialization/deserialization
  - Message construction utilities

- `module.modulemap` - Clang module map for Swift import

## What is PJRT?

**PJRT (Plugin JAX Runtime)** is a standard C API for ML runtime execution developed by Google.

### Purpose
- Vendor-neutral interface for ML accelerators
- Supports CPU, GPU, TPU, and custom hardware
- Used by JAX, PyTorch/XLA, and TensorFlow

### Architecture
```
ML Framework (JAX, PyTorch, TensorFlow)
           ↓
      PJRT C API (stable interface)
           ↓
    PJRT Plugin (vendor-specific)
           ↓
  Hardware (CPU, GPU, TPU, etc.)
```

## Why C Wrappers?

The raw PJRT C API is designed for C/C++ and has patterns that don't map well to Swift:

### Raw C API Challenges
1. Complex struct initialization
2. Manual memory management
3. Enum type mismatches
4. Nested callbacks
5. Opaque pointer casting

### PJRTSimpleWrapper Solutions
1. Simple init functions
2. Explicit create/destroy pairs
3. Int32 for enums (Swift-friendly)
4. Simplified callback signatures
5. Typed opaque pointers

## Example: Error Handling

### Raw C API
```c
typedef struct {
    PJRT_Error_Code code;
    char* message;
    // ... more fields
} PJRT_Error;

PJRT_Error* error = SomeFunction();
if (error) {
    // Complex error extraction
}
```

### PJRTSimpleWrapper
```c
typedef enum {
    SW_PJRT_Error_OK = 0,
    SW_PJRT_Error_INVALID_ARGUMENT = 3,
    // ... more cases
} SW_PJRT_Error_Code;

SW_PJRT_Error_Code error = SimplifiedFunction();
if (error != SW_PJRT_Error_OK) {
    // Simple integer comparison
}
```

## Module Organization

### Layer 1: Raw PJRT API
`pjrt_c_api.h` and `pjrt_c_api_cpu.h`
- Direct from XLA project
- Unmodified official headers
- Complete API specification

### Layer 2: Simple Wrappers
`PJRTSimpleWrapper.h`
- Swift-friendly wrappers
- Simplified types
- Clear ownership semantics

### Layer 3: Protocol Helpers
`PJRTProtoHelper.h`
- Helpers for protobuf serialization
- Message construction
- Type conversions

### Layer 4: Swift API
In `SwiftIRXLA/PJRT/`
- High-level Swift classes
- Automatic memory management
- Swifty error handling

## Usage from Swift

```swift
import PJRTCWrappers

// Create CPU client (via wrapper)
let client: OpaquePointer = pjrt_create_cpu_client()

// Get device count
var deviceCount: Int32 = 0
let error = pjrt_client_get_device_count(client, &deviceCount)

guard error == SW_PJRT_Error_OK else {
    // Handle error
}

print("Devices: \(deviceCount)")

// Cleanup
pjrt_client_destroy(client)
```

## Key Functions (Simplified API)

### Client Management
```c
// Create client for specific platform
void* pjrt_create_cpu_client();
void* pjrt_create_gpu_client();
void pjrt_client_destroy(void* client);
```

### Device Query
```c
int32_t pjrt_client_get_device_count(void* client, int32_t* count);
int32_t pjrt_get_device(void* client, int32_t index, void** device);
```

### Buffer Operations
```c
int32_t pjrt_buffer_from_host_data(
    void* client,
    void* data,
    size_t size,
    int32_t dtype,
    int64_t* dims,
    size_t ndims,
    void** buffer
);

int32_t pjrt_buffer_to_host(void* buffer, void* data, size_t size);
void pjrt_buffer_destroy(void* buffer);
```

### Compilation and Execution
```c
int32_t pjrt_compile(
    void* client,
    const char* mlir_module,
    void** executable
);

int32_t pjrt_execute(
    void* executable,
    void** input_buffers,
    size_t num_inputs,
    void*** output_buffers,
    size_t* num_outputs
);

void pjrt_executable_destroy(void* executable);
```

## Memory Management Pattern

All wrapper functions follow a consistent pattern:

```c
// Creation returns opaque pointer
void* resource = pjrt_create_something(...);

// Usage returns error code
int32_t error = pjrt_use_resource(resource, ...);

// Destruction is explicit
pjrt_resource_destroy(resource);
```

This maps cleanly to Swift's manual memory management or can be wrapped in Swift classes with `deinit`.

## Dependencies

### External
- XLA PJRT implementation
  - CPU plugin: `pjrt_c_api_cpu_plugin.so`
  - GPU plugin: `pjrt_c_api_gpu_plugin.so`
  - TPU plugin: `pjrt_c_api_tpu_plugin.so`

### Internal
- None (this is a pure C layer)

## Build Notes

### Module Map
The `module.modulemap` file allows Swift to import C headers:
```
module PJRTCWrappers {
    header "pjrt_c_api.h"
    header "pjrt_c_api_cpu.h"
    header "PJRTSimpleWrapper.h"
    header "PJRTProtoHelper.h"
    export *
}
```

### Linking
When using PJRT, you must link against the plugin library:
```swift
// Package.swift
.linkedLibrary("PJRTProtoHelper")
// Plus runtime loading of plugin .so/.dylib
```

## Platform Support

### CPU (Always Available)
- `pjrt_c_api_cpu.h`
- Single-threaded or multi-threaded execution
- Useful for debugging and development

### GPU (NVIDIA, AMD)
- CUDA plugin for NVIDIA
- ROCm plugin for AMD
- Requires driver installation

### TPU (Google Cloud)
- Google's custom ML accelerator
- Only available on Google Cloud Platform

## Related Modules

- **SwiftIRCore** - Uses similar C wrapper pattern
- **SwiftIRXLA/PJRT/** - Swift API built on these wrappers
- **SwiftIRXLA/Backend/** - XLA integration using PJRT

## Design Principles

### Simplicity
Wrappers should be as simple as possible while maintaining full functionality.

### Ownership Clarity
Every resource has clear create/destroy pairs.

### Error Handling
Consistent int32 error codes throughout.

### Swift Compatibility
All types chosen for smooth Swift interop:
- `void*` for opaque pointers
- `int32_t` for enums and error codes
- `size_t` for sizes
- Simple structs without unions

---

**Next Steps**: See [SwiftIRXLA/PJRT](../SwiftIRXLA/) for Swift API using these wrappers.
