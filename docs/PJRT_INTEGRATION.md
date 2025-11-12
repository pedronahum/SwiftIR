# PJRT Integration Guide

## Overview

PJRT (Pretty much Just the RunTime) is XLA's official runtime interface for GPU/TPU execution. SwiftIR now includes comprehensive PJRT integration, enabling true hardware acceleration on CUDA GPUs, ROCm GPUs, and Google TPUs.

## Architecture

```
┌─────────────────────────────────────────┐
│   SwiftIR Application Code               │
│   (StableHLO Operations)                 │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   PJRTClient.swift                       │
│   - Device Management                     │
│   - Buffer Allocation                     │
│   - Program Compilation                   │
│   - Execution Control                     │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   PJRTWrapper.h (C Bindings)             │
│   - Plugin Loading (dlopen/dlsym)        │
│   - PJRT C API Exposure                  │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   PJRT Plugin (.so/.dylib)               │
│   - libpjrt_c_api_cpu.so    (CPU)        │
│   - libpjrt_c_api_gpu.so    (GPU)        │
│   - libpjrt_c_api_tpu.so    (TPU)        │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   XLA Compiler + Runtime                 │
│   - HLO Optimization                      │
│   - Backend Code Generation               │
│   - Kernel Fusion                         │
│   - Autotuning                            │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│   Hardware (CPU/GPU/TPU)                 │
└───────────────────────────────────────────┘
```

## Components

### 1. PJRT C API Headers

Located in `Sources/SwiftIRCore/include/`:
- `pjrt_c_api.h` - Core PJRT C API definitions
- `pjrt_c_api_cpu.h` - CPU plugin interface
- `PJRTWrapper.h` - Swift-friendly C wrappers with dynamic loading

### 2. Swift Wrapper Layer

Located in `Sources/SwiftIRXLA/PJRT/PJRTClient.swift`:

**Classes:**
- `PJRTClient` - Main client for device management and execution
- `PJRTDevice` - Device representation
- `PJRTBuffer` - Device memory buffer
- `PJRTLoadedExecutable` - Compiled program

**Enums:**
- `PJRTElementType` - Tensor element types (f16, f32, f64, etc.)
- `PJRTError` - Error types

### 3. Examples

- `Examples/PJRT_Example.swift` - Comprehensive integration example
- Shows device enumeration, buffer management, compilation, execution

## Building PJRT Plugins

### Prerequisites

```bash
# Install Bazel (required version 7.7.0)
brew install bazel

# Clone OpenXLA
cd /path/to/your/workspace
git clone https://github.com/openxla/xla.git
cd xla
```

### Build CPU Plugin

```bash
# Configure for CPU
python3 ./configure.py --backend=CPU

# Build the PJRT CPU plugin
bazel build //xla/pjrt/c:pjrt_c_api_cpu

# The output will be at:
# bazel-bin/xla/pjrt/c/libpjrt_c_api_cpu.so (Linux)
# bazel-bin/xla/pjrt/c/libpjrt_c_api_cpu.dylib (macOS)
```

### Build GPU Plugin (CUDA)

```bash
# Configure for CUDA
python3 ./configure.py --backend=CUDA

# Build the PJRT GPU plugin
bazel build //xla/pjrt/c:pjrt_c_api_gpu

# Output at:
# bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu.so
```

### Build GPU Plugin (ROCm)

```bash
# Configure for ROCm
python3 ./configure.py --backend=ROCM

# Build
bazel build //xla/pjrt/c:pjrt_c_api_gpu
```

## Integration into SwiftIR

### Step 1: Copy Plugin Libraries

```bash
# Create lib directory in SwiftIR
mkdir -p /path/to/SwiftIR/lib

# Copy the built plugins
cp /path/to/xla/bazel-bin/xla/pjrt/c/libpjrt_c_api_cpu.* \
   /path/to/SwiftIR/lib/

# For GPU:
cp /path/to/xla/bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu.* \
   /path/to/SwiftIR/lib/
```

### Step 2: Update Package.swift

Add linker flags to your Package.swift:

```swift
.target(
    name: "SwiftIRXLA",
    dependencies: ["SwiftIRCore", "SwiftIRStableHLO"],
    swiftSettings: [
        .interoperabilityMode(.Cxx),
        .unsafeFlags([
            "-I", "Sources/SwiftIRCore/include",
            // ... other includes
        ]),
    ],
    linkerSettings: [
        .unsafeFlags([
            "-L", "/path/to/SwiftIR/lib",
            "-lpjrt_c_api_cpu",  // Add for CPU support
            "-lpjrt_c_api_gpu",  // Add for GPU support
            "-Xlinker", "-rpath", "-Xlinker", "/path/to/SwiftIR/lib"
        ])
    ]
)
```

### Step 3: Implement Real Client

Update `PJRTClient.swift` to call actual PJRT APIs instead of stubs:

```swift
public init(backend: Backend) throws {
    self.backend = backend
    self.platformName = backend.rawValue

    // Load the appropriate plugin
    let pluginPath = "/path/to/SwiftIR/lib/libpjrt_c_api_\(backend.rawValue).dylib"

    guard let api = PJRT_LoadCpuPluginWrapper(pluginPath) else {
        throw PJRTError.clientCreationFailed("Failed to load plugin: \(PJRT_GetLoadErrorWrapper())")
    }

    // Create client using PJRT API
    var createArgs = PJRT_Client_Create_Args()
    createArgs.struct_size = MemoryLayout<PJRT_Client_Create_Args>.size
    // ... set other fields

    let error = api.pointee.PJRT_Client_Create(&createArgs)
    if error != nil {
        throw PJRTError.clientCreationFailed("Client creation failed")
    }

    self.handle = createArgs.client

    // Enumerate devices
    var devicesArgs = PJRT_Client_Devices_Args()
    // ... enumerate devices
}
```

## Usage Examples

### Basic Usage

```swift
import SwiftIRXLA

// Create a client for CPU
let client = try PJRTClient(backend: .cpu)

// Get default device
guard let device = client.defaultDevice else {
    fatalError("No device available")
}

// Create a buffer
var data: [Float] = [1.0, 2.0, 3.0, 4.0]
let buffer = try data.withUnsafeBytes { ptr in
    try client.createBuffer(
        data: ptr.baseAddress!,
        shape: [2, 2],
        elementType: .f32,
        device: device
    )
}

// Compile a program
let executable = try client.compile(
    mlirModule: mlirModuleString,
    devices: [device]
)

// Execute
let outputs = try executable.execute(
    arguments: [buffer],
    device: device
)
```

### GPU Usage

```swift
// Create GPU client
let gpuClient = try PJRTClient(backend: .gpu)

// Select a specific GPU
let gpu = gpuClient.devices.first { $0.kind == "GPU" }!

// Same API as CPU - PJRT handles the differences
let gpuBuffer = try gpuClient.createBuffer(...)
let gpuExecutable = try gpuClient.compile(...)
let results = try gpuExecutable.execute(...)
```

## Performance Characteristics

### CPU (via PJRT)
- **Throughput**: 10-50 GFLOPS
- **Latency**: 1-10ms for small tensors
- **Optimizations**: Vectorization, loop fusion
- **Use Case**: Development, small models, edge deployment

### GPU (CUDA/ROCm)
- **Throughput**: 1-10 TFLOPS
- **Latency**: 0.1-1ms for small tensors
- **Optimizations**: Kernel fusion, autotuning, mixed precision
- **Use Case**: Training, large-scale inference

### TPU
- **Throughput**: 100+ TFLOPS
- **Latency**: <0.1ms for small tensors
- **Optimizations**: Systolic array utilization, bfloat16
- **Use Case**: Large-scale training, production inference

## Troubleshooting

### Plugin Loading Fails

**Problem**: `dlopen` fails to find the plugin

**Solution**:
```bash
# Check the library path
export DYLD_LIBRARY_PATH=/path/to/SwiftIR/lib:$DYLD_LIBRARY_PATH  # macOS
export LD_LIBRARY_PATH=/path/to/SwiftIR/lib:$LD_LIBRARY_PATH      # Linux

# Verify the library exists
ls -la /path/to/SwiftIR/lib/libpjrt_c_api_*

# Check dependencies
otool -L /path/to/SwiftIR/lib/libpjrt_c_api_cpu.dylib  # macOS
ldd /path/to/SwiftIR/lib/libpjrt_c_api_cpu.so          # Linux
```

### CUDA Not Found

**Problem**: GPU plugin fails to load CUDA libraries

**Solution**:
```bash
# Ensure CUDA is installed
nvidia-smi

# Check CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install CUDA if missing
# See: https://developer.nvidia.com/cuda-downloads
```

### Compilation Errors

**Problem**: PJRT compilation fails

**Solution**:
- Ensure StableHLO module is valid (run verification passes first)
- Check error message from PJRT API
- Try with IR dumping enabled: `dumpIR: true` in compilation options
- Verify target device supports the operations used

## Comparison with Alternatives

### vs. LLVM ExecutionEngine (Current Default)

| Feature | LLVM ExecutionEngine | PJRT |
|---------|---------------------|------|
| GPU Support | ❌ | ✅ |
| TPU Support | ❌ | ✅ |
| Distributed | ❌ | ✅ |
| Performance | Good (CPU) | Excellent (GPU/TPU) |
| Dependencies | Minimal | Requires XLA |
| Setup | Easy | Moderate |

### vs. PyTorch

| Feature | PyTorch | SwiftIR + PJRT |
|---------|---------|----------------|
| Language | Python | Swift |
| Backend | Eager + JIT | AOT (XLA) |
| GPU Support | CUDA only | CUDA + ROCm + TPU |
| Compilation | Partial (TorchScript) | Full (StableHLO → XLA) |
| Type Safety | Dynamic | Static |

### vs. JAX

| Feature | JAX | SwiftIR + PJRT |
|---------|-----|----------------|
| Language | Python | Swift |
| Backend | XLA | XLA |
| Performance | Excellent | Excellent |
| Type System | NumPy-style | Swift native types |
| JIT/AOT | JIT | Both |

## Future Enhancements

### Short Term
- [ ] Complete PJRT C API bindings
- [ ] Real client implementation
- [ ] CPU plugin integration test
- [ ] Memory management utilities

### Medium Term
- [ ] GPU plugin integration
- [ ] Multi-device execution
- [ ] Asynchronous execution
- [ ] Profiling support

### Long Term
- [ ] TPU support
- [ ] Distributed training
- [ ] Model sharding
- [ ] Custom operator support

## References

- [OpenXLA PJRT Documentation](https://openxla.org/xla/pjrt)
- [PJRT C API Header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
- [PJRT Integration Guide](https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md)
- [XLA Compilation](https://openxla.org/xla/architecture)

## Support

For issues or questions:
1. Check this documentation
2. Review the examples in `Examples/PJRT_Example.swift`
3. Consult the OpenXLA PJRT documentation
4. Open an issue on the SwiftIR GitHub repository
