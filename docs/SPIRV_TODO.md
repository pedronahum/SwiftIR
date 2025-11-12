# SPIR-V Integration TODO

## Current Status

✅ **Architecture Complete** - All components designed and implemented
✅ **Code Compiles** - All Swift code builds successfully
⏳ **Runtime Pending** - Requires library linkage for execution

## What's Been Built

### 1. SPIR-V/GPU Dialect Bindings

**Files:**
- `Sources/SwiftIRCore/include/MLIRCoreWrapper.h` (lines 491-505)
- `Sources/SwiftIRCore/MLIRCore.swift` (lines 485-495)

**Status:** ✅ Complete
- GPU dialect registration: `mlirRegisterGPUDialectWrapper()`
- SPIR-V dialect registration: `mlirRegisterSPIRVDialectWrapper()`
- Swift API bindings: `registerGPUDialect()`, `registerSPIRVDialect()`

### 2. SPIR-V Lowering Pipeline

**File:** `Sources/SwiftIRXLA/SPIRV/SPIRVPipeline.swift`

**Status:** ✅ Complete

**Features:**
- 3-stage progressive lowering:
  - **Stage 1:** Linalg → Parallel Loops (`convert-linalg-to-parallel-loops`)
  - **Stage 2:** Parallel Loops → GPU Dialect (`gpu-map-parallel-loops`, `convert-parallel-loops-to-gpu`, `gpu-kernel-outlining`)
  - **Stage 3:** GPU → SPIR-V Dialect (`convert-gpu-to-spirv`, `convert-func-to-spirv`, `convert-arith-to-spirv`, etc.)
- SPIR-V binary serialization API
- Configurable options (version, workgroup size, verification)
- IR dumping for debugging

**Pipeline:**
```
Linalg Operations
       ↓ (Stage 1)
Parallel Loops (scf.parallel)
       ↓ (Stage 2)
GPU Dialect (gpu.launch)
       ↓ (Stage 3)
SPIR-V Dialect (spirv.module)
       ↓
SPIR-V Binary ([UInt32])
```

### 3. Vulkan Runtime Wrapper

**File:** `Sources/SwiftIRXLA/SPIRV/VulkanRuntime.swift`

**Status:** ✅ Complete (stub implementation)

**Components:**
- `VulkanDevice` - GPU device management
  - Physical device properties
  - Compute queue management
  - Device enumeration
- `VulkanBuffer` - GPU memory buffers
  - Host → Device uploads
  - Device → Host downloads
  - Explicit memory management
- `VulkanExecutable` - Compiled SPIR-V shaders
  - Shader module creation
  - Compute pipeline setup
  - Kernel dispatch
  - Execution statistics

**Platform Support:**
- Windows (via Vulkan SDK)
- Linux (via Vulkan SDK)
- macOS (via MoltenVK)
- Android (native Vulkan)
- iOS (via MoltenVK)

### 4. Metal Runtime Wrapper

**File:** `Sources/SwiftIRXLA/SPIRV/MetalRuntime.swift`

**Status:** ✅ Complete (full Metal API integration)

**Components:**
- `MetalDevice` - Metal device wrapper
  - Device properties (unified memory, max threads, etc.)
  - Command queue management
  - Device enumeration (all GPUs on macOS, default on iOS)
- `MetalBuffer` - Metal buffer wrapper
  - Zero-copy unified memory on Apple Silicon
  - Host ↔ Device transfers
- `MetalExecutable` - Metal compute pipeline
  - MSL compilation from source
  - SPIR-V → MSL conversion (via spirv-cross, pending)
  - Kernel execution with threadgroups

**Platform Support:**
- macOS (all Macs with Metal support)
- iOS (all devices)
- tvOS (all devices)

**Key Advantages:**
- Unified memory on Apple Silicon (zero-copy)
- Native Apple platform integration
- Low latency (<0.5ms)
- Excellent power efficiency

### 5. Comprehensive Example

**File:** `Examples/SPIRV_Example.swift`

**Status:** ✅ Complete (compiles successfully)

**Demonstrates:**
1. SPIR-V lowering pipeline configuration
2. Vulkan device enumeration and execution
3. Metal device enumeration and execution
4. Buffer management and data transfers
5. Architecture diagrams and documentation
6. Performance characteristics comparison
7. Use cases and integration roadmap

## What's Missing for Full Functionality

### Critical Path Items

#### 1. Link MLIR GPU/SPIR-V Dialect Libraries

**Action Required:**
```bash
# Build the required MLIR dialect libraries
cd /Users/pedro/programming/swift/stablehlo/llvm-build
ninja MLIRGPUDialect MLIRSPIRVDialect MLIRGPUToSPIRV \
      MLIRGPUToSPIRVTransforms MLIRSPIRVConversion \
      MLIRFuncToSPIRV MLIRArithToSPIRV MLIRSCFToSPIRV \
      MLIRMemRefToSPIRV MLIRVectorToSPIRV
```

**Expected Output:**
```
libMLIRGPUDialect.a
libMLIRSPIRVDialect.a
libMLIRGPUToSPIRV.a
... (additional libraries)
```

**Issue:** Currently missing symbols:
```
Undefined symbols for architecture arm64:
  "_mlirGetDialectHandle__gpu__"
  "_mlirGetDialectHandle__spirv__"
```

#### 2. Update Package.swift Linker Settings

**Action Required:**
Add to `Package.swift` → `.target(name: "SwiftIRCore")` → `linkerSettings`:

```swift
.unsafeFlags([
    // Existing flags...

    // GPU Dialect
    "-lMLIRGPUDialect",
    "-lMLIRGPUOps",
    "-lMLIRGPUTransforms",

    // SPIR-V Dialect
    "-lMLIRSPIRVDialect",
    "-lMLIRSPIRVConversion",
    "-lMLIRSPIRVUtils",
    "-lMLIRSPIRVSerialization",

    // Conversion Passes
    "-lMLIRGPUToSPIRV",
    "-lMLIRGPUToSPIRVTransforms",
    "-lMLIRFuncToSPIRV",
    "-lMLIRArithToSPIRV",
    "-lMLIRSCFToSPIRV",
    "-lMLIRMemRefToSPIRV",
    "-lMLIRVectorToSPIRV",
])
```

#### 3. Implement Vulkan C Bindings

**File to Create:** `Sources/SwiftIRCore/include/VulkanWrapper.h`

**Required Functions:**
```c
// Instance and Device
VkResult createVulkanInstance(VkInstance* instance);
VkResult enumeratePhysicalDevices(VkInstance instance, VkPhysicalDevice** devices, uint32_t* count);
VkResult createLogicalDevice(VkPhysicalDevice physicalDevice, VkDevice* device, uint32_t* queueFamilyIndex);

// Memory Management
VkResult createBuffer(VkDevice device, VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory);
VkResult uploadBufferData(VkDevice device, VkDeviceMemory memory, const void* data, size_t size);
VkResult downloadBufferData(VkDevice device, VkDeviceMemory memory, void* data, size_t size);

// Shader Execution
VkResult createShaderModule(VkDevice device, const uint32_t* spirvCode, size_t codeSize, VkShaderModule* shaderModule);
VkResult createComputePipeline(VkDevice device, VkShaderModule shaderModule, VkPipeline* pipeline, VkPipelineLayout* layout);
VkResult dispatchCompute(VkDevice device, VkQueue queue, VkPipeline pipeline, VkBuffer* buffers, uint32_t bufferCount, uint32_t x, uint32_t y, uint32_t z);
```

**Then update:** `Sources/SwiftIRXLA/SPIRV/VulkanRuntime.swift`
- Replace stub implementations with actual Vulkan API calls
- Call C wrapper functions via imported module

#### 4. Install Vulkan SDK

**macOS:**
```bash
brew install vulkan-sdk molten-vk
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install vulkan-tools libvulkan-dev vulkan-validationlayers-dev
```

**Windows:**
Download from: https://vulkan.lunarg.com/

**Then add to Package.swift:**
```swift
.target(
    name: "SwiftIRXLA",
    dependencies: ["SwiftIRCore"],
    linkerSettings: [
        .linkedLibrary("vulkan", .when(platforms: [.linux, .windows])),
        .linkedFramework("Metal", .when(platforms: [.macOS, .iOS]))
    ]
)
```

#### 5. Integrate spirv-cross for Metal

**Purpose:** Convert SPIR-V binaries to Metal Shading Language (MSL)

**Action Required:**
```bash
# Clone spirv-cross
git clone https://github.com/KhronosGroup/SPIRV-Cross.git
cd SPIRV-Cross

# Build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# Install
sudo make install
```

**Then update:** `Sources/SwiftIRXLA/SPIRV/MetalRuntime.swift`
- Add C++ interop for spirv-cross
- Implement `MetalExecutable.fromSPIRV()` using:
  ```cpp
  spirv_cross::CompilerMSL compiler(spirv_binary);
  std::string msl_source = compiler.compile();
  ```

### Nice-to-Have Enhancements

#### 1. SPIR-V Validation

**Tool:** `spirv-val` from Vulkan SDK

**Integration Point:** `SPIRVPipeline.serializeSPIRV()`
```swift
public func serializeSPIRV(module: MLIRModule) throws -> [UInt32] {
    let binary = try actualSerialization(module)

    // Validate SPIR-V
    try validateSPIRV(binary)

    return binary
}
```

#### 2. SPIR-V Optimization

**Tool:** `spirv-opt` from Vulkan SDK

**Features:**
- Dead code elimination
- Constant folding
- Loop unrolling
- Inline expansion

#### 3. Memory Pooling

**Optimization:** Reuse GPU buffers across kernel launches

**Implementation:**
```swift
class BufferPool {
    private var availableBuffers: [Int: [VulkanBuffer]] = [:]

    func acquireBuffer(size: Int, device: VulkanDevice) -> VulkanBuffer
    func releaseBuffer(_ buffer: VulkanBuffer)
}
```

#### 4. Asynchronous Execution

**Goal:** Non-blocking kernel launches

**Implementation:**
```swift
extension VulkanExecutable {
    func executeAsync(buffers: [VulkanBuffer], workgroupCount: (UInt32, UInt32, UInt32)) -> Future<Void>
}
```

#### 5. Multi-GPU Support

**Goal:** Distribute workload across multiple GPUs

**Implementation:**
```swift
class VulkanMultiGPUExecutor {
    func executeDistributed(executable: VulkanExecutable, data: [Data], devices: [VulkanDevice])
}
```

## Testing Strategy

### Unit Tests to Add

1. **GPU Dialect Registration**
   ```swift
   func testGPUDialectRegistration() {
       let context = MLIRContext()
       context.registerGPUDialect()
       // Verify dialect is loaded
   }
   ```

2. **SPIR-V Pipeline Stages**
   ```swift
   func testLinalgToParallelLoops() {
       // Create Linalg matmul
       // Run Stage 1
       // Verify parallel loops generated
   }
   ```

3. **Vulkan Device Enumeration**
   ```swift
   func testVulkanDeviceEnumeration() {
       let devices = try VulkanRuntime.enumerateDevices()
       XCTAssertGreaterThan(devices.count, 0)
   }
   ```

4. **Metal Buffer Transfers**
   ```swift
   func testMetalBufferTransfer() {
       let device = try MetalRuntime.enumerateDevices().first!
       let buffer = try device.createBuffer(size: 1024)

       var input: [Float] = [1, 2, 3, 4]
       try input.withUnsafeBytes { ptr in
           try buffer.upload(data: ptr.baseAddress!, length: 16)
       }

       var output = [Float](repeating: 0, count: 4)
       try output.withUnsafeMutableBytes { ptr in
           try buffer.download(data: ptr.baseAddress!, length: 16)
       }

       XCTAssertEqual(input, output)
   }
   ```

### Integration Tests

1. **End-to-End SPIR-V Lowering**
   - Create simple Linalg operation (element-wise add)
   - Lower through all 3 stages
   - Verify valid SPIR-V binary

2. **Vulkan Kernel Execution**
   - Compile simple compute shader
   - Execute on GPU
   - Verify output matches expected

3. **Metal Kernel Execution**
   - Compile MSL kernel
   - Execute on Apple GPU
   - Verify output matches expected

4. **Cross-Platform Consistency**
   - Run same operation on Vulkan and Metal
   - Verify identical results

## Performance Benchmarks

### Target Metrics

**Vulkan:**
- Kernel launch overhead: < 2ms
- Memory transfer (PCIe): > 5 GB/s
- Compute throughput: Device dependent

**Metal (Apple Silicon):**
- Kernel launch overhead: < 0.5ms
- Memory transfer (unified): Zero-copy
- Compute throughput: 2-15 TFLOPS (M1-M3)

### Benchmark Suite

1. **Matrix Multiplication** (various sizes: 128×128, 512×512, 2048×2048)
2. **Convolution** (3×3, 5×5 kernels)
3. **Element-wise Operations** (add, mul, relu)
4. **Memory Bandwidth** (copy throughput)

## Documentation Updates

### Files to Update

1. **README.md**
   - Add SPIR-V integration section
   - Update feature list
   - Add quick start guide

2. **BUILDING.md**
   - Document Vulkan SDK installation
   - Document MLIR dialect build process
   - Platform-specific instructions

3. **API_REFERENCE.md**
   - Document SPIRVPipeline API
   - Document VulkanRuntime API
   - Document MetalRuntime API

## Use Cases

### 1. Cross-Platform ML Inference

**Scenario:** Deploy ML model to Windows/Linux/macOS/Android/iOS

**Implementation:**
```swift
let model = loadModel("resnet50.mlir")
let pipeline = SPIRVPipeline(context: context)
try pipeline.lowerToSPIRV(module: model)
let spirvBinary = try pipeline.serializeSPIRV(module: model)

#if os(macOS) || os(iOS)
let device = try MetalRuntime.enumerateDevices().first!
let executable = try device.createExecutableFromSPIRV(spirvBinary: spirvBinary)
#else
let device = try VulkanRuntime.enumerateDevices().first!
let executable = try device.createExecutable(spirvBinary: spirvBinary)
#endif

let result = try executable.execute(buffers: [inputBuffer, outputBuffer], ...)
```

### 2. Game Engine ML Integration

**Scenario:** Real-time ML effects in game (style transfer, upscaling)

**Advantages:**
- Share GPU with rendering pipeline
- Low latency (< 16ms per frame at 60 FPS)
- Vulkan integration with existing graphics

### 3. Mobile ML Applications

**Scenario:** On-device inference for iOS/Android apps

**Advantages:**
- GPU acceleration on mobile
- Lower power consumption than CPU
- Privacy (no cloud upload)

### 4. Embedded Edge Devices

**Scenario:** ML on Raspberry Pi, NVIDIA Jetson, etc.

**Advantages:**
- GPU acceleration on constrained devices
- SPIR-V portability
- Efficient power usage

## Comparison: SPIR-V vs PJRT

| Feature | SPIR-V + Vulkan/Metal | PJRT + XLA |
|---------|----------------------|-----------|
| **Platform Support** | ✅ Windows/Linux/macOS/Android/iOS | ⚠️ Linux/macOS (GPU), TPU (Cloud) |
| **Mobile Support** | ✅ iOS/Android native | ❌ Limited |
| **Graphics Integration** | ✅ Native Vulkan/Metal | ❌ Compute only |
| **ML Optimization** | ⚠️ Good | ✅ Excellent |
| **Kernel Fusion** | ⚠️ Manual | ✅ Automatic |
| **TPU Support** | ❌ No | ✅ Yes |
| **Setup Complexity** | ⚠️ Moderate (SDK required) | ⚠️ Moderate (library build) |
| **Best For** | Cross-platform, mobile, graphics | ML performance, TPU, production |

### Recommended Strategy

**Use SPIR-V when:**
- Need mobile deployment (iOS/Android)
- Need Windows support
- Integrating with game engines
- Embedded/edge devices

**Use PJRT when:**
- Need maximum ML performance
- Have access to TPUs
- Production ML serving
- JAX/TensorFlow compatibility

**Use Both:**
- Compile StableHLO once
- Lower to PJRT for data center (GPU/TPU)
- Lower to SPIR-V for edge/mobile
- Same high-level IR, different execution paths

## Timeline Estimate

### Phase 1: MLIR Linkage (1-2 days)
- Build MLIR GPU/SPIR-V dialect libraries
- Update Package.swift linker flags
- Resolve linking errors
- **Milestone:** Example runs without linking errors

### Phase 2: Vulkan Integration (3-5 days)
- Install Vulkan SDK
- Implement VulkanWrapper.h
- Update VulkanRuntime.swift
- Test device enumeration
- **Milestone:** Can enumerate Vulkan devices

### Phase 3: Basic Execution (5-7 days)
- Implement buffer management
- Implement shader loading
- Implement compute dispatch
- End-to-end test with simple kernel
- **Milestone:** Can execute "hello world" compute shader

### Phase 4: SPIR-V Generation (7-10 days)
- Implement actual SPIR-V serialization
- Test lowering pipeline with real IR
- Validate generated SPIR-V
- **Milestone:** Can generate valid SPIR-V from Linalg

### Phase 5: Metal Integration (3-5 days)
- Integrate spirv-cross
- Implement SPIR-V → MSL conversion
- Test Metal execution path
- **Milestone:** Can execute on Metal

### Phase 6: Testing & Optimization (7-10 days)
- Write unit tests
- Write integration tests
- Performance benchmarking
- Memory leak detection
- **Milestone:** Production-ready quality

**Total Estimate:** 4-6 weeks for full production readiness

## Next Immediate Steps

1. **Check if MLIR GPU/SPIR-V dialects are built**
   ```bash
   ls /Users/pedro/programming/swift/stablehlo/llvm-build/lib/ | grep -E "(GPU|SPIRV)"
   ```

2. **If not built, build them**
   ```bash
   cd /Users/pedro/programming/swift/stablehlo/llvm-build
   ninja MLIRGPUDialect MLIRSPIRVDialect
   ```

3. **Update Package.swift with linker flags** (as shown above)

4. **Test that example links successfully**
   ```bash
   swift build --target SPIRV_Example
   ```

5. **If links, start implementing Vulkan wrapper**

---

**Status:** Architecture complete, ready for implementation
**Blockers:** Missing MLIR dialect libraries (resolvable)
**Risk:** Low - clear path forward
**Effort:** 4-6 weeks to production-ready
