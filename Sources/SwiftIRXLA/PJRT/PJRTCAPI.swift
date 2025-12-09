//===-- PJRTCAPI.swift - PJRT C API Swift Wrapper --------*- Swift -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// Swift wrapper for simplified PJRT C API
//
//===------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import PJRTCAPI  // C module defined in PJRTCWrappers/include/module.modulemap

// MARK: - XLA Optimization Level

/// XLA backend optimization level
///
/// Controls how much optimization XLA applies during compilation.
/// Higher levels result in slower compilation but potentially faster execution.
///
/// Reference: https://github.com/openxla/xla - xla_backend_optimization_level in DebugOptions
public enum XLAOptimizationLevel: Sendable {
    /// Use XLA's default optimization (typically O2)
    case `default`
    /// No optimization - fastest compile, slowest execution
    case O0
    /// Basic optimization
    case O1
    /// Standard optimization (recommended, GPU requires >= 2)
    case O2
    /// Maximum optimization - slowest compile, best execution
    case O3

    /// Convert to C API value
    var cValue: SW_XLA_OptLevel {
        switch self {
        case .default: return SW_XLA_OPT_DEFAULT
        case .O0: return SW_XLA_OPT_O0
        case .O1: return SW_XLA_OPT_O1
        case .O2: return SW_XLA_OPT_O2
        case .O3: return SW_XLA_OPT_O3
        }
    }
}

/// Low-level Swift wrapper for PJRT C API
///
/// This provides Swift-friendly access to the PJRT C API functions
/// via the simplified C wrapper layer.
internal class PJRTCAPI {
    /// Backend type for this API instance
    internal let backend: String

    /// Path to the plugin library
    private let pluginPath: String

    /// Initialize by loading a PJRT plugin
    ///
    /// - Parameters:
    ///   - backend: Backend name ("cpu", "gpu", "tpu")
    ///   - pluginPath: Path to the PJRT plugin library
    /// - Throws: PJRTError if plugin cannot be loaded
    init(backend: String, pluginPath: String) throws {
        self.backend = backend
        self.pluginPath = pluginPath

        try loadPlugin()
    }

    deinit {
        // NOTE: We intentionally do NOT unload the plugin here to avoid segfaults
        // when PJRTBuffer/PJRTLoadedExecutable deinit methods try to destroy resources
        // after the plugin has been unloaded. The plugin will remain loaded for the
        // lifetime of the process, which is acceptable since we typically only load
        // one PJRT plugin per process.
        //
        // unloadPlugin()
    }

    /// Load the PJRT plugin library
    private func loadPlugin() throws {
        let errorCode = PJRT_LoadPlugin(pluginPath)
        if errorCode != SW_PJRT_Error_OK {
            if let errorMsg = PJRT_GetLastError() {
                let msg = String(cString: errorMsg)
                throw PJRTError.clientCreationFailed("Failed to load plugin '\(pluginPath)': \(msg)")
            }
            throw PJRTError.clientCreationFailed("Failed to load plugin '\(pluginPath)': error code \(errorCode.rawValue)")
        }

        print("✓ PJRT \(backend.uppercased()) plugin loaded successfully")
        print("  Plugin: \(pluginPath)")
    }

    /// Unload the PJRT plugin
    private func unloadPlugin() {
        PJRT_UnloadPlugin()
    }

    /// Create a PJRT client
    ///
    /// - Returns: Opaque pointer to PJRT_Client
    /// - Throws: PJRTError if client creation fails
    func createClient() throws -> OpaquePointer {
        var client: UnsafeMutableRawPointer? = nil
        let errorCode = PJRT_CreateClient(&client)

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.clientCreationFailed("PJRT_CreateClient failed with code \(errorCode.rawValue)")
        }

        guard let clientPtr = client else {
            throw PJRTError.clientCreationFailed("PJRT_CreateClient returned NULL client")
        }

        return OpaquePointer(clientPtr)
    }

    /// Destroy a PJRT client
    func destroyClient(_ client: OpaquePointer) {
        PJRT_DestroyClient(UnsafeMutableRawPointer(client))
    }

    /// Get platform name from client
    func getPlatformName(_ client: OpaquePointer) throws -> String {
        var namePtr: UnsafePointer<CChar>? = nil
        let errorCode = PJRT_GetPlatformName(UnsafeMutableRawPointer(client), &namePtr)

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.clientCreationFailed("PJRT_GetPlatformName failed with code \(errorCode.rawValue)")
        }

        guard let name = namePtr else {
            throw PJRTError.clientCreationFailed("PJRT_GetPlatformName returned NULL")
        }

        return String(cString: name)
    }

    /// Get addressable devices from client
    func getAddressableDevices(_ client: OpaquePointer) throws -> [OpaquePointer] {
        var devicesPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>? = nil
        var numDevices: size_t = 0

        let errorCode = PJRT_GetAddressableDevices(
            UnsafeMutableRawPointer(client),
            &devicesPtr,
            &numDevices
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.clientCreationFailed("PJRT_GetAddressableDevices failed with code \(errorCode.rawValue)")
        }

        guard let devices = devicesPtr else {
            return []
        }

        // Convert device array to Swift array
        var result: [OpaquePointer] = []
        for i in 0..<numDevices {
            if let device = devices[i] {
                result.append(OpaquePointer(device))
            }
        }

        return result
    }

    /// Get device ID
    func getDeviceId(_ device: OpaquePointer) throws -> Int32 {
        var deviceId: Int32 = 0
        let errorCode = PJRT_GetDeviceId(UnsafeMutableRawPointer(device), &deviceId)

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.deviceNotFound(0)
        }

        return deviceId
    }

    /// Get device kind (e.g., "CPU", "GPU")
    func getDeviceKind(_ device: OpaquePointer) throws -> String {
        var kindPtr: UnsafePointer<CChar>? = nil
        let errorCode = PJRT_GetDeviceKind(UnsafeMutableRawPointer(device), &kindPtr)

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.deviceNotFound(0)
        }

        guard let kind = kindPtr else {
            throw PJRTError.deviceNotFound(0)
        }

        return String(cString: kind)
    }

    /// Host buffer semantics for buffer creation
    public enum HostBufferSemantics {
        /// Data is copied during the call, buffer can be modified after return
        case immutableOnlyDuringCall
        /// Data must remain valid until transfer completes
        case immutableUntilTransferCompletes
        /// Zero-copy: host memory used directly, must remain valid for buffer lifetime
        case zeroCopy
        /// Mutable zero-copy: for buffer donation - memory can be reused for output
        case mutableZeroCopy

        public var toCType: SW_PJRT_HostBufferSemantics {
            switch self {
            case .immutableOnlyDuringCall: return SW_PJRT_HostBuffer_ImmutableOnlyDuringCall
            case .immutableUntilTransferCompletes: return SW_PJRT_HostBuffer_ImmutableUntilTransferCompletes
            case .zeroCopy: return SW_PJRT_HostBuffer_ZeroCopy
            case .mutableZeroCopy: return SW_PJRT_HostBuffer_MutableZeroCopy
            }
        }

        /// Initialize from PJRTClient.HostBufferSemantics
        public init(_ clientSemantics: PJRTClient.HostBufferSemantics) {
            switch clientSemantics {
            case .immutableOnlyDuringCall: self = .immutableOnlyDuringCall
            case .immutableUntilTransferCompletes: self = .immutableUntilTransferCompletes
            case .zeroCopy: self = .zeroCopy
            case .mutableZeroCopy: self = .mutableZeroCopy
            }
        }
    }

    /// Create a buffer from host data
    func createBuffer(
        client: OpaquePointer,
        data: UnsafeRawPointer,
        type: SW_PJRT_Buffer_Type,
        dims: [Int64],
        device: OpaquePointer,
        semantics: HostBufferSemantics = .immutableOnlyDuringCall
    ) throws -> OpaquePointer {
        var buffer: UnsafeMutableRawPointer? = nil

        let errorCode = dims.withUnsafeBufferPointer { dimsPtr in
            PJRT_CreateBufferWithSemantics(
                UnsafeMutableRawPointer(client),
                data,
                type,
                dimsPtr.baseAddress,
                dims.count,
                UnsafeMutableRawPointer(device),
                semantics.toCType,
                &buffer
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_CreateBuffer failed with code \(errorCode.rawValue)")
        }

        guard let bufferPtr = buffer else {
            throw PJRTError.bufferCreationFailed("PJRT_CreateBuffer returned NULL buffer")
        }

        return OpaquePointer(bufferPtr)
    }

    /// Create a buffer from host data using cached descriptor (fast path)
    ///
    /// When `isSameShape` is true and the slot was previously used, this avoids
    /// re-initializing the args structure and only updates the data pointer.
    /// This reduces per-buffer overhead for repeated same-shape inputs.
    ///
    /// - Parameters:
    ///   - client: PJRT client
    ///   - data: Host data pointer
    ///   - type: Buffer element type
    ///   - dims: Tensor dimensions
    ///   - device: Target device
    ///   - semantics: Host buffer semantics
    ///   - cachedSlot: Cache slot (0-15) for this input
    ///   - isSameShape: Whether the shape matches the previous call for this slot
    /// - Returns: PJRT buffer handle
    func createBufferFast(
        client: OpaquePointer,
        data: UnsafeRawPointer,
        type: SW_PJRT_Buffer_Type,
        dims: [Int64],
        device: OpaquePointer,
        semantics: HostBufferSemantics = .immutableOnlyDuringCall,
        cachedSlot: Int,
        isSameShape: Bool
    ) throws -> OpaquePointer {
        var buffer: UnsafeMutableRawPointer? = nil

        let errorCode = dims.withUnsafeBufferPointer { dimsPtr in
            PJRT_CreateBufferFast(
                UnsafeMutableRawPointer(client),
                data,
                type,
                dimsPtr.baseAddress,
                dims.count,
                UnsafeMutableRawPointer(device),
                semantics.toCType,
                Int32(cachedSlot),
                isSameShape,
                &buffer
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_CreateBufferFast failed with code \(errorCode.rawValue)")
        }

        guard let bufferPtr = buffer else {
            throw PJRTError.bufferCreationFailed("PJRT_CreateBufferFast returned NULL buffer")
        }

        return OpaquePointer(bufferPtr)
    }

    /// Create multiple buffers in a single batched call
    /// This reduces FFI overhead by batching multiple buffer creations together
    func createBuffersBatched(
        client: OpaquePointer,
        dataPointers: [UnsafeRawPointer],
        types: [SW_PJRT_Buffer_Type],
        dimsArrays: [[Int64]],
        device: OpaquePointer,
        semantics: HostBufferSemantics
    ) throws -> [OpaquePointer] {
        let numBuffers = dataPointers.count
        guard numBuffers > 0,
              numBuffers == types.count,
              numBuffers == dimsArrays.count else {
            throw PJRTError.bufferCreationFailed("Mismatched array sizes for batched buffer creation")
        }

        // Prepare arrays of pointers for C API
        var outBuffers = [UnsafeMutableRawPointer?](repeating: nil, count: numBuffers)
        var numDimsArray = dimsArrays.map { size_t($0.count) }

        // Convert UnsafeRawPointer to UnsafeMutableRawPointer? for C API compatibility
        var mutableDataPtrs = dataPointers.map { UnsafeMutableRawPointer(mutating: $0) as UnsafeMutableRawPointer? }

        // We need to keep the dims arrays alive during the call
        let errorCode = mutableDataPtrs.withUnsafeMutableBufferPointer { dataPtrs in
            types.withUnsafeBufferPointer { typesPtrs in
                numDimsArray.withUnsafeMutableBufferPointer { numDimsPtrs in
                    outBuffers.withUnsafeMutableBufferPointer { outBufPtrs in
                        // Create array of pointers to dims arrays
                        var dimsPtrs = [UnsafePointer<Int64>?](repeating: nil, count: numBuffers)
                        for i in 0..<numBuffers {
                            dimsPtrs[i] = dimsArrays[i].withUnsafeBufferPointer { $0.baseAddress }
                        }

                        return dimsPtrs.withUnsafeMutableBufferPointer { dimsPtrsBuf in
                            PJRT_CreateBuffersBatched(
                                UnsafeMutableRawPointer(client),
                                numBuffers,
                                dataPtrs.baseAddress,
                                typesPtrs.baseAddress,
                                dimsPtrsBuf.baseAddress,
                                numDimsPtrs.baseAddress,
                                UnsafeMutableRawPointer(device),
                                semantics.toCType,
                                outBufPtrs.baseAddress
                            )
                        }
                    }
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_CreateBuffersBatched failed with code \(errorCode.rawValue)")
        }

        // Convert to OpaquePointers
        return try outBuffers.map { ptr in
            guard let ptr = ptr else {
                throw PJRTError.bufferCreationFailed("PJRT_CreateBuffersBatched returned NULL buffer")
            }
            return OpaquePointer(ptr)
        }
    }

    /// Transfer buffer from device to host (synchronous - waits for completion)
    func bufferToHost(buffer: OpaquePointer, destination: UnsafeMutableRawPointer, size: Int) throws {
        let errorCode = PJRT_BufferToHost(
            UnsafeMutableRawPointer(buffer),
            destination,
            size
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_BufferToHost failed with code \(errorCode.rawValue)")
        }
    }

    /// Transfer buffer from device to host (asynchronous - returns immediately)
    /// Returns an event handle that must be awaited before accessing the destination data
    func bufferToHostAsync(buffer: OpaquePointer, destination: UnsafeMutableRawPointer, size: Int) throws -> OpaquePointer? {
        var event: UnsafeMutableRawPointer? = nil
        let errorCode = PJRT_BufferToHostAsync(
            UnsafeMutableRawPointer(buffer),
            destination,
            size,
            &event
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_BufferToHostAsync failed with code \(errorCode.rawValue)")
        }

        return event.map { OpaquePointer($0) }
    }

    /// Wait for an event to complete
    func eventAwait(_ event: OpaquePointer) throws {
        let errorCode = PJRT_EventAwait(UnsafeMutableRawPointer(event))
        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_EventAwait failed with code \(errorCode.rawValue)")
        }
    }

    /// Destroy an event (must be called after awaiting)
    func eventDestroy(_ event: OpaquePointer) {
        PJRT_EventDestroy(UnsafeMutableRawPointer(event))
    }

    /// Wait for an event to complete and destroy it
    func eventAwaitAndDestroy(_ event: OpaquePointer?) throws {
        guard let event = event else { return }
        let errorCode = PJRT_EventAwaitAndDestroy(UnsafeMutableRawPointer(event))
        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_EventAwaitAndDestroy failed with code \(errorCode.rawValue)")
        }
    }

    /// Destroy a buffer
    func destroyBuffer(_ buffer: OpaquePointer) {
        PJRT_DestroyBuffer(UnsafeMutableRawPointer(buffer))
    }

    /// Get buffer dimensions
    func getBufferDimensions(buffer: OpaquePointer) throws -> [Int64] {
        var dimsPtr: UnsafePointer<Int64>? = nil
        var numDims: size_t = 0

        let errorCode = PJRT_GetBufferDimensions(
            UnsafeMutableRawPointer(buffer),
            &dimsPtr,
            &numDims
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_GetBufferDimensions failed with code \(errorCode.rawValue)")
        }

        guard let dims = dimsPtr else {
            return []
        }

        return Array(UnsafeBufferPointer(start: dims, count: Int(numDims)))
    }

    /// Get buffer size in bytes
    func getBufferOnDeviceSizeInBytes(buffer: OpaquePointer) throws -> Int {
        var size: size_t = 0

        let errorCode = PJRT_GetBufferOnDeviceSizeInBytes(
            UnsafeMutableRawPointer(buffer),
            &size
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_GetBufferOnDeviceSizeInBytes failed with code \(errorCode.rawValue)")
        }

        return Int(size)
    }

    /// Compile an MLIR module
    func compile(client: OpaquePointer, mlirModule: String) throws -> OpaquePointer {
        return try compile(client: client, mlirModule: mlirModule, xlaOptLevel: .default)
    }

    /// Compile an MLIR module with specified XLA optimization level
    func compile(client: OpaquePointer, mlirModule: String, xlaOptLevel: XLAOptimizationLevel) throws -> OpaquePointer {
        var executable: UnsafeMutableRawPointer? = nil

        let errorCode = mlirModule.withCString { modulePtr in
            PJRT_CompileWrapperWithOptLevel(
                UnsafeMutableRawPointer(client),
                modulePtr,
                xlaOptLevel.cValue,
                &executable
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.compilationFailed("PJRT_CompileWrapper failed with code \(errorCode.rawValue)")
        }

        guard let executablePtr = executable else {
            throw PJRTError.compilationFailed("PJRT_CompileWrapper returned NULL executable")
        }

        return OpaquePointer(executablePtr)
    }

    /// Destroy an executable
    func destroyExecutable(_ executable: OpaquePointer) {
        PJRT_DestroyExecutable(UnsafeMutableRawPointer(executable))
    }

    /// Execute a compiled program
    func execute(
        executable: OpaquePointer,
        inputs: [OpaquePointer]
    ) throws -> [OpaquePointer] {
        // Convert to optional raw pointers as expected by C API
        var inputPtrs: [UnsafeMutableRawPointer?] = inputs.map { UnsafeMutableRawPointer($0) }
        var outputsPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>? = nil
        var numOutputs: size_t = 0

        let errorCode = inputPtrs.withUnsafeMutableBufferPointer { inputsBuffer in
            PJRT_ExecuteWrapper(
                UnsafeMutableRawPointer(executable),
                inputsBuffer.baseAddress,
                inputs.count,
                &outputsPtr,
                &numOutputs
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteWrapper failed with code \(errorCode.rawValue)")
        }

        guard let outputs = outputsPtr else {
            return []
        }

        // Convert output buffers to Swift array
        var result: [OpaquePointer] = []
        for i in 0..<numOutputs {
            if let output = outputs[i] {
                result.append(OpaquePointer(output))
            }
        }

        return result
    }

    /// Execute a compiled program with buffer donation support
    ///
    /// When buffer donation is enabled, input buffers can be reused for outputs
    /// with matching shapes. This avoids allocating new output buffers.
    ///
    /// - Parameters:
    ///   - executable: Compiled PJRT executable
    ///   - inputs: Input buffer handles
    ///   - nonDonatableIndices: Input indices that should NOT be donated (empty = donate all eligible)
    /// - Returns: Output buffer handles
    /// - Note: After execution, donated input buffers become invalid and must not be used.
    func executeWithDonation(
        executable: OpaquePointer,
        inputs: [OpaquePointer],
        nonDonatableIndices: [Int64] = []
    ) throws -> [OpaquePointer] {
        var inputPtrs: [UnsafeMutableRawPointer?] = inputs.map { UnsafeMutableRawPointer($0) }
        var outputsPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>? = nil
        var numOutputs: size_t = 0

        let errorCode = inputPtrs.withUnsafeMutableBufferPointer { inputsBuffer in
            nonDonatableIndices.withUnsafeBufferPointer { indicesBuffer in
                PJRT_ExecuteWithDonation(
                    UnsafeMutableRawPointer(executable),
                    inputsBuffer.baseAddress,
                    inputs.count,
                    indicesBuffer.baseAddress,
                    indicesBuffer.count,
                    &outputsPtr,
                    &numOutputs
                )
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteWithDonation failed with code \(errorCode.rawValue)")
        }

        guard let outputs = outputsPtr else {
            return []
        }

        // Convert output buffers to Swift array
        var result: [OpaquePointer] = []
        for i in 0..<numOutputs {
            if let output = outputs[i] {
                result.append(OpaquePointer(output))
            }
        }

        return result
    }

    /// Increase external reference count for a buffer
    /// Must be called when taking ownership of a buffer from PJRT
    func bufferIncRef(_ buffer: OpaquePointer) throws {
        let errorCode = PJRT_Buffer_IncRefCount(UnsafeMutableRawPointer(buffer))
        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_Buffer_IncRefCount failed with code \(errorCode.rawValue)")
        }
    }

    /// Decrease external reference count for a buffer
    /// Must be called before destroying a buffer
    func bufferDecRef(_ buffer: OpaquePointer) throws {
        let errorCode = PJRT_Buffer_DecRefCount(UnsafeMutableRawPointer(buffer))
        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_Buffer_DecRefCount failed with code \(errorCode.rawValue)")
        }
    }

    // MARK: - Fast/Optimized APIs

    /// Fast buffer to host async (uses designated initializers)
    func bufferToHostAsyncFast(
        buffer: OpaquePointer,
        destination: UnsafeMutableRawPointer,
        size: Int
    ) throws -> OpaquePointer? {
        var event: UnsafeMutableRawPointer? = nil

        let errorCode = PJRT_BufferToHostAsyncFast(
            UnsafeMutableRawPointer(buffer),
            destination,
            size,
            &event
        )

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_BufferToHostAsyncFast failed with code \(errorCode.rawValue)")
        }

        return event.map { OpaquePointer($0) }
    }

    /// Fast await and destroy event
    func eventAwaitAndDestroyFast(event: OpaquePointer) throws {
        let errorCode = PJRT_EventAwaitAndDestroyFast(UnsafeMutableRawPointer(event))
        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferTransferFailed("PJRT_EventAwaitAndDestroyFast failed with code \(errorCode.rawValue)")
        }
    }

    /// Execute and transfer to host in a single FFI call
    /// This combines execute + async D2H + await into a single C call to reduce overhead.
    ///
    /// - Parameters:
    ///   - executable: Compiled executable
    ///   - inputBuffers: Input buffer handles
    ///   - outputSizes: Size in bytes for each output buffer
    /// - Returns: Array of Float arrays containing the output data
    func executeAndTransfer(
        executable: OpaquePointer,
        inputBuffers: [OpaquePointer],
        outputSizes: [Int]
    ) throws -> [[Float]] {
        // Convert input buffers to raw pointers
        var inputPtrs: [UnsafeMutableRawPointer?] = inputBuffers.map { UnsafeMutableRawPointer($0) }

        // Allocate output arrays
        var output0 = [Float](repeating: 0, count: outputSizes[0] / MemoryLayout<Float>.stride)
        var output1 = outputSizes.count > 1 ? [Float](repeating: 0, count: outputSizes[1] / MemoryLayout<Float>.stride) : []

        var outDataSizes: [size_t] = outputSizes.map { size_t($0) }
        var actualOutputs: size_t = 0

        let errorCode = inputPtrs.withUnsafeMutableBufferPointer { inputsBuffer in
            output0.withUnsafeMutableBytes { out0Buffer in
                output1.withUnsafeMutableBytes { out1Buffer in
                    outDataSizes.withUnsafeMutableBufferPointer { sizesBuffer in
                        // Create array of output pointers
                        var outPtrs: [UnsafeMutableRawPointer?] = [
                            out0Buffer.baseAddress,
                            outputSizes.count > 1 ? out1Buffer.baseAddress : nil
                        ]
                        return outPtrs.withUnsafeMutableBufferPointer { outPtrsBuffer in
                            PJRT_ExecuteAndTransfer(
                                UnsafeMutableRawPointer(executable),
                                inputsBuffer.baseAddress,
                                inputBuffers.count,
                                outPtrsBuffer.baseAddress,
                                sizesBuffer.baseAddress,
                                outputSizes.count,
                                &actualOutputs
                            )
                        }
                    }
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteAndTransfer failed with code \(errorCode.rawValue)")
        }

        var outputs: [[Float]] = [output0]
        if outputSizes.count > 1 {
            outputs.append(output1)
        }
        return Array(outputs.prefix(Int(actualOutputs)))
    }

    /// Hot path: Full H2D + Execute + D2H in a single FFI call
    ///
    /// This is the most optimized execution path, combining all operations into
    /// a single C call to minimize FFI overhead.
    ///
    /// - Parameters:
    ///   - client: PJRT client handle
    ///   - executable: PJRT executable handle
    ///   - device: PJRT device handle
    ///   - inputData: Array of raw pointers to input float data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    ///   - semantics: Host buffer semantics
    func executeHotPath(
        client: OpaquePointer,
        executable: OpaquePointer,
        device: OpaquePointer,
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int],
        semantics: HostBufferSemantics
    ) throws {
        // Convert to C-compatible arrays
        var inputPtrs = inputData.map { UnsafeRawPointer($0) }
        var inputSizesU = inputSizes.map { size_t($0) }
        var outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }
        var outputSizesU = outputSizes.map { size_t($0) }

        let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            inputSizesU.withUnsafeBufferPointer { inputSizesBuf in
                outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                    outputSizesU.withUnsafeBufferPointer { outputSizesBuf in
                        // Cast the buffer pointers to match C expectations
                        let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                        let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                        return PJRT_ExecuteHotPath(
                            UnsafeMutableRawPointer(client),
                            UnsafeMutableRawPointer(executable),
                            UnsafeMutableRawPointer(device),
                            inputDataPtr,
                            inputSizesBuf.baseAddress,
                            inputData.count,
                            outputDataPtr,
                            outputSizesBuf.baseAddress,
                            outputData.count,
                            semantics.toCType
                        )
                    }
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteHotPath failed with code \(errorCode.rawValue)")
        }
    }

    // MARK: - Profiled Execution

    /// Execute with detailed timing breakdown
    /// This profiles the complete execution path: H2D + Execute + D2H
    ///
    /// - Parameters:
    ///   - client: PJRT client handle
    ///   - executable: PJRT executable handle
    ///   - device: PJRT device handle
    ///   - inputData: Array of raw pointers to input float data
    ///   - inputSizes: Number of elements in each input
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    ///   - semantics: Host buffer semantics
    /// - Returns: PJRTExecutionTiming with breakdown of where time was spent
    func executeProfiled(
        client: OpaquePointer,
        executable: OpaquePointer,
        device: OpaquePointer,
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int],
        semantics: HostBufferSemantics
    ) throws -> PJRTExecutionTiming {
        // Convert to C-compatible arrays
        var inputPtrs = inputData.map { UnsafeRawPointer($0) }
        let inputSizesU = inputSizes.map { size_t($0) }
        var outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }
        let outputSizesU = outputSizes.map { size_t($0) }

        var timing = SW_PJRT_ExecutionTiming()

        let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            inputSizesU.withUnsafeBufferPointer { inputSizesBuf in
                outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                    outputSizesU.withUnsafeBufferPointer { outputSizesBuf in
                        // Cast the buffer pointers to match C expectations
                        let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                        let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                        return PJRT_ExecuteProfiled(
                            UnsafeMutableRawPointer(client),
                            UnsafeMutableRawPointer(executable),
                            UnsafeMutableRawPointer(device),
                            inputDataPtr,
                            inputSizesBuf.baseAddress,
                            inputData.count,
                            outputDataPtr,
                            outputSizesBuf.baseAddress,
                            outputData.count,
                            semantics.toCType,
                            &timing
                        )
                    }
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteProfiled failed with code \(errorCode.rawValue)")
        }

        return PJRTExecutionTiming(from: timing)
    }
}

// MARK: - PJRT Execution Timing

/// Timing breakdown for PJRT execution operations
///
/// All times are in microseconds for easier interpretation.
/// Use `printSummary()` to display a formatted breakdown.
public struct PJRTExecutionTiming {
    /// Time spent creating input buffers (H2D transfer initiation) in microseconds
    public var h2dCreateUs: Double
    /// Time spent executing the kernel in microseconds
    public var executeUs: Double
    /// Time spent initiating D2H transfers (async start) in microseconds
    public var d2hInitiateUs: Double
    /// Time spent awaiting D2H completion (sync) in microseconds
    public var d2hAwaitUs: Double
    /// Time spent destroying buffers (cleanup) in microseconds
    public var bufferDestroyUs: Double
    /// Total end-to-end time in microseconds
    public var totalUs: Double
    /// Number of input buffers
    public var numInputs: Int
    /// Number of output buffers
    public var numOutputs: Int

    /// Initialize from C timing structure
    init(from cTiming: SW_PJRT_ExecutionTiming) {
        self.h2dCreateUs = Double(cTiming.h2d_create_ns) / 1000.0
        self.executeUs = Double(cTiming.execute_ns) / 1000.0
        self.d2hInitiateUs = Double(cTiming.d2h_initiate_ns) / 1000.0
        self.d2hAwaitUs = Double(cTiming.d2h_await_ns) / 1000.0
        self.bufferDestroyUs = Double(cTiming.buffer_destroy_ns) / 1000.0
        self.totalUs = Double(cTiming.total_ns) / 1000.0
        self.numInputs = Int(cTiming.num_inputs)
        self.numOutputs = Int(cTiming.num_outputs)
    }

    /// Print a summary of the timing breakdown
    public func printSummary() {
        print("PJRT Execution Timing Breakdown:")
        print("  H2D Buffer Creation: \(String(format: "%.1f", h2dCreateUs)) μs")
        print("  Execute:             \(String(format: "%.1f", executeUs)) μs")
        print("  D2H Initiate:        \(String(format: "%.1f", d2hInitiateUs)) μs")
        print("  D2H Await:           \(String(format: "%.1f", d2hAwaitUs)) μs")
        print("  Buffer Destroy:      \(String(format: "%.1f", bufferDestroyUs)) μs")
        print("  Total:               \(String(format: "%.1f", totalUs)) μs")
        print("  Inputs: \(numInputs), Outputs: \(numOutputs)")
    }
}

// MARK: - PJRT Async Execution API

// MARK: - PJRT Buffer Pool

/// Swift wrapper for PJRT buffer pool
///
/// A buffer pool caches configuration for repeated executions with the same
/// input shapes. This enables zero-copy semantics for optimal performance.
///
/// Note: Due to PJRT's design, input buffers are consumed after each execution.
/// The pool optimizes by using zero-copy semantics and caching shape metadata.
public class PJRTBufferPool {
    private var pool: SW_PJRT_BufferPool
    private let inputSizes: [Int]

    /// Initialize a buffer pool for repeated execution
    ///
    /// - Parameters:
    ///   - client: PJRT client handle
    ///   - device: PJRT device handle
    ///   - executable: PJRT executable handle
    ///   - inputSizes: Number of elements for each input
    /// - Throws: PJRTError on failure
    public init(
        client: OpaquePointer,
        device: OpaquePointer,
        executable: OpaquePointer,
        inputSizes: [Int]
    ) throws {
        self.pool = SW_PJRT_BufferPool()
        self.inputSizes = inputSizes

        var inputSizesU = inputSizes.map { size_t($0) }

        let errorCode = inputSizesU.withUnsafeMutableBufferPointer { sizesBuf in
            PJRT_BufferPoolCreate(
                UnsafeMutableRawPointer(client),
                UnsafeMutableRawPointer(device),
                UnsafeMutableRawPointer(executable),
                sizesBuf.baseAddress,
                sizesBuf.count,
                &pool
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_BufferPoolCreate failed with code \(errorCode.rawValue)")
        }
    }

    deinit {
        PJRT_BufferPoolDestroy(&pool)
    }

    /// Execute using the pooled configuration with zero-copy semantics
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input float data
    ///   - inputSizes: Number of elements in each input (must match pool configuration)
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - outputSizes: Number of elements in each output
    /// - Returns: Timing breakdown for the execution
    /// - Throws: PJRTError on failure
    public func execute(
        inputData: [UnsafeRawPointer],
        inputSizes: [Int],
        outputData: [UnsafeMutableRawPointer],
        outputSizes: [Int]
    ) throws -> PJRTExecutionTiming {
        let inputPtrs = inputData.map { UnsafeRawPointer($0) }
        let inputSizesU = inputSizes.map { size_t($0) }
        let outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }
        let outputSizesU = outputSizes.map { size_t($0) }

        var timing = SW_PJRT_ExecutionTiming()

        let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            inputSizesU.withUnsafeBufferPointer { inputSizesBuf in
                outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                    outputSizesU.withUnsafeBufferPointer { outputSizesBuf in
                        let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                        let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                        return PJRT_ExecutePooled(
                            &pool,
                            inputDataPtr,
                            inputSizesBuf.baseAddress,
                            inputData.count,
                            outputDataPtr,
                            outputSizesBuf.baseAddress,
                            outputData.count,
                            &timing
                        )
                    }
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecutePooled failed with code \(errorCode.rawValue)")
        }

        return PJRTExecutionTiming(from: timing)
    }
}

/// Execute with OnReady callbacks instead of blocking await
/// This uses PJRT_Event_OnReady for D2H completion instead of PJRT_Event_Await.
/// Returns timing breakdown for comparison with blocking version.
public func pjrtExecuteWithCallbacks(
    client: OpaquePointer,
    executable: OpaquePointer,
    device: OpaquePointer,
    inputData: [UnsafeRawPointer],
    inputSizes: [Int],
    outputData: [UnsafeMutableRawPointer],
    outputSizes: [Int],
    semantics: PJRTClient.HostBufferSemantics
) throws -> PJRTExecutionTiming {
    // Convert to C-compatible arrays
    let inputPtrs = inputData.map { UnsafeRawPointer($0) }
    let inputSizesU = inputSizes.map { size_t($0) }
    let outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }
    let outputSizesU = outputSizes.map { size_t($0) }

    // Convert to PJRTCAPI semantics
    let capiSemantics = PJRTCAPI.HostBufferSemantics(semantics)

    var timing = SW_PJRT_ExecutionTiming()

    let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
        inputSizesU.withUnsafeBufferPointer { inputSizesBuf in
            outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                outputSizesU.withUnsafeBufferPointer { outputSizesBuf in
                    // Cast the buffer pointers to match C expectations
                    let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                    let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                    return PJRT_ExecuteWithCallbacks(
                        UnsafeMutableRawPointer(client),
                        UnsafeMutableRawPointer(executable),
                        UnsafeMutableRawPointer(device),
                        inputDataPtr,
                        inputSizesBuf.baseAddress,
                        inputData.count,
                        outputDataPtr,
                        outputSizesBuf.baseAddress,
                        outputData.count,
                        capiSemantics.toCType,
                        &timing
                    )
                }
            }
        }
    }

    if errorCode != SW_PJRT_Error_OK {
        throw PJRTError.executionFailed("PJRT_ExecuteWithCallbacks failed with code \(errorCode.rawValue)")
    }

    return PJRTExecutionTiming(from: timing)
}

// MARK: - Enhanced Buffer Pool V2

/// Enhanced buffer pool V2 with optimized execution path
///
/// This pool uses pre-initialized structures and batched transfers to minimize
/// per-call overhead. It's designed for repeated executions with the same shapes.
///
/// ## Performance Optimizations
/// - Pre-allocated H2D args structures (only data pointers updated per call)
/// - Batched D2H transfers (all initiated before any awaited)
/// - Thread-local storage for intermediate buffers
/// - Zero-copy semantics where possible
///
/// ## Usage Example
/// ```swift
/// let pool = try PJRTBufferPoolV2(
///     client: client.clientHandle!,
///     device: device.handle,
///     executable: exec.handle,
///     inputSizes: [100_000],
///     outputSizes: [100_000, 100_000]
/// )
///
/// // Execute many times with minimal overhead
/// for _ in 0..<1000 {
///     try pool.execute(inputData: [inputs], outputData: [outputs])
/// }
/// ```
public class PJRTBufferPoolV2 {
    fileprivate var pool: SW_PJRT_BufferPoolV2
    private let inputSizes: [Int]
    private let outputSizes: [Int]

    /// Number of executions performed
    public var executionCount: UInt64 {
        return pool.execution_count
    }

    /// Initialize an enhanced buffer pool
    ///
    /// - Parameters:
    ///   - client: PJRT client handle
    ///   - device: PJRT device handle
    ///   - executable: PJRT executable handle
    ///   - inputSizes: Number of F32 elements for each input (max 8)
    ///   - outputSizes: Number of F32 elements for each output (max 4)
    /// - Throws: PJRTError on failure
    public init(
        client: OpaquePointer,
        device: OpaquePointer,
        executable: OpaquePointer,
        inputSizes: [Int],
        outputSizes: [Int]
    ) throws {
        self.pool = SW_PJRT_BufferPoolV2()
        self.inputSizes = inputSizes
        self.outputSizes = outputSizes

        var inputSizesU = inputSizes.map { size_t($0) }
        var outputSizesU = outputSizes.map { size_t($0) }

        let errorCode = inputSizesU.withUnsafeMutableBufferPointer { inputBuf in
            outputSizesU.withUnsafeMutableBufferPointer { outputBuf in
                PJRT_BufferPoolV2Create(
                    UnsafeMutableRawPointer(client),
                    UnsafeMutableRawPointer(device),
                    UnsafeMutableRawPointer(executable),
                    inputBuf.baseAddress,
                    inputBuf.count,
                    outputBuf.baseAddress,
                    outputBuf.count,
                    &pool
                )
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.bufferCreationFailed("PJRT_BufferPoolV2Create failed with code \(errorCode.rawValue)")
        }
    }

    deinit {
        PJRT_BufferPoolV2Destroy(&pool)
    }

    /// Execute with optimized path
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input F32 data
    ///   - outputData: Array of raw pointers to preallocated output buffers
    ///   - timing: Optional timing breakdown (nil to skip timing)
    /// - Throws: PJRTError on failure
    public func execute(
        inputData: [UnsafeRawPointer],
        outputData: [UnsafeMutableRawPointer],
        timing: inout PJRTExecutionTiming?
    ) throws {
        let inputPtrs = inputData.map { UnsafeRawPointer($0) }
        let outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }

        var cTiming = SW_PJRT_ExecutionTiming()
        let wantTiming = timing != nil

        let errorCode: SW_PJRT_Error_Code = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                if wantTiming {
                    return PJRT_ExecutePooledV2(
                        &pool,
                        inputDataPtr,
                        outputDataPtr,
                        &cTiming
                    )
                } else {
                    return PJRT_ExecutePooledV2(
                        &pool,
                        inputDataPtr,
                        outputDataPtr,
                        nil
                    )
                }
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecutePooledV2 failed with code \(errorCode.rawValue)")
        }

        if wantTiming {
            timing = PJRTExecutionTiming(from: cTiming)
        }
    }

    /// Execute with timing (convenience method)
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input F32 data
    ///   - outputData: Array of raw pointers to preallocated output buffers
    /// - Returns: Timing breakdown
    /// - Throws: PJRTError on failure
    public func executeWithTiming(
        inputData: [UnsafeRawPointer],
        outputData: [UnsafeMutableRawPointer]
    ) throws -> PJRTExecutionTiming {
        var timing: PJRTExecutionTiming? = PJRTExecutionTiming(from: SW_PJRT_ExecutionTiming())
        try execute(inputData: inputData, outputData: outputData, timing: &timing)
        return timing!
    }

    /// Execute without timing (fastest path)
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input F32 data
    ///   - outputData: Array of raw pointers to preallocated output buffers
    /// - Throws: PJRTError on failure
    public func execute(
        inputData: [UnsafeRawPointer],
        outputData: [UnsafeMutableRawPointer]
    ) throws {
        var timing: PJRTExecutionTiming? = nil
        try execute(inputData: inputData, outputData: outputData, timing: &timing)
    }

    /// Ultra-fast execution path
    ///
    /// This uses PJRT_ExecuteUltraFast which has the lowest per-call overhead.
    /// No timing is provided.
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input F32 data
    ///   - outputData: Array of raw pointers to preallocated output buffers
    /// - Throws: PJRTError on failure
    public func executeUltraFast(
        inputData: [UnsafeRawPointer],
        outputData: [UnsafeMutableRawPointer]
    ) throws {
        let inputPtrs = inputData.map { UnsafeRawPointer($0) }
        let outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }

        let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                return PJRT_ExecuteUltraFast(
                    &pool,
                    inputDataPtr,
                    outputDataPtr
                )
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_ExecuteUltraFast failed with code \(errorCode.rawValue)")
        }
    }
}

// MARK: - Batched Transfer API

/// Batched H2D transfer - creates multiple device buffers from host data
///
/// This batches multiple buffer creation calls to reduce per-buffer overhead.
///
/// - Parameters:
///   - client: PJRT client handle
///   - device: PJRT device handle
///   - data: Array of host data pointers (F32)
///   - sizes: Array of element counts for each buffer
///   - semantics: Host buffer semantics to use
/// - Returns: Array of device buffer handles
/// - Throws: PJRTError on failure
public func pjrtBatchedH2DTransfer(
    client: OpaquePointer,
    device: OpaquePointer,
    data: [UnsafeRawPointer],
    sizes: [Int],
    semantics: PJRTClient.HostBufferSemantics
) throws -> [OpaquePointer] {
    guard data.count == sizes.count else {
        throw PJRTError.bufferCreationFailed("data and sizes arrays must have same length")
    }
    guard data.count <= 8 else {
        throw PJRTError.bufferCreationFailed("Maximum 8 buffers in batched transfer")
    }

    let dataPtrs = data.map { UnsafeRawPointer($0) }
    let sizesU = sizes.map { size_t($0) }
    var outBuffers = [UnsafeMutableRawPointer?](repeating: nil, count: data.count)

    let capiSemantics = PJRTCAPI.HostBufferSemantics(semantics)

    let errorCode = dataPtrs.withUnsafeBufferPointer { dataBuf in
        sizesU.withUnsafeBufferPointer { sizesBuf in
            outBuffers.withUnsafeMutableBufferPointer { outBuf in
                let dataPtr = UnsafeRawPointer(dataBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)

                return PJRT_BatchedH2DTransfer(
                    UnsafeMutableRawPointer(client),
                    UnsafeMutableRawPointer(device),
                    dataPtr,
                    sizesBuf.baseAddress,
                    data.count,
                    capiSemantics.toCType,
                    outBuf.baseAddress
                )
            }
        }
    }

    if errorCode != SW_PJRT_Error_OK {
        throw PJRTError.bufferCreationFailed("PJRT_BatchedH2DTransfer failed with code \(errorCode.rawValue)")
    }

    return outBuffers.compactMap { $0.map { OpaquePointer($0) } }
}

/// Batched D2H transfer - transfers multiple buffers from device to host
///
/// This batches transfer initiation and awaiting to enable overlap.
///
/// - Parameters:
///   - buffers: Array of device buffer handles
///   - destinations: Array of destination host pointers
///   - sizes: Array of sizes in bytes
///   - destroyBuffers: Whether to destroy source buffers after transfer
/// - Throws: PJRTError on failure
public func pjrtBatchedD2HTransfer(
    buffers: [OpaquePointer],
    destinations: [UnsafeMutableRawPointer],
    sizes: [Int],
    destroyBuffers: Bool
) throws {
    guard buffers.count == destinations.count && buffers.count == sizes.count else {
        throw PJRTError.bufferTransferFailed("buffers, destinations, and sizes arrays must have same length")
    }
    guard buffers.count <= 8 else {
        throw PJRTError.bufferTransferFailed("Maximum 8 buffers in batched transfer")
    }

    var bufferPtrs = buffers.map { UnsafeMutableRawPointer($0) as UnsafeMutableRawPointer? }
    var destPtrs = destinations.map { UnsafeMutableRawPointer($0) as UnsafeMutableRawPointer? }
    let sizesU = sizes.map { size_t($0) }

    let errorCode = bufferPtrs.withUnsafeMutableBufferPointer { bufBuf in
        destPtrs.withUnsafeMutableBufferPointer { destBuf in
            sizesU.withUnsafeBufferPointer { sizesBuf in
                return PJRT_BatchedD2HTransfer(
                    bufBuf.baseAddress,
                    destBuf.baseAddress,
                    sizesBuf.baseAddress,
                    buffers.count,
                    destroyBuffers
                )
            }
        }
    }

    if errorCode != SW_PJRT_Error_OK {
        throw PJRTError.bufferTransferFailed("PJRT_BatchedD2HTransfer failed with code \(errorCode.rawValue)")
    }
}

// MARK: - Pipelined Execution API

/// Pipelined execution for overlapping compute and transfer
///
/// The pipeline allows multiple executions to be "in flight" simultaneously,
/// hiding D2H transfer latency by overlapping it with the next execution.
///
/// ## How It Works
/// ```
/// Execution 1: [H2D][EXEC][D2H transfer...........]
/// Execution 2:           [H2D][EXEC][D2H transfer...........]
/// Execution 3:                     [H2D][EXEC][D2H transfer...]
///                              ↑ overlap hides latency!
/// ```
///
/// ## Usage Example
/// ```swift
/// let pipeline = try PJRTExecutionPipeline(pool: pool, depth: 2)
///
/// // Submit multiple executions
/// for i in 0..<100 {
///     try pipeline.submit(inputData: inputs[i], outputData: outputs[i])
/// }
///
/// // Flush to ensure all complete
/// try pipeline.flush()
/// ```
///
/// ## Performance Impact
/// With depth=2, D2H latency is almost completely hidden after warmup.
/// Effective throughput approaches: 1 / (H2D + Execute) instead of 1 / (H2D + Execute + D2H)
public class PJRTExecutionPipeline {
    private var pipeline: SW_PJRT_ExecutionPipeline
    private let pool: PJRTBufferPoolV2

    /// Number of in-flight executions currently in the pipeline
    public var inFlight: Int {
        return Int(pipeline.in_flight)
    }

    /// Total number of submissions
    public var totalSubmissions: UInt64 {
        return pipeline.total_submissions
    }

    /// Total number of completions
    public var totalCompletions: UInt64 {
        return pipeline.total_completions
    }

    /// Initialize a pipelined execution context
    ///
    /// - Parameters:
    ///   - pool: Buffer pool V2 to use (must remain valid for pipeline lifetime)
    ///   - depth: Pipeline depth (1-4). Higher values provide more overlap but use more memory.
    ///            Recommended: 2 for most cases.
    /// - Throws: PJRTError on failure
    public init(pool: PJRTBufferPoolV2, depth: Int = 2) throws {
        self.pool = pool
        self.pipeline = SW_PJRT_ExecutionPipeline()

        let errorCode = withUnsafeMutablePointer(to: &pool.pool) { poolPtr in
            PJRT_PipelineCreate(poolPtr, depth, &pipeline)
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_PipelineCreate failed with code \(errorCode.rawValue)")
        }
    }

    deinit {
        PJRT_PipelineDestroy(&pipeline)
    }

    /// Submit an execution to the pipeline (non-blocking if slots available)
    ///
    /// This performs H2D + Execute + initiates D2H, then returns immediately.
    /// If all pipeline slots are full, this blocks until one completes.
    ///
    /// - Parameters:
    ///   - inputData: Array of raw pointers to input F32 data
    ///   - outputData: Array of raw pointers to preallocated output buffers
    /// - Throws: PJRTError on failure
    public func submit(
        inputData: [UnsafeRawPointer],
        outputData: [UnsafeMutableRawPointer]
    ) throws {
        let inputPtrs = inputData.map { UnsafeRawPointer($0) }
        let outputPtrs = outputData.map { UnsafeMutableRawPointer($0) }

        let errorCode = inputPtrs.withUnsafeBufferPointer { inputPtrsBuf in
            outputPtrs.withUnsafeBufferPointer { outputPtrsBuf in
                let inputDataPtr = UnsafeRawPointer(inputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeRawPointer?.self)
                let outputDataPtr = UnsafeMutableRawPointer(mutating: outputPtrsBuf.baseAddress!).assumingMemoryBound(to: UnsafeMutableRawPointer?.self)

                return PJRT_PipelineSubmit(
                    &pipeline,
                    inputDataPtr,
                    outputDataPtr
                )
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_PipelineSubmit failed with code \(errorCode.rawValue)")
        }
    }

    /// Wait for the oldest in-flight execution to complete
    ///
    /// - Throws: PJRTError on failure or if pipeline is empty
    public func awaitOne() throws {
        let errorCode = PJRT_PipelineAwaitOne(&pipeline, nil)

        if errorCode == SW_PJRT_Error_NOT_FOUND {
            throw PJRTError.executionFailed("Pipeline is empty - no executions to await")
        }

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_PipelineAwaitOne failed with code \(errorCode.rawValue)")
        }
    }

    /// Flush the pipeline - wait for all in-flight executions to complete
    ///
    /// - Throws: PJRTError on failure
    public func flush() throws {
        let errorCode = PJRT_PipelineFlush(&pipeline)

        if errorCode != SW_PJRT_Error_OK {
            throw PJRTError.executionFailed("PJRT_PipelineFlush failed with code \(errorCode.rawValue)")
        }
    }

    /// Get pipeline statistics
    ///
    /// - Returns: Tuple of (submissions, completions, averageWaitMicroseconds)
    public func getStats() -> (submissions: UInt64, completions: UInt64, avgWaitUs: Double) {
        var submissions: UInt64 = 0
        var completions: UInt64 = 0
        var avgWaitUs: Double = 0

        PJRT_PipelineGetStats(&pipeline, &submissions, &completions, &avgWaitUs)

        return (submissions, completions, avgWaitUs)
    }
}
