//===-- PJRTCAPI.swift - PJRT C API Swift Wrapper --------*- Swift -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// Swift wrapper for simplified PJRT C API
//
//===------------------------------------------------------------------===//

import Foundation
import SwiftIRCore
import PJRTCAPI  // C module defined in PJRTCWrappers/include/module.modulemap

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

    /// Create a buffer from host data
    func createBuffer(
        client: OpaquePointer,
        data: UnsafeRawPointer,
        type: SW_PJRT_Buffer_Type,
        dims: [Int64],
        device: OpaquePointer
    ) throws -> OpaquePointer {
        var buffer: UnsafeMutableRawPointer? = nil

        let errorCode = dims.withUnsafeBufferPointer { dimsPtr in
            PJRT_CreateBuffer(
                UnsafeMutableRawPointer(client),
                data,
                type,
                dimsPtr.baseAddress,
                dims.count,
                UnsafeMutableRawPointer(device),
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

    /// Transfer buffer from device to host
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
        var executable: UnsafeMutableRawPointer? = nil

        let errorCode = mlirModule.withCString { modulePtr in
            PJRT_CompileWrapper(
                UnsafeMutableRawPointer(client),
                modulePtr,
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
}
