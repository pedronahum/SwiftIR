// PJRTProfiler.swift - PJRT Profiler Extension bindings
//
// This module provides Swift bindings to the PJRT profiler extension,
// which is embedded in the PJRT CPU plugin with profiler support.
//
// The PJRT profiler extension lives in the same binary as the PJRT client,
// ensuring that HLO modules registered during compilation are captured by
// the MetadataCollector in the profiler.
//
// IMPORTANT: This module supports two loading modes:
// 1. Dynamic mode (via ProfilerBindingsDynamic): Used when SwiftIRJupyter loads
//    libPJRTSimpleWrapper.so dynamically. This ensures we share the same g_api
//    global as SwiftIRJupyter.
// 2. Static mode (via PJRTCAPI): Used when running standalone (e.g., ProfilerDemo)
//    where PJRTSimpleWrapper.c is statically linked.
//
// Usage:
//   // After PJRTClient is initialized:
//   guard PJRTProfiler.isAvailable else { return }
//   let profiler = try PJRTProfiler.create()
//   try profiler.start()
//   // ... do XLA computations ...
//   try profiler.stop()
//   let data = try profiler.collectData()

import Foundation
import PJRTCAPI

// MARK: - Dynamic/Static Binding Selection

/// Helper to select between dynamic and static bindings
/// Dynamic bindings are preferred because they share g_api with SwiftIRJupyter
private enum ProfilerBinding {
    /// Try to load dynamic bindings and check if profiler is available
    static func loadDynamicIfAvailable() -> Bool {
        do {
            try ProfilerBindingsDynamic.shared.load()
            return ProfilerBindingsDynamic.shared.hasProfilerExtension()
        } catch {
            return false
        }
    }

    /// Check if dynamic bindings are loaded and have profiler extension
    static var dynamicAvailable: Bool {
        ProfilerBindingsDynamic.shared.isLoaded && ProfilerBindingsDynamic.shared.hasProfilerExtension()
    }

    /// Check if static bindings have profiler extension
    static var staticAvailable: Bool {
        PJRT_HasProfilerExtension()
    }
}

// MARK: - PJRT Profiler Errors

/// Errors from the PJRT profiler extension
public enum PJRTProfilerError: Error, CustomStringConvertible {
    case notAvailable
    case creationFailed
    case startFailed
    case stopFailed
    case collectDataFailed
    case internalError(code: Int32)

    public var description: String {
        switch self {
        case .notAvailable:
            return "PJRT profiler extension not available. Make sure PJRT plugin is loaded first."
        case .creationFailed:
            return "Failed to create PJRT profiler"
        case .startFailed:
            return "Failed to start PJRT profiler"
        case .stopFailed:
            return "Failed to stop PJRT profiler"
        case .collectDataFailed:
            return "Failed to collect PJRT profiler data"
        case .internalError(let code):
            return "PJRT profiler internal error: \(code)"
        }
    }
}

// MARK: - PJRT Profiler

/// High-level Swift interface to the PJRT profiler extension
public final class PJRTProfiler {
    /// Handle to the underlying PJRT profiler
    private var handle: OpaquePointer?

    /// Whether the profiler has been started
    private var isStarted: Bool = false

    /// Default ProfileOptions protobuf with:
    /// - version = 1 (enable non-default behavior)
    /// - host_tracer_level = 2 (detailed tracing)
    /// - device_tracer_level = 1 (enable device tracing)
    /// - enable_hlo_proto = true (capture XLA HLO modules)
    ///
    /// Wire format: field 2 (host)=2, field 3 (device)=1, field 5 (version)=1, field 7 (hlo)=true
    private static let defaultProfileOptions = Data([0x10, 0x02, 0x18, 0x01, 0x28, 0x01, 0x38, 0x01])

    /// Whether we're using dynamic bindings (shared g_api with SwiftIRJupyter)
    nonisolated(unsafe) private static var useDynamic: Bool = false

    /// Check if PJRT profiler extension is available
    /// Note: PJRT plugin must be loaded first (by creating a PJRTClient)
    ///
    /// This property tries dynamic bindings first (for SwiftIRJupyter compatibility),
    /// then falls back to static bindings (for standalone use).
    public static var isAvailable: Bool {
        // First try dynamic bindings - this shares g_api with SwiftIRJupyter
        if ProfilerBinding.loadDynamicIfAvailable() {
            useDynamic = true
            return true
        }
        // Fall back to static bindings
        if ProfilerBinding.staticAvailable {
            useDynamic = false
            return true
        }
        return false
    }

    /// Create a new PJRT profiler
    /// - Parameter options: Serialized ProfileOptions protobuf (optional, uses defaults if nil)
    /// - Throws: PJRTProfilerError if creation fails
    public init(options: Data? = nil) throws {
        guard Self.isAvailable else {
            throw PJRTProfilerError.notAvailable
        }

        // Use provided options or default options with HLO proto enabled
        let effectiveOptions = options ?? Self.defaultProfileOptions

        if Self.useDynamic {
            // Use dynamic bindings (shared g_api with SwiftIRJupyter)
            let profilerHandle = try effectiveOptions.withUnsafeBytes { optionsPtr -> UnsafeMutableRawPointer in
                try ProfilerBindingsDynamic.shared.profilerCreate(
                    options: optionsPtr.baseAddress?.assumingMemoryBound(to: CChar.self),
                    optionsSize: effectiveOptions.count
                )
            }
            self.handle = OpaquePointer(profilerHandle)
        } else {
            // Use static bindings
            var profilerHandle: UnsafeMutableRawPointer?

            let result = effectiveOptions.withUnsafeBytes { optionsPtr in
                PJRT_ProfilerCreate(
                    optionsPtr.baseAddress?.assumingMemoryBound(to: CChar.self),
                    effectiveOptions.count,
                    &profilerHandle
                )
            }

            guard result == SW_PJRT_Error_OK, let handle = profilerHandle else {
                throw PJRTProfilerError.creationFailed
            }

            self.handle = OpaquePointer(handle)
        }
    }

    /// Create a PJRT profiler (static factory)
    public static func create(options: Data? = nil) throws -> PJRTProfiler {
        return try PJRTProfiler(options: options)
    }

    deinit {
        if let handle = handle {
            if Self.useDynamic {
                ProfilerBindingsDynamic.shared.profilerDestroy(UnsafeMutableRawPointer(handle))
            } else {
                _ = PJRT_ProfilerDestroy(UnsafeMutableRawPointer(handle))
            }
        }
    }

    /// Start profiling
    public func start() throws {
        guard let handle = handle else {
            throw PJRTProfilerError.notAvailable
        }

        if Self.useDynamic {
            try ProfilerBindingsDynamic.shared.profilerStart(UnsafeMutableRawPointer(handle))
        } else {
            let result = PJRT_ProfilerStart(UnsafeMutableRawPointer(handle))
            guard result == SW_PJRT_Error_OK else {
                throw PJRTProfilerError.startFailed
            }
        }

        isStarted = true
    }

    /// Stop profiling
    public func stop() throws {
        guard let handle = handle else {
            throw PJRTProfilerError.notAvailable
        }

        if Self.useDynamic {
            try ProfilerBindingsDynamic.shared.profilerStop(UnsafeMutableRawPointer(handle))
        } else {
            let result = PJRT_ProfilerStop(UnsafeMutableRawPointer(handle))
            guard result == SW_PJRT_Error_OK else {
                throw PJRTProfilerError.stopFailed
            }
        }

        isStarted = false
    }

    /// Collect profiler data as XSpace protobuf
    /// - Returns: Serialized XSpace protobuf data
    public func collectData() throws -> Data {
        guard let handle = handle else {
            throw PJRTProfilerError.notAvailable
        }

        var size: Int = 0

        if Self.useDynamic {
            // First call to get size
            try ProfilerBindingsDynamic.shared.profilerCollectData(
                UnsafeMutableRawPointer(handle),
                buffer: nil,
                size: &size
            )

            guard size > 0 else {
                throw PJRTProfilerError.collectDataFailed
            }

            print("PJRTProfiler: Allocated buffer size: \(size)")

            // Allocate buffer and collect data
            var buffer = [UInt8](repeating: 0, count: size)
            try buffer.withUnsafeMutableBufferPointer { bufferPtr in
                try ProfilerBindingsDynamic.shared.profilerCollectData(
                    UnsafeMutableRawPointer(handle),
                    buffer: bufferPtr.baseAddress,
                    size: &size
                )
            }

            print("PJRTProfiler: Actual data size: \(size)")

            // Use only the actual data size
            var actualSize = size
            while actualSize > 0 && buffer[actualSize - 1] == 0 {
                actualSize -= 1
                print("PJRTProfiler: Removing trailing null byte, new size: \(actualSize)")
            }

            let rawData = Data(buffer[0..<actualSize])
            return repairXSpace(rawData)
        } else {
            // Static bindings path
            var result = PJRT_ProfilerCollectData(
                UnsafeMutableRawPointer(handle),
                nil,
                &size
            )

            guard result == SW_PJRT_Error_OK, size > 0 else {
                throw PJRTProfilerError.collectDataFailed
            }

            let allocatedSize = size
            print("PJRTProfiler: Allocated buffer size: \(allocatedSize)")

            // Allocate buffer and collect data
            var buffer = [UInt8](repeating: 0, count: size)
            result = buffer.withUnsafeMutableBytes { bufferPtr in
                PJRT_ProfilerCollectData(
                    UnsafeMutableRawPointer(handle),
                    bufferPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                    &size
                )
            }

            guard result == SW_PJRT_Error_OK else {
                throw PJRTProfilerError.collectDataFailed
            }

            print("PJRTProfiler: Actual data size: \(size)")

            // Use only the actual data size (size may have been updated by the second call)
            // Also check for and remove any trailing null bytes which are invalid protobuf
            var actualSize = size
            while actualSize > 0 && buffer[actualSize - 1] == 0 {
                actualSize -= 1
                print("PJRTProfiler: Removing trailing null byte, new size: \(actualSize)")
            }

            let rawData = Data(buffer[0..<actualSize])
            // Check for corruption and repair if needed
            let repairedData = repairXSpace(rawData)
            return repairedData
        }
    }

    /// Process XSpace data from PJRT profiler
    ///
    /// This function:
    /// 1. Validates that Task Environment plane is present (required by TensorBoard)
    /// 2. Adds hostname field if missing (field 4 in XSpace)
    ///
    /// The hostname field is required by TensorBoard for multi-host profile aggregation
    /// and display purposes.
    private func repairXSpace(_ data: Data) -> Data {
        // Debug: Check what's in the data
        let taskEnvPattern = "Task Environment".data(using: .utf8)!
        if data.range(of: taskEnvPattern) != nil {
            print("PJRTProfiler: Task Environment plane found (required by TensorBoard)")
        } else {
            print("PJRTProfiler: Warning - Task Environment plane not found")
        }

        // Check if hostname field is already present
        // The hostname field should appear after the last XPlane (field 1) closes
        // We look for the pattern: } followed by field 4 tag (0x22)
        let hasHostname = checkForHostnameField(in: data)

        if hasHostname {
            print("PJRTProfiler: Hostname field present")
            return data
        }

        // Add hostname field (field 4) to XSpace
        print("PJRTProfiler: Adding missing hostname field")
        return addHostnameField(to: data)
    }

    /// Check if the XSpace data contains a hostname field (field 4)
    /// The hostname appears at the end of the XSpace, after all XPlane messages
    private func checkForHostnameField(in data: Data) -> Bool {
        // Look for field 4 tag (0x22) followed by a length and hostname string
        // This should appear at the top level, after the last XPlane closes
        // We scan from the end since hostname is typically the last field

        guard data.count > 10 else { return false }

        // Simple heuristic: check if the data ends with a hostname-like pattern
        // Field 4 tag = 0x22, followed by length, followed by string
        for i in stride(from: data.count - 5, through: max(0, data.count - 100), by: -1) {
            if data[i] == 0x22 {
                // Check if this looks like a valid length-delimited field
                let lenByte = data[i + 1]
                if lenByte > 0 && lenByte < 64 {  // Reasonable hostname length
                    // Check if remaining bytes match the length
                    let expectedEnd = i + 2 + Int(lenByte)
                    if expectedEnd == data.count {
                        // Verify it looks like a hostname (alphanumeric with dashes)
                        let hostnameStart = i + 2
                        let hostnameBytes = data[hostnameStart..<data.count]
                        if let hostname = String(data: hostnameBytes, encoding: .utf8),
                           hostname.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == "." }) {
                            return true
                        }
                    }
                }
            }
        }
        return false
    }

    /// Add hostname field (field 4) to XSpace protobuf data
    /// XSpace.hostnames is a repeated string field (field number 4)
    /// Wire format: tag (0x22) + length + string bytes
    private func addHostnameField(to data: Data) -> Data {
        let hostname = ProcessInfo.processInfo.hostName
        guard let hostnameBytes = hostname.data(using: .utf8) else {
            print("PJRTProfiler: Failed to encode hostname")
            return data
        }

        // Build the field: tag (0x22) + varint length + string
        var field = Data()
        field.append(0x22)  // Field 4, wire type 2 (length-delimited)

        // Encode length as varint
        var length = hostnameBytes.count
        while length > 0x7F {
            field.append(UInt8((length & 0x7F) | 0x80))
            length >>= 7
        }
        field.append(UInt8(length))

        // Append hostname bytes
        field.append(hostnameBytes)

        print("PJRTProfiler: Added hostname '\(hostname)' (\(field.count) bytes)")

        // Append to existing data
        var result = data
        result.append(field)
        return result
    }
    
    /// Rename a metadata string in the XSpace data
    /// This is used to rename "_pt" to "hlo_proto" to fix TensorBoard visualization
    private func renameMetadata(in data: Data, from oldName: String, to newName: String) -> Data {
        var repaired = data
        let oldBytes = oldName.data(using: .utf8)!
        let newBytes = newName.data(using: .utf8)!
        
        // We look for the pattern:
        // 0x2a (Field 5 - stat_metadata)
        //   <len>
        //     ...
        //     0x12 (Field 2 - XStatMetadata value)
        //       <len>
        //         ...
        //         0x12 (Field 2 - name)
        //           <len>
        //             oldName
        
        // Simplified search: Look for 0x12 <len> oldName
        // And verify it's inside a stat_metadata entry if possible, or just be aggressive since "_pt" is rare.
        // 0x12 is Field 2.
        
        var i = 0
        while i < repaired.count - oldBytes.count - 2 {
            if repaired[i] == 0x12 { // Field 2 (Name)
                // Check length
                let lenIdx = i + 1
                // Assuming short strings for now (1 byte length)
                // oldName is "_pt" (3 bytes), so length is 3.
                if repaired[lenIdx] == UInt8(oldBytes.count) {
                    let strIdx = lenIdx + 1
                    if repaired.subdata(in: strIdx..<strIdx+oldBytes.count) == oldBytes {
                        // Found it!
                        // Now we need to find the parent message (XStatMetadata) to update its length.
                        // The parent message is likely Field 2 of a Map Entry.
                        // Map Entry: 0x2a <len> ... 0x12 <parent_len> <parent_content>
                        
                        // We need to walk back to find the start of the parent message.
                        // This is tricky without a full parser.
                        // But we know the parent contains this string.
                        
                        // Let's try a simpler approach:
                        // Just replace the string and update the immediate length.
                        // AND update the length of the *containing* length-delimited field if it's close.
                        
                        // Actually, if we change size, we invalidate ALL parent lengths.
                        // But we only care about the Map Entry length (Field 5).
                        // And the XPlane length (which we fix later in repairXSpace).
                        
                        // So we need to fix:
                        // 1. The string length (easy).
                        // 2. The XStatMetadata length (Field 2 of Map Entry).
                        // 3. The Map Entry length (Field 5 of XPlane).
                        
                        // Let's scan backwards to find these containers.
                        
                        // Backtrack to find 0x12 (parent tag)
                        var parentStart: Int?
                        for j in 1...20 { // Heuristic: header shouldn't be too far
                            let idx = i - j
                            if idx < 0 { break }
                            if repaired[idx] == 0x12 {
                                // Check if the length matches the distance to current point + remaining
                                // This is hard.
                                parentStart = idx
                                break
                            }
                        }
                        
                        // Backtrack further to find 0x2a (grandparent tag - Map Entry)
                        var grandParentStart: Int?
                        if let pStart = parentStart {
                            for j in 1...20 {
                                let idx = pStart - j
                                if idx < 0 { break }
                                if repaired[idx] == 0x2a {
                                    grandParentStart = idx
                                    break
                                }
                            }
                        }
                        
                        if let pStart = parentStart, let gpStart = grandParentStart {
                            print("PJRTProfiler: Renaming metadata '\(oldName)' to '\(newName)' at offset \(i)")
                            
                            // 1. Calculate size difference
                            let sizeDiff = newBytes.count - oldBytes.count
                            
                            // 2. Update string
                            // Insert new bytes
                            repaired.replaceSubrange(strIdx..<strIdx+oldBytes.count, with: newBytes)
                            // Update string length (assuming 1 byte for now, hlo_proto is 9 bytes)
                            repaired[lenIdx] = UInt8(newBytes.count)
                            
                            // 3. Update Parent (XStatMetadata) Length
                            // pStart points to 0x12. pStart+1 is length.
                            // We need to read varint, add sizeDiff, write back.
                            // If varint size changes, we have to shift again!
                            
                            func patchVarint(at index: Int, add diff: Int) -> Int {
                                var val: UInt64 = 0
                                var shift: UInt64 = 0
                                var k = index
                                while k < repaired.count {
                                    let b = repaired[k]
                                    val |= (UInt64(b & 0x7F) << shift)
                                    shift += 7
                                    k += 1
                                    if (b & 0x80) == 0 { break }
                                }
                                let oldLenLen = k - index
                                let oldLen = val
                                let newLen = oldLen + UInt64(diff)
                                
                                var newLenBytes: [UInt8] = []
                                var v = newLen
                                while true {
                                    let b = UInt8(v & 0x7F)
                                    v >>= 7
                                    if v == 0 {
                                        newLenBytes.append(b)
                                        break
                                    } else {
                                        newLenBytes.append(b | 0x80)
                                    }
                                }
                                
                                let newLenLen = newLenBytes.count
                                let lenDiff = newLenLen - oldLenLen
                                
                                if lenDiff != 0 {
                                    if lenDiff > 0 {
                                        repaired.insert(contentsOf: [UInt8](repeating: 0, count: lenDiff), at: index)
                                    } else {
                                        repaired.removeSubrange(index..<index - lenDiff)
                                    }
                                }
                                
                                for x in 0..<newLenBytes.count {
                                    repaired[index + x] = newLenBytes[x]
                                }
                                
                                return lenDiff
                            }
                            
                            // Patch parent length
                            // pStart is 0x12. Length starts at pStart + 1.
                            let pLenDiff = patchVarint(at: pStart + 1, add: sizeDiff)
                            
                            // 4. Update Grandparent (Map Entry) Length
                            // gpStart is 0x2a. Length starts at gpStart + 1.
                            // Note: gpStart might have shifted if pLenDiff != 0?
                            // No, gpStart is before pStart.
                            // But the content *inside* grandparent grew by sizeDiff + pLenDiff.
                            let gpLenDiff = patchVarint(at: gpStart + 1, add: sizeDiff + pLenDiff)
                            
                            // Adjust indices for next iteration
                            i += sizeDiff + pLenDiff + gpLenDiff
                        }
                    }
                }
            }
            i += 1
        }
        
        return repaired
    }

    /// Export collected data to file
    /// - Parameters:
    ///   - data: XSpace protobuf data
    ///   - filepath: Path to output file
    public static func exportToFile(_ data: Data, filepath: String) throws {
        try data.withUnsafeBytes { dataPtr in
            try filepath.withCString { cPath in
                guard let fd = fopen(cPath, "wb") else {
                    throw PJRTProfilerError.internalError(code: -2)
                }
                defer { fclose(fd) }

                let written = fwrite(
                    dataPtr.baseAddress,
                    1,
                    data.count,
                    fd
                )

                guard written == data.count else {
                    throw PJRTProfilerError.internalError(code: -3)
                }
            }
        }
    }

    // MARK: - TraceMe API

    /// Check if TraceMe API is available in the PJRT plugin
    ///
    /// When true, traces created via `PJRTProfiler.traced()` will appear in
    /// the same profile as XLA internal traces (compilation, execution, etc).
    public static var hasTraceMeApi: Bool {
        // Try dynamic first
        if ProfilerBindingsDynamic.shared.isLoaded {
            return ProfilerBindingsDynamic.shared.hasTraceMeApi()
        }
        // Try to load dynamic
        do {
            try ProfilerBindingsDynamic.shared.load()
            if ProfilerBindingsDynamic.shared.hasTraceMeApi() {
                return true
            }
        } catch {
            // Ignore - fall through to static
        }
        // Fall back to static
        return PJRT_HasTraceMeApi()
    }

    /// Check if tracing is currently active at the given level
    ///
    /// Use this to avoid expensive string operations when tracing is disabled.
    /// - Parameter level: Trace level (1=critical, 2=info, 3=verbose)
    /// - Returns: true if tracing is active at this level
    public static func traceMeActive(level: Int32 = 1) -> Bool {
        if ProfilerBindingsDynamic.shared.isLoaded {
            return ProfilerBindingsDynamic.shared.traceMeActive(level: level)
        }
        return PJRT_TraceMeActive(level)
    }

    /// Start a TraceMe activity
    ///
    /// Returns an activity ID that must be passed to `traceMeStop()` to end the trace.
    /// The activity will appear in the TensorBoard Trace Viewer alongside XLA operations.
    ///
    /// - Parameters:
    ///   - name: Name to display in the Trace Viewer
    ///   - level: Trace level (1=critical/always shown, 2=info, 3=verbose)
    /// - Returns: Activity ID (or 0 if tracing is disabled)
    public static func traceMeStart(_ name: String, level: Int32 = 1) -> Int64 {
        if ProfilerBindingsDynamic.shared.isLoaded {
            return ProfilerBindingsDynamic.shared.traceMeStart(name, level: level)
        }
        return name.withCString { cName in
            PJRT_TraceMeStart(cName, level)
        }
    }

    /// Stop a TraceMe activity
    ///
    /// - Parameter activityId: Activity ID returned by `traceMeStart()`
    public static func traceMeStop(_ activityId: Int64) {
        if ProfilerBindingsDynamic.shared.isLoaded {
            ProfilerBindingsDynamic.shared.traceMeStop(activityId)
        } else {
            PJRT_TraceMeStop(activityId)
        }
    }

    /// Record an instant trace event (a point in time, no duration)
    ///
    /// - Parameters:
    ///   - name: Name to display in the Trace Viewer
    ///   - level: Trace level (1=critical, 2=info, 3=verbose)
    public static func traceMeInstant(_ name: String, level: Int32 = 1) {
        if ProfilerBindingsDynamic.shared.isLoaded {
            ProfilerBindingsDynamic.shared.traceMeInstant(name, level: level)
        } else {
            name.withCString { cName in
                PJRT_TraceMeInstant(cName, level)
            }
        }
    }

    /// Execute a block within a PJRT TraceMe scope
    ///
    /// The trace will appear in the same profile as XLA internal traces,
    /// making it easy to correlate Swift code with XLA operations.
    ///
    /// - Parameters:
    ///   - name: Name to display in the Trace Viewer
    ///   - level: Trace level (1=critical, 2=info, 3=verbose)
    ///   - block: Code to execute within the trace scope
    /// - Returns: Result of the block
    @discardableResult
    public static func traced<T>(_ name: String, level: Int32 = 1, _ block: () throws -> T) rethrows -> T {
        let activityId = traceMeStart(name, level: level)
        defer { traceMeStop(activityId) }
        return try block()
    }

    /// Execute an async block within a PJRT TraceMe scope
    @available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
    @discardableResult
    public static func traced<T>(_ name: String, level: Int32 = 1, _ block: () async throws -> T) async rethrows -> T {
        let activityId = traceMeStart(name, level: level)
        defer { traceMeStop(activityId) }
        return try await block()
    }
}

// MARK: - Global TraceMe Functions (use PJRT TraceMe when available)

/// Execute a block within a trace scope using PJRT TraceMe if available
///
/// This function automatically uses the PJRT plugin's TraceMe when profiling with
/// PJRTProfiler, ensuring Swift traces appear alongside XLA internal traces.
/// Falls back to ProfilerBindings TraceMe when PJRT TraceMe is not available.
///
/// - Parameters:
///   - name: Name to display in the Trace Viewer
///   - level: Trace level (1=critical, 2=info, 3=verbose)
///   - block: Code to execute within the trace scope
/// - Returns: Result of the block
@discardableResult
public func pjrtTraced<T>(_ name: String, level: Int32 = 1, _ block: () throws -> T) rethrows -> T {
    if PJRTProfiler.hasTraceMeApi {
        return try PJRTProfiler.traced(name, level: level, block)
    } else {
        // Fall back to ProfilerBindings TraceMe
        return try traced(name, level: level, block)
    }
}

/// Execute an async block within a trace scope using PJRT TraceMe if available
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
@discardableResult
public func pjrtTraced<T>(_ name: String, level: Int32 = 1, _ block: () async throws -> T) async rethrows -> T {
    if PJRTProfiler.hasTraceMeApi {
        return try await PJRTProfiler.traced(name, level: level, block)
    } else {
        // Fall back to ProfilerBindings TraceMe
        return try await traced(name, level: level, block)
    }
}

/// Create a training step marker using PJRT TraceMe if available
///
/// TensorBoard's Overview Page uses step markers to calculate step timing.
/// This function creates a trace event that TensorBoard recognizes.
///
/// - Parameters:
///   - step: The step number (0-indexed)
///   - block: The training step code to execute
/// - Returns: Result of the block
@discardableResult
public func pjrtTrainStep<T>(_ step: Int, _ block: () throws -> T) rethrows -> T {
    let name = "TrainStep#step_num=\(step),_r=1#"
    if PJRTProfiler.hasTraceMeApi {
        return try PJRTProfiler.traced(name, level: 1, block)
    } else {
        return try trainStep(step, block)
    }
}

/// Async version of pjrtTrainStep
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
@discardableResult
public func pjrtTrainStep<T>(_ step: Int, _ block: () async throws -> T) async rethrows -> T {
    let name = "TrainStep#step_num=\(step),_r=1#"
    if PJRTProfiler.hasTraceMeApi {
        return try await PJRTProfiler.traced(name, level: 1, block)
    } else {
        return try await trainStep(step, block)
    }
}
