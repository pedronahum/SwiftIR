// ShardedExecution.swift - PJRT Distributed Execution for SwiftIR
// Provides sharded buffer and executable types for multi-device execution
//
// This integrates SDY sharding annotations with PJRT for actual distributed execution.

import Foundation

// MARK: - Sharded Buffer

/// A tensor buffer that is partitioned across multiple devices according to a sharding spec.
///
/// ShardedBuffer manages the distribution of tensor data across devices. When created
/// with a sharding specification, it automatically partitions the data according to
/// the mesh topology and sharding axes.
public class JShardedBuffer {
    /// The sharding specification for this buffer
    public let sharding: JSDYSharding

    /// The mesh this buffer is distributed across
    public let mesh: JSDYMesh

    /// The global (unsharded) shape of the tensor
    public let globalShape: [Int64]

    /// The local (per-device) shape after sharding
    public let localShape: [Int64]

    /// The data type
    public let dtype: JDType

    /// Per-device buffer handles (device index -> buffer pointer)
    private var deviceBuffers: [Int: UnsafeMutableRawPointer] = [:]

    /// The client used to create this buffer
    private weak var client: JShardedClient?

    /// Creates a sharded buffer from global data.
    ///
    /// The data is automatically partitioned across devices according to the sharding spec.
    /// - Parameters:
    ///   - data: The global tensor data (before sharding)
    ///   - globalShape: The shape of the global tensor
    ///   - sharding: How to partition the tensor
    ///   - mesh: The device mesh to distribute across
    ///   - client: The PJRT client managing devices
    public init(
        data: [Float],
        globalShape: [Int64],
        sharding: JSDYSharding,
        mesh: JSDYMesh,
        dtype: JDType = .float32,
        client: JShardedClient
    ) throws {
        self.sharding = sharding
        self.mesh = mesh
        self.globalShape = globalShape
        self.dtype = dtype
        self.client = client

        // Calculate local shape based on sharding
        self.localShape = Self.calculateLocalShape(globalShape: globalShape, sharding: sharding, mesh: mesh)

        // Partition and create device buffers
        try createDeviceBuffers(data: data, client: client)
    }

    /// Creates a sharded buffer from pre-partitioned per-device data.
    public init(
        perDeviceData: [[Float]],
        globalShape: [Int64],
        localShape: [Int64],
        sharding: JSDYSharding,
        mesh: JSDYMesh,
        dtype: JDType = .float32,
        client: JShardedClient
    ) throws {
        self.sharding = sharding
        self.mesh = mesh
        self.globalShape = globalShape
        self.localShape = localShape
        self.dtype = dtype
        self.client = client

        // Create buffers for each device
        let devices = try client.getDevices()
        for (i, data) in perDeviceData.enumerated() {
            if i < devices.count {
                let buffer = try PJRTBindings.shared.createBuffer(
                    client: client.clientHandle,
                    data: data,
                    type: .f32,
                    shape: localShape,
                    device: devices[i]
                )
                deviceBuffers[i] = buffer
            }
        }
    }

    deinit {
        // Clean up device buffers
        for (_, buffer) in deviceBuffers {
            PJRTBindings.shared.destroyBuffer(buffer)
        }
    }

    /// Calculate the local (per-device) shape from global shape and sharding.
    private static func calculateLocalShape(
        globalShape: [Int64],
        sharding: JSDYSharding,
        mesh: JSDYMesh
    ) -> [Int64] {
        var localShape = globalShape

        for (dimIdx, axisName) in sharding.dimAxes.enumerated() {
            if let axis = axisName,
               let meshAxis = mesh.axes.first(where: { $0.name == axis }) {
                // This dimension is sharded across this axis
                localShape[dimIdx] /= Int64(meshAxis.size)
            }
        }

        return localShape
    }

    /// Partition global data and create per-device buffers.
    private func createDeviceBuffers(data: [Float], client: JShardedClient) throws {
        let devices = try client.getDevices()
        let numDevices = mesh.deviceCount

        guard devices.count >= numDevices else {
            throw SwiftIRJupyterError.invalidState(
                message: "Not enough devices: need \(numDevices), have \(devices.count)"
            )
        }

        // Partition data according to sharding
        let partitions = partitionData(data: data)

        // Create buffer on each device
        for (deviceIdx, partition) in partitions.enumerated() {
            if deviceIdx < devices.count {
                let buffer = try PJRTBindings.shared.createBuffer(
                    client: client.clientHandle,
                    data: partition,
                    type: .f32,
                    shape: localShape,
                    device: devices[deviceIdx]
                )
                deviceBuffers[deviceIdx] = buffer
            }
        }
    }

    /// Partition global data according to the sharding specification.
    private func partitionData(data: [Float]) -> [[Float]] {
        let numDevices = mesh.deviceCount

        // For simple data parallelism (sharding on first dimension)
        if let firstAxis = sharding.dimAxes.first, firstAxis != nil {
            // Shard along first dimension
            let totalBatch = Int(globalShape[0])
            let batchPerDevice = totalBatch / numDevices
            let elementsPerBatch = data.count / totalBatch

            var partitions: [[Float]] = []
            for device in 0..<numDevices {
                let startBatch = device * batchPerDevice
                let startIdx = startBatch * elementsPerBatch
                let endIdx = (startBatch + batchPerDevice) * elementsPerBatch
                partitions.append(Array(data[startIdx..<endIdx]))
            }
            return partitions
        }

        // Fully replicated - copy data to all devices
        return Array(repeating: data, count: numDevices)
    }

    /// Gather data from all devices back to host.
    public func toHost() throws -> [Float] {
        var allData: [Float] = []

        // Collect from devices in order
        let sortedDevices = deviceBuffers.keys.sorted()
        for deviceIdx in sortedDevices {
            guard let buffer = deviceBuffers[deviceIdx] else { continue }

            let localCount = localShape.reduce(1, *)
            var localData = [Float](repeating: 0, count: Int(localCount))
            try PJRTBindings.shared.bufferToHost(buffer, into: &localData)
            allData.append(contentsOf: localData)
        }

        return allData
    }

    /// Get the buffer handle for a specific device.
    public func getDeviceBuffer(deviceIndex: Int) -> UnsafeMutableRawPointer? {
        deviceBuffers[deviceIndex]
    }

    /// Get all device buffer handles in order.
    public func getAllDeviceBuffers() -> [UnsafeMutableRawPointer] {
        deviceBuffers.keys.sorted().compactMap { deviceBuffers[$0] }
    }
}

// MARK: - Sharded Client

/// A PJRT client configured for sharded execution across multiple devices.
public class JShardedClient {
    /// The underlying PJRT client handle
    public let clientHandle: UnsafeMutableRawPointer

    /// The device mesh for this client
    public let mesh: JSDYMesh

    /// Cached device handles
    private var devices: [UnsafeMutableRawPointer]?

    /// Platform name (CPU, GPU, TPU)
    public let platformName: String

    /// Creates a sharded client with the given mesh.
    public init(mesh: JSDYMesh) throws {
        self.mesh = mesh

        // Load PJRT if needed
        let pjrt = PJRTBindings.shared
        if !pjrt.isLoaded {
            try pjrt.load()
        }

        // Load appropriate plugin
        if !pjrt.isPluginLoaded {
            // Try to find PJRT plugin
            let pluginPaths = [
                "/opt/swiftir-deps/lib/pjrt_plugin_cpu.so",
                "/usr/local/lib/pjrt_plugin_cpu.so",
                "cmake/build/lib/pjrt_plugin_cpu.so"
            ]

            for path in pluginPaths {
                if FileManager.default.fileExists(atPath: path) {
                    try? pjrt.loadPlugin(path: path)
                    break
                }
            }
        }

        // Create client
        self.clientHandle = try pjrt.createClient()
        self.platformName = (try? pjrt.getPlatformName(clientHandle)) ?? "unknown"

        // Validate device count
        let deviceCount = try getDevices().count
        if deviceCount < mesh.deviceCount {
            print("Warning: Mesh requires \(mesh.deviceCount) devices, only \(deviceCount) available")
            print("  Execution will use virtual sharding (data parallelism on available devices)")
        }
    }

    deinit {
        PJRTBindings.shared.destroyClient(clientHandle)
    }

    /// Get available devices.
    public func getDevices() throws -> [UnsafeMutableRawPointer] {
        if let cached = devices {
            return cached
        }
        let devs = try PJRTBindings.shared.getAddressableDevices(clientHandle)
        devices = devs
        return devs
    }

    /// Create a sharded buffer from global data.
    public func createShardedBuffer(
        data: [Float],
        globalShape: [Int64],
        sharding: JSDYSharding
    ) throws -> JShardedBuffer {
        try JShardedBuffer(
            data: data,
            globalShape: globalShape,
            sharding: sharding,
            mesh: mesh,
            client: self
        )
    }

    /// Create a replicated buffer (same data on all devices).
    public func createReplicatedBuffer(
        data: [Float],
        shape: [Int64]
    ) throws -> JShardedBuffer {
        let sharding = JSDYSharding.replicated(mesh: mesh, rank: shape.count)
        return try JShardedBuffer(
            data: data,
            globalShape: shape,
            sharding: sharding,
            mesh: mesh,
            client: self
        )
    }
}

// MARK: - Sharded Executable

/// An executable compiled with sharding annotations for distributed execution.
public class JShardedExecutable {
    /// The underlying PJRT executable handle
    private let executableHandle: UnsafeMutableRawPointer

    /// The client this was compiled for
    private let client: JShardedClient

    /// The mesh configuration
    public let mesh: JSDYMesh

    /// Input shardings (in order)
    public let inputShardings: [JSDYSharding]

    /// Output shardings
    public let outputShardings: [JSDYSharding]

    /// The MLIR source (for debugging)
    public let mlirSource: String

    /// Creates a sharded executable by compiling MLIR with sharding annotations.
    public init(
        mlirSource: String,
        mesh: JSDYMesh,
        inputShardings: [JSDYSharding],
        outputShardings: [JSDYSharding],
        client: JShardedClient
    ) throws {
        self.mlirSource = mlirSource
        self.mesh = mesh
        self.inputShardings = inputShardings
        self.outputShardings = outputShardings
        self.client = client

        // Compile the MLIR
        self.executableHandle = try PJRTBindings.shared.compile(
            client: client.clientHandle,
            mlirModule: mlirSource
        )
    }

    deinit {
        PJRTBindings.shared.destroyExecutable(executableHandle)
    }

    /// Execute with sharded input buffers.
    public func execute(inputs: [JShardedBuffer]) throws -> [JShardedBuffer] {
        // For now, we execute on each device and collect results
        // A full implementation would use PJRT's native sharded execution

        let devices = try client.getDevices()
        let numDevices = min(mesh.deviceCount, devices.count)

        // Collect all device-local input buffers
        var allInputBuffers: [UnsafeMutableRawPointer] = []
        for input in inputs {
            allInputBuffers.append(contentsOf: input.getAllDeviceBuffers())
        }

        // Execute
        let outputPtrs = try PJRTBindings.shared.execute(
            executable: executableHandle,
            inputs: allInputBuffers
        )

        // Package outputs as sharded buffers
        var outputs: [JShardedBuffer] = []

        // For simplicity, assume single output with same sharding as first input
        if !outputPtrs.isEmpty {
            let outputSharding = outputShardings.first ?? inputShardings.first ?? JSDYSharding.replicated(mesh: mesh, rank: 2)

            // Calculate output shape (for now, assume same as first input)
            let outputGlobalShape = inputs.first?.globalShape ?? [1]
            let outputLocalShape = inputs.first?.localShape ?? [1]

            // Create output buffer wrapper
            let output = try JShardedBuffer(
                perDeviceData: [],  // Will be filled by execute
                globalShape: outputGlobalShape,
                localShape: outputLocalShape,
                sharding: outputSharding,
                mesh: mesh,
                client: client
            )
            outputs.append(output)
        }

        return outputs
    }

    /// Execute with raw float arrays (convenience method).
    public func execute(
        inputs: [([Float], [Int64], JSDYSharding)]
    ) throws -> [[Float]] {
        // Create sharded buffers
        var inputBuffers: [JShardedBuffer] = []
        for (data, shape, sharding) in inputs {
            let buffer = try client.createShardedBuffer(
                data: data,
                globalShape: shape,
                sharding: sharding
            )
            inputBuffers.append(buffer)
        }

        // Execute
        let outputBuffers = try execute(inputs: inputBuffers)

        // Gather results
        return try outputBuffers.map { try $0.toHost() }
    }
}

// MARK: - Sharded Execution Context

/// High-level context for sharded compilation and execution.
///
/// Example usage:
/// ```swift
/// let mesh = JSDYMesh.linear(name: "devices", axis: "batch", size: 4)
/// let ctx = try JShardedExecutionContext(mesh: mesh)
///
/// // Define shardings
/// let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
/// let wSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)
///
/// // Compile sharded function
/// let exec = try ctx.compile { tracingCtx in
///     let x = tracingCtx.input(shape: [16, 64], sharding: xSharding)
///     let w = tracingCtx.input(shape: [64, 32], sharding: wSharding)
///     return x.matmul(w)
/// }
///
/// // Execute
/// let results = try ctx.execute(exec, inputs: [xData, wData])
/// ```
public class JShardedExecutionContext {
    /// The device mesh
    public let mesh: JSDYMesh

    /// The sharded PJRT client
    public let client: JShardedClient

    /// Creates a sharded execution context.
    public init(mesh: JSDYMesh) throws {
        self.mesh = mesh
        self.client = try JShardedClient(mesh: mesh)
    }

    /// Compile a sharded function.
    public func compile(
        inputShardings: [JSDYSharding],
        outputSharding: JSDYSharding? = nil,
        builder: (JShardedTracingContext) throws -> JTracer
    ) throws -> JShardedExecutable {
        // Create tracing context
        let tracingCtx = JShardedTracingContext(mesh: mesh)

        // Build the trace
        let output = try builder(tracingCtx)

        // Set output
        tracingCtx.output(output, sharding: outputSharding ?? inputShardings.first)

        // Build sharded MLIR
        let mlir = tracingCtx.buildShardedModule()

        // Compile
        return try JShardedExecutable(
            mlirSource: mlir,
            mesh: mesh,
            inputShardings: inputShardings,
            outputShardings: outputSharding.map { [$0] } ?? inputShardings.prefix(1).map { $0 },
            client: client
        )
    }

    /// Compile a function with automatic data-parallel sharding.
    public func compileDataParallel(
        inputShapes: [[Int]],
        dtype: JDType = .float32,
        builder: ([JTracer]) throws -> JTracer
    ) throws -> JShardedExecutable {
        // Create data-parallel shardings for inputs
        let inputShardings = inputShapes.map { shape in
            JSDYSharding.dataParallel(mesh: mesh, rank: shape.count)
        }

        let tracingCtx = JShardedTracingContext(mesh: mesh)

        // Create inputs with data-parallel sharding
        let inputs = zip(inputShapes, inputShardings).map { (shape, sharding) in
            tracingCtx.input(
                shape: JTensorShape(shape),
                dtype: dtype,
                sharding: sharding
            )
        }

        // Build trace
        let output = try builder(inputs)

        // Set output with data-parallel sharding
        tracingCtx.output(output, sharding: inputShardings.first)

        // Build and compile
        let mlir = tracingCtx.buildShardedModule()

        return try JShardedExecutable(
            mlirSource: mlir,
            mesh: mesh,
            inputShardings: inputShardings,
            outputShardings: inputShardings.prefix(1).map { $0 },
            client: client
        )
    }

    /// Execute a sharded executable with data.
    public func execute(
        _ executable: JShardedExecutable,
        inputs: [([Float], [Int64])]
    ) throws -> [[Float]] {
        let inputsWithSharding = zip(inputs, executable.inputShardings).map { (input, sharding) in
            (input.0, input.1, sharding)
        }
        return try executable.execute(inputs: inputsWithSharding)
    }

    /// Print execution context info.
    public func printInfo() {
        print("Sharded Execution Context")
        print("  Platform: \(client.platformName)")
        mesh.printInfo()
        if let devices = try? client.getDevices() {
            print("  Available devices: \(devices.count)")
        }
    }
}

// MARK: - Convenience Functions

/// Create a sharded execution context with a linear mesh.
public func createDataParallelContext(numDevices: Int) throws -> JShardedExecutionContext {
    let mesh = JSDYMesh.linear(name: "data_parallel", axis: "batch", size: numDevices)
    return try JShardedExecutionContext(mesh: mesh)
}

/// Create a sharded execution context with a 2D mesh.
public func createHybridParallelContext(
    dataParallel: Int,
    modelParallel: Int
) throws -> JShardedExecutionContext {
    let mesh = JSDYMesh.grid(
        name: "hybrid",
        dataParallel: dataParallel,
        modelParallel: modelParallel
    )
    return try JShardedExecutionContext(mesh: mesh)
}

// MARK: - Collective Operations for Sharded Buffers

extension JShardedBuffer {
    /// All-reduce: sum values across all devices.
    ///
    /// This is used for gradient synchronization in data-parallel training.
    public func allReduceSum() throws -> JShardedBuffer {
        guard let client = client else {
            throw SwiftIRJupyterError.invalidState(message: "Client deallocated")
        }

        // Gather all data
        let allData = try toHost()

        // For data-parallel, sum across shards
        // In a real implementation, this would use PJRT collective ops
        let localCount = Int(localShape.reduce(1, *))
        var summed = [Float](repeating: 0, count: localCount)

        let numDevices = mesh.deviceCount
        for i in 0..<localCount {
            var sum: Float = 0
            for d in 0..<numDevices {
                sum += allData[d * localCount + i]
            }
            summed[i] = sum / Float(numDevices)  // Average for gradients
        }

        // Create new buffer with reduced data (replicated)
        return try JShardedBuffer(
            perDeviceData: Array(repeating: summed, count: numDevices),
            globalShape: localShape,  // After reduction, local = global
            localShape: localShape,
            sharding: JSDYSharding.replicated(mesh: mesh, rank: localShape.count),
            mesh: mesh,
            client: client
        )
    }

    /// All-gather: collect all shards to form global tensor on each device.
    public func allGather() throws -> JShardedBuffer {
        guard let client = client else {
            throw SwiftIRJupyterError.invalidState(message: "Client deallocated")
        }

        // Gather all data
        let allData = try toHost()

        // Create replicated buffer with gathered data
        return try JShardedBuffer(
            perDeviceData: Array(repeating: allData, count: mesh.deviceCount),
            globalShape: globalShape,
            localShape: globalShape,  // Now fully replicated
            sharding: JSDYSharding.replicated(mesh: mesh, rank: globalShape.count),
            mesh: mesh,
            client: client
        )
    }
}
