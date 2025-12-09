/// JupyterShardingBenchmark.swift - Performance benchmarks for SDY Sharding (SwiftIRJupyter Module)
///
/// Benchmarks mesh creation, sharding annotation generation, MLIR generation,
/// and compares sharded vs non-sharded tracing overhead for the Jupyter API.

import SwiftIRJupyter
import Foundation

// MARK: - Benchmark Utilities

/// Simple benchmark timer
struct JBenchmarkTimer {
    let name: String
    let iterations: Int
    var times: [Double] = []

    init(name: String, iterations: Int = 1000) {
        self.name = name
        self.iterations = iterations
    }

    mutating func measure(_ block: () -> Void) {
        // Warmup
        for _ in 0..<Swift.min(10, iterations / 10) {
            block()
        }

        // Actual measurements
        times.removeAll()
        for _ in 0..<iterations {
            let start = DispatchTime.now()
            block()
            let end = DispatchTime.now()
            let nanos = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
            times.append(nanos / 1_000_000) // Convert to milliseconds
        }
    }

    var mean: Double {
        times.reduce(0, +) / Double(times.count)
    }

    var median: Double {
        let sorted = times.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[mid-1] + sorted[mid]) / 2 : sorted[mid]
    }

    var min: Double {
        times.min() ?? 0
    }

    var max: Double {
        times.max() ?? 0
    }

    var stddev: Double {
        let m = mean
        let variance = times.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(times.count)
        return variance.squareRoot()
    }

    func report() {
        print("  \(name):")
        print("    Iterations: \(iterations)")
        print("    Mean:   \(String(format: "%.4f", mean)) ms")
        print("    Median: \(String(format: "%.4f", median)) ms")
        print("    Min:    \(String(format: "%.4f", min)) ms")
        print("    Max:    \(String(format: "%.4f", max)) ms")
        print("    Stddev: \(String(format: "%.4f", stddev)) ms")
        print("    Throughput: \(String(format: "%.0f", 1000.0 / mean)) ops/sec")
    }
}

// MARK: - Benchmark 1: Mesh Creation

func benchmarkJupyterMeshCreation() {
    print("\n=== Benchmark 1: Jupyter Mesh Creation ===\n")

    // Linear mesh
    var linearTimer = JBenchmarkTimer(name: "Linear Mesh (1D)", iterations: 10000)
    linearTimer.measure {
        _ = JSDYMesh.linear(name: "dp", axis: "batch", size: 8)
    }
    linearTimer.report()

    // Grid mesh
    var gridTimer = JBenchmarkTimer(name: "Grid Mesh (2D)", iterations: 10000)
    gridTimer.measure {
        _ = JSDYMesh.grid(name: "hybrid", dataParallel: 4, modelParallel: 8)
    }
    gridTimer.report()

    // TPU mesh
    var tpuTimer = JBenchmarkTimer(name: "TPU Mesh (2D)", iterations: 10000)
    tpuTimer.measure {
        _ = JSDYMesh.tpuMesh(name: "tpu", x: 4, y: 8)
    }
    tpuTimer.report()

    // JDeviceMesh (legacy API)
    var deviceMeshTimer = JBenchmarkTimer(name: "JDeviceMesh (legacy)", iterations: 10000)
    deviceMeshTimer.measure {
        _ = JDeviceMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)
    }
    deviceMeshTimer.report()
}

// MARK: - Benchmark 2: Sharding Annotation Generation

func benchmarkJupyterShardingAnnotation() {
    print("\n=== Benchmark 2: Jupyter Sharding Annotation Generation ===\n")

    let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)

    // Replicated sharding
    var replicatedTimer = JBenchmarkTimer(name: "Replicated Sharding", iterations: 10000)
    replicatedTimer.measure {
        _ = JSDYSharding.replicated(mesh: mesh, rank: 4)
    }
    replicatedTimer.report()

    // Data parallel sharding
    var dpTimer = JBenchmarkTimer(name: "Data Parallel Sharding", iterations: 10000)
    dpTimer.measure {
        _ = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "data")
    }
    dpTimer.report()

    // Model parallel sharding
    var mpTimer = JBenchmarkTimer(name: "Model Parallel Sharding", iterations: 10000)
    mpTimer.measure {
        _ = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
    }
    mpTimer.report()

    // MLIR attribute text generation
    let sharding = JSDYSharding(mesh: mesh, axes: ["data", nil, "model", nil])
    var mlirTimer = JBenchmarkTimer(name: "MLIR Attribute Text", iterations: 10000)
    mlirTimer.measure {
        _ = sharding.mlirAttributeText
    }
    mlirTimer.report()

    // Legacy JShardingSpec
    let deviceMesh = JDeviceMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)
    var legacyTimer = JBenchmarkTimer(name: "JShardingSpec (legacy)", iterations: 10000)
    legacyTimer.measure {
        _ = JShardingSpec.dataParallel(mesh: deviceMesh, rank: 2)
    }
    legacyTimer.report()
}

// MARK: - Benchmark 3: MLIR Generation with Sharding

func benchmarkJupyterMLIRGeneration() {
    print("\n=== Benchmark 3: Jupyter MLIR Generation with Sharding ===\n")

    // Small model: simple matmul
    var smallTimer = JBenchmarkTimer(name: "Small Model (1 matmul)", iterations: 1000)
    smallTimer.measure {
        let mesh = JSDYMesh.linear(name: "dp", axis: "batch", size: 4)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let wSharding = JSDYSharding.replicated(mesh: mesh, rank: 2)

        let x = ctx.input(shape: JTensorShape([16, 64]), dtype: .float32, sharding: xSharding)
        let w = ctx.input(shape: JTensorShape([64, 32]), dtype: .float32, sharding: wSharding)

        let y = x.matmul(w)
        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "small")
    }
    smallTimer.report()

    // Medium model: MLP with 3 layers
    var mediumTimer = JBenchmarkTimer(name: "Medium Model (3-layer MLP)", iterations: 500)
    mediumTimer.measure {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: JTensorShape([256, 256]), dtype: .float32, sharding: wSharding)
        let w3 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h1 = x.matmul(w1).relu()
        let h2 = h1.matmul(w2).relu()
        let y = h2.matmul(w3)

        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "mlp")
    }
    mediumTimer.report()

    // Large model: deeper network with more operations
    var largeTimer = JBenchmarkTimer(name: "Large Model (6-layer + ops)", iterations: 200)
    largeTimer.measure {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 8)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: JTensorShape([64, 512]), dtype: .float32, sharding: xSharding)
        var h = x

        // 6 layers
        for i in 0..<6 {
            let w = ctx.input(shape: JTensorShape([512, 512]), dtype: .float32, sharding: wSharding)
            h = h.matmul(w)
            if i < 5 {
                h = h.relu()
            }
        }

        ctx.output(h, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "deep")
    }
    largeTimer.report()
}

// MARK: - Benchmark 4: Sharded vs Non-Sharded Tracing Overhead

func benchmarkJupyterShardingOverhead() {
    print("\n=== Benchmark 4: Jupyter Sharded vs Non-Sharded Tracing Overhead ===\n")

    // Non-sharded tracing (using regular JTracingContext)
    var nonShardedTimer = JBenchmarkTimer(name: "Non-Sharded Tracing", iterations: 1000)
    nonShardedTimer.measure {
        let ctx = JTracingContext()

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32)
        let w2 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32)

        let h = x.matmul(w1).relu()
        let _ = h.matmul(w2)
    }
    nonShardedTimer.report()

    // Sharded tracing (using JShardedTracingContext)
    var shardedTimer = JBenchmarkTimer(name: "Sharded Tracing", iterations: 1000)
    shardedTimer.measure {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h = x.matmul(w1).relu()
        let _ = h.matmul(w2)
    }
    shardedTimer.report()

    // Calculate overhead
    let overhead = (shardedTimer.mean - nonShardedTimer.mean) / nonShardedTimer.mean * 100
    print("\n  Sharding Overhead: \(String(format: "%.1f", overhead))%")

    // Full pipeline comparison (tracing + MLIR generation)
    print("\n  --- Full Pipeline (Tracing + MLIR Generation) ---\n")

    var nonShardedFullTimer = JBenchmarkTimer(name: "Non-Sharded Full Pipeline", iterations: 500)
    nonShardedFullTimer.measure {
        let ctx = JTracingContext()

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32)
        let w2 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32)

        let h = x.matmul(w1).relu()
        let y = h.matmul(w2)

        ctx.output(y)
        _ = ctx.buildModule(name: "test")
    }
    nonShardedFullTimer.report()

    var shardedFullTimer = JBenchmarkTimer(name: "Sharded Full Pipeline", iterations: 500)
    shardedFullTimer.measure {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: JTensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: JTensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h = x.matmul(w1).relu()
        let y = h.matmul(w2)

        ctx.output(y, sharding: xSharding)
        _ = ctx.buildShardedModule(name: "test")
    }
    shardedFullTimer.report()

    let fullOverhead = (shardedFullTimer.mean - nonShardedFullTimer.mean) / nonShardedFullTimer.mean * 100
    print("\n  Full Pipeline Overhead: \(String(format: "%.1f", fullOverhead))%")
}

// MARK: - Benchmark 5: Scaling with Model Size

func benchmarkJupyterScaling() {
    print("\n=== Benchmark 5: Jupyter Scaling with Model Size ===\n")

    let sizes = [1, 2, 4, 8, 16, 32]

    print("  Layers | Mean (ms) | Throughput (ops/s)")
    print("  -------|-----------|-------------------")

    for numLayers in sizes {
        var timer = JBenchmarkTimer(name: "\(numLayers) layers", iterations: 100)
        timer.measure {
            let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
            let ctx = JShardedTracingContext(mesh: mesh)

            let xSharding = JSDYSharding(mesh: mesh, axes: ["data", nil])
            let wSharding = JSDYSharding(mesh: mesh, axes: [nil, "model"])

            let x = ctx.input(shape: JTensorShape([32, 256]), dtype: .float32, sharding: xSharding)
            var h = x

            for i in 0..<numLayers {
                let w = ctx.input(shape: JTensorShape([256, 256]), dtype: .float32, sharding: wSharding)
                h = h.matmul(w)
                if i < numLayers - 1 {
                    h = h.relu()
                }
            }

            ctx.output(h, sharding: xSharding)
            _ = ctx.buildShardedModule(name: "scale_test")
        }

        let throughput = 1000.0 / timer.mean
        print("  \(String(format: "%6d", numLayers)) | \(String(format: "%9.4f", timer.mean)) | \(String(format: "%17.0f", throughput))")
    }
}

// MARK: - Benchmark 6: API Comparison (Jupyter vs Legacy)

func benchmarkJupyterAPIComparison() {
    print("\n=== Benchmark 6: Jupyter API Comparison (SDY vs Legacy) ===\n")

    // New SDY API
    var sdyTimer = JBenchmarkTimer(name: "JSDYMesh + JSDYSharding API", iterations: 1000)
    sdyTimer.measure {
        let mesh = JSDYMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)
        let ctx = JShardedTracingContext(mesh: mesh)

        let xSharding = JSDYSharding.dataParallel(mesh: mesh, rank: 2)
        let wSharding = JSDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1)

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w = ctx.input(shape: JTensorShape([128, 64]), dtype: .float32, sharding: wSharding)

        let y = x.matmul(w)
        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "sdy_test")
    }
    sdyTimer.report()

    // Legacy JDeviceMesh + JShardingSpec API (without MLIR generation)
    var legacyTimer = JBenchmarkTimer(name: "JDeviceMesh + JShardingSpec API", iterations: 1000)
    legacyTimer.measure {
        let mesh = JDeviceMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)
        let _ = JShardingSpec.dataParallel(mesh: mesh, rank: 2)
        // Note: modelParallel uses dataParallel with different axis in Jupyter API
        let _ = JShardingSpec.dataParallel(mesh: mesh, rank: 2)

        // Convert to SDY for MLIR generation
        let sdyMesh = mesh.toSDYMesh()
        let ctx = JShardedTracingContext(mesh: sdyMesh)

        let xSharding = JSDYSharding.dataParallel(mesh: sdyMesh, rank: 2)
        let wSharding = JSDYSharding.modelParallel(mesh: sdyMesh, rank: 2, featureDim: 1)

        let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w = ctx.input(shape: JTensorShape([128, 64]), dtype: .float32, sharding: wSharding)

        let y = x.matmul(w)
        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "legacy_test")
    }
    legacyTimer.report()

    let diff = legacyTimer.mean - sdyTimer.mean
    let diffPercent = diff / sdyTimer.mean * 100
    print("\n  SDY API is \(String(format: "%.1f", abs(diffPercent)))% \(diff > 0 ? "faster" : "slower") than Legacy+SDY bridge")
}

// MARK: - Benchmark 7: Mesh Topology Comparison

func benchmarkJupyterMeshTopology() {
    print("\n=== Benchmark 7: Jupyter Mesh Topology Comparison ===\n")

    // Same total devices (16), different topologies
    let topologies: [(String, JSDYMesh)] = [
        ("Linear 16x1", JSDYMesh.linear(name: "linear", axis: "x", size: 16)),
        ("Grid 4x4", JSDYMesh.grid(name: "grid", dataParallel: 4, modelParallel: 4)),
        ("Grid 2x8", JSDYMesh.grid(name: "grid", dataParallel: 2, modelParallel: 8)),
        ("TPU 4x4", JSDYMesh.tpuMesh(name: "tpu", x: 4, y: 4)),
    ]

    print("  Topology    | Mesh (ms) | Sharding (ms) | MLIR (ms) | Total (ms)")
    print("  ------------|-----------|---------------|-----------|----------")

    for (name, mesh) in topologies {
        // Mesh creation time
        var meshTimer = JBenchmarkTimer(name: "mesh", iterations: 1000)
        meshTimer.measure {
            _ = mesh.mlirText
        }

        // Sharding creation time
        var shardingTimer = JBenchmarkTimer(name: "sharding", iterations: 1000)
        let axes: [String?] = mesh.axes.count == 1 ? [mesh.axes[0].name, nil] :
                              [mesh.axes[0].name, mesh.axes[1].name]
        shardingTimer.measure {
            _ = JSDYSharding(mesh: mesh, axes: axes)
        }

        // Full MLIR generation
        var mlirTimer = JBenchmarkTimer(name: "mlir", iterations: 500)
        mlirTimer.measure {
            let ctx = JShardedTracingContext(mesh: mesh)
            let sharding = JSDYSharding(mesh: mesh, axes: axes)

            let x = ctx.input(shape: JTensorShape([32, 128]), dtype: .float32, sharding: sharding)
            let w = ctx.input(shape: JTensorShape([128, 64]), dtype: .float32, sharding: sharding)
            let y = x.matmul(w)
            ctx.output(y, sharding: sharding)

            _ = ctx.buildShardedModule(name: "test")
        }

        let total = meshTimer.mean + shardingTimer.mean + mlirTimer.mean
        print("  \(name.padding(toLength: 11, withPad: " ", startingAt: 0)) | \(String(format: "%9.4f", meshTimer.mean)) | \(String(format: "%13.4f", shardingTimer.mean)) | \(String(format: "%9.4f", mlirTimer.mean)) | \(String(format: "%9.4f", total))")
    }
}

// MARK: - Main

print("╔══════════════════════════════════════════════════════════════╗")
print("║     SwiftIRJupyter SDY Sharding Performance Benchmarks       ║")
print("║     (Jupyter/Colab API)                                      ║")
print("╚══════════════════════════════════════════════════════════════╝")

let totalStart = DispatchTime.now()

benchmarkJupyterMeshCreation()
benchmarkJupyterShardingAnnotation()
benchmarkJupyterMLIRGeneration()
benchmarkJupyterShardingOverhead()
benchmarkJupyterScaling()
benchmarkJupyterAPIComparison()
benchmarkJupyterMeshTopology()

let totalEnd = DispatchTime.now()
let totalSeconds = Double(totalEnd.uptimeNanoseconds - totalStart.uptimeNanoseconds) / 1_000_000_000

print("\n" + String(repeating: "=", count: 60))
print("Total benchmark time: \(String(format: "%.2f", totalSeconds)) seconds")
print(String(repeating: "=", count: 60))
