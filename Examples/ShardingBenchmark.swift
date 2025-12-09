/// ShardingBenchmark.swift - Performance benchmarks for SDY Sharding (Main SwiftIR Module)
///
/// Benchmarks mesh creation, sharding annotation generation, MLIR generation,
/// and compares sharded vs non-sharded tracing overhead.

import SwiftIR
import Foundation

// MARK: - Benchmark Utilities

/// Simple benchmark timer
struct BenchmarkTimer {
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

func benchmarkMeshCreation() {
    print("\n=== Benchmark 1: Mesh Creation ===\n")

    // Linear mesh
    var linearTimer = BenchmarkTimer(name: "Linear Mesh (1D)", iterations: 10000)
    linearTimer.measure {
        _ = SDYMesh.linear(name: "dp", axis: "batch", size: 8)
    }
    linearTimer.report()

    // Grid mesh
    var gridTimer = BenchmarkTimer(name: "Grid Mesh (2D)", iterations: 10000)
    gridTimer.measure {
        _ = SDYMesh.grid(name: "hybrid", dataParallel: 4, modelParallel: 8)
    }
    gridTimer.report()

    // TPU mesh
    var tpuTimer = BenchmarkTimer(name: "TPU Mesh (2D)", iterations: 10000)
    tpuTimer.measure {
        _ = SDYMesh.tpuMesh(name: "tpu", x: 4, y: 8)
    }
    tpuTimer.report()

    // Cube mesh
    var cubeTimer = BenchmarkTimer(name: "Cube Mesh (3D)", iterations: 10000)
    cubeTimer.measure {
        _ = SDYMesh.cube(name: "3d", data: 2, tensor: 4, pipeline: 2)
    }
    cubeTimer.report()
}

// MARK: - Benchmark 2: Sharding Annotation Generation

func benchmarkShardingAnnotation() {
    print("\n=== Benchmark 2: Sharding Annotation Generation ===\n")

    let mesh = SDYMesh.grid(name: "mesh", dataParallel: 4, modelParallel: 8)

    // Replicated sharding
    var replicatedTimer = BenchmarkTimer(name: "Replicated Sharding", iterations: 10000)
    replicatedTimer.measure {
        _ = SDYSharding.replicated(mesh: mesh, rank: 4)
    }
    replicatedTimer.report()

    // Data parallel sharding
    var dpTimer = BenchmarkTimer(name: "Data Parallel Sharding", iterations: 10000)
    dpTimer.measure {
        _ = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "data")
    }
    dpTimer.report()

    // Model parallel sharding
    var mpTimer = BenchmarkTimer(name: "Model Parallel Sharding", iterations: 10000)
    mpTimer.measure {
        _ = SDYSharding.modelParallel(mesh: mesh, rank: 2, featureDim: 1, modelAxis: "model")
    }
    mpTimer.report()

    // Column parallel sharding
    var colTimer = BenchmarkTimer(name: "Column Parallel Sharding", iterations: 10000)
    colTimer.measure {
        _ = SDYSharding.columnParallel(mesh: mesh, rank: 2, axis: "model")
    }
    colTimer.report()

    // MLIR attribute text generation
    let sharding = SDYSharding(mesh: mesh, axes: ["data", nil, "model", nil])
    var mlirTimer = BenchmarkTimer(name: "MLIR Attribute Text", iterations: 10000)
    mlirTimer.measure {
        _ = sharding.mlirAttributeText
    }
    mlirTimer.report()
}

// MARK: - Benchmark 3: MLIR Generation with Sharding

func benchmarkMLIRGeneration() {
    print("\n=== Benchmark 3: MLIR Generation with Sharding ===\n")

    // Small model: simple matmul
    var smallTimer = BenchmarkTimer(name: "Small Model (1 matmul)", iterations: 1000)
    smallTimer.measure {
        let mesh = SDYMesh.linear(name: "dp", axis: "batch", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding.dataParallel(mesh: mesh, rank: 2, batchAxis: "batch")
        let wSharding = SDYSharding.replicated(mesh: mesh, rank: 2)

        let x = ctx.input(shape: TensorShape([16, 64]), dtype: .float32, sharding: xSharding)
        let w = ctx.input(shape: TensorShape([64, 32]), dtype: .float32, sharding: wSharding)

        let y = x.tracer.matmul(w.tracer)
        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "small")
    }
    smallTimer.report()

    // Medium model: MLP with 3 layers
    var mediumTimer = BenchmarkTimer(name: "Medium Model (3-layer MLP)", iterations: 500)
    mediumTimer.measure {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: TensorShape([256, 256]), dtype: .float32, sharding: wSharding)
        let w3 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h1 = x.tracer.matmul(w1.tracer).relu()
        let h2 = h1.matmul(w2.tracer).relu()
        let y = h2.matmul(w3.tracer)

        ctx.output(y, sharding: xSharding)

        _ = ctx.buildShardedModule(name: "mlp")
    }
    mediumTimer.report()

    // Large model: deeper network with more operations
    var largeTimer = BenchmarkTimer(name: "Large Model (6-layer + ops)", iterations: 200)
    largeTimer.measure {
        let mesh = SDYMesh.cube(name: "mesh", data: 2, tensor: 4, pipeline: 2)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = SDYSharding(mesh: mesh, axes: [nil, "tensor"])

        let x = ctx.input(shape: TensorShape([64, 512]), dtype: .float32, sharding: xSharding)
        var h = x.tracer

        // 6 layers
        for i in 0..<6 {
            let w = ctx.input(shape: TensorShape([512, 512]), dtype: .float32, sharding: wSharding)
            h = h.matmul(w.tracer)
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

func benchmarkShardingOverhead() {
    print("\n=== Benchmark 4: Sharded vs Non-Sharded Tracing Overhead ===\n")

    // Minimal sharding: ShardedTracingContext with no explicit shardings
    var minimalTimer = BenchmarkTimer(name: "Minimal Sharding (no annotations)", iterations: 1000)
    minimalTimer.measure {
        let mesh = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32)
        let w2 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32)

        let h = x.tracer.matmul(w1.tracer).relu()
        let _ = h.matmul(w2.tracer)
    }
    minimalTimer.report()

    // Sharded tracing (using ShardedTracingContext with full annotations)
    var shardedTimer = BenchmarkTimer(name: "Full Sharding (with annotations)", iterations: 1000)
    shardedTimer.measure {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h = x.tracer.matmul(w1.tracer).relu()
        let _ = h.matmul(w2.tracer)
    }
    shardedTimer.report()

    // Calculate overhead
    let overhead = (shardedTimer.mean - minimalTimer.mean) / minimalTimer.mean * 100
    print("\n  Annotation Overhead: \(String(format: "%.1f", overhead))%")

    // Full pipeline comparison (tracing + MLIR generation)
    print("\n  --- Full Pipeline (Tracing + MLIR Generation) ---\n")

    var minimalFullTimer = BenchmarkTimer(name: "Minimal Full Pipeline", iterations: 500)
    minimalFullTimer.measure {
        let mesh = SDYMesh.linear(name: "mesh", axis: "x", size: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32)
        let w2 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32)

        let h = x.tracer.matmul(w1.tracer).relu()
        let y = h.matmul(w2.tracer)

        ctx.output(y)
        _ = ctx.buildShardedModule(name: "test")
    }
    minimalFullTimer.report()

    var shardedFullTimer = BenchmarkTimer(name: "Sharded Full Pipeline", iterations: 500)
    shardedFullTimer.measure {
        let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
        let ctx = ShardedTracingContext(mesh: mesh)

        let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
        let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])

        let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: xSharding)
        let w1 = ctx.input(shape: TensorShape([128, 256]), dtype: .float32, sharding: wSharding)
        let w2 = ctx.input(shape: TensorShape([256, 64]), dtype: .float32, sharding: wSharding)

        let h = x.tracer.matmul(w1.tracer).relu()
        let y = h.matmul(w2.tracer)

        ctx.output(y, sharding: xSharding)
        _ = ctx.buildShardedModule(name: "test")
    }
    shardedFullTimer.report()

    let fullOverhead = (shardedFullTimer.mean - minimalFullTimer.mean) / minimalFullTimer.mean * 100
    print("\n  Full Pipeline Overhead: \(String(format: "%.1f", fullOverhead))%")
}

// MARK: - Benchmark 5: Scaling with Model Size

func benchmarkScaling() {
    print("\n=== Benchmark 5: Scaling with Model Size ===\n")

    let sizes = [1, 2, 4, 8, 16, 32]

    print("  Layers | Mean (ms) | Throughput (ops/s)")
    print("  -------|-----------|-------------------")

    for numLayers in sizes {
        var timer = BenchmarkTimer(name: "\(numLayers) layers", iterations: 100)
        timer.measure {
            let mesh = SDYMesh.grid(name: "mesh", dataParallel: 2, modelParallel: 4)
            let ctx = ShardedTracingContext(mesh: mesh)

            let xSharding = SDYSharding(mesh: mesh, axes: ["data", nil])
            let wSharding = SDYSharding(mesh: mesh, axes: [nil, "model"])

            let x = ctx.input(shape: TensorShape([32, 256]), dtype: .float32, sharding: xSharding)
            var h = x.tracer

            for i in 0..<numLayers {
                let w = ctx.input(shape: TensorShape([256, 256]), dtype: .float32, sharding: wSharding)
                h = h.matmul(w.tracer)
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

// MARK: - Benchmark 6: Mesh Topology Comparison

func benchmarkMeshTopology() {
    print("\n=== Benchmark 6: Mesh Topology Comparison ===\n")

    // Same total devices (16), different topologies
    let topologies: [(String, SDYMesh)] = [
        ("Linear 16x1", SDYMesh.linear(name: "linear", axis: "x", size: 16)),
        ("Grid 4x4", SDYMesh.grid(name: "grid", dataParallel: 4, modelParallel: 4)),
        ("Grid 2x8", SDYMesh.grid(name: "grid", dataParallel: 2, modelParallel: 8)),
        ("Cube 2x2x4", SDYMesh.cube(name: "cube", data: 2, tensor: 2, pipeline: 4)),
        ("Cube 2x4x2", SDYMesh.cube(name: "cube", data: 2, tensor: 4, pipeline: 2)),
    ]

    print("  Topology    | Mesh (ms) | Sharding (ms) | MLIR (ms) | Total (ms)")
    print("  ------------|-----------|---------------|-----------|----------")

    for (name, mesh) in topologies {
        // Mesh creation time
        var meshTimer = BenchmarkTimer(name: "mesh", iterations: 1000)
        meshTimer.measure {
            _ = mesh.mlirText
        }

        // Sharding creation time
        var shardingTimer = BenchmarkTimer(name: "sharding", iterations: 1000)
        let axes: [String?] = mesh.axes.count == 1 ? [mesh.axes[0].name, nil] :
                              mesh.axes.count == 2 ? [mesh.axes[0].name, mesh.axes[1].name] :
                              [mesh.axes[0].name, mesh.axes[1].name]
        shardingTimer.measure {
            _ = SDYSharding(mesh: mesh, axes: axes)
        }

        // Full MLIR generation
        var mlirTimer = BenchmarkTimer(name: "mlir", iterations: 500)
        mlirTimer.measure {
            let ctx = ShardedTracingContext(mesh: mesh)
            let sharding = SDYSharding(mesh: mesh, axes: axes)

            let x = ctx.input(shape: TensorShape([32, 128]), dtype: .float32, sharding: sharding)
            let w = ctx.input(shape: TensorShape([128, 64]), dtype: .float32, sharding: sharding)
            let y = x.tracer.matmul(w.tracer)
            ctx.output(y, sharding: sharding)

            _ = ctx.buildShardedModule(name: "test")
        }

        let total = meshTimer.mean + shardingTimer.mean + mlirTimer.mean
        print("  \(name.padding(toLength: 11, withPad: " ", startingAt: 0)) | \(String(format: "%9.4f", meshTimer.mean)) | \(String(format: "%13.4f", shardingTimer.mean)) | \(String(format: "%9.4f", mlirTimer.mean)) | \(String(format: "%9.4f", total))")
    }
}

// MARK: - Main

print("╔══════════════════════════════════════════════════════════════╗")
print("║     SwiftIR SDY Sharding Performance Benchmarks              ║")
print("║     (Main SwiftIR Module)                                    ║")
print("╚══════════════════════════════════════════════════════════════╝")

let totalStart = DispatchTime.now()

benchmarkMeshCreation()
benchmarkShardingAnnotation()
benchmarkMLIRGeneration()
benchmarkShardingOverhead()
benchmarkScaling()
benchmarkMeshTopology()

let totalEnd = DispatchTime.now()
let totalSeconds = Double(totalEnd.uptimeNanoseconds - totalStart.uptimeNanoseconds) / 1_000_000_000

print("\n" + String(repeating: "=", count: 60))
print("Total benchmark time: \(String(format: "%.2f", totalSeconds)) seconds")
print(String(repeating: "=", count: 60))
