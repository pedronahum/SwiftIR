//===-- PipelinedBenchmark.swift - Pipelined Execution Benchmark --*- Swift -*-===//
//
// SwiftIR Pipelined Execution Benchmark
//
// Compares sequential vs pipelined execution to measure throughput improvement
// from overlapping D2H transfers with the next computation.
//
//===----------------------------------------------------------------------===//

import SwiftIRXLA
import Foundation

@main
struct PipelinedBenchmark {
    static func main() throws {
        print("=".repeated(70))
        print("SwiftIR Pipelined Execution Benchmark")
        print("=".repeated(70))
        print()

        // Initialize PJRT client
        print("Initializing PJRT CPU client...")
        let client = try PJRTClient(backend: .cpu)

        guard let device = client.addressableDevices.first else {
            fatalError("No addressable devices found")
        }

        // Test configuration
        let batchSize = 100_000
        let totalExecutions = 500
        let warmupExecutions = 20

        print()
        print("Configuration:")
        print("  Batch size: \(batchSize)")
        print("  Total executions: \(totalExecutions)")
        print("  Warmup: \(warmupExecutions)")
        print()

        // Create test MLIR module (simple vectorized computation)
        let mlirModule = """
        module @pipelined_benchmark {
            func.func @main(%arg0: tensor<\(batchSize)xf32>) -> (tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>) {
                // y1 = x^2
                %0 = stablehlo.multiply %arg0, %arg0 : tensor<\(batchSize)xf32>
                // y2 = x + x^2
                %1 = stablehlo.add %arg0, %0 : tensor<\(batchSize)xf32>
                return %0, %1 : tensor<\(batchSize)xf32>, tensor<\(batchSize)xf32>
            }
        }
        """

        // Compile the module
        print("Compiling MLIR module...")
        let executable = try client.compile(mlirModule: mlirModule)
        print("  Compilation successful")
        print()

        // Prepare test data - use multiple input buffers to simulate streaming
        var inputBuffers: [[Float]] = []
        for i in 0..<4 {
            var buffer = [Float](repeating: 0, count: batchSize)
            for j in 0..<batchSize {
                buffer[j] = Float(i * batchSize + j) * 0.0001
            }
            inputBuffers.append(buffer)
        }

        var output1 = [Float](repeating: 0, count: batchSize)
        var output2 = [Float](repeating: 0, count: batchSize)

        // =================================================================
        // Benchmark 1: Sequential (non-pipelined) execution
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 1: Sequential Execution (baseline)")
        print("-".repeated(70))

        let poolV2 = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )

        // Warmup
        for i in 0..<warmupExecutions {
            let inputIdx = i % inputBuffers.count
            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV2.execute(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }
        }

        // Benchmark
        let seqStart = DispatchTime.now()
        for i in 0..<totalExecutions {
            let inputIdx = i % inputBuffers.count
            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV2.execute(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }
        }
        let seqEnd = DispatchTime.now()
        let seqTotalNs = seqEnd.uptimeNanoseconds - seqStart.uptimeNanoseconds
        let seqAvgUs = Double(seqTotalNs) / Double(totalExecutions) / 1000.0
        let seqThroughput = Double(totalExecutions) / (Double(seqTotalNs) / 1_000_000_000.0)

        print("  Total time: \(String(format: "%.1f", Double(seqTotalNs) / 1_000_000.0)) ms")
        print("  Average per execution: \(String(format: "%.1f", seqAvgUs)) μs")
        print("  Throughput: \(String(format: "%.0f", seqThroughput)) executions/sec")
        print()

        // =================================================================
        // Benchmark 2: Pipelined execution (depth 2)
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 2: Pipelined Execution (depth=2)")
        print("-".repeated(70))

        // Create a new pool for pipelined execution
        let pipelinePool = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )

        let pipeline = try PJRTExecutionPipeline(pool: pipelinePool, depth: 2)

        // Prepare multiple output buffers for pipeline
        var pipeOutputs1 = [[Float]](repeating: [Float](repeating: 0, count: batchSize), count: 4)
        var pipeOutputs2 = [[Float]](repeating: [Float](repeating: 0, count: batchSize), count: 4)

        // Warmup with pipelined execution
        for i in 0..<warmupExecutions {
            let inputIdx = i % inputBuffers.count
            let outputIdx = i % 4

            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try pipeOutputs1[outputIdx].withUnsafeMutableBytes { out1Ptr in
                    try pipeOutputs2[outputIdx].withUnsafeMutableBytes { out2Ptr in
                        try pipeline.submit(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }

            // Collect result from oldest if pipeline is full
            if pipeline.inFlight >= 2 {
                try pipeline.awaitOne()
            }
        }
        try pipeline.flush()

        // Reset pipeline stats
        let pipelinePool2 = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )
        let pipeline2 = try PJRTExecutionPipeline(pool: pipelinePool2, depth: 2)

        // Benchmark pipelined execution
        let pipeStart = DispatchTime.now()
        for i in 0..<totalExecutions {
            let inputIdx = i % inputBuffers.count
            let outputIdx = i % 4

            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try pipeOutputs1[outputIdx].withUnsafeMutableBytes { out1Ptr in
                    try pipeOutputs2[outputIdx].withUnsafeMutableBytes { out2Ptr in
                        try pipeline2.submit(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }

            // Await oldest result if pipeline is full
            if pipeline2.inFlight >= 2 {
                try pipeline2.awaitOne()
            }
        }
        // Flush remaining
        try pipeline2.flush()
        let pipeEnd = DispatchTime.now()
        let pipeTotalNs = pipeEnd.uptimeNanoseconds - pipeStart.uptimeNanoseconds
        let pipeAvgUs = Double(pipeTotalNs) / Double(totalExecutions) / 1000.0
        let pipeThroughput = Double(totalExecutions) / (Double(pipeTotalNs) / 1_000_000_000.0)

        let (submissions, completions, avgWaitUs) = pipeline2.getStats()

        print("  Total time: \(String(format: "%.1f", Double(pipeTotalNs) / 1_000_000.0)) ms")
        print("  Average per execution: \(String(format: "%.1f", pipeAvgUs)) μs")
        print("  Throughput: \(String(format: "%.0f", pipeThroughput)) executions/sec")
        print("  Speedup vs sequential: \(String(format: "%.2fx", seqThroughput / pipeThroughput * pipeThroughput / seqThroughput))")
        print()
        print("  Pipeline stats:")
        print("    Submissions: \(submissions)")
        print("    Completions: \(completions)")
        print("    Avg wait time: \(String(format: "%.1f", avgWaitUs)) μs")
        print()

        // =================================================================
        // Benchmark 3: Pipelined execution (depth 4)
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 3: Pipelined Execution (depth=4)")
        print("-".repeated(70))

        let pipelinePool4 = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )
        let pipeline4 = try PJRTExecutionPipeline(pool: pipelinePool4, depth: 4)

        // Warmup
        for i in 0..<warmupExecutions {
            let inputIdx = i % inputBuffers.count
            let outputIdx = i % 4

            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try pipeOutputs1[outputIdx].withUnsafeMutableBytes { out1Ptr in
                    try pipeOutputs2[outputIdx].withUnsafeMutableBytes { out2Ptr in
                        try pipeline4.submit(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }

            if pipeline4.inFlight >= 4 {
                try pipeline4.awaitOne()
            }
        }
        try pipeline4.flush()

        // Fresh pipeline for benchmark
        let pipelinePool4b = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )
        let pipeline4b = try PJRTExecutionPipeline(pool: pipelinePool4b, depth: 4)

        // Benchmark
        let pipe4Start = DispatchTime.now()
        for i in 0..<totalExecutions {
            let inputIdx = i % inputBuffers.count
            let outputIdx = i % 4

            try inputBuffers[inputIdx].withUnsafeBytes { inputPtr in
                try pipeOutputs1[outputIdx].withUnsafeMutableBytes { out1Ptr in
                    try pipeOutputs2[outputIdx].withUnsafeMutableBytes { out2Ptr in
                        try pipeline4b.submit(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }

            if pipeline4b.inFlight >= 4 {
                try pipeline4b.awaitOne()
            }
        }
        try pipeline4b.flush()
        let pipe4End = DispatchTime.now()
        let pipe4TotalNs = pipe4End.uptimeNanoseconds - pipe4Start.uptimeNanoseconds
        let pipe4AvgUs = Double(pipe4TotalNs) / Double(totalExecutions) / 1000.0
        let pipe4Throughput = Double(totalExecutions) / (Double(pipe4TotalNs) / 1_000_000_000.0)

        let (submissions4, completions4, avgWaitUs4) = pipeline4b.getStats()

        print("  Total time: \(String(format: "%.1f", Double(pipe4TotalNs) / 1_000_000.0)) ms")
        print("  Average per execution: \(String(format: "%.1f", pipe4AvgUs)) μs")
        print("  Throughput: \(String(format: "%.0f", pipe4Throughput)) executions/sec")
        print("  Speedup vs sequential: \(String(format: "%.2fx", pipeThroughput / seqThroughput * pipe4Throughput / pipeThroughput))")
        print()
        print("  Pipeline stats:")
        print("    Submissions: \(submissions4)")
        print("    Completions: \(completions4)")
        print("    Avg wait time: \(String(format: "%.1f", avgWaitUs4)) μs")
        print()

        // =================================================================
        // Summary
        // =================================================================
        print("=".repeated(70))
        print("Summary (\(totalExecutions) executions, batch size \(batchSize))")
        print("=".repeated(70))
        print()
        print("                        Total (ms)    Avg (μs)    Throughput (exec/s)    Speedup")
        print(String(format: "  Sequential (baseline)   %8.1f    %8.1f           %8.0f      1.00x",
                     Double(seqTotalNs) / 1_000_000.0, seqAvgUs, seqThroughput))
        print(String(format: "  Pipelined (depth=2)     %8.1f    %8.1f           %8.0f      %.2fx",
                     Double(pipeTotalNs) / 1_000_000.0, pipeAvgUs, pipeThroughput, seqAvgUs / pipeAvgUs))
        print(String(format: "  Pipelined (depth=4)     %8.1f    %8.1f           %8.0f      %.2fx",
                     Double(pipe4TotalNs) / 1_000_000.0, pipe4AvgUs, pipe4Throughput, seqAvgUs / pipe4AvgUs))
        print()

        // Verify correctness with one final execution
        print("Verifying output correctness...")
        let testInput: Float = 0.5
        inputBuffers[0][0] = testInput
        try inputBuffers[0].withUnsafeBytes { inputPtr in
            try output1.withUnsafeMutableBytes { out1Ptr in
                try output2.withUnsafeMutableBytes { out2Ptr in
                    try poolV2.execute(
                        inputData: [inputPtr.baseAddress!],
                        outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                    )
                }
            }
        }
        let expected1 = testInput * testInput
        let expected2 = testInput + expected1
        if abs(output1[0] - expected1) < 1e-5 && abs(output2[0] - expected2) < 1e-5 {
            print("  ✓ Outputs correct!")
        } else {
            print("  ✗ Output mismatch: got (\(output1[0]), \(output2[0])), expected (\(expected1), \(expected2))")
        }

        print()
        print("Benchmark complete!")
    }
}

extension String {
    func repeated(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}
