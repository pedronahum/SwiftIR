//===-- BufferPoolBenchmark.swift - Buffer Pool Performance Test --*- Swift -*-===//
//
// SwiftIR Buffer Pool V2 and Batched Transfer Benchmark
//
// This benchmark compares execution performance across different optimization levels:
// 1. Buffer Pool V1 (PJRT_ExecutePooled)
// 2. Buffer Pool V2 with timing (PJRT_ExecutePooledV2)
// 3. Buffer Pool V2 without timing
// 4. Ultra-fast path (PJRT_ExecuteUltraFast)
//
//===----------------------------------------------------------------------===//

import SwiftIRXLA
import Foundation

@main
struct BufferPoolBenchmark {
    static func main() throws {
        print("=".repeated(70))
        print("SwiftIR Buffer Pool V2 & Batched Transfer Benchmark")
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
        let warmupIterations = 10
        let benchmarkIterations = 100

        print()
        print("Configuration:")
        print("  Batch size: \(batchSize)")
        print("  Warmup iterations: \(warmupIterations)")
        print("  Benchmark iterations: \(benchmarkIterations)")
        print()

        // Create test MLIR module (simple vectorized computation)
        let mlirModule = """
        module @buffer_pool_benchmark {
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

        // Prepare test data
        var inputData = [Float](repeating: 0, count: batchSize)
        for i in 0..<batchSize {
            inputData[i] = Float(i) * 0.0001
        }

        var output1 = [Float](repeating: 0, count: batchSize)
        var output2 = [Float](repeating: 0, count: batchSize)

        // =================================================================
        // Benchmark 1: Buffer Pool V1 (baseline)
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 1: Buffer Pool V1 (PJRT_ExecutePooled) - Baseline")
        print("-".repeated(70))

        let poolV1 = try PJRTBufferPool(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize]
        )

        // Warmup
        for _ in 0..<warmupIterations {
            _ = try inputData.withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV1.execute(
                            inputData: [inputPtr.baseAddress!],
                            inputSizes: [batchSize],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!],
                            outputSizes: [batchSize, batchSize]
                        )
                    }
                }
            }
        }

        // Benchmark
        var poolV1Timings: [Double] = []
        for _ in 0..<benchmarkIterations {
            let timing = try inputData.withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV1.execute(
                            inputData: [inputPtr.baseAddress!],
                            inputSizes: [batchSize],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!],
                            outputSizes: [batchSize, batchSize]
                        )
                    }
                }
            }
            poolV1Timings.append(timing.totalUs)
        }

        let poolV1Avg = poolV1Timings.reduce(0, +) / Double(benchmarkIterations)
        let poolV1Min = poolV1Timings.min()!

        print("  Average: \(String(format: "%.1f", poolV1Avg)) μs")
        print("  Min:     \(String(format: "%.1f", poolV1Min)) μs")
        print()

        // Print first timing breakdown
        let v1DetailedTiming = try inputData.withUnsafeBytes { inputPtr in
            try output1.withUnsafeMutableBytes { out1Ptr in
                try output2.withUnsafeMutableBytes { out2Ptr in
                    try poolV1.execute(
                        inputData: [inputPtr.baseAddress!],
                        inputSizes: [batchSize],
                        outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!],
                        outputSizes: [batchSize, batchSize]
                    )
                }
            }
        }
        print("  Timing breakdown:")
        print("    H2D:      \(String(format: "%6.1f", v1DetailedTiming.h2dCreateUs)) μs")
        print("    Execute:  \(String(format: "%6.1f", v1DetailedTiming.executeUs)) μs")
        print("    D2H Init: \(String(format: "%6.1f", v1DetailedTiming.d2hInitiateUs)) μs")
        print("    D2H Await:\(String(format: "%6.1f", v1DetailedTiming.d2hAwaitUs)) μs")
        print("    Cleanup:  \(String(format: "%6.1f", v1DetailedTiming.bufferDestroyUs)) μs")
        print()

        // =================================================================
        // Benchmark 2: Buffer Pool V2 with timing
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 2: Buffer Pool V2 with timing (PJRT_ExecutePooledV2)")
        print("-".repeated(70))

        let poolV2 = try PJRTBufferPoolV2(
            client: client.clientHandle!,
            device: device.deviceHandle!,
            executable: executable.executableHandle!,
            inputSizes: [batchSize],
            outputSizes: [batchSize, batchSize]
        )

        // Warmup
        for _ in 0..<warmupIterations {
            try inputData.withUnsafeBytes { inputPtr in
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
        var poolV2Timings: [Double] = []
        for _ in 0..<benchmarkIterations {
            let timing = try inputData.withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV2.executeWithTiming(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }
            poolV2Timings.append(timing.totalUs)
        }

        let poolV2Avg = poolV2Timings.reduce(0, +) / Double(benchmarkIterations)
        let poolV2Min = poolV2Timings.min()!

        print("  Average: \(String(format: "%.1f", poolV2Avg)) μs")
        print("  Min:     \(String(format: "%.1f", poolV2Min)) μs")
        print("  Speedup vs V1: \(String(format: "%.2fx", poolV1Avg / poolV2Avg))")
        print()

        // Print timing breakdown
        let v2DetailedTiming = try inputData.withUnsafeBytes { inputPtr in
            try output1.withUnsafeMutableBytes { out1Ptr in
                try output2.withUnsafeMutableBytes { out2Ptr in
                    try poolV2.executeWithTiming(
                        inputData: [inputPtr.baseAddress!],
                        outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                    )
                }
            }
        }
        print("  Timing breakdown:")
        print("    H2D:      \(String(format: "%6.1f", v2DetailedTiming.h2dCreateUs)) μs")
        print("    Execute:  \(String(format: "%6.1f", v2DetailedTiming.executeUs)) μs")
        print("    D2H Init: \(String(format: "%6.1f", v2DetailedTiming.d2hInitiateUs)) μs")
        print("    D2H Await:\(String(format: "%6.1f", v2DetailedTiming.d2hAwaitUs)) μs")
        print("    Cleanup:  \(String(format: "%6.1f", v2DetailedTiming.bufferDestroyUs)) μs")
        print()

        // =================================================================
        // Benchmark 3: Buffer Pool V2 without timing (no timing overhead)
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 3: Buffer Pool V2 without timing (no timing overhead)")
        print("-".repeated(70))

        // Warmup
        for _ in 0..<warmupIterations {
            try inputData.withUnsafeBytes { inputPtr in
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

        // Benchmark (wall clock time)
        let v2NoTimingStart = DispatchTime.now()
        for _ in 0..<benchmarkIterations {
            try inputData.withUnsafeBytes { inputPtr in
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
        let v2NoTimingEnd = DispatchTime.now()
        let v2NoTimingTotalNs = v2NoTimingEnd.uptimeNanoseconds - v2NoTimingStart.uptimeNanoseconds
        let v2NoTimingAvg = Double(v2NoTimingTotalNs) / Double(benchmarkIterations) / 1000.0

        print("  Average: \(String(format: "%.1f", v2NoTimingAvg)) μs (wall clock)")
        print("  Speedup vs V1: \(String(format: "%.2fx", poolV1Avg / v2NoTimingAvg))")
        print()

        // =================================================================
        // Benchmark 4: Ultra-fast path
        // =================================================================
        print("-".repeated(70))
        print("Benchmark 4: Ultra-fast path (PJRT_ExecuteUltraFast)")
        print("-".repeated(70))

        // Warmup
        for _ in 0..<warmupIterations {
            try inputData.withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV2.executeUltraFast(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }
        }

        // Benchmark (wall clock time)
        let ultraFastStart = DispatchTime.now()
        for _ in 0..<benchmarkIterations {
            try inputData.withUnsafeBytes { inputPtr in
                try output1.withUnsafeMutableBytes { out1Ptr in
                    try output2.withUnsafeMutableBytes { out2Ptr in
                        try poolV2.executeUltraFast(
                            inputData: [inputPtr.baseAddress!],
                            outputData: [out1Ptr.baseAddress!, out2Ptr.baseAddress!]
                        )
                    }
                }
            }
        }
        let ultraFastEnd = DispatchTime.now()
        let ultraFastTotalNs = ultraFastEnd.uptimeNanoseconds - ultraFastStart.uptimeNanoseconds
        let ultraFastAvg = Double(ultraFastTotalNs) / Double(benchmarkIterations) / 1000.0

        print("  Average: \(String(format: "%.1f", ultraFastAvg)) μs (wall clock)")
        print("  Speedup vs V1: \(String(format: "%.2fx", poolV1Avg / ultraFastAvg))")
        print()

        // =================================================================
        // Summary
        // =================================================================
        print("=".repeated(70))
        print("Summary (batch size: \(batchSize), \(benchmarkIterations) iterations)")
        print("=".repeated(70))
        print()
        print("                             Average (μs)    Min (μs)    Speedup")
        print("  Buffer Pool V1 (baseline)    \(String(format: "%8.1f", poolV1Avg))      \(String(format: "%7.1f", poolV1Min))     1.00x")
        print("  Buffer Pool V2 (w/ timing)   \(String(format: "%8.1f", poolV2Avg))      \(String(format: "%7.1f", poolV2Min))     \(String(format: "%.2fx", poolV1Avg / poolV2Avg))")
        print("  Buffer Pool V2 (no timing)   \(String(format: "%8.1f", v2NoTimingAvg))         N/A     \(String(format: "%.2fx", poolV1Avg / v2NoTimingAvg))")
        print("  Ultra-fast path              \(String(format: "%8.1f", ultraFastAvg))         N/A     \(String(format: "%.2fx", poolV1Avg / ultraFastAvg))")
        print()

        // Verify correctness
        print("Verifying output correctness...")
        var correct = true
        for i in 0..<min(10, batchSize) {
            let expected1 = inputData[i] * inputData[i]
            let expected2 = inputData[i] + expected1
            if abs(output1[i] - expected1) > 1e-5 || abs(output2[i] - expected2) > 1e-5 {
                print("  ERROR at index \(i): got (\(output1[i]), \(output2[i])), expected (\(expected1), \(expected2))")
                correct = false
            }
        }
        if correct {
            print("  All outputs correct!")
        }

        print()
        print("V2 Execution count: \(poolV2.executionCount)")
        print("Benchmark complete!")
    }
}

extension String {
    func repeated(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}
