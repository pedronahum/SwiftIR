//===-- StableHLO_CNN.swift - StableHLO CNN Example -------*- Swift -*-===//
//
// SwiftIR - Phase 11: StableHLO CNN Example
// Demonstrates building a complete CNN using StableHLO operations
// Designed for XLA compilation and GPU/TPU execution
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA

/// Complete CNN example using StableHLO for XLA compilation
///
/// This example demonstrates:
/// - Convolutional layers with stride and padding
/// - Batch normalization
/// - Activation functions (ReLU)
/// - Max pooling
/// - Global average pooling
/// - Dense (fully connected) layers
/// - Complete forward pass through a ResNet-style network
func buildStableHLOCNN() {
    print("=" * 70)
    print("StableHLO CNN Example - Optimized for XLA Compilation")
    print("=" * 70)

    // Create MLIR context and load StableHLO dialect
    let context = MLIRContext()
    context.loadAllDialects()
    _ = loadStablehloDialect(context)

    let f32 = FloatType.f32(context: context)
    let builder = IRBuilder(context: context)
    let module = MLIRModule(context: context)
    let loc = MLIRLocation.unknown(in: context)

    print("\n📦 Building StableHLO CNN Network...")
    print("   Architecture: Input → Conv → BN → ReLU → Pool → Conv → BN → ReLU → GAP → Dense")

    // Create function signature
    // Input: [batch=1, height=224, width=224, channels=3] (ImageNet-like input)
    let inputType = RankedTensorType(shape: [1, 224, 224, 3], elementType: f32, context: context)

    // Weights and parameters for the network
    let conv1KernelType = RankedTensorType(shape: [7, 7, 3, 64], elementType: f32, context: context)
    let bn1ParamsType = RankedTensorType(shape: [64], elementType: f32, context: context)
    let conv2KernelType = RankedTensorType(shape: [3, 3, 64, 128], elementType: f32, context: context)
    let bn2ParamsType = RankedTensorType(shape: [128], elementType: f32, context: context)
    let denseWeightsType = RankedTensorType(shape: [128, 1000], elementType: f32, context: context)

    // Build the network
    print("\n🔧 Layer Details:")
    print("   1. Conv1 (7x7, stride 2, 64 filters)")
    print("      Input:  [1, 224, 224, 3]")
    print("      Output: [1, 112, 112, 64]")

    // Simulate building the network with operation builders
    // In real usage, these would be actual tensor values
    let block = MLIRBlock(
        arguments: [
            inputType.typeHandle,
            conv1KernelType.typeHandle,
            bn1ParamsType.typeHandle, bn1ParamsType.typeHandle, bn1ParamsType.typeHandle, bn1ParamsType.typeHandle,
            conv2KernelType.typeHandle,
            bn2ParamsType.typeHandle, bn2ParamsType.typeHandle, bn2ParamsType.typeHandle, bn2ParamsType.typeHandle,
            denseWeightsType.typeHandle
        ],
        context: context
    )

    let input = block.getArgument(0)
    let conv1Kernel = block.getArgument(1)
    let bn1Scale = block.getArgument(2)
    let bn1Offset = block.getArgument(3)
    let bn1Mean = block.getArgument(4)
    let bn1Variance = block.getArgument(5)
    let conv2Kernel = block.getArgument(6)
    let bn2Scale = block.getArgument(7)
    let bn2Offset = block.getArgument(8)
    let bn2Mean = block.getArgument(9)
    let bn2Variance = block.getArgument(10)
    let denseWeights = block.getArgument(11)

    // Layer 1: Conv → BN → ReLU → MaxPool
    let conv1 = StableHLOMLOps.conv2d(
        input: input,
        kernel: conv1Kernel,
        strides: [2, 2],           // Stride 2 for downsampling
        padding: [(3, 3), (3, 3)], // Padding to maintain spatial dims
        in: builder
    )

    let bn1 = StableHLOMLOps.batchNorm(
        conv1,
        scale: bn1Scale,
        offset: bn1Offset,
        mean: bn1Mean,
        variance: bn1Variance,
        epsilon: 1e-5,
        in: builder
    )

    let relu1 = StableHLOMLOps.relu(bn1, in: builder)

    print("   2. MaxPool (3x3, stride 2)")
    print("      Input:  [1, 112, 112, 64]")
    print("      Output: [1, 56, 56, 64]")

    let pool1 = StableHLOMLOps.maxPool2d(
        relu1,
        windowSize: [3, 3],
        strides: [2, 2],
        padding: [(1, 1), (1, 1)],
        in: builder
    )

    // Layer 2: Conv → BN → ReLU
    print("   3. Conv2 (3x3, stride 1, 128 filters)")
    print("      Input:  [1, 56, 56, 64]")
    print("      Output: [1, 56, 56, 128]")

    let conv2 = StableHLOMLOps.conv2d(
        input: pool1,
        kernel: conv2Kernel,
        strides: [1, 1],
        padding: [(1, 1), (1, 1)],
        in: builder
    )

    let bn2 = StableHLOMLOps.batchNorm(
        conv2,
        scale: bn2Scale,
        offset: bn2Offset,
        mean: bn2Mean,
        variance: bn2Variance,
        epsilon: 1e-5,
        in: builder
    )

    let relu2 = StableHLOMLOps.relu(bn2, in: builder)

    // Global Average Pooling
    print("   4. GlobalAveragePool")
    print("      Input:  [1, 56, 56, 128]")
    print("      Output: [1, 1, 1, 128]")

    let gap = StableHLOMLOps.avgPool2d(
        relu2,
        windowSize: [56, 56],  // Pool entire spatial dimensions
        strides: [1, 1],
        padding: [(0, 0), (0, 0)],
        in: builder
    )

    // Reshape for dense layer
    print("   5. Reshape for Dense Layer")
    print("      Input:  [1, 1, 1, 128]")
    print("      Output: [1, 128]")

    let flattened = StableHLOMLOps.reshape(gap, to: [1, 128], in: builder)

    // Dense layer (classification)
    print("   6. Dense (Fully Connected)")
    print("      Input:  [1, 128]")
    print("      Output: [1, 1000] (ImageNet classes)")

    let logits = StableHLOMLOps.matmul(lhs: flattened, rhs: denseWeights, in: builder)

    // Verify all operations created
    let allOps = [conv1, bn1, relu1, pool1, conv2, bn2, relu2, gap, flattened, logits]
    let allValid = allOps.allSatisfy { !$0.isNull }

    if allValid {
        print("\n✅ Network built successfully!")
        print("   Total layers: 6")
        print("   Operations created: \(allOps.count)")
    } else {
        print("\n❌ Error: Some operations failed to create")
    }

    // Print optimization opportunity
    print("\n🚀 Optimization Pipeline:")
    print("   Next steps for XLA compilation:")
    print("   1. Apply stablehlo-canonicalize-dynamism pass")
    print("   2. Apply stablehlo-refine-shapes pass")
    print("   3. Compile to XLA for GPU/TPU execution")

    // Demonstrate the pipeline
    print("\n📊 Preparing module for XLA compilation...")

    let pipeline = StableHLOPipeline(context: context)

    // Note: In a real scenario, we would add all operations to a function
    // and then run the pipeline. For this demo, we just show the API.
    print("   ✓ Pipeline created")
    print("   ✓ StableHLO passes registered")
    print("   ✓ Ready for: pipeline.prepareForXLA(module)")

    // Performance characteristics
    print("\n⚡ Performance Characteristics:")
    print("   • All operations are tensor-level (no loops)")
    print("   • Fully parallelizable across batch and spatial dimensions")
    print("   • Optimized for GPU/TPU execution via XLA")
    print("   • Supports mixed precision training")
    print("   • Can be exported to StableHLO portable format")

    // Show typical batch sizes
    print("\n📈 Typical Configurations:")
    print("   Training:   batch_size = 32-256")
    print("   Inference:  batch_size = 1 (real-time)")
    print("   GPU memory: ~2-4 GB for batch_size = 32")

    print("\n" + "=" * 70)
    print("✨ StableHLO CNN Example Complete!")
    print("=" * 70)
}

// MARK: - Helper for Creating String Repetition

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run the example
buildStableHLOCNN()
