//===-- StableHLO_ResNet.swift - StableHLO ResNet Example -*-Swift-*-===//
//
// SwiftIR - Phase 11: StableHLO ResNet Example
// Demonstrates building a ResNet-style network with residual connections
// Optimized for XLA compilation and distributed training
//
//===------------------------------------------------------------------===//

import SwiftIRCore
import SwiftIRTypes
import SwiftIRStableHLO
import SwiftIRXLA
import MLIRCoreWrapper

/// ResNet-style network using StableHLO
///
/// Key features:
/// - Residual connections (skip connections)
/// - Bottleneck blocks
/// - Batch normalization
/// - Strided convolutions for downsampling
/// - Global average pooling
func buildStableHLOResNet() {
    print("=" * 70)
    print("StableHLO ResNet Example - Residual Learning Framework")
    print("=" * 70)

    let context = MLIRContext()
    context.loadAllDialects()
    _ = loadStablehloDialect(context)

    let f32 = FloatType.f32(context: context)
    let builder = IRBuilder(context: context)

    print("\n🏗️  Building ResNet-18 Style Network")
    print("   Key Innovation: Residual connections (y = F(x) + x)")

    // Define tensor shapes for a residual block
    // Block input/output: [batch=4, height=56, width=56, channels=64]
    let blockInputType = RankedTensorType(shape: [4, 56, 56, 64], elementType: f32, context: context)

    // Convolution kernels for bottleneck block
    let conv1x1Type = RankedTensorType(shape: [1, 1, 64, 64], elementType: f32, context: context)
    let conv3x3Type = RankedTensorType(shape: [3, 3, 64, 64], elementType: f32, context: context)

    // Batch norm parameters
    let bnParamsType = RankedTensorType(shape: [64], elementType: f32, context: context)

    print("\n📦 Residual Block Architecture:")
    print("   Input → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+residual) → ReLU")

    let block = MLIRBlock(
        arguments: [
            blockInputType.typeHandle,
            conv3x3Type.typeHandle, conv3x3Type.typeHandle,
            bnParamsType.typeHandle, bnParamsType.typeHandle, bnParamsType.typeHandle, bnParamsType.typeHandle,
            bnParamsType.typeHandle, bnParamsType.typeHandle, bnParamsType.typeHandle, bnParamsType.typeHandle
        ],
        context: context
    )

    let input = block.getArgument(0)
    let conv1Kernel = block.getArgument(1)
    let conv2Kernel = block.getArgument(2)

    // First batch norm params
    let bn1Scale = block.getArgument(3)
    let bn1Offset = block.getArgument(4)
    let bn1Mean = block.getArgument(5)
    let bn1Variance = block.getArgument(6)

    // Second batch norm params
    let bn2Scale = block.getArgument(7)
    let bn2Offset = block.getArgument(8)
    let bn2Mean = block.getArgument(9)
    let bn2Variance = block.getArgument(10)

    // RESIDUAL BLOCK: Branch pathway
    print("\n🔀 Branch Pathway (F(x)):")

    // First conv layer
    print("   1. Conv 3x3 (stride 1, padding 1)")
    let conv1 = StableHLOMLOps.conv2d(
        input: input,
        kernel: conv1Kernel,
        strides: [1, 1],
        padding: [(1, 1), (1, 1)],
        in: builder
    )

    let bn1 = StableHLOMLOps.batchNorm(
        conv1,
        scale: bn1Scale,
        offset: bn1Offset,
        mean: bn1Mean,
        variance: bn1Variance,
        in: builder
    )

    let relu1 = StableHLOMLOps.relu(bn1, in: builder)

    // Second conv layer
    print("   2. Conv 3x3 (stride 1, padding 1)")
    let conv2 = StableHLOMLOps.conv2d(
        input: relu1,
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
        in: builder
    )

    // RESIDUAL CONNECTION: Add skip connection
    print("\n➕ Skip Connection:")
    print("   output = F(x) + x")
    let residualSum = StableHLOMLOps.add(bn2, input, in: builder)

    // Final activation
    let output = StableHLOMLOps.relu(residualSum, in: builder)

    // Verify shapes preserved through residual block
    let outputType = output.getType()
    let inputTypeCheck = input.getType()

    if mlirTypeIsARankedTensorWrapper(outputType) && mlirTypeIsARankedTensorWrapper(inputTypeCheck) {
        let preserved = (0..<4).allSatisfy { dim in
            mlirShapedTypeGetDimSizeWrapper(outputType, dim) ==
            mlirShapedTypeGetDimSizeWrapper(inputTypeCheck, dim)
        }

        if preserved {
            print("   ✓ Shape preserved: [4, 56, 56, 64]")
        }
    }

    // Demonstrate a downsampling residual block
    print("\n⬇️  Downsampling Residual Block:")
    print("   Used for transitioning between stages")

    let downsampleInputType = RankedTensorType(shape: [4, 56, 56, 64], elementType: f32, context: context)
    let downsampleConvType = RankedTensorType(shape: [3, 3, 64, 128], elementType: f32, context: context)
    let downsampleSkipType = RankedTensorType(shape: [1, 1, 64, 128], elementType: f32, context: context)  // 1x1 conv for matching dimensions

    let dsBlock = MLIRBlock(
        arguments: [
            downsampleInputType.typeHandle,
            downsampleConvType.typeHandle,
            downsampleSkipType.typeHandle
        ],
        context: context
    )

    let dsInput = dsBlock.getArgument(0)
    let dsConvKernel = dsBlock.getArgument(1)
    let dsSkipKernel = dsBlock.getArgument(2)

    // Main branch: stride-2 convolution
    print("   Branch: Conv 3x3 stride 2 (64→128 channels)")
    let dsConv = StableHLOMLOps.conv2d(
        input: dsInput,
        kernel: dsConvKernel,
        strides: [2, 2],  // Downsample spatial dimensions
        padding: [(1, 1), (1, 1)],
        in: builder
    )

    // Skip connection: 1x1 stride-2 convolution to match dimensions
    print("   Skip:   Conv 1x1 stride 2 (64→128 channels)")
    let dsSkip = StableHLOMLOps.conv2d(
        input: dsInput,
        kernel: dsSkipKernel,
        strides: [2, 2],
        padding: [(0, 0), (0, 0)],
        in: builder
    )

    // Add residual
    let dsSum = StableHLOMLOps.add(dsConv, dsSkip, in: builder)

    // Verify downsampled output
    let dsOutputType = dsSum.getType()
    if mlirTypeIsARankedTensorWrapper(dsOutputType) {
        let H = mlirShapedTypeGetDimSizeWrapper(dsOutputType, 1)
        let W = mlirShapedTypeGetDimSizeWrapper(dsOutputType, 2)
        let C = mlirShapedTypeGetDimSizeWrapper(dsOutputType, 3)
        print("   ✓ Output shape: [4, \(H), \(W), \(C)]")
        print("   ✓ Spatial dims halved, channels doubled")
    }

    // Complete ResNet architecture overview
    print("\n🎯 Complete ResNet-18 Architecture:")
    print("   Stage 0: Conv 7x7 stride 2, MaxPool 3x3 stride 2")
    print("   Stage 1: 2 × Residual Block (64 channels)")
    print("   Stage 2: 2 × Residual Block (128 channels, first with stride 2)")
    print("   Stage 3: 2 × Residual Block (256 channels, first with stride 2)")
    print("   Stage 4: 2 × Residual Block (512 channels, first with stride 2)")
    print("   Final:   Global Average Pool → Dense (1000 classes)")

    // Key advantages
    print("\n💡 Residual Learning Advantages:")
    print("   ✓ Enables training very deep networks (50-152+ layers)")
    print("   ✓ Mitigates vanishing gradient problem")
    print("   ✓ Faster convergence during training")
    print("   ✓ Better feature reuse")
    print("   ✓ Gradients flow directly through skip connections")

    // XLA optimization benefits
    print("\n🚀 XLA Compilation Benefits:")
    print("   • Fuses batch norm + activation into single kernel")
    print("   • Optimizes residual addition")
    print("   • Eliminates temporary buffers where possible")
    print("   • Generates specialized code for fixed batch sizes")
    print("   • Maximizes memory bandwidth utilization")

    // Training characteristics
    print("\n📊 Training Characteristics:")
    print("   Batch size:     256 (distributed across 8 GPUs)")
    print("   Learning rate:  0.1 with cosine decay")
    print("   Optimizer:      SGD with momentum 0.9")
    print("   Epochs:         90-120")
    print("   Augmentation:   RandomCrop, RandomFlip, ColorJitter")

    // Performance metrics
    print("\n⚡ Performance (ImageNet):")
    print("   ResNet-18:  69.8% top-1, 89.1% top-5")
    print("   ResNet-50:  76.2% top-1, 93.0% top-5")
    print("   Throughput: ~1000 images/sec (V100 GPU)")

    print("\n" + "=" * 70)
    print("✨ StableHLO ResNet Example Complete!")
    print("=" * 70)
}

// MARK: - Helper for String Repetition

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run the example
buildStableHLOResNet()
