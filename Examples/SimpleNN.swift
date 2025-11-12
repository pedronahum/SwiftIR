//===-- SimpleNN.swift - Simple Neural Network Demo ------*- Swift -*-===//
//
// SwiftIR - Phase 7 Demo
// Demonstrates building a neural network layer using SwiftIRXLA
//
//===------------------------------------------------------------------===//

import SwiftIRXLA
import SwiftIRCore
import SwiftIRTypes
import SwiftIRBuilders

/// Demonstrates building a simple neural network layer in MLIR
///
/// This example shows:
/// - Creating tensors for weights, inputs, and biases
/// - Matrix multiplication (linear transformation)
/// - Bias addition
/// - ReLU activation function
/// - Generating valid MLIR IR
///
/// The network computes: output = relu(input @ weights + bias)
func buildSimpleNeuralNetworkLayer() {
    print(String(repeating: "=", count: 70))
    print("SwiftIR Phase 7 Demo: Simple Neural Network Layer")
    print(String(repeating: "=", count: 70))
    print()

    // Create MLIR context and register necessary dialects
    let ctx = MLIRContext()
    TensorDialect.register(with: ctx)
    ctx.loadDialect("linalg")
    ctx.loadDialect("arith")
    ctx.loadDialect("func")

    print("📦 Registered dialects: tensor, linalg, arith, func")
    print()

    // Build the neural network layer
    print("🏗️  Building neural network layer...")
    print("   Architecture: [batch=32, input=784] -> [batch=32, hidden=128]")
    print("   Operation: relu(X @ W + b)")
    print()

    let module = MLIRModule(context: ctx)
    let builder = IRBuilder(context: ctx)

    // Create function signature
    let f32 = FloatType.f32(context: ctx)

    // Input tensor type: [32, 784] (batch size, input features)
    let inputType = RankedTensorType(shape: [32, 784], elementType: f32, context: ctx)

    // Weights tensor type: [784, 128] (input features, hidden units)
    let weightsType = RankedTensorType(shape: [784, 128], elementType: f32, context: ctx)

    // Bias tensor type: [32, 128] (batch size, hidden units)
    let biasType = RankedTensorType(shape: [32, 128], elementType: f32, context: ctx)

    // Output tensor type: [32, 128]
    let outputType = RankedTensorType(shape: [32, 128], elementType: f32, context: ctx)

    // Build function
    let funcType = FunctionType(
        inputs: [inputType.typeHandle, weightsType.typeHandle, biasType.typeHandle],
        results: [outputType.typeHandle],
        context: ctx
    )

    // Create entry block with arguments
    let entryBlock = MLIRBlock(
        arguments: [inputType.typeHandle, weightsType.typeHandle, biasType.typeHandle],
        context: ctx
    )

    // Set insertion point to the block
    builder.setInsertionPoint(entryBlock)

    // Get block arguments
    let input = entryBlock.getArgument(0)
    let weights = entryBlock.getArgument(1)
    let bias = entryBlock.getArgument(2)

    print("✅ Created function with inputs:")
    print("   - input:   tensor<32x784xf32>")
    print("   - weights: tensor<784x128xf32>")
    print("   - bias:    tensor<32x128xf32>")
    print()

    // Step 1: Linear transformation (matrix multiplication)
    print("🔢 Step 1: Matrix multiplication (input @ weights)")
    let linear = MLOps.matmul(lhs: input, rhs: weights, in: builder)
    print("   Result: tensor<32x128xf32>")
    print()

    // Step 2: Add bias
    print("➕ Step 2: Add bias (linear + bias)")
    let withBias = MLOps.add(linear, bias, in: builder)
    print("   Result: tensor<32x128xf32>")
    print()

    // Step 3: Apply ReLU activation
    print("⚡ Step 3: ReLU activation (max(0, withBias))")
    let activated = MLOps.relu(withBias, in: builder)
    print("   Result: tensor<32x128xf32>")
    print()

    // Return the result
    let returnOp = Func.`return`([activated], location: builder.unknownLocation(), context: ctx)
    builder.insert(returnOp)

    // Create the function operation
    let function = Func.function(
        name: "neural_layer",
        type: funcType,
        location: builder.unknownLocation(),
        context: ctx
    )
    .setBody(entryBlock)
    .build()

    module.append(function)

    print("✅ Neural network layer built successfully!")
    print()
    print(String(repeating: "=", count: 70))
    print("Generated MLIR IR:")
    print(String(repeating: "=", count: 70))
    print()
    print(module.dump())
    print()
    print(String(repeating: "=", count: 70))
    print("✨ Demo complete! The neural network layer is ready for execution.")
    print(String(repeating: "=", count: 70))
}

/// Demonstrates element-wise operations on tensors
func buildElementWiseOperations() {
    print()
    print(String(repeating: "=", count: 70))
    print("Bonus Demo: Element-wise Tensor Operations")
    print(String(repeating: "=", count: 70))
    print()

    let ctx = MLIRContext()
    TensorDialect.register(with: ctx)
    ctx.loadDialect("arith")
    ctx.loadDialect("func")

    let module = MLIRModule(context: ctx)
    let builder = IRBuilder(context: ctx)

    let f32 = FloatType.f32(context: ctx)
    let tensorType = RankedTensorType(shape: [10, 10], elementType: f32, context: ctx)

    let funcType = FunctionType(
        inputs: [tensorType.typeHandle, tensorType.typeHandle],
        results: [tensorType.typeHandle],
        context: ctx
    )

    let entryBlock = MLIRBlock(
        arguments: [tensorType.typeHandle, tensorType.typeHandle],
        context: ctx
    )

    builder.setInsertionPoint(entryBlock)

    let a = entryBlock.getArgument(0)
    let b = entryBlock.getArgument(1)

    print("📊 Computing: (a + b) * (a - b) / a")
    print()

    let sum = MLOps.add(a, b, in: builder)
    print("✅ a + b")

    let diff = MLOps.sub(a, b, in: builder)
    print("✅ a - b")

    let product = MLOps.mul(sum, diff, in: builder)
    print("✅ (a + b) * (a - b)")

    let result = MLOps.div(product, a, in: builder)
    print("✅ ((a + b) * (a - b)) / a")
    print()

    let returnOp = Func.`return`([result], location: builder.unknownLocation(), context: ctx)
    builder.insert(returnOp)

    let function = Func.function(
        name: "elementwise_ops",
        type: funcType,
        location: builder.unknownLocation(),
        context: ctx
    )
    .setBody(entryBlock)
    .build()

    module.append(function)

    print("Generated MLIR:")
    print(String(repeating: "-", count: 70))
    print(module.dump())
    print(String(repeating: "-", count: 70))
    print()
}

/// Demonstrates a 2-layer neural network
func buildTwoLayerNetwork() {
    print()
    print(String(repeating: "=", count: 70))
    print("Advanced Demo: Two-Layer Neural Network")
    print(String(repeating: "=", count: 70))
    print()

    let ctx = MLIRContext()
    TensorDialect.register(with: ctx)
    ctx.loadDialect("linalg")
    ctx.loadDialect("arith")
    ctx.loadDialect("math")
    ctx.loadDialect("func")

    let module = MLIRModule(context: ctx)
    let builder = IRBuilder(context: ctx)

    let f32 = FloatType.f32(context: ctx)

    // Network architecture: 784 -> 128 -> 10
    print("🧠 Network Architecture:")
    print("   Input:  784 features (e.g., MNIST 28x28 images)")
    print("   Hidden: 128 neurons with ReLU")
    print("   Output: 10 classes with Softmax")
    print()

    // Input: [batch, 784]
    let inputType = RankedTensorType(shape: [32, 784], elementType: f32, context: ctx)

    // Layer 1 weights: [784, 128]
    let w1Type = RankedTensorType(shape: [784, 128], elementType: f32, context: ctx)
    let b1Type = RankedTensorType(shape: [32, 128], elementType: f32, context: ctx)

    // Layer 2 weights: [128, 10]
    let w2Type = RankedTensorType(shape: [128, 10], elementType: f32, context: ctx)
    let b2Type = RankedTensorType(shape: [32, 10], elementType: f32, context: ctx)

    // Output: [batch, 10]
    let outputType = RankedTensorType(shape: [32, 10], elementType: f32, context: ctx)

    let funcType = FunctionType(
        inputs: [inputType.typeHandle, w1Type.typeHandle, b1Type.typeHandle,
                 w2Type.typeHandle, b2Type.typeHandle],
        results: [outputType.typeHandle],
        context: ctx
    )

    let entryBlock = MLIRBlock(
        arguments: [inputType.typeHandle, w1Type.typeHandle, b1Type.typeHandle,
                   w2Type.typeHandle, b2Type.typeHandle],
        context: ctx
    )

    builder.setInsertionPoint(entryBlock)

    let x = entryBlock.getArgument(0)
    let w1 = entryBlock.getArgument(1)
    let b1 = entryBlock.getArgument(2)
    let w2 = entryBlock.getArgument(3)
    let b2 = entryBlock.getArgument(4)

    // Layer 1: relu(x @ w1 + b1)
    print("📍 Layer 1: Hidden layer with ReLU")
    let hidden_linear = MLOps.matmul(lhs: x, rhs: w1, in: builder)
    let hidden_bias = MLOps.add(hidden_linear, b1, in: builder)
    let hidden = MLOps.relu(hidden_bias, in: builder)
    print("   ✅ Output: tensor<32x128xf32>")
    print()

    // Layer 2: softmax(hidden @ w2 + b2)
    print("📍 Layer 2: Output layer with Softmax")
    let output_linear = MLOps.matmul(lhs: hidden, rhs: w2, in: builder)
    let output_bias = MLOps.add(output_linear, b2, in: builder)
    let output = MLOps.softmax(output_bias, in: builder)
    print("   ✅ Output: tensor<32x10xf32>")
    print()

    let returnOp = Func.`return`([output], location: builder.unknownLocation(), context: ctx)
    builder.insert(returnOp)

    let function = Func.function(
        name: "two_layer_network",
        type: funcType,
        location: builder.unknownLocation(),
        context: ctx
    )
    .setBody(entryBlock)
    .build()

    module.append(function)

    print("Generated MLIR:")
    print(String(repeating: "-", count: 70))
    print(module.dump())
    print(String(repeating: "-", count: 70))
    print()
}

// MARK: - Main Entry Point

@main
struct SimpleNNDemo {
    static func main() {
        print()
        print("🎯 SwiftIR Phase 7: ML Operations Demo")
        print("   Building Neural Networks with MLIR in Swift")
        print()

        // Run demos
        buildSimpleNeuralNetworkLayer()
        buildElementWiseOperations()
        buildTwoLayerNetwork()

        print(String(repeating: "=", count: 70))
        print("🎉 All demos completed successfully!")
        print(String(repeating: "=", count: 70))
        print()
        print("What you've seen:")
        print("  ✅ Tensor operations (empty, create)")
        print("  ✅ Linear algebra (matrix multiplication)")
        print("  ✅ Element-wise operations (add, sub, mul, div)")
        print("  ✅ Activation functions (ReLU, Softmax)")
        print("  ✅ Complete neural network layers")
        print("  ✅ Valid MLIR IR generation")
        print()
        print("Next steps:")
        print("  → This MLIR can be compiled to LLVM IR")
        print("  → Then to native code for CPU/GPU execution")
        print("  → Or converted to XLA for TPU execution (Phase 9)")
        print()
    }
}
