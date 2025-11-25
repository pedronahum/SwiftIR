# SwiftIR Symbolic Automatic Differentiation

**Complete reverse-mode automatic differentiation system for SwiftIR, powered by Swift's `_Differentiation` module and StableHLO operations.**

## Overview

The SymbolicAD module provides a production-ready automatic differentiation system that:

- ✅ **Integrates with Swift's native differentiation** - Uses `@differentiable`, `@derivative`, and VJP (Vector-Jacobian Product)
- ✅ **Traces to StableHLO MLIR** - Generates portable, optimizable intermediate representation
- ✅ **Executes on PJRT** - Compiles and runs on CPUs, GPUs, and TPUs
- ✅ **Comprehensive operation coverage** - 80+ differentiable operations with proper gradient implementations
- ✅ **Production tested** - 300+ tests across 38 phases validating correctness

## Architecture

```
Swift Code with @differentiable
          ↓
   DifferentiableTracer
   (symbolic execution)
          ↓
    MLIR StableHLO IR
          ↓
   XLA Compilation
          ↓
    PJRT Execution
   (forward + gradient)
```

### Key Components

1. **[ADIntegration.swift](ADIntegration.swift)** - Core automatic differentiation implementation
   - `DifferentiableTracer` - Symbolic tensor type that builds MLIR graphs
   - 80+ differentiable operations with VJPs
   - Arithmetic, transcendentals, matrix ops, convolutions, normalizations, loss functions

2. **[BackendCompilation.swift](BackendCompilation.swift)** - MLIR code generation and StableHLO compilation
   - Converts traced operations to MLIR text
   - Handles generic operation syntax for StableHLO v1.13+
   - Manages regions for operations like scatter

3. **[RuntimeIntegration.swift](RuntimeIntegration.swift)** - PJRT execution interface
   - Compiles MLIR to executable code
   - Executes forward pass and gradient computation
   - Type-safe tensor I/O

## Quick Start

### Basic Example: Gradient Computation

```swift
import SwiftIR
import _Differentiation

// Define a differentiable function
@differentiable(reverse)
func squaredNorm(_ x: DifferentiableTracer) -> DifferentiableTracer {
    return diffSum(x * x)
}

// Compile to PJRT
let gradFunc = try compileGradientForPJRT(
    input: TensorSpec(shape: [4], dtype: .float32)
) { x in
    squaredNorm(x)
}

// Execute with gradient
let input: [Float] = [1.0, 2.0, 3.0, 4.0]
let gradient = try gradFunc.gradient(input)

print(gradient)  // [2.0, 4.0, 6.0, 8.0] = 2*x
```

### Neural Network Layer Example

```swift
@differentiable(reverse)
func denseLayer(
    _ input: DifferentiableTracer,
    weights: DifferentiableTracer,
    bias: DifferentiableTracer
) -> DifferentiableTracer {
    let matmul = diffMatmul(input, weights)
    let withBias = matmul + bias
    return diffReLU(withBias)
}

// Compile and train
let gradFunc = try compileGradientForPJRT(
    input: TensorSpec(shape: [32, 128], dtype: .float32),
    weights: TensorSpec(shape: [128, 64], dtype: .float32),
    bias: TensorSpec(shape: [64], dtype: .float32)
) { input, weights, bias in
    let output = denseLayer(input, weights: weights, bias: bias)
    // ... compute loss
    return loss
}

// Get gradients for all parameters
let (loss, grads) = try gradFunc.forwardWithGradient(inputData, weightsData, biasData)
```

## Implemented Operations

### Basic Arithmetic (Phases 9-10)

- **Element-wise**: `+`, `-`, `*`, `/` (with broadcasting)
- **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`
- **Selection**: `diffSelect(condition, trueVals, falseVals)`

### Transcendental Functions (Phases 11-18)

- **Exponential/Log**: `diffExp`, `diffLog`, `diffLog1p`
- **Power**: `diffPow`, `diffSqrt`, `diffRsqrt`
- **Trigonometric**: `diffSin`, `diffCos`, `diffTan`
- **Inverse Trig**: `diffAsin`, `diffAcos`, `diffAtan`, `diffAtan2`
- **Hyperbolic**: `diffSinh`, `diffCosh`, `diffTanh`
- **Inverse Hyperbolic**: `diffAsinh`, `diffAcosh`, `diffAtanh`

### Matrix Operations (Phase 21)

- **Matrix Multiplication**: `diffMatmul(a, b)` - Full gradient support for batched matmul
- **Transpose**: Implicit in gradient computation

### Activation Functions (Phases 22, 32)

- **ReLU**: `diffReLU(x)` - Rectified Linear Unit
- **Leaky ReLU**: `diffLeakyReLU(x, negativeSlope: 0.01)`
- **ELU**: `diffELU(x, alpha: 1.0)` - Exponential Linear Unit
- **SELU**: `diffSELU(x)` - Scaled ELU
- **Sigmoid**: `diffSigmoid(x)` - Logistic function
- **Tanh**: `diffTanh(x)` - Hyperbolic tangent
- **Softplus**: `diffSoftplus(x)` - Smooth approximation of ReLU
- **Swish/SiLU**: `diffSwish(x)` - Self-gated activation
- **GELU**: `diffGELU(x)` - Gaussian Error Linear Unit
- **Mish**: `diffMish(x)` - Self-regularized activation

### Reduction Operations (Phases 27)

- **Sum**: `diffSum(x)` - Reduce all elements to scalar
- **Mean**: `diffMean(x)` - Average of all elements
- **Max**: `diffMax(x)` - Maximum element (with gradient routing)
- **Min**: `diffMin(x)` - Minimum element

### Broadcasting & Shape Operations (Phases 24, 28, 35)

- **Broadcast**: Automatic broadcasting for binary operations
- **Reshape**: `diffReshape(x, shape: [...])`
- **Flatten**: `diffFlatten(x, startDim: 1, endDim: -1)`
- **Permute**: `diffPermute(x, dims: [0, 3, 1, 2])` - Dimension reordering
- **Squeeze**: `diffSqueeze(x, dims: [...])` - Remove singleton dimensions
- **Unsqueeze**: `diffUnsqueeze(x, dims: [...])` - Add singleton dimensions

### Convolution Operations (Phase 29)

- **Conv2D**: `diffConv2D(input, kernel, stride, padding, dilation)`
  - Full gradient support for both input and kernel
  - Flexible padding: "SAME" or "VALID"
  - Strided and dilated convolutions

### Pooling Operations (Phase 33)

- **Max Pooling**: `diffMaxPool2D(x, kernelSize, stride, padding)`
- **Average Pooling**: `diffAvgPool2D(x, kernelSize, stride, padding)`
- Gradient routing for max pooling (only max elements receive gradients)

### Normalization (Phase 34)

- **Batch Normalization**: `diffBatchNorm2D(x, scale, bias, epsilon)`
  - Per-channel normalization across batch dimension
  - Full gradient support for input, scale, and bias

- **Layer Normalization**: `diffLayerNorm(x, scale, bias, epsilon)`
  - Normalizes across feature dimensions
  - Common in transformers and NLP

### Dropout (Phase 36)

- **Dropout**: `diffDropout(x, probability, training)`
  - Training mode: scales by 1/(1-p) for unbiased expectation
  - Inference mode: scales by (1-p) for consistent output

### Embedding Operations (Phase 37)

- **Gather**: `diffGather(embeddings, indices)` - Embedding lookup
  - Efficient sparse gradient accumulation
  - Critical for NLP and recommendation systems

- **Scatter**: `diffScatter(base, indices, updates)` - Sparse updates
  - Used internally for gather gradients
  - Supports add reduction for gradient accumulation

### Loss Functions

#### Phase 31: Core Loss Functions

- **Mean Squared Error**: `diffMSELoss(predictions, targets)`
  - L2 loss for regression tasks
  - Gradient: `2 * (pred - target) / n`

- **Cross-Entropy**: `diffCrossEntropyLoss(logits, labels)`
  - Classification loss with numerical stability
  - Combined softmax + log + negative log-likelihood

- **Softmax**: `diffSoftmax(x)` - Probability distribution
  - Numerically stable implementation with max subtraction

#### Phase 38: Advanced Loss Functions

- **Huber Loss**: `diffHuberLoss(predictions, targets, delta)`
  - Robust regression loss
  - Quadratic for small errors, linear for large (reduces outlier impact)
  - Smooth transition at delta threshold

- **KL Divergence**: `diffKLDivergence(predictions, targets)`
  - Measures distribution divergence
  - KL(P||Q) = sum(P * log(P/Q))
  - Numerical stability with epsilon clipping

### Utility Operations

- **Absolute Value**: `diffAbs(x)` - Element-wise absolute value
- **Sign**: `diffSign(x)` - Element-wise sign (-1, 0, 1)
- **Negate**: `diffNegate(x)` - Element-wise negation
- **Clip**: `diffClip(x, min, max)` - Value clamping

## Gradient Implementation Details

### VJP Pattern

All differentiable operations follow the Vector-Jacobian Product (VJP) pattern:

```swift
@differentiable(reverse)
public func diffOperation(_ x: DifferentiableTracer) -> DifferentiableTracer {
    // Forward pass: build MLIR operation
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("No active MLIRBuilder")
    }

    let result = builder.freshSSA()
    builder.addOperation(MLIROperation(
        result: result,
        opName: "stablehlo.operation",
        operands: [x.irValue],
        resultType: tensorType(shape: x.shape, dtype: x.dtype)
    ))

    return DifferentiableTracer(irValue: result, shape: x.shape, dtype: x.dtype)
}

@derivative(of: diffOperation)
public func diffOperationVJP(
    _ x: DifferentiableTracer
) -> (value: DifferentiableTracer, pullback: (DifferentiableTracer) -> DifferentiableTracer) {
    let y = diffOperation(x)

    func pullback(_ dy: DifferentiableTracer) -> DifferentiableTracer {
        // Compute gradient: dy * (∂operation/∂x)
        // Build MLIR operations for gradient computation
        return gradientExpression
    }

    return (y, pullback)
}
```

### Broadcast-aware Gradients

Operations handle broadcasting automatically:

```swift
// Forward: [2, 3] + [3] → [2, 3] (broadcast)
let y = a + b

// Backward: gradient needs reduction to match original shapes
// dy: [2, 3] → dA: [2, 3] (same shape)
//           → dB: [3]    (sum over broadcast dimension)
```

The gradient computation includes `reduceSum` helper to handle dimension reduction for broadcasted operands.

### Complex Gradients

**Convolution gradients** involve three separate computations:
1. **Input gradient**: Convolution with transposed kernel (gradient convolution)
2. **Kernel gradient**: Convolution of input with output gradient
3. Both use StableHLO convolution with adjusted dimension numbers

**Pooling gradients** require:
1. **Max pooling**: Route gradients only to maximum elements (using comparison masks)
2. **Average pooling**: Distribute gradients evenly across pooling window

**Normalization gradients** compute:
1. Input gradients through normalized values
2. Scale/bias gradients via reduction across batch/feature dimensions

## Testing Strategy

### Comprehensive Test Coverage (300+ tests)

Each operation includes tests for:

1. **Forward pass correctness** - Verify output values match expected
2. **Gradient correctness** - Validate gradients against analytical derivatives
3. **Gradient shapes** - Ensure gradient shapes match input shapes
4. **Broadcasting behavior** - Test all broadcast scenarios
5. **Edge cases** - Zero inputs, negative values, boundary conditions
6. **Numerical stability** - Operations remain stable with extreme values

### Test Organization by Phase

- **Phase 9**: Validation framework
- **Phase 10**: Fixed MLIR operations
- **Phase 11**: Gradient execution
- **Phases 12-15**: Transcendentals and validations
- **Phase 16**: Gradient-based differentiation
- **Phases 17-20**: Advanced transcendentals and operations
- **Phase 21**: Matrix multiplication
- **Phase 22**: ReLU activation
- **Phase 23**: Multi-input operations
- **Phase 24**: Broadcasting
- **Phase 25**: Reduction gradients
- **Phase 26**: Trigonometric functions
- **Phase 27**: Reductions
- **Phase 28**: Shape operations
- **Phase 29**: Convolutions
- **Phase 30**: Element-wise operations
- **Phase 31**: Softmax and loss functions
- **Phase 32**: Activation functions
- **Phase 33**: Pooling operations
- **Phase 34**: Normalization (BatchNorm, LayerNorm)
- **Phase 35**: Advanced shape operations
- **Phase 36**: Dropout
- **Phase 37**: Gather/Scatter for embeddings
- **Phase 38**: Advanced loss functions (Huber, KL divergence)

### Example Test Pattern

```swift
func testOperationGradient() throws {
    let gradFunc = try compileGradientForPJRT(
        input: TensorSpec(shape: [4], dtype: .float32)
    ) { x in
        return diffOperation(x)
    }

    let input: [Float] = [1.0, 2.0, 3.0, 4.0]
    let gradient = try gradFunc.gradient(input)

    // Verify gradient shape
    XCTAssertEqual(gradient.count, 4)

    // Verify gradient values (analytical solution)
    XCTAssertEqual(gradient[0], expectedGrad0, accuracy: 0.01)
    XCTAssertEqual(gradient[1], expectedGrad1, accuracy: 0.01)
    // ...
}
```

## StableHLO v1.13 Compatibility

The implementation uses StableHLO v1.13.4 with proper syntax support:

### Generic Operation Syntax

Certain operations require generic MLIR syntax instead of custom assembly:

```mlir
// Scatter with reduction region
%result = "stablehlo.scatter"(%base, %indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
}) {
  scatter_dimension_numbers = #stablehlo.scatter<...>,
  unique_indices = false
} : (tensor<4xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<4xf32>

// Gather with dimension configuration
%result = "stablehlo.gather"(%embeddings, %indices) {
  dimension_numbers = #stablehlo.gather<...>,
  slice_sizes = array<i64: 1, 8>,
  indices_are_sorted = false
} : (tensor<10x8xf32>, tensor<1xi32>) -> tensor<1x8xf32>
```

### Type System

Proper handling of scalar tensors:

```swift
// Scalar tensor: tensor<f32> (not tensor<xf32>)
func tensorType(shape: [Int], dtype: DType) -> String {
    if shape.isEmpty {
        return "tensor<\(dtype.rawValue)>"  // Scalar
    } else {
        return "tensor<\(shape.map(String.init).joined(separator: "x"))x\(dtype.rawValue)>"
    }
}
```

## Known Limitations & Future Work

### Current Limitations

1. **Single precision only** - Currently focused on Float32, Float64 support planned
2. **Scalar reductions** - Most reductions reduce to scalar; axis-specific reductions coming
3. **Static shapes** - Dynamic shapes not yet supported
4. **CPU execution** - GPU execution requires MLIR rebuild (see main README)

### Planned Enhancements

1. **Higher-order gradients** - Forward-mode AD for Hessians
2. **Checkpointing** - Memory-efficient gradient computation for large models
3. **Custom VJPs** - User-defined gradient implementations
4. **Gradient accumulation** - Efficient batch processing
5. **Mixed precision** - FP16/BF16 training support
6. **Gradient clipping** - Built-in gradient norm clipping

## Performance Considerations

### Compilation

- **Graph building**: Symbolic trace is fast (Swift execution)
- **MLIR generation**: String concatenation overhead is minimal
- **XLA compilation**: One-time cost, executable is cached
- **Tip**: Compile once, execute many times

### Execution

- **CPU**: Optimized by XLA with vectorization and fusion
- **Memory**: In-place operations where possible
- **Parallelism**: Automatic PJRT parallelization across cores

### Optimization Strategies

1. **Operation fusion** - XLA automatically fuses compatible operations
2. **Memory layout** - XLA optimizes tensor layouts for cache efficiency
3. **Constant folding** - Compile-time evaluation of constants
4. **Dead code elimination** - Unused operations are removed

## Integration Examples

### Training Loop

```swift
// Model parameters
var weights = initializeWeights()
var bias = initializeBias()

// Compile gradient function once
let gradFunc = try compileGradientForPJRT(
    input: TensorSpec(shape: [batchSize, inputDim], dtype: .float32),
    weights: TensorSpec(shape: [inputDim, outputDim], dtype: .float32),
    bias: TensorSpec(shape: [outputDim], dtype: .float32)
) { input, w, b in
    let logits = diffMatmul(input, w) + b
    let probs = diffSoftmax(logits)
    return diffCrossEntropyLoss(probs, labels)
}

// Training loop
for epoch in 0..<numEpochs {
    for batch in dataLoader {
        // Forward + backward in one call
        let (loss, grads) = try gradFunc.forwardWithGradient(
            batch.inputs,
            weights,
            bias
        )

        // Update parameters
        weights = sgdUpdate(weights, grads.0, learningRate)
        bias = sgdUpdate(bias, grads.1, learningRate)

        print("Epoch \(epoch), Loss: \(loss)")
    }
}
```

### Custom Layer

```swift
@differentiable(reverse)
func residualBlock(
    _ x: DifferentiableTracer,
    conv1Kernel: DifferentiableTracer,
    conv2Kernel: DifferentiableTracer
) -> DifferentiableTracer {
    // First convolution
    let conv1 = diffConv2D(x, kernel: conv1Kernel,
                          stride: (1, 1), padding: "SAME")
    let relu1 = diffReLU(conv1)

    // Second convolution
    let conv2 = diffConv2D(relu1, kernel: conv2Kernel,
                          stride: (1, 1), padding: "SAME")

    // Residual connection
    return diffReLU(conv2 + x)
}
```

## Debugging Tips

### Inspecting Generated MLIR

```swift
// Enable debug output
let gradFunc = try compileGradientForPJRT(...) { x in
    // Your computation
}

// MLIR is printed to stderr during compilation
// Look for the generated StableHLO operations
```

### Gradient Checking

```swift
// Numerical gradient for verification
func numericalGradient(f: (Float) -> Float, x: Float, eps: Float = 1e-4) -> Float {
    return (f(x + eps) - f(x - eps)) / (2 * eps)
}

// Compare with automatic gradient
let autoGrad = try gradFunc.gradient([x])[0]
let numGrad = numericalGradient(f, x)
assert(abs(autoGrad - numGrad) < 1e-3, "Gradient mismatch!")
```

### Common Issues

1. **Shape mismatches** - Check tensor shapes at each operation
2. **NaN gradients** - Add epsilon to divisions and logs
3. **Exploding gradients** - Use gradient clipping or lower learning rate
4. **Zero gradients** - Verify operations are differentiable

## Contributing

When adding new operations:

1. **Implement forward pass** - Add operation to `ADIntegration.swift`
2. **Mark as differentiable** - Use `@differentiable(reverse)` attribute
3. **Implement VJP** - Add `@derivative` function with pullback
4. **Add MLIR generation** - Update `BackendCompilation.swift` if needed
5. **Write tests** - Add to appropriate Phase test file
6. **Test gradients** - Verify correctness with analytical derivatives
7. **Test shapes** - Ensure broadcasting and reduction work correctly

## References

- [Swift Differentiation](https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md)
- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [XLA Documentation](https://www.tensorflow.org/xla)
- [PJRT Plugin API](https://github.com/openxla/xla/tree/main/xla/pjrt/c)

---

**SwiftIR SymbolicAD: Production-ready automatic differentiation for modern ML in Swift.**
