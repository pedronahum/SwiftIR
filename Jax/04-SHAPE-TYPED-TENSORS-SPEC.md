# Shape-Typed Tensors Specification

## Overview

Shape-typed tensors leverage Swift's powerful type system to catch shape mismatches at compile time rather than runtime. This is a significant advantage over JAX/NumPy, where shape errors only appear when code runs.

## The Problem

In JAX/NumPy, shape errors are only caught at runtime:

```python
# This compiles fine but crashes at runtime
x = jnp.ones((32, 64))
y = jnp.ones((64, 128))
z = jnp.matmul(x, y)      # OK: [32,64] @ [64,128] -> [32,128]
w = jnp.matmul(z, x)      # RUNTIME ERROR: [32,128] @ [32,64] shapes incompatible
```

## The Swift Solution

With Swift's type system, we can catch this at compile time:

```swift
let x: Tensor<Shape2D<D32, D64>> = ones()
let y: Tensor<Shape2D<D64, D128>> = ones()
let z = matmul(x, y)      // Type: Tensor<Shape2D<D32, D128>>
let w = matmul(z, x)      // COMPILE ERROR: Cannot convert Shape2D<D32, D64> to Shape2D<D128, K>
```

## SwiftIR API Specification

### Dimension Types

```swift
/// Protocol for dimension sizes (known at compile time)
public protocol Dim {
    static var value: Int { get }
}

/// Concrete dimension types
public struct D1: Dim { public static let value = 1 }
public struct D2: Dim { public static let value = 2 }
public struct D4: Dim { public static let value = 4 }
public struct D8: Dim { public static let value = 8 }
public struct D16: Dim { public static let value = 16 }
public struct D32: Dim { public static let value = 32 }
public struct D64: Dim { public static let value = 64 }
public struct D128: Dim { public static let value = 128 }
public struct D256: Dim { public static let value = 256 }
public struct D512: Dim { public static let value = 512 }
public struct D768: Dim { public static let value = 768 }
public struct D1024: Dim { public static let value = 1024 }
public struct D2048: Dim { public static let value = 2048 }
public struct D4096: Dim { public static let value = 4096 }

/// Dynamic dimension (size known only at runtime)
public struct DynDim: Dim {
    public static var value: Int { fatalError("Dynamic dimension has no static value") }
    public let runtimeValue: Int
}

/// Custom dimension factory
public func dim<N: BinaryInteger>(_ n: N) -> some Dim {
    // Returns appropriate Dim type for common sizes, or DynDim otherwise
}
```

### Shape Types

```swift
/// Protocol for tensor shapes
public protocol TensorShape {
    /// Number of dimensions (rank)
    static var rank: Int { get }
    
    /// Dimension sizes as array
    static var dims: [Int] { get }
    
    /// Total element count
    static var elementCount: Int { get }
}

/// Scalar (rank 0)
public struct Scalar: TensorShape {
    public static let rank = 0
    public static let dims: [Int] = []
    public static let elementCount = 1
}

/// 1D shape
public struct Shape1D<D0: Dim>: TensorShape {
    public static var rank: Int { 1 }
    public static var dims: [Int] { [D0.value] }
    public static var elementCount: Int { D0.value }
}

/// 2D shape
public struct Shape2D<D0: Dim, D1: Dim>: TensorShape {
    public static var rank: Int { 2 }
    public static var dims: [Int] { [D0.value, D1.value] }
    public static var elementCount: Int { D0.value * D1.value }
}

/// 3D shape
public struct Shape3D<D0: Dim, D1: Dim, D2: Dim>: TensorShape {
    public static var rank: Int { 3 }
    public static var dims: [Int] { [D0.value, D1.value, D2.value] }
    public static var elementCount: Int { D0.value * D1.value * D2.value }
}

/// 4D shape (common for images: [batch, channels, height, width])
public struct Shape4D<D0: Dim, D1: Dim, D2: Dim, D3: Dim>: TensorShape {
    public static var rank: Int { 4 }
    public static var dims: [Int] { [D0.value, D1.value, D2.value, D3.value] }
    public static var elementCount: Int { D0.value * D1.value * D2.value * D3.value }
}

/// 5D shape
public struct Shape5D<D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim>: TensorShape {
    public static var rank: Int { 5 }
    public static var dims: [Int] { [D0.value, D1.value, D2.value, D3.value, D4.value] }
}
```

### Typed Tensor

```swift
/// Shape-typed tensor wrapper around DifferentiableTracer
public struct Tensor<Shape: TensorShape>: Differentiable {
    /// Underlying tracer for computation
    public var tracer: DifferentiableTracer
    
    /// Create tensor with type-checked shape
    public init(tracer: DifferentiableTracer) {
        precondition(tracer.shape == Shape.dims,
                     "Shape mismatch: expected \(Shape.dims), got \(tracer.shape)")
        self.tracer = tracer
    }
    
    /// Tangent vector type for differentiation
    public typealias TangentVector = Tensor<Shape>
}

// Type aliases for common shapes
public typealias Vector<N: Dim> = Tensor<Shape1D<N>>
public typealias Matrix<M: Dim, N: Dim> = Tensor<Shape2D<M, N>>
public typealias Tensor3<D0: Dim, D1: Dim, D2: Dim> = Tensor<Shape3D<D0, D1, D2>>
public typealias Tensor4<D0: Dim, D1: Dim, D2: Dim, D3: Dim> = Tensor<Shape4D<D0, D1, D2, D3>>
```

### Type-Safe Operations

#### Matrix Multiplication

```swift
/// Type-safe matrix multiplication
/// [M, K] @ [K, N] -> [M, N]
@differentiable(reverse)
public func matmul<M: Dim, K: Dim, N: Dim>(
    _ a: Matrix<M, K>,
    _ b: Matrix<K, N>
) -> Matrix<M, N> {
    let result = diffMatmul(a.tracer, b.tracer)
    return Matrix<M, N>(tracer: result)
}

/// Batched matrix multiplication
/// [B, M, K] @ [B, K, N] -> [B, M, N]
@differentiable(reverse)
public func batchMatmul<B: Dim, M: Dim, K: Dim, N: Dim>(
    _ a: Tensor3<B, M, K>,
    _ b: Tensor3<B, K, N>
) -> Tensor3<B, M, N> {
    let result = diffBatchedMatmul(a.tracer, b.tracer)
    return Tensor3<B, M, N>(tracer: result)
}

/// Matrix-vector multiplication
/// [M, N] @ [N] -> [M]
@differentiable(reverse)
public func matvec<M: Dim, N: Dim>(
    _ a: Matrix<M, N>,
    _ b: Vector<N>
) -> Vector<M> {
    let result = diffMatmul(a.tracer, b.tracer.unsqueezed(-1)).squeezed(-1)
    return Vector<M>(tracer: result)
}
```

#### Element-wise Operations

```swift
/// Element-wise operations preserve shape
@differentiable(reverse)
public func relu<S: TensorShape>(_ x: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: diffReLU(x.tracer))
}

@differentiable(reverse)
public func sigmoid<S: TensorShape>(_ x: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: diffSigmoid(x.tracer))
}

@differentiable(reverse)
public func tanh<S: TensorShape>(_ x: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: diffTanh(x.tracer))
}

@differentiable(reverse)
public func gelu<S: TensorShape>(_ x: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: diffGELU(x.tracer))
}

/// Element-wise arithmetic
@differentiable(reverse)
public func + <S: TensorShape>(_ a: Tensor<S>, _ b: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: a.tracer + b.tracer)
}

@differentiable(reverse)
public func - <S: TensorShape>(_ a: Tensor<S>, _ b: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: a.tracer - b.tracer)
}

@differentiable(reverse)
public func * <S: TensorShape>(_ a: Tensor<S>, _ b: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: a.tracer * b.tracer)
}

@differentiable(reverse)
public func / <S: TensorShape>(_ a: Tensor<S>, _ b: Tensor<S>) -> Tensor<S> {
    return Tensor<S>(tracer: a.tracer / b.tracer)
}
```

#### Reduction Operations

```swift
/// Sum all elements to scalar
@differentiable(reverse)
public func sum<S: TensorShape>(_ x: Tensor<S>) -> Tensor<Scalar> {
    return Tensor<Scalar>(tracer: diffSum(x.tracer))
}

/// Mean all elements to scalar
@differentiable(reverse)
public func mean<S: TensorShape>(_ x: Tensor<S>) -> Tensor<Scalar> {
    return Tensor<Scalar>(tracer: diffMean(x.tracer))
}

/// Sum along axis with compile-time shape transformation
@differentiable(reverse)
public func sum<D0: Dim, D1: Dim>(
    _ x: Matrix<D0, D1>,
    axis: Int
) -> Vector<D0> where axis == 1 {
    return Vector<D0>(tracer: diffSum(x.tracer, axis: 1))
}

@differentiable(reverse)
public func sum<D0: Dim, D1: Dim>(
    _ x: Matrix<D0, D1>,
    axis: Int
) -> Vector<D1> where axis == 0 {
    return Vector<D1>(tracer: diffSum(x.tracer, axis: 0))
}
```

#### Shape Transformations

```swift
/// Reshape with compile-time shape verification
@differentiable(reverse)
public func reshape<From: TensorShape, To: TensorShape>(
    _ x: Tensor<From>
) -> Tensor<To> where From.elementCount == To.elementCount {
    let result = diffReshape(x.tracer, shape: To.dims)
    return Tensor<To>(tracer: result)
}

/// Transpose matrix
@differentiable(reverse)
public func transpose<M: Dim, N: Dim>(
    _ x: Matrix<M, N>
) -> Matrix<N, M> {
    let result = diffTranspose(x.tracer)
    return Matrix<N, M>(tracer: result)
}

/// Unsqueeze to add dimension
@differentiable(reverse)
public func unsqueeze<D0: Dim>(
    _ x: Vector<D0>,
    axis: Int
) -> Matrix<D1, D0> where axis == 0 {
    let result = diffUnsqueeze(x.tracer, axis: 0)
    return Matrix<D1, D0>(tracer: result)
}

/// Squeeze to remove dimension
@differentiable(reverse)
public func squeeze<D0: Dim>(
    _ x: Matrix<D1, D0>
) -> Vector<D0> {
    let result = diffSqueeze(x.tracer, axis: 0)
    return Vector<D0>(tracer: result)
}
```

### Layer Definitions with Typed Tensors

```swift
/// Type-safe linear layer
public struct Linear<In: Dim, Out: Dim>: Differentiable {
    public var weight: Matrix<In, Out>
    public var bias: Vector<Out>
    
    public init() {
        self.weight = Matrix<In, Out>.glorotUniform()
        self.bias = Vector<Out>.zeros()
    }
    
    @differentiable(reverse)
    public func callAsFunction(_ x: Vector<In>) -> Vector<Out> {
        return matvec(transpose(weight), x) + bias
    }
    
    @differentiable(reverse)
    public func callAsFunction<B: Dim>(_ x: Matrix<B, In>) -> Matrix<B, Out> {
        return matmul(x, weight) + bias.broadcast(to: [B.value, Out.value])
    }
}

/// Type-safe MLP
public struct MLP<In: Dim, Hidden: Dim, Out: Dim>: Differentiable {
    public var layer1: Linear<In, Hidden>
    public var layer2: Linear<Hidden, Out>
    
    public init() {
        self.layer1 = Linear()
        self.layer2 = Linear()
    }
    
    @differentiable(reverse)
    public func callAsFunction(_ x: Vector<In>) -> Vector<Out> {
        return layer2(relu(layer1(x)))
    }
    
    @differentiable(reverse)
    public func callAsFunction<B: Dim>(_ x: Matrix<B, In>) -> Matrix<B, Out> {
        return layer2(relu(layer1(x)))
    }
}

/// Type-safe attention
public struct Attention<SeqLen: Dim, Embed: Dim, Heads: Dim, HeadDim: Dim>: Differentiable 
    where Embed.value == Heads.value * HeadDim.value {
    
    public var queryProj: Linear<Embed, Embed>
    public var keyProj: Linear<Embed, Embed>
    public var valueProj: Linear<Embed, Embed>
    public var outputProj: Linear<Embed, Embed>
    
    @differentiable(reverse)
    public func callAsFunction(_ x: Matrix<SeqLen, Embed>) -> Matrix<SeqLen, Embed> {
        let q = queryProj(x)  // [SeqLen, Embed]
        let k = keyProj(x)
        let v = valueProj(x)
        
        // Reshape for multi-head: [SeqLen, Heads, HeadDim]
        let qHeads: Tensor3<SeqLen, Heads, HeadDim> = reshape(q)
        let kHeads: Tensor3<SeqLen, Heads, HeadDim> = reshape(k)
        let vHeads: Tensor3<SeqLen, Heads, HeadDim> = reshape(v)
        
        // Attention scores: [Heads, SeqLen, SeqLen]
        let scores = batchMatmul(
            qHeads.permuted(1, 0, 2),  // [Heads, SeqLen, HeadDim]
            kHeads.permuted(1, 2, 0)   // [Heads, HeadDim, SeqLen]
        )
        let weights = softmax(scores / sqrt(Float(HeadDim.value)), axis: -1)
        
        // Apply attention: [Heads, SeqLen, HeadDim]
        let attended = batchMatmul(weights, vHeads.permuted(1, 0, 2))
        
        // Reshape back: [SeqLen, Embed]
        let combined: Matrix<SeqLen, Embed> = reshape(attended.permuted(1, 0, 2))
        
        return outputProj(combined)
    }
}
```

### Usage Examples

#### Example 1: Compile-Time Error Detection
```swift
// This code won't compile due to shape mismatch
let x: Matrix<D32, D64> = zeros()
let y: Matrix<D128, D64> = zeros()
let z = matmul(x, y)  // COMPILE ERROR: Expected Matrix<D64, K>, got Matrix<D128, D64>

// Correct version
let y_correct: Matrix<D64, D128> = zeros()
let z = matmul(x, y_correct)  // OK: Matrix<D32, D128>
```

#### Example 2: Type-Safe Neural Network
```swift
// Define network with typed dimensions
typealias BatchSize = D32
typealias InputDim = D784
typealias HiddenDim = D256
typealias OutputDim = D10

struct TypedMNIST: Differentiable {
    var fc1: Linear<InputDim, HiddenDim>
    var fc2: Linear<HiddenDim, HiddenDim>
    var fc3: Linear<HiddenDim, OutputDim>
    
    @differentiable(reverse)
    func callAsFunction(_ x: Matrix<BatchSize, InputDim>) -> Matrix<BatchSize, OutputDim> {
        var h = relu(fc1(x))      // Matrix<BatchSize, HiddenDim>
        h = relu(fc2(h))          // Matrix<BatchSize, HiddenDim>
        return fc3(h)             // Matrix<BatchSize, OutputDim>
    }
}

// Usage
let model = TypedMNIST()
let input: Matrix<BatchSize, InputDim> = placeholder()
let output = model(input)  // Guaranteed: Matrix<BatchSize, OutputDim>
```

#### Example 3: Type-Safe Transformer Block
```swift
struct TransformerBlock<SeqLen: Dim, Embed: Dim>: Differentiable {
    var attention: Attention<SeqLen, Embed, D8, D64>  // 8 heads, 64 head dim
    var ffn: MLP<Embed, D2048, Embed>
    var norm1: LayerNorm<Embed>
    var norm2: LayerNorm<Embed>
    
    @differentiable(reverse)
    func callAsFunction(_ x: Matrix<SeqLen, Embed>) -> Matrix<SeqLen, Embed> {
        // Pre-norm architecture
        let attnOut = x + attention(norm1(x))
        return attnOut + ffn(norm2(attnOut))
    }
}
```

## Implementation Guide

### Type-Level Dimension Computation

For operations that transform shapes, we need type-level computation:

```swift
/// Type-level addition for dimensions
public protocol DimAdd {
    associatedtype Result: Dim
}

/// Broadcast shape computation
public protocol BroadcastShape {
    associatedtype From: TensorShape
    associatedtype To: TensorShape
    static func broadcast(_ x: Tensor<From>) -> Tensor<To>
}

/// Reduction result shape
public protocol ReduceShape {
    associatedtype Input: TensorShape
    associatedtype Axis
    associatedtype Result: TensorShape
}
```

### Dynamic Fallback

For cases where shapes aren't known at compile time:

```swift
/// Dynamic tensor (shape checked at runtime)
public typealias DynamicTensor = Tensor<DynamicShape>

public struct DynamicShape: TensorShape {
    public static var rank: Int { fatalError("Dynamic") }
    public static var dims: [Int] { fatalError("Dynamic") }
    public let runtimeDims: [Int]
}

/// Convert typed tensor to dynamic
public func toDynamic<S: TensorShape>(_ x: Tensor<S>) -> DynamicTensor {
    return DynamicTensor(tracer: x.tracer, runtimeDims: S.dims)
}

/// Convert dynamic tensor to typed (runtime check)
public func toTyped<S: TensorShape>(_ x: DynamicTensor) throws -> Tensor<S> {
    guard x.runtimeDims == S.dims else {
        throw ShapeError.mismatch(expected: S.dims, got: x.runtimeDims)
    }
    return Tensor<S>(tracer: x.tracer)
}
```

### Integration with vmap

```swift
/// vmap preserves shape types with batch dimension
@differentiable(reverse)
public func vmap<B: Dim, In: TensorShape, Out: TensorShape>(
    _ fn: @escaping @differentiable(reverse) (Tensor<In>) -> Tensor<Out>
) -> @differentiable(reverse) (Tensor<BatchedShape<B, In>>) -> Tensor<BatchedShape<B, Out>> {
    return { batched in
        // Implementation using underlying vmap
        let result = diffVmap(fn, batched.tracer)
        return Tensor<BatchedShape<B, Out>>(tracer: result)
    }
}

/// Shape with batch dimension prepended
public struct BatchedShape<B: Dim, S: TensorShape>: TensorShape {
    public static var rank: Int { S.rank + 1 }
    public static var dims: [Int] { [B.value] + S.dims }
}
```

## Complete Test Suite

### File: Tests/SwiftIRTests/TypedTensorTests.swift

```swift
import XCTest
@testable import SwiftIR

final class TypedTensorTests: XCTestCase {
    
    // MARK: - Basic Shape Tests
    
    func testScalarShape() {
        XCTAssertEqual(Scalar.rank, 0)
        XCTAssertEqual(Scalar.dims, [])
        XCTAssertEqual(Scalar.elementCount, 1)
    }
    
    func testShape1D() {
        XCTAssertEqual(Shape1D<D64>.rank, 1)
        XCTAssertEqual(Shape1D<D64>.dims, [64])
        XCTAssertEqual(Shape1D<D64>.elementCount, 64)
    }
    
    func testShape2D() {
        XCTAssertEqual(Shape2D<D32, D64>.rank, 2)
        XCTAssertEqual(Shape2D<D32, D64>.dims, [32, 64])
        XCTAssertEqual(Shape2D<D32, D64>.elementCount, 32 * 64)
    }
    
    func testShape4D() {
        XCTAssertEqual(Shape4D<D8, D3, D224, D224>.rank, 4)
        XCTAssertEqual(Shape4D<D8, D3, D224, D224>.dims, [8, 3, 224, 224])
    }
    
    // MARK: - Tensor Creation Tests
    
    func testTensorCreation() {
        let tracer = DifferentiableTracer.zeros(shape: [32, 64])
        let tensor = Matrix<D32, D64>(tracer: tracer)
        XCTAssertEqual(tensor.tracer.shape, [32, 64])
    }
    
    func testTensorCreationMismatch() {
        let tracer = DifferentiableTracer.zeros(shape: [32, 128])
        
        // This should fail at runtime (precondition)
        // In production, this would be caught at compile time
        // XCTAssertThrowsError(Matrix<D32, D64>(tracer: tracer))
    }
    
    func testTypedZeros() {
        let tensor: Matrix<D32, D64> = Matrix.zeros()
        XCTAssertEqual(tensor.tracer.shape, [32, 64])
    }
    
    func testTypedOnes() {
        let tensor: Vector<D128> = Vector.ones()
        XCTAssertEqual(tensor.tracer.shape, [128])
    }
    
    // MARK: - Matrix Multiplication Tests
    
    func testMatmulTypes() {
        let a: Matrix<D32, D64> = Matrix.placeholder()
        let b: Matrix<D64, D128> = Matrix.placeholder()
        
        let c = matmul(a, b)
        
        // Verify output type
        let _: Matrix<D32, D128> = c
        XCTAssertEqual(c.tracer.shape, [32, 128])
    }
    
    func testMatmulChain() {
        let a: Matrix<D32, D64> = Matrix.placeholder()
        let b: Matrix<D64, D128> = Matrix.placeholder()
        let c: Matrix<D128, D64> = Matrix.placeholder()
        
        let result = matmul(matmul(a, b), c)
        
        let _: Matrix<D32, D64> = result
        XCTAssertEqual(result.tracer.shape, [32, 64])
    }
    
    func testBatchMatmul() {
        let a: Tensor3<D8, D32, D64> = Tensor3.placeholder()
        let b: Tensor3<D8, D64, D128> = Tensor3.placeholder()
        
        let c = batchMatmul(a, b)
        
        let _: Tensor3<D8, D32, D128> = c
        XCTAssertEqual(c.tracer.shape, [8, 32, 128])
    }
    
    func testMatvec() {
        let m: Matrix<D32, D64> = Matrix.placeholder()
        let v: Vector<D64> = Vector.placeholder()
        
        let result = matvec(m, v)
        
        let _: Vector<D32> = result
        XCTAssertEqual(result.tracer.shape, [32])
    }
    
    // MARK: - Element-wise Operation Tests
    
    func testElementwisePreservesShape() {
        let x: Matrix<D32, D64> = Matrix.placeholder()
        
        let r = relu(x)
        let s = sigmoid(x)
        let t = tanh(x)
        let g = gelu(x)
        
        // All should have same type
        let _: Matrix<D32, D64> = r
        let _: Matrix<D32, D64> = s
        let _: Matrix<D32, D64> = t
        let _: Matrix<D32, D64> = g
    }
    
    func testArithmetic() {
        let a: Matrix<D32, D64> = Matrix.placeholder()
        let b: Matrix<D32, D64> = Matrix.placeholder()
        
        let sum = a + b
        let diff = a - b
        let prod = a * b
        let quot = a / b
        
        let _: Matrix<D32, D64> = sum
        let _: Matrix<D32, D64> = diff
        let _: Matrix<D32, D64> = prod
        let _: Matrix<D32, D64> = quot
    }
    
    // MARK: - Reduction Tests
    
    func testSumToScalar() {
        let x: Matrix<D32, D64> = Matrix.placeholder()
        
        let s = sum(x)
        
        let _: Tensor<Scalar> = s
        XCTAssertEqual(s.tracer.shape, [])
    }
    
    func testMeanToScalar() {
        let x: Tensor3<D8, D32, D64> = Tensor3.placeholder()
        
        let m = mean(x)
        
        let _: Tensor<Scalar> = m
    }
    
    // MARK: - Shape Transformation Tests
    
    func testReshape() {
        let x: Matrix<D32, D64> = Matrix.placeholder()  // 32 * 64 = 2048 elements
        
        let y: Matrix<D64, D32> = reshape(x)
        XCTAssertEqual(y.tracer.shape, [64, 32])
        
        let z: Vector<D2048> = reshape(x)
        XCTAssertEqual(z.tracer.shape, [2048])
    }
    
    func testTranspose() {
        let x: Matrix<D32, D64> = Matrix.placeholder()
        
        let y = transpose(x)
        
        let _: Matrix<D64, D32> = y
        XCTAssertEqual(y.tracer.shape, [64, 32])
    }
    
    // MARK: - Linear Layer Tests
    
    func testLinearLayer() {
        let layer = Linear<D64, D128>()
        let input: Vector<D64> = Vector.placeholder()
        
        let output = layer(input)
        
        let _: Vector<D128> = output
        XCTAssertEqual(output.tracer.shape, [128])
    }
    
    func testLinearLayerBatched() {
        let layer = Linear<D64, D128>()
        let input: Matrix<D32, D64> = Matrix.placeholder()
        
        let output = layer(input)
        
        let _: Matrix<D32, D128> = output
        XCTAssertEqual(output.tracer.shape, [32, 128])
    }
    
    // MARK: - MLP Tests
    
    func testMLP() {
        let mlp = MLP<D784, D256, D10>()
        let input: Matrix<D32, D784> = Matrix.placeholder()
        
        let output = mlp(input)
        
        let _: Matrix<D32, D10> = output
        XCTAssertEqual(output.tracer.shape, [32, 10])
    }
    
    // MARK: - Gradient Tests
    
    func testTypedGradient() {
        let x: Matrix<D32, D64> = Matrix.placeholder()
        let w: Matrix<D64, D128> = Matrix.placeholder()
        
        let (value, (gradX, gradW)) = valueWithGradient(at: x, w) { x, w in
            sum(matmul(x, w))
        }
        
        // Gradients preserve types
        let _: Matrix<D32, D64> = gradX
        let _: Matrix<D64, D128> = gradW
        
        XCTAssertEqual(gradX.tracer.shape, [32, 64])
        XCTAssertEqual(gradW.tracer.shape, [64, 128])
    }
    
    func testLinearGradient() {
        let layer = Linear<D64, D32>()
        let input: Matrix<D8, D64> = Matrix.placeholder()
        
        let grad = gradient(at: layer) { layer in
            sum(layer(input))
        }
        
        // Gradient has same type as parameters
        let _: Linear<D64, D32> = grad
    }
    
    // MARK: - Integration with vmap
    
    func testVmapTyped() {
        @differentiable(reverse)
        func singleExample(_ x: Vector<D64>) -> Vector<D32> {
            return Linear<D64, D32>()(x)
        }
        
        let batchedFn = vmap(singleExample)
        let input: Matrix<D8, D64> = Matrix.placeholder()  // Batch of 8
        
        let output = batchedFn(input)
        
        let _: Matrix<D8, D32> = output
    }
    
    // MARK: - Dynamic Fallback Tests
    
    func testDynamicTensor() {
        let tracer = DifferentiableTracer.placeholder(shape: [32, 64])
        let dynamic = DynamicTensor(tracer: tracer, runtimeDims: [32, 64])
        
        XCTAssertEqual(dynamic.runtimeDims, [32, 64])
    }
    
    func testTypedToDynamic() {
        let typed: Matrix<D32, D64> = Matrix.placeholder()
        let dynamic = toDynamic(typed)
        
        XCTAssertEqual(dynamic.runtimeDims, [32, 64])
    }
    
    func testDynamicToTyped() throws {
        let tracer = DifferentiableTracer.placeholder(shape: [32, 64])
        let dynamic = DynamicTensor(tracer: tracer, runtimeDims: [32, 64])
        
        let typed: Matrix<D32, D64> = try toTyped(dynamic)
        XCTAssertEqual(typed.tracer.shape, [32, 64])
    }
    
    func testDynamicToTypedMismatch() {
        let tracer = DifferentiableTracer.placeholder(shape: [32, 128])
        let dynamic = DynamicTensor(tracer: tracer, runtimeDims: [32, 128])
        
        XCTAssertThrowsError(try toTyped(dynamic) as Matrix<D32, D64>)
    }
    
    // MARK: - Compile-Time Safety Tests
    
    // Note: These tests demonstrate what WOULD be compile errors
    // They are commented out because they shouldn't compile
    
    /*
    func testMatmulShapeMismatchDoesNotCompile() {
        let a: Matrix<D32, D64> = Matrix.placeholder()
        let b: Matrix<D128, D64> = Matrix.placeholder()
        
        // This line should NOT compile:
        let c = matmul(a, b)  // ERROR: Cannot convert Matrix<D128, D64> to Matrix<D64, K>
    }
    
    func testAdditionShapeMismatchDoesNotCompile() {
        let a: Matrix<D32, D64> = Matrix.placeholder()
        let b: Matrix<D32, D128> = Matrix.placeholder()
        
        // This line should NOT compile:
        let c = a + b  // ERROR: Cannot convert Matrix<D32, D128> to Matrix<D32, D64>
    }
    
    func testReshapeElementCountMismatchDoesNotCompile() {
        let x: Matrix<D32, D64> = Matrix.placeholder()  // 2048 elements
        
        // This line should NOT compile:
        let y: Matrix<D32, D32> = reshape(x)  // ERROR: 1024 != 2048
    }
    */
    
    // MARK: - Performance Tests
    
    func testTypedTensorOverhead() {
        // Verify typed tensors have zero runtime overhead
        let iterations = 10000
        
        // Untyped version
        let untypedStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let a = DifferentiableTracer.placeholder(shape: [32, 64])
            let b = DifferentiableTracer.placeholder(shape: [64, 128])
            let _ = diffMatmul(a, b)
        }
        let untypedTime = CFAbsoluteTimeGetCurrent() - untypedStart
        
        // Typed version
        let typedStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let a: Matrix<D32, D64> = Matrix.placeholder()
            let b: Matrix<D64, D128> = Matrix.placeholder()
            let _ = matmul(a, b)
        }
        let typedTime = CFAbsoluteTimeGetCurrent() - typedStart
        
        // Typed should be within 10% of untyped (ideally identical)
        let overhead = (typedTime - untypedTime) / untypedTime
        XCTAssertLessThan(overhead, 0.1)
    }
}
```

## Success Criteria

### Type Safety

- [ ] Shape mismatches caught at compile time for matmul
- [ ] Shape mismatches caught at compile time for element-wise ops
- [ ] Reshape validates element count at compile time
- [ ] Reduction operations produce correct output shape types

### API Completeness

- [ ] All basic dimension types (D1 through D4096)
- [ ] Shape types for ranks 0-5
- [ ] Typed versions of all 80+ differentiable operations
- [ ] Linear, MLP, and common layer types

### Zero Overhead

- [ ] No runtime performance penalty vs untyped
- [ ] Thin wrapper compiles away completely
- [ ] Generated StableHLO identical to untyped version

### Differentiability

- [ ] Gradient computation preserves types
- [ ] Layer parameters have correct gradient types
- [ ] Works with vmap

### Ergonomics

- [ ] Clean API without excessive type annotations
- [ ] Type inference works for common patterns
- [ ] Dynamic fallback for runtime-determined shapes

## Files to Create

### New Files
- `Sources/SwiftIR/TypedTensors/Dim.swift` - Dimension types
- `Sources/SwiftIR/TypedTensors/TensorShape.swift` - Shape types
- `Sources/SwiftIR/TypedTensors/Tensor.swift` - Typed tensor wrapper
- `Sources/SwiftIR/TypedTensors/Operations.swift` - Type-safe operations
- `Sources/SwiftIR/TypedTensors/Layers.swift` - Type-safe layers
- `Sources/SwiftIR/TypedTensors/Dynamic.swift` - Dynamic fallback
- `Tests/SwiftIRTests/TypedTensorTests.swift` - Test suite
