# VMAP - Automatic Vectorization Specification

## Overview

`vmap` (vectorized map) automatically transforms a function that operates on single examples into one that operates on batches. This is essential for efficient ML training and inference.

## JAX Reference

```python
# JAX vmap usage
def single_example(x, w):
    return jnp.dot(x, w)  # [features] @ [features, classes] -> [classes]

batched = jax.vmap(single_example, in_axes=(0, None))
# Now: [batch, features] @ [features, classes] -> [batch, classes]
```

## SwiftIR API Specification

### Primary API

```swift
/// Vectorize a function over specified input axes
/// - Parameters:
///   - fn: The function to vectorize
///   - inAxes: Which axis to batch over for each input (nil = broadcast)
///   - outAxes: Which axis the batch dimension appears in output (default: 0)
/// - Returns: A vectorized version of the function
public func vmap<T, U>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    inAxes: VmapAxes = .init(0),
    outAxes: Int = 0
) -> @differentiable(reverse) (T) -> U
```

### VmapAxes Type

```swift
/// Specifies batch axes for vmap inputs
public struct VmapAxes {
    public let axes: [Int?]
    
    /// Single axis for single input
    public init(_ axis: Int?) {
        self.axes = [axis]
    }
    
    /// Multiple axes for tuple inputs
    public init(_ axes: Int?...) {
        self.axes = axes
    }
    
    /// Array initializer
    public init(axes: [Int?]) {
        self.axes = axes
    }
}
```

### Usage Examples

#### Example 1: Simple Batching
```swift
@differentiable(reverse)
func processOne(_ x: DifferentiableTracer) -> DifferentiableTracer {
    return diffReLU(x)  // [features] -> [features]
}

let batchedProcess = vmap(processOne)
// Input: [batch, features] -> Output: [batch, features]
```

#### Example 2: Matrix Multiplication with Broadcasting
```swift
@differentiable(reverse)
func forward(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
    return diffMatmul(x, w)  // [in] @ [in, out] -> [out]
}

let batchedForward = vmap(forward, inAxes: .init(0, nil))
// x: [batch, in], w: [in, out] -> [batch, out]
// w is broadcast (shared) across batch
```

#### Example 3: Per-Example Gradients
```swift
@differentiable(reverse)
func loss(_ x: DifferentiableTracer, _ y: DifferentiableTracer) -> DifferentiableTracer {
    let pred = model(x)
    return diffMSELoss(pred, y)  // scalar loss per example
}

// Get gradient for each example separately (useful for differential privacy)
let perExampleGrad = vmap(
    { (x, y) in gradient(at: params) { p in loss(x, y) } },
    inAxes: .init(0, 0)
)
```

#### Example 4: Nested vmap
```swift
@differentiable(reverse)
func elementOp(_ a: DifferentiableTracer, _ b: DifferentiableTracer) -> DifferentiableTracer {
    return a * b  // scalar * scalar -> scalar
}

// First vmap: batch over inner dimension
let rowOp = vmap(elementOp, inAxes: .init(0, 0))  
// [cols] * [cols] -> [cols]

// Second vmap: batch over outer dimension  
let matrixOp = vmap(rowOp, inAxes: .init(0, 0))
// [rows, cols] * [rows, cols] -> [rows, cols]
```

## Implementation Guide

### Core Data Structures

```swift
/// Tracks batch dimensions through tracing
public struct BatchTracer {
    /// The underlying tracer
    public var tracer: DifferentiableTracer
    
    /// Which dimension is the batch dimension (nil if not batched)
    public var batchAxis: Int?
    
    /// The batch size
    public var batchSize: Int?
}

/// Context for vmap transformation
public class VmapContext {
    /// Current batch size being traced
    public var batchSize: Int
    
    /// Mapping from original tracer IDs to batched tracers
    public var tracerMapping: [Int: BatchTracer] = [:]
    
    /// Stack for nested vmaps
    public static var contextStack: [VmapContext] = []
    
    public static var current: VmapContext? {
        contextStack.last
    }
}
```

### Implementation Steps

#### Step 1: Batch Dimension Injection

```swift
/// Add batch dimension to input tracer
func batchInput(
    _ tracer: DifferentiableTracer,
    axis: Int?,
    batchSize: Int
) -> BatchTracer {
    guard let axis = axis else {
        // nil axis = broadcast this input
        return BatchTracer(tracer: tracer, batchAxis: nil, batchSize: nil)
    }
    
    // Insert batch dimension at specified axis
    var newShape = tracer.shape
    newShape.insert(batchSize, at: axis)
    
    let batchedTracer = DifferentiableTracer(
        shape: newShape,
        dtype: tracer.dtype,
        operation: .vmapBatchedInput(original: tracer, axis: axis)
    )
    
    return BatchTracer(tracer: batchedTracer, batchAxis: axis, batchSize: batchSize)
}
```

#### Step 2: Operation Batching Rules

Each operation needs a batching rule. Here's the pattern:

```swift
/// Protocol for operations that can be batched
protocol BatchableOperation {
    /// Transform operation to handle batch dimension
    static func batchingRule(
        inputs: [BatchTracer],
        originalOp: Operation
    ) -> BatchTracer
}

/// Example: Element-wise operations (trivial batching)
extension ElementWiseOps: BatchableOperation {
    static func batchingRule(
        inputs: [BatchTracer],
        originalOp: Operation
    ) -> BatchTracer {
        // Element-wise ops just work with extra dimension
        let batchAxis = inputs.compactMap { $0.batchAxis }.first
        let batchSize = inputs.compactMap { $0.batchSize }.first
        
        // Broadcast non-batched inputs
        let broadcastedInputs = inputs.map { input -> DifferentiableTracer in
            if input.batchAxis == nil, let batchAxis = batchAxis, let batchSize = batchSize {
                return broadcastToBatch(input.tracer, axis: batchAxis, size: batchSize)
            }
            return input.tracer
        }
        
        // Apply original operation
        let result = applyOp(originalOp, inputs: broadcastedInputs)
        
        return BatchTracer(tracer: result, batchAxis: batchAxis, batchSize: batchSize)
    }
}

/// Example: Reduction operations
extension ReductionOps: BatchableOperation {
    static func batchingRule(
        inputs: [BatchTracer],
        originalOp: Operation
    ) -> BatchTracer {
        let input = inputs[0]
        guard let batchAxis = input.batchAxis else {
            // Not batched, apply normally
            let result = applyOp(originalOp, inputs: [input.tracer])
            return BatchTracer(tracer: result, batchAxis: nil, batchSize: nil)
        }
        
        // Adjust reduction axis to account for batch dimension
        var reduceAxis = originalOp.axis
        if reduceAxis >= batchAxis {
            reduceAxis += 1
        }
        
        let adjustedOp = originalOp.withAxis(reduceAxis)
        let result = applyOp(adjustedOp, inputs: [input.tracer])
        
        // Batch dimension preserved (shifted if needed)
        let newBatchAxis = reduceAxis > batchAxis ? batchAxis : batchAxis - 1
        return BatchTracer(tracer: result, batchAxis: newBatchAxis, batchSize: input.batchSize)
    }
}

/// Example: Matmul batching
extension MatmulOp: BatchableOperation {
    static func batchingRule(
        inputs: [BatchTracer],
        originalOp: Operation
    ) -> BatchTracer {
        let (a, b) = (inputs[0], inputs[1])
        
        switch (a.batchAxis, b.batchAxis) {
        case (nil, nil):
            // Neither batched - normal matmul
            let result = diffMatmul(a.tracer, b.tracer)
            return BatchTracer(tracer: result, batchAxis: nil, batchSize: nil)
            
        case (let axis?, nil):
            // Left batched: [batch, m, k] @ [k, n] -> [batch, m, n]
            // Use batched matmul (broadcasts right matrix)
            let result = diffBatchedMatmul(a.tracer, b.tracer)
            return BatchTracer(tracer: result, batchAxis: axis, batchSize: a.batchSize)
            
        case (nil, let axis?):
            // Right batched: [m, k] @ [batch, k, n] -> [batch, m, n]
            // Transpose, batched matmul, transpose back
            let result = diffBatchedMatmulRight(a.tracer, b.tracer)
            return BatchTracer(tracer: result, batchAxis: 0, batchSize: b.batchSize)
            
        case (let axisA?, let axisB?):
            // Both batched - must have same batch size
            assert(a.batchSize == b.batchSize, "Batch sizes must match")
            let result = diffBatchedMatmul(a.tracer, b.tracer)
            return BatchTracer(tracer: result, batchAxis: 0, batchSize: a.batchSize)
        }
    }
}
```

#### Step 3: Main vmap Implementation

```swift
public func vmap<T, U>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    inAxes: VmapAxes = .init(0),
    outAxes: Int = 0
) -> @differentiable(reverse) (T) -> U {
    
    return { (input: T) -> U in
        // 1. Create vmap context
        let batchSize = inferBatchSize(input, axes: inAxes)
        let context = VmapContext(batchSize: batchSize)
        VmapContext.contextStack.append(context)
        defer { VmapContext.contextStack.removeLast() }
        
        // 2. Create batched input tracers
        let batchedInputs = createBatchedInputs(input, axes: inAxes, batchSize: batchSize)
        
        // 3. Trace the function (operations will use batching rules)
        let batchedOutput = fn(batchedInputs as! T)
        
        // 4. Move batch axis to output position
        let result = moveBatchAxis(batchedOutput, toAxis: outAxes)
        
        return result as! U
    }
}

/// Infer batch size from inputs
func inferBatchSize<T>(_ input: T, axes: VmapAxes) -> Int {
    // Extract tracers from input
    let tracers = extractTracers(input)
    
    for (tracer, axis) in zip(tracers, axes.axes) {
        if let axis = axis {
            return tracer.shape[axis]
        }
    }
    
    fatalError("At least one input must have a batch axis")
}
```

#### Step 4: StableHLO Generation

vmap should generate efficient batched StableHLO:

```swift
/// Generate StableHLO for batched operation
func generateBatchedStableHLO(_ op: Operation, batchInfo: BatchInfo) -> String {
    switch op {
    case .matmul(let a, let b):
        // Use stablehlo.dot_general for batched matmul
        return """
        %\(op.resultId) = stablehlo.dot_general %\(a.id), %\(b.id),
            batching_dims = [0] x [0],
            contracting_dims = [\(a.rank-1)] x [\(b.rank-2)]
            : (tensor<\(a.shape.hlotype)>, tensor<\(b.shape.hlotype)>) -> tensor<\(op.resultShape.hlotype)>
        """
        
    case .broadcast(let x, let targetShape):
        // Use stablehlo.broadcast_in_dim
        return """
        %\(op.resultId) = stablehlo.broadcast_in_dim %\(x.id),
            dims = [\(op.broadcastDims.map(String.init).joined(separator: ", "))]
            : (tensor<\(x.shape.hlotype)>) -> tensor<\(targetShape.hlotype)>
        """
        
    default:
        // Element-wise ops work naturally with batched tensors
        return generateStandardStableHLO(op)
    }
}
```

### Gradient Support

vmap must compose with differentiation:

```swift
/// VJP rule for vmap
@derivative(of: vmap)
func vmapVJP<T, U>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    inAxes: VmapAxes,
    outAxes: Int
) -> (value: @differentiable(reverse) (T) -> U, 
      pullback: (@escaping (T) -> U) -> @differentiable(reverse) (T) -> U) {
    
    let vmapped = vmap(fn, inAxes: inAxes, outAxes: outAxes)
    
    return (vmapped, { gradFn in
        // The gradient of a vmapped function is the vmap of the gradient
        vmap(gradFn, inAxes: .init(outAxes), outAxes: inAxes.axes[0] ?? 0)
    })
}
```

## Complete Test Suite

### File: Tests/SwiftIRTests/VmapTests.swift

```swift
import XCTest
@testable import SwiftIR

final class VmapTests: XCTestCase {
    
    // MARK: - Basic Functionality Tests
    
    func testVmapIdentity() {
        // vmap over identity function
        @differentiable(reverse)
        func identity(_ x: DifferentiableTracer) -> DifferentiableTracer { x }
        
        let batched = vmap(identity)
        
        let input = DifferentiableTracer.placeholder(shape: [4, 8])  // [batch, features]
        let output = batched(input)
        
        XCTAssertEqual(output.shape, [4, 8])
    }
    
    func testVmapElementwise() {
        // vmap over element-wise operation
        @differentiable(reverse)
        func relu(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        let batchedRelu = vmap(relu)
        
        let input = DifferentiableTracer.placeholder(shape: [32, 64])
        let output = batchedRelu(input)
        
        XCTAssertEqual(output.shape, [32, 64])
    }
    
    func testVmapReduction() {
        // vmap over reduction should reduce each batch element
        @differentiable(reverse)
        func sumAll(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSum(x)  // [features] -> scalar
        }
        
        let batchedSum = vmap(sumAll)
        
        let input = DifferentiableTracer.placeholder(shape: [16, 32])  // [batch, features]
        let output = batchedSum(input)
        
        XCTAssertEqual(output.shape, [16])  // [batch] of scalars
    }
    
    func testVmapMatmul() {
        // vmap over matmul with weight broadcasting
        @differentiable(reverse)
        func linear(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(x, w)  // [in] @ [in, out] -> [out]
        }
        
        let batchedLinear = vmap(linear, inAxes: .init(0, nil))
        
        let x = DifferentiableTracer.placeholder(shape: [8, 64])   // [batch, in]
        let w = DifferentiableTracer.placeholder(shape: [64, 32])  // [in, out]
        let output = batchedLinear(x, w)
        
        XCTAssertEqual(output.shape, [8, 32])  // [batch, out]
    }
    
    func testVmapBothInputsBatched() {
        // vmap with both inputs batched
        @differentiable(reverse)
        func dot(_ a: DifferentiableTracer, _ b: DifferentiableTracer) -> DifferentiableTracer {
            return diffSum(a * b)  // [features] * [features] -> scalar
        }
        
        let batchedDot = vmap(dot, inAxes: .init(0, 0))
        
        let a = DifferentiableTracer.placeholder(shape: [16, 64])
        let b = DifferentiableTracer.placeholder(shape: [16, 64])
        let output = batchedDot(a, b)
        
        XCTAssertEqual(output.shape, [16])  // [batch] of scalars
    }
    
    // MARK: - Axis Specification Tests
    
    func testVmapCustomInAxis() {
        // Batch dimension not at position 0
        @differentiable(reverse)
        func process(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        let batchedProcess = vmap(process, inAxes: .init(1))
        
        let input = DifferentiableTracer.placeholder(shape: [64, 8, 32])  // [features, batch, more]
        let output = batchedProcess(input)
        
        // Output should have batch at axis 1
        XCTAssertEqual(output.shape, [64, 8, 32])
    }
    
    func testVmapCustomOutAxis() {
        // Move batch dimension to different output position
        @differentiable(reverse)
        func process(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        let batchedProcess = vmap(process, inAxes: .init(0), outAxes: 1)
        
        let input = DifferentiableTracer.placeholder(shape: [8, 64])  // [batch, features]
        let output = batchedProcess(input)
        
        XCTAssertEqual(output.shape, [64, 8])  // [features, batch]
    }
    
    // MARK: - Nested vmap Tests
    
    func testNestedVmap() {
        // Double vmap for matrix operations
        @differentiable(reverse)
        func scalarOp(_ a: DifferentiableTracer, _ b: DifferentiableTracer) -> DifferentiableTracer {
            return a * b + a
        }
        
        let rowOp = vmap(scalarOp, inAxes: .init(0, 0))
        let matrixOp = vmap(rowOp, inAxes: .init(0, 0))
        
        let a = DifferentiableTracer.placeholder(shape: [4, 8])
        let b = DifferentiableTracer.placeholder(shape: [4, 8])
        let output = matrixOp(a, b)
        
        XCTAssertEqual(output.shape, [4, 8])
    }
    
    func testTripleNestedVmap() {
        // Triple vmap for 3D tensor operations
        @differentiable(reverse)
        func elementOp(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffTanh(x)
        }
        
        let level1 = vmap(elementOp)
        let level2 = vmap(level1)
        let level3 = vmap(level2)
        
        let input = DifferentiableTracer.placeholder(shape: [2, 3, 4])
        let output = level3(input)
        
        XCTAssertEqual(output.shape, [2, 3, 4])
    }
    
    // MARK: - Gradient Tests
    
    func testVmapGradientSimple() {
        // Gradient through vmapped function
        @differentiable(reverse)
        func square(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return x * x
        }
        
        let batchedSquare = vmap(square)
        
        let x = DifferentiableTracer.placeholder(shape: [4, 8])
        let (value, grad) = valueWithGradient(at: x) { x in
            diffSum(batchedSquare(x))
        }
        
        // d/dx sum(x^2) = 2x
        XCTAssertEqual(grad.shape, [4, 8])
        // Numerical check would verify grad ≈ 2 * x
    }
    
    func testVmapGradientMatmul() {
        // Gradient through batched matmul
        @differentiable(reverse)
        func linear(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(x, w)
        }
        
        let batchedLinear = vmap(linear, inAxes: .init(0, nil))
        
        let x = DifferentiableTracer.placeholder(shape: [8, 64])
        let w = DifferentiableTracer.placeholder(shape: [64, 32])
        
        let (value, (gradX, gradW)) = valueWithGradient(at: x, w) { x, w in
            diffSum(batchedLinear(x, w))
        }
        
        XCTAssertEqual(gradX.shape, [8, 64])
        XCTAssertEqual(gradW.shape, [64, 32])
    }
    
    func testVmapGradientNumerical() {
        // Numerical gradient verification
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSigmoid(x)
        }
        
        let batchedF = vmap(f)
        
        let xValues: [Float] = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
        let x = DifferentiableTracer.constant(xValues, shape: [2, 4])
        
        let (_, grad) = valueWithGradient(at: x) { x in
            diffSum(batchedF(x))
        }
        
        // Compile and execute
        let (mlir, _) = compileGradientForPJRT(grad)
        let numericalGrad = computeNumericalGradient(batchedF, at: xValues, epsilon: 1e-5)
        
        // Compare
        let analyticalGrad = executePJRT(mlir, inputs: xValues)
        XCTAssertEqual(analyticalGrad, numericalGrad, accuracy: 1e-4)
    }
    
    // MARK: - Edge Cases
    
    func testVmapEmptyBatch() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        let batched = vmap(f)
        let input = DifferentiableTracer.placeholder(shape: [0, 8])  // Empty batch
        let output = batched(input)
        
        XCTAssertEqual(output.shape, [0, 8])
    }
    
    func testVmapSingletonBatch() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffReLU(x)
        }
        
        let batched = vmap(f)
        let input = DifferentiableTracer.placeholder(shape: [1, 8])  // Batch size 1
        let output = batched(input)
        
        XCTAssertEqual(output.shape, [1, 8])
    }
    
    func testVmapLargeBatch() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(x, DifferentiableTracer.placeholder(shape: [64, 64]))
        }
        
        let batched = vmap(f, inAxes: .init(0))
        let input = DifferentiableTracer.placeholder(shape: [1024, 64])  // Large batch
        let output = batched(input)
        
        XCTAssertEqual(output.shape, [1024, 64])
    }
    
    // MARK: - Integration Tests
    
    func testVmapWithWhileLoop() {
        // vmap composed with while loop
        @differentiable(reverse)
        func iterate(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffWhileLoop(
                initialState: x,
                condition: { state, i in i < 10 },
                body: { state in diffReLU(state * 0.9) }
            )
        }
        
        let batchedIterate = vmap(iterate)
        
        let input = DifferentiableTracer.placeholder(shape: [4, 8])
        let output = batchedIterate(input)
        
        XCTAssertEqual(output.shape, [4, 8])
    }
    
    func testVmapWithScan() {
        // vmap composed with scan (once scan is implemented)
        // This test validates that vmap and scan compose correctly
    }
    
    func testVmapMLPForward() {
        // Full MLP forward pass with vmap
        let hiddenSize = 128
        let outputSize = 10
        
        @differentiable(reverse)
        func mlpSingle(
            _ x: DifferentiableTracer,
            _ w1: DifferentiableTracer,
            _ w2: DifferentiableTracer
        ) -> DifferentiableTracer {
            let h = diffReLU(diffMatmul(x, w1))
            return diffMatmul(h, w2)
        }
        
        let batchedMLP = vmap(mlpSingle, inAxes: .init(0, nil, nil))
        
        let x = DifferentiableTracer.placeholder(shape: [32, 64])
        let w1 = DifferentiableTracer.placeholder(shape: [64, hiddenSize])
        let w2 = DifferentiableTracer.placeholder(shape: [hiddenSize, outputSize])
        
        let output = batchedMLP(x, w1, w2)
        
        XCTAssertEqual(output.shape, [32, outputSize])
    }
    
    // MARK: - Performance Tests
    
    func testVmapCompilationTime() {
        @differentiable(reverse)
        func complex(_ x: DifferentiableTracer) -> DifferentiableTracer {
            var y = x
            for _ in 0..<10 {
                y = diffReLU(diffMatmul(y, DifferentiableTracer.placeholder(shape: [64, 64])))
            }
            return y
        }
        
        let batched = vmap(complex, inAxes: .init(0))
        
        let input = DifferentiableTracer.placeholder(shape: [32, 64])
        
        let start = CFAbsoluteTimeGetCurrent()
        let _ = batched(input)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Compilation should be fast (< 1 second)
        XCTAssertLessThan(elapsed, 1.0)
    }
    
    func testVmapVsManualBatching() {
        // Compare vmap to manually batched code
        let batchSize = 64
        let features = 128
        
        // Manual batching
        @differentiable(reverse)
        func manualBatched(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
            // Directly write batched matmul
            return diffMatmul(x, w)  // [batch, in] @ [in, out] -> [batch, out]
        }
        
        // vmap batching
        @differentiable(reverse)
        func single(_ x: DifferentiableTracer, _ w: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(x, w)  // [in] @ [in, out] -> [out]
        }
        let vmapBatched = vmap(single, inAxes: .init(0, nil))
        
        let x = DifferentiableTracer.placeholder(shape: [batchSize, features])
        let w = DifferentiableTracer.placeholder(shape: [features, features])
        
        // Both should produce same shape
        let manual = manualBatched(x, w)
        let vmapped = vmapBatched(x, w)
        
        XCTAssertEqual(manual.shape, vmapped.shape)
        
        // Generated MLIR should be equivalent
        // (Performance should be within 5% of manual)
    }
    
    // MARK: - StableHLO Generation Tests
    
    func testVmapStableHLOGeneration() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(x, DifferentiableTracer.placeholder(shape: [64, 32]))
        }
        
        let batched = vmap(f, inAxes: .init(0))
        let input = DifferentiableTracer.placeholder(shape: [8, 64])
        let output = batched(input)
        
        let mlir = generateStableHLO(output)
        
        // Should contain batched matmul (dot_general with batching_dims)
        XCTAssertTrue(mlir.contains("stablehlo.dot_general"))
        XCTAssertTrue(mlir.contains("batching_dims"))
    }
}
```

## Success Criteria

### Functional Requirements

- [ ] Basic vmap over element-wise operations
- [ ] vmap over reductions (sum, mean, max, min)
- [ ] vmap over matmul with various axis configurations
- [ ] vmap with `inAxes` specifying different batch positions
- [ ] vmap with `outAxes` moving batch dimension
- [ ] vmap with `nil` axes for broadcasting
- [ ] Nested vmap (2+ levels)
- [ ] vmap gradient computation (VJP)
- [ ] vmap + gradient numerical correctness
- [ ] vmap + diffWhileLoop composition
- [ ] vmap with all 80+ differentiable ops

### Performance Requirements

- [ ] Compilation time < 1s for typical networks
- [ ] Runtime overhead < 5% vs manual batching
- [ ] Memory usage equivalent to manual batching
- [ ] Generated StableHLO uses efficient batched ops (dot_general)

### Edge Cases

- [ ] Empty batch (size 0)
- [ ] Singleton batch (size 1)
- [ ] Large batch (1000+)
- [ ] High-dimensional tensors (4D+)
- [ ] Mixed batched/unbatched inputs

### Documentation

- [ ] API documentation with examples
- [ ] Migration guide from manual batching
- [ ] Performance tuning guide

## Error Messages

Implement clear error messages:

```swift
// Batch size mismatch
"vmap: Batch sizes don't match. Input 0 has batch size 32 at axis 0, but input 1 has batch size 16 at axis 0."

// No batch axis specified
"vmap: At least one input must have a batch axis. All axes are nil."

// Invalid axis
"vmap: Invalid batch axis 3 for input with shape [2, 4]. Axis must be in range 0..<2."

// Shape mismatch after batching
"vmap: Shape mismatch in operation 'matmul'. After batching, got shapes [8, 32, 64] and [64, 128] which are incompatible."
```

## Files to Create/Modify

### New Files
- `Sources/SwiftIR/SymbolicAD/Vmap.swift` - Main implementation
- `Sources/SwiftIR/SymbolicAD/BatchingRules.swift` - Per-operation batching rules
- `Tests/SwiftIRTests/VmapTests.swift` - Test suite

### Modified Files
- `Sources/SwiftIR/SymbolicAD/DifferentiableTracer.swift` - Add batch tracking
- `Sources/SwiftIR/SymbolicAD/BackendCompilation.swift` - Batched StableHLO generation
- `Sources/SwiftIR/SymbolicAD/ADIntegration.swift` - Register batching rules for all ops
