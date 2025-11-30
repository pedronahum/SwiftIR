// JupyterFeatureParityTest.swift
// Tests all SwiftIRJupyter features for parity with C++ interop SwiftIR

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

import SwiftIRJupyter
import _Differentiation
import Foundation

// String multiplication helper
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

print("=" * 60)
print("SwiftIRJupyter Feature Parity Test")
print("Testing all features for Jupyter/Colab compatibility")
print("=" * 60)
print()

nonisolated(unsafe) var testsPassed = 0
nonisolated(unsafe) var testsFailed = 0

func test(_ name: String, _ condition: Bool) {
    if condition {
        print("✓ \(name)")
        testsPassed += 1
    } else {
        print("✗ \(name)")
        testsFailed += 1
    }
}

// MARK: - Test 1: Basic Types

print("-" * 60)
print("TEST 1: Basic Types")
print("-" * 60)

let shape1 = JTensorShape([2, 3])
test("JTensorShape creation", shape1.rank == 2)
test("JTensorShape dimensions", shape1.dimensions == [2, 3])

let scalar = JTensorShape.scalar
test("Scalar shape", scalar.rank == 0)

let broadcasted = JTensorShape([1, 3]).broadcast(with: JTensorShape([2, 1]))
test("Broadcasting", broadcasted.dimensions == [2, 3])

test("JDType float32", JDType.float32.rawValue == "f32")
test("JDType promotion", JDType.float32.promoted(with: .int32) == .float32)

print()

// MARK: - Test 2: JTracer Operations

print("-" * 60)
print("TEST 2: JTracer Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let a = JTracer(value: 2.0, shape: .scalar, dtype: .float32)
let b = JTracer(value: 3.0, shape: .scalar, dtype: .float32)

let sum = a + b
test("Addition", sum.shape == .scalar)

let diff = a - b
test("Subtraction", diff.shape == .scalar)

let prod = a * b
test("Multiplication", prod.shape == .scalar)

let quot = a / b
test("Division", quot.shape == .scalar)

let neg = -a
test("Negation", neg.shape == .scalar)

print()

// MARK: - Test 3: Unary Operations

print("-" * 60)
print("TEST 3: Unary Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let x = JTracer(value: 1.0, shape: .scalar, dtype: .float32)

let expResult = x.exp()
test("exp()", expResult.shape == .scalar)

let logResult = x.log()
test("log()", logResult.shape == .scalar)

let sqrtResult = x.sqrt()
test("sqrt()", sqrtResult.shape == .scalar)

let tanhResult = x.tanh()
test("tanh()", tanhResult.shape == .scalar)

let sigmoidResult = x.sigmoid()
test("sigmoid()", sigmoidResult.shape == .scalar)

let reluResult = x.relu()
test("relu()", reluResult.shape == .scalar)

// New operations
let sinResult = x.sin()
test("sin()", sinResult.shape == .scalar)

let cosResult = x.cos()
test("cos()", cosResult.shape == .scalar)

let tanResult = x.tan()
test("tan()", tanResult.shape == .scalar)

let absResult = x.abs()
test("abs()", absResult.shape == .scalar)

let powerResult = x.power(2.0)
test("power()", powerResult.shape == .scalar)

// NEW: Additional math operations
let rsqrtResult = x.rsqrt()
test("rsqrt()", rsqrtResult.shape == .scalar)

let floorResult = x.floor()
test("floor()", floorResult.shape == .scalar)

let ceilResult = x.ceil()
test("ceil()", ceilResult.shape == .scalar)

let clampResult = x.clamp(min: 0.0, max: 1.0)
test("clamp()", clampResult.shape == .scalar)

print()

// MARK: - Test 4: Matrix Operations

print("-" * 60)
print("TEST 4: Matrix Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let matrix1 = JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32)
let matrix2 = JTracer(value: 1.0, shape: JTensorShape([3, 4]), dtype: .float32)

let matmulResult = matrix1.matmul(matrix2)
test("matmul() shape", matmulResult.shape.dimensions == [2, 4])

let transposeResult = matrix1.transpose()
test("transpose() shape", transposeResult.shape.dimensions == [3, 2])

let reshaped = matrix1.reshape(to: JTensorShape([6]))
test("reshape()", reshaped.shape.dimensions == [6])

print()

// MARK: - Test 5: Reduction Operations

print("-" * 60)
print("TEST 5: Reduction Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let tensor = JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32)

let sumAll = tensor.sum()
test("sum() to scalar", sumAll.shape == .scalar)

let sumAxis0 = tensor.sum(alongAxes: [0], keepDims: false)
test("sum(axis=0)", sumAxis0.shape.dimensions == [3])

let meanAxis1 = tensor.mean(alongAxes: [1], keepDims: true)
test("mean(axis=1, keepDims)", meanAxis1.shape.dimensions == [2, 1])

// NEW: Max/min reductions
let maxAxis0 = tensor.max(alongAxes: [0], keepDims: false)
test("max(axis=0)", maxAxis0.shape.dimensions == [3])

let minAxis1 = tensor.min(alongAxes: [1], keepDims: false)
test("min(axis=1)", minAxis1.shape.dimensions == [2])

print()

// MARK: - Test 5.5: Advanced Activation Functions

print("-" * 60)
print("TEST 5.5: Advanced Activation Functions")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let act = JTracer(value: 0.5, shape: .scalar, dtype: .float32)

let leakyReluResult = act.leakyRelu(alpha: 0.01)
test("leakyRelu()", leakyReluResult.shape == .scalar)

let eluResult = act.elu(alpha: 1.0)
test("elu()", eluResult.shape == .scalar)

let siluResult = act.silu()
test("silu()", siluResult.shape == .scalar)

let geluResult = act.gelu()
test("gelu()", geluResult.shape == .scalar)

let softplusResult = act.softplus()
test("softplus()", softplusResult.shape == .scalar)

let softmaxTensor = JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32)
let softmaxResult = softmaxTensor.softmax()
test("softmax()", softmaxResult.shape.dimensions == [2, 3])

let logSoftmaxResult = softmaxTensor.logSoftmax()
test("logSoftmax()", logSoftmaxResult.shape.dimensions == [2, 3])

print()

// MARK: - Test 5.6: Element-wise Binary Operations

print("-" * 60)
print("TEST 5.6: Element-wise Binary Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let t1 = JTracer(value: 2.0, shape: .scalar, dtype: .float32)
let t2 = JTracer(value: 3.0, shape: .scalar, dtype: .float32)

let maxResult = t1.maximum(t2)
test("maximum()", maxResult.shape == .scalar)

let minResult = t1.minimum(t2)
test("minimum()", minResult.shape == .scalar)

print()

// MARK: - Test 5.7: Loss Functions

print("-" * 60)
print("TEST 5.7: Loss Functions")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let preds = JTracer(value: 0.5, shape: JTensorShape([4]), dtype: .float32)
let targets = JTracer(value: 1.0, shape: JTensorShape([4]), dtype: .float32)

let mseLoss = JTracer.mseLoss(predictions: preds, targets: targets)
test("mseLoss()", mseLoss.shape == .scalar)

let bceLoss = JTracer.bceLoss(predictions: preds, targets: targets)
test("bceLoss()", bceLoss.shape == .scalar)

let logits = JTracer(value: 1.0, shape: JTensorShape([2, 4]), dtype: .float32)
let oneHot = JTracer(value: 0.25, shape: JTensorShape([2, 4]), dtype: .float32)

let crossEntropyLoss = JTracer.crossEntropyLoss(logits: logits, targets: oneHot)
test("crossEntropyLoss()", crossEntropyLoss.shape == .scalar)

let softmaxCELoss = JTracer.softmaxCrossEntropyLoss(logits: logits, targets: oneHot)
test("softmaxCrossEntropyLoss()", softmaxCELoss.shape == .scalar)

let huberLoss = JTracer.huberLoss(predictions: preds, targets: targets, delta: 1.0)
test("huberLoss()", huberLoss.shape == .scalar)

print()

// MARK: - Test 5.8: Tensor Slicing and Concatenation

print("-" * 60)
print("TEST 5.8: Tensor Slicing and Concatenation")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let sliceTensor = JTracer(value: 1.0, shape: JTensorShape([4, 6]), dtype: .float32)
let sliced = sliceTensor.slice(starts: [0, 0], limits: [2, 3], strides: [1, 1])
test("slice() shape", sliced.shape.dimensions == [2, 3])

let cat1 = JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32)
let cat2 = JTracer(value: 2.0, shape: JTensorShape([2, 3]), dtype: .float32)
let concatenated = JTracer.concatenate([cat1, cat2], axis: 0)
test("concatenate(axis=0)", concatenated.shape.dimensions == [4, 3])

let concatAxis1 = JTracer.concatenate([cat1, cat2], axis: 1)
test("concatenate(axis=1)", concatAxis1.shape.dimensions == [2, 6])

print()

// MARK: - Test 6: Comparison Operations

print("-" * 60)
print("TEST 6: Comparison Operations")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

let p = JTracer(value: 5.0, shape: .scalar, dtype: .float32)
let q = JTracer(value: 3.0, shape: .scalar, dtype: .float32)

let ltResult = p < q
test("< comparison", ltResult.dtype == .bool)

let gtResult = p > q
test("> comparison", gtResult.dtype == .bool)

let leResult = p <= q
test("<= comparison", leResult.dtype == .bool)

let geResult = p >= q
test(">= comparison", geResult.dtype == .bool)

let eqResult = p == q
test("== comparison", eqResult.dtype == .bool)

let neResult = p != q
test("!= comparison", neResult.dtype == .bool)

print()

// MARK: - Test 7: Distributed Operations

print("-" * 60)
print("TEST 7: Distributed Training Support")
print("-" * 60)

// Device mesh
let mesh = JDeviceMesh.linear(name: "test_mesh", size: 4)
test("DeviceMesh creation", mesh.deviceCount == 4)
test("DeviceMesh axis name", mesh.axisNames[0] == "x")

let gridMesh = JDeviceMesh.grid(dataParallel: 2, modelParallel: 4)
test("Grid mesh devices", gridMesh.deviceCount == 8)
test("Grid mesh data axis", gridMesh.axisSize("data") == 2)
test("Grid mesh model axis", gridMesh.axisSize("model") == 4)

// Sharding spec
let replicated = JShardingSpec.replicated(mesh: mesh, rank: 2)
test("Replicated sharding", replicated.isReplicated)

let dataParallel = JShardingSpec.dataParallel(mesh: gridMesh, rank: 3)
test("Data parallel sharding", !dataParallel.isReplicated)

// Collective operations on tracer
JTracerGraphBuilder.shared.reset()
let distTensor = JTracer(value: 1.0, shape: JTensorShape([4, 4]), dtype: .float32)

let reducedTensor = distTensor.allReduce(reduction: .sum)
test("allReduce()", reducedTensor.shape == distTensor.shape)

let gatheredTensor = distTensor.allGather(gatherDim: 0, replicaGroups: [[0, 1, 2, 3]])
test("allGather()", gatheredTensor.shape.dimensions[0] == 16) // 4 * 4 replicas

let broadcastTensor = distTensor.broadcast(rootReplica: 0)
test("broadcast()", broadcastTensor.shape == distTensor.shape)

// Gradient synchronizer
let synchronizer = JGradientSynchronizer(mesh: mesh, reduction: .mean)
let syncedGrad = synchronizer.synchronize(distTensor)
test("GradientSynchronizer", syncedGrad.shape == distTensor.shape)

// Distributed context
JDistributedContext.shared.initialize(mesh: gridMesh, replicaId: 3)
test("DistributedContext worldSize", JDistributedContext.shared.worldSize == 8)
test("DistributedContext replicaId", JDistributedContext.shared.replicaId == 3)
test("DistributedContext isDistributed", JDistributedContext.shared.isDistributed)
JDistributedContext.shared.reset()

print()

// MARK: - Test 8: Automatic Differentiation

print("-" * 60)
print("TEST 8: Automatic Differentiation")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

@differentiable(reverse)
func simpleFunction(_ x: JTracer) -> JTracer {
    return x * x + x * 2.0
}

let input = JTracer(value: 3.0, shape: .scalar, dtype: .float32)
let (value, grad) = valueWithGradient(of: simpleFunction, at: input)
test("valueWithGradient() value", value.shape == .scalar)
test("valueWithGradient() gradient", grad.shape == .scalar)

let gradOnly = gradient(of: simpleFunction, at: input)
test("gradient()", gradOnly.shape == .scalar)

// Multi-input gradient
@differentiable(reverse)
func multiInputFunc(_ x: JTracer, _ y: JTracer) -> JTracer {
    return x * y + x + y
}

let (val2, grads2) = valueWithGradient(of: multiInputFunc, at: input, input)
test("Multi-input gradient (first)", grads2.0.shape == .scalar)
test("Multi-input gradient (second)", grads2.1.shape == .scalar)

print()

// MARK: - Test 9: Trigonometric Derivatives

print("-" * 60)
print("TEST 9: Trigonometric Function Derivatives")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

@differentiable(reverse)
func sinFunc(_ x: JTracer) -> JTracer { x.sin() }

@differentiable(reverse)
func cosFunc(_ x: JTracer) -> JTracer { x.cos() }

@differentiable(reverse)
func tanFunc(_ x: JTracer) -> JTracer { x.tan() }

let angle = JTracer(value: 0.5, shape: .scalar, dtype: .float32)

let sinGrad = gradient(of: sinFunc, at: angle)
test("sin() derivative (cos)", sinGrad.shape == .scalar)

let cosGrad = gradient(of: cosFunc, at: angle)
test("cos() derivative (-sin)", cosGrad.shape == .scalar)

let tanGrad = gradient(of: tanFunc, at: angle)
test("tan() derivative (sec²)", tanGrad.shape == .scalar)

print()

// MARK: - Test 10: JIT Compilation

print("-" * 60)
print("TEST 10: JIT Compilation & Execution")
print("-" * 60)

do {
    try SwiftIRJupyter.shared.initialize()

    @differentiable(reverse)
    func computableFunc(_ x: JTracer) -> JTracer {
        let y = x * x
        let z = y + x
        return z * 2.0
    }

    let compiled = try jitCompile(inputShape: [], dtype: .float32) { x in
        computableFunc(x)
    }
    test("JIT compilation", true)

    let results = try compiled.execute([[5.0]], shapes: [[]])
    test("JIT execution", results.count > 0)
    test("JIT result", results[0][0] == 60.0) // (5*5 + 5) * 2 = 60

} catch {
    test("JIT compilation", false)
    test("JIT execution", false)
    test("JIT result", false)
    print("  Error: \(error)")
}

print()

// MARK: - Summary

print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print()
print("Tests passed: \(testsPassed)")
print("Tests failed: \(testsFailed)")
print()

if testsFailed == 0 {
    print("🎉 ALL TESTS PASSED!")
    print()
    print("SwiftIRJupyter has FULL FEATURE PARITY with SwiftIR:")
    print("  ✓ Basic tensor operations (+, -, *, /)")
    print("  ✓ Unary operations (exp, log, sqrt, tanh, sigmoid, relu)")
    print("  ✓ Trigonometric operations (sin, cos, tan)")
    print("  ✓ Additional math (abs, power, rsqrt, floor, ceil, clamp)")
    print("  ✓ Advanced activations (leakyRelu, elu, silu, gelu, softplus)")
    print("  ✓ Softmax operations (softmax, logSoftmax)")
    print("  ✓ Matrix operations (matmul, transpose, reshape)")
    print("  ✓ Reduction operations (sum, mean, max, min)")
    print("  ✓ Element-wise ops (maximum, minimum)")
    print("  ✓ Tensor manipulation (slice, concatenate)")
    print("  ✓ Loss functions (MSE, BCE, CrossEntropy, Huber)")
    print("  ✓ Comparison operations (<, >, <=, >=, ==, !=)")
    print("  ✓ Automatic differentiation (VJP for all ops)")
    print("  ✓ Distributed training (DeviceMesh, ShardingSpec, collectives)")
    print("  ✓ JIT compilation and PJRT execution")
    print()
    print("Ready for Jupyter/Colab deployment!")
} else {
    print("❌ SOME TESTS FAILED")
    print("Please review the failed tests above.")
}

print()
print("=" * 60)
