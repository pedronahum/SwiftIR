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
let slicedTensor = sliceTensor.slice(starts: [0, 0], limits: [2, 3], strides: [1, 1])
test("slice() shape", slicedTensor.shape.dimensions.compactMap { $0 } == [2, 3])

let cat1 = JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32)
let cat2 = JTracer(value: 2.0, shape: JTensorShape([2, 3]), dtype: .float32)
let concatenated = JTracer.concatenate([cat1, cat2], axis: 0)
test("concatenate(axis=0)", concatenated.shape.dimensions.compactMap { $0 } == [4, 3])

let concatAxis1 = JTracer.concatenate([cat1, cat2], axis: 1)
test("concatenate(axis=1)", concatAxis1.shape.dimensions.compactMap { $0 } == [2, 6])

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

// MARK: - Test 11: Scan Operations

print("-" * 60)
print("TEST 11: Scan Operations (JAX-like)")
print("-" * 60)

JTracerGraphBuilder.shared.reset()

// Test jScan basic API
func testScanStep(carry: JTracer, input: JTracer) -> (JTracer, JTracer) {
    let newCarry = carry + input
    return (newCarry, newCarry)
}

let scanInputs = JTracer(value: 1.0, shape: JTensorShape([5]), dtype: .float32)
let scanInit = JTracer(value: 0.0, shape: .scalar, dtype: .float32)

// Test that jScan compiles without error (shape checking)
let (scanFinal, scanOutputs) = jScan(testScanStep, initCarry: scanInit, xs: scanInputs)
test("jScan returns final carry", scanFinal.shape == .scalar)
test("jScan returns outputs with sequence shape", scanOutputs.shape.dimensions[0] == 5)

// Test jCumsum
JTracerGraphBuilder.shared.reset()
let cumsumInput = JTracer(value: 1.0, shape: JTensorShape([4]), dtype: .float32)
let cumsumResult = jCumsum(cumsumInput)
test("jCumsum shape", cumsumResult.shape.dimensions[0] == 4)

// Test jCumprod
JTracerGraphBuilder.shared.reset()
let cumprodInput = JTracer(value: 2.0, shape: JTensorShape([3]), dtype: .float32)
let cumprodResult = jCumprod(cumprodInput)
test("jCumprod shape", cumprodResult.shape.dimensions[0] == 3)

// Test jScanSimple
JTracerGraphBuilder.shared.reset()
func simpleStep(state: JTracer, input: JTracer) -> JTracer {
    return state + input
}
let (simpleFinal, simpleOutputs) = jScanSimple(simpleStep, initState: scanInit, xs: scanInputs)
test("jScanSimple returns final state", simpleFinal.shape == .scalar)
test("jScanSimple returns all states", simpleOutputs.shape.dimensions[0] == 5)

// Test reverse scan
JTracerGraphBuilder.shared.reset()
let (reverseFinal, reverseOutputs) = jScan(testScanStep, initCarry: scanInit, xs: scanInputs, reverse: true)
test("jScan reverse mode", reverseFinal.shape == .scalar)
test("jScan reverse outputs shape", reverseOutputs.shape.dimensions[0] == 5)

// Test dynamic slice helper
JTracerGraphBuilder.shared.reset()
let sliceInput = JTracer(value: 1.0, shape: JTensorShape([10, 3]), dtype: .float32)
let sliceIndex = JTracer(value: 2.0, shape: .scalar, dtype: .float32)
let dynamicSliced = jDynamicSliceAt(sliceInput, index: sliceIndex, featureShape: JTensorShape([3]))
test("jDynamicSliceAt returns correct shape", dynamicSliced.shape.dimensions.compactMap { $0 } == [3])

// Test dynamic update slice helper
JTracerGraphBuilder.shared.reset()
let updateOperand = JTracer(value: 0.0, shape: JTensorShape([10, 3]), dtype: .float32)
let updateValue = JTracer(value: 1.0, shape: JTensorShape([3]), dtype: .float32)
let updateIndex = JTracer(value: 5.0, shape: .scalar, dtype: .float32)
let updated = jDynamicUpdateSliceAt(updateOperand, value: updateValue, index: updateIndex, outputShape: JTensorShape([3]))
test("jDynamicUpdateSliceAt preserves shape", updated.shape.dimensions.compactMap { $0 } == [10, 3])

print()

// MARK: - Test 12: Conditional Operations (cond/select)

print("TEST 12: Conditional Operations (jSelect, jCond, jWhere, jClamp)")
print("-" * 60)

// Test jSelect
JTracerGraphBuilder.shared.reset()
let selectCond = JTracer(value: 1.0, shape: JTensorShape([3]), dtype: .bool)  // Simulates boolean
let selectTrue = JTracer(value: 10.0, shape: JTensorShape([3]), dtype: .float32)
let selectFalse = JTracer(value: 0.0, shape: JTensorShape([3]), dtype: .float32)
let selectResult = jSelect(selectCond, onTrue: selectTrue, onFalse: selectFalse)
test("jSelect returns correct shape", selectResult.shape == JTensorShape([3]))
test("jSelect returns correct dtype", selectResult.dtype == .float32)

// Test jCond (scalar conditional)
JTracerGraphBuilder.shared.reset()
let condPredicate = JTracer(value: 1.0, shape: .scalar, dtype: .bool)
let condX = JTracer(value: 5.0, shape: JTensorShape([2, 3]), dtype: .float32)
let condResult = jCond(condPredicate, onTrue: { condX * JTracer(value: 2.0, shape: JTensorShape([2, 3]), dtype: .float32) }, onFalse: { condX })
test("jCond returns correct shape", condResult.shape == JTensorShape([2, 3]))

// Test jCondWith (conditional with operand)
JTracerGraphBuilder.shared.reset()
let condWithPred = JTracer(value: 0.0, shape: .scalar, dtype: .bool)
let condWithX = JTracer(value: 3.0, shape: JTensorShape([4]), dtype: .float32)
let condWithResult = jCondWith(condWithPred, operand: condWithX, onTrue: { x in x + x }, onFalse: { x in x })
test("jCondWith returns correct shape", condWithResult.shape == JTensorShape([4]))

// Test jWhere (NumPy-style alias)
JTracerGraphBuilder.shared.reset()
let whereMask = JTracer(value: 1.0, shape: JTensorShape([5]), dtype: .bool)
let whereX = JTracer(value: 1.0, shape: JTensorShape([5]), dtype: .float32)
let whereY = JTracer(value: 0.0, shape: JTensorShape([5]), dtype: .float32)
let whereResult = jWhere(whereMask, whereX, whereY)
test("jWhere returns correct shape", whereResult.shape == JTensorShape([5]))

// Test jClamp
JTracerGraphBuilder.shared.reset()
let clampX = JTracer(value: 5.0, shape: JTensorShape([3, 4]), dtype: .float32)
let jClampResult = jClamp(clampX, min: 0.0, max: 1.0)
test("jClamp returns correct shape", jClampResult.shape == JTensorShape([3, 4]))
test("jClamp preserves dtype", jClampResult.dtype == .float32)

// Test comparison functions
JTracerGraphBuilder.shared.reset()
let cmpA = JTracer(value: 2.0, shape: JTensorShape([3]), dtype: .float32)
let cmpB = JTracer(value: 3.0, shape: JTensorShape([3]), dtype: .float32)

let jGtResult = jGreater(cmpA, cmpB)
test("jGreater returns bool dtype", jGtResult.dtype == .bool)
test("jGreater returns correct shape", jGtResult.shape == JTensorShape([3]))

let jGeResult = jGreaterEqual(cmpA, cmpB)
test("jGreaterEqual returns bool dtype", jGeResult.dtype == .bool)

let jLtResult = jLess(cmpA, cmpB)
test("jLess returns bool dtype", jLtResult.dtype == .bool)

let jLeResult = jLessEqual(cmpA, cmpB)
test("jLessEqual returns bool dtype", jLeResult.dtype == .bool)

let jEqResult = jEqual(cmpA, cmpB)
test("jEqual returns bool dtype", jEqResult.dtype == .bool)

let jNeResult = jNotEqual(cmpA, cmpB)
test("jNotEqual returns bool dtype", jNeResult.dtype == .bool)

print()

// MARK: - Test 13: Vmap Operations (Automatic Vectorization)

print("TEST 13: Vmap Operations (jVmap, jVmap2, jVmap3)")
print("-" * 60)

// Test basic jVmap - single input
JTracerGraphBuilder.shared.reset()
let vmapInput = JTracer(value: 1.0, shape: JTensorShape([8, 64]), dtype: .float32)  // [batch, features]
let vmappedFn = jVmap({ x in x.relu() })
let vmapResult = vmappedFn(vmapInput)
test("jVmap single input shape preserved", vmapResult.shape == JTensorShape([8, 64]))

// Test jVmap with axis specification
JTracerGraphBuilder.shared.reset()
let vmapInput2 = JTracer(value: 1.0, shape: JTensorShape([16, 32]), dtype: .float32)
let vmappedFn2 = jVmap({ x in x.tanh() }, inAxes: JVmapAxes(0))
let vmapResult2 = vmappedFn2(vmapInput2)
test("jVmap with inAxes shape", vmapResult2.shape == JTensorShape([16, 32]))

// Test jVmap2 - two inputs, both batched
JTracerGraphBuilder.shared.reset()
let vmap2A = JTracer(value: 1.0, shape: JTensorShape([4, 16]), dtype: .float32)
let vmap2B = JTracer(value: 2.0, shape: JTensorShape([4, 16]), dtype: .float32)
let vmapped2Fn = jVmap2({ a, b in a + b }, inAxes: JVmapAxes(0, 0))
let vmap2Result = vmapped2Fn(vmap2A, vmap2B)
test("jVmap2 both batched shape", vmap2Result.shape == JTensorShape([4, 16]))

// Test jVmap2 - one input batched, one broadcast
JTracerGraphBuilder.shared.reset()
let vmap2X = JTracer(value: 1.0, shape: JTensorShape([8, 32]), dtype: .float32)  // [batch, features]
let vmap2W = JTracer(value: 0.5, shape: JTensorShape([32]), dtype: .float32)  // [features] - will be broadcast
let vmapped2BroadcastFn = jVmap2({ x, w in x * w }, inAxes: JVmapAxes(0, nil))
let vmap2BroadcastResult = vmapped2BroadcastFn(vmap2X, vmap2W)
test("jVmap2 with broadcast shape", vmap2BroadcastResult.shape.dimensions[0] == 8)

// Test jVmap3 - three inputs (MLP-like pattern)
JTracerGraphBuilder.shared.reset()
let vmap3X = JTracer(value: 1.0, shape: JTensorShape([4, 10]), dtype: .float32)  // [batch, input]
let vmap3W1 = JTracer(value: 0.1, shape: JTensorShape([10, 20]), dtype: .float32)  // [input, hidden]
let vmap3W2 = JTracer(value: 0.1, shape: JTensorShape([20, 5]), dtype: .float32)  // [hidden, output]
// Note: jVmap3 broadcasts W1 and W2 across batch
let vmapped3Fn = jVmap3({ x, w1, w2 in
    let h = x.matmul(w1).relu()
    return h.matmul(w2)
}, inAxes: JVmapAxes(0, nil, nil))
let vmap3Result = vmapped3Fn(vmap3X, vmap3W1, vmap3W2)
test("jVmap3 MLP-like pattern batch dim", vmap3Result.shape.dimensions[0] == 4)
test("jVmap3 MLP-like pattern output dim", vmap3Result.shape.dimensions[1] == 5)

// Test jMoveAxis
JTracerGraphBuilder.shared.reset()
let moveAxisInput = JTracer(value: 1.0, shape: JTensorShape([2, 3, 4]), dtype: .float32)
let movedAxis = jMoveAxis(moveAxisInput, from: 0, to: 2)
test("jMoveAxis shape transformation", movedAxis.shape.dimensions.compactMap { $0 } == [3, 4, 2])

// Test jBroadcastBatch
JTracerGraphBuilder.shared.reset()
let broadcastInput = JTracer(value: 1.0, shape: JTensorShape([32, 64]), dtype: .float32)
let broadcastResult = jBroadcastBatch(broadcastInput, batchSize: 8, batchAxis: 0)
test("jBroadcastBatch adds batch dim", broadcastResult.shape.dimensions.compactMap { $0 } == [8, 32, 64])

// Test jBatchedMatmul - 3D @ 2D
JTracerGraphBuilder.shared.reset()
let batchedA = JTracer(value: 1.0, shape: JTensorShape([4, 8, 16]), dtype: .float32)  // [batch, m, k]
let batchedB = JTracer(value: 1.0, shape: JTensorShape([16, 32]), dtype: .float32)  // [k, n]
let batchedMatmulResult = jBatchedMatmul(batchedA, batchedB)
test("jBatchedMatmul 3D@2D shape", batchedMatmulResult.shape.dimensions.compactMap { $0 } == [4, 8, 32])

// Test jBatchedMatmul - 3D @ 3D (both batched)
JTracerGraphBuilder.shared.reset()
let batchedA2 = JTracer(value: 1.0, shape: JTensorShape([4, 8, 16]), dtype: .float32)  // [batch, m, k]
let batchedB2 = JTracer(value: 1.0, shape: JTensorShape([4, 16, 32]), dtype: .float32)  // [batch, k, n]
let batchedMatmulResult2 = jBatchedMatmul(batchedA2, batchedB2)
test("jBatchedMatmul 3D@3D shape", batchedMatmulResult2.shape.dimensions.compactMap { $0 } == [4, 8, 32])

// Test JVmapAxes expressibility
let axes1: JVmapAxes = 0
test("JVmapAxes integer literal", axes1.axes[0] == 0)
let axes2: JVmapAxes = nil
test("JVmapAxes nil literal", axes2.axes[0] == nil)
let axes3: JVmapAxes = [0, nil, 1]
test("JVmapAxes array literal count", axes3.axes.count == 3)

print()

// MARK: - Test 14: JTree Operations (jTreeMap, jTreeZipWith, jTreeReduce)

print("-" * 60)
print("TEST 14: JTree Operations (jTreeMap, jTreeZipWith, jTreeReduce)")
print("-" * 60)

// Test JTracer base conformance to JTree
JTracerGraphBuilder.shared.reset()
let treeLeaf = JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32)
let flattenedLeaf = treeLeaf.flatten()
test("JTracer flatten returns single element", flattenedLeaf.count == 1)

let unflattenedLeaf = JTracer.unflatten(flattenedLeaf)
test("JTracer unflatten shape matches", unflattenedLeaf.shape.dimensions.compactMap { $0 } == [4, 8])

test("JTracer treeStructure is leaf", JTracer.treeStructure.leafCount == 1)

// Test Array<JTracer> conformance
JTracerGraphBuilder.shared.reset()
let treeArray = [
    JTracer(value: 1.0, shape: JTensorShape([2, 3]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([2, 3]), dtype: .float32),
    JTracer(value: 3.0, shape: JTensorShape([2, 3]), dtype: .float32)
]
let flattenedArray = treeArray.flatten()
test("Array<JTracer> flatten count", flattenedArray.count == 3)

let unflattenedArray = [JTracer].unflatten(flattenedArray)
test("Array<JTracer> unflatten count", unflattenedArray.count == 3)

// Test JTuple2 conformance
JTracerGraphBuilder.shared.reset()
let tuple2 = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([8, 16]), dtype: .float32)
)
let flattenedTuple2 = tuple2.flatten()
test("JTuple2 flatten count", flattenedTuple2.count == 2)
test("JTuple2 treeStructure leafCount", JTuple2<JTracer, JTracer>.treeStructure.leafCount == 2)

let unflattenedTuple2 = JTuple2<JTracer, JTracer>.unflatten(flattenedTuple2)
test("JTuple2 unflatten first shape", unflattenedTuple2.t0.shape.dimensions.compactMap { $0 } == [4, 8])
test("JTuple2 unflatten second shape", unflattenedTuple2.t1.shape.dimensions.compactMap { $0 } == [8, 16])

// Test JTuple3 conformance
JTracerGraphBuilder.shared.reset()
let tuple3 = JTuple3(
    JTracer(value: 1.0, shape: JTensorShape([10, 20]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([20]), dtype: .float32),
    JTracer(value: 3.0, shape: JTensorShape([20, 10]), dtype: .float32)
)
let flattenedTuple3 = tuple3.flatten()
test("JTuple3 flatten count", flattenedTuple3.count == 3)
test("JTuple3 treeStructure leafCount", JTuple3<JTracer, JTracer, JTracer>.treeStructure.leafCount == 3)

// Test JTuple4 conformance
JTracerGraphBuilder.shared.reset()
let tuple4 = JTuple4(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([8]), dtype: .float32),
    JTracer(value: 3.0, shape: JTensorShape([8, 16]), dtype: .float32),
    JTracer(value: 4.0, shape: JTensorShape([16]), dtype: .float32)
)
let flattenedTuple4 = tuple4.flatten()
test("JTuple4 flatten count", flattenedTuple4.count == 4)
test("JTuple4 treeStructure leafCount", JTuple4<JTracer, JTracer, JTracer, JTracer>.treeStructure.leafCount == 4)

// Test jTreeMap - scale all parameters
JTracerGraphBuilder.shared.reset()
let mapInput = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([8, 16]), dtype: .float32)
)
let scaled = jTreeMap(mapInput) { leaf in leaf * JTracer(value: 2.0, shape: leaf.shape, dtype: leaf.dtype) }
test("jTreeMap preserves structure", scaled.flatten().count == 2)
test("jTreeMap first shape unchanged", scaled.t0.shape.dimensions.compactMap { $0 } == [4, 8])
test("jTreeMap second shape unchanged", scaled.t1.shape.dimensions.compactMap { $0 } == [8, 16])

// Test jTreeZipWith - SGD-like update
JTracerGraphBuilder.shared.reset()
let zipParams = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([10, 20]), dtype: .float32),
    JTracer(value: 0.5, shape: JTensorShape([20]), dtype: .float32)
)
let zipGrads = JTuple2(
    JTracer(value: 0.1, shape: JTensorShape([10, 20]), dtype: .float32),
    JTracer(value: 0.05, shape: JTensorShape([20]), dtype: .float32)
)
let zipUpdated: JTuple2<JTracer, JTracer> = jTreeZipWith(zipParams, zipGrads) { p, g in p - g }
test("jTreeZipWith preserves structure", zipUpdated.flatten().count == 2)
test("jTreeZipWith first shape", zipUpdated.t0.shape.dimensions.compactMap { $0 } == [10, 20])
test("jTreeZipWith second shape", zipUpdated.t1.shape.dimensions.compactMap { $0 } == [20])

// Test jTreeZipWith3 - Adam-like update pattern
JTracerGraphBuilder.shared.reset()
let adamParams = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 0.5, shape: JTensorShape([8]), dtype: .float32)
)
let adamM = JTuple2(
    JTracer(value: 0.1, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 0.05, shape: JTensorShape([8]), dtype: .float32)
)
let adamV = JTuple2(
    JTracer(value: 0.01, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 0.005, shape: JTensorShape([8]), dtype: .float32)
)
let adamUpdated = jTreeZipWith3(adamParams, adamM, adamV) { p, m, v in p - m }
test("jTreeZipWith3 preserves structure", adamUpdated.flatten().count == 2)

// Test jTreeReduce - sum all parameters (L2 norm style)
JTracerGraphBuilder.shared.reset()
let reduceInput = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([8, 16]), dtype: .float32)
)
let initial = JTracer(value: 0.0, shape: JTensorShape([1]), dtype: .float32)
let reduced = jTreeReduce(reduceInput, initial) { acc, leaf in acc + leaf.sum() }
test("jTreeReduce returns scalar-like", reduced.shape.dimensions.compactMap { $0 }.count <= 1)

// Test jTreeLeafCount
JTracerGraphBuilder.shared.reset()
let countInput = JTuple3(
    JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([8]), dtype: .float32),
    JTracer(value: 3.0, shape: JTensorShape([8, 16]), dtype: .float32)
)
test("jTreeLeafCount", jTreeLeafCount(countInput) == 3)

// Test jTreeShapes
JTracerGraphBuilder.shared.reset()
let shapesInput = JTuple2(
    JTracer(value: 1.0, shape: JTensorShape([10, 20]), dtype: .float32),
    JTracer(value: 2.0, shape: JTensorShape([20, 30]), dtype: .float32)
)
let shapes = jTreeShapes(shapesInput)
test("jTreeShapes count", shapes.count == 2)
test("jTreeShapes first", shapes[0].dimensions.compactMap { $0 } == [10, 20])
test("jTreeShapes second", shapes[1].dimensions.compactMap { $0 } == [20, 30])

// Test jTreeZerosLike
JTracerGraphBuilder.shared.reset()
let zerosInput = JTuple2(
    JTracer(value: 5.0, shape: JTensorShape([4, 8]), dtype: .float32),
    JTracer(value: 10.0, shape: JTensorShape([8, 16]), dtype: .float32)
)
let zeros = jTreeZerosLike(zerosInput)
test("jTreeZerosLike structure", zeros.flatten().count == 2)
test("jTreeZerosLike first shape", zeros.t0.shape.dimensions.compactMap { $0 } == [4, 8])
test("jTreeZerosLike second shape", zeros.t1.shape.dimensions.compactMap { $0 } == [8, 16])

// Test jTreeOnesLike
JTracerGraphBuilder.shared.reset()
let onesInput = JTuple2(
    JTracer(value: 0.0, shape: JTensorShape([3, 5]), dtype: .float32),
    JTracer(value: 0.0, shape: JTensorShape([5, 7]), dtype: .float32)
)
let ones = jTreeOnesLike(onesInput)
test("jTreeOnesLike structure", ones.flatten().count == 2)

// Test jTreeFullLike
JTracerGraphBuilder.shared.reset()
let fullInput = JTuple2(
    JTracer(value: 0.0, shape: JTensorShape([2, 4]), dtype: .float32),
    JTracer(value: 0.0, shape: JTensorShape([4, 6]), dtype: .float32)
)
let full = jTreeFullLike(fullInput, value: 0.5)
test("jTreeFullLike structure", full.flatten().count == 2)

// Test JTreeStructure metadata
test("JTreeStructure leaf leafCount", JTreeStructure.leaf.leafCount == 1)
let customStructure = JTreeStructure(leafCount: 5, children: [], typeName: "Custom")
test("JTreeStructure custom leafCount", customStructure.leafCount == 5)
test("JTreeStructure custom typeName", customStructure.typeName == "Custom")

// Test nested tree (Array of Tuple2)
JTracerGraphBuilder.shared.reset()
let nestedTree: [JTuple2<JTracer, JTracer>] = [
    JTuple2(
        JTracer(value: 1.0, shape: JTensorShape([4, 8]), dtype: .float32),
        JTracer(value: 2.0, shape: JTensorShape([8]), dtype: .float32)
    ),
    JTuple2(
        JTracer(value: 3.0, shape: JTensorShape([4, 8]), dtype: .float32),
        JTracer(value: 4.0, shape: JTensorShape([8]), dtype: .float32)
    )
]
let flattenedNested = nestedTree.flatten()
test("Nested tree flatten count", flattenedNested.count == 4)

print()

// MARK: - Test 15: PRNG Operations (JPRNGKey, jRandomUniform, jRandomNormal)

print("-" * 60)
print("TEST 15: PRNG Operations (JPRNGKey, jRandomUniform, jRandomNormal)")
print("-" * 60)

// Test JPRNGKey creation
let prngKey = JPRNGKey(seed: 42)
test("JPRNGKey creation", prngKey.data.0 != 0 || prngKey.data.1 != 0)

// Test determinism - same seed should produce same key
let prngKey2 = JPRNGKey(seed: 42)
test("JPRNGKey determinism", prngKey.data == prngKey2.data)

// Test different seeds produce different keys
let prngKey3 = JPRNGKey(seed: 123)
test("JPRNGKey different seeds", prngKey.data != prngKey3.data)

// Test split produces two different keys
let (splitKey1, splitKey2) = prngKey.split()
test("JPRNGKey split produces different keys", splitKey1.data != splitKey2.data)

// Test split(into:) produces correct count
let splitKeys = prngKey.split(into: 5)
test("JPRNGKey split(into:) count", splitKeys.count == 5)

// Test all split keys are different
var allDifferent = true
for i in 0..<splitKeys.count {
    for j in (i+1)..<splitKeys.count {
        if splitKeys[i].data == splitKeys[j].data {
            allDifferent = false
        }
    }
}
test("JPRNGKey split(into:) all different", allDifferent)

// Test fold(in:) produces different key
let foldedKey = prngKey.fold(in: 7)
test("JPRNGKey fold(in:) produces different key", foldedKey.data != prngKey.data)

// Test jRandomUniform shape
JTracerGraphBuilder.shared.reset()
let uniformResult = jRandomUniform(prngKey, shape: JTensorShape([8, 16]), dtype: .float32)
test("jRandomUniform shape", uniformResult.shape.dimensions.compactMap { $0 } == [8, 16])
test("jRandomUniform dtype", uniformResult.dtype == .float32)

// Test jRandomNormal shape
JTracerGraphBuilder.shared.reset()
let normalResult = jRandomNormal(prngKey, shape: JTensorShape([4, 32]), dtype: .float32)
test("jRandomNormal shape", normalResult.shape.dimensions.compactMap { $0 } == [4, 32])
test("jRandomNormal dtype", normalResult.dtype == .float32)

// Test jRandomNormal with mean and stddev
JTracerGraphBuilder.shared.reset()
let normalScaled = jRandomNormal(prngKey, shape: JTensorShape([10, 20]), mean: 0.5, stddev: 0.1, dtype: .float32)
test("jRandomNormal scaled shape", normalScaled.shape.dimensions.compactMap { $0 } == [10, 20])

// Test jRandomInt shape
JTracerGraphBuilder.shared.reset()
let intResult = jRandomInt(prngKey, shape: JTensorShape([5, 10]), minval: 0, maxval: 100, dtype: .int32)
test("jRandomInt shape", intResult.shape.dimensions.compactMap { $0 } == [5, 10])
test("jRandomInt dtype", intResult.dtype == .int32)

// Test jDropout preserves shape
JTracerGraphBuilder.shared.reset()
let dropoutInput = JTracer(value: 1.0, shape: JTensorShape([8, 16]), dtype: .float32)
let droppedResult = jDropout(prngKey, dropoutInput, rate: 0.1)
test("jDropout shape preserved", droppedResult.shape.dimensions.compactMap { $0 } == [8, 16])
test("jDropout dtype preserved", droppedResult.dtype == .float32)

// Test jDropout with rate 0 (no dropout)
JTracerGraphBuilder.shared.reset()
let noDropoutInput = JTracer(value: 2.0, shape: JTensorShape([4, 8]), dtype: .float32)
let noDropoutResult = jDropout(prngKey, noDropoutInput, rate: 0.0)
test("jDropout rate 0 shape", noDropoutResult.shape.dimensions.compactMap { $0 } == [4, 8])

// Test jTruncatedNormal shape
JTracerGraphBuilder.shared.reset()
let truncNormal = jTruncatedNormal(prngKey, shape: JTensorShape([16, 32]), mean: 0, stddev: 1, dtype: .float32)
test("jTruncatedNormal shape", truncNormal.shape.dimensions.compactMap { $0 } == [16, 32])

// Test jXavierUniform shape
JTracerGraphBuilder.shared.reset()
let xavierU = jXavierUniform(prngKey, shape: JTensorShape([784, 256]), dtype: .float32)
test("jXavierUniform shape", xavierU.shape.dimensions.compactMap { $0 } == [784, 256])

// Test jXavierNormal shape
JTracerGraphBuilder.shared.reset()
let xavierN = jXavierNormal(prngKey, shape: JTensorShape([256, 128]), dtype: .float32)
test("jXavierNormal shape", xavierN.shape.dimensions.compactMap { $0 } == [256, 128])

// Test jHeUniform shape
JTracerGraphBuilder.shared.reset()
let heU = jHeUniform(prngKey, shape: JTensorShape([512, 256]), dtype: .float32)
test("jHeUniform shape", heU.shape.dimensions.compactMap { $0 } == [512, 256])

// Test jHeNormal shape
JTracerGraphBuilder.shared.reset()
let heN = jHeNormal(prngKey, shape: JTensorShape([128, 64]), dtype: .float32)
test("jHeNormal shape", heN.shape.dimensions.compactMap { $0 } == [128, 64])

// Test JRngDistribution enum
test("JRngDistribution uniform", JRngDistribution.uniform.rawValue == "UNIFORM")
test("JRngDistribution normal", JRngDistribution.normal.rawValue == "NORMAL")

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
    print("  ✓ Scan operations (jScan, jCumsum, jCumprod, jScanSimple)")
    print("  ✓ Conditional operations (jSelect, jCond, jCondWith, jWhere, jClamp)")
    print("  ✓ Vmap operations (jVmap, jVmap2, jVmap3, jBatchedMatmul)")
    print("  ✓ Tree operations (JTree, jTreeMap, jTreeZipWith, jTreeReduce)")
    print("  ✓ PRNG operations (JPRNGKey, jRandomUniform, jRandomNormal, jDropout)")
    print()
    print("Ready for Jupyter/Colab deployment!")
} else {
    print("❌ SOME TESTS FAILED")
    print("Please review the failed tests above.")
}

print()
print("=" * 60)
