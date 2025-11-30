# Higher-Order Automatic Differentiation Specification

## Overview

Higher-order AD enables computing Hessians (second derivatives), Jacobians, and efficient Hessian-vector products. Essential for second-order optimization, uncertainty quantification, and scientific computing.

## JAX Reference

```python
hessian = jax.hessian(loss_fn)(x)
hvp = jax.jvp(jax.grad(loss_fn), (x,), (v,))[1]  # Hessian-vector product
jacobian = jax.jacfwd(fn)(x)  # Forward-mode Jacobian
```

## SwiftIR API Specification

### Hessian

```swift
/// Compute full Hessian matrix (second derivatives)
/// For f: R^n -> R, returns n×n matrix
public func diffHessian<T: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> DifferentiableTracer,
    at x: T
) -> DifferentiableTracer  // Shape: [n, n]

/// Diagonal of Hessian (more efficient than full)
public func diffHessianDiag<T: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> DifferentiableTracer,
    at x: T
) -> T.TangentVector
```

### Hessian-Vector Product (HVP)

```swift
/// Efficient Hessian-vector product via forward-over-reverse
/// Computes H @ v without materializing H
public func diffHVP<T: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> DifferentiableTracer,
    at x: T,
    vector v: T.TangentVector
) -> T.TangentVector
```

### Jacobian

```swift
/// Forward-mode Jacobian (efficient for f: R^n -> R^m where n < m)
public func diffJacobianForward<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    at x: T
) -> DifferentiableTracer  // Shape: [m, n]

/// Reverse-mode Jacobian (efficient for f: R^n -> R^m where m < n)
public func diffJacobianReverse<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    at x: T
) -> DifferentiableTracer  // Shape: [m, n]
```

### JVP (Forward-Mode AD)

```swift
/// Jacobian-vector product (forward-mode differentiation)
/// Computes (f(x), J @ v) in a single forward pass
public func diffJVP<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    at x: T,
    tangent v: T.TangentVector
) -> (value: U, tangent: U.TangentVector)
```

### Usage Examples

```swift
// Full Hessian for small problems
@differentiable(reverse)
func loss(_ x: DifferentiableTracer) -> DifferentiableTracer {
    return diffSum(x * x * x)  // sum(x^3)
}

let x = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
let H = diffHessian(loss, at: x)  // 3x3 Hessian

// HVP for Newton-CG (large scale)
@differentiable(reverse)
func neuralNetLoss(_ params: Params) -> DifferentiableTracer {
    return crossEntropy(model(params, x), y)
}

let v = randomDirection(like: params)
let Hv = diffHVP(neuralNetLoss, at: params, vector: v)

// Use in conjugate gradient:
func newtonCG(_ params: Params, _ grad: Params) -> Params {
    var x = Params.zeros()
    var r = grad
    var p = r
    
    for _ in 0..<maxIter {
        let Hp = diffHVP(neuralNetLoss, at: params, vector: p)
        let alpha = dot(r, r) / dot(p, Hp)
        x = x + alpha * p
        let rNew = r - alpha * Hp
        let beta = dot(rNew, rNew) / dot(r, r)
        r = rNew
        p = rNew + beta * p
    }
    return x
}

// Jacobian for physics simulation
@differentiable(reverse)
func dynamics(_ state: DifferentiableTracer) -> DifferentiableTracer {
    // dx/dt = f(x)
    return nonlinearDynamics(state)
}

let J = diffJacobianForward(dynamics, at: currentState)
// Use J for stability analysis, linearization, etc.
```

## Implementation Guide

### HVP via Forward-over-Reverse

```swift
public func diffHVP<T: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> DifferentiableTracer,
    at x: T,
    vector v: T.TangentVector
) -> T.TangentVector {
    // Step 1: Define gradient function
    let gradFn = { (x: T) -> T.TangentVector in
        gradient(at: x, of: fn)
    }
    
    // Step 2: Apply JVP to gradient function
    // This computes d/dt [∇f(x + t*v)] at t=0 = H @ v
    let (_, hvp) = diffJVP(gradFn, at: x, tangent: v)
    
    return hvp
}
```

### Full Hessian (for small problems)

```swift
public func diffHessian(
    _ fn: @escaping @differentiable(reverse) (DifferentiableTracer) -> DifferentiableTracer,
    at x: DifferentiableTracer
) -> DifferentiableTracer {
    let n = x.shape.reduce(1, *)
    
    // Compute Hessian column by column using HVP
    var columns: [DifferentiableTracer] = []
    
    for i in 0..<n {
        // e_i = one-hot vector
        var oneHot = [Float](repeating: 0, count: n)
        oneHot[i] = 1.0
        let ei = DifferentiableTracer.constant(oneHot, shape: x.shape)
        
        // H[:, i] = H @ e_i
        let column = diffHVP(fn, at: x, vector: ei)
        columns.append(column.flattened())
    }
    
    return diffStack(columns, axis: 1)  // [n, n]
}
```

### JVP (Forward-Mode)

```swift
public func diffJVP<T: Differentiable, U: Differentiable>(
    _ fn: @escaping @differentiable(reverse) (T) -> U,
    at x: T,
    tangent v: T.TangentVector
) -> (value: U, tangent: U.TangentVector) {
    // Forward-mode AD: propagate tangents forward
    // Implemented via dual numbers or tracing
    
    // Create dual number representation
    let dualX = makeDual(x, tangent: v)
    
    // Trace through function
    let dualY = fn(dualX.primal)  // Need to extend tracer
    
    // Extract primal and tangent
    return (extractPrimal(dualY), extractTangent(dualY))
}
```

## Test Suite

```swift
final class HigherOrderADTests: XCTestCase {
    
    func testHessianQuadratic() {
        // f(x) = x^T A x / 2, Hessian = A
        let A = DifferentiableTracer.constant([
            1, 2,
            2, 5
        ], shape: [2, 2])
        
        @differentiable(reverse)
        func quadratic(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return 0.5 * diffSum(x * diffMatmul(A, x))
        }
        
        let x = DifferentiableTracer.constant([1.0, 1.0], shape: [2])
        let H = diffHessian(quadratic, at: x)
        
        let result = compileAndExecute(H)
        XCTAssertEqual(result, [1, 2, 2, 5], accuracy: 1e-5)
    }
    
    func testHVPCorrectness() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSum(x * x * x)  // f = sum(x^3), H = diag(6x)
        }
        
        let x = DifferentiableTracer.constant([1.0, 2.0, 3.0], shape: [3])
        let v = DifferentiableTracer.constant([1.0, 1.0, 1.0], shape: [3])
        
        let hvp = diffHVP(f, at: x, vector: v)
        
        // H @ v = [6*1, 6*2, 6*3] @ [1,1,1] = [6, 12, 18]
        let result = compileAndExecute(hvp)
        XCTAssertEqual(result, [6, 12, 18], accuracy: 1e-5)
    }
    
    func testJVPLinear() {
        @differentiable(reverse)
        func linear(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffMatmul(A, x)  // f(x) = Ax, Jacobian = A
        }
        
        let x = DifferentiableTracer.placeholder(shape: [3])
        let v = DifferentiableTracer.constant([1, 0, 0], shape: [3])
        
        let (_, tangent) = diffJVP(linear, at: x, tangent: v)
        
        // J @ e_0 = first column of A
        XCTAssertEqual(tangent.shape, [2])  // Output dimension
    }
    
    func testHessianNumerical() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSum(diffSin(x) * x)
        }
        
        let xVals: [Float] = [0.5, 1.0]
        let x = DifferentiableTracer.constant(xVals, shape: [2])
        
        let analyticalH = compileAndExecute(diffHessian(f, at: x))
        let numericalH = computeNumericalHessian(f, at: xVals, epsilon: 1e-4)
        
        XCTAssertEqual(analyticalH, numericalH, accuracy: 1e-3)
    }
    
    func testHVPvsFullHessian() {
        @differentiable(reverse)
        func f(_ x: DifferentiableTracer) -> DifferentiableTracer {
            return diffSum(diffExp(x))
        }
        
        let x = DifferentiableTracer.constant([1.0, 2.0], shape: [2])
        let v = DifferentiableTracer.constant([1.0, 2.0], shape: [2])
        
        // HVP directly
        let hvpDirect = compileAndExecute(diffHVP(f, at: x, vector: v))
        
        // Full Hessian then multiply
        let H = compileAndExecute(diffHessian(f, at: x))
        let vArr = compileAndExecute(v)
        let hvpFull = matmul2x2(H, vArr)  // Manual 2x2 matmul
        
        XCTAssertEqual(hvpDirect, hvpFull, accuracy: 1e-5)
    }
}
```

## Success Criteria

- [ ] Correct Hessian for quadratic functions (H = A)
- [ ] HVP matches H @ v computed via full Hessian
- [ ] Numerical gradient check passes for Hessian
- [ ] JVP computes correct tangents
- [ ] Jacobian correct for linear functions (J = A)
- [ ] HVP efficient (O(n) not O(n²))
- [ ] Works with all differentiable ops

## Files to Create

- `Sources/SwiftIR/SymbolicAD/HigherOrderAD.swift`
- `Tests/SwiftIRTests/HigherOrderADTests.swift`
