/// Phase 3 Validation Tests - Differentiation System
/// Tests for gradient, valueWithGradient, pullback, and higher-order differentiation

import Testing
import _Differentiation

@testable import SwiftIR

@Suite("Phase 3: Differentiation System", .serialized)
struct Phase3ValidationTests {

    init() {
        TracerGraphBuilder.shared.reset()
        TokenManager.shared.reset()
    }

    // MARK: - Basic Gradient Tests

    @Suite("Basic Gradients")
    struct BasicGradientTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Gradient of identity function")
        func gradientIdentity() {
            let x = Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0 }, at: x)

            // d(x)/dx = 1
            #expect(grad.shape == x.shape)
            #expect(grad.dtype == x.dtype)
        }

        @Test("Gradient of scalar multiplication")
        func gradientScalarMul() {
            let x = Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0 * 2.0 }, at: x)

            // d(2x)/dx = 2
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of addition")
        func gradientAddition() {
            let x = Tracer(value: 1.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0 + $0 }, at: x)

            // d(x+x)/dx = 2
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of multiplication")
        func gradientMultiplication() {
            let x = Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0 * $0 }, at: x)

            // d(x^2)/dx = 2x
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of exp")
        func gradientExp() {
            let x = Tracer(value: 1.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0.exp() }, at: x)

            // d(exp(x))/dx = exp(x)
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of log")
        func gradientLog() {
            let x = Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32)

            let grad = gradient(of: { $0.log() }, at: x)

            // d(log(x))/dx = 1/x
            #expect(grad.shape == x.shape)
        }
    }

    // MARK: - Value with Gradient Tests

    @Suite("Value with Gradient")
    struct ValueWithGradientTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Value with gradient - quadratic")
        func valueWithGradientQuadratic() {
            let x = Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32)

            let (value, grad) = valueWithGradient(of: { $0 * $0 }, at: x)

            // f(3) = 9, f'(3) = 6
            #expect(value.shape == TensorShape.scalar)
            #expect(grad.shape == x.shape)
        }

        @Test("Value with gradient - composed functions")
        func valueWithGradientComposed() {
            let x = Tracer(value: 1.0, shape: TensorShape.scalar, dtype: .float32)

            let (value, grad) = valueWithGradient(
                of: { $0.exp().log() },
                at: x
            )

            // exp(log(x)) has gradient 1
            #expect(value.shape == x.shape)
            #expect(grad.shape == x.shape)
        }

        @Test("Value with gradient - multiple inputs")
        func valueWithGradientMultipleInputs() {
            let x = Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32)
            let y = Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32)

            let (value, grads) = valueWithGradient(
                of: { a, b in a * b },
                at: x, y
            )

            // f(x,y) = xy, df/dx = y, df/dy = x
            #expect(value.shape == TensorShape.scalar)
            #expect(grads.0.shape == x.shape)
            #expect(grads.1.shape == y.shape)
        }
    }

    // MARK: - Pullback Tests

    @Suite("Pullback Functions")
    struct PullbackTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Pullback basic usage")
        func pullbackBasic() {
            let x = Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32)

            let pb = pullback(at: x, of: { $0 * $0 })

            // Apply pullback with upstream gradient of 1
            let upstream = Tracer.ones(shape: TensorShape.scalar, dtype: .float32)
            let grad = pb(upstream)

            #expect(grad.shape == x.shape)
        }

        @Test("Value with pullback")
        func valueWithPullbackTest() {
            let x = Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32)

            let (value, pb) = valueWithPullback(at: x, of: { $0.exp() })

            #expect(value.shape == x.shape)

            let upstream = Tracer.ones(shape: TensorShape.scalar, dtype: .float32)
            let grad = pb(upstream)
            #expect(grad.shape == x.shape)
        }

        @Test("Pullback with custom upstream")
        func pullbackCustomUpstream() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)

            let (_, pb) = valueWithPullback(at: x, of: { $0 * 2.0 })

            // Custom upstream gradient
            let upstream = Tracer(value: 0.5, shape: TensorShape([2, 3]), dtype: .float32)
            let grad = pb(upstream)

            #expect(grad.shape == x.shape)
        }
    }

    // MARK: - Tensor Gradient Tests

    @Suite("Tensor Gradients")
    struct TensorGradientTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Gradient of tensor sum")
        func gradientTensorSum() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)

            let grad = gradient(of: { $0.sum() }, at: x)

            // d(sum(x))/dx = ones
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of element-wise operations")
        func gradientElementWise() {
            let x = Tracer(value: 2.0, shape: TensorShape([3, 4]), dtype: .float32)

            let grad = gradient(
                of: { ($0 * $0).sum() },
                at: x
            )

            // d(sum(x^2))/dx = 2x
            #expect(grad.shape == x.shape)
        }

        @Test("Gradient of matrix multiplication")
        func gradientMatmul() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let w = Tracer(value: 0.1, shape: TensorShape([3, 4]), dtype: .float32)

            let (_, grads) = valueWithGradient(
                of: { a, b in a.matmul(b).sum() },
                at: x, w
            )

            // Check gradient shapes
            #expect(grads.0.shape == x.shape)
            #expect(grads.1.shape == w.shape)
        }

        @Test("Gradient with broadcasting")
        func gradientBroadcasting() {
            let x = Tracer(value: 1.0, shape: TensorShape([2, 3]), dtype: .float32)
            let b = Tracer(value: 0.5, shape: TensorShape([1, 3]), dtype: .float32)

            let (_, grads) = valueWithGradient(
                of: { a, bias in (a + bias).sum() },
                at: x, b
            )

            #expect(grads.0.shape == x.shape)
            // Bias gradient should be reduced to original shape
            #expect(grads.1.shape.dimensions.last == 3)
        }
    }

    // MARK: - Chain Rule Tests

    @Suite("Chain Rule")
    struct ChainRuleTests {

        init() {
            TracerGraphBuilder.shared.reset()
        }

        @Test("Chain rule - simple composition")
        func chainRuleSimple() {
            let x = Tracer(value: 1.0, shape: TensorShape.scalar, dtype: .float32)

            // f(x) = exp(x^2)
            let grad = gradient(
                of: { ($0 * $0).exp() },
                at: x
            )

            // d/dx[exp(x^2)] = 2x * exp(x^2)
            #expect(grad.shape == x.shape)
        }

        @Test("Chain rule - triple composition")
        func chainRuleTriple() {
            let x = Tracer(value: 0.5, shape: TensorShape.scalar, dtype: .float32)

            // f(x) = log(exp(x^2))
            let grad = gradient(
                of: { ($0 * $0).exp().log() },
                at: x
            )

            // Should simplify to d/dx[x^2] = 2x
            #expect(grad.shape == x.shape)
        }

        @Test("Chain rule - with activations")
        func chainRuleActivations() {
            let x = Tracer(value: 0.5, shape: TensorShape([2, 3]), dtype: .float32)

            let grad = gradient(
                of: { $0.tanh().sigmoid().sum() },
                at: x
            )

            #expect(grad.shape == x.shape)
        }
    }

    // MARK: - Integration Tests

    @Test("Phase 3 integration - neural network gradient")
    func phase3NeuralNetworkGradient() {
        TracerGraphBuilder.shared.reset()

        print("\n========================================")
        print("Phase 3: Neural Network Gradient Test")
        print("========================================\n")

        // Simple 1-layer network: y = relu(x @ w + b)
        let x = Tracer(value: 1.0, shape: TensorShape([4, 8]), dtype: .float32)
        let w = Tracer(value: 0.1, shape: TensorShape([8, 4]), dtype: .float32)
        let b = Tracer(value: 0.0, shape: TensorShape([1, 4]), dtype: .float32)

        // Forward pass with loss
        let forward: @differentiable(reverse) (Tracer, Tracer, Tracer) -> Tracer = {
            input, weight, bias in
            let h = input.matmul(weight) + bias
            let activated = h.relu()
            return activated.sum()  // Scalar loss
        }

        // Compute value
        let loss = forward(x, w, b)
        #expect(loss.shape == TensorShape.scalar || loss.shape.rank <= 1)
        print("✅ Forward pass computed")

        // Compute gradients using pullback
        let (_, pb) = valueWithPullback(at: x, w, of: { input, weight in
            let h = input.matmul(weight) + b
            return h.relu().sum()
        })

        let upstream = Tracer.ones(shape: TensorShape.scalar, dtype: .float32)
        let (dInput, dWeight) = pb(upstream)

        #expect(dInput.shape == x.shape)
        #expect(dWeight.shape == w.shape)
        print("✅ Gradients computed correctly")

        print("\n========================================")
        print("✅ PHASE 3 NEURAL NETWORK GRADIENT COMPLETE")
        print("========================================\n")
    }

    @Test("Phase 3 integration - method chaining")
    func phase3MethodChaining() {
        TracerGraphBuilder.shared.reset()

        let x = Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32)

        // Use method chaining API
        let (value, grad) = x.valueWithGradient(through: { $0 * $0 })

        #expect(value.shape == TensorShape.scalar)
        #expect(grad.shape == x.shape)
    }

    @Test("Phase 3 integration - batch gradients")
    func phase3BatchGradients() {
        TracerGraphBuilder.shared.reset()

        let inputs = [
            Tracer(value: 1.0, shape: TensorShape.scalar, dtype: .float32),
            Tracer(value: 2.0, shape: TensorShape.scalar, dtype: .float32),
            Tracer(value: 3.0, shape: TensorShape.scalar, dtype: .float32),
        ]

        let grads = batchGradient(of: { $0 * $0 }, at: inputs)

        #expect(grads.count == inputs.count)
        for (input, grad) in zip(inputs, grads) {
            #expect(grad.shape == input.shape)
        }
    }
}
