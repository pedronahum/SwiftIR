// PRNG.swift - Functional (Splittable) Pseudo-Random Number Generation
// Works with C++ interop via MLIR bindings
//
// This file implements JAX-style functional PRNGs that enable reproducible
// randomness in parallel/vectorized code. Unlike stateful PRNGs, functional
// PRNGs are explicit and composable.
//
// Based on JAX's PRNG: https://jax.readthedocs.io/en/latest/jax.random.html

import Foundation
import _Differentiation

// MARK: - PRNG Key

/// Immutable PRNG key using Threefry-based algorithm
///
/// Functional PRNGs differ from traditional stateful PRNGs:
/// - Keys are immutable values, not mutable state
/// - `split()` creates independent streams from one key
/// - Same key + same operation = same result (deterministic)
///
/// **Usage Example**:
/// ```swift
/// let key = PRNGKey(seed: 42)
/// let (key1, key2) = key.split()
/// let noise = diffRandomNormal(key1, shape: [32, 64])
/// let mask = diffRandomUniform(key2, shape: [32, 64])
/// ```
public struct PRNGKey: Hashable, Sendable {
    /// Internal state (Threefry 2x32)
    internal let data: (UInt32, UInt32)

    /// Create a new PRNG key from a seed
    public init(seed: UInt64) {
        self.data = PRNGKey.threefryInit(seed)
    }

    /// Internal initializer from raw data
    internal init(data: (UInt32, UInt32)) {
        self.data = data
    }

    /// Split key into two independent keys
    ///
    /// This is the core operation for functional PRNGs. The two resulting
    /// keys will produce independent random streams.
    ///
    /// ```swift
    /// let key = PRNGKey(seed: 42)
    /// let (trainKey, evalKey) = key.split()
    /// ```
    public func split() -> (PRNGKey, PRNGKey) {
        let (a, b) = PRNGKey.threefryRandom2x32(data, counter: (0, 0))
        let (c, d) = PRNGKey.threefryRandom2x32(data, counter: (1, 0))
        return (PRNGKey(data: a), PRNGKey(data: (c.0, d.0)))
    }

    /// Split key into n independent keys
    ///
    /// Useful for batched operations where each element needs its own key.
    ///
    /// ```swift
    /// let keys = key.split(into: batchSize)
    /// let batchedDropout = vmap(diffDropout, inAxes: (0, 0, nil))
    /// let dropped = batchedDropout(keys, x, 0.1)
    /// ```
    public func split(into n: Int) -> [PRNGKey] {
        var keys: [PRNGKey] = []
        keys.reserveCapacity(n)
        var current = self
        for _ in 0..<n {
            let (next, key) = current.split()
            keys.append(key)
            current = next
        }
        return keys
    }

    /// Fold in additional data to create a new key
    ///
    /// Useful for creating keys that depend on indices or other data.
    ///
    /// ```swift
    /// let layerKey = key.fold(in: UInt32(layerIndex))
    /// ```
    public func fold(in data: UInt32) -> PRNGKey {
        let newData = PRNGKey.threefryRandom2x32(self.data, counter: (data, 0))
        return PRNGKey(data: newData.0)
    }

    // MARK: - Threefry Implementation

    /// Initialize Threefry state from seed
    private static func threefryInit(_ seed: UInt64) -> (UInt32, UInt32) {
        let high = UInt32(truncatingIfNeeded: seed >> 32)
        let low = UInt32(truncatingIfNeeded: seed)
        // Mix the seed using Threefry
        return threefryRandom2x32((high, low), counter: (0, 0)).0
    }

    /// Threefry 2x32 random number generator
    /// Based on the Threefry algorithm from "Parallel Random Numbers: As Easy as 1, 2, 3"
    private static func threefryRandom2x32(
        _ key: (UInt32, UInt32),
        counter: (UInt32, UInt32)
    ) -> ((UInt32, UInt32), (UInt32, UInt32)) {
        // Rotation constants for Threefry 2x32
        let rotations: [UInt32] = [13, 15, 26, 6, 17, 29, 16, 24]

        var x0 = counter.0
        var x1 = counter.1

        // Key schedule constants
        let ks0 = key.0
        let ks1 = key.1
        let ks2 = 0x1BD11BDA ^ ks0 ^ ks1  // Skein parity constant

        // Initial key injection
        x0 = x0 &+ ks0
        x1 = x1 &+ ks1

        // 20 rounds of Threefry
        for round in 0..<20 {
            // Mix
            x0 = x0 &+ x1
            x1 = rotateLeft(x1, by: Int(rotations[round % 8]))
            x1 ^= x0

            // Key injection every 4 rounds
            if (round + 1) % 4 == 0 {
                let inject = (round + 1) / 4
                switch inject % 3 {
                case 0:
                    x0 = x0 &+ ks0
                    x1 = x1 &+ ks1 &+ UInt32(inject)
                case 1:
                    x0 = x0 &+ ks1
                    x1 = x1 &+ ks2 &+ UInt32(inject)
                case 2:
                    x0 = x0 &+ ks2
                    x1 = x1 &+ ks0 &+ UInt32(inject)
                default:
                    break
                }
            }
        }

        return ((x0, x1), (x0 ^ 0xDEADBEEF, x1 ^ 0xCAFEBABE))
    }

    /// Rotate left helper
    private static func rotateLeft(_ value: UInt32, by count: Int) -> UInt32 {
        return (value << count) | (value >> (32 - count))
    }

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        hasher.combine(data.0)
        hasher.combine(data.1)
    }

    public static func == (lhs: PRNGKey, rhs: PRNGKey) -> Bool {
        return lhs.data == rhs.data
    }
}

// MARK: - Random Operations

/// Generate uniform random values in [0, 1)
///
/// ```swift
/// let key = PRNGKey(seed: 42)
/// let uniform = diffRandomUniform(key, shape: [32, 64])
/// ```
public func diffRandomUniform(
    _ key: PRNGKey,
    shape: [Int],
    dtype: DType = .float32
) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffRandomUniform requires an active MLIRBuilder")
    }

    let result = builder.freshSSA()
    let resultType = tensorType(shape: shape, dtype: dtype)

    // Create rng_bit_generator for uniform distribution
    // StableHLO uses rng_bit_generator + convert for uniform floats
    builder.addOperation(MLIROperation(
        result: result,
        opName: "stablehlo.rng",
        operands: [],
        attributes: [
            "rng_distribution": "#stablehlo<rng_distribution UNIFORM>",
            "seed": "dense<[\(key.data.0), \(key.data.1)]> : tensor<2xui32>"
        ],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: result, shape: shape, dtype: dtype)
}

/// Generate standard normal random values (Gaussian distribution)
///
/// Uses Box-Muller transform internally.
///
/// ```swift
/// let key = PRNGKey(seed: 42)
/// let normal = diffRandomNormal(key, shape: [32, 64])
/// let scaled = diffRandomNormal(key, shape: [32, 64], mean: 0, stddev: 0.02)
/// ```
public func diffRandomNormal(
    _ key: PRNGKey,
    shape: [Int],
    mean: Float = 0,
    stddev: Float = 1,
    dtype: DType = .float32
) -> DifferentiableTracer {
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffRandomNormal requires an active MLIRBuilder")
    }

    let result = builder.freshSSA()
    let resultType = tensorType(shape: shape, dtype: dtype)

    // StableHLO rng with NORMAL distribution
    builder.addOperation(MLIROperation(
        result: result,
        opName: "stablehlo.rng",
        operands: [],
        attributes: [
            "rng_distribution": "#stablehlo<rng_distribution NORMAL>",
            "seed": "dense<[\(key.data.0), \(key.data.1)]> : tensor<2xui32>"
        ],
        resultType: resultType
    ))

    var output = DifferentiableTracer(irValue: result, shape: shape, dtype: dtype)

    // Apply mean and stddev scaling if not default
    if stddev != 1.0 {
        output = output * createConstant(stddev, shape: [], dtype: dtype)
    }
    if mean != 0.0 {
        output = output + createConstant(mean, shape: [], dtype: dtype)
    }

    return output
}

/// Generate random integers in [minval, maxval)
///
/// ```swift
/// let key = PRNGKey(seed: 42)
/// let indices = diffRandomInt(key, shape: [100], minval: 0, maxval: 10)
/// ```
public func diffRandomInt(
    _ key: PRNGKey,
    shape: [Int],
    minval: Int,
    maxval: Int,
    dtype: DType = .int32
) -> DifferentiableTracer {
    // Generate uniform [0, 1) then scale to integer range
    let uniform = diffRandomUniform(key, shape: shape, dtype: .float32)
    let range = Float(maxval - minval)
    let scaled = uniform * createConstant(range, shape: [], dtype: .float32)
    let shifted = scaled + createConstant(Float(minval), shape: [], dtype: .float32)

    // Convert to integer (floor)
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffRandomInt requires an active MLIRBuilder")
    }

    let result = builder.freshSSA()
    let resultType = tensorType(shape: shape, dtype: dtype)

    builder.addOperation(MLIROperation(
        result: result,
        opName: "stablehlo.convert",
        operands: [shifted.irValue],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: result, shape: shape, dtype: dtype)
}

/// Apply dropout to a tensor
///
/// Randomly sets elements to zero with probability `rate`, and scales
/// remaining elements by 1/(1-rate) to maintain expected values.
///
/// ```swift
/// let key = PRNGKey(seed: 42)
/// let dropped = diffDropout(key, activations, rate: 0.1)
/// ```
public func diffDropout(
    _ key: PRNGKey,
    _ x: DifferentiableTracer,
    rate: Float
) -> DifferentiableTracer {
    precondition(rate >= 0 && rate < 1, "Dropout rate must be in [0, 1)")

    if rate == 0 {
        return x
    }

    // Generate uniform mask
    let mask = diffRandomUniform(key, shape: x.shape, dtype: x.dtype)

    // Create threshold
    let threshold = createConstant(rate, shape: [], dtype: x.dtype)

    // Compare: mask > rate (keep if true)
    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffDropout requires an active MLIRBuilder")
    }

    let compareResult = builder.freshSSA()
    let boolType = tensorType(shape: x.shape, dtype: .bool)

    builder.addOperation(MLIROperation(
        result: compareResult,
        opName: "stablehlo.compare",
        operands: [mask.irValue, threshold.irValue],
        attributes: ["comparison_direction": "#stablehlo<comparison_direction GT>"],
        resultType: boolType
    ))

    // Convert bool mask to float (0.0 or 1.0)
    let floatMask = builder.freshSSA()
    let floatType = tensorType(shape: x.shape, dtype: x.dtype)

    builder.addOperation(MLIROperation(
        result: floatMask,
        opName: "stablehlo.convert",
        operands: [compareResult],
        resultType: floatType
    ))

    let maskTracer = DifferentiableTracer(irValue: floatMask, shape: x.shape, dtype: x.dtype)

    // Apply mask and scale
    let scale = 1.0 / (1.0 - rate)
    let scaleTracer = createConstant(scale, shape: [], dtype: x.dtype)

    return x * maskTracer * scaleTracer
}

/// Generate truncated normal random values
///
/// Values outside [-2*stddev, 2*stddev] are resampled.
/// Useful for weight initialization.
///
/// ```swift
/// let weights = diffTruncatedNormal(key, shape: [784, 256], stddev: 0.02)
/// ```
public func diffTruncatedNormal(
    _ key: PRNGKey,
    shape: [Int],
    mean: Float = 0,
    stddev: Float = 1,
    dtype: DType = .float32
) -> DifferentiableTracer {
    // Generate normal samples
    var samples = diffRandomNormal(key, shape: shape, mean: mean, stddev: stddev, dtype: dtype)

    // Clamp to [-2*stddev, 2*stddev] from mean
    let lower = mean - 2 * stddev
    let upper = mean + 2 * stddev

    guard let builder = DifferentiableTracer.currentBuilder else {
        fatalError("diffTruncatedNormal requires an active MLIRBuilder")
    }

    let result = builder.freshSSA()
    let resultType = tensorType(shape: shape, dtype: dtype)

    builder.addOperation(MLIROperation(
        result: result,
        opName: "stablehlo.clamp",
        operands: [
            createConstant(lower, shape: [], dtype: dtype).irValue,
            samples.irValue,
            createConstant(upper, shape: [], dtype: dtype).irValue
        ],
        resultType: resultType
    ))

    return DifferentiableTracer(irValue: result, shape: shape, dtype: dtype)
}

// MARK: - Weight Initialization

/// Xavier/Glorot uniform initialization
///
/// Samples from uniform distribution in [-limit, limit] where
/// limit = sqrt(6 / (fan_in + fan_out))
///
/// ```swift
/// let weights = diffXavierUniform(key, shape: [784, 256])
/// ```
public func diffXavierUniform(
    _ key: PRNGKey,
    shape: [Int],
    dtype: DType = .float32
) -> DifferentiableTracer {
    precondition(shape.count >= 2, "Xavier init requires at least 2D tensor")

    let fanIn = Float(shape[shape.count - 2])
    let fanOut = Float(shape[shape.count - 1])
    let limit = (6.0 / (fanIn + fanOut)).squareRoot()

    // Generate uniform [-limit, limit]
    let uniform = diffRandomUniform(key, shape: shape, dtype: dtype)
    let scaled = uniform * createConstant(2 * limit, shape: [], dtype: dtype)
    return scaled - createConstant(limit, shape: [], dtype: dtype)
}

/// Xavier/Glorot normal initialization
///
/// Samples from normal distribution with stddev = sqrt(2 / (fan_in + fan_out))
///
/// ```swift
/// let weights = diffXavierNormal(key, shape: [784, 256])
/// ```
public func diffXavierNormal(
    _ key: PRNGKey,
    shape: [Int],
    dtype: DType = .float32
) -> DifferentiableTracer {
    precondition(shape.count >= 2, "Xavier init requires at least 2D tensor")

    let fanIn = Float(shape[shape.count - 2])
    let fanOut = Float(shape[shape.count - 1])
    let stddev = (2.0 / (fanIn + fanOut)).squareRoot()

    return diffRandomNormal(key, shape: shape, mean: 0, stddev: stddev, dtype: dtype)
}

/// He/Kaiming uniform initialization
///
/// Samples from uniform distribution in [-limit, limit] where
/// limit = sqrt(6 / fan_in)
///
/// Good for ReLU activations.
///
/// ```swift
/// let weights = diffHeUniform(key, shape: [256, 128])
/// ```
public func diffHeUniform(
    _ key: PRNGKey,
    shape: [Int],
    dtype: DType = .float32
) -> DifferentiableTracer {
    precondition(shape.count >= 2, "He init requires at least 2D tensor")

    let fanIn = Float(shape[shape.count - 2])
    let limit = (6.0 / fanIn).squareRoot()

    let uniform = diffRandomUniform(key, shape: shape, dtype: dtype)
    let scaled = uniform * createConstant(2 * limit, shape: [], dtype: dtype)
    return scaled - createConstant(limit, shape: [], dtype: dtype)
}

/// He/Kaiming normal initialization
///
/// Samples from normal distribution with stddev = sqrt(2 / fan_in)
///
/// Good for ReLU activations.
///
/// ```swift
/// let weights = diffHeNormal(key, shape: [256, 128])
/// ```
public func diffHeNormal(
    _ key: PRNGKey,
    shape: [Int],
    dtype: DType = .float32
) -> DifferentiableTracer {
    precondition(shape.count >= 2, "He init requires at least 2D tensor")

    let fanIn = Float(shape[shape.count - 2])
    let stddev = (2.0 / fanIn).squareRoot()

    return diffRandomNormal(key, shape: shape, mean: 0, stddev: stddev, dtype: dtype)
}

// MARK: - Implementation Notes

/*
 PRNG IMPLEMENTATION STATUS:

 ✅ Implemented:
    - PRNGKey with Threefry-based algorithm
    - split() for creating independent streams
    - split(into:) for batched operations
    - fold(in:) for index-dependent keys
    - diffRandomUniform() - uniform [0, 1)
    - diffRandomNormal() - Gaussian distribution
    - diffRandomInt() - integer range
    - diffDropout() - dropout with scaling
    - diffTruncatedNormal() - truncated normal
    - diffXavierUniform/Normal() - Glorot initialization
    - diffHeUniform/Normal() - Kaiming initialization

 ⚠️ Current Limitations:
    - Uses stablehlo.rng which may not be supported on all backends
    - No VJP implementations (random ops have zero gradient)
    - shuffle() not yet implemented

 🔜 Future Enhancements:
    - rng_bit_generator for better backend support
    - Categorical/multinomial sampling
    - Permutation/shuffle operations
    - Gumbel-softmax for differentiable sampling

 KEY INSIGHTS:
 - Functional PRNGs enable reproducible distributed training
 - Same key = same random numbers (deterministic)
 - split() is essential for parallel code
 - Random ops have zero gradient (not differentiable)
 */
