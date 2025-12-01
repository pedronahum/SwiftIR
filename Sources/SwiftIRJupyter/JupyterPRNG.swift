// JupyterPRNG.swift - Functional (Splittable) Pseudo-Random Number Generation
// Pure Swift - works without C++ interop via dlopen/dlsym
//
// This file implements JAX-style functional PRNGs that enable reproducible
// randomness in parallel/vectorized code. Unlike stateful PRNGs, functional
// PRNGs are explicit and composable.
//
// Based on JAX's PRNG: https://jax.readthedocs.io/en/latest/jax.random.html

import Foundation

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
/// let key = JPRNGKey(seed: 42)
/// let (key1, key2) = key.split()
/// let noise = jRandomNormal(key1, shape: JTensorShape([32, 64]))
/// let mask = jRandomUniform(key2, shape: JTensorShape([32, 64]))
/// ```
public struct JPRNGKey: Hashable, Sendable {
    /// Internal state (Threefry 2x32)
    public let data: (UInt32, UInt32)

    /// Create a new PRNG key from a seed
    public init(seed: UInt64) {
        self.data = JPRNGKey.threefryInit(seed)
    }

    /// Internal initializer from raw data
    public init(data: (UInt32, UInt32)) {
        self.data = data
    }

    /// Split key into two independent keys
    ///
    /// This is the core operation for functional PRNGs. The two resulting
    /// keys will produce independent random streams.
    ///
    /// ```swift
    /// let key = JPRNGKey(seed: 42)
    /// let (trainKey, evalKey) = key.split()
    /// ```
    public func split() -> (JPRNGKey, JPRNGKey) {
        let (a, b) = JPRNGKey.threefryRandom2x32(data, counter: (0, 0))
        let (c, d) = JPRNGKey.threefryRandom2x32(data, counter: (1, 0))
        return (JPRNGKey(data: a), JPRNGKey(data: (c.0, d.0)))
    }

    /// Split key into n independent keys
    ///
    /// Useful for batched operations where each element needs its own key.
    ///
    /// ```swift
    /// let keys = key.split(into: batchSize)
    /// ```
    public func split(into n: Int) -> [JPRNGKey] {
        var keys: [JPRNGKey] = []
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
    public func fold(in data: UInt32) -> JPRNGKey {
        let newData = JPRNGKey.threefryRandom2x32(self.data, counter: (data, 0))
        return JPRNGKey(data: newData.0)
    }

    // MARK: - Threefry Implementation

    /// Initialize Threefry state from seed
    private static func threefryInit(_ seed: UInt64) -> (UInt32, UInt32) {
        let high = UInt32(truncatingIfNeeded: seed >> 32)
        let low = UInt32(truncatingIfNeeded: seed)
        return threefryRandom2x32((high, low), counter: (0, 0)).0
    }

    /// Threefry 2x32 random number generator
    /// Based on the Threefry algorithm from "Parallel Random Numbers: As Easy as 1, 2, 3"
    private static func threefryRandom2x32(
        _ key: (UInt32, UInt32),
        counter: (UInt32, UInt32)
    ) -> ((UInt32, UInt32), (UInt32, UInt32)) {
        let rotations: [UInt32] = [13, 15, 26, 6, 17, 29, 16, 24]

        var x0 = counter.0
        var x1 = counter.1

        let ks0 = key.0
        let ks1 = key.1
        let ks2 = 0x1BD11BDA ^ ks0 ^ ks1

        x0 = x0 &+ ks0
        x1 = x1 &+ ks1

        for round in 0..<20 {
            x0 = x0 &+ x1
            x1 = rotateLeft(x1, by: Int(rotations[round % 8]))
            x1 ^= x0

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

    private static func rotateLeft(_ value: UInt32, by count: Int) -> UInt32 {
        return (value << count) | (value >> (32 - count))
    }

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        hasher.combine(data.0)
        hasher.combine(data.1)
    }

    public static func == (lhs: JPRNGKey, rhs: JPRNGKey) -> Bool {
        return lhs.data == rhs.data
    }
}

// MARK: - Random Operations

/// Generate uniform random values in [0, 1)
///
/// ```swift
/// let key = JPRNGKey(seed: 42)
/// let uniform = jRandomUniform(key, shape: JTensorShape([32, 64]))
/// ```
public func jRandomUniform(
    _ key: JPRNGKey,
    shape: JTensorShape,
    dtype: JDType = .float32
) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let resultId = builder.createRng(
        key: key,
        shape: shape,
        dtype: dtype,
        distribution: .uniform
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: shape,
        dtype: dtype,
        version: JTracer.incrementVersion()
    )
}

/// Generate standard normal random values (Gaussian distribution)
///
/// ```swift
/// let key = JPRNGKey(seed: 42)
/// let normal = jRandomNormal(key, shape: JTensorShape([32, 64]))
/// let scaled = jRandomNormal(key, shape: JTensorShape([32, 64]), mean: 0, stddev: 0.02)
/// ```
public func jRandomNormal(
    _ key: JPRNGKey,
    shape: JTensorShape,
    mean: Float = 0,
    stddev: Float = 1,
    dtype: JDType = .float32
) -> JTracer {
    let builder = JTracerGraphBuilder.shared

    let resultId = builder.createRng(
        key: key,
        shape: shape,
        dtype: dtype,
        distribution: .normal
    )

    var output = JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: shape,
        dtype: dtype,
        version: JTracer.incrementVersion()
    )

    // Apply mean and stddev scaling if not default
    if stddev != 1.0 {
        let scale = JTracer(value: Double(stddev), shape: JTensorShape([1]), dtype: dtype)
        output = output * scale
    }
    if mean != 0.0 {
        let offset = JTracer(value: Double(mean), shape: JTensorShape([1]), dtype: dtype)
        output = output + offset
    }

    return output
}

/// Generate random integers in [minval, maxval)
///
/// ```swift
/// let key = JPRNGKey(seed: 42)
/// let indices = jRandomInt(key, shape: JTensorShape([100]), minval: 0, maxval: 10)
/// ```
public func jRandomInt(
    _ key: JPRNGKey,
    shape: JTensorShape,
    minval: Int,
    maxval: Int,
    dtype: JDType = .int32
) -> JTracer {
    // Generate uniform [0, 1) then scale to integer range
    let uniform = jRandomUniform(key, shape: shape, dtype: .float32)

    let range = JTracer(value: Double(maxval - minval), shape: JTensorShape([1]), dtype: .float32)
    let scaled = uniform * range

    let offset = JTracer(value: Double(minval), shape: JTensorShape([1]), dtype: .float32)
    let shifted = scaled + offset

    // Convert to integer (floor via convert)
    let builder = JTracerGraphBuilder.shared
    let resultId = builder.createConvert(
        input: shifted.valueId,
        inputDtype: .float32,
        outputDtype: dtype,
        shape: shape
    )

    return JTracer(
        irValue: JMLIRValue(id: resultId),
        shape: shape,
        dtype: dtype,
        version: JTracer.incrementVersion()
    )
}

/// Apply dropout to a tensor
///
/// Randomly sets elements to zero with probability `rate`, and scales
/// remaining elements by 1/(1-rate) to maintain expected values.
///
/// ```swift
/// let key = JPRNGKey(seed: 42)
/// let dropped = jDropout(key, activations, rate: 0.1)
/// ```
public func jDropout(
    _ key: JPRNGKey,
    _ x: JTracer,
    rate: Float
) -> JTracer {
    precondition(rate >= 0 && rate < 1, "Dropout rate must be in [0, 1)")

    if rate == 0 {
        return x
    }

    // Generate uniform mask
    let mask = jRandomUniform(key, shape: x.shape, dtype: x.dtype)

    // Create threshold and compare
    let threshold = JTracer(value: Double(rate), shape: JTensorShape([1]), dtype: x.dtype)
    let keepMask = jGreater(mask, threshold)

    // Convert bool mask to float (0.0 or 1.0)
    let builder = JTracerGraphBuilder.shared
    let floatMaskId = builder.createConvert(
        input: keepMask.valueId,
        inputDtype: .bool,
        outputDtype: x.dtype,
        shape: x.shape
    )

    let floatMask = JTracer(
        irValue: JMLIRValue(id: floatMaskId),
        shape: x.shape,
        dtype: x.dtype,
        version: JTracer.incrementVersion()
    )

    // Apply mask and scale
    let scale = 1.0 / (1.0 - Double(rate))
    let scaleTracer = JTracer(value: scale, shape: JTensorShape([1]), dtype: x.dtype)

    return x * floatMask * scaleTracer
}

/// Generate truncated normal random values
///
/// Values outside [-2*stddev, 2*stddev] are clamped.
///
/// ```swift
/// let weights = jTruncatedNormal(key, shape: JTensorShape([784, 256]), stddev: 0.02)
/// ```
public func jTruncatedNormal(
    _ key: JPRNGKey,
    shape: JTensorShape,
    mean: Float = 0,
    stddev: Float = 1,
    dtype: JDType = .float32
) -> JTracer {
    let samples = jRandomNormal(key, shape: shape, mean: mean, stddev: stddev, dtype: dtype)

    let lower = Double(mean - 2 * stddev)
    let upper = Double(mean + 2 * stddev)

    return jClamp(samples, min: lower, max: upper)
}

// MARK: - Weight Initialization

/// Xavier/Glorot uniform initialization
///
/// ```swift
/// let weights = jXavierUniform(key, shape: JTensorShape([784, 256]))
/// ```
public func jXavierUniform(
    _ key: JPRNGKey,
    shape: JTensorShape,
    dtype: JDType = .float32
) -> JTracer {
    let dims = shape.dimensions.compactMap { $0 }
    precondition(dims.count >= 2, "Xavier init requires at least 2D tensor")

    let fanIn = Float(dims[dims.count - 2])
    let fanOut = Float(dims[dims.count - 1])
    let limit = sqrt(6.0 / (fanIn + fanOut))

    let uniform = jRandomUniform(key, shape: shape, dtype: dtype)
    let scale = JTracer(value: Double(2 * limit), shape: JTensorShape([1]), dtype: dtype)
    let offset = JTracer(value: Double(limit), shape: JTensorShape([1]), dtype: dtype)

    return uniform * scale - offset
}

/// Xavier/Glorot normal initialization
///
/// ```swift
/// let weights = jXavierNormal(key, shape: JTensorShape([784, 256]))
/// ```
public func jXavierNormal(
    _ key: JPRNGKey,
    shape: JTensorShape,
    dtype: JDType = .float32
) -> JTracer {
    let dims = shape.dimensions.compactMap { $0 }
    precondition(dims.count >= 2, "Xavier init requires at least 2D tensor")

    let fanIn = Float(dims[dims.count - 2])
    let fanOut = Float(dims[dims.count - 1])
    let stddev = sqrt(2.0 / (fanIn + fanOut))

    return jRandomNormal(key, shape: shape, mean: 0, stddev: stddev, dtype: dtype)
}

/// He/Kaiming uniform initialization (good for ReLU)
///
/// ```swift
/// let weights = jHeUniform(key, shape: JTensorShape([256, 128]))
/// ```
public func jHeUniform(
    _ key: JPRNGKey,
    shape: JTensorShape,
    dtype: JDType = .float32
) -> JTracer {
    let dims = shape.dimensions.compactMap { $0 }
    precondition(dims.count >= 2, "He init requires at least 2D tensor")

    let fanIn = Float(dims[dims.count - 2])
    let limit = sqrt(6.0 / fanIn)

    let uniform = jRandomUniform(key, shape: shape, dtype: dtype)
    let scale = JTracer(value: Double(2 * limit), shape: JTensorShape([1]), dtype: dtype)
    let offset = JTracer(value: Double(limit), shape: JTensorShape([1]), dtype: dtype)

    return uniform * scale - offset
}

/// He/Kaiming normal initialization (good for ReLU)
///
/// ```swift
/// let weights = jHeNormal(key, shape: JTensorShape([256, 128]))
/// ```
public func jHeNormal(
    _ key: JPRNGKey,
    shape: JTensorShape,
    dtype: JDType = .float32
) -> JTracer {
    let dims = shape.dimensions.compactMap { $0 }
    precondition(dims.count >= 2, "He init requires at least 2D tensor")

    let fanIn = Float(dims[dims.count - 2])
    let stddev = sqrt(2.0 / fanIn)

    return jRandomNormal(key, shape: shape, mean: 0, stddev: stddev, dtype: dtype)
}

// MARK: - RNG Distribution Enum

/// Distribution type for random number generation
public enum JRngDistribution: String {
    case uniform = "UNIFORM"
    case normal = "NORMAL"
}

// MARK: - Graph Builder Extensions

extension JTracerGraphBuilder {
    /// Create a random number generation operation
    public func createRng(
        key: JPRNGKey,
        shape: JTensorShape,
        dtype: JDType,
        distribution: JRngDistribution
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.rng(
            id: id,
            keyData: key.data,
            shape: shape,
            dtype: dtype,
            distribution: distribution
        )
        addOperation(op)
        return id
    }

    /// Create a convert operation (for type casting)
    public func createConvert(
        input: UInt64,
        inputDtype: JDType,
        outputDtype: JDType,
        shape: JTensorShape
    ) -> UInt64 {
        let id = getNextId()

        let op = JTracedOperation.convert(
            id: id,
            input: input,
            inputDtype: inputDtype,
            outputDtype: outputDtype,
            shape: shape
        )
        addOperation(op)
        return id
    }
}

// MARK: - Implementation Notes

/*
 JPRNG IMPLEMENTATION STATUS:

 ✅ Implemented:
    - JPRNGKey with Threefry-based algorithm
    - split() for creating independent streams
    - split(into:) for batched operations
    - fold(in:) for index-dependent keys
    - jRandomUniform() - uniform [0, 1)
    - jRandomNormal() - Gaussian distribution
    - jRandomInt() - integer range
    - jDropout() - dropout with scaling
    - jTruncatedNormal() - truncated normal
    - jXavierUniform/Normal() - Glorot initialization
    - jHeUniform/Normal() - Kaiming initialization

 ⚠️ Current Limitations:
    - Requires adding rng and convert operations to JTracedOperation
    - No VJP implementations (random ops have zero gradient)
    - shuffle() not yet implemented

 🔜 Future Enhancements:
    - Categorical/multinomial sampling
    - Permutation/shuffle operations
    - Gumbel-softmax for differentiable sampling

 KEY INSIGHTS:
 - Functional PRNGs enable reproducible distributed training
 - Same key = same random numbers (deterministic)
 - split() is essential for parallel code
 - Random ops have zero gradient (not differentiable)
 */
