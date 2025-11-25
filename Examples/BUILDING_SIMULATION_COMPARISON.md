# Building Thermal Simulation: SwiftIR vs Standard Swift

This example compares two approaches to automatic differentiation in Swift for physics-based simulation, inspired by [PassiveLogic's differentiable-swift-examples](https://github.com/PassiveLogic/differentiable-swift-examples).

## What This Demonstrates

A **building thermal dynamics simulation** with automatic differentiation:

- **Physics**: Energy balance equation for radiant floor heating
- **Numerical method**: Euler integration (20 timesteps, 0.1s each)
- **Autodiff**: Compute gradients for parameter optimization
- **Benchmark**: Compare forward pass vs gradient computation performance

## Two Implementations

### 1. SwiftIR Implementation ([BuildingSimulation_Simple.swift](BuildingSimulation_Simple.swift))

**Approach:**
```
Swift @differentiable code
         ↓
   Symbolic tracing
         ↓
   MLIR StableHLO IR
         ↓
   XLA compilation
         ↓
   PJRT execution
```

**Characteristics:**
- ✅ One-time compilation cost (~200-500ms)
- ✅ XLA optimizations (fusion, vectorization)
- ✅ Portable IR (can run on GPU/TPU)
- ✅ Same runtime as JAX/TensorFlow
- ⚠️ Compilation overhead for small workloads

**Run:**
```bash
cd SwiftIR
source /etc/profile.d/swiftir.sh
swift run BuildingSimulation_Simple
```

### 2. Standard Swift Implementation (PassiveLogic)

**Approach:**
```
Swift @differentiable code
         ↓
Swift compiler's AD
         ↓
Native Swift execution
```

**Characteristics:**
- ✅ No compilation overhead
- ✅ Pure Swift (no external dependencies)
- ✅ Fast for small workloads
- ⚠️ Limited to CPU execution
- ⚠️ No cross-framework portability

**Run:**
```bash
git clone https://github.com/PassiveLogic/differentiable-swift-examples
cd differentiable-swift-examples/Benchmarks/BuildingSimulation/Swift
swift run
```

## Expected Performance Comparison

### SwiftIR (This Implementation)

```
Compilation: ~200-500ms (one-time)
Forward pass: ~50-200μs
Gradient:     ~150-600μs
Overhead:     2-3x
```

### Standard Swift (PassiveLogic)

```
Compilation: N/A (interpreted)
Forward pass: ~5-20μs
Gradient:     ~30-100μs
Overhead:     3-5x
```

## Key Differences

| Aspect | SwiftIR | Standard Swift |
|--------|---------|----------------|
| **Compilation** | One-time MLIR/XLA compilation | No compilation |
| **Optimizations** | XLA fusion, vectorization | Swift compiler optimizations |
| **Portability** | StableHLO (GPU/TPU ready) | CPU only |
| **Small workloads** | Slower (compilation cost) | Faster |
| **Large workloads** | Faster (XLA optimizations) | Slower |
| **Scalability** | Excellent (hardware acceleration) | Limited (CPU bound) |

## When to Use Each

### Use SwiftIR when:
- ✅ Running many simulations (amortize compilation)
- ✅ Need GPU/TPU acceleration
- ✅ Want portable IR for deployment
- ✅ Building production ML systems
- ✅ Integrating with JAX/TensorFlow ecosystem

### Use Standard Swift when:
- ✅ One-off simulations or prototyping
- ✅ Pure Swift environment (no MLIR/XLA)
- ✅ Small, quick experiments
- ✅ Educational/learning purposes
- ✅ Minimal dependencies required

## Physics Model

### Simplified Thermal Dynamics

```
dT/dt = (P_heating - P_loss) / (m × Cp)

Where:
  T          = temperature (°C)
  P_heating  = heating power input (W)
  P_loss     = heat loss to ambient (W)
  m          = mass (kg)
  Cp         = specific heat capacity (J/kg·K)
```

### Discretization

```swift
// Euler forward integration
T(t+dt) = T(t) + (P_net × dt) / (m × Cp)

Where:
  P_net = P_heating - k × (T - T_ambient)
  k     = thermal conductance (W/K)
  dt    = timestep (0.1s)
```

### Parameters

```
Slab (concrete floor):
  • Mass: 22,700 kg (100m² × 0.1m × 2,270 kg/m³)
  • Specific heat: 880 J/(kg·K)
  • Thermal mass: 19.98 MJ/K

Heating system:
  • Power: 1,000 W (radiant floor)

Environment:
  • Ambient temp: 15°C
  • Heat loss: 100 W/K

Target:
  • Final temperature: 27.345°C (after 2 seconds)
```

## Gradient Computation

Both implementations compute gradients of the **loss function** with respect to **all input parameters**:

```swift
@differentiable(reverse)
func computeLoss(
    _ initialTemp: Float,
    _ heatingPower: Float,
    _ ambientTemp: Float,
    _ thermalMass: Float,
    _ thermalConductance: Float
) -> Float {
    let finalTemp = runSimulation(...)
    return (finalTemp - targetTemp)²  // L2 loss
}

// Compute gradients
let grads = gradient(at: initialTemp, heatingPower, ..., of: computeLoss)
```

These gradients enable:
- **Parameter estimation**: Learn unknown physical parameters from observations
- **Optimal control**: Find heating schedule to reach target temperature
- **Model calibration**: Tune model to match real-world measurements

## Use Cases for This Approach

### 1. Building Energy Optimization
Learn optimal heating/cooling schedules to minimize energy while maintaining comfort.

### 2. System Identification
Estimate physical parameters (thermal mass, insulation) from temperature measurements.

### 3. Predictive Control
Forecast future temperatures and optimize control actions.

### 4. Digital Twins
Create differentiable models of physical systems for simulation and optimization.

## Extended Example: Parameter Estimation

```swift
// Given: observed temperature after 2s = 27.345°C
// Unknown: actual heating power (nominal = 1000W)

var heatingPower: Float = 800.0  // Initial guess
let learningRate: Float = 0.1

for epoch in 0..<100 {
    let grads = gradient(at: heatingPower) { power in
        computeLoss(20.0, power, 15.0, 19_976_000, 100.0)
    }

    // Gradient descent
    heatingPower -= learningRate * grads

    if epoch % 10 == 0 {
        let loss = computeLoss(20.0, heatingPower, 15.0, 19_976_000, 100.0)
        print("Epoch \(epoch): power = \(heatingPower)W, loss = \(loss)")
    }
}

print("Learned heating power: \(heatingPower)W")
```

## Running the Comparison

### Quick Comparison

```bash
# 1. Run SwiftIR version
cd SwiftIR
source /etc/profile.d/swiftir.sh
time swift run BuildingSimulation_Simple > swiftir_results.txt

# 2. Run standard Swift version
cd /tmp
git clone https://github.com/PassiveLogic/differentiable-swift-examples
cd differentiable-swift-examples/Benchmarks/BuildingSimulation/Swift
time swift run > standard_results.txt

# 3. Compare
diff swiftir_results.txt standard_results.txt
```

### Analyzing Results

Look for:
- **Compilation time**: SwiftIR has upfront cost
- **Execution time**: Per-trial forward/gradient times
- **Gradient overhead**: Ratio of gradient/forward time
- **Accuracy**: Both should achieve similar loss values

## Implementation Notes

### SwiftIR Specific

1. **Scalar operations**: SwiftIR works with scalar tracers (`shape: []`)
2. **Unrolled loops**: 20 timesteps explicitly unrolled (current limitation)
3. **Compilation**: One-time MLIR/XLA compilation before execution
4. **Type safety**: All operations verified at Swift compile time

### Standard Swift Specific

1. **Differentiable protocol**: All types conform to `Differentiable`
2. **Derivatives**: `@derivative` attribute defines gradient rules
3. **Value semantics**: Structs (not classes) for efficiency
4. **Pure Swift**: No external dependencies beyond `_Differentiation`

## Future Enhancements

### For SwiftIR Version

- [ ] Add control flow (proper loops instead of unrolling)
- [ ] Multi-zone thermal model (tensor operations)
- [ ] GPU execution (when GPU support is enabled)
- [ ] Batched simulation (multiple scenarios in parallel)
- [ ] Higher-order derivatives (Hessian for optimization)

### For Comparison

- [ ] Larger simulation (100+ zones)
- [ ] Longer timesteps (100+ steps)
- [ ] Batch processing (1000+ scenarios)
- [ ] Memory usage comparison
- [ ] Energy consumption measurement

## References

### SwiftIR
- [SwiftIR Repository](https://github.com/pedronahum/SwiftIR)
- [SymbolicAD Documentation](../Sources/SwiftIR/SymbolicAD/README.md)
- [XLA Compiler](https://www.tensorflow.org/xla)
- [PJRT Runtime](https://github.com/openxla/xla/tree/main/xla/pjrt)

### PassiveLogic Example
- [Differentiable Swift Examples](https://github.com/PassiveLogic/differentiable-swift-examples)
- [Building Simulation Code](https://github.com/PassiveLogic/differentiable-swift-examples/blob/main/Benchmarks/BuildingSimulation/Swift/main.swift)

### Swift Differentiation
- [Swift Differentiation Documentation](https://github.com/apple/swift/blob/main/docs/DifferentiableProgramming.md)
- [Swift for TensorFlow](https://github.com/tensorflow/swift)

## License

This example is released under Apache 2.0 License, same as SwiftIR.

The PassiveLogic comparison code is used for educational purposes under their license.

---

**SwiftIR: Modern ML compilation in Swift - Type-safe, portable, and production-ready.**
