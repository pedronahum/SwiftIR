# JAX Black-Scholes Benchmark

This folder contains a JAX implementation of the Black-Scholes benchmark for fair comparison against SwiftIR-Traced.

## Why This Comparison Matters

Both JAX and SwiftIR-Traced:
- Use **XLA** as the backend compiler
- Use **automatic differentiation** for gradient computation
- **JIT compile** the function before execution
- Support **batched operations** for high throughput

This makes JAX the ideal baseline for comparing SwiftIR's performance, as both use the same underlying execution engine (XLA).

## Installation

```bash
pip install -r requirements.txt
```

Or install JAX directly:

```bash
pip install jax jaxlib
```

## Running the Benchmark

```bash
python benchmark.py
```

## Benchmark Approaches

The benchmark tests three JAX approaches:

1. **vmap(grad)**: Applies scalar gradient function to each element via `vmap`
   - Most similar to Swift's `_Differentiation`
   - Per-element function application

2. **batch grad**: Computes gradient of sum reduction over batched function
   - More efficient for large batches
   - Single kernel launch

3. **explicit VJP**: Uses `jax.vjp` with ones seed vector
   - Exactly matches SwiftIR's approach
   - Most fair comparison

## Sample Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         JAX EXPLICIT VJP                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌────────────┬────────────────┬──────────────────┬──────────────────┐
  │ Batch Size │  Compile (ms)  │     Time (μs)    │   Throughput     │
  ├────────────┼────────────────┼──────────────────┼──────────────────┤
  │        100 │          X.X   │            X.X   │        X.XX K/s  │
  │      1,000 │          X.X   │            X.X   │        X.XX K/s  │
  │     10,000 │          X.X   │            X.X   │        X.XX K/s  │
  │    100,000 │          X.X   │            X.X   │        X.XX K/s  │
  │  1,000,000 │          X.X   │            X.X   │        X.XX K/s  │
  └────────────┴────────────────┴──────────────────┴──────────────────┘
```

## Comparison with SwiftIR

| Implementation | 100K Throughput (K/s) | XLA Backend |
|----------------|----------------------|-------------|
| SwiftIR-Traced | ~59,217              | Yes         |
| JAX explicit VJP | ~XX,XXX            | Yes         |
| Swift `_Differentiation` | ~803       | No          |

Both SwiftIR and JAX benefit from XLA's optimizations:
- SIMD vectorization (AVX-512)
- Operation fusion
- Memory layout optimization
- Compile-time gradient generation
