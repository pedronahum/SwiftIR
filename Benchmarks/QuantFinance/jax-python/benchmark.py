#!/usr/bin/env python3
"""
JAX Black-Scholes Benchmark
===========================

This benchmark implements Black-Scholes option pricing with automatic
differentiation (Delta computation) using JAX with JIT compilation.

This provides a fair comparison against SwiftIR-Traced, as both:
- Use XLA as the backend compiler
- Use automatic differentiation for gradient computation
- Compile the function before execution (JIT)

Key features:
- Separates compile time from execution time
- Can dump HLO/StableHLO IR for comparison with SwiftIR's MLIR

Usage:
    pip install jax jaxlib
    python benchmark.py
    python benchmark.py --dump-hlo  # Dump HLO IR
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import time
from functools import partial
from typing import Tuple, List
import numpy as np
import argparse
import os

# Ensure we're using CPU for fair comparison
jax.config.update('jax_platform_name', 'cpu')

# ============================================================================
# Black-Scholes Implementation
# ============================================================================

def normal_cdf_approx(x: jnp.ndarray) -> jnp.ndarray:
    """Approximate Normal CDF using sigmoid (same as Swift implementation)."""
    return jax.nn.sigmoid(x * 1.702)


def black_scholes_call_single(
    spot: float,
    strike: float = 100.0,
    rate: float = 0.05,
    maturity: float = 1.0,
    volatility: float = 0.2
) -> float:
    """Black-Scholes call option price (scalar version)."""
    stdev = volatility * jnp.sqrt(maturity)
    log_sk = jnp.log(spot / strike)
    drift = (rate + 0.5 * volatility * volatility) * maturity
    d1 = (log_sk + drift) / stdev
    d2 = d1 - stdev
    df = jnp.exp(-rate * maturity)

    nd1 = normal_cdf_approx(d1)
    nd2 = normal_cdf_approx(d2)

    return spot * nd1 - strike * df * nd2


def black_scholes_call_batch(
    spots: jnp.ndarray,
    strike: float = 100.0,
    rate: float = 0.05,
    maturity: float = 1.0,
    volatility: float = 0.2
) -> jnp.ndarray:
    """Black-Scholes call option prices (batched version)."""
    stdev = volatility * jnp.sqrt(maturity)
    log_sk = jnp.log(spots / strike)
    drift = (rate + 0.5 * volatility * volatility) * maturity
    d1 = (log_sk + drift) / stdev
    d2 = d1 - stdev
    df = jnp.exp(-rate * maturity)

    nd1 = normal_cdf_approx(d1)
    nd2 = normal_cdf_approx(d2)

    return spots * nd1 - strike * df * nd2


# ============================================================================
# Gradient Functions (not JIT-decorated - we control compilation explicitly)
# ============================================================================

def price_and_delta_vmap(spots: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute prices and deltas using vmap over scalar gradient."""
    grad_fn = value_and_grad(black_scholes_call_single)
    prices, deltas = vmap(grad_fn)(spots)
    return prices, deltas


def price_and_delta_batch(spots: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute prices and deltas using batched gradient with sum reduction."""
    def loss_fn(spots):
        return black_scholes_call_batch(spots).sum()

    prices = black_scholes_call_batch(spots)
    deltas = grad(loss_fn)(spots)
    return prices, deltas


def price_and_delta_explicit(spots: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute prices and deltas using explicit VJP with ones seed."""
    prices, vjp_fn = jax.vjp(black_scholes_call_batch, spots)
    # Use ones as seed (same as SwiftIR's seed vector)
    deltas, = vjp_fn(jnp.ones_like(prices))
    return prices, deltas


def price_only(spots: jnp.ndarray) -> jnp.ndarray:
    """Compute prices only (no gradient)."""
    return black_scholes_call_batch(spots)


# ============================================================================
# HLO/StableHLO Dumping
# ============================================================================

def dump_hlo(fn, spots: jnp.ndarray, name: str, output_dir: str = "hlo_dumps"):
    """Dump HLO IR for a function."""
    os.makedirs(output_dir, exist_ok=True)

    # Lower to StableHLO
    lowered = jit(fn).lower(spots)

    # Get StableHLO text
    stablehlo_text = lowered.as_text()
    stablehlo_path = os.path.join(output_dir, f"{name}_stablehlo.txt")
    with open(stablehlo_path, "w") as f:
        f.write(stablehlo_text)
    print(f"  Wrote StableHLO to: {stablehlo_path}")

    # Compile and get HLO
    compiled = lowered.compile()

    # Get HLO text (optimized)
    hlo_text = compiled.as_text()
    hlo_path = os.path.join(output_dir, f"{name}_hlo.txt")
    with open(hlo_path, "w") as f:
        f.write(hlo_text)
    print(f"  Wrote HLO to: {hlo_path}")

    return stablehlo_text, hlo_text


# ============================================================================
# Benchmark Infrastructure with Separated Compile/Execute Timing
# ============================================================================

def run_benchmark_separated(
    fn,
    spots: jnp.ndarray,
    trials: int = 100,
    warmup: int = 10,
    name: str = "benchmark"
) -> dict:
    """Run a benchmark with truly separated compile and execution timing."""

    # Step 1: Measure TRUE compile time (lower + compile)
    compile_start = time.perf_counter()
    lowered = jit(fn).lower(spots)
    lowered_time = time.perf_counter() - compile_start

    compile_start2 = time.perf_counter()
    compiled = lowered.compile()
    compile_time = time.perf_counter() - compile_start2

    total_compile_ms = (lowered_time + compile_time) * 1000
    lowered_ms = lowered_time * 1000
    compile_only_ms = compile_time * 1000

    # Step 2: Warmup the compiled function (should be fast - no compilation)
    for _ in range(warmup):
        result = compiled(spots)
        jax.block_until_ready(result)

    # Step 3: Measure PURE execution time
    times = []
    sample_prices = None
    sample_deltas = None

    for i in range(trials):
        start = time.perf_counter()
        result = compiled(spots)
        jax.block_until_ready(result)  # Ensure async completion
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if i == 0:
            if isinstance(result, tuple):
                sample_prices = np.array(result[0][:5])
                sample_deltas = np.array(result[1][:5])
            else:
                sample_prices = np.array(result[:5])

    avg_time_us = np.mean(times) * 1_000_000
    min_time_us = np.min(times) * 1_000_000
    max_time_us = np.max(times) * 1_000_000
    std_time_us = np.std(times) * 1_000_000
    throughput_k = len(spots) / np.mean(times) / 1000

    return {
        "name": name,
        "batch_size": len(spots),
        "lowering_time_ms": lowered_ms,
        "compile_time_ms": compile_only_ms,
        "total_compile_ms": total_compile_ms,
        "avg_time_us": avg_time_us,
        "min_time_us": min_time_us,
        "max_time_us": max_time_us,
        "std_time_us": std_time_us,
        "throughput_k": throughput_k,
        "sample_prices": sample_prices,
        "sample_deltas": sample_deltas,
    }


def main():
    parser = argparse.ArgumentParser(description="JAX Black-Scholes Benchmark")
    parser.add_argument("--dump-hlo", action="store_true",
                        help="Dump HLO/StableHLO IR for comparison")
    parser.add_argument("--output-dir", default="hlo_dumps",
                        help="Directory for HLO dumps")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     JAX Black-Scholes Benchmark (Separated Compile/Execute Timing)           ║
║     (jax.jit + jax.grad + XLA CPU backend)                                   ║
║                                                                              ║
║     Fair comparison with SwiftIR-Traced (both use XLA)                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Print JAX configuration
    print(f"  JAX version: {jax.__version__}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print()

    # Dump HLO if requested
    if args.dump_hlo:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Dumping HLO/StableHLO IR                                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        # Use batch size 1000 for readable HLO
        spots = jnp.array([80.0 + 40.0 * (i % 100) / 100.0 for i in range(1000)],
                          dtype=jnp.float32)

        dump_hlo(price_and_delta_explicit, spots, "black_scholes_fwd_grad", args.output_dir)
        dump_hlo(price_only, spots, "black_scholes_fwd_only", args.output_dir)
        print()

    trials = 100
    batch_sizes = [100, 1000, 10000, 100000, 1000000]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Running Benchmarks at Multiple Batch Sizes                  ║")
    print("║  (Compile and Execute times measured SEPARATELY)             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Trials per benchmark: {trials}")
    print()

    # Focus on explicit VJP (most comparable to SwiftIR)
    approaches = [
        ("JAX explicit VJP", price_and_delta_explicit),
    ]

    all_results = {name: [] for name, _ in approaches}

    for size in batch_sizes:
        print(f"  Batch size: {size:,}")

        # Generate input data (same pattern as Swift)
        spots = jnp.array([80.0 + 40.0 * (i % 100) / 100.0 for i in range(size)],
                          dtype=jnp.float32)

        for name, fn in approaches:
            result = run_benchmark_separated(fn, spots, trials=trials, name=name)
            all_results[name].append(result)
            print(f"    Lowering:  {result['lowering_time_ms']:.2f} ms")
            print(f"    Compile:   {result['compile_time_ms']:.2f} ms")
            print(f"    Execute:   {result['avg_time_us']:.1f} μs (±{result['std_time_us']:.1f})")
            print(f"    Throughput: {result['throughput_k']:.2f} K/s")
        print()

    # Print detailed tables
    for name, results in all_results.items():
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  {name.upper():^74} ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)

        print("  COMPILATION TIME:")
        print("  ┌────────────┬──────────────┬──────────────┬──────────────┐")
        print("  │ Batch Size │ Lowering(ms) │ Compile(ms)  │   Total(ms)  │")
        print("  ├────────────┼──────────────┼──────────────┼──────────────┤")
        for r in results:
            print(f"  │ {r['batch_size']:>10,} │ {r['lowering_time_ms']:>10.2f} │ "
                  f"{r['compile_time_ms']:>10.2f} │ {r['total_compile_ms']:>10.2f} │")
        print("  └────────────┴──────────────┴──────────────┴──────────────┘")
        print()

        print("  EXECUTION TIME (pure execution, no compilation):")
        print("  ┌────────────┬──────────────────┬──────────────────┬──────────────────┐")
        print("  │ Batch Size │   Avg Time (μs)  │   Min Time (μs)  │   Throughput     │")
        print("  ├────────────┼──────────────────┼──────────────────┼──────────────────┤")
        for r in results:
            print(f"  │ {r['batch_size']:>10,} │ {r['avg_time_us']:>14.1f} │ "
                  f"{r['min_time_us']:>14.1f} │ {r['throughput_k']:>12.2f} K/s │")
        print("  └────────────┴──────────────────┴──────────────────┴──────────────────┘")
        print()

    # Sample results validation
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SAMPLE RESULTS VALIDATION                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Use the explicit VJP results for validation (most comparable to SwiftIR)
    if all_results["JAX explicit VJP"]:
        r = all_results["JAX explicit VJP"][0]
        print(f"  Spot prices:     80.0, 80.4, 80.8, 81.2, 81.6")
        if r['sample_prices'] is not None:
            prices_str = ", ".join(f"{p:.4f}" for p in r['sample_prices'])
            print(f"  Call prices:     {prices_str}")
        if r['sample_deltas'] is not None:
            deltas_str = ", ".join(f"{d:.4f}" for d in r['sample_deltas'])
            print(f"  Deltas (∂C/∂S):  {deltas_str}")
    print()

    # Comparison summary
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COMPARISON SUMMARY (at batch size 100,000)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Find 100k results
    for name, results in all_results.items():
        for r in results:
            if r['batch_size'] == 100000:
                print(f"  {name}:")
                print(f"    Compile time:    {r['total_compile_ms']:.2f} ms")
                print(f"    Execute time:    {r['avg_time_us']:.1f} μs")
                print(f"    Throughput:      {r['throughput_k']:,.0f} K options/sec")

    print()
    print("  Compare with SwiftIR-Traced at 100K batch:")
    print("    Compile time:    ~28-45 ms")
    print("    Execute time:    ~1,688 μs (fwd+grad)")
    print("    Throughput:      ~59,217 K/s")
    print()
    print("  Compare with Swift _Differentiation: ~803 K/s at 100K batch")
    print()

    if args.dump_hlo:
        print(f"  HLO dumps saved to: {args.output_dir}/")
        print()


if __name__ == "__main__":
    main()
