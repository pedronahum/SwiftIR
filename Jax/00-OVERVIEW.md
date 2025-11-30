# SwiftIR Enhancement Specifications

## Overview

This document set provides comprehensive specifications for extending SwiftIR to match and exceed JAX capabilities while leveraging Swift's unique type system advantages.

## Document Index

| # | Feature | Priority | Complexity | File |
|---|---------|----------|------------|------|
| 01 | vmap - Automatic Vectorization | HIGH | High | `01-VMAP-SPEC.md` |
| 02 | scan - Sequence Processing | HIGH | Medium | `02-SCAN-SPEC.md` |
| 03 | cond/switch - Conditionals | HIGH | Medium | `03-COND-SPEC.md` |
| 04 | Shape-Typed Tensors | HIGH | High | `04-SHAPE-TYPED-TENSORS-SPEC.md` |
| 05 | Gradient Checkpointing | MEDIUM | Medium | `05-CHECKPOINTING-SPEC.md` |
| 06 | Functional PRNG | MEDIUM | Low | `06-PRNG-SPEC.md` |
| 07 | Higher-Order Differentiation | MEDIUM | High | `07-HIGHER-ORDER-AD-SPEC.md` |
| 08 | Protocol-Based Pytrees | MEDIUM | Medium | `08-PYTREES-SPEC.md` |

## Implementation Order

### Phase 1: Core Transformations (Weeks 1-4)
1. **vmap** - Essential for efficient batching in ML workloads
2. **scan** - Cleaner API for sequential models (RNNs, time series)
3. **cond/switch** - Differentiable control flow

### Phase 2: Type Safety (Weeks 5-6)
4. **Shape-Typed Tensors** - Swift's killer feature for compile-time safety
5. **Protocol-Based Pytrees** - Ergonomic parameter trees

### Phase 3: Advanced Features (Weeks 7-8)
6. **Gradient Checkpointing** - Memory efficiency for large models
7. **Functional PRNG** - Reproducibility
8. **Higher-Order Differentiation** - Hessians, JVPs

## Repository Structure

```
Sources/SwiftIR/
├── SymbolicAD/
│   ├── ADIntegration.swift      # Existing: 80+ differentiable ops
│   ├── DifferentiableWhile.swift # Existing: while loop support
│   ├── BackendCompilation.swift  # Existing: MLIR generation
│   ├── PJRTExecution.swift       # Existing: runtime execution
│   │
│   ├── Vmap.swift               # NEW: vmap transformation
│   ├── Scan.swift               # NEW: scan transformation
│   ├── Cond.swift               # NEW: conditional execution
│   ├── Checkpointing.swift      # NEW: gradient checkpointing
│   ├── PRNG.swift               # NEW: functional random
│   └── HigherOrderAD.swift      # NEW: hessian, jvp
│
├── TypedTensors/                 # NEW MODULE
│   ├── TensorShape.swift        # Shape type definitions
│   ├── Tensor.swift             # Shape-typed tensor wrapper
│   ├── Operations.swift         # Type-safe operations
│   └── Broadcasting.swift       # Broadcast shape inference
│
└── Pytrees/                      # NEW MODULE
    ├── DiffTree.swift           # Protocol definition
    ├── TreeFlatten.swift        # Flatten/unflatten
    └── TreeMap.swift            # Tree transformations

Tests/SwiftIRTests/
├── VmapTests.swift              # NEW
├── ScanTests.swift              # NEW
├── CondTests.swift              # NEW
├── TypedTensorTests.swift       # NEW
├── CheckpointingTests.swift     # NEW
├── PRNGTests.swift              # NEW
├── HigherOrderADTests.swift     # NEW
└── PytreeTests.swift            # NEW
```

## Dependencies on Existing Code

All new features build on these existing components:

### DifferentiableTracer
```swift
// From existing codebase - the core tracing type
public class DifferentiableTracer: Differentiable {
    public var id: Int
    public var shape: [Int]
    public var dtype: DType
    // ... gradient tracking, operation recording
}
```

### StableHLO Generation
```swift
// From BackendCompilation.swift
func generateStableHLO(from tracer: DifferentiableTracer) -> String
```

### PJRT Execution
```swift
// From PJRTExecution.swift
func compileAndExecute(mlir: String, inputs: [Float]) -> [Float]
```

## Success Criteria Summary

| Feature | Key Metrics |
|---------|-------------|
| vmap | 95%+ of JAX vmap functionality, <5% overhead vs manual batching |
| scan | O(1) compile time, matches diffWhileLoop performance |
| cond | Both branches traced, correct gradient flow |
| Shape-Typed | Compile-time shape errors, zero runtime overhead |
| Checkpointing | 50%+ memory reduction, <2x compute overhead |
| PRNG | Deterministic, splittable, matches JAX semantics |
| Higher-Order AD | Correct Hessians, efficient HVP via forward-mode |
| Pytrees | Automatic flatten/unflatten, works with all transformations |

## Testing Strategy

Each feature requires:
1. **Unit tests** - Individual function correctness
2. **Gradient tests** - Numerical gradient checking
3. **Integration tests** - Composition with other features
4. **Performance tests** - Benchmarks against baselines
5. **Edge case tests** - Empty tensors, single elements, large batches

## Getting Started

For Claude Code, start with each spec file in order. Each contains:
- Detailed API specification
- Implementation guide with code structure
- Complete test cases
- Success criteria checklist

Begin with `01-VMAP-SPEC.md` and proceed sequentially.
