# Integration Plan: Swapping `survi` Cover Tree with `covertreex` (Rust)

This document outlines the steps to replace the Python/Numba cover tree implementation in `survi-v2` with the optimized Rust implementation provided by `covertreex`.

## 1. Dependencies

Add `covertreex` to `survi-v2/packages/survi/pyproject.toml`:

```toml
[project.optional-dependencies]
# ...
rust_tree = [
    "covertreex>=0.0.2",
]
```

## 2. Adapter Implementation

Use `survi_adapter.py` (provided) to bridge `survi`'s backend to `covertreex`.

The adapter now supports:
- **RBF & Matern 5/2** (Isotropic & ARD): Optimized NumPy implementation.
- **Generic JAX Kernels**: Fallback using JAX-on-CPU for complex or custom kernels.

### Deep Kernel Optimization Strategy

For **Deep Learned Kernels** (e.g. `NeuralFiniteFeatureKernel` or Deep Kernel Learning), relying on the generic JAX fallback works but may be slow due to repeated neural network inference on small blocks.

**Recommended Optimization:**
1.  Precompute the embeddings $\Phi(X)$ for all points.
2.  Pass these embeddings as the "dataset" to `covertreex` (instead of raw $X$).
3.  Use a simple kernel (Linear or RBF) on these embeddings.

If using `NeuralFiniteFeatureKernel`, the kernel is linear on features $\phi(x)$. The adapter can automatically detect this if you expose the features, but for now, the JAX fallback is the safest starting point.

## 3. Modification of `cover_tree.py`

In `survi/models/selectors/cover_tree.py`, modify `CoverTreeCorrelationNeighborSelector.select` and `select_for_prediction`.

### Select Method

```python
        # ... inside select method ...
        
        # CHECK FOR RUST ENGINE
        use_rust = HAS_COVERTREEX and os.getenv("SURVI_USE_RUST_TREE", "1") == "1"
        
        if use_rust:
            from .survi_adapter import adapt_backend
            
            # Adapt the backend (automatically detects RBF/Matern or uses JAX fallback)
            host_backend = adapt_backend(backend, kernel_strategy=kernel_strategy, unconstrained_params=unconstrained_kernel_params)
            
            # Extract variance/lengthscale just for "info" passed to engine params
            # (Actual kernel logic is in host_backend.kernel_provider)
            # We can pass dummy values if using generic provider
            params = kernel_strategy.constrain_params(unconstrained_kernel_params)
            s2 = float(params.get('signal_variance', 1.0))
            
            # Build and Query using Rust Hilbert Engine
            tree = cx_engine.build_tree(
                X_np, 
                engine="rust-hilbert", 
                residual_backend=host_backend,
                residual_params={
                    "variance": s2,
                    "chunk_size": 512
                }
            )
            
            # Run Query
            query_indices = np.arange(n, dtype=np.int64)
            neighbors, distances = tree.knn(query_indices, k=k, return_distances=True)
            
            # Pad -1s if needed and return
            out = np.full((n, k), -1, dtype=np.int32)
            # ... copy logic ...
                
            return jnp.array(neighbors, dtype=jnp.int32)
```

## 4. Verification

Run the `survi` test suite with `SURVI_USE_RUST_TREE=1`. Ensure that `Matern52` tests pass (verifying the adapter logic).
