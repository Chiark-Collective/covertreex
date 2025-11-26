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
- **Deep Embeddings**: Pass `points_override` to use embeddings $\Phi(X)$ instead of $X$.
- **Generic JAX Kernels**: Fallback using JAX-on-CPU.

## 3. Modification of `cover_tree.py`

In `survi/models/selectors/cover_tree.py`, modify `CoverTreeCorrelationNeighborSelector.select` and `select_for_prediction`.

### Select Method

```python
        # ... inside select method ...
        
        # CHECK FOR RUST ENGINE
        use_rust = HAS_COVERTREEX and os.getenv("SURVI_USE_RUST_TREE", "1") == "1"
        
        if use_rust:
            from .survi_adapter import adapt_backend
            
            # 1. Handle Deep Kernels (Embeddings)
            # If the kernel strategy has a '_features' method (e.g. NeuralFiniteFeatureKernel)
            # or if you are using Deep Kernel Learning, extract embeddings here.
            points_override = None
            # Example:
            # if hasattr(kernel_strategy, "embed"):
            #     points_override = np.asarray(kernel_strategy.embed(X_np, unconstrained_kernel_params))
            
            # 2. Adapt Backend
            host_backend = adapt_backend(
                backend, 
                kernel_strategy=kernel_strategy, 
                unconstrained_params=unconstrained_kernel_params,
                points_override=points_override
            )
            
            # 3. Extract Parameters for Rust Engine
            # The Rust engine implements RBF (0) and Matern52 (1) natively.
            # It needs to know which one to use.
            params = kernel_strategy.constrain_params(unconstrained_kernel_params)
            s2 = float(params.get('signal_variance', 1.0))
            
            # Detect kernel type (0=RBF, 1=Matern52) from survi Enum
            # Default to 0 (RBF)
            k_type = getattr(kernel_strategy, "kernel_id", 0)
            if isinstance(k_type, int):
                pass 
            else: 
                k_type = 0 # Fallback or map enum names
            
            # 4. Build and Query
            tree = cx_engine.build_tree(
                X_np, 
                engine="rust-hilbert", 
                residual_backend=host_backend,
                residual_params={
                    "variance": s2,
                    "chunk_size": 512,
                    "kernel_type": int(k_type)
                }
            )
            
            # Run Query
            query_indices = np.arange(n, dtype=np.int64)
            neighbors, distances = tree.knn(query_indices, k=k, return_distances=True)
            
            # ... pad/format and return ...
            return jnp.array(neighbors, dtype=jnp.int32)
```

## 4. Verification

Run the `survi` test suite with `SURVI_USE_RUST_TREE=1`.
