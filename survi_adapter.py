import numpy as np
import jax
import jax.numpy as jnp
import os

# Mock imports for demonstration if libraries are missing
try:
    from covertreex.metrics.residual import ResidualCorrHostData
except ImportError:
    class ResidualCorrHostData:
        def __init__(self, **kwargs): pass

def _rbf_kernel_numpy(x_all, variance, lengthscale):
    """Returns a provider for RBF kernel."""
    var = float(variance)
    ls = np.asarray(lengthscale)
    is_ard = ls.ndim > 0
    
    def provider(row_indices, col_indices):
        rows = x_all[row_indices]
        cols = x_all[col_indices]
        diff = rows[:, None, :] - cols[None, :, :]
        if is_ard:
             diff = diff / ls[None, None, :]
        else:
             diff = diff / ls
        sq_dist = np.sum(diff**2, axis=2)
        return var * np.exp(-0.5 * sq_dist)
    return provider

def _matern52_kernel_numpy(x_all, variance, lengthscale):
    """Returns a provider for Matern 5/2 kernel."""
    var = float(variance)
    ls = np.asarray(lengthscale)
    is_ard = ls.ndim > 0
    sqrt5 = np.sqrt(5.0)

    def provider_iso(row_indices, col_indices):
        rows = x_all[row_indices]
        cols = x_all[col_indices]
        diff = rows[:, None, :] - cols[None, :, :]
        diff = diff / float(ls)
        d2 = np.sum(diff**2, axis=2)
        r = np.sqrt(np.maximum(d2, 0.0))
        a = sqrt5 * r
        return var * (1.0 + a + a**2/3.0) * np.exp(-a)

    def provider_ard(row_indices, col_indices):
        rows = x_all[row_indices]
        cols = x_all[col_indices]
        diff = (rows[:, None, :] - cols[None, :, :]) / ls[None, None, :]
        d2 = np.sum(diff**2, axis=2)
        r = np.sqrt(np.maximum(d2, 0.0))
        # Match survi's ARD definition: a = sqrt(5) * r / ls_geom
        # But 'r' here is already scaled by ls vector (Mahalanobis distance).
        # In survi AnisotropicKernel: r = sqrt(d2), and then a = sqrt(5)*r/ls_geom.
        # This effectively double-scales. We mirror it to ensure exact parity.
        ls_geom = np.exp(np.mean(np.log(ls)))
        a = sqrt5 * r / ls_geom
        return var * (1.0 + a + a**2/3.0) * np.exp(-a)

    return provider_ard if is_ard else provider_iso

def _make_jax_fallback_provider(kernel_strategy, unconstrained_params, X_all):
    """Creates a provider that calls back into JAX on the CPU."""
    try:
        cpu_dev = jax.devices("cpu")[0]
    except Exception:
        cpu_dev = None

    params_cpu = jax.device_put(unconstrained_params, cpu_dev)
    X_cpu = jax.device_put(X_all, cpu_dev)
    
    @jax.jit(backend="cpu")
    def _block_compute(idx1, idx2):
        x1 = X_cpu[idx1]
        x2 = X_cpu[idx2]
        return kernel_strategy(x1, x2, params_cpu)
        
    def provider(row_indices, col_indices):
        idx1 = jnp.array(row_indices)
        idx2 = jnp.array(col_indices)
        k_block = _block_compute(idx1, idx2)
        return np.asarray(k_block)
        
    return provider

def adapt_backend(survi_backend, kernel_strategy=None, unconstrained_params=None, points_override=None):
    """
    Adapts a survi `_ResidualCorrBackend` to a covertreex `ResidualCorrHostData`.
    
    Args:
        survi_backend: Instance of `_ResidualCorrBackend`.
        kernel_strategy: The survi KernelStrategy (e.g. IsotropicKernel).
        unconstrained_params: The parameters dict.
        points_override: Optional (N, D) array to use instead of survi_backend.X.
                        Useful for Deep Kernels where we operate on embeddings.
    """
    # 1. Extract Data
    v_matrix = np.asarray(survi_backend.V, dtype=np.float32)
    p_diag = np.asarray(survi_backend.p_diag, dtype=np.float32)
    kernel_diag = np.asarray(survi_backend.Kdiag, dtype=np.float32)
    
    # Use override points (embeddings) if provided, else raw X
    if points_override is not None:
        X_host = np.asarray(points_override, dtype=np.float64)
    else:
        X_host = np.asarray(survi_backend.X, dtype=np.float64)

    # 2. Identify Kernel Type and Params
    provider = None
    is_fast_path = False
    
    if kernel_strategy:
        cls_name = kernel_strategy.__class__.__name__
        
        if cls_name in ("IsotropicKernel", "AnisotropicKernel"):
            params = kernel_strategy.constrain_params(unconstrained_params)
            s2 = float(params['signal_variance'])
            ls_jax = params.get('lengthscales', params.get('lengthscale'))
            ls = np.asarray(ls_jax)
            
            kid = getattr(kernel_strategy, "kernel_id", None)
            
            if kid == 0: # RBF
                provider = _rbf_kernel_numpy(X_host, s2, ls)
                is_fast_path = True
            elif kid == 1: # Matern52
                provider = _matern52_kernel_numpy(X_host, s2, ls)
                is_fast_path = True

    if not is_fast_path:
        if kernel_strategy is None:
             kernel_strategy = getattr(survi_backend, "ks", None)
             unconstrained_params = getattr(survi_backend, "pu", None)
             
        if kernel_strategy:
            # Fallback to JAX (works for Deep Kernels, Mixtures, etc.)
            provider = _make_jax_fallback_provider(kernel_strategy, unconstrained_params, X_host)
        else:
            raise ValueError("Cannot adapt backend: missing KernelStrategy and no fast path found.")

    def point_decoder(values):
        return np.asarray(values, dtype=np.int64).reshape(-1)
        
    # Optional: For Deep Kernels, pass the embeddings (X_host) as 'kernel_points_f32'
    # This helps the Rust tree optimize spatial indexing (Hilbert sort) and SGEMM.
    kernel_points = X_host.astype(np.float32)
    row_norms = np.sum(kernel_points**2, axis=1)

    return ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=provider,
        point_decoder=point_decoder,
        chunk_size=512,
        kernel_points_f32=kernel_points,
        kernel_row_norms_f32=row_norms
    )

