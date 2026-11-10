"""Selected PIVOT operators, adapted from the research implementation.

See the supplement README for the correspondence with the paper.
"""
import jax
import jax.numpy as jnp
import numpy as np


def gated_residual(shared_features, shift_features, regime_gate):
    """The residual_raw branch of _factorized_fusion (Eq. 4)."""
    return shared_features + regime_gate * shift_features


def bounded_price(logit, lower, upper):
    """The bounded_sigmoid branch of the normalized price head (Eq. 12)."""
    return lower + (upper - lower) * jax.nn.sigmoid(logit)

def _masked_mean(values, mask, fallback):
    mask = mask.astype(values.dtype)
    count = jnp.sum(mask)
    masked = jnp.sum(values * mask) / (count + 1.0e-12)
    return jnp.where(count > 0.0, masked, fallback)

def _sample_tail_cvar(values, tail_fraction: float):
    flat = jnp.reshape(values, (-1,))
    n = flat.shape[0]
    if n == 0:
        return jnp.array(0.0)
    fraction = float(np.clip(tail_fraction, 1.0 / max(1, n), 1.0))
    threshold = jnp.quantile(flat, 1.0 - fraction)
    tail_mask = (flat >= threshold).astype(flat.dtype)
    return _masked_mean(flat, tail_mask, jnp.mean(flat))


def mean_sample_cvar(squared_error, tail_fraction=0.2, mixture_weight=0.5):
    """Mean and empirical tail loss (Eqs. 8–10)."""
    mse = jnp.mean(squared_error)
    tail = _sample_tail_cvar(squared_error, tail_fraction)
    return (1.0 - mixture_weight) * mse + mixture_weight * tail
