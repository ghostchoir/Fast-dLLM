"""Hadamard rotation wrappers for NVFP4 KV cache quantization.

Uses the Dao-AILab fast-hadamard-transform CUDA kernel when available,
with a pure PyTorch fallback for environments where the CUDA extension
cannot be compiled (e.g. torch version mismatch).

Normalized Hadamard: forward and inverse both apply H·x / √n,
making the transform orthogonal (norm-preserving).

Usage:
    from quantization.hadamard import hadamard_transform, inverse_hadamard_transform

    x_rot = hadamard_transform(x)              # H·x / √n
    x_back = inverse_hadamard_transform(x_rot) # H·x_rot / √n  ≈ x
"""

import math
import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fht
    from fast_hadamard_transform.fast_hadamard_transform_interface import (
        hadamard_transform_12N as _fht_12n,
    )
    _HAS_FHT = True
except ImportError:
    _HAS_FHT = False


# ---------------------------------------------------------------------------
# Pure PyTorch fallbacks
# ---------------------------------------------------------------------------

def _hadamard_pytorch(x, scale):
    """Walsh-Hadamard transform on last dim (must be power of 2)."""
    n = x.shape[-1]
    orig_dtype = x.dtype
    x = x.float()
    h = 1
    while h < n:
        # Butterfly: split into pairs of size h, compute sum/diff
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], n)
        h *= 2
    return (x * scale).to(orig_dtype)


# Exact 12×12 Hadamard matrix from fast-hadamard-transform CUDA kernel
_H12 = None

def _get_h12(device):
    global _H12
    if _H12 is not None and _H12.device == device:
        return _H12
    _H12 = torch.tensor([
        [+1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
        [-1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1],
        [+1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1],
        [+1, -1, -1, -1, +1, -1, -1, +1, -1, +1, +1, -1],
        [+1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1],
        [+1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1],
        [+1, +1, -1, -1, +1, +1, +1, -1, +1, +1, -1, -1],
        [+1, -1, -1, +1, +1, -1, -1, -1, +1, -1, -1, +1],
        [+1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
        [+1, -1, -1, +1, -1, +1, +1, -1, -1, -1, +1, -1],
        [+1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1],
        [+1, -1, +1, -1, -1, +1, -1, +1, +1, -1, -1, -1],
    ], dtype=torch.float32, device=device)
    return _H12


def _hadamard_12n_pytorch(x, scale):
    """Hadamard for dim = 12 × power_of_2 (Kronecker: H_12 ⊗ H_N)."""
    n = x.shape[-1]
    m = n // 12
    orig_shape = x.shape
    orig_dtype = x.dtype
    x = x.reshape(*orig_shape[:-1], 12, m).float()

    # Apply H_m (power-of-2 Walsh-Hadamard) on last dim
    h = 1
    while h < m:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*orig_shape[:-1], 12, m)
        h *= 2

    # Apply H_12 on the 12-dim: einsum('ij,...j...->...i...')
    H12 = _get_h12(x.device)
    x = torch.einsum('ij,...jk->...ik', H12, x)

    return (x * scale).reshape(orig_shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hadamard_transform(x, scale=None):
    """Forward normalized Hadamard: H·x / √n on last dim."""
    if scale is None:
        scale = 1.0 / math.sqrt(x.shape[-1])
    if _HAS_FHT:
        return _fht(x, scale=scale)
    return _hadamard_pytorch(x, scale)


def inverse_hadamard_transform(x, scale=None):
    """Inverse normalized Hadamard: H·x / √n (same as forward for orthogonal H/√n)."""
    if scale is None:
        scale = 1.0 / math.sqrt(x.shape[-1])
    if _HAS_FHT:
        return _fht(x, scale=scale)
    return _hadamard_pytorch(x, scale)


def hadamard_transform_12N(x, scale=None):
    """Normalized Hadamard for dim = 12 × power_of_2 (e.g. 1536 = 12×128).

    Uses the specialized CUDA kernel for Kronecker product H_12 ⊗ H_N.
    Self-inverse with normalized scale: (H_12N/√n)² = I.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(x.shape[-1])
    if _HAS_FHT:
        return _fht_12n(x, scale=scale)
    return _hadamard_12n_pytorch(x, scale)
