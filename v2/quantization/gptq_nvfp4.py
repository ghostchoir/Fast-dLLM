"""GPTQ/GPTAQ weight optimization for NVFP4 format.

Implements:
- GPTQ (Frantar et al., 2022) adapted for NVFP4 two-level quantization
- GPTAQ (Li et al., ICML 2025) asymmetric calibration extension

GPTAQ adds ~20 lines to GPTQ: it collects full-precision layer inputs alongside
quantized inputs, computes an asymmetry error matrix dXXT, and uses a perturbation
matrix P to correct weight updates — matching quantized layer outputs to the
full-precision model's outputs rather than to already-quantized inputs.
"""

import torch
import torch.nn as nn
from quantization.quantizer import (
    _round_to_fp4_e2m1,
    FP4_E2M1_MAX,
    pseudo_nvfp4_quantize_tensor,
)

SKIP_MODULES = {"lm_head", "embed_tokens"}


@torch.no_grad()
def quantize_nvfp4_block(block, global_scale):
    """Quantize a [rows, block_size] block atomically with consistent local scale.

    All columns share the same per-row local FP8 scale, matching what eval
    (pseudo_nvfp4_quantize_tensor_with_global_scale) would compute.
    """
    block_norm = block / global_scale
    local_amax = block_norm.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    local_scale = local_amax / FP4_E2M1_MAX
    local_scale_fp8 = local_scale.to(torch.float8_e4m3fn).float()
    local_scale_fp8 = local_scale_fp8.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)

    scaled_vals = block_norm / local_scale_fp8
    quantized_vals = _round_to_fp4_e2m1(scaled_vals)
    dequant = quantized_vals * local_scale_fp8 * global_scale

    return dequant


@torch.no_grad()
def quantize_nvfp4_full(W, global_scale, block_size=16):
    """Quantize a full weight matrix with NVFP4 using given global scale.

    Returns dequantized weights. Used for MSE-optimal scale search.
    """
    rows, cols = W.shape
    # Pad columns to multiple of block_size
    pad = (block_size - cols % block_size) % block_size
    if pad > 0:
        W_pad = torch.nn.functional.pad(W, (0, pad))
    else:
        W_pad = W

    W_pad = W_pad.reshape(rows, -1, block_size)  # [rows, n_blocks, block_size]
    W_norm = W_pad / global_scale

    local_amax = W_norm.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    local_scale = local_amax / FP4_E2M1_MAX
    local_scale_fp8 = local_scale.to(torch.float8_e4m3fn).float()
    local_scale_fp8 = local_scale_fp8.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)

    scaled_vals = W_norm / local_scale_fp8
    quantized_vals = _round_to_fp4_e2m1(scaled_vals)
    dequant = quantized_vals * local_scale_fp8 * global_scale

    dequant = dequant.reshape(rows, -1)[:, :cols]
    return dequant


@torch.no_grad()
def optimize_global_scale(W, n_iters=20, n_grid=16, block_size=16):
    """Find MSE-optimal global scale via grid search.

    Searches around the amax-based default scale to find the value that
    minimizes weight MSE after NVFP4 quantization.

    Args:
        W: [out_features, in_features] weight matrix
        n_iters: number of search iterations
        n_grid: number of grid points per iteration

    Returns:
        optimal global_scale (scalar tensor)
    """
    gs = W.abs().amax().clamp(min=1e-12) / FP4_E2M1_MAX

    best_gs = gs.clone()
    W_q = quantize_nvfp4_full(W, best_gs, block_size)
    best_mse = ((W - W_q) ** 2).sum()

    # Coarse-to-fine grid search
    lo, hi = 0.5, 1.5  # search range as fraction of current gs
    for iteration in range(n_iters):
        factors = torch.linspace(lo, hi, n_grid, device=W.device)
        for f in factors:
            gs_try = gs * f
            if gs_try <= 0:
                continue
            W_q = quantize_nvfp4_full(W, gs_try, block_size)
            mse = ((W - W_q) ** 2).sum()
            if mse < best_mse:
                best_mse = mse
                best_gs = gs_try

        # Narrow search range around best
        best_f = best_gs / gs
        spread = (hi - lo) / (2 * n_grid)
        lo = max(0.1, (best_f - spread * n_grid / 2).item())
        hi = (best_f + spread * n_grid / 2).item()

    return best_gs


@torch.no_grad()
def gptq_quantize_linear(W, H, global_scale, blocksize=128, nvfp4_block_size=16,
                          damp_percent=0.01, act_order=True):
    """Run GPTQ on a single weight matrix with atomic NVFP4 block quantization.

    Each NVFP4 block (16 columns) is quantized atomically with a single local
    scale. Per-column errors are then distributed to subsequent columns via the
    standard GPTQ Hessian-weighted update.

    Args:
        W: [out_features, in_features] weight matrix (float32)
        H: [in_features, in_features] Hessian X^T X (float32)
        global_scale: scalar, MSE-optimized or amax/6.0
        blocksize: GPTQ lazy batch size (must be multiple of nvfp4_block_size)
        nvfp4_block_size: NVFP4 micro-block size (16)
        damp_percent: Hessian damping fraction
        act_order: if True, reorder columns by Hessian diagonal (descending)

    Returns:
        W_q: GPTQ-optimized dequantized weights [out_features, in_features]
        loss: Average GPTQ loss (weighted squared error)
    """
    assert blocksize % nvfp4_block_size == 0, \
        f"blocksize ({blocksize}) must be multiple of nvfp4_block_size ({nvfp4_block_size})"
    rows, columns = W.shape
    W = W.clone().float()
    H = H.clone().float()

    # --- Act-order: reorder columns by descending Hessian diagonal ---
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]

    # Handle dead columns (zero input variance)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Dampen Hessian
    damp = damp_percent * torch.diag(H).mean()
    diag_idx = torch.arange(columns, device=H.device)
    H[diag_idx, diag_idx] += damp

    # Compute H^{-1} and its upper Cholesky factor
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        extra = 0.1 * torch.diag(H).mean()
        H[diag_idx, diag_idx] += extra
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)

    H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)

    total_loss = 0.0

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = H_inv_cho[i1:i2, i1:i2]

        # Process NVFP4 blocks atomically within this GPTQ block
        for nvfp4_start in range(0, count, nvfp4_block_size):
            nvfp4_end = min(nvfp4_start + nvfp4_block_size, count)
            nvfp4_cols = nvfp4_end - nvfp4_start

            # Quantize entire NVFP4 block at once (consistent local scale)
            block = W1[:, nvfp4_start:nvfp4_end]
            block_q = quantize_nvfp4_block(block, global_scale)
            block_err = block - block_q  # [rows, nvfp4_cols]

            # Per-column error distribution to columns AFTER this NVFP4 block
            for j in range(nvfp4_cols):
                col = nvfp4_start + j
                d = Hinv1[col, col]
                err = block_err[:, j] / d
                Err1[:, col] = err
                total_loss += (block_err[:, j] ** 2 / d).sum().item()

                # Distribute to remaining columns (after this NVFP4 block)
                if nvfp4_end < count:
                    W1[:, nvfp4_end:] -= (
                        err.unsqueeze(1) * Hinv1[col, nvfp4_end:].unsqueeze(0)
                    )

            # Store quantized values
            W1[:, nvfp4_start:nvfp4_end] = block_q

        W[:, i1:i2] = W1
        if i2 < columns:
            W[:, i2:] -= Err1 @ H_inv_cho[i1:i2, i2:]

    # --- Inverse permutation to restore original column order ---
    if perm is not None:
        inv_perm = torch.argsort(perm)
        W = W[:, inv_perm]

    avg_loss = total_loss / (rows * columns)
    return W, avg_loss


@torch.no_grad()
def gptaq_quantize_linear(W, H, dXXT, global_scale, alpha=0.25, blocksize=128,
                           nvfp4_block_size=16, damp_percent=0.01):
    """Run GPTAQ on a single weight matrix with atomic NVFP4 block quantization.

    GPTAQ extends GPTQ with asymmetric calibration (Li et al., ICML 2025).
    It uses a perturbation matrix P derived from the difference between
    full-precision and quantized layer inputs to correct weight updates.

    Args:
        W: [out_features, in_features] weight matrix (float32)
        H: [in_features, in_features] Hessian X^T X (float32)
        dXXT: [in_features, in_features] asymmetry error matrix
              dXXT = (FP_input - Q_input) @ Q_input^T, accumulated over samples
        global_scale: scalar, amax(W) / 6.0
        alpha: scaling factor for perturbation (default 0.25 from paper)
        blocksize: GPTQ lazy batch size (must be multiple of nvfp4_block_size)
        nvfp4_block_size: NVFP4 micro-block size (16)
        damp_percent: Hessian damping fraction

    Returns:
        W_q: GPTAQ-optimized dequantized weights [out_features, in_features]
        loss: Average GPTQ loss (weighted squared error)
    """
    assert blocksize % nvfp4_block_size == 0, \
        f"blocksize ({blocksize}) must be multiple of nvfp4_block_size ({nvfp4_block_size})"
    rows, columns = W.shape
    W = W.clone().float()
    H = H.clone().float()
    dXXT = dXXT.clone().float()

    # Handle dead columns (zero input variance)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Dampen Hessian
    damp = damp_percent * torch.diag(H).mean()
    diag_idx = torch.arange(columns, device=H.device)
    H[diag_idx, diag_idx] += damp

    # Compute H^{-1} and its upper Cholesky factor
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        extra = 0.1 * torch.diag(H).mean()
        H[diag_idx, diag_idx] += extra
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)

    H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)

    # GPTAQ: compute perturbation matrix P
    P = alpha * ((dXXT @ H_inv_cho.T).triu_(diagonal=1)) @ H_inv_cho
    del dXXT

    total_loss = 0.0

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = H_inv_cho[i1:i2, i1:i2]
        P1 = P[i1:i2, i1:i2]

        # Process NVFP4 blocks atomically within this GPTQ block
        for nvfp4_start in range(0, count, nvfp4_block_size):
            nvfp4_end = min(nvfp4_start + nvfp4_block_size, count)
            nvfp4_cols = nvfp4_end - nvfp4_start

            # Quantize entire NVFP4 block at once (consistent local scale)
            block = W1[:, nvfp4_start:nvfp4_end]
            block_q = quantize_nvfp4_block(block, global_scale)
            block_err = block - block_q  # [rows, nvfp4_cols]

            # Per-column error distribution to columns AFTER this NVFP4 block
            for j in range(nvfp4_cols):
                col = nvfp4_start + j
                d = Hinv1[col, col]
                w = W1[:, col]  # current weight column (pre-quant value)
                err = block_err[:, j] / d
                Err1[:, col] = err
                total_loss += (block_err[:, j] ** 2 / d).sum().item()

                # GPTAQ: distribute error with perturbation correction
                if nvfp4_end < count:
                    W1[:, nvfp4_end:] -= (
                        err.unsqueeze(1) * Hinv1[col, nvfp4_end:].unsqueeze(0)
                        - w.unsqueeze(1) * P1[col, nvfp4_end:].unsqueeze(0)
                    )

            # Store quantized values
            W1[:, nvfp4_start:nvfp4_end] = block_q

        W[:, i1:i2] = W1
        if i2 < columns:
            # GPTAQ: inter-block update with perturbation correction
            W[:, i2:] -= Err1 @ H_inv_cho[i1:i2, i2:] - W1 @ P[i1:i2, i2:]

    avg_loss = total_loss / (rows * columns)
    return W, avg_loss
