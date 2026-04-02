"""KV cache fake quantization for evaluation and analysis.

Provides dynamic (per-token) and calibrated (frozen-scale) fake quantization
for DynamicCache KV tensors.  Supports INT4, INT8, FP8, and NVFP4 modes.

Calibrated modes (FP8, NVFP4) match vLLM production behavior: per-layer
scales are computed once during a calibration prefill and frozen thereafter.
"""

from contextlib import nullcontext
from typing import Optional

import torch

from quantization.quantizer import (
    pseudo_quantize_tensor,
    pseudo_fp8_quantize_tensor,
    pseudo_nvfp4_quantize_tensor,
    pseudo_nvfp4_quantize_tensor_with_global_scale,
    FP4_E2M1_MAX,
)
from quantization.hadamard import hadamard_transform, inverse_hadamard_transform


# ---------------------------------------------------------------------------
# Block Hadamard rotation helpers
# ---------------------------------------------------------------------------

def block_hadamard_transform(x, block_size=16):
    """Apply independent Hadamard within each block of block_size along last dim."""
    *batch_dims, D = x.shape
    x = x.reshape(*batch_dims, D // block_size, block_size)
    x = hadamard_transform(x)
    return x.reshape(*batch_dims, D)


def inverse_block_hadamard_transform(x, block_size=16):
    """Inverse of block_hadamard_transform."""
    *batch_dims, D = x.shape
    x = x.reshape(*batch_dims, D // block_size, block_size)
    x = inverse_hadamard_transform(x)
    return x.reshape(*batch_dims, D)


def build_zigzag_permutation(channel_amax, block_size=16):
    """Build zigzag permutation distributing high-magnitude channels across blocks.

    DuQuant-style: sort by magnitude, assign in back-and-forth order so no
    single block gets a concentration of outlier channels.
    """
    D = len(channel_amax)
    n_blocks = D // block_size
    sorted_idx = channel_amax.argsort(descending=True)
    block_slots = [[] for _ in range(n_blocks)]
    forward = True
    for i, ch_idx in enumerate(sorted_idx):
        block_pos = i % n_blocks
        if not forward:
            block_pos = n_blocks - 1 - block_pos
        block_slots[block_pos].append(ch_idx.item())
        if (i + 1) % n_blocks == 0:
            forward = not forward
    perm = []
    for block in block_slots:
        perm.extend(block)
    return torch.tensor(perm, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dynamic (per-token) KV cache quantization
# ---------------------------------------------------------------------------

def quantize_kv_cache(past_key_values, kv_quant_mode: str):
    """Apply fake quantization to all KV cache tensors in-place.

    Per-token quantization: each [head_dim]-sized vector (one token, one head)
    gets its own scale factor.

    Args:
        past_key_values: DynamicCache with .key_cache and .value_cache lists.
        kv_quant_mode: 'int4', 'int8', 'fp8', or 'nvfp4'.
    """
    if kv_quant_mode == 'none' or past_key_values is None:
        return

    num_layers = len(past_key_values.key_cache)
    for layer_idx in range(num_layers):
        for cache_list in (past_key_values.key_cache, past_key_values.value_cache):
            tensor = cache_list[layer_idx]           # [B, H, T, D]
            orig_shape = tensor.shape
            orig_dtype = tensor.dtype
            tensor_2d = tensor.reshape(-1, orig_shape[-1])  # [B*H*T, D]

            if kv_quant_mode == 'nvfp4':
                q = pseudo_nvfp4_quantize_tensor(
                    tensor_2d.float(), per_tensor_global=True,
                ).to(orig_dtype)
            elif kv_quant_mode == 'fp8':
                q = pseudo_fp8_quantize_tensor(
                    tensor_2d.float(), per_tensor=True,
                ).to(orig_dtype)
            else:
                n_bit = 4 if kv_quant_mode == 'int4' else 8
                q = pseudo_quantize_tensor(
                    tensor_2d.float(),
                    n_bit=n_bit,
                    zero_point=True,
                    w_group_size=-1,
                ).to(orig_dtype)

            cache_list[layer_idx] = q.reshape(orig_shape)


# ---------------------------------------------------------------------------
# FP8 calibrated (frozen-scale) KV cache quantization
# ---------------------------------------------------------------------------

class FP8KVCacheScales:
    """Frozen FP8 scale factors for KV cache quantization (vLLM-style).

    One FP32 scalar per K/V per layer, computed once during calibration.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_scales: list[torch.Tensor] = [None] * num_layers
        self.v_scales: list[torch.Tensor] = [None] * num_layers
        self.calibrated = False


@torch.no_grad()
def calibrate_fp8_kv_scales(
    model,
    calibration_input_ids: torch.Tensor,
    block_size: int,
    quant_context=None,
) -> FP8KVCacheScales:
    """Run a calibration prefill to compute frozen FP8 KV cache scales.

    Args:
        model: The transformer model.
        calibration_input_ids: [1, prompt_len] token ids for calibration.
        block_size: Block size for the forward pass.
        quant_context: Optional context manager (e.g. FakeQuantContext) to
            apply during calibration.  None if the model is already
            permanently quantized.

    Returns:
        FP8KVCacheScales with frozen per-layer scales.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(
            f"Calibration prompt ({calibration_input_ids.shape[1]} tokens) "
            f"must be >= block_size ({block_size})"
        )

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = FP8KVCacheScales(num_layers)

    for layer_idx in range(num_layers):
        k_amax = past_kv.key_cache[layer_idx].float().abs().amax().clamp(min=1e-12)
        v_amax = past_kv.value_cache[layer_idx].float().abs().amax().clamp(min=1e-12)
        scales.k_scales[layer_idx] = (k_amax / fp8_max).to(torch.float32)
        scales.v_scales[layer_idx] = (v_amax / fp8_max).to(torch.float32)

    scales.calibrated = True
    return scales


def quantize_kv_cache_fp8_frozen(
    past_key_values,
    scales: FP8KVCacheScales,
    num_old_tokens: int,
    bf16_window_size: int = 0,
):
    """Apply frozen-scale FP8 quantization to only newly-appended KV entries.

    Matches vLLM behavior: each entry is quantized once at write time with
    a frozen scale. Old entries (positions < num_old_tokens) are never touched.

    Args:
        past_key_values: DynamicCache with .key_cache / .value_cache.
        scales: Frozen FP8KVCacheScales from calibration.
        num_old_tokens: Number of T-dim tokens already quantized.
        bf16_window_size: Number of most recent tokens to keep in BF16.
            When > 0, only tokens before (total_len - bf16_window_size) are
            quantized, implementing a KIVI-style sliding window.

    Returns:
        int: The number of tokens that have been quantized (new num_old_tokens).
    """
    assert scales.calibrated, "Scales must be calibrated before use"
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max  # 448.0
    num_layers = len(past_key_values.key_cache)
    total_len = past_key_values.key_cache[0].shape[2]
    quantize_end = max(num_old_tokens, total_len - bf16_window_size) if bf16_window_size > 0 else total_len

    for layer_idx in range(num_layers):
        for cache_list, scale in [
            (past_key_values.key_cache, scales.k_scales[layer_idx]),
            (past_key_values.value_cache, scales.v_scales[layer_idx]),
        ]:
            tensor = cache_list[layer_idx]  # [B, H, T, D]
            if quantize_end <= num_old_tokens:
                continue
            new_slice = tensor[:, :, num_old_tokens:quantize_end, :]
            orig_dtype = new_slice.dtype
            x_scaled = new_slice.float() / scale
            # Clamp to FP8 representable range — FP8 e4m3fn has no inf,
            # so out-of-range values would become NaN without clamping.
            # This matches production behavior (Triton stores saturate).
            x_scaled = x_scaled.clamp(-fp8_max, fp8_max)
            x_fp8 = x_scaled.to(fp8_dtype)
            x_deq = x_fp8.to(torch.float32) * scale
            tensor[:, :, num_old_tokens:quantize_end, :] = x_deq.to(orig_dtype)

    return quantize_end


# ---------------------------------------------------------------------------
# NVFP4 calibrated (frozen-global-scale) KV cache quantization
# ---------------------------------------------------------------------------

class NVFP4KVCacheScales:
    """Frozen NVFP4 global scale factors for KV cache quantization.

    One FP32 global scalar per K/V per layer, computed once during calibration.
    Local FP8-E4M3 scales are NOT stored — they are computed dynamically at
    write time per block of 16 elements (matching NVIDIA Blackwell HW).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_scales: list[torch.Tensor] = [None] * num_layers
        self.v_scales: list[torch.Tensor] = [None] * num_layers
        self.calibrated = False


class NVFP4KVCacheScalesExtended(NVFP4KVCacheScales):
    """Extended NVFP4 scales with optional permutation, per-channel scales, block rotation."""

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.k_perms = [None] * num_layers
        self.k_inv_perms = [None] * num_layers
        self.k_per_channel_scales = [None] * num_layers
        self.k_per_channel_means = [None] * num_layers
        self.k_pre_shift_means = [None] * num_layers  # original-domain means (subtracted BEFORE rotation)
        self.use_block_rotation = [False] * num_layers


@torch.no_grad()
def calibrate_nvfp4_kv_scales(
    model,
    calibration_input_ids: torch.Tensor,
    block_size: int,
    quant_context=None,
) -> NVFP4KVCacheScales:
    """Run a calibration prefill to compute frozen NVFP4 KV cache global scales.

    Same pattern as calibrate_fp8_kv_scales but uses FP4_E2M1_MAX (6.0) instead
    of FP8 max (448.0).

    Args:
        model: The transformer model.
        calibration_input_ids: [1, prompt_len] token ids for calibration.
        block_size: Block size for the forward pass.
        quant_context: Optional context manager (e.g. FakeQuantContext) to
            apply during calibration.

    Returns:
        NVFP4KVCacheScales with frozen per-layer global scales.
    """
    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(
            f"Calibration prompt ({calibration_input_ids.shape[1]} tokens) "
            f"must be >= block_size ({block_size})"
        )

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScales(num_layers)

    for layer_idx in range(num_layers):
        # Compute amax on Hadamard-rotated data, since quantization
        # operates in the rotated domain (rotation disperses outliers).
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()
        k_rotated = hadamard_transform(k_data)
        v_rotated = hadamard_transform(v_data)
        k_amax = k_rotated.abs().amax().clamp(min=1e-12)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

    scales.calibrated = True
    return scales


def quantize_kv_cache_nvfp4_frozen(
    past_key_values,
    scales: NVFP4KVCacheScales,
    num_old_tokens: int,
    bf16_window_size: int = 0,
    variant: str = "frozen",
):
    """Apply NVFP4 quantization to newly-appended KV entries.

    Args:
        past_key_values: DynamicCache with .key_cache / .value_cache.
        scales: Frozen NVFP4KVCacheScales from calibration (ignored for 'dynamic').
        num_old_tokens: Number of T-dim tokens already quantized.
        bf16_window_size: Number of most recent tokens to keep in BF16.
        variant: Quantization variant:
            - 'frozen': Per-layer frozen global scale + Hadamard (default)
            - 'dynamic': Per-token dynamic global scale + Hadamard
            - 'k_only': Only quantize K (frozen scale + Hadamard), V stays BF16
            - 'v_only': Only quantize V (frozen scale + Hadamard), K stays BF16
            - 'per_head': Per-head frozen global scale + Hadamard
            - 'no_hadamard': Per-layer frozen global scale, no Hadamard rotation

    Returns:
        int: The number of tokens that have been quantized (new num_old_tokens).
    """
    if scales is not None:
        assert scales.calibrated, "Scales must be calibrated before use"
    num_layers = len(past_key_values.key_cache)
    total_len = past_key_values.key_cache[0].shape[2]
    quantize_end = max(num_old_tokens, total_len - bf16_window_size) if bf16_window_size > 0 else total_len

    use_hadamard = variant not in ("no_hadamard",)
    skip_k = variant == "v_only"
    skip_v = variant == "k_only"

    for layer_idx in range(num_layers):
        for cache_list, global_scale, is_key in [
            (past_key_values.key_cache, scales.k_scales[layer_idx] if scales else None, True),
            (past_key_values.value_cache, scales.v_scales[layer_idx] if scales else None, False),
        ]:
            if (skip_k and is_key) or (skip_v and not is_key):
                continue
            tensor = cache_list[layer_idx]  # [B, H, T, D]
            if quantize_end <= num_old_tokens:
                continue
            new_slice = tensor[:, :, num_old_tokens:quantize_end, :]
            orig_dtype = new_slice.dtype
            orig_shape = new_slice.shape

            if variant == "per_head":
                # Per-head frozen global scales: process each head separately
                B, H, T_new, D = orig_shape
                for h in range(H):
                    head_slice = new_slice[:, h, :, :].reshape(-1, D).float()
                    if use_hadamard:
                        head_slice = hadamard_transform(head_slice)
                    head_scale = global_scale[h] if global_scale.dim() > 0 else global_scale
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(head_slice, head_scale)
                    if use_hadamard:
                        q = inverse_hadamard_transform(q)
                    tensor[:, h, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(B, T_new, D)
            elif variant == "dynamic":
                # Per-token dynamic global scales (most precise FP4)
                flat = new_slice.reshape(-1, orig_shape[-1]).float()
                if use_hadamard:
                    flat = hadamard_transform(flat)
                q = pseudo_nvfp4_quantize_tensor(flat, per_tensor_global=False)
                if use_hadamard:
                    q = inverse_hadamard_transform(q)
                tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)
            elif variant in ("block_rot", "block_rot_clip", "block_rot_shift", "block_rot_shift_clip", "hybrid", "local_rot", "perm_hadamard", "pre_shift_block_rot"):
                flat = new_slice.reshape(-1, orig_shape[-1]).float()
                if is_key:
                    k_perm = getattr(scales, 'k_perms', [None] * num_layers)[layer_idx]
                    if k_perm is not None:
                        flat = flat[:, k_perm]
                    # Pre-rotation mean shift (original domain, before Hadamard)
                    k_pre_mean = getattr(scales, 'k_pre_shift_means', [None] * num_layers)[layer_idx]
                    if k_pre_mean is not None:
                        flat = flat - k_pre_mean.unsqueeze(0)
                    use_block = getattr(scales, 'use_block_rotation', [False] * num_layers)[layer_idx]
                    if use_block:
                        flat = block_hadamard_transform(flat)
                    else:
                        flat = hadamard_transform(flat)
                    # Post-rotation mean shift (rotated domain)
                    k_mean = getattr(scales, 'k_per_channel_means', [None] * num_layers)[layer_idx]
                    if k_mean is not None:
                        flat = flat - k_mean.unsqueeze(0)
                    k_pc = getattr(scales, 'k_per_channel_scales', [None] * num_layers)[layer_idx]
                    if k_pc is not None:
                        q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, k_pc.unsqueeze(0))
                    else:
                        q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, global_scale)
                    # Add post-rotation mean back after dequantization
                    if k_mean is not None:
                        q = q + k_mean.unsqueeze(0)
                    if use_block:
                        q = inverse_block_hadamard_transform(q)
                    else:
                        q = inverse_hadamard_transform(q)
                    # Add pre-rotation mean back (original domain)
                    if k_pre_mean is not None:
                        q = q + k_pre_mean.unsqueeze(0)
                    if k_perm is not None:
                        q = q[:, scales.k_inv_perms[layer_idx]]
                else:
                    flat = hadamard_transform(flat)
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, global_scale)
                    q = inverse_hadamard_transform(q)
                tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)
            elif variant in ("bias_subtract", "bias_sub_no_rot", "bias_sub_block_rot"):
                # Subtract k_proj.bias before rotation + quantize, add back after
                rot = getattr(scales, 'rotation_mode', 'global')
                def _rotate(x):
                    if rot == "global": return hadamard_transform(x)
                    if rot == "block": return block_hadamard_transform(x)
                    return x
                def _inv_rotate(x):
                    if rot == "global": return inverse_hadamard_transform(x)
                    if rot == "block": return inverse_block_hadamard_transform(x)
                    return x

                # KV LAC: learnable clip ratios (saved by QAT with --kv_lac)
                kv_lac = getattr(scales, 'kv_lac_ratios', None)
                kv_key = 'k' if is_key else 'v'
                lac_ratio = None
                if kv_lac is not None and layer_idx in kv_lac:
                    lac_ratio = kv_lac[layer_idx].get(kv_key)

                flat = new_slice.reshape(-1, orig_shape[-1]).float()
                if is_key:
                    k_bias = getattr(scales, 'k_biases', [None] * num_layers)[layer_idx]
                    if k_bias is not None:
                        B, H, T_new, D = orig_shape
                        bias_expanded = k_bias.unsqueeze(1).expand(H, T_new, D).reshape(-1, D)
                        bias_expanded = bias_expanded.unsqueeze(0).expand(B, -1, -1).reshape(-1, D)
                        flat = flat - bias_expanded.to(flat.device)
                    if lac_ratio is not None:
                        amax = flat.abs().amax().clamp(min=1e-12)
                        flat = flat.clamp(-lac_ratio * amax, lac_ratio * amax)
                    flat = _rotate(flat)
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, global_scale)
                    q = _inv_rotate(q)
                    if k_bias is not None:
                        q = q + bias_expanded.to(q.device)
                else:
                    if lac_ratio is not None:
                        amax = flat.abs().amax().clamp(min=1e-12)
                        flat = flat.clamp(-lac_ratio * amax, lac_ratio * amax)
                    flat = _rotate(flat)
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, global_scale)
                    q = _inv_rotate(q)
                tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)
            else:
                # frozen / k_only / v_only / no_hadamard / attn_aware
                flat = new_slice.reshape(-1, orig_shape[-1]).float()
                if use_hadamard:
                    flat = hadamard_transform(flat)
                q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, global_scale)
                if use_hadamard:
                    q = inverse_hadamard_transform(q)
                tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)

    return quantize_end


# ---------------------------------------------------------------------------
# NVFP4 per-head calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_per_head(
    model,
    calibration_input_ids: torch.Tensor,
    block_size: int,
    quant_context=None,
) -> NVFP4KVCacheScales:
    """Calibrate per-head NVFP4 global scales (one scale per KV head per layer)."""
    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(
            f"Calibration prompt ({calibration_input_ids.shape[1]} tokens) "
            f"must be >= block_size ({block_size})"
        )

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScales(num_layers)

    for layer_idx in range(num_layers):
        # [B, H, T, D] -> per-head amax over [B, T, D]
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()
        k_rotated = hadamard_transform(k_data)
        v_rotated = hadamard_transform(v_data)
        # amax per head: [H]
        k_amax = k_rotated.abs().amax(dim=(0, 2, 3)).clamp(min=1e-12)
        v_amax = v_rotated.abs().amax(dim=(0, 2, 3)).clamp(min=1e-12)
        scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# K quantization sensitivity analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_k_quantization_sensitivity(
    model,
    calibration_input_ids: torch.Tensor,
    block_size: int,
    quant_context=None,
):
    """Measure per-layer and per-head K quantization error (MSE and cosine).

    Returns a dict with per-layer and per-head error metrics to guide
    mixed-precision decisions (which layers/heads need BF16 K).
    """
    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    results = {
        "num_layers": num_layers,
        "per_layer_mse": [],      # [num_layers]
        "per_layer_cos": [],      # [num_layers]  (1 - cosine_sim)
        "per_head_mse": [],       # [num_layers][H]
        "per_head_cos": [],       # [num_layers][H]
    }

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()  # [B, H, T, D]
        B, H, T, D = k_data.shape

        # Hadamard rotate (same as quantization pipeline)
        k_rotated = hadamard_transform(k_data)

        # Compute per-layer frozen scale (amax over entire layer)
        k_amax = k_rotated.abs().amax().clamp(min=1e-12)
        k_global_scale = (k_amax / FP4_E2M1_MAX).to(torch.float32)

        # Quantize
        flat = k_rotated.reshape(-1, D)
        k_quant = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, k_global_scale)
        k_quant = k_quant.reshape(B, H, T, D)

        # Inverse Hadamard to get quantized K in original domain
        k_deq = inverse_hadamard_transform(k_quant)

        # Per-layer error
        mse = (k_data - k_deq).pow(2).mean().item()
        cos_err = 1.0 - torch.nn.functional.cosine_similarity(
            k_data.reshape(-1, D), k_deq.reshape(-1, D), dim=1
        ).mean().item()
        results["per_layer_mse"].append(mse)
        results["per_layer_cos"].append(cos_err)

        # Per-head error
        head_mse = []
        head_cos = []
        for h in range(H):
            k_h = k_data[:, h, :, :].reshape(-1, D)
            k_dq_h = k_deq[:, h, :, :].reshape(-1, D)
            head_mse.append((k_h - k_dq_h).pow(2).mean().item())
            head_cos.append(1.0 - torch.nn.functional.cosine_similarity(
                k_h, k_dq_h, dim=1
            ).mean().item())
        results["per_head_mse"].append(head_mse)
        results["per_head_cos"].append(head_cos)

    return results


# ---------------------------------------------------------------------------
# Strategy 1: Attention-aware scale calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_attn_aware(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    n_alphas=50,
    quant_context=None,
):
    """Calibrate NVFP4 scales using attention-aware loss (Gram matrix proxy).

    For target layers, grid-searches the scale alpha that minimizes
    ||K K^T - K_quant K_quant^T||^2 instead of ||K - K_quant||^2.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScales(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        k_rotated = hadamard_transform(k_data)
        k_amax = k_rotated.abs().amax().clamp(min=1e-12)

        if layer_idx not in target_layers:
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            B, H, T, D = k_data.shape
            k_flat_orig = k_data.reshape(B * H, T, D)
            gram_orig = torch.bmm(k_flat_orig, k_flat_orig.transpose(1, 2))

            best_alpha, best_loss = 1.0, float('inf')
            for alpha in torch.linspace(0.02, 1.0, n_alphas):
                scale = alpha * k_amax / FP4_E2M1_MAX
                k_flat = k_rotated.reshape(-1, D)
                k_quant = pseudo_nvfp4_quantize_tensor_with_global_scale(k_flat, scale)
                k_deq = inverse_hadamard_transform(k_quant.reshape(B, H, T, D))

                k_flat_deq = k_deq.reshape(B * H, T, D)
                gram_quant = torch.bmm(k_flat_deq, k_flat_deq.transpose(1, 2))
                loss = ((gram_orig - gram_quant) ** 2).mean().item()

                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha.item()

            scales.k_scales[layer_idx] = (best_alpha * k_amax / FP4_E2M1_MAX).to(torch.float32)
            print(f"  [attn_aware] Layer {layer_idx}: best_alpha={best_alpha:.3f}, "
                  f"gram_loss={best_loss:.6f}")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 2: Random permutation before Hadamard
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_with_perm(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    n_random_perms=200,
    quant_context=None,
):
    """Calibrate NVFP4 scales with random permutation search before Hadamard."""
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            B, H, T, D = k_data.shape
            k_flat = k_data.reshape(-1, D)

            # Baseline: identity permutation
            k_rot = hadamard_transform(k_flat)
            amax = k_rot.abs().amax().clamp(min=1e-12)
            scale = amax / FP4_E2M1_MAX
            k_q = pseudo_nvfp4_quantize_tensor_with_global_scale(k_rot, scale)
            k_deq = inverse_hadamard_transform(k_q)
            baseline_mse = (k_flat - k_deq).pow(2).mean().item()
            best_mse = baseline_mse
            best_perm = None
            print(f"  [perm_hadamard] Layer {layer_idx}: baseline MSE={baseline_mse:.4f}")

            for _ in range(n_random_perms):
                perm = torch.randperm(D, device=k_flat.device)
                k_perm = k_flat[:, perm]
                k_rot = hadamard_transform(k_perm)
                amax_p = k_rot.abs().amax().clamp(min=1e-12)
                scale_p = amax_p / FP4_E2M1_MAX
                k_q = pseudo_nvfp4_quantize_tensor_with_global_scale(k_rot, scale_p)
                k_deq = inverse_hadamard_transform(k_q)
                inv_perm = perm.argsort()
                k_deq = k_deq[:, inv_perm]
                mse = (k_flat - k_deq).pow(2).mean().item()
                if mse < best_mse:
                    best_mse = mse
                    best_perm = perm

            if best_perm is not None:
                scales.k_perms[layer_idx] = best_perm
                scales.k_inv_perms[layer_idx] = best_perm.argsort()
                k_perm = k_flat[:, best_perm]
                k_rot = hadamard_transform(k_perm)
                amax = k_rot.abs().amax().clamp(min=1e-12)
                scales.k_scales[layer_idx] = (amax / FP4_E2M1_MAX).to(torch.float32)
                improv = (1 - best_mse / baseline_mse) * 100
                print(f"  [perm_hadamard] Layer {layer_idx}: best MSE={best_mse:.4f} "
                      f"({improv:.1f}% improvement)")
            else:
                k_rot = hadamard_transform(k_flat)
                amax = k_rot.abs().amax().clamp(min=1e-12)
                scales.k_scales[layer_idx] = (amax / FP4_E2M1_MAX).to(torch.float32)
                print(f"  [perm_hadamard] Layer {layer_idx}: identity is best")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 3: Local block rotation + zigzag permutation + per-channel scaling
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_local_rot(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    quant_context=None,
):
    """Calibrate NVFP4 with zigzag permutation + block rotation + per-channel scaling."""
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H, T, D = k_data.shape

            # Zigzag permutation based on original-domain channel magnitudes
            ch_amax = k_data.abs().amax(dim=(0, 1, 2))  # [D]
            perm = build_zigzag_permutation(ch_amax, rot_block_size)
            inv_perm = perm.argsort()
            scales.k_perms[layer_idx] = perm.to(k_data.device)
            scales.k_inv_perms[layer_idx] = inv_perm.to(k_data.device)

            # Apply permutation + block Hadamard
            k_perm = k_data[..., perm.to(k_data.device)]
            k_rotated = block_hadamard_transform(k_perm, rot_block_size)

            # Per-channel amax in rotated domain
            per_ch_amax = k_rotated.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)
            scales.k_per_channel_scales[layer_idx] = (per_ch_amax / FP4_E2M1_MAX).to(torch.float32)

            # Dummy per-tensor scale for compatibility
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

            ratio = per_ch_amax.max() / per_ch_amax.min()
            print(f"  [local_rot] Layer {layer_idx}: zigzag + block rot + per-channel "
                  f"(range [{per_ch_amax.min():.2f}, {per_ch_amax.max():.2f}], ratio={ratio:.1f}x)")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 4: Block rotation + per-channel scaling (no permutation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_block_rot(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    quant_context=None,
):
    """Calibrate NVFP4 with block-diagonal Hadamard + per-channel scaling (no perm)."""
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            k_rotated = block_hadamard_transform(k_data, rot_block_size)

            per_ch_amax = k_rotated.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)
            scales.k_per_channel_scales[layer_idx] = (per_ch_amax / FP4_E2M1_MAX).to(torch.float32)

            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

            ratio = per_ch_amax.max() / per_ch_amax.min()
            print(f"  [block_rot] Layer {layer_idx}: block rot + per-channel "
                  f"(range [{per_ch_amax.min():.2f}, {per_ch_amax.max():.2f}], ratio={ratio:.1f}x)")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 4b: Block rotation + per-channel scaling + per-block clipping
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_block_rot_clip(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    n_alphas=30,
    quant_context=None,
):
    """Block rotation + per-channel scaling with per-block MSE-optimal clipping.

    For each NVFP4 block of 16 channels, grid-searches a clip ratio alpha that
    minimizes MSE. Different blocks get different clip ratios — the outlier block
    benefits from aggressive clipping while normal blocks stay near alpha=1.0.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H, T, D = k_data.shape
            n_blocks = D // rot_block_size

            k_rotated = block_hadamard_transform(k_data, rot_block_size)

            # Per-channel amax (baseline, no clipping)
            per_ch_amax = k_rotated.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)  # [D]

            # Grid-search clip ratio per NVFP4 block of 16
            k_flat = k_data.reshape(-1, D)  # original domain for MSE
            k_rot_flat = k_rotated.reshape(-1, D)  # rotated domain for quantization
            best_per_ch_scale = per_ch_amax.clone() / FP4_E2M1_MAX

            for blk in range(n_blocks):
                ch_start = blk * rot_block_size
                ch_end = ch_start + rot_block_size
                blk_amax = per_ch_amax[ch_start:ch_end]  # [16]

                best_alpha, best_mse = 1.0, float('inf')
                for alpha in torch.linspace(0.1, 1.0, n_alphas):
                    trial_scale = best_per_ch_scale.clone()
                    trial_scale[ch_start:ch_end] = alpha * blk_amax / FP4_E2M1_MAX
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                        k_rot_flat, trial_scale.unsqueeze(0))
                    q_orig = inverse_block_hadamard_transform(q.reshape(B, H, T, D), rot_block_size)
                    mse = (k_data - q_orig).pow(2).mean().item()
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha.item()

                best_per_ch_scale[ch_start:ch_end] = best_alpha * blk_amax / FP4_E2M1_MAX
                print(f"    Block {blk} (ch {ch_start}-{ch_end-1}): "
                      f"alpha={best_alpha:.3f}, amax_range=[{blk_amax.min():.1f}, {blk_amax.max():.1f}]")

            scales.k_per_channel_scales[layer_idx] = best_per_ch_scale.to(torch.float32)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

            # Report total MSE
            q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                k_rot_flat, best_per_ch_scale.unsqueeze(0))
            q_orig = inverse_block_hadamard_transform(q.reshape(B, H, T, D), rot_block_size)
            final_mse = (k_data - q_orig).pow(2).mean().item()
            print(f"  [block_rot_clip] Layer {layer_idx}: final MSE={final_mse:.4f}")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 5: Block rotation + mean shift + per-channel scaling (+ optional clip)
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_block_rot_shift(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    clip=False,
    n_alphas=30,
    quant_context=None,
):
    """Block rotation + per-channel mean shift + per-channel scaling.

    Subtracts per-channel mean in the rotated domain before quantization.
    This removes constant biases from outlier channels, dramatically reducing
    the dynamic range that NVFP4 must represent.

    Storage overhead: 128 FP32 means per target layer = 512 bytes.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H, T, D = k_data.shape

            k_rotated = block_hadamard_transform(k_data, rot_block_size)

            # Per-channel mean in rotated domain
            per_ch_mean = k_rotated.mean(dim=(0, 1, 2))  # [D]
            scales.k_per_channel_means[layer_idx] = per_ch_mean.to(torch.float32)

            # Residual after mean subtraction
            k_centered = k_rotated - per_ch_mean
            residual_amax = k_centered.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)  # [D]
            per_ch_scale = residual_amax / FP4_E2M1_MAX  # [D]

            if clip:
                # Grid-search per-block clip ratio on the residual
                n_blocks = D // rot_block_size
                k_centered_flat = k_centered.reshape(-1, D)
                for blk in range(n_blocks):
                    ch_s = blk * rot_block_size
                    ch_e = ch_s + rot_block_size
                    blk_amax = residual_amax[ch_s:ch_e]
                    best_alpha, best_mse = 1.0, float('inf')
                    for alpha in torch.linspace(0.1, 1.0, n_alphas):
                        trial = per_ch_scale.clone()
                        trial[ch_s:ch_e] = alpha * blk_amax / FP4_E2M1_MAX
                        q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                            k_centered_flat, trial.unsqueeze(0))
                        q_shifted = q + per_ch_mean.unsqueeze(0)
                        q_orig = inverse_block_hadamard_transform(
                            q_shifted.reshape(B, H, T, D), rot_block_size)
                        mse = (k_data - q_orig).pow(2).mean().item()
                        if mse < best_mse:
                            best_mse = mse
                            best_alpha = alpha.item()
                    per_ch_scale[ch_s:ch_e] = best_alpha * blk_amax / FP4_E2M1_MAX
                    print(f"    Block {blk} (ch {ch_s}-{ch_e-1}): alpha={best_alpha:.3f}, "
                          f"resid_amax=[{blk_amax.min():.1f}, {blk_amax.max():.1f}]")

            scales.k_per_channel_scales[layer_idx] = per_ch_scale.to(torch.float32)
            # Fallback scalar scale (not used at runtime for this layer)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

            # Report MSE
            k_centered_flat = k_centered.reshape(-1, D)
            q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                k_centered_flat, per_ch_scale.unsqueeze(0))
            q_shifted = q + per_ch_mean.unsqueeze(0)
            q_orig = inverse_block_hadamard_transform(
                q_shifted.reshape(B, H, T, D), rot_block_size)
            final_mse = (k_data - q_orig).pow(2).mean().item()
            tag = "block_rot_shift_clip" if clip else "block_rot_shift"
            print(f"  [{tag}] Layer {layer_idx}: final MSE={final_mse:.4f}")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Strategy 6: Hybrid channel-adaptive (mean-shift for const-bias, clip for variable)
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_hybrid(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    mean_ratio_threshold=3.0,
    n_alphas=30,
    quant_context=None,
):
    """Hybrid channel-adaptive: mean-shift for constant-bias, clip for variable.

    For each rotated channel in target layers:
      - If |mean| / residual_amax > threshold: "constant-bias" channel
        → subtract mean, use fine scale (residual_amax / 6)
      - Else: "variable" channel
        → no mean subtraction, use per-block MSE-optimal clipped scale

    The per-channel mean vector has zeros for variable channels, so the
    existing runtime mean-shift code path works unchanged.

    Storage: 128 FP32 means + 128 FP32 scales per target layer.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H, T, D = k_data.shape
            n_blocks = D // rot_block_size

            k_rotated = block_hadamard_transform(k_data, rot_block_size)

            # Per-channel statistics in rotated domain
            per_ch_mean = k_rotated.mean(dim=(0, 1, 2))  # [D]
            per_ch_amax = k_rotated.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)  # [D]
            k_centered = k_rotated - per_ch_mean
            residual_amax = k_centered.abs().amax(dim=(0, 1, 2)).clamp(min=1e-12)  # [D]

            # Classify channels: constant-bias vs variable
            ratio = per_ch_mean.abs() / residual_amax  # [D]
            is_const_bias = ratio > mean_ratio_threshold  # [D] bool
            n_const = is_const_bias.sum().item()
            n_var = D - n_const

            print(f"  [hybrid] Layer {layer_idx}: {n_const} constant-bias channels "
                  f"(ratio>{mean_ratio_threshold}), {n_var} variable channels")

            # Build hybrid mean: keep mean for const-bias, zero for variable
            hybrid_mean = torch.where(is_const_bias, per_ch_mean, torch.zeros_like(per_ch_mean))
            scales.k_per_channel_means[layer_idx] = hybrid_mean.to(torch.float32)

            # Build hybrid input: mean-shifted for const-bias, raw for variable
            # k_hybrid[d] = k_rotated[d] - hybrid_mean[d]
            #   const-bias channels: k_rotated[d] - mean[d] = residual (small)
            #   variable channels:   k_rotated[d] - 0 = k_rotated[d] (unchanged)
            k_hybrid = k_rotated - hybrid_mean

            # Initial per-channel scale:
            #   const-bias: residual_amax / 6 (fine scale)
            #   variable:   per_ch_amax / 6 (full range, no clip yet)
            per_ch_scale = torch.where(
                is_const_bias,
                residual_amax / FP4_E2M1_MAX,
                per_ch_amax / FP4_E2M1_MAX,
            )

            # Grid-search per-block clip alpha (only affects variable channels)
            k_hybrid_flat = k_hybrid.reshape(-1, D)
            for blk in range(n_blocks):
                ch_s = blk * rot_block_size
                ch_e = ch_s + rot_block_size
                blk_const = is_const_bias[ch_s:ch_e]

                # Skip blocks that are entirely constant-bias (no clipping needed)
                if blk_const.all():
                    print(f"    Block {blk} (ch {ch_s}-{ch_e-1}): all constant-bias, alpha=1.000")
                    continue

                # For variable channels in this block, search clip alpha
                blk_amax_var = per_ch_amax[ch_s:ch_e]  # amax for variable channels
                blk_amax_const = residual_amax[ch_s:ch_e]  # resid_amax for const-bias

                best_alpha, best_mse = 1.0, float('inf')
                for alpha in torch.linspace(0.05, 1.0, n_alphas):
                    trial = per_ch_scale.clone()
                    # Only clip the variable channels in this block
                    for i in range(rot_block_size):
                        d = ch_s + i
                        if not is_const_bias[d]:
                            trial[d] = alpha * per_ch_amax[d] / FP4_E2M1_MAX

                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                        k_hybrid_flat, trial.unsqueeze(0))
                    q_shifted = q + hybrid_mean.unsqueeze(0)
                    q_orig = inverse_block_hadamard_transform(
                        q_shifted.reshape(B, H, T, D), rot_block_size)
                    mse = (k_data - q_orig).pow(2).mean().item()
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha.item()

                # Apply best alpha to variable channels in this block
                for i in range(rot_block_size):
                    d = ch_s + i
                    if not is_const_bias[d]:
                        per_ch_scale[d] = best_alpha * per_ch_amax[d] / FP4_E2M1_MAX

                n_var_blk = (~blk_const).sum().item()
                n_const_blk = blk_const.sum().item()
                print(f"    Block {blk} (ch {ch_s}-{ch_e-1}): "
                      f"{n_const_blk} const-bias + {n_var_blk} variable, "
                      f"var_alpha={best_alpha:.3f}")

            scales.k_per_channel_scales[layer_idx] = per_ch_scale.to(torch.float32)
            # Fallback scalar scale
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

            # Report final MSE
            q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                k_hybrid_flat, per_ch_scale.unsqueeze(0))
            q_shifted = q + hybrid_mean.unsqueeze(0)
            q_orig = inverse_block_hadamard_transform(
                q_shifted.reshape(B, H, T, D), rot_block_size)
            final_mse = (k_data - q_orig).pow(2).mean().item()
            print(f"  [hybrid] Layer {layer_idx}: final MSE={final_mse:.4f}")

    scales.calibrated = True
    return scales


@torch.no_grad()
def calibrate_nvfp4_kv_scales_pre_shift_block_rot(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    mean_ratio_threshold=5.0,
    n_alphas=50,
    quant_context=None,
):
    """Pre-rotation mean shift + block Hadamard + per-channel scale + per-block clip.

    For target layers:
      1. Identify constant-bias channels in ORIGINAL domain (|mean|/std > threshold)
      2. Subtract their means BEFORE block Hadamard rotation
      3. Block Hadamard on the shifted data → much more uniform distribution
      4. Per-channel scale + per-block clip search in rotated domain

    This prevents the Hadamard from spreading constant biases across all 16
    channels in a block, which inflates local scales and wastes precision.

    Storage: D FP32 pre-shift means + D FP32 per-channel scales per target layer.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H, T, D = k_data.shape
            n_blocks = D // rot_block_size

            # Step 1: Identify constant-bias dims in original domain
            orig_mean = k_data.mean(dim=(0, 1, 2))  # [D]
            orig_std = k_data.std(dim=(0, 1, 2)).clamp(min=1e-12)  # [D]
            ratio = orig_mean.abs() / orig_std
            is_const_bias = ratio > mean_ratio_threshold

            # Build pre-shift mean: only for constant-bias dims
            pre_shift_mean = torch.where(is_const_bias, orig_mean, torch.zeros_like(orig_mean))
            scales.k_pre_shift_means[layer_idx] = pre_shift_mean.to(torch.float32)

            n_shifted = is_const_bias.sum().item()
            shifted_dims = is_const_bias.nonzero(as_tuple=True)[0].tolist()
            print(f"  [pre_shift] Layer {layer_idx}: shifting {n_shifted} original-domain dims: {shifted_dims}")
            for d in shifted_dims:
                print(f"    dim {d}: mean={orig_mean[d]:.1f}, std={orig_std[d]:.1f}, |mean|/std={ratio[d]:.1f}")

            # Step 2: Subtract pre-shift mean, then block Hadamard
            k_shifted = k_data - pre_shift_mean
            k_rotated = block_hadamard_transform(k_shifted, rot_block_size)

            # Step 3: Per-channel scale in rotated domain
            k_rot_flat = k_rotated.reshape(-1, D)
            per_ch_amax = k_rot_flat.abs().amax(dim=0).clamp(min=1e-12)  # [D]
            per_ch_scale = per_ch_amax / FP4_E2M1_MAX  # [D]

            # Step 4: Per-block clip search (MSE in original domain)
            for blk in range(n_blocks):
                ch_s = blk * rot_block_size
                ch_e = ch_s + rot_block_size
                blk_amax = per_ch_amax[ch_s:ch_e]
                best_alpha, best_mse = 1.0, float('inf')
                for alpha in torch.linspace(0.05, 1.0, n_alphas):
                    trial = per_ch_scale.clone()
                    trial[ch_s:ch_e] = alpha * blk_amax / FP4_E2M1_MAX
                    q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                        k_rot_flat, trial.unsqueeze(0))
                    q_orig = inverse_block_hadamard_transform(
                        q.reshape(B, H, T, D), rot_block_size)
                    q_orig = q_orig + pre_shift_mean  # add pre-shift mean back
                    mse = (k_data - q_orig).pow(2).mean().item()
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha.item()
                per_ch_scale[ch_s:ch_e] = best_alpha * blk_amax / FP4_E2M1_MAX
                print(f"    Block {blk} (ch {ch_s}-{ch_e-1}): alpha={best_alpha:.3f}, mse={best_mse:.4f}")

            scales.k_per_channel_scales[layer_idx] = per_ch_scale.to(torch.float32)
            # Fallback scalar scale
            k_amax_global = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax_global / FP4_E2M1_MAX).to(torch.float32)

            # Report final MSE
            q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                k_rot_flat, per_ch_scale.unsqueeze(0))
            q_orig = inverse_block_hadamard_transform(
                q.reshape(B, H, T, D), rot_block_size)
            q_orig = q_orig + pre_shift_mean
            final_mse = (k_data - q_orig).pow(2).mean().item()
            print(f"  [pre_shift] Layer {layer_idx}: final MSE={final_mse:.4f}")

    scales.calibrated = True
    return scales


@torch.no_grad()
def calibrate_nvfp4_kv_scales_attn_loss(
    model,
    calibration_input_ids,
    block_size,
    target_layers=None,
    rot_block_size=16,
    n_alphas=50,
    quant_context=None,
):
    """Block rotation + per-channel scale + per-block clip using ATTENTION loss.

    Instead of minimizing K MSE, minimizes:
        ||softmax(Q·K^T/√d) - softmax(Q·K_quant^T/√d)||²_F

    This calibrates scales based on what matters for downstream attention,
    not raw K reconstruction fidelity.
    """
    if target_layers is None:
        target_layers = {0}

    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(f"Calibration prompt too short for block_size={block_size}")

    # Step 1: Forward pass with hooks to capture Q (post-RoPE) for target layers
    captured_q = {}

    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model
    attn_module = raw_model.model.layers[0].self_attn
    _apply_rotary_pos_emb = type(attn_module).forward.__globals__['apply_rotary_pos_emb']

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # Called with all kwargs — extract hidden_states and position_embeddings
            hidden_states = kwargs.get('hidden_states', args[0] if args else None)
            position_embeddings = kwargs.get('position_embeddings')
            cos, sin = position_embeddings
            hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
            captured_q[layer_idx] = query_states.detach().float()
        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        h = raw_model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True, update_past_key_values=True, block_size=block_size,
        )

    for h in hooks:
        h.remove()

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)
    scales = NVFP4KVCacheScalesExtended(num_layers)

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()
        v_data = past_kv.value_cache[layer_idx].float()

        v_rotated = hadamard_transform(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        if layer_idx not in target_layers:
            k_rotated = hadamard_transform(k_data)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)
        else:
            scales.use_block_rotation[layer_idx] = True
            B, H_kv, T, D = k_data.shape
            n_blocks = D // rot_block_size

            # Get captured Q for this layer: [B, H_q, T, D]
            q_data = captured_q[layer_idx]  # [B, 12, T, 128]
            H_q = q_data.shape[1]
            num_kv_groups = H_q // H_kv  # 6

            # Compute reference attention scores: softmax(Q·K^T/√d)
            # Use GQA: expand K to match Q heads
            k_expanded = k_data.unsqueeze(2).expand(B, H_kv, num_kv_groups, T, D)
            k_expanded = k_expanded.reshape(B, H_q, T, D)  # [B, 12, T, 128]
            scale_factor = D ** 0.5
            # Compute in chunks to save memory (T can be 2048)
            # Full attention: [B, H_q, T, T] — 12*2048*2048*4 = 192MB, manageable
            attn_ref = torch.matmul(q_data, k_expanded.transpose(-2, -1)) / scale_factor
            attn_ref = torch.softmax(attn_ref, dim=-1)  # [B, H_q, T, T]

            # Block Hadamard on K
            k_rotated = block_hadamard_transform(k_data, rot_block_size)
            k_rot_flat = k_rotated.reshape(-1, D)

            # Per-channel amax
            per_ch_amax = k_rot_flat.abs().amax(dim=0).clamp(min=1e-12)
            best_per_ch_scale = per_ch_amax.clone() / FP4_E2M1_MAX

            # Grid-search clip ratio per block using attention loss
            for blk in range(n_blocks):
                ch_s = blk * rot_block_size
                ch_e = ch_s + rot_block_size
                blk_amax = per_ch_amax[ch_s:ch_e]

                best_alpha, best_loss = 1.0, float('inf')
                for alpha in torch.linspace(0.05, 1.0, n_alphas):
                    trial_scale = best_per_ch_scale.clone()
                    trial_scale[ch_s:ch_e] = alpha * blk_amax / FP4_E2M1_MAX
                    q_nvfp4 = pseudo_nvfp4_quantize_tensor_with_global_scale(
                        k_rot_flat, trial_scale.unsqueeze(0))
                    k_quant = inverse_block_hadamard_transform(
                        q_nvfp4.reshape(B, H_kv, T, D), rot_block_size)

                    # Compute attention with quantized K
                    k_q_expanded = k_quant.unsqueeze(2).expand(B, H_kv, num_kv_groups, T, D)
                    k_q_expanded = k_q_expanded.reshape(B, H_q, T, D)
                    attn_quant = torch.matmul(q_data, k_q_expanded.transpose(-2, -1)) / scale_factor
                    attn_quant = torch.softmax(attn_quant, dim=-1)

                    loss = (attn_ref - attn_quant).pow(2).mean().item()
                    if loss < best_loss:
                        best_loss = loss
                        best_alpha = alpha.item()

                best_per_ch_scale[ch_s:ch_e] = best_alpha * blk_amax / FP4_E2M1_MAX

                # Also compute MSE for logging
                trial_scale = best_per_ch_scale.clone()
                q_nvfp4 = pseudo_nvfp4_quantize_tensor_with_global_scale(
                    k_rot_flat, trial_scale.unsqueeze(0))
                k_quant = inverse_block_hadamard_transform(
                    q_nvfp4.reshape(B, H_kv, T, D), rot_block_size)
                mse = (k_data - k_quant).pow(2).mean().item()
                print(f"    Block {blk} (ch {ch_s}-{ch_e-1}): alpha={best_alpha:.3f}, "
                      f"attn_loss={best_loss:.2e}, K_MSE={mse:.4f}")

            scales.k_per_channel_scales[layer_idx] = best_per_ch_scale.to(torch.float32)
            k_amax_global = k_rotated.abs().amax().clamp(min=1e-12)
            scales.k_scales[layer_idx] = (k_amax_global / FP4_E2M1_MAX).to(torch.float32)

            # Final MSE and attention loss
            q_nvfp4 = pseudo_nvfp4_quantize_tensor_with_global_scale(
                k_rot_flat, best_per_ch_scale.unsqueeze(0))
            k_quant = inverse_block_hadamard_transform(
                q_nvfp4.reshape(B, H_kv, T, D), rot_block_size)
            final_mse = (k_data - k_quant).pow(2).mean().item()
            k_q_expanded = k_quant.unsqueeze(2).expand(B, H_kv, num_kv_groups, T, D)
            k_q_expanded = k_q_expanded.reshape(B, H_q, T, D)
            attn_final = torch.matmul(q_data, k_q_expanded.transpose(-2, -1)) / scale_factor
            attn_final = torch.softmax(attn_final, dim=-1)
            final_attn_loss = (attn_ref - attn_final).pow(2).mean().item()
            print(f"  [attn_loss] Layer {layer_idx}: K_MSE={final_mse:.4f}, "
                  f"attn_loss={final_attn_loss:.2e}")

    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Bias-aware NVFP4 calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_nvfp4_kv_scales_bias_subtract(
    model,
    calibration_input_ids: torch.Tensor,
    block_size: int,
    rotation: str = "global",
    quant_context=None,
) -> NVFP4KVCacheScales:
    """Calibrate NVFP4 scales after subtracting k_proj.bias from K cache.

    The k_proj bias (a learned constant per channel) dominates the K cache
    magnitude in early layers (layer 0: bias amax=316 vs residual amax=14.5).
    By subtracting it before rotation + quantization and adding it back after,
    we quantize only the data-dependent residual, dramatically reducing error.

    Args:
        rotation: "global" (128-dim Hadamard), "block" (8x 16-dim Hadamard),
                  or "none" (no rotation).
    """
    prefill_len = (calibration_input_ids.shape[1] // block_size) * block_size
    if prefill_len == 0:
        raise ValueError(
            f"Calibration prompt ({calibration_input_ids.shape[1]} tokens) "
            f"must be >= block_size ({block_size})"
        )

    def rotate(x):
        if rotation == "global":
            return hadamard_transform(x)
        elif rotation == "block":
            return block_hadamard_transform(x)
        return x

    ctx = quant_context if quant_context is not None else nullcontext()
    with ctx:
        out = model.forward(
            input_ids=calibration_input_ids[:, :prefill_len],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )

    past_kv = out.past_key_values
    num_layers = len(past_kv.key_cache)

    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model

    scales = NVFP4KVCacheScales(num_layers)
    scales.k_biases = [None] * num_layers
    scales.rotation_mode = rotation

    for layer_idx in range(num_layers):
        k_data = past_kv.key_cache[layer_idx].float()  # [B, H, T, D]
        v_data = past_kv.value_cache[layer_idx].float()
        B, H, T, D = k_data.shape

        # Get k_proj.bias: [num_kv_heads * head_dim] -> [1, H, 1, D]
        k_bias = raw_model.model.layers[layer_idx].self_attn.k_proj.bias.data.float()
        k_bias = k_bias.reshape(1, H, 1, D).to(k_data.device)
        scales.k_biases[layer_idx] = k_bias.squeeze(0).squeeze(1)  # [H, D] for storage

        # Subtract bias, then rotate
        k_residual = k_data - k_bias
        k_rotated = rotate(k_residual)

        k_amax = k_rotated.abs().amax().clamp(min=1e-12)
        scales.k_scales[layer_idx] = (k_amax / FP4_E2M1_MAX).to(torch.float32)

        # V uses same rotation (no bias subtraction needed)
        v_rotated = rotate(v_data)
        v_amax = v_rotated.abs().amax().clamp(min=1e-12)
        scales.v_scales[layer_idx] = (v_amax / FP4_E2M1_MAX).to(torch.float32)

        # Log improvement
        orig_amax = rotate(k_data).abs().amax().item()
        new_amax = k_rotated.abs().amax().item()
        bias_amax = k_bias.abs().max().item()
        if bias_amax > 1.0:
            rot_label = {"global": "H", "block": "BH", "none": ""}[rotation]
            if rot_label:
                print(f"  [bias_sub/{rotation}] Layer {layer_idx}: "
                      f"bias_amax={bias_amax:.1f}, "
                      f"{rot_label}(K) amax {orig_amax:.1f} -> {rot_label}(K-bias) amax {new_amax:.1f} "
                      f"({orig_amax/new_amax:.1f}x reduction)")
            else:
                print(f"  [bias_sub/{rotation}] Layer {layer_idx}: "
                      f"bias_amax={bias_amax:.1f}, "
                      f"K amax {orig_amax:.1f} -> (K-bias) amax {new_amax:.1f} "
                      f"({orig_amax/new_amax:.1f}x reduction)")

    scales.calibrated = True
    return scales


def load_alpaca_calib_sequences(tokenizer, num_samples=128, max_length=2048,
                                 min_length=32):
    """Load Alpaca training conversations as calibration sequences.

    Formats each conversation with apply_chat_template, truncates from the left
    if longer than max_length, and returns the longest num_samples sequences.
    """
    import json
    import os

    alpaca_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "alpaca",
        "train_conversation", "train_52002.json",
    )
    alpaca_path = os.path.normpath(alpaca_path)
    with open(alpaca_path, "r") as f:
        data = json.load(f)

    sequences = []
    for instance in data["instances"]:
        messages = instance["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        if len(ids) > max_length:
            ids = ids[-max_length:]  # truncate from the left
        if len(ids) >= min_length:
            sequences.append(ids.unsqueeze(0))  # [1, seq_len]

    # Sort by length descending, take top num_samples
    sequences.sort(key=lambda x: x.shape[1], reverse=True)
    sequences = sequences[:num_samples]
    lengths = [s.shape[1] for s in sequences]
    print(f"[Alpaca calib] Loaded {len(sequences)} sequences "
          f"(len range: {min(lengths)}-{max(lengths)}, "
          f"median: {lengths[len(lengths)//2]})")
    return sequences


def generate_calib_sequences(model, tokenizer, num_samples=128,
                              max_new_tokens=256, block_size=32,
                              threshold=0.9):
    """Generate calibration sequences using BF16 model on Alpaca prompts.

    Loads Alpaca user prompts, generates responses with block-diffusion
    sampling, returns prompt+response token sequences.
    """
    import json
    import os
    import types
    from generation_functions import Fast_dLLM_QwenForCausalLM

    alpaca_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "alpaca",
        "train_conversation", "train_52002.json",
    )
    alpaca_path = os.path.normpath(alpaca_path)
    with open(alpaca_path, "r") as f:
        data = json.load(f)

    # Extract user prompts, format with chat template
    prompts = []
    for instance in data["instances"]:
        messages = instance["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
        if not user_msg:
            continue
        chat = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        prompts.append(ids)

    # Sort by length descending, take top num_samples
    prompts.sort(key=lambda x: len(x), reverse=True)
    prompts = prompts[:num_samples]
    print(f"[Generate calib] Generating {len(prompts)} sequences "
          f"(prompt len range: {len(prompts[-1])}-{len(prompts[0])})")

    # Monkey-patch batch_sample onto model
    if not hasattr(model, "batch_sample"):
        model.batch_sample = types.MethodType(
            Fast_dLLM_QwenForCausalLM.batch_sample, model)

    device = next(model.parameters()).device
    mask_id = 151665
    stop_token = 151645
    small_block_size = 8

    sequences = []
    for i, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids.unsqueeze(0).to(device)
        seq_len = torch.tensor([len(prompt_ids)], device=device)
        try:
            output_ids = model.batch_sample(
                input_ids=input_ids,
                tokenizer=tokenizer,
                block_size=block_size,
                max_new_tokens=max_new_tokens,
                small_block_size=small_block_size,
                min_len=len(prompt_ids),
                seq_len=seq_len,
                mask_id=mask_id,
                threshold=threshold,
                stop_token=stop_token,
                use_block_cache=False,
                top_p=0.95,
                temperature=0.0,
            )
            sequences.append(output_ids[0:1].cpu())  # [1, total_len]
        except Exception as e:
            print(f"  [Generate calib] Warning: sample {i} failed: {e}")
            # Fall back to just the prompt
            sequences.append(input_ids.cpu())

        if (i + 1) % 10 == 0:
            print(f"  [Generate calib] {i+1}/{len(prompts)} done")

    lengths = [s.shape[1] for s in sequences]
    print(f"[Generate calib] Generated {len(sequences)} sequences "
          f"(len range: {min(lengths)}-{max(lengths)})")
    return sequences


@torch.no_grad()
def calibrate_nvfp4_kv_scales_bias_subtract_multi(
    model,
    calibration_sequences,
    block_size: int,
    rotation: str = "global",
    quant_context=None,
) -> NVFP4KVCacheScales:
    """Calibrate NVFP4 scales with bias subtraction using multiple sequences.

    Runs a separate forward pass for each sequence and takes the max amax
    per layer across all sequences. This produces more representative scales
    than concatenating unrelated text into a single sequence.
    """
    def rotate(x):
        if rotation == "global":
            return hadamard_transform(x)
        elif rotation == "block":
            return block_hadamard_transform(x)
        return x

    raw_model = model.module if hasattr(model, 'module') else model
    device = next(model.parameters()).device
    ctx = quant_context if quant_context is not None else nullcontext()

    # Initialize running max amax per layer (will be set on first sequence)
    max_k_amax = None
    max_v_amax = None
    num_layers = None

    for seq_idx, seq in enumerate(calibration_sequences):
        seq = seq.to(device)
        prefill_len = (seq.shape[1] // block_size) * block_size
        if prefill_len == 0:
            continue

        with ctx:
            out = model.forward(
                input_ids=seq[:, :prefill_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )

        past_kv = out.past_key_values

        if num_layers is None:
            num_layers = len(past_kv.key_cache)
            max_k_amax = [torch.tensor(0.0, device=device)] * num_layers
            max_v_amax = [torch.tensor(0.0, device=device)] * num_layers

        for layer_idx in range(num_layers):
            k_data = past_kv.key_cache[layer_idx].float()
            v_data = past_kv.value_cache[layer_idx].float()
            B, H, T, D = k_data.shape

            k_bias = raw_model.model.layers[layer_idx].self_attn.k_proj.bias.data.float()
            k_bias = k_bias.reshape(1, H, 1, D).to(device)

            k_residual = k_data - k_bias
            k_rotated = rotate(k_residual)
            k_amax = k_rotated.abs().amax().clamp(min=1e-12)
            max_k_amax[layer_idx] = torch.max(max_k_amax[layer_idx], k_amax)

            v_rotated = rotate(v_data)
            v_amax = v_rotated.abs().amax().clamp(min=1e-12)
            max_v_amax[layer_idx] = torch.max(max_v_amax[layer_idx], v_amax)

        # Free memory
        del out, past_kv
        torch.cuda.empty_cache()

        if (seq_idx + 1) % 20 == 0:
            print(f"  [Multi-calib] {seq_idx+1}/{len(calibration_sequences)} sequences done")

    if num_layers is None:
        raise ValueError("No valid calibration sequences (all too short?)")

    # Build scales
    scales = NVFP4KVCacheScales(num_layers)
    scales.k_biases = [None] * num_layers
    scales.rotation_mode = rotation

    for layer_idx in range(num_layers):
        scales.k_scales[layer_idx] = (max_k_amax[layer_idx] / FP4_E2M1_MAX).to(torch.float32)
        scales.v_scales[layer_idx] = (max_v_amax[layer_idx] / FP4_E2M1_MAX).to(torch.float32)

        k_bias = raw_model.model.layers[layer_idx].self_attn.k_proj.bias.data.float()
        H_heads = k_bias.shape[0] // (raw_model.config.hidden_size // raw_model.config.num_attention_heads)
        D_head = raw_model.config.hidden_size // raw_model.config.num_attention_heads
        scales.k_biases[layer_idx] = k_bias.reshape(H_heads, D_head)

        bias_amax = k_bias.abs().max().item()
        if bias_amax > 1.0:
            rot_label = {"global": "H", "block": "BH", "none": ""}[rotation]
            print(f"  [bias_sub_multi/{rotation}] Layer {layer_idx}: "
                  f"bias_amax={bias_amax:.1f}, "
                  f"max_k_amax={max_k_amax[layer_idx].item():.2f}, "
                  f"k_scale={scales.k_scales[layer_idx].item():.4f}")

    print(f"[Multi-calib] Calibrated {num_layers} layers from "
          f"{len(calibration_sequences)} sequences (rotation={rotation})")
    scales.calibrated = True
    return scales


# ---------------------------------------------------------------------------
# Mixed-precision K quantization
# ---------------------------------------------------------------------------

def quantize_kv_cache_mixed_k(
    past_key_values,
    scales: NVFP4KVCacheScales,
    fp8_k_scales,
    num_old_tokens: int,
    bf16_window_size: int = 0,
    k_mode: str = "fp8",
    k_bf16_layers: set = None,
    k_bf16_heads: set = None,
):
    """Mixed-precision KV cache: FP4 for V, configurable precision for K.

    Args:
        past_key_values: DynamicCache.
        scales: NVFP4 scales for V (and FP4 K layers if any).
        fp8_k_scales: FP8 scales for K (used when k_mode='fp8'). Can be None.
        num_old_tokens: Tokens already quantized.
        bf16_window_size: KIVI-style BF16 window.
        k_mode: How to handle K:
            - 'fp8': All K layers use FP8 + Hadamard
            - 'mixed_layer': Some layers BF16, rest FP4
            - 'mixed_head': Some heads BF16, rest FP4
        k_bf16_layers: Set of layer indices whose K stays BF16 (for mixed_layer).
        k_bf16_heads: Set of (layer_idx, head_idx) tuples whose K stays BF16
                      (for mixed_head).

    Returns:
        int: quantize_end (new num_old_tokens).
    """
    if scales is not None:
        assert scales.calibrated, "Scales must be calibrated before use"
    num_layers = len(past_key_values.key_cache)
    total_len = past_key_values.key_cache[0].shape[2]
    quantize_end = max(num_old_tokens, total_len - bf16_window_size) if bf16_window_size > 0 else total_len

    if quantize_end <= num_old_tokens:
        return quantize_end

    k_bf16_layers = k_bf16_layers or set()
    k_bf16_heads = k_bf16_heads or set()

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max  # 448.0

    for layer_idx in range(num_layers):
        # --- V: always FP4 with Hadamard ---
        v_tensor = past_key_values.value_cache[layer_idx]  # [B, H, T, D]
        v_slice = v_tensor[:, :, num_old_tokens:quantize_end, :]
        orig_dtype = v_slice.dtype
        orig_shape = v_slice.shape
        v_global_scale = scales.v_scales[layer_idx] if scales else None

        flat = v_slice.reshape(-1, orig_shape[-1]).float()
        flat = hadamard_transform(flat)
        q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, v_global_scale)
        q = inverse_hadamard_transform(q)
        v_tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)

        # --- K: depends on mode ---
        k_tensor = past_key_values.key_cache[layer_idx]  # [B, H, T, D]
        k_slice = k_tensor[:, :, num_old_tokens:quantize_end, :]
        orig_shape = k_slice.shape
        B, H, T_new, D = orig_shape

        if k_mode == "fp8":
            # FP8 quantize all K with Hadamard
            flat = k_slice.reshape(-1, D).float()
            flat = hadamard_transform(flat)
            if fp8_k_scales is not None:
                scale = fp8_k_scales.k_scales[layer_idx]
                x_scaled = flat / scale
                x_scaled = x_scaled.clamp(-fp8_max, fp8_max)
                x_fp8 = x_scaled.to(fp8_dtype)
                q = x_fp8.to(torch.float32) * scale
            else:
                # Dynamic FP8
                amax = flat.abs().amax().clamp(min=1e-12)
                scale = amax / fp8_max
                x_scaled = (flat / scale).clamp(-fp8_max, fp8_max)
                q = x_scaled.to(fp8_dtype).to(torch.float32) * scale
            q = inverse_hadamard_transform(q)
            k_tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)

        elif k_mode == "mixed_layer":
            if layer_idx in k_bf16_layers:
                pass  # Skip — K stays BF16
            else:
                # FP4 quantize K
                k_global_scale = scales.k_scales[layer_idx] if scales else None
                flat = k_slice.reshape(-1, D).float()
                flat = hadamard_transform(flat)
                q = pseudo_nvfp4_quantize_tensor_with_global_scale(flat, k_global_scale)
                q = inverse_hadamard_transform(q)
                k_tensor[:, :, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(orig_shape)

        elif k_mode == "mixed_head":
            k_global_scale = scales.k_scales[layer_idx] if scales else None
            for h in range(H):
                if (layer_idx, h) in k_bf16_heads:
                    continue  # Skip — this head stays BF16
                head_slice = k_slice[:, h, :, :].reshape(-1, D).float()
                head_slice = hadamard_transform(head_slice)
                q = pseudo_nvfp4_quantize_tensor_with_global_scale(head_slice, k_global_scale)
                q = inverse_hadamard_transform(q)
                k_tensor[:, h, num_old_tokens:quantize_end, :] = q.to(orig_dtype).reshape(B, T_new, D)

    return quantize_end
