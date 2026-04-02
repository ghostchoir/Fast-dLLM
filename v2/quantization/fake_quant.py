"""Permanent fake quantization for lm-eval-harness evaluation."""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from quantization.quantizer import (
    pseudo_quantize_tensor,
    pseudo_fp8_quantize_tensor,
    pseudo_nvfp4_quantize_tensor,
    pseudo_nvfp4_quantize_tensor_with_global_scale,
)

# Scheme -> (n_bit, group_size) for integer weight quantization
WEIGHT_QUANT_CONFIG = {
    "w4g128": (4, 128),
    "w8a8":   (8, -1),    # per-channel for weights
}

# FP8-based schemes
FP8_SCHEMES = {"fp-w8a8"}

# NVFP4-based schemes
NVFP4_SCHEMES = {"nvfp-w4a4"}

# Schemes that also quantize activations
ACT_QUANT_SCHEMES = {"w8a8", "fp-w8a8", "nvfp-w4a4"}

# All valid schemes
ALL_QUANT_SCHEMES = set(WEIGHT_QUANT_CONFIG) | FP8_SCHEMES | NVFP4_SCHEMES

# Modules to skip
SKIP_MODULES = {"lm_head", "embed_tokens"}


def _load_learned_scales(model_path: Optional[str]) -> Optional[Dict[str, float]]:
    """Load learned_scales.pt from model directory if it exists."""
    if model_path is None:
        return None
    scales_path = os.path.join(model_path, "learned_scales.pt")
    if os.path.exists(scales_path):
        scales = torch.load(scales_path, map_location="cpu", weights_only=True)
        print(f"[FakeQuant] Loaded {len(scales)} learned weight scales from {scales_path}")
        return scales
    return None


def _load_learned_lac(model_path: Optional[str]) -> Optional[Dict[str, float]]:
    """Load learned_lac.pt (LAC clip ratios) from model directory if it exists."""
    if model_path is None:
        return None
    lac_path = os.path.join(model_path, "learned_lac.pt")
    if os.path.exists(lac_path):
        lac = torch.load(lac_path, map_location="cpu", weights_only=True)
        print(f"[FakeQuant] Loaded {len(lac)} LAC clip ratios from {lac_path}")
        return lac
    return None


def _load_act_clip_values(model_path: Optional[str]) -> Optional[Dict[str, float]]:
    """Load act_clip_values.pt (fixed activation clip vals) from model directory."""
    if model_path is None:
        return None
    path = os.path.join(model_path, "act_clip_values.pt")
    if os.path.exists(path):
        vals = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[FakeQuant] Loaded {len(vals)} fixed activation clip values from {path}")
        return vals
    return None


def _load_act_outlier_blocks(model_path: Optional[str]) -> Optional[Dict[str, list]]:
    """Load act_outlier_blocks.pt (dual global scale config) from model directory."""
    if model_path is None:
        return None
    path = os.path.join(model_path, "act_outlier_blocks.pt")
    if os.path.exists(path):
        blocks = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[FakeQuant] Loaded {len(blocks)} act outlier block configs from {path}")
        return blocks
    return None


def _is_gptq_model(model_path: Optional[str]) -> bool:
    """Check if model directory contains a GPTQ marker file."""
    if model_path is None:
        return False
    marker = os.path.join(model_path, "gptq_optimized.json")
    return os.path.exists(marker)


def apply_fake_quantization(model, quant_scheme, model_path=None, r4_block_size=0,
                            act_outlier_channels=None, skip_quant_modules=None):
    """
    Permanently fake-quantize all nn.Linear weights in-place and install
    activation quantization hooks for W+A schemes.

    If ``model_path`` points to a directory containing ``learned_scales.pt``
    (saved by distill_qat.py when training with ``--learnable_nvfp4_scales``),
    those scales are used instead of the default amax-derived RTN scales.

    If ``model_path`` contains ``gptq_optimized.json``, weight re-quantization
    is skipped (weights are already GPTQ-optimized dequantized values).
    Activation quantization hooks are still applied.

    If ``r4_block_size`` > 0, applies R4 online rotation (SpinQuant R4):
    pre-rotates down_proj columns and applies block Hadamard to activations
    before quantization.

    Supported schemes:
        w4g128    : INT4 weights, group_size=128, no activation quantization
        w8a8      : INT8 weights per-channel, INT8 activations per-token
        fp-w8a8   : FP8 weights per-tensor, FP8 activations per-tensor
        nvfp-w4a4 : NVFP4 weights per-tensor, NVFP4 activations per-tensor
    """
    assert quant_scheme in ALL_QUANT_SCHEMES, \
        f"Unknown quant_scheme '{quant_scheme}'. Choose from {sorted(ALL_QUANT_SCHEMES)}"

    is_fp8 = quant_scheme in FP8_SCHEMES
    is_nvfp4 = quant_scheme in NVFP4_SCHEMES
    if not is_fp8 and not is_nvfp4:
        n_bit, group_size = WEIGHT_QUANT_CONFIG[quant_scheme]

    gptq_skip_weight_quant = _is_gptq_model(model_path) if is_nvfp4 else False
    if gptq_skip_weight_quant:
        print("[FakeQuant] GPTQ model detected — skipping weight re-quantization")

    learned_scales = _load_learned_scales(model_path) if is_nvfp4 else None
    learned_lac = _load_learned_lac(model_path) if is_nvfp4 else None
    act_clip_vals = _load_act_clip_values(model_path) if is_nvfp4 else None
    act_outlier_blocks = _load_act_outlier_blocks(model_path) if is_nvfp4 else None

    # Build outlier block config from act_outlier_channels if provided and no file loaded
    if act_outlier_channels is not None and act_outlier_blocks is None and is_nvfp4:
        act_outlier_blocks = {}
        for layer_idx, channels in act_outlier_channels.items():
            mod_name = f"model.layers.{layer_idx}.mlp.down_proj"
            block_ids = sorted(set(ch // 16 for ch in channels))
            act_outlier_blocks[mod_name] = block_ids
            print(f"  [ActOutlier] {mod_name}: channels {channels} -> outlier blocks {block_ids}")
        print(f"[ActOutlier] Built dual global scale config for {len(act_outlier_blocks)} layers from act_outlier_channels")

    # R4 online rotation: identify down_proj modules for activation rotation hooks.
    # Weight pre-rotation is NOT done here — QAT bakes it into saved weights.
    # Only the online activation rotation (block Hadamard before quantization) is needed.
    down_proj_ids = set()
    if r4_block_size > 0:
        for layer_module in model.model.layers:
            down_proj_ids.add(id(layer_module.mlp.down_proj))
        print(f"[FakeQuant] R4 online rotation enabled for {len(down_proj_ids)} "
              f"down_proj layers (block_size={r4_block_size})")

    count = 0
    learned_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in SKIP_MODULES):
            continue
        if skip_quant_modules and any(skip in name for skip in skip_quant_modules):
            print(f"[FakeQuant] SKIPPING quantization for {name}")
            continue

        # Quantize weights in-place (skip for GPTQ models — already optimized)
        if gptq_skip_weight_quant:
            pass  # weights are already GPTQ-optimized dequantized values
        else:
            w = module.weight.data
            if is_nvfp4:
                if learned_scales is not None and name in learned_scales:
                    gs = torch.tensor(learned_scales[name], dtype=torch.float32)
                    w_q = pseudo_nvfp4_quantize_tensor_with_global_scale(
                        w.float(), gs,
                    ).to(w.dtype)
                    learned_count += 1
                else:
                    w_q = pseudo_nvfp4_quantize_tensor(
                        w.float(), per_tensor_global=True,
                    ).to(w.dtype)
            elif is_fp8:
                w_q = pseudo_fp8_quantize_tensor(w.float(), per_tensor=True).to(w.dtype)
            else:
                w_q = pseudo_quantize_tensor(
                    w.float(), n_bit=n_bit, zero_point=True, w_group_size=group_size,
                ).to(w.dtype)
            module.weight.data.copy_(w_q)
        count += 1

        # Activation hooks for w8a8 / fp-w8a8 / nvfp-w4a4
        if quant_scheme in ACT_QUANT_SCHEMES:
            is_r4_target = r4_block_size > 0 and id(module) in down_proj_ids
            lac_ratio = learned_lac.get(name) if learned_lac is not None else None
            fixed_clip = act_clip_vals.get(name) if act_clip_vals is not None else None
            outlier_bids = act_outlier_blocks.get(name) if act_outlier_blocks is not None else None
            if is_nvfp4:
                def make_nvfp4_hook(apply_r4=False, block_size=16, clip_ratio=None,
                                    act_clip_val=None, outlier_block_ids=None):
                    def hook(mod, inputs):
                        x = inputs[0]
                        if apply_r4:
                            from quantization.kv_cache_quant import block_hadamard_transform
                            x = block_hadamard_transform(x, block_size)
                        if clip_ratio is not None:
                            amax = x.abs().amax().clamp(min=1e-12)
                            clip_val = clip_ratio * amax
                            x = x.clamp(-clip_val, clip_val)
                        if act_clip_val is not None:
                            x = x.clamp(-act_clip_val, act_clip_val)
                        x_q = pseudo_nvfp4_quantize_tensor(
                            x.float(), per_tensor_global=True,
                            outlier_block_ids=outlier_block_ids,
                        ).to(x.dtype)
                        return (x_q,) + inputs[1:]
                    return hook
                module.register_forward_pre_hook(
                    make_nvfp4_hook(is_r4_target, r4_block_size, lac_ratio, fixed_clip, outlier_bids))
            elif is_fp8:
                def make_fp8_hook():
                    def hook(mod, inputs):
                        x = inputs[0]
                        x_q = pseudo_fp8_quantize_tensor(
                            x.float(), per_tensor=True,
                        ).to(x.dtype)
                        return (x_q,) + inputs[1:]
                    return hook
                module.register_forward_pre_hook(make_fp8_hook())
            else:
                def make_int_hook(n_bit):
                    def hook(mod, inputs):
                        x = inputs[0]
                        orig_shape = x.shape
                        orig_dtype = x.dtype
                        x_q = pseudo_quantize_tensor(
                            x.reshape(-1, orig_shape[-1]).float(),
                            n_bit=n_bit, zero_point=True, w_group_size=-1,
                        ).to(orig_dtype).reshape(orig_shape)
                        return (x_q,) + inputs[1:]
                    return hook
                module.register_forward_pre_hook(make_int_hook(8))

    msg = f"[FakeQuant] Applied '{quant_scheme}' to {count} Linear modules"
    if learned_count > 0:
        msg += f" ({learned_count} with learned scales)"
    if learned_lac is not None:
        msg += f" ({len(learned_lac)} with LAC)"
    if act_clip_vals is not None:
        msg += f" ({len(act_clip_vals)} with fixed act clip)"
    if act_outlier_blocks is not None:
        msg += f" ({len(act_outlier_blocks)} with dual global scale)"
    print(msg)
