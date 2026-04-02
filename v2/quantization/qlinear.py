"""QLinear: nn.Linear subclass with STE-based fake quantizers for QAT.

Supports independent quantization of weights, input activations, and
output (KV cache) with INT4/8, FP8, and NVFP4 formats.
"""

import sys
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.quantizer import (
    SteIntAsymQuantizer,
    SteFp8Quantizer,
    SteNvfp4Quantizer,
    SteNvfp4QuantizerFrozenScale,
    UniformAffineQuantizer,
)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_weight_quantizer(
    quant_type: str,
    bit: int = 4,
    group_size: int = 128,
    enabled: bool = True,
    learnable_qp: bool = False,
    weight: Optional[torch.Tensor] = None,
    learnable_global_scale: bool = False,
) -> nn.Module:
    """Create an STE-based quantizer for weights."""
    if quant_type == "int":
        return SteIntAsymQuantizer(
            bit=bit, w_group_size=group_size,
            enabled=enabled, learnable_qp=learnable_qp, weight=weight,
        )
    elif quant_type == "fp8":
        return SteFp8Quantizer(per_tensor=True, enabled=enabled)
    elif quant_type == "nvfp4":
        return SteNvfp4Quantizer(
            enabled=enabled,
            learnable_global_scale=learnable_global_scale,
        )
    raise ValueError(f"Unknown weight quant type: {quant_type}")


def create_act_quantizer(
    quant_type: str,
    bit: int = 8,
) -> nn.Module:
    """Create a quantizer for activations or KV cache outputs."""
    if quant_type == "int":
        return UniformAffineQuantizer(
            n_bits=bit, dynamic=True, dynamic_method="per_token",
        )
    elif quant_type == "fp8":
        return SteFp8Quantizer(per_tensor=True)
    elif quant_type == "nvfp4":
        return SteNvfp4Quantizer()
    raise ValueError(f"Unknown activation quant type: {quant_type}")


# ---------------------------------------------------------------------------
# QLinear
# ---------------------------------------------------------------------------

class QLinear(nn.Linear):
    """Linear layer with optional STE-based fake quantization on weights,
    input activations, and output (KV cache for k_proj/v_proj).

    Args:
        in_features, out_features, bias: Same as nn.Linear.
        compute_dtype: Dtype for computation (default bf16).
        w_quant_type: Weight quantizer format ('int', 'fp8', 'nvfp4').
        w_bit: Bit-width for INT weight quantization.
        w_group_size: Group size for INT weight quantization (-1 = per-channel).
        a_quant_type: Input activation quantizer ('none', 'int', 'fp8', 'nvfp4').
        a_bit: Bit-width for INT activation quantization.
        kv_quant_type: Output KV cache quantizer ('none', 'int', 'fp8', 'nvfp4').
        kv_bit: Bit-width for INT KV cache quantization.
        is_kv_proj: Whether this is a k_proj or v_proj (enables output quantizer).
        quantizer_enabled: Initial enabled state for all quantizers.
        learnable_qp: Whether to use learnable quantization parameters (INT only).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        # Weight quantization
        w_quant_type: str = "int",
        w_bit: int = 4,
        w_group_size: int = 128,
        # Input activation quantization
        a_quant_type: str = "none",
        a_bit: int = 8,
        # Output (KV cache) quantization
        kv_quant_type: str = "none",
        kv_bit: int = 8,
        is_kv_proj: bool = False,
        # Control
        quantizer_enabled: bool = True,
        learnable_qp: bool = False,
        weight: Optional[torch.Tensor] = None,
        learnable_global_scale: bool = False,
        lwc: bool = False,
        lac: bool = False,
    ):
        super().__init__(in_features, out_features, bias)
        self.compute_dtype = compute_dtype

        # Weight quantizer
        self.w_quant_type = w_quant_type
        self.weight_quantizer = create_weight_quantizer(
            w_quant_type, w_bit, w_group_size,
            enabled=quantizer_enabled,
            learnable_qp=learnable_qp,
            weight=weight,
            learnable_global_scale=learnable_global_scale,
        )

        # Learnable Weight Clipping (OmniQuant LWC, per-NVFP4-group)
        if lwc:
            num_groups = out_features * (in_features // 16)
            self.lwc_upbound = nn.Parameter(torch.ones(num_groups, 1) * 4.0)
        else:
            self.lwc_upbound = None

        # Learnable Activation Clipping (LAC, per-tensor)
        # sigmoid(4.0) = 0.982 → clips top ~2%, gradient 0.018 (7x larger than init=6)
        if lac:
            self.lac_upbound = nn.Parameter(torch.tensor(3.0))
        else:
            self.lac_upbound = None

        # Input activation quantizer
        self.a_quant_type = a_quant_type
        if a_quant_type != "none":
            self.input_quantizer = create_act_quantizer(a_quant_type, a_bit)
        else:
            self.input_quantizer = None

        # Output (KV cache) quantizer -- only for k_proj / v_proj
        self.kv_quant_type = kv_quant_type
        if is_kv_proj and kv_quant_type != "none":
            self.output_quantizer = create_act_quantizer(kv_quant_type, kv_bit)
        else:
            self.output_quantizer = None

        # Fixed activation clipping (for outlier layers like down_proj at L04/L05/L26)
        # Set externally via set_act_clip_values() after model conversion
        self.act_clip_val = None

        # R4 online rotation (block Hadamard on input before quantization)
        # Set externally via enable_r4_rotation() for down_proj modules
        self.r4_block_size = 0  # 0 = disabled

    def enable_weight_quant(self):
        self.weight_quantizer.enable()

    def disable_weight_quant(self):
        self.weight_quantizer.disable()

    @torch.no_grad()
    def bake_weights(self):
        """Apply LWC + fake quantization permanently to weights and disable both."""
        w = self.weight
        if self.lwc_upbound is not None:
            w_grouped = w.reshape(-1, 16)
            group_amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            clip_val = torch.sigmoid(self.lwc_upbound) * group_amax
            w = w_grouped.clamp(-clip_val, clip_val).reshape(self.weight.shape)
        if hasattr(self.weight_quantizer, 'quantize_weight'):
            quantized = self.weight_quantizer.quantize_weight(w)
        else:
            quantized = self.weight_quantizer(w)
        self.weight.data.copy_(quantized.to(self.weight.dtype))
        self.weight_quantizer.disable()
        self.lwc_upbound = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_dtype = x.dtype
        x = x.to(self.compute_dtype)

        # 0. R4 online rotation (block Hadamard before quantization)
        if self.r4_block_size > 0:
            from quantization.kv_cache_quant import block_hadamard_transform
            x = block_hadamard_transform(x, self.r4_block_size)

        # 0b. LAC: learnable activation clipping before quantization
        if self.lac_upbound is not None:
            with torch.no_grad():
                amax = x.abs().amax().clamp(min=1e-12)
            clip_val = torch.sigmoid(self.lac_upbound) * amax
            x = x.clamp(-clip_val, clip_val)

        # 0c. Fixed activation clipping (calibrated p99.9 for outlier layers)
        if self.act_clip_val is not None:
            x = x.clamp(-self.act_clip_val, self.act_clip_val)

        # 1. Quantize input activation
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)

        # 2. Apply LWC (learnable weight clipping) if enabled
        w = self.weight
        if self.lwc_upbound is not None:
            w_grouped = w.reshape(-1, 16)  # [num_groups, 16]
            with torch.no_grad():
                group_amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            clip_val = torch.sigmoid(self.lwc_upbound) * group_amax
            w = w_grouped.clamp(-clip_val, clip_val).reshape(self.weight.shape)

        # 3. Quantize weight via STE
        quantized_weight = self.weight_quantizer(w).to(self.compute_dtype)

        # 4. F.linear in compute_dtype (BF16) — matches inference precision.
        # CUDA tensor cores use FP32 accumulation internally for BF16 inputs.
        b = self.bias.to(self.compute_dtype) if self.bias is not None else None
        out = F.linear(x, quantized_weight, b)

        # 5. Quantize output for KV cache (k_proj / v_proj only)
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)

        return out.to(inp_dtype)

    def extra_repr(self):
        parts = [
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
            f'w_quant={self.w_quant_type}',
        ]
        if self.a_quant_type != "none":
            parts.append(f'a_quant={self.a_quant_type}')
        if self.output_quantizer is not None:
            parts.append(f'kv_quant={self.kv_quant_type}')
        return ', '.join(parts)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

SKIP_MODULES = {"lm_head", "embed_tokens"}


def convert_model_to_quant(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    # Weight
    w_quant_type: str = "int",
    w_bit: int = 4,
    w_group_size: int = 128,
    # Activation
    a_quant_type: str = "none",
    a_bit: int = 8,
    # KV cache (applied to k_proj, v_proj output)
    kv_quant_type: str = "none",
    kv_bit: int = 8,
    # Control
    compute_dtype: torch.dtype = torch.bfloat16,
    quantizer_enabled: bool = True,
    learnable_qp: bool = False,
    learnable_global_scale: bool = False,
    lwc: bool = False,
    lac: bool = False,
) -> Tuple[nn.Module, List[str]]:
    """Replace nn.Linear modules with QLinear for QAT.

    Recursively walks the model. k_proj and v_proj get output quantizers
    for KV cache quantization. lm_head and embed_tokens are skipped.

    Returns:
        (model, list_of_converted_module_names)
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = list(SKIP_MODULES)

    converted = []

    def _convert(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, QLinear):
                if any(skip in name for skip in modules_to_not_convert):
                    _convert(child, full_name)
                    continue

                is_kv = ("k_proj" in name or "v_proj" in name)
                ql = QLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    w_quant_type=w_quant_type, w_bit=w_bit, w_group_size=w_group_size,
                    a_quant_type=a_quant_type, a_bit=a_bit,
                    kv_quant_type=kv_quant_type, kv_bit=kv_bit,
                    is_kv_proj=is_kv,
                    quantizer_enabled=quantizer_enabled,
                    learnable_qp=learnable_qp,
                    weight=child.weight if learnable_qp else None,
                    learnable_global_scale=learnable_global_scale,
                    lwc=lwc,
                    lac=lac,
                )
                # Copy weight and bias
                ql.weight = child.weight
                if child.bias is not None:
                    ql.bias = child.bias

                # Move any newly-created params to same device as weight
                device = child.weight.device
                for pname, param in ql.named_parameters():
                    if param.device != device:
                        param.data = param.data.to(device)

                setattr(module, name, ql)
                converted.append(full_name)
            else:
                _convert(child, full_name)

    _convert(model)
    return model, converted


def set_act_clip_values(model: nn.Module, clip_layers: Dict[int, float]):
    """Set fixed activation clip values on specific down_proj modules.

    Args:
        model: Model with QLinear modules (after convert_model_to_quant).
        clip_layers: {layer_idx: clip_val} — calibrated p99.9 thresholds
            for down_proj input activations at outlier layers.
    """
    count = 0
    for layer_idx, clip_val in clip_layers.items():
        layer = model.model.layers[layer_idx]
        down_proj = layer.mlp.down_proj
        if isinstance(down_proj, QLinear):
            down_proj.act_clip_val = clip_val
            count += 1
            print(f"  [ActClip] Layer {layer_idx} down_proj: clip_val={clip_val:.1f}")
    print(f"[ActClip] Set fixed clipping on {count} layers")


def set_act_outlier_channels(model: nn.Module, layer_channels: Dict[int, List[int]]):
    """Set outlier channel indices on specific down_proj input quantizers.

    Outlier channels get their own FP32 global scale in NVFP4, while normal
    blocks share a global scale computed without the outlier blocks. This
    prevents extreme outlier channels from inflating the global scale and
    killing precision for the other 99%+ of blocks.

    Args:
        model: Model with QLinear modules (after convert_model_to_quant).
        layer_channels: {layer_idx: [channel_indices]}
            e.g. {4: [610], 5: [8729], 26: [2538]}
    """
    count = 0
    for layer_idx, channels in layer_channels.items():
        layer = model.model.layers[layer_idx]
        down_proj = layer.mlp.down_proj
        if isinstance(down_proj, QLinear) and down_proj.input_quantizer is not None:
            block_ids = sorted(set(ch // 16 for ch in channels))
            iq = down_proj.input_quantizer
            iq.outlier_block_ids = block_ids
            # Register learnable outlier scale params if learnable_global_scale
            if iq.learnable_global_scale and iq._outlier_scales is None:
                iq._outlier_scales = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.0)) for _ in block_ids
                ])
            count += 1
            learnable_tag = " (learnable)" if iq.learnable_global_scale else ""
            print(f"  [ActOutlier] Layer {layer_idx} down_proj: "
                  f"channels {channels} -> outlier blocks {block_ids}{learnable_tag}")
    print(f"[ActOutlier] Set dual global scale on {count} layers")


def convert_quant_to_linear(model: nn.Module) -> nn.Module:
    """Bake weights and replace QLinear with nn.Linear for deployment."""

    def _convert(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, QLinear):
                child.bake_weights()
                with torch.no_grad():
                    new_linear = nn.Linear(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    new_linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        new_linear.bias.data.copy_(child.bias.data)
                setattr(module, name, new_linear)
            else:
                _convert(child)

    _convert(model)
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_qlinear_modules(
    model: nn.Module,
    layer_idx: Optional[int] = None,
    proj_type: Optional[str] = None,
) -> Dict[str, QLinear]:
    """Find QLinear modules with optional filtering."""
    results = {}
    for full_name, child in model.named_modules():
        if not isinstance(child, QLinear):
            continue
        if proj_type is not None and proj_type not in full_name:
            continue
        if layer_idx is not None:
            # Extract layer index from name like "model.layers.5.self_attn.k_proj"
            parts = full_name.split('.')
            found_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    found_idx = int(parts[i + 1])
                    break
            if found_idx != layer_idx:
                continue
        results[full_name] = child
    return results


def set_qlinear_quantizer_state(model: nn.Module, enabled: bool):
    """Toggle all quantizers in QLinear modules."""
    for _, child in model.named_modules():
        if isinstance(child, QLinear):
            child.weight_quantizer.enabled = enabled
            if child.input_quantizer is not None:
                child.input_quantizer.enable = enabled
            if child.output_quantizer is not None:
                child.output_quantizer.enable = enabled


def bake_all_weights(model: nn.Module):
    """Bake and freeze all QLinear weights."""
    for _, child in model.named_modules():
        if isinstance(child, QLinear):
            child.bake_weights()


@torch.no_grad()
def prepare_model_for_eval_save(
    model: nn.Module,
    save_path: str,
    save_fn=None,
):
    """Bake LWC + extract learned scales, save, then restore weights.

    Non-destructive: weights are restored after saving so training can
    continue (safe to call at checkpoint time).

    Args:
        model: QAT model with QLinear modules.
        save_path: Directory to save ``learned_scales.pt`` into.
        save_fn: Callable that performs the actual model save (e.g.
            ``lambda: (model.save_pretrained(path), tok.save_pretrained(path))``).
            Called while LWC is baked into weights.
    """
    import os

    originals: Dict[str, torch.Tensor] = {}
    learned_scales: Dict[str, float] = {}

    for full_name, child in model.named_modules():
        if not isinstance(child, QLinear):
            continue

        # 1. Bake LWC: apply clipping, stash original weights
        if child.lwc_upbound is not None:
            originals[full_name] = child.weight.data.clone()
            w_grouped = child.weight.data.reshape(-1, 16)
            group_amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            clip_val = torch.sigmoid(child.lwc_upbound) * group_amax
            child.weight.data.copy_(
                w_grouped.clamp(-clip_val, clip_val).reshape(child.weight.shape)
            )

        # 2. Extract learned global scale
        wq = child.weight_quantizer
        if (isinstance(wq, SteNvfp4Quantizer)
                and wq.learnable_global_scale
                and wq._global_scale is not None):
            learned_scales[full_name] = torch.exp(wq._global_scale).item()

    # Save learned scales
    if learned_scales:
        os.makedirs(save_path, exist_ok=True)
        scales_path = os.path.join(save_path, "learned_scales.pt")
        torch.save(learned_scales, scales_path)
        print(f"  Saved {len(learned_scales)} learned weight scales → {scales_path}")

    # Perform the actual model save (with LWC baked into weights)
    if save_fn is not None:
        save_fn()

    # Restore original weights so training can continue
    for full_name, child in model.named_modules():
        if full_name in originals:
            child.weight.data.copy_(originals[full_name])


# ---------------------------------------------------------------------------
# Post-RoPE KV cache quantization (attached to attention modules)
# ---------------------------------------------------------------------------

def _create_kv_quantizer(kv_quant_type, frozen_scale, learnable_global_scale=False,
                          lac=False):
    """Create a KV quantizer for a single layer's K or V."""
    if kv_quant_type == "nvfp4":
        return SteNvfp4QuantizerFrozenScale(
            frozen_scale,
            learnable_global_scale=learnable_global_scale,
            lac=lac,
        )
    else:
        raise ValueError(
            f"Unsupported kv_quant_type for frozen-scale attach: {kv_quant_type}. "
            f"Currently only 'nvfp4' is supported."
        )


def attach_kv_quantizers(model, kv_scales, kv_quant_type="nvfp4", rotation="global",
                         learnable_kv_scales=False, lac=False):
    """Attach KV cache quantizers to attention modules (post-RoPE).

    Instead of quantizing at the k_proj/v_proj output (pre-RoPE, inside QLinear),
    this attaches quantizers that fire after RoPE application, matching eval-time
    behavior in ``kv_cache_quant.py``.

    The attention class's forward method is monkey-patched to inject quantization
    between RoPE and the cache-update / attention-computation steps.

    Args:
        model: The transformer model (e.g. Fast_dLLM_QwenForCausalLM).
        kv_scales: Calibrated scale object (NVFP4KVCacheScales or FP8KVCacheScales)
            with per-layer ``.k_scales`` and ``.v_scales`` lists.
        kv_quant_type: Quantization format ('nvfp4').
        rotation: Rotation mode for Hadamard transform:
            - 'global': Full 128-dim Hadamard (default)
            - 'block': Block-diagonal 8x16-dim Hadamard
            - 'none': No rotation
    """
    # Find all attention modules by looking for modules with q_proj + k_proj + v_proj
    attn_modules = []
    for name, module in model.named_modules():
        if (hasattr(module, 'q_proj') and hasattr(module, 'k_proj')
                and hasattr(module, 'v_proj') and hasattr(module, 'o_proj')
                and hasattr(module, 'layer_idx')):
            attn_modules.append((name, module))

    if not attn_modules:
        raise RuntimeError("No attention modules found in the model.")

    # Check if bias subtraction data is available
    k_biases = getattr(kv_scales, 'k_biases', None)

    # Attach per-layer quantizers as submodules
    for name, attn in attn_modules:
        layer_idx = attn.layer_idx
        attn.k_quantizer = _create_kv_quantizer(
            kv_quant_type, kv_scales.k_scales[layer_idx],
            learnable_global_scale=learnable_kv_scales,
            lac=lac,
        )
        attn.v_quantizer = _create_kv_quantizer(
            kv_quant_type, kv_scales.v_scales[layer_idx],
            learnable_global_scale=learnable_kv_scales,
            lac=lac,
        )
        # Flag for live bias subtraction (reads k_proj.bias at runtime)
        if k_biases is not None and k_biases[layer_idx] is not None:
            attn._kv_quant_bias_subtract = True
        attn._kv_quant_rotation = rotation

    # Monkey-patch the attention class's forward method (all instances share it)
    attn_cls = type(attn_modules[0][1])
    if hasattr(attn_cls, '_original_forward_before_kv_quant'):
        # Already patched
        return

    attn_cls._original_forward_before_kv_quant = attn_cls.forward

    # Get references to functions from the modeling module
    modeling_module = sys.modules[attn_cls.__module__]
    _apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb
    _fused_flex_attention = modeling_module.fused_flex_attention
    _ALL_ATTENTION_FUNCTIONS = modeling_module.ALL_ATTENTION_FUNCTIONS

    def _patched_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        update_past_key_values=False,
        block_past_key_values=None,
        replace_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.training:
            # Split for [x_t, x_0] complementary masking
            q_1 = query_states[:, :, :query_states.shape[2] // 2]
            q_2 = query_states[:, :, query_states.shape[2] // 2:]
            k_1 = key_states[:, :, :key_states.shape[2] // 2]
            k_2 = key_states[:, :, key_states.shape[2] // 2:]
            q_1, k_1 = _apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = _apply_rotary_pos_emb(q_2, k_2, cos, sin)
            query_states = torch.cat((q_1, q_2), dim=-2)
            key_states = torch.cat((k_1, k_2), dim=-2)
        else:
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # >>> KV QUANTIZATION (post-RoPE, matching eval-time behavior) <<<
        from quantization.hadamard import hadamard_transform, inverse_hadamard_transform
        from quantization.kv_cache_quant import (
            block_hadamard_transform, inverse_block_hadamard_transform,
        )

        rot = getattr(self, '_kv_quant_rotation', 'global')
        def _rotate(x):
            if rot == "global": return hadamard_transform(x)
            if rot == "block": return block_hadamard_transform(x)
            return x
        def _inv_rotate(x):
            if rot == "global": return inverse_hadamard_transform(x)
            if rot == "block": return inverse_block_hadamard_transform(x)
            return x

        kv_dtype = key_states.dtype
        active_mask = getattr(type(self), '_kv_quant_active_mask', None)
        if hasattr(self, 'k_quantizer') and self.k_quantizer is not None:
            key_pre_quant = key_states  # save reference before bias sub + quant
            # Bias subtraction: subtract live k_proj.bias before rotation+quantize.
            # The bias cancels in softmax (shift-invariance) so we don't add it back.
            # Use .detach() so bias receives gradients only from k_proj, not from
            # the subtraction path (which would create conflicting gradients).
            if getattr(self, '_kv_quant_bias_subtract', False):
                num_kv_heads = key_states.shape[1]
                head_dim = key_states.shape[3]
                live_bias = self.k_proj.bias.detach().view(num_kv_heads, head_dim)
                key_states = key_states - live_bias.unsqueeze(0).unsqueeze(2)
            key_states = _rotate(key_states)
            key_states = self.k_quantizer(key_states)
            key_states = _inv_rotate(key_states)
            key_states = key_states.to(kv_dtype)
            # Restore active positions to BF16 (not quantized at inference)
            if active_mask is not None:
                mask_4d = active_mask[:, None, :, None]  # [B, 1, T, 1]
                key_states = torch.where(mask_4d, key_pre_quant, key_states)
        if hasattr(self, 'v_quantizer') and self.v_quantizer is not None:
            val_pre_quant = value_states
            value_states = _rotate(value_states)
            value_states = self.v_quantizer(value_states)
            value_states = _inv_rotate(value_states)
            value_states = value_states.to(kv_dtype)
            if active_mask is not None:
                mask_4d = active_mask[:, None, :, None]
                value_states = torch.where(mask_4d, val_pre_quant, value_states)

        # Collect KV for distillation (post-quant for student, post-RoPE for teacher)
        if getattr(type(self), '_collect_kv', False):
            _store = getattr(type(self), '_kv_store', None)
            if _store is not None:
                _store[self.layer_idx] = (key_states, value_states)

        # Cache updates
        if block_past_key_values is not None:
            if len(block_past_key_values) <= self.layer_idx:
                cache_kwargs = {
                    "sin": sin, "cos": cos, "cache_position": cache_position,
                }
                key_states, value_states = block_past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]
                block_cache_key_states[
                    :, :,
                    replace_position:replace_position + key_states.shape[2]
                ] = key_states
                block_cache_value_states[
                    :, :,
                    replace_position:replace_position + value_states.shape[2]
                ] = value_states
                key_states = block_cache_key_states
                value_states = block_cache_value_states

        if past_key_value is not None:
            if update_past_key_values:
                cache_kwargs = {
                    "sin": sin, "cos": cos, "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            elif len(past_key_value) > self.layer_idx:
                key_states = torch.cat(
                    (past_key_value[self.layer_idx][0], key_states), dim=-2
                )
                value_states = torch.cat(
                    (past_key_value[self.layer_idx][1], value_states), dim=-2
                )

        # Attention computation
        if self.training:
            attn_output = _fused_flex_attention(
                query_states, key_states, value_states, mask=attention_mask
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attention_interface = _ALL_ATTENTION_FUNCTIONS["sdpa"]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                is_causal=False,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    attn_cls.forward = _patched_attention_forward
    print(
        f"[KV Quant] Attached {kv_quant_type} quantizers to "
        f"{len(attn_modules)} attention layers (post-RoPE, rotation={rotation}"
        f"{', bias_subtract' if k_biases is not None else ''})"
    )
