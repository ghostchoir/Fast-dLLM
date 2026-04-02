"""SmoothQuant-style partial gamma absorption for activation quantization.

Redistributes RMSNorm gamma magnitude from activations to weights.
For each RMSNorm -> Linear boundary:
    s_j = |gamma_j|^alpha
    gamma_new_j = gamma_j / s_j = sign(gamma_j) * |gamma_j|^(1-alpha)
    W_new[:, j] = W[:, j] * s_j

Mathematically equivalent (zero accuracy loss before quantization).
Must be called BEFORE convert_model_to_quant().
"""

import torch


@torch.no_grad()
def apply_smooth_scaling(model, alpha=0.5, min_gamma=1e-4):
    """Apply SmoothQuant partial gamma absorption in-place.

    Args:
        model: HuggingFace CausalLM model (Qwen2.5 architecture).
        alpha: Smoothing strength. 0 = no change, 1 = full absorption.
            0.5 = balanced (standard SmoothQuant).
            Higher values push more difficulty to weights.
        min_gamma: Minimum |gamma| clamp to avoid division issues.

    Returns:
        list of dicts with diagnostic stats per layer.
    """
    stats = []

    for layer_idx, layer in enumerate(model.model.layers):
        layer_stats = {}

        # --- Boundary 1: input_layernorm -> q/k/v_proj ---
        gamma1 = layer.input_layernorm.weight.data.float()
        orig_dtype1 = layer.input_layernorm.weight.dtype
        layer_stats["gamma1_range"] = (gamma1.min().item(), gamma1.max().item())
        layer_stats["gamma1_cv"] = (gamma1.std() / gamma1.abs().mean().clamp(min=1e-12)).item()

        s1 = gamma1.abs().clamp(min=min_gamma).pow(alpha)
        gamma1_new = gamma1 / s1

        layer_stats["gamma1_new_range"] = (gamma1_new.min().item(), gamma1_new.max().item())
        layer_stats["gamma1_new_cv"] = (gamma1_new.std() / gamma1_new.abs().mean().clamp(min=1e-12)).item()

        layer.input_layernorm.weight.data = gamma1_new.to(orig_dtype1)
        for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj,
                     layer.self_attn.v_proj]:
            proj.weight.data.mul_(s1.to(proj.weight.dtype))

        # --- Boundary 2: post_attention_layernorm -> gate/up_proj ---
        gamma2 = layer.post_attention_layernorm.weight.data.float()
        orig_dtype2 = layer.post_attention_layernorm.weight.dtype
        layer_stats["gamma2_range"] = (gamma2.min().item(), gamma2.max().item())
        layer_stats["gamma2_cv"] = (gamma2.std() / gamma2.abs().mean().clamp(min=1e-12)).item()

        s2 = gamma2.abs().clamp(min=min_gamma).pow(alpha)
        gamma2_new = gamma2 / s2

        layer_stats["gamma2_new_range"] = (gamma2_new.min().item(), gamma2_new.max().item())
        layer_stats["gamma2_new_cv"] = (gamma2_new.std() / gamma2_new.abs().mean().clamp(min=1e-12)).item()

        layer.post_attention_layernorm.weight.data = gamma2_new.to(orig_dtype2)
        for proj in [layer.mlp.gate_proj, layer.mlp.up_proj]:
            proj.weight.data.mul_(s2.to(proj.weight.dtype))

        stats.append(layer_stats)

    # Skip final norm -> lm_head (lm_head is not quantized)
    return stats
