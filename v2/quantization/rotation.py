"""R1+R2 rotation absorption for SpinQuant-style weight rotation.

R1 (residual stream rotation): Hadamard rotation at every RMSNorm boundary.
    Absorbs RMSNorm γ into consuming weights, then applies rotation matrix.
    Uses hadamard_transform_12N for hidden_size = 12 × power_of_2.

R2 (per-head V rotation): Hadamard rotation on V values per attention head.
    Absorbed into v_proj (output side) and o_proj (input side).
    Uses standard hadamard_transform for head_dim (power of 2).

Both rotations are absorbed into weights → zero inference overhead.
Must be called BEFORE convert_model_to_quant().
"""

import math

import torch
import torch.nn as nn

from quantization.hadamard import hadamard_transform, hadamard_transform_12N


@torch.no_grad()
def absorb_r1_r2(model, apply_r1=True, apply_r2=True):
    """One-time absorption of R1+R2 rotations into model weights.

    Modifies weights in-place. Sets all RMSNorm γ=1.
    Unties lm_head from embed_tokens if R1 is applied.

    Args:
        model: HuggingFace CausalLM model (Qwen2.5 architecture).
        apply_r1: Apply R1 (residual stream rotation via H_12N).
        apply_r2: Apply R2 (per-head V rotation via H_128).
    """
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_size // num_heads

    r1_scale = 1.0 / math.sqrt(hidden_size)
    r2_scale = 1.0 / math.sqrt(head_dim)

    def rotate_last_dim_r1(W):
        """W @ R1: Hadamard rotation on last dim (input features = hidden_size)."""
        return hadamard_transform_12N(
            W.float(), scale=r1_scale,
        ).to(W.dtype)

    def rotate_first_dim_r1(W):
        """R1 @ W: Hadamard rotation on first dim (output features = hidden_size).

        Since R1 is symmetric: R1 @ W = (W^T @ R1)^T
        """
        return hadamard_transform_12N(
            W.T.contiguous().float(), scale=r1_scale,
        ).T.contiguous().to(W.dtype)

    if apply_r1:
        # Untie lm_head from embed_tokens (they need different transforms)
        if model.config.tie_word_embeddings:
            model.config.tie_word_embeddings = False
            lm_head_w = model.lm_head.weight.data.clone()
            model.lm_head.weight = nn.Parameter(lm_head_w)

        for layer in model.model.layers:
            gamma1 = layer.input_layernorm.weight.data.clone()
            gamma2 = layer.post_attention_layernorm.weight.data.clone()

            # --- Consuming weights: absorb γ then apply R1 ---

            # Attention: absorb γ1 into q/k/v_proj
            for proj in [layer.self_attn.q_proj,
                         layer.self_attn.k_proj,
                         layer.self_attn.v_proj]:
                proj.weight.data.mul_(gamma1)  # W * γ[None,:] broadcast
                proj.weight.data = rotate_last_dim_r1(proj.weight.data)
                # bias unchanged (added after matmul, in output space)

            # MLP: absorb γ2 into gate/up_proj
            for proj in [layer.mlp.gate_proj, layer.mlp.up_proj]:
                proj.weight.data.mul_(gamma2)
                proj.weight.data = rotate_last_dim_r1(proj.weight.data)

            # --- Producing weights: apply R1 to output ---

            # o_proj: weight + bias
            layer.self_attn.o_proj.weight.data = rotate_first_dim_r1(
                layer.self_attn.o_proj.weight.data)
            if layer.self_attn.o_proj.bias is not None:
                b = layer.self_attn.o_proj.bias.data.float()
                layer.self_attn.o_proj.bias.data = hadamard_transform_12N(
                    b.unsqueeze(0), scale=r1_scale,
                ).squeeze(0).to(layer.self_attn.o_proj.bias.dtype)

            # down_proj: weight only (no bias in Qwen MLP)
            layer.mlp.down_proj.weight.data = rotate_first_dim_r1(
                layer.mlp.down_proj.weight.data)

            # Set γ = 1 (RMSNorm now just divides by RMS)
            layer.input_layernorm.weight.data.fill_(1.0)
            layer.post_attention_layernorm.weight.data.fill_(1.0)

        # Final norm → lm_head: absorb γ_final then R1
        gamma_final = model.model.norm.weight.data.clone()
        model.lm_head.weight.data.mul_(gamma_final)
        model.lm_head.weight.data = rotate_last_dim_r1(model.lm_head.weight.data)
        model.model.norm.weight.data.fill_(1.0)

        # Embed tokens: rotate each embedding vector
        model.model.embed_tokens.weight.data = rotate_last_dim_r1(
            model.model.embed_tokens.weight.data)

    if apply_r2:
        for layer in model.model.layers:
            attn = layer.self_attn

            # v_proj: R2 on output side per KV head
            # Weight shape: (num_kv_heads * head_dim, in_features)
            vw = attn.v_proj.weight.data
            vw_heads = vw.reshape(num_kv_heads, head_dim, -1).float()
            # R2 @ W_h per head: transpose to (nkv, in, hd), hadamard on last dim
            vw_heads = hadamard_transform(
                vw_heads.transpose(1, 2), scale=r2_scale,
            ).transpose(1, 2)
            attn.v_proj.weight.data = vw_heads.reshape_as(vw).to(vw.dtype)

            # v_proj bias: R2 @ b per head
            if attn.v_proj.bias is not None:
                vb = attn.v_proj.bias.data
                vb_heads = vb.reshape(num_kv_heads, head_dim).float()
                vb_heads = hadamard_transform(vb_heads, scale=r2_scale)
                attn.v_proj.bias.data = vb_heads.reshape_as(vb).to(vb.dtype)

            # o_proj: R2 on input side per Q head (12 head blocks of 128)
            # Weight shape: (hidden_size, num_heads * head_dim)
            ow = attn.o_proj.weight.data
            ow_blocks = ow.reshape(ow.shape[0], num_heads, head_dim).float()
            ow_blocks = hadamard_transform(ow_blocks, scale=r2_scale)
            attn.o_proj.weight.data = ow_blocks.reshape_as(ow).to(ow.dtype)
            # o_proj bias: NOT affected by R2 (only by R1)
