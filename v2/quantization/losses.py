"""Knowledge distillation losses for QAT Phase 3.

Provides:
  - compute_kd_loss: JSD/FKL/RKL logit distillation on masked positions
  - compute_kv_distill_loss: Hidden-state MSE as KV cache distillation proxy
  - compute_post_quant_kv_loss: Direct MSE on post-quant KV vs teacher KV
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Divergence primitives
# ---------------------------------------------------------------------------

def _fkl_div(student_log_prob, teacher_prob, eps=1e-6):
    return F.kl_div(student_log_prob, teacher_prob + eps, reduction="none")


def _rkl_div(teacher_log_prob, student_prob, eps=1e-6):
    return F.kl_div(teacher_log_prob, student_prob + eps, reduction="none")


def _js_div(teacher_prob, student_prob, teacher_log_prob, student_log_prob, eps=1e-6):
    c_prob = 0.5 * teacher_prob + 0.5 * student_prob + eps
    kl = (0.5 * F.kl_div(student_log_prob, c_prob, reduction='none')
          + 0.5 * F.kl_div(teacher_log_prob, c_prob, reduction='none'))
    return torch.nan_to_num(kl, 0, 0, 0)


# ---------------------------------------------------------------------------
# Logit distillation loss
# ---------------------------------------------------------------------------

def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    loss_type: str = "jsd",
    token_weight_mode: str = "none",
    focal_gamma: float = 0.5,
) -> torch.Tensor:
    """Compute KD loss on masked positions only.

    Args:
        student_logits: [B, T, V] student logits.
        teacher_logits: [B, T, V] teacher logits (detached inside).
        mask: [B, T] float tensor, 1.0 at positions to distill (masked tokens).
        temperature: Softmax temperature.
        loss_type: 'jsd', 'fkl', 'rkl', 'ce', or 'mse'.
            'ce': cross-entropy with teacher hard labels (argmax).
            'mse': mean squared error on raw logits (no softmax).
        token_weight_mode: Per-token weighting strategy.
            'none': uniform (default). 'entropy': sqrt(teacher entropy).
            'focal': (kl/kl_mean)^gamma. 'confidence': teacher max prob.
        focal_gamma: Exponent for focal weighting (default 0.5).

    Returns:
        Scalar loss (mean over masked positions) * temperature^2.
    """
    t_logits = teacher_logits.detach() / temperature
    s_logits = student_logits / temperature

    if loss_type == "ce":
        # Cross-entropy with teacher hard labels (argmax)
        teacher_labels = teacher_logits.detach().argmax(dim=-1)  # [B, T]
        # Use unnormalized student logits / temperature
        per_token = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            teacher_labels.view(-1),
            reduction="none",
        ).view(s_logits.shape[0], s_logits.shape[1])  # [B, T]
    elif loss_type == "mse":
        # MSE on raw logits (no softmax)
        per_token = ((s_logits - t_logits) ** 2).mean(dim=-1)  # [B, T]
    elif loss_type == "fkl":
        teacher_probs = F.softmax(t_logits, dim=-1)
        student_log_probs = F.log_softmax(s_logits, dim=-1)
        per_token = _fkl_div(student_log_probs, teacher_probs).sum(dim=-1)
    elif loss_type == "rkl":
        teacher_log_probs = F.log_softmax(t_logits, dim=-1)
        student_probs = F.softmax(s_logits, dim=-1)
        per_token = _rkl_div(teacher_log_probs, student_probs).sum(dim=-1)
    elif loss_type == "jsd":
        teacher_probs = F.softmax(t_logits, dim=-1)
        teacher_log_probs = F.log_softmax(t_logits, dim=-1)
        student_probs = F.softmax(s_logits, dim=-1)
        student_log_probs = F.log_softmax(s_logits, dim=-1)
        per_token = _js_div(
            teacher_probs, student_probs,
            teacher_log_probs, student_log_probs,
        ).sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # --- Per-token weighting ---
    if token_weight_mode == "entropy":
        t_probs_w = F.softmax(t_logits, dim=-1)
        log_t_w = torch.log(t_probs_w.clamp(min=1e-8))
        entropy = -(t_probs_w * log_t_w).sum(dim=-1)  # [B, T]
        raw_w = torch.sqrt(entropy)
    elif token_weight_mode == "focal":
        pt_det = per_token.detach()
        kl_mean = (pt_det * mask).sum() / mask.sum().clamp(min=1.0)
        raw_w = (pt_det / kl_mean.clamp(min=1e-8)).clamp(min=0.01).pow(focal_gamma)
    elif token_weight_mode == "confidence":
        t_probs_w = F.softmax(t_logits, dim=-1)
        raw_w = t_probs_w.max(dim=-1).values  # [B, T]
    else:
        raw_w = None

    if raw_w is not None:
        # Normalize weights to mean=1 over masked positions
        w_masked = raw_w * mask
        w_mean = w_masked.sum() / mask.sum().clamp(min=1.0)
        weights = w_masked / w_mean.clamp(min=1e-8)
        masked_loss = (per_token * weights).sum()
    else:
        masked_loss = (per_token * mask).sum()

    num_masked = mask.sum().clamp(min=1.0)
    return (masked_loss / num_masked) * (temperature ** 2)


# ---------------------------------------------------------------------------
# KV cache distillation via hidden-state MSE
# ---------------------------------------------------------------------------

def compute_kv_distill_loss(
    student_hidden_states,
    teacher_hidden_states,
    cache_mask: torch.Tensor,
    layer_stride: int = 4,
) -> torch.Tensor:
    """MSE loss on hidden states at cached positions as KV cache distillation proxy.

    KV cache entries are linear projections of hidden states, so matching
    hidden states ensures KV values match. Computed on a subset of layers
    (every ``layer_stride``-th) for memory efficiency.

    Accepts hidden_states as either:
      - A tuple of [B, L, D] tensors (standard HF, one per layer), or
      - A single [B, L, D] tensor (custom models returning final hidden state).

    Args:
        student_hidden_states: Hidden states from the student model.
        teacher_hidden_states: Hidden states from the teacher model.
        cache_mask: [B, L] float tensor, 1.0 at positions whose KV is cached
            during inference (prefix blocks, completed sub-blocks, and suffix
            sub-blocks per DualCache).
        layer_stride: Only use every N-th layer (default 4).

    Returns:
        Scalar MSE loss averaged over selected layers and cached positions.
    """
    # Normalize to list-of-tensors regardless of input format
    if isinstance(student_hidden_states, torch.Tensor):
        student_layers = [student_hidden_states]
        teacher_layers = [teacher_hidden_states]
    else:
        student_layers = list(student_hidden_states)
        teacher_layers = list(teacher_hidden_states)

    B, L, D = student_layers[0].shape
    num_cached = cache_mask.sum().clamp(min=1.0)

    total_loss = torch.tensor(0.0, device=student_layers[0].device)
    num_layers = 0

    for i in range(0, len(student_layers), layer_stride):
        s = student_layers[i]       # [B, L, D]
        t = teacher_layers[i].detach()

        # MSE per token: [B, L]
        mse = ((s - t) ** 2).mean(dim=-1)
        masked_mse = (mse * cache_mask).sum() / num_cached
        total_loss = total_loss + masked_mse
        num_layers += 1

    return total_loss / max(num_layers, 1)


# ---------------------------------------------------------------------------
# Post-quant KV cache distillation (direct KV matching)
# ---------------------------------------------------------------------------

def compute_post_quant_kv_loss(
    student_kv: Dict[int, tuple],
    teacher_kv: Dict[int, tuple],
    cache_mask: torch.Tensor,
    k_biases: Optional[Dict[int, torch.Tensor]] = None,
    layer_stride: int = 1,
) -> torch.Tensor:
    """MSE loss between student's post-quant KV and teacher's unquantized KV.

    Directly minimizes the KV cache quantization error that affects inference.
    More targeted than hidden-state MSE (which is an indirect proxy).

    Args:
        student_kv: {layer_idx: (K, V)} collected from student forward.
            K,V are [B, num_kv_heads, T, head_dim], post-quantization.
        teacher_kv: {layer_idx: (K, V)} collected from teacher forward.
            K,V are [B, num_kv_heads, T, head_dim], post-RoPE (unquantized).
        cache_mask: [B, T] float tensor, 1.0 at cached positions.
        k_biases: {layer_idx: [num_kv_heads, head_dim]} bias tensors.
            If provided, subtract from teacher K for fair comparison
            (student K has bias subtracted before quantization).
        layer_stride: Only use every N-th layer (default 1 = all layers).

    Returns:
        Scalar MSE loss averaged over selected layers and cached positions.
    """
    mask_4d = cache_mask[:, None, :, None]  # [B, 1, T, 1]
    num_cached = cache_mask.sum().clamp(min=1.0)

    total_loss = torch.tensor(0.0, device=cache_mask.device)
    num_layers = 0

    for layer_idx in sorted(teacher_kv.keys()):
        if layer_idx % layer_stride != 0:
            continue
        if layer_idx not in student_kv:
            continue

        t_k, t_v = teacher_kv[layer_idx]
        s_k, s_v = student_kv[layer_idx]

        # Bias correction: student K has bias subtracted, teacher K doesn't
        if k_biases is not None and layer_idx in k_biases:
            bias = k_biases[layer_idx]  # [num_kv_heads, head_dim]
            t_k = t_k - bias.unsqueeze(0).unsqueeze(2)

        # MSE on cached positions only
        k_mse = ((s_k - t_k.detach()) ** 2 * mask_4d).sum() / num_cached
        v_mse = ((s_v - t_v.detach()) ** 2 * mask_4d).sum() / num_cached
        total_loss = total_loss + k_mse + v_mse
        num_layers += 1

    return total_loss / max(num_layers, 1)
