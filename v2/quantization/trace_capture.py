"""Capture denoising traces from a frozen BF16 teacher model.

Follows the same algorithm as generation_functions.py:batch_sample but
records every intermediate state for distillation training in Phase 3.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


@dataclass
class DenoisingStep:
    """One snapshot of the denoising process."""
    full_input_ids: torch.Tensor     # [prefix_len + block_size] token ids
    mask_positions: torch.Tensor     # [block_size] bool, True where masked
    prefix_len: int                  # length of completed prefix
    sub_block_start: int             # active sub-block start within block
    sub_block_end: int               # active sub-block end within block
    target_ids: torch.Tensor         # [block_size] ground truth tokens
    unmask_this_step: torch.Tensor   # [block_size] bool, True at positions unmasked this step
    block_idx: int = -1              # which block (0..max_blocks-1) this step belongs to


def sample_with_top_p(logits, top_p=0.95, temperature=0.0):
    """Top-p sampling matching generation_functions.py."""
    if temperature == 0.0:
        probs = F.softmax(logits, dim=-1)
        tokens = logits.argmax(dim=-1)
        return tokens, probs

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond top-p threshold
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    tokens = torch.multinomial(sorted_probs.view(-1, sorted_probs.size(-1)), 1)
    tokens = tokens.view(*sorted_probs.shape[:-1])
    tokens = sorted_indices.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    return tokens, probs


@torch.no_grad()
def capture_denoising_traces(
    model,
    tokenizer,
    input_ids_list: List[torch.Tensor],
    block_size: int = 32,
    small_block_size: int = 8,
    max_blocks: int = 4,
    threshold: float = 0.95,
    top_p: float = 0.95,
    temperature: float = 0.0,
    mask_id: int = FAST_DLLM_MASK_ID,
    stop_token: int = FAST_DLLM_STOP_TOKEN,
) -> List[DenoisingStep]:
    """Capture denoising traces from a teacher model.

    For each prompt, runs the block diffusion generation algorithm and
    records a DenoisingStep at each denoising iteration.

    Args:
        model: The teacher model (frozen BF16).
        tokenizer: Tokenizer for the model.
        input_ids_list: List of [1, prompt_len] token id tensors.
        block_size: Tokens per block.
        small_block_size: Sub-block size.
        max_blocks: Maximum blocks to generate per prompt.
        threshold: Unmask confidence threshold.
        top_p: Top-p sampling parameter.
        temperature: Sampling temperature.
        mask_id: Mask token id.
        stop_token: Stop token id.

    Returns:
        List of DenoisingStep across all prompts and blocks.
    """
    all_steps = []
    num_small_blocks = block_size // small_block_size
    device = next(model.parameters()).device

    for prompt_idx, input_ids in enumerate(input_ids_list):
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Prefill prompt
        prefill_len = (input_ids.shape[1] // block_size) * block_size
        past_key_values = None

        if prefill_len > 0:
            output = model.forward(
                input_ids=input_ids[:, :prefill_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            past_key_values = output.past_key_values
            if input_ids.shape[1] % block_size == 0:
                next_token = output.logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        finished = False

        for block_idx in range(max_blocks):
            if finished:
                break

            # Initialize block with mask tokens
            prompt_length = input_ids.shape[1]
            pad_len = block_size - (prompt_length % block_size)
            x_mask = mask_id * torch.ones(
                (1, pad_len), device=device, dtype=torch.long,
            )
            x_t = torch.cat([input_ids, x_mask], dim=1)

            # Ground truth: we don't have it, so store mask_id as placeholder
            # (during actual distillation, the teacher logits are the target)
            target_ids = x_t[0, -block_size:].clone()

            for sb_idx in range(num_small_blocks):
                sb_start = sb_idx * small_block_size
                sb_end = sb_start + small_block_size

                start = -block_size + sb_start
                end = None if block_size == sb_end else -block_size + sb_end

                while True:
                    mask_idx = (x_t[:, -block_size:] == mask_id)
                    if mask_idx[:, sb_start:sb_end].sum() == 0:
                        break

                    # Forward pass on current block
                    logits = model.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=False,
                    ).logits

                    # Apply token shift
                    logits = torch.cat(
                        [logits[:, :1, :], logits[:, :-1, :]], dim=1,
                    )
                    logits = logits[:, sb_start:sb_end]

                    # Sample and compute unmask decision
                    x_1, p_1t = sample_with_top_p(
                        logits, top_p=top_p, temperature=temperature,
                    )
                    x1_p = torch.gather(
                        p_1t, dim=-1, index=x_1.unsqueeze(-1),
                    ).squeeze(-1)
                    x1_p = torch.where(
                        mask_idx[:, sb_start:sb_end], x1_p, -torch.inf,
                    )

                    unmask_idx = (x1_p > threshold)
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[
                        torch.arange(x_1.shape[0]), max_prob_idx
                    ] = True
                    unmask_idx = unmask_idx & mask_idx[:, sb_start:sb_end]

                    # Map sub-block unmask to block-level
                    unmask_block = torch.zeros(block_size, dtype=torch.bool,
                                               device=device)
                    unmask_block[sb_start:sb_end] = unmask_idx[0]

                    # Record step (before applying unmasking)
                    step = DenoisingStep(
                        full_input_ids=x_t[0].cpu().clone(),
                        mask_positions=mask_idx[0].cpu().clone(),
                        prefix_len=prompt_length,
                        sub_block_start=sb_start,
                        sub_block_end=sb_end,
                        target_ids=target_ids.cpu().clone(),
                        unmask_this_step=unmask_block.cpu().clone(),
                        block_idx=block_idx,
                    )
                    all_steps.append(step)

                    # Apply unmasking
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                    # Check for stop token
                    if (x_1 == stop_token).any() and unmask_idx.any():
                        finished = True

            # Completed block: update KV cache
            mask_idx = (x_t[:, -block_size:] == mask_id)
            if mask_idx.sum() == 0:
                output = model.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                past_key_values = output.past_key_values
                next_token = output.logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)

            input_ids = x_t

        if (prompt_idx + 1) % 100 == 0:
            print(f"  Captured {len(all_steps)} steps from {prompt_idx + 1} prompts")

    return all_steps
