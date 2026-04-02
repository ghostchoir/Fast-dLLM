"""AWQ-style scale and clip search adapted for NVFP4 W4A4 quantization.

Used as a PTQ pre-processing step before QAT distillation.
Finds per-channel scales and per-block clip values that minimize
layer-wise reconstruction MSE under NVFP4 quantization.

Algorithm adapted from: https://github.com/mit-han-lab/llm-awq
"""

import argparse
import gc
import os
import sys
import functools

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantization.quantizer import pseudo_nvfp4_quantize_tensor


# ---------------------------------------------------------------------------
# NVFP4 fake quantization wrappers
# ---------------------------------------------------------------------------

def nvfp4_quant_weight(w):
    """NVFP4 fake quant for a weight tensor (per-tensor global scale)."""
    return pseudo_nvfp4_quantize_tensor(
        w.float(), nvfp4_block_size=16, per_tensor_global=True,
    ).to(w.dtype)


def nvfp4_quant_activation(x):
    """NVFP4 fake quant for an activation tensor (per-tensor global scale)."""
    return pseudo_nvfp4_quantize_tensor(
        x.float(), nvfp4_block_size=16, per_tensor_global=True,
    ).to(x.dtype)


# ---------------------------------------------------------------------------
# Scale absorption helpers (from AWQ, format-agnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """Absorb per-channel scales: divide LayerNorm weights, multiply Linear inputs."""
    # ln.weight /= scales
    ln.weight.div_(scales.to(ln.weight.device))
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales.to(ln.bias.device))
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
        if fc.bias is not None:
            # bias is added after matmul, not affected by input scale
            pass


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """Absorb per-channel scales between two Linear layers.

    fc1 output channels divided by scales, fc2 input channels multiplied.
    """
    assert fc1.out_features == fc2.in_features
    # fc1.weight[out, in] -> divide output dim
    fc1.weight.div_(scales.view(-1, 1).to(fc1.weight.device))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1).to(fc1.bias.device))
    # fc2.weight[out, in] -> multiply input dim
    fc2.weight.mul_(scales.view(1, -1).to(fc2.weight.device))


# ---------------------------------------------------------------------------
# Activation statistics collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_act_scale(x):
    """Per-channel mean absolute activation magnitude."""
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def collect_layer_inputs(model, tokenizer, dataset_name="allenai/c4",
                         n_samples=128, seqlen=512):
    """Collect input activations for each transformer layer using hooks.

    Returns: list of tensors, one per layer, shape [n_samples * seqlen, hidden_dim].
    """
    device = next(model.parameters()).device
    model.eval()

    # Prepare calibration tokens
    dataset = load_dataset(dataset_name, "en", split="train", streaming=True)
    texts = []
    for sample in dataset:
        texts.append(sample["text"])
        if len(texts) >= n_samples:
            break

    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=seqlen,
    )
    input_ids = enc["input_ids"][:n_samples].to(device)

    # Collect inputs to each layer using hooks
    num_layers = len(model.model.layers)
    layer_inputs = [[] for _ in range(num_layers)]
    layer_kwargs = [None] * num_layers
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            layer_inputs[layer_idx].append(hidden.detach().cpu())
            if layer_kwargs[layer_idx] is None:
                # Capture kwargs for block forward (position_embeddings, etc.)
                kw = {}
                if "position_embeddings" in kwargs:
                    kw["position_embeddings"] = tuple(
                        t.detach().cpu() for t in kwargs["position_embeddings"]
                    )
                if "attention_mask" in kwargs:
                    kw["attention_mask"] = kwargs["attention_mask"].detach().cpu()
                layer_kwargs[layer_idx] = kw
            return None
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=True))

    # Run calibration forward passes in batches
    batch_size = min(4, n_samples)
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size]
            try:
                model(input_ids=batch)
            except Exception:
                pass  # Some layers may error; we have what we need

    for h in hooks:
        h.remove()

    # Concatenate
    result = []
    for i in range(num_layers):
        if layer_inputs[i]:
            cat = torch.cat(layer_inputs[i], dim=0)  # [batch*seqlen, hidden]
            result.append(cat)
        else:
            result.append(None)

    return result, layer_kwargs


@torch.no_grad()
def collect_linear_input_features(model, tokenizer, dataset_name="allenai/c4",
                                  n_samples=128, seqlen=512):
    """Collect input features for each Linear module for clip search.

    Returns: dict mapping module_name -> Tensor [n_tokens, in_features].
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = load_dataset(dataset_name, "en", split="train", streaming=True)
    texts = []
    for sample in dataset:
        texts.append(sample["text"])
        if len(texts) >= n_samples:
            break

    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=seqlen,
    )
    input_ids = enc["input_ids"][:n_samples].to(device)

    features = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, args):
            x = args[0].detach().cpu().reshape(-1, args[0].shape[-1])
            if name not in features:
                features[name] = [x]
            else:
                features[name].append(x)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    batch_size = min(4, n_samples)
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size]
            try:
                model(input_ids=batch)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    # Concatenate and subsample to manageable size
    max_tokens = 512
    result = {}
    for name, feat_list in features.items():
        cat = torch.cat(feat_list, dim=0)
        if cat.shape[0] > max_tokens:
            idx = torch.randperm(cat.shape[0])[:max_tokens]
            cat = cat[idx]
        result[name] = cat

    return result


# ---------------------------------------------------------------------------
# Scale search (AWQ-style, adapted for NVFP4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _capture_fc_output(block, fc_module, x, kwargs):
    """Capture a Linear module's output activation during a block forward pass."""
    device = next(block.parameters()).device
    captured = []

    def hook(mod, inp, out):
        captured.append(out.detach())

    h = fc_module.register_forward_hook(hook)
    kw = {k: v.to(device) if isinstance(v, torch.Tensor) else
          tuple(t.to(device) for t in v) if isinstance(v, tuple) else v
          for k, v in kwargs.items()}
    block(x.to(device), **kw)
    h.remove()
    return captured[0] if captured else None


@torch.no_grad()
def search_scale_for_boundary(block, prev_op, linears, x, kwargs,
                              n_grid=20, w4a4=False, act_scale=None):
    """Grid search for best per-channel scaling at one boundary.

    Args:
        block: The transformer layer module (for block-level MSE).
        prev_op: The preceding LayerNorm or Linear module.
        linears: List of Linear modules whose input channels to scale.
        x: Input hidden states to the block [batch, seq, hidden].
        kwargs: Block forward kwargs (position_embeddings, attention_mask, etc.).
        n_grid: Number of grid points for ratio search.
        w4a4: If True, also NVFP4-quantize activations in forward pass.
        act_scale: Pre-computed per-channel activation scale. If None, computed from x.
                   For fc→fc boundaries, pass the intermediate activation scale.

    Returns:
        best_scales: Tensor of per-channel scaling factors.
        best_error: Best MSE achieved.
    """
    device = next(block.parameters()).device
    x = x.to(device)
    kw = {k: v.to(device) if isinstance(v, torch.Tensor) else
          tuple(t.to(device) for t in v) if isinstance(v, tuple) else v
          for k, v in kwargs.items()}

    # Reference output (BF16, no quantization)
    org_out = block(x, **kw)
    if isinstance(org_out, tuple):
        org_out = org_out[0]

    x_max = act_scale.to(device) if act_scale is not None else get_act_scale(x)
    best_error = float("inf")
    best_scales = None
    org_sd = {k: v.clone() for k, v in block.state_dict().items()}

    for ratio_idx in range(n_grid):
        ratio = ratio_idx / n_grid
        if ratio_idx == 0:
            # ratio=0 means no scaling — test as baseline
            scales = torch.ones_like(x_max)
        else:
            scales = x_max.pow(ratio).clamp(min=1e-4)
            scales = scales / (scales.max() * scales.min()).sqrt()

        # Apply scales: absorb into prev_op and linears
        is_norm = hasattr(prev_op, "weight") and prev_op.weight.dim() == 1
        if is_norm:
            # RMSNorm or LayerNorm: 1D weight
            prev_op.weight.data.div_(scales.to(prev_op.weight.device))
            if hasattr(prev_op, "bias") and prev_op.bias is not None:
                prev_op.bias.data.div_(scales.to(prev_op.bias.device))
        else:
            # Linear (fc-to-fc): divide output channels
            prev_op.weight.data.div_(scales.view(-1, 1).to(prev_op.weight.device))
            if prev_op.bias is not None:
                prev_op.bias.data.div_(scales.view(-1).to(prev_op.bias.device))

        for fc in linears:
            fc.weight.data.mul_(scales.view(1, -1).to(fc.weight.device))

        # NVFP4 fake-quant all Linear weights in the block
        orig_weights = {}
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                orig_weights[name] = mod.weight.data.clone()
                mod.weight.data = nvfp4_quant_weight(mod.weight.data)

        # Forward pass (optionally with activation quantization)
        if w4a4:
            x_q = nvfp4_quant_activation(x)
            out = block(x_q, **kw)
        else:
            out = block(x, **kw)

        if isinstance(out, tuple):
            out = out[0]
        loss = (org_out - out).float().pow(2).mean().item()

        if loss < best_error:
            best_error = loss
            best_scales = scales.clone()

        # Restore block state
        block.load_state_dict(org_sd)

    return best_scales, best_error


# ---------------------------------------------------------------------------
# Clip search (AWQ-style, adapted for NVFP4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_layer_nvfp4(w, input_feat, n_grid=20, max_shrink=0.5, device="cuda"):
    """Per-NVFP4-block clip search for a single Linear layer.

    Args:
        w: Weight tensor [out_features, in_features].
        input_feat: Input activations [n_tokens, in_features].
        n_grid: Grid points for shrinkage search.
        max_shrink: Maximum fraction to shrink (0.5 = test up to 50% clip).
        device: Device to run computation on.

    Returns:
        best_max_val: Tensor [out_features, n_blocks, 1] of per-block clip values (CPU).
    """
    assert w.dim() == 2
    block_size = 16  # NVFP4 block

    # Reshape into blocks and move to GPU
    # w: [out, in] -> [out, 1, n_blocks, block_size]
    # input_feat: [n_tokens, in] -> [1, n_tokens, n_blocks, block_size]
    n_blocks = w.shape[1] // block_size
    w = w.reshape(w.shape[0], 1, n_blocks, block_size).to(device)
    input_feat = input_feat.reshape(1, input_feat.shape[0], n_blocks, block_size).to(device)

    # Process output channels in batches to avoid OOM
    oc_batch = 256 if w.shape[0] % 256 == 0 else (64 if w.shape[0] % 64 == 0 else w.shape[0])
    best_max_val_all = []

    for i_b in range(0, w.shape[0], oc_batch):
        w_batch = w[i_b : i_b + oc_batch]  # [batch, 1, n_blocks, 16]
        org_max_val = w_batch.abs().amax(dim=-1, keepdim=True)  # [batch, 1, n_blocks, 1]
        best_max_val = org_max_val.clone()
        min_errs = torch.full_like(org_max_val, float("inf"))

        # Baseline output (unquantized)
        org_out = (input_feat * w_batch).sum(dim=-1)  # [batch, n_tokens, n_blocks]

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            cur_w = torch.clamp(w_batch, -max_val, max_val)

            # NVFP4 quantize each block
            cur_w_flat = cur_w.reshape(-1, block_size).float()
            q_w_flat = pseudo_nvfp4_quantize_tensor(
                cur_w_flat, nvfp4_block_size=block_size, per_tensor_global=False,
            )
            q_w = q_w_flat.reshape(cur_w.shape).to(cur_w.dtype)

            cur_out = (input_feat * q_w).sum(dim=-1)  # [batch, n_tokens, n_blocks]
            # Per-block MSE averaged over tokens -> [batch, 1, n_blocks, 1]
            err = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True).unsqueeze(-1)

            improved = err < min_errs
            min_errs[improved] = err[improved]
            best_max_val[improved] = max_val[improved]

        best_max_val_all.append(best_max_val.cpu())

    # [out_features, 1, n_blocks, 1] -> [out_features, n_blocks, 1]
    return torch.cat(best_max_val_all, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# Main AWQ-NVFP4 pipeline
# ---------------------------------------------------------------------------

def run_awq_nvfp4(model, tokenizer, n_samples=128, seqlen=512, n_grid=20,
                  w4a4=False, dataset_name="allenai/c4", clip_only=False):
    """Run AWQ-style scale search + clip search with NVFP4 quantization.

    1. Collect calibration activations
    2. Per-layer scale search (grid over 20 ratios) → absorb scales  [skip if clip_only]
    3. Per-layer clip search (grid over 10 thresholds) → return clip values

    Args:
        model: The model (will be modified in-place with absorbed scales).
        tokenizer: Tokenizer for calibration data.
        n_samples: Number of calibration samples.
        seqlen: Sequence length for calibration.
        n_grid: Grid points for search.
        w4a4: If True, include activation quantization in scale search MSE.
        dataset_name: HuggingFace dataset for calibration.
        clip_only: If True, skip scale search and only run clip search on original weights.

    Returns:
        clip_values: Dict[module_name, Tensor] of per-block clip values.
        scale_log: List of (boundary_name, best_ratio, best_error) tuples.
    """
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    scale_log = []

    if not clip_only:
        print(f"[AWQ-NVFP4] Collecting calibration data ({n_samples} samples, seqlen {seqlen})...")
        layer_inputs, layer_kwargs = collect_layer_inputs(
            model, tokenizer, dataset_name=dataset_name,
            n_samples=n_samples, seqlen=seqlen,
        )

        # -------------------------------------------------------------------
        # Phase 1: Scale search & absorption
        # -------------------------------------------------------------------
        print(f"\n[AWQ-NVFP4] Phase 1: Scale search ({n_grid} grid points, w4a4={w4a4})")

        for layer_idx in tqdm(range(num_layers), desc="Scale search"):
            layer = model.model.layers[layer_idx]
            x = layer_inputs[layer_idx]
            if x is None:
                print(f"  Layer {layer_idx}: no calibration data, skipping")
                continue

            kw = layer_kwargs[layer_idx] if layer_kwargs[layer_idx] else {}

            # Boundary 1: input_layernorm -> [q_proj, k_proj, v_proj]
            scales, err = search_scale_for_boundary(
                block=layer,
                prev_op=layer.input_layernorm,
                linears=[layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                x=x, kwargs=kw, n_grid=n_grid, w4a4=w4a4,
            )
            scale_ln_fcs(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                scales.to(device),
            )
            scale_log.append((f"L{layer_idx}.ln1->qkv", err))

            # Boundary 2: v_proj -> o_proj — SKIP for GQA (dim mismatch)

            # Boundary 3: post_attention_layernorm -> [gate_proj, up_proj]
            scales, err = search_scale_for_boundary(
                block=layer,
                prev_op=layer.post_attention_layernorm,
                linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
                x=x, kwargs=kw, n_grid=n_grid, w4a4=w4a4,
            )
            scale_ln_fcs(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj],
                scales.to(device),
            )
            scale_log.append((f"L{layer_idx}.ln2->gate,up", err))

            # Boundary 4: up_proj -> down_proj
            # Need intermediate activation scale (dim=8960), not block input (dim=1536)
            up_out = _capture_fc_output(layer, layer.mlp.up_proj, x, kw)
            up_act_scale = get_act_scale(up_out) if up_out is not None else None
            del up_out
            scales, err = search_scale_for_boundary(
                block=layer,
                prev_op=layer.mlp.up_proj,
                linears=[layer.mlp.down_proj],
                x=x, kwargs=kw, n_grid=n_grid, w4a4=w4a4,
                act_scale=up_act_scale,
            )
            scale_fc_fc(
                layer.mlp.up_proj, layer.mlp.down_proj,
                scales.to(device),
            )
            scale_log.append((f"L{layer_idx}.up->down", err))

            # Update activations for next layer (re-run with absorbed scales)
            if layer_idx + 1 < num_layers:
                x_dev = x.to(device)
                kw_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else
                          tuple(t.to(device) for t in v) if isinstance(v, tuple) else v
                          for k, v in kw.items()}
                out = layer(x_dev, **kw_dev)
                if isinstance(out, tuple):
                    out = out[0]
                layer_inputs[layer_idx + 1] = out.detach().cpu()
                # Free old input
                layer_inputs[layer_idx] = None

        print(f"  Scale search complete. {len(scale_log)} boundaries processed.")
    else:
        print(f"[AWQ-NVFP4] Clip-only mode: skipping scale search")

    # -----------------------------------------------------------------------
    # Phase 2: Clip search
    # -----------------------------------------------------------------------
    print(f"\n[AWQ-NVFP4] Phase 2: Clip search ({n_grid} grid points)")
    print("  Collecting post-scale input features...")
    linear_feats = collect_linear_input_features(
        model, tokenizer, dataset_name=dataset_name,
        n_samples=n_samples, seqlen=seqlen,
    )

    clip_values = {}
    for name, module in tqdm(list(model.named_modules()), desc="Clip search"):
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        if name not in linear_feats:
            continue

        w = module.weight.data.clone().cpu()
        feat = linear_feats[name]

        # Skip if dimensions aren't NVFP4-block-aligned
        if w.shape[1] % 16 != 0:
            continue

        best_max_val = clip_layer_nvfp4(w, feat, n_grid=n_grid)
        clip_values[name] = best_max_val.cpu()

    print(f"  Clip search complete. {len(clip_values)} layers processed.")

    # Clean up
    if not clip_only:
        del layer_inputs
    del linear_feats
    gc.collect()
    torch.cuda.empty_cache()

    return clip_values, scale_log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AWQ-NVFP4 PTQ preprocessing")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save scaled model + clip values")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=512,
                        help="Calibration sequence length")
    parser.add_argument("--n_grid", type=int, default=20,
                        help="Grid points for scale/clip search")
    parser.add_argument("--w4a4", action="store_true",
                        help="Include activation NVFP4 in scale search MSE")
    parser.add_argument("--dataset", type=str, default="allenai/c4",
                        help="Calibration dataset")
    parser.add_argument("--clip_only", action="store_true",
                        help="Skip scale search, only run clip search on original weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[AWQ-NVFP4] Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    clip_values, scale_log = run_awq_nvfp4(
        model, tokenizer,
        n_samples=args.n_samples, seqlen=args.seqlen,
        n_grid=args.n_grid, w4a4=args.w4a4,
        dataset_name=args.dataset,
        clip_only=args.clip_only,
    )

    if not args.clip_only:
        # Save scaled model
        print(f"\n[AWQ-NVFP4] Saving scaled model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Save clip values
    clip_path = os.path.join(args.output_dir, "awq_clip.pt")
    torch.save(clip_values, clip_path)
    print(f"  Saved {len(clip_values)} clip configs to {clip_path}")

    if not args.clip_only:
        # Save scale log for analysis
        log_path = os.path.join(args.output_dir, "awq_scale_log.pt")
        torch.save(scale_log, log_path)
        print(f"  Saved scale search log to {log_path}")

    # Print summary
    print(f"\n[AWQ-NVFP4] Summary:")
    print(f"  Model: {args.model_path}")
    print(f"  Mode: {'clip-only' if args.clip_only else 'scale+clip'}")
    print(f"  Calibration: {args.n_samples} samples, seqlen {args.seqlen}")
    print(f"  Grid: {args.n_grid} points, w4a4={args.w4a4}")
    print(f"  Scale boundaries: {len(scale_log)}")
    print(f"  Clip layers: {len(clip_values)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
