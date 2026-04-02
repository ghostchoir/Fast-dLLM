# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import itertools
import torch
import re
import datetime
import torch.distributed as dist
from pathlib import Path

# Set NCCL timeout to 2 days to prevent watchdog kills during slow generation
import os
os.environ.setdefault("NCCL_TIMEOUT", "172800000")
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=2))
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import time
import types
import generation_functions
from quantization.fake_quant import apply_fake_quantization, ALL_QUANT_SCHEMES
from quantization.kv_cache_quant import (
    calibrate_fp8_kv_scales, calibrate_nvfp4_kv_scales,
    calibrate_nvfp4_kv_scales_per_head,
    calibrate_nvfp4_kv_scales_attn_aware,
    calibrate_nvfp4_kv_scales_with_perm,
    calibrate_nvfp4_kv_scales_local_rot,
    calibrate_nvfp4_kv_scales_block_rot,
    calibrate_nvfp4_kv_scales_block_rot_clip,
    calibrate_nvfp4_kv_scales_block_rot_shift,
    calibrate_nvfp4_kv_scales_hybrid,
    calibrate_nvfp4_kv_scales_pre_shift_block_rot,
    calibrate_nvfp4_kv_scales_attn_loss,
    calibrate_nvfp4_kv_scales_bias_subtract,
    analyze_k_quantization_sensitivity,
    quantize_kv_cache, quantize_kv_cache_fp8_frozen, quantize_kv_cache_nvfp4_frozen,
    quantize_kv_cache_mixed_k,
)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("fast_dllm_v2")
class Fast_dLLM_v2EvalHarness(LM):
    def __init__(
        self,
        model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',
        device="cuda",
        show_speed=False,
        max_new_tokens=2048,
        batch_size=32,
        mask_id=151665,
        use_block_cache=False,
        small_block_size=8,
        bd_size=32,
        threshold=0.9,
        quant_scheme=None,
        kv_quant=False,
        kv_quant_mode=None,
        kv_bf16_window=0,
        kv_quant_variant='frozen',
        r4_block_size=0,
        act_outlier_channels=None,
        skip_quant_modules=None,
        calib_data=None,
        **kwargs,
    ):

        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
        self.model.eval()

        # Parse act_outlier_channels: "4:610.5:8729.26:2538" -> {layer: [channels]}
        # Uses '.' separator (not ',') because model_args uses ',' as key=value delimiter
        parsed_outlier_channels = None
        if act_outlier_channels:
            parsed_outlier_channels = {}
            for entry in act_outlier_channels.split('.'):
                layer_str, ch_str = entry.split(':')
                layer_idx = int(layer_str)
                parsed_outlier_channels.setdefault(layer_idx, []).append(int(ch_str))

        # Parse skip_quant_modules: "layers.4.mlp.down_proj|layers.5.mlp.down_proj" -> list
        parsed_skip_modules = None
        if skip_quant_modules:
            parsed_skip_modules = skip_quant_modules.split('|')

        r4_block_size = int(r4_block_size) if r4_block_size else 0
        if quant_scheme is not None:
            assert quant_scheme in ALL_QUANT_SCHEMES, \
                f"Unknown quant_scheme '{quant_scheme}'. Choose from {sorted(ALL_QUANT_SCHEMES)}"
            apply_fake_quantization(
                self.model, quant_scheme, model_path=model_path,
                r4_block_size=r4_block_size,
                act_outlier_channels=parsed_outlier_channels,
                skip_quant_modules=parsed_skip_modules,
            )

        self.model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, self.model)

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.show_speed = show_speed
        self.max_new_tokens = max_new_tokens
        self.batch_size = int(batch_size)
        self.mask_id = mask_id
        self.model_path = model_path
        self.use_block_cache = use_block_cache
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.bd_size = bd_size
        self.quant_scheme = quant_scheme
        self.calib_data = calib_data

        # KV cache quantization
        # kv_quant_mode can be set explicitly (none/int4/int8/fp8/nvfp4),
        # or derived from quant_scheme when kv_quant=True.
        _KV_MODE_MAP = {'fp-w8a8': 'fp8', 'nvfp-w4a4': 'nvfp4', 'w8a8': 'int8', 'w4g128': 'int4'}
        if isinstance(kv_quant, str):
            kv_quant = kv_quant.lower() in ('true', '1', 'yes')
        if kv_quant_mode is not None and kv_quant_mode != 'none':
            # Explicit kv_quant_mode overrides everything
            self.kv_quant_mode = kv_quant_mode
        elif kv_quant:
            # Legacy: derive from quant_scheme
            self.kv_quant_mode = _KV_MODE_MAP.get(quant_scheme)
        else:
            self.kv_quant_mode = None
        self.kv_bf16_window = int(kv_bf16_window)
        self.kv_quant_variant = kv_quant_variant
        self.fp8_kv_scales = None
        self.nvfp4_kv_scales = None
        self.fp8_k_only_scales = None   # FP8 scales for K in mixed mode
        self.k_bf16_layers = None       # Set of layer indices for mixed_layer
        self.k_bf16_heads = None        # Set of (layer, head) tuples for mixed_head
        self.k_sensitivity = None       # Sensitivity analysis results

        if self.kv_quant_mode in ('fp8', 'nvfp4'):
            if self.kv_quant_variant == 'dynamic':
                print(f"[KVQuant] Using dynamic per-token NVFP4 KV cache quantization")
            elif self.kv_quant_variant in ('k_only', 'v_only', 'no_hadamard'):
                self._calibrate_nvfp4(model_path)
            elif self.kv_quant_variant == 'per_head':
                self._calibrate_nvfp4_per_head(model_path)
            elif self.kv_quant_variant == 'k_fp8_v_fp4':
                # FP8 for K, FP4 for V: need both scale types
                self._calibrate_nvfp4(model_path)  # V scales
                self._calibrate_fp8_k(model_path)   # K scales
            elif self.kv_quant_variant.startswith('mixed_layer'):
                # mixed_layer_N: keep top-N most sensitive layers' K in BF16
                self._calibrate_nvfp4(model_path)
                self._setup_mixed_layer(model_path)
            elif self.kv_quant_variant.startswith('mixed_head'):
                # mixed_head_N: keep top-N most sensitive heads' K in BF16
                self._calibrate_nvfp4(model_path)
                self._setup_mixed_head(model_path)
            elif self.kv_quant_variant == 'attn_aware':
                self._calibrate_nvfp4_attn_aware(model_path)
            elif self.kv_quant_variant == 'perm_hadamard':
                self._calibrate_nvfp4_perm_hadamard(model_path)
            elif self.kv_quant_variant == 'local_rot':
                self._calibrate_nvfp4_local_rot(model_path)
            elif self.kv_quant_variant == 'block_rot':
                self._calibrate_nvfp4_block_rot(model_path)
            elif self.kv_quant_variant == 'block_rot_clip':
                self._calibrate_nvfp4_block_rot_clip(model_path)
            elif self.kv_quant_variant == 'block_rot_shift':
                self._calibrate_nvfp4_block_rot_shift(model_path, clip=False)
            elif self.kv_quant_variant == 'block_rot_shift_clip':
                self._calibrate_nvfp4_block_rot_shift(model_path, clip=True)
            elif self.kv_quant_variant == 'hybrid':
                self._calibrate_nvfp4_hybrid(model_path)
            elif self.kv_quant_variant == 'pre_shift_block_rot':
                self._calibrate_nvfp4_pre_shift_block_rot(model_path)
            elif self.kv_quant_variant == 'attn_loss':
                self._calibrate_nvfp4_attn_loss(model_path)
            elif self.kv_quant_variant in ('bias_subtract', 'bias_sub_no_rot', 'bias_sub_block_rot'):
                self._calibrate_nvfp4_bias_subtract(model_path)
            else:
                self._calibrate_nvfp4(model_path)
        elif self.kv_quant_mode:
            print(f"[KVQuant] Using dynamic {self.kv_quant_mode.upper()} KV cache quantization")

    def _calibrate_nvfp4(self, model_path):
        """Calibrate per-layer frozen NVFP4 scales (or load from checkpoint)."""
        loaded_scales = self._try_load_kv_scales(model_path)
        if loaded_scales is not None:
            self.nvfp4_kv_scales = loaded_scales
            print(f"[KVQuant] Loaded saved NVFP4 KV scales for {loaded_scales.num_layers} layers")
        else:
            calib_ids = self._get_calib_ids()
            self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales(self.model, calib_ids, block_size=self.bd_size)
            print(f"[KVQuant] Calibrated NVFP4 KV scales for {self.nvfp4_kv_scales.num_layers} layers")

    def _calibrate_nvfp4_per_head(self, model_path):
        """Calibrate per-head frozen NVFP4 scales."""
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_per_head(self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated per-head NVFP4 KV scales for {self.nvfp4_kv_scales.num_layers} layers")

    def _calibrate_nvfp4_attn_aware(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_attn_aware(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated attention-aware NVFP4 KV scales")

    def _calibrate_nvfp4_perm_hadamard(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_with_perm(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated permutation+Hadamard NVFP4 KV scales")

    def _calibrate_nvfp4_local_rot(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_local_rot(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated local rotation NVFP4 KV scales")

    def _calibrate_nvfp4_block_rot(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_block_rot(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated block rotation NVFP4 KV scales")

    def _calibrate_nvfp4_block_rot_clip(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_block_rot_clip(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated block rotation + clipping NVFP4 KV scales")

    def _calibrate_nvfp4_block_rot_shift(self, model_path, clip=False):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_block_rot_shift(
            self.model, calib_ids, block_size=self.bd_size, clip=clip)
        tag = "shift+clip" if clip else "shift"
        print(f"[KVQuant] Calibrated block rotation + {tag} NVFP4 KV scales")

    def _calibrate_nvfp4_hybrid(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_hybrid(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated hybrid channel-adaptive NVFP4 KV scales")

    def _calibrate_nvfp4_pre_shift_block_rot(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_pre_shift_block_rot(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated pre-shift + block rotation NVFP4 KV scales")

    def _calibrate_nvfp4_attn_loss(self, model_path):
        calib_ids = self._get_calib_ids()
        self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_attn_loss(
            self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated attention-loss block rotation NVFP4 KV scales")

    def _calibrate_nvfp4_bias_subtract(self, model_path):
        rot_map = {
            'bias_subtract': 'global',
            'bias_sub_no_rot': 'none',
            'bias_sub_block_rot': 'block',
        }
        rotation = rot_map.get(self.kv_quant_variant, 'global')

        # Try loading saved KV scales from checkpoint (matches training calibration)
        # Skip if user explicitly provides calib_data (wants custom calibration)
        loaded_scales = None if self.calib_data else self._try_load_kv_scales(model_path)
        if loaded_scales is not None:
            loaded_scales.rotation_mode = rotation
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            num_layers = loaded_scales.num_layers
            loaded_scales.k_biases = [None] * num_layers
            for i in range(num_layers):
                k_bias = raw_model.model.layers[i].self_attn.k_proj.bias.data.float()
                num_kv_heads = raw_model.config.num_key_value_heads
                head_dim = raw_model.config.hidden_size // raw_model.config.num_attention_heads
                loaded_scales.k_biases[i] = k_bias.reshape(num_kv_heads, head_dim)
            loaded_scales.calibrated = True
            self.nvfp4_kv_scales = loaded_scales
            print(f"[KVQuant] Loaded saved bias-subtract KV scales for "
                  f"{num_layers} layers (rotation={rotation})")
        else:
            calib_ids = self._get_calib_ids()
            self.nvfp4_kv_scales = calibrate_nvfp4_kv_scales_bias_subtract(
                self.model, calib_ids, block_size=self.bd_size, rotation=rotation)
            print(f"[KVQuant] Calibrated bias-subtract NVFP4 KV scales (rotation={rotation})")

        # Load KV LAC ratios if available (saved by QAT with --kv_lac)
        kv_lac_path = os.path.join(model_path, "learned_kv_lac.pt") if model_path else None
        if kv_lac_path and os.path.exists(kv_lac_path):
            kv_lac = torch.load(kv_lac_path, map_location="cpu", weights_only=True)
            self.nvfp4_kv_scales.kv_lac_ratios = kv_lac
            print(f"[KVQuant] Loaded {len(kv_lac)} KV LAC clip ratios")

    def _calibrate_fp8_k(self, model_path):
        """Calibrate FP8 scales for K cache only."""
        calib_ids = self._get_calib_ids()
        self.fp8_k_only_scales = calibrate_fp8_kv_scales(self.model, calib_ids, block_size=self.bd_size)
        print(f"[KVQuant] Calibrated FP8 K scales for {self.fp8_k_only_scales.num_layers} layers")

    def _run_sensitivity_analysis(self):
        """Run K quantization sensitivity analysis."""
        if self.k_sensitivity is None:
            calib_ids = self._get_calib_ids()
            self.k_sensitivity = analyze_k_quantization_sensitivity(
                self.model, calib_ids, block_size=self.bd_size
            )
        return self.k_sensitivity

    def _setup_mixed_layer(self, model_path):
        """Setup mixed-precision per-layer: keep top-N sensitive layers' K in BF16."""
        # Parse N from variant name: mixed_layer_7 => keep top 7 layers BF16
        parts = self.kv_quant_variant.split('_')
        n_bf16 = int(parts[2]) if len(parts) > 2 else 7  # default: 25% of 28
        sens = self._run_sensitivity_analysis()
        # Rank layers by cosine error (higher = more sensitive)
        layer_errors = list(enumerate(sens["per_layer_cos"]))
        layer_errors.sort(key=lambda x: x[1], reverse=True)
        self.k_bf16_layers = set(idx for idx, _ in layer_errors[:n_bf16])
        print(f"[KVQuant] Mixed-layer K: {n_bf16}/{sens['num_layers']} layers BF16 "
              f"(layers {sorted(self.k_bf16_layers)})")
        # Print sensitivity ranking
        for rank, (idx, err) in enumerate(layer_errors):
            marker = " <-- BF16" if idx in self.k_bf16_layers else ""
            print(f"  Layer {idx:2d}: cos_err={err:.6f}{marker}")

    def _setup_mixed_head(self, model_path):
        """Setup mixed-precision per-head: keep top-N sensitive heads' K in BF16."""
        # Parse N from variant name: mixed_head_14 => keep top 14 heads BF16
        parts = self.kv_quant_variant.split('_')
        n_bf16 = int(parts[2]) if len(parts) > 2 else 14  # default: 25% of 56 (28*2)
        sens = self._run_sensitivity_analysis()
        # Flatten per-head errors: list of (layer_idx, head_idx, cos_err)
        head_errors = []
        for layer_idx, head_errs in enumerate(sens["per_head_cos"]):
            for head_idx, err in enumerate(head_errs):
                head_errors.append((layer_idx, head_idx, err))
        head_errors.sort(key=lambda x: x[2], reverse=True)
        self.k_bf16_heads = set((l, h) for l, h, _ in head_errors[:n_bf16])
        print(f"[KVQuant] Mixed-head K: {n_bf16}/{len(head_errors)} heads BF16")
        for rank, (l, h, err) in enumerate(head_errors):
            marker = " <-- BF16" if (l, h) in self.k_bf16_heads else ""
            print(f"  Layer {l:2d} Head {h}: cos_err={err:.6f}{marker}")

    def _get_calib_ids(self):
        """Load calibration data from a local file or C4 dataset."""
        if self.calib_data and os.path.isfile(self.calib_data):
            with open(self.calib_data, 'r') as f:
                calib_text = f.read()
            calib_ids = self.tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.device)
            print(f"[KVQuant] Calibrating on {self.calib_data} ({calib_ids.shape[1]} tokens)...")
        else:
            from datasets import load_dataset
            dataset_name = self.calib_data or "allenai/c4"
            c4 = load_dataset(dataset_name, "en", split="validation", streaming=True)
            calib_text = " ".join(x["text"] for x in itertools.islice(c4, 128))
            calib_ids = self.tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.device)
            print(f"[KVQuant] Calibrating on {dataset_name} ({calib_ids.shape[1]} tokens)...")
        return calib_ids

    def _try_load_kv_scales(self, model_path):
        """Try to load KV quantizer global scales saved by distill_qat.py.

        The distillation checkpoint saves k_quantizer.global_scale and
        v_quantizer.global_scale buffers for each attention layer.  If found,
        build an NVFP4KVCacheScales (or FP8KVCacheScales) object from them
        instead of recalibrating from C4.
        """
        from quantization.kv_cache_quant import NVFP4KVCacheScales, FP8KVCacheScales
        from safetensors import safe_open
        import glob as glob_mod

        # Look for safetensors or pytorch_model files
        safetensor_files = sorted(glob_mod.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            return None

        # Scan for k_quantizer/v_quantizer keys
        k_scales = {}
        v_scales = {}
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    if "k_quantizer.global_scale" in key:
                        # Extract layer index from e.g. "model.layers.5.self_attn.k_quantizer.global_scale"
                        parts = key.split(".")
                        for i, p in enumerate(parts):
                            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_idx = int(parts[i + 1])
                                k_scales[layer_idx] = f.get_tensor(key)
                                break
                    elif "v_quantizer.global_scale" in key:
                        parts = key.split(".")
                        for i, p in enumerate(parts):
                            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_idx = int(parts[i + 1])
                                v_scales[layer_idx] = f.get_tensor(key)
                                break

        if not k_scales or not v_scales:
            return None

        num_layers = max(max(k_scales), max(v_scales)) + 1
        if len(k_scales) != num_layers or len(v_scales) != num_layers:
            print(f"[KVQuant] Warning: found partial KV scales ({len(k_scales)} K, {len(v_scales)} V, "
                  f"expected {num_layers}). Falling back to C4 calibration.")
            return None

        if self.kv_quant_mode == 'nvfp4':
            scales = NVFP4KVCacheScales(num_layers)
            for i in range(num_layers):
                scales.k_scales[i] = k_scales[i]
                scales.v_scales[i] = v_scales[i]
            scales.calibrated = True
            return scales
        elif self.kv_quant_mode == 'fp8':
            scales = FP8KVCacheScales(num_layers)
            for i in range(num_layers):
                scales.k_scales[i] = k_scales[i]
                scales.v_scales[i] = v_scales[i]
            scales.calibrated = True
            return scales

        return None

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_path
    
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt, tokenize=False)
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc


    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        batch[:, prompt_index.sum()] = self.mask_id

        batch = torch.cat([batch.to(self.device), torch.full((b, self.bd_size-batch.shape[1]%self.bd_size), self.mask_id, dtype=torch.long, device=self.device)], dim=1)
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id

        return batch

    @torch.no_grad()
    def get_logits(self, batch):
        logits = self.model(batch).logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []

        perturbed_seq = self._forward_process(seq.clone(), prompt_index)

        mask_indices = perturbed_seq == self.mask_id

        logits = self.get_logits(perturbed_seq)
        seq = torch.cat([seq.to(self.device), torch.full((seq.shape[0], self.bd_size-seq.shape[1]%self.bd_size), -100, dtype=torch.long, device=self.device)], dim=1)
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
        loss = loss.sum()
        loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)


    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out
    
    def generate_until(self, requests):
        output = [None] * len(requests)  # pre-allocate output list
        num_tokens = 0
        
        start_time = time.time()
        
        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))
        
        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []
        
        if current_batch:
            batched_requests.append(current_batch)

        for _, batch in enumerate(tqdm(batched_requests, desc="Generating...")):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []
            
            for orig_idx, req in batch:
                question = req.args[0]
                
                if req.task_name.startswith('minerva_math'):
                    question = question.replace("Solution:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                elif req.task_name.startswith('gsm8k'):
                    question = question.replace("Answer:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([input_ids, torch.full((1, max_len - input_ids.shape[1]), self.mask_id, dtype=torch.long, device=self.device)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            # Build KV cache quantization callback
            kv_quant_fn = None
            _win = self.kv_bf16_window
            _var = self.kv_quant_variant
            if self.kv_quant_mode == 'fp8' and self.fp8_kv_scales:
                kv_quant_fn = lambda past_kv, num_old_tokens: quantize_kv_cache_fp8_frozen(past_kv, self.fp8_kv_scales, num_old_tokens, bf16_window_size=_win)
            elif self.kv_quant_mode == 'nvfp4':
                if _var == 'k_fp8_v_fp4' or _var.startswith('mixed_layer') or _var.startswith('mixed_head'):
                    # Mixed-precision: use quantize_kv_cache_mixed_k
                    _scales = self.nvfp4_kv_scales
                    _fp8_k = self.fp8_k_only_scales
                    _k_mode = 'fp8' if _var == 'k_fp8_v_fp4' else ('mixed_layer' if _var.startswith('mixed_layer') else 'mixed_head')
                    _k_bf16_layers = self.k_bf16_layers
                    _k_bf16_heads = self.k_bf16_heads
                    kv_quant_fn = lambda past_kv, num_old_tokens: quantize_kv_cache_mixed_k(
                        past_kv, _scales, _fp8_k, num_old_tokens,
                        bf16_window_size=_win, k_mode=_k_mode,
                        k_bf16_layers=_k_bf16_layers, k_bf16_heads=_k_bf16_heads,
                    )
                else:
                    _scales = self.nvfp4_kv_scales  # may be None for 'dynamic' variant
                    kv_quant_fn = lambda past_kv, num_old_tokens: quantize_kv_cache_nvfp4_frozen(past_kv, _scales, num_old_tokens, bf16_window_size=_win, variant=_var)
            elif self.kv_quant_mode in ('int4', 'int8'):
                kv_quant_fn = lambda past_kv, num_old_tokens: quantize_kv_cache(past_kv, self.kv_quant_mode)

            with torch.no_grad():
                if self.accelerator is not None:
                    generated_ids = self.accelerator.unwrap_model(self.model).mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                        kv_quant_fn=kv_quant_fn,
                    )
                else:
                    generated_ids = self.model.mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                        kv_quant_fn=kv_quant_fn,
                    )
            
            # extract new generated tokens, and keep original index order
            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos]:], 
                    skip_special_tokens=True
                )
            
                # count token number
                if self.show_speed:
                    num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()
                
                # put result in the correct original index position
                output[orig_idx] = generated_answer

                print('=' * 20)
                print('question: ', req.args[0])
                print('answer: ', generated_answer)
                print('=' * 20, end='\n\n')
            
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
