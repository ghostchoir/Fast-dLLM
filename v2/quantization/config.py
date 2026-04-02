"""Predefined quantization configs for QAT.

Each config specifies format and bit-width for weights, activations, and KV cache
independently. Individual args can override any preset value.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

QUANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "W4": {
        "w_quant_type": "int",
        "w_bit": 4,
        "w_group_size": 128,
        "a_quant_type": "none",
        "a_bit": 8,
        "kv_quant_type": "none",
        "kv_bit": 8,
    },
    "W4A8": {
        "w_quant_type": "int",
        "w_bit": 4,
        "w_group_size": 128,
        "a_quant_type": "int",
        "a_bit": 8,
        "kv_quant_type": "int",
        "kv_bit": 8,
    },
    "W8A8": {
        "w_quant_type": "int",
        "w_bit": 8,
        "w_group_size": -1,
        "a_quant_type": "int",
        "a_bit": 8,
        "kv_quant_type": "int",
        "kv_bit": 8,
    },
    "FP8": {
        "w_quant_type": "fp8",
        "w_bit": 8,
        "w_group_size": -1,
        "a_quant_type": "fp8",
        "a_bit": 8,
        "kv_quant_type": "fp8",
        "kv_bit": 8,
    },
    "NVFP4": {
        "w_quant_type": "nvfp4",
        "w_bit": 4,
        "w_group_size": -1,
        "a_quant_type": "nvfp4",
        "a_bit": 4,
        "kv_quant_type": "nvfp4",
        "kv_bit": 4,
    },
}


def get_quant_config(
    preset: str = "W4",
    # Individual overrides
    w_quant_type: str = None,
    w_bit: int = None,
    w_group_size: int = None,
    a_quant_type: str = None,
    a_bit: int = None,
    kv_quant_type: str = None,
    kv_bit: int = None,
) -> Dict[str, Any]:
    """Get a quantization config from a preset with optional overrides.

    Args:
        preset: Name of preset config (W4, W4A8, W8A8, FP8, NVFP4).
        w_quant_type .. kv_bit: Individual overrides (None = use preset).

    Returns:
        Dict with keys: w_quant_type, w_bit, w_group_size,
                        a_quant_type, a_bit, kv_quant_type, kv_bit.
    """
    if preset not in QUANT_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from {sorted(QUANT_PRESETS)}"
        )

    config = dict(QUANT_PRESETS[preset])

    # Apply individual overrides
    overrides = {
        "w_quant_type": w_quant_type,
        "w_bit": w_bit,
        "w_group_size": w_group_size,
        "a_quant_type": a_quant_type,
        "a_bit": a_bit,
        "kv_quant_type": kv_quant_type,
        "kv_bit": kv_bit,
    }
    for key, val in overrides.items():
        if val is not None:
            config[key] = val

    return config


def add_quant_args(parser):
    """Add quantization arguments to an argument parser."""
    group = parser.add_argument_group("Quantization")
    group.add_argument("--quant_config", type=str, default="W4",
                        choices=sorted(QUANT_PRESETS),
                        help="Preset quantization config")
    group.add_argument("--w_quant_type", type=str, default=None,
                        choices=["int", "fp8", "nvfp4"],
                        help="Override weight quantizer type")
    group.add_argument("--w_bit", type=int, default=None,
                        help="Override weight bit-width")
    group.add_argument("--w_group_size", type=int, default=None,
                        help="Override weight group size (-1 for per-channel)")
    group.add_argument("--a_quant_type", type=str, default=None,
                        choices=["none", "int", "fp8", "nvfp4"],
                        help="Override activation quantizer type")
    group.add_argument("--a_bit", type=int, default=None,
                        help="Override activation bit-width")
    group.add_argument("--kv_quant_type", type=str, default=None,
                        choices=["none", "int", "fp8", "nvfp4"],
                        help="Override KV cache quantizer type")
    group.add_argument("--kv_bit", type=int, default=None,
                        help="Override KV cache bit-width")
    group.add_argument("--learnable_qp", action="store_true",
                        help="Use learnable quantization parameters (INT only)")
    return group
