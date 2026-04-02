import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Inlined from rlm_quant/utils.py to avoid external dependency
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, w_group_size=-1,
    inplace=False, get_scale_zp=False, clip_ratio=1.0,
):
    org_w_shape = w.shape
    if w_group_size > 0:
        assert org_w_shape[-1] % w_group_size == 0
        w = w.reshape(-1, w_group_size)
    elif w_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True) * clip_ratio
        min_val = w.amin(dim=1, keepdim=True) * clip_ratio
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True) * clip_ratio
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


class STE(torch.autograd.Function):
    """Memory-efficient fake quantization with straight-through estimator."""

    @staticmethod
    def forward(ctx, x, scales, zeros, min_int, max_int):
        x_int = torch.round(x / scales) + zeros
        ctx.save_for_backward((x_int >= min_int) & (x_int <= max_int))
        return (x_int.clamp(min_int, max_int) - zeros) * scales

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None, None, None, None


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min, max) - x).detach() + x


class SteIntAsymQuantizer(nn.Module):
    """
    Straight-Through Estimator based Integer Asymmetric Quantizer.

    Args:
        bit: Number of quantization bits (default: 4)
        w_group_size: Group size for quantization (default: 128, -1 for per-channel)
        enabled: Whether quantization is active (default: True)
    """
    def __init__(
        self,
        bit=4,
        w_group_size=128,
        enabled=True,
        learnable_qp=False,
        weight=None,
    ):
        super().__init__()
        self.w_group_size = w_group_size
        self.bit = bit
        self.learnable_qp = learnable_qp
        self._enabled = enabled
        self._update_bounds()

        if self.learnable_qp:
            x = weight.reshape(-1, self.w_group_size)
            xmin = x.amin([-1], keepdim=True)
            xmax = x.amax([-1], keepdim=True)
            range = xmax - xmin
            scales = range / (2**self.bit - 1)
            scales = scales.clamp(min=1e-4, max=1e4)
            zeros = -(xmin / scales).clamp(min=-1e4, max=1e4)
            self.scales = nn.Parameter(scales)
            self.zeros = nn.Parameter(zeros.round())

    def _update_bounds(self):
        """Update min/max integer bounds based on bit width."""
        self.max_int = 2 ** self.bit - 1
        self.min_int = 0

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def enable(self):
        """Enable quantization."""
        self._enabled = True

    def disable(self):
        """Disable quantization (pass-through mode)."""
        self._enabled = False

    def set_bit(self, bit: int):
        """Change quantization bit width."""
        self.bit = bit
        self._update_bounds()

    def forward(self, x):
        # Pass-through if disabled or bit >= 16
        if not self._enabled or self.bit >= 16:
            return x

        org_w_shape = x.shape

        if self.w_group_size > 0:
            assert org_w_shape[-1] % self.w_group_size == 0
            x = x.reshape(-1, self.w_group_size)
        elif self.w_group_size == -1:
            x = x.reshape(-1, org_w_shape[-1])

        assert x.dim() == 2

        if self.learnable_qp:
            scales = clamp_ste(GradMultiply.apply(self.scales, 10.0), 1e-4, 1e4)
            zeros = clamp_ste(round_ste(GradMultiply.apply(self.zeros, 10.0)), self.min_int, self.max_int)
            x = ((round_ste(x / scales) + zeros).clamp(self.min_int, self.max_int) - zeros) * scales
        else:
            with torch.no_grad():
                max_val = x.amax(dim=1, keepdim=True)
                min_val = x.amin(dim=1, keepdim=True)
                scales = (max_val - min_val).clamp(min=1e-5) / self.max_int
                zeros = (-torch.round(min_val / scales)).clamp(self.min_int, self.max_int)
            x = STE.apply(x, scales, zeros, self.min_int, self.max_int)

        return x.reshape(org_w_shape)

    @torch.no_grad()
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to weight without STE (for baking weights).
        Returns the quantized weight in the same dtype.
        """
        if not self._enabled or self.bit >= 16:
            return weight.clone()

        org_shape = weight.shape
        org_dtype = weight.dtype

        if self.w_group_size > 0:
            assert org_shape[-1] % self.w_group_size == 0
            w = weight.reshape(-1, self.w_group_size)
        elif self.w_group_size == -1:
            w = weight.reshape(-1, org_shape[-1])

        if self.learnable_qp:
            scales = self.scales.clamp(1e-4, 1e4)
            zeros = torch.round(self.zeros).clamp(self.min_int, self.max_int)
        else:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            scales = (max_val - min_val).clamp(min=1e-5) / self.max_int
            zeros = (-torch.round(min_val / scales)).clamp(self.min_int, self.max_int)

        w_int = torch.round(w / scales) + zeros
        w_int = w_int.clamp(self.min_int, self.max_int)
        w_quant = (w_int - zeros) * scales

        return w_quant.reshape(org_shape).to(org_dtype)

    def extra_repr(self):
        return f'bit={self.bit}, group_size={self.w_group_size}, enabled={self._enabled}, learnable_qp={self.learnable_qp}'


class SteN2F3Quantizer(nn.Module):
    def __init__(self, w_group_size=128, nf_dtype=torch.float32, enabled=True):
        super().__init__()
        self.w_group_size = w_group_size
        self.nf_dtype = nf_dtype
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x):
        if not self._enabled:
            return x

        org_w_shape = x.shape

        if self.w_group_size > 0:
            assert org_w_shape[-1] % self.w_group_size == 0
            qx = x.reshape(-1, self.w_group_size)
        elif self.w_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        max_val = qx.amax(dim=1, keepdim=True)
        min_val = qx.amin(dim=1, keepdim=True)

        scale_pos = torch.abs(max_val)
        scale_neg = torch.abs(min_val)

        dev = qx.device
        x_pos = torch.zeros_like(qx)
        x_neg = torch.zeros_like(qx)
        x_pos = torch.where(qx >= 0, qx, x_pos)
        x_neg = torch.where(qx < 0, qx, x_neg)
        q_pos = x_pos / scale_pos
        q_neg = x_neg / scale_neg

        q_pos, q_neg = self.round_pass(q_pos, q_neg, dev)

        qx = q_pos * scale_pos + q_neg * scale_neg
        qx = qx.reshape(org_w_shape).to(torch.bfloat16)

        return qx

    def round_n2f3(self, q_pos, q_neg, dev):
        q_pos = torch.where(q_pos >= 0.8114928305149078, torch.tensor(1.0, dtype=self.nf_dtype).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.8114928305149078) & (q_pos >= 0.5024898052215576), torch.tensor(0.6229856610298157, dtype=self.nf_dtype).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.5024898052215576) & (q_pos >= 0.2826657369732857), torch.tensor(0.3819939494132996, dtype=self.nf_dtype).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.2826657369732857) & (q_pos >= 0.0916687622666359), torch.tensor(0.1833375245332718, dtype=self.nf_dtype).to(dev), q_pos)
        q_pos = torch.where(q_pos < 0.0916687622666359, torch.tensor(0, dtype=self.nf_dtype).to(dev), q_pos)

        q_neg = torch.where(q_neg >= -0.1234657019376755, torch.tensor(0, dtype=self.nf_dtype).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.1234657019376755) & (q_neg >= -0.39097706973552704), torch.tensor(-0.2469314038753510, dtype=self.nf_dtype).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.39097706973552704) & (q_neg >= -0.7675113677978516), torch.tensor(-0.5350227355957031, dtype=self.nf_dtype).to(dev), q_neg)
        q_neg = torch.where(q_neg < -0.7675113677978516, torch.tensor(-1.0, dtype=self.nf_dtype).to(dev), q_neg)

        return q_pos, q_neg

    def round_pass(self, q_pos, q_neg, dev):
        y_grad_pos, y_grad_neg = q_pos, q_neg
        y_pos, y_neg = self.round_n2f3(q_pos, q_neg, dev)
        return (y_pos - y_grad_pos).detach() + y_grad_pos, (y_neg - y_grad_neg).detach() + y_grad_neg


CLIPMIN = 1e-5


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self, n_bits: int = 8, symmetric: bool = False, per_channel_axes=[],
        metric="minmax", dynamic=False, dynamic_method="per_cluster",
        group_size=None, shape=None, lwc=False, disable_zero_point=False,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None
        self.scale = None
        self.zero_point = None
        self.round_zero_point = None
        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc

        init_value = 2.
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        self.sigmoid = nn.Sigmoid()
        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, :-self.deficiency]
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2**(self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


class Step(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = (sign + 1) / 2
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


# ---------------------------------------------------------------------------
# FP8 (E4M3FN) helpers — native on Hopper+, software emulation on Ampere
# ---------------------------------------------------------------------------

# Keep numeric constants explicit so fallback works even on environments where
# FP8 dtype kernels are unavailable on the current accelerator.
FP8_E4M3FN_MAX = 448.0
FP8_E4M3FN_MIN_POS = 2.0 ** -9


def _build_fp8_e4m3fn_positive_values():
    values = [0.0]

    # Subnormals (exp=0, mantissa=1..7): m * 2^-9
    for m in range(1, 8):
        values.append(m * FP8_E4M3FN_MIN_POS)

    # Normals (exp=1..14, mantissa=0..7), bias=7
    for exp_bits in range(1, 15):
        exp = exp_bits - 7
        base = 2.0 ** exp
        for mantissa in range(8):
            values.append((1.0 + mantissa / 8.0) * base)

    # Extended finite range for e4m3fn (exp=15, mantissa=0..6), mantissa=7 is NaN.
    base = 2.0 ** 8
    for mantissa in range(7):
        values.append((1.0 + mantissa / 8.0) * base)

    return torch.tensor(values, dtype=torch.float32)


_FP8_E4M3FN_POS_VALUES = _build_fp8_e4m3fn_positive_values()
_FP8_E4M3FN_BOUNDARIES = (
    _FP8_E4M3FN_POS_VALUES[:-1] + _FP8_E4M3FN_POS_VALUES[1:]
) * 0.5


def _device_supports_native_fp8_round(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    # Hopper (SM90) or newer supports native FP8 tensor core path.
    return major >= 9


def _select_fp8_round_backend(device: torch.device, backend: str = "auto") -> str:
    if backend not in {"auto", "native", "emulated"}:
        raise ValueError(
            f"Unsupported FP8 backend '{backend}'. "
            "Choose from {'auto', 'native', 'emulated'}."
        )
    if backend == "auto":
        return "native" if _device_supports_native_fp8_round(device) else "emulated"
    return backend


def _round_to_fp8_e4m3fn_emulated(x: torch.Tensor) -> torch.Tensor:
    """Software FP8-E4M3FN round-trip in FP32 domain."""
    dev = x.device
    finite = torch.isfinite(x)
    x_finite = torch.where(
        finite, x, torch.zeros_like(x)
    ).clamp(min=-FP8_E4M3FN_MAX, max=FP8_E4M3FN_MAX)

    sign = x_finite.sign()
    mag = x_finite.abs()
    boundaries = _FP8_E4M3FN_BOUNDARIES.to(dev)
    pos_values = _FP8_E4M3FN_POS_VALUES.to(dev)

    idx = torch.bucketize(mag, boundaries)
    rounded = sign * pos_values[idx]

    # Match finite-no-inf behavior: +/-inf saturate, NaN stays NaN.
    pos_inf_mask = torch.isposinf(x)
    neg_inf_mask = torch.isneginf(x)
    nan_mask = torch.isnan(x)
    rounded = torch.where(pos_inf_mask, torch.full_like(rounded, FP8_E4M3FN_MAX), rounded)
    rounded = torch.where(neg_inf_mask, torch.full_like(rounded, -FP8_E4M3FN_MAX), rounded)
    rounded = torch.where(nan_mask, torch.full_like(rounded, float("nan")), rounded)
    return rounded


def round_to_fp8_e4m3fn(x: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Round tensor to FP8-E4M3FN grid and return in the same dtype as *x*."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def pseudo_fp8_quantize_tensor(x, per_tensor=True):
    """
    Fake FP8 quantization using native-or-emulated FP8 E4M3FN rounding.

    Args:
        x: Input tensor (any dtype).
        per_tensor: If True, one scale for the entire tensor.
                    If False, per-row scale (x must be 2D).
    Returns:
        Fake-quantized tensor in the original dtype.
    """
    orig_dtype = x.dtype
    x = x.float()
    fp8_max = FP8_E4M3FN_MAX

    if per_tensor:
        amax = x.abs().amax().clamp(min=1e-12)
        scale = amax / fp8_max
        x_scaled = x / scale
        x_deq = round_to_fp8_e4m3fn(x_scaled) * scale
    else:
        assert x.dim() == 2, "per-row FP8 quantization requires 2D input"
        amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = amax / fp8_max
        x_scaled = x / scale
        x_deq = round_to_fp8_e4m3fn(x_scaled) * scale

    return x_deq.to(orig_dtype)


# ---------------------------------------------------------------------------
# NVFP4 (E2M1) quantization
# ---------------------------------------------------------------------------

# The 8 non-negative representable values of FP4 E2M1
_FP4_E2M1_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Decision boundaries (midpoints) between consecutive positive values
_FP4_E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00])
FP4_E2M1_MAX = 6.0


def _round_to_fp4_e2m1(x):
    """Snap each element of *x* to the nearest FP4 E2M1 value.

    Uses ``torch.bucketize`` for vectorised, GPU-friendly rounding.
    Values whose magnitude exceeds 6.0 are saturated (no NaN).
    """
    sign = x.sign()
    a = x.abs()
    dev = x.device
    boundaries = _FP4_E2M1_BOUNDARIES.to(dev)
    pos_values = _FP4_E2M1_POS_VALUES.to(dev)
    # bucketize returns the index of the bucket each value falls into
    idx = torch.bucketize(a, boundaries)          # shape same as x
    rounded = pos_values[idx]                      # lookup
    return sign * rounded


def _percentile_amax(x_abs_flat, percentile, subsample=1_000_000):
    """Compute robust amax using percentile with optional subsampling."""
    if x_abs_flat.numel() > subsample:
        idx = torch.randint(x_abs_flat.numel(), (subsample,), device=x_abs_flat.device)
        sample = x_abs_flat[idx]
    else:
        sample = x_abs_flat
    return torch.quantile(sample, percentile).clamp(min=1e-12)


def pseudo_nvfp4_quantize_tensor(x, nvfp4_block_size=16, per_tensor_global=True,
                                 outlier_block_ids=None, channel_perm=None,
                                 channel_inv_perm=None, act_percentile=None):
    """Fake NVFP4 quantization with two-level scaling.

    1. Global FP32 scale (per-tensor or per-row)
    2. Local FP8-E4M3 scale per block of ``nvfp4_block_size`` elements

    Args:
        x: Input tensor (float32 recommended).
        nvfp4_block_size: Block size for local scales (default 16, NVIDIA spec).
        per_tensor_global: If True, one global scale for the whole tensor.
                           If False, per-row global scale (x must be 2D).
        outlier_block_ids: Optional list of block indices that get their own
            FP32 global scale, while normal blocks share a scale computed
            without the outlier blocks (dual global scale).
    Returns:
        Fake-quantized tensor in the same dtype as *x*.
    """
    orig_dtype = x.dtype
    x = x.float()
    orig_shape = x.shape

    # Reshape last dim into blocks of nvfp4_block_size
    flat = x.reshape(-1, x.shape[-1])
    if channel_perm is not None:
        flat = flat[:, channel_perm]
    N, D = flat.shape
    assert D % nvfp4_block_size == 0, \
        f"Last dim ({D}) must be divisible by nvfp4_block_size ({nvfp4_block_size})"
    num_blocks = D // nvfp4_block_size
    blocked = flat.reshape(N, num_blocks, nvfp4_block_size)

    # --- Global decode scale s_dec = amax / (FP4_MAX * FP8_MAX) ---
    # Following NVIDIA spec (Appendix B, Eq 1): s_enc = FP4_MAX * FP8_MAX / amax
    # s_dec = 1/s_enc stored in FP32 for decoding.
    _use_dual = outlier_block_ids is not None and per_tensor_global
    if _use_dual:
        # Check if all blocks are outliers — if so, skip dual scale (no benefit)
        valid_outlier_ids = [bid for bid in outlier_block_ids if bid < num_blocks]
        if len(valid_outlier_ids) >= num_blocks:
            _use_dual = False
    if _use_dual:
        # Dual global scale: outlier blocks get their own s_dec, normal blocks share one
        non_outlier_mask = torch.ones(num_blocks, dtype=torch.bool, device=x.device)
        for bid in valid_outlier_ids:
            non_outlier_mask[bid] = False
        if act_percentile is not None:
            normal_amax = _percentile_amax(blocked[:, non_outlier_mask, :].abs().flatten(), act_percentile)
        else:
            normal_amax = blocked[:, non_outlier_mask, :].abs().amax().clamp(min=1e-12)
        s_dec_normal = normal_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)
        block_s_dec = torch.full(
            (1, num_blocks, 1), s_dec_normal.item(), device=x.device)
        for bid in valid_outlier_ids:
            if act_percentile is not None:
                bid_amax = _percentile_amax(blocked[:, bid:bid+1, :].abs().flatten(), act_percentile)
            else:
                bid_amax = blocked[:, bid:bid+1, :].abs().amax().clamp(min=1e-12)
            block_s_dec[0, bid, 0] = (bid_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)).item()

        # Local decode scale on ORIGINAL blocked values (Eq 2)
        local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        local_dec = local_amax / FP4_E2M1_MAX
        # Scale into FP8 range before cast (Eq 3)
        local_dec_fp8 = (local_dec / block_s_dec).clamp(max=FP8_E4M3FN_MAX).to(torch.float8_e4m3fn).to(torch.float32)

        # Encode + quantize (Eq 4)
        scaled_vals = blocked / (local_dec_fp8 * block_s_dec)
        quantized_vals = _round_to_fp4_e2m1(scaled_vals)

        # Decode (Eq 5)
        deq = quantized_vals * local_dec_fp8 * block_s_dec
    else:
        if per_tensor_global:
            if act_percentile is not None:
                global_amax = _percentile_amax(x.abs().flatten(), act_percentile)
            else:
                global_amax = x.abs().amax().clamp(min=1e-12)
            s_dec = global_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)
        else:
            assert x.dim() == 2, "per-row global scale requires 2D input"
            global_amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            s_dec = global_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)

        s_dec_bcast = s_dec if per_tensor_global else s_dec.reshape(-1, 1, 1)

        # Local decode scale on ORIGINAL blocked values (Eq 2)
        local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        local_dec = local_amax / FP4_E2M1_MAX
        # Scale into FP8 range before cast (Eq 3)
        local_dec_fp8 = (local_dec / s_dec_bcast).clamp(max=FP8_E4M3FN_MAX).to(torch.float8_e4m3fn).to(torch.float32)

        # Encode + quantize (Eq 4)
        scaled_vals = blocked / (local_dec_fp8 * s_dec_bcast)
        quantized_vals = _round_to_fp4_e2m1(scaled_vals)

        # Decode (Eq 5)
        deq = quantized_vals * local_dec_fp8 * s_dec_bcast

    x_deq = deq.reshape(N, D)
    if channel_inv_perm is not None:
        x_deq = x_deq[:, channel_inv_perm]
    x_deq = x_deq.reshape(orig_shape)

    return x_deq.to(orig_dtype)


def pseudo_nvfp4_quantize_tensor_with_global_scale(
    x, global_scale, nvfp4_block_size=16,
):
    """Fake NVFP4 quantization with a **pre-computed frozen** global scale.

    Used for calibrated KV cache quantization where the global scale was
    determined during a calibration pass and frozen, while local scales are
    still computed dynamically per block.

    The global_scale is s_dec = amax / (FP4_MAX * FP8_MAX), following the
    NVIDIA spec.  For backward compatibility, old-convention scales
    (amax / FP4_MAX, typically > 1) are auto-detected and converted.

    Args:
        x: Input tensor (float32 recommended).
        global_scale: Pre-computed FP32 global decode scale (s_dec).
        nvfp4_block_size: Block size for local scales (default 16).
    Returns:
        Fake-quantized tensor in the same dtype as *x*.
    """
    orig_dtype = x.dtype
    x = x.float()
    s_dec = global_scale.float()

    # Backward compat: old convention stored amax/6 (typically > 1)
    if s_dec.max().item() > 1.0:
        s_dec = s_dec / FP8_E4M3FN_MAX

    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    N, D = flat.shape
    assert D % nvfp4_block_size == 0, \
        f"Last dim ({D}) must be divisible by nvfp4_block_size ({nvfp4_block_size})"
    blocked = flat.reshape(N, D // nvfp4_block_size, nvfp4_block_size)

    # Local decode scale on original values, scaled into FP8 range
    local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    local_dec = local_amax / FP4_E2M1_MAX
    local_dec_fp8 = (local_dec / s_dec).clamp(max=FP8_E4M3FN_MAX).to(torch.float8_e4m3fn).to(torch.float32)

    # Encode + quantize
    scaled_vals = blocked / (local_dec_fp8 * s_dec)
    quantized_vals = _round_to_fp4_e2m1(scaled_vals)

    # Decode
    deq = quantized_vals * local_dec_fp8 * s_dec
    x_deq = deq.reshape(N, D).reshape(orig_shape)

    return x_deq.to(orig_dtype)


# ---------------------------------------------------------------------------
# Differentiable NVFP4 helpers for QAT
# ---------------------------------------------------------------------------

class FP8Round(Function):
    """STE for FP8 E4M3 round-trip.

    ``x.to(torch.float8_e4m3fn)`` breaks the autograd graph in PyTorch
    (gradient becomes None).  This custom Function restores the gradient
    via straight-through estimation.
    """
    @staticmethod
    def forward(ctx, input):
        return input.to(torch.float8_e4m3fn).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class SteFp8Quantizer(nn.Module):
    """Differentiable FP8 E4M3 quantizer with per-tensor scaling for QAT.

    Uses FP8Round (STE) for the FP8 cast so gradients flow through.

    Args:
        per_tensor: If True, one scale for the entire tensor.
                    If False, per-row scale (input must be 2D).
        enabled: Whether quantization is active.
    """

    def __init__(self, per_tensor=True, enabled=True):
        super().__init__()
        self.per_tensor = per_tensor
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x):
        if not self._enabled:
            return x
        fp8_max = FP8_E4M3FN_MAX
        x_f = x.float()
        with torch.no_grad():
            if self.per_tensor:
                amax = x_f.abs().amax().clamp(min=1e-12)
            else:
                amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = amax / fp8_max
        x_scaled = x_f / scale
        x_fp8 = FP8Round.apply(x_scaled)  # STE through FP8 cast
        return (x_fp8 * scale).to(x.dtype)

    @torch.no_grad()
    def quantize_weight(self, weight):
        """Apply fake FP8 quantization without STE (for baking weights)."""
        if not self._enabled:
            return weight.clone()
        return pseudo_fp8_quantize_tensor(
            weight.float(), per_tensor=self.per_tensor,
        ).to(weight.dtype)

    def extra_repr(self):
        return f'per_tensor={self.per_tensor}, enabled={self._enabled}'


def round_pass_fp4(x):
    """Differentiable FP4 E2M1 rounding via STE.

    Forward: snap to nearest FP4 value.  Backward: identity.
    """
    y = _round_to_fp4_e2m1(x)
    return (y - x).detach() + x


class SteNvfp4Quantizer(nn.Module):
    """Differentiable NVFP4 quantizer for quantization-aware training.

    Two-level scaling: global FP32 scale + local FP8-E4M3 scale per block
    of ``nvfp4_block_size`` elements, following the NVIDIA Blackwell NVFP4
    format.

    Args:
        nvfp4_block_size: Elements per micro-block (default 16, NVIDIA spec).
        enabled: Whether quantization is active.
        learnable_local_scale: If True, local scales become learnable
            ``nn.Parameter``s (initialized from data on first forward).
            Default False matches HW dynamic scaling.
        learnable_global_scale: If True, the global scale becomes a learnable
            ``nn.Parameter`` in log space (LSQ-style, following OmniQuant).
            Default False computes from tensor amax each forward.
    """

    def __init__(
        self,
        nvfp4_block_size=16,
        enabled=True,
        learnable_local_scale=False,
        learnable_global_scale=False,
        learnable_act_clip: str = 'none',
        in_features: int = 0,
    ):
        super().__init__()
        self.nvfp4_block_size = nvfp4_block_size
        self._enabled = enabled
        self.learnable_local_scale = learnable_local_scale
        self.learnable_global_scale = learnable_global_scale
        self._local_scales = None    # lazily initialized as nn.Parameter
        self.outlier_block_ids = None  # set externally for dual global scale
        self._outlier_scales = None   # nn.ParameterList for learnable dual scale
        self.channel_perm = None      # [D] LongTensor or None
        self.channel_inv_perm = None  # [D] LongTensor or None
        # Pre-allocate for DDP compatibility (DDP requires all params at wrap time).
        # Value 0.0 = log(1.0), updated from data on first forward.
        if learnable_global_scale:
            self._global_scale = nn.Parameter(torch.tensor(0.0))
            self._global_scale_initialized = False
        else:
            self._global_scale = None

        # Improved Learnable Activation Clipping (LAC v2)
        # Uses clamp(param, 0.01, 1.0) instead of sigmoid for gradient=1 in range.
        self.learnable_act_clip = learnable_act_clip
        if learnable_act_clip == 'scalar':
            self._clip_factor = nn.Parameter(torch.tensor(0.98))
        elif learnable_act_clip == 'channel':
            assert in_features > 0, "in_features required for channel LAC"
            self._clip_factor = nn.Parameter(torch.ones(in_features) * 0.98)
        else:
            self._clip_factor = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x):
        if not self._enabled:
            return x

        orig_shape = x.shape
        x_f = x.float()

        # Reshape last dim into blocks
        flat = x_f.reshape(-1, x_f.shape[-1])
        # Channel reorder: sort channels by magnitude before blocking
        if self.channel_perm is not None:
            flat = flat[:, self.channel_perm]
        N, D = flat.shape
        assert D % self.nvfp4_block_size == 0
        num_blocks = D // self.nvfp4_block_size

        # --- Improved LAC: learnable activation clipping before quantization ---
        if self.learnable_act_clip == 'scalar' and self._clip_factor is not None:
            clip_ratio = torch.clamp(self._clip_factor, 0.01, 1.0)
            with torch.no_grad():
                token_amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            flat = flat.clamp(-clip_ratio * token_amax, clip_ratio * token_amax)
        elif self.learnable_act_clip == 'channel' and self._clip_factor is not None:
            clip_ratio = torch.clamp(self._clip_factor, 0.01, 1.0)  # [D]
            with torch.no_grad():
                ch_amax = flat.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)  # [1, D]
            flat = flat.clamp(-clip_ratio * ch_amax, clip_ratio * ch_amax)

        blocked = flat.reshape(N, num_blocks, self.nvfp4_block_size)

        # --- Global decode scale s_dec = amax / (FP4_MAX * FP8_MAX) ---
        # Following NVIDIA spec: local scales are computed on original values
        # and scaled into FP8 range [0, 448] before casting.
        fp8_min_pos = FP8_E4M3FN_MIN_POS
        use_dual_scale = self.outlier_block_ids is not None
        if use_dual_scale:
            # Check if all blocks are outliers — if so, skip dual scale (no benefit)
            valid_outlier_ids = [bid for bid in self.outlier_block_ids if bid < num_blocks]
            if len(valid_outlier_ids) >= num_blocks:
                use_dual_scale = False
        if use_dual_scale:
            with torch.no_grad():
                non_outlier_mask = torch.ones(num_blocks, dtype=torch.bool, device=x.device)
                for bid in valid_outlier_ids:
                    non_outlier_mask[bid] = False
                normal_amax = blocked[:, non_outlier_mask, :].abs().amax().clamp(min=1e-12)
                s_dec_normal = normal_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)
                block_s_dec = torch.full(
                    (1, num_blocks, 1), s_dec_normal.item(), device=x.device)
                for bid in valid_outlier_ids:
                    bid_amax = blocked[:, bid:bid+1, :].abs().amax().clamp(min=1e-12)
                    block_s_dec[0, bid, 0] = (bid_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)).item()

            # Local decode on original values, scaled into FP8 range
            with torch.no_grad():
                local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                local_dec = local_amax / FP4_E2M1_MAX

            local_dec_fp8 = FP8Round.apply((local_dec / block_s_dec).clamp(max=FP8_E4M3FN_MAX))
            local_dec_fp8 = local_dec_fp8.clamp(min=fp8_min_pos)

            # Encode + FP4 quantize
            scaled_vals = blocked / (local_dec_fp8 * block_s_dec)
            quantized_vals = round_pass_fp4(scaled_vals)

            # Decode
            deq = quantized_vals * local_dec_fp8 * block_s_dec
        else:
            with torch.no_grad():
                global_amax = x_f.abs().amax().clamp(min=1e-12)
                data_s_dec = global_amax / (FP4_E2M1_MAX * FP8_E4M3FN_MAX)

            if self.learnable_global_scale:
                if not self._global_scale_initialized:
                    # Initialize in log space from data statistics (first forward)
                    self._global_scale.data.copy_(data_s_dec.log().detach())
                    self._global_scale_initialized = True
                s_dec = torch.exp(self._global_scale)   # differentiable
            else:
                s_dec = data_s_dec                       # detached

            # Local decode on original values, scaled into FP8 range
            with torch.no_grad():
                local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                local_dec = local_amax / FP4_E2M1_MAX

            if self.learnable_local_scale:
                if self._local_scales is None:
                    self._local_scales = nn.Parameter((local_dec / s_dec).clamp(max=FP8_E4M3FN_MAX).detach().clone())
                local_dec_fp8 = FP8Round.apply(self._local_scales.clamp(max=FP8_E4M3FN_MAX))
            else:
                local_dec_fp8 = FP8Round.apply((local_dec / s_dec).clamp(max=FP8_E4M3FN_MAX))

            local_dec_fp8 = local_dec_fp8.clamp(min=fp8_min_pos)

            # Encode + FP4 quantize (differentiable via STE)
            scaled_vals = blocked / (local_dec_fp8 * s_dec)
            quantized_vals = round_pass_fp4(scaled_vals)

            # Decode
            deq = quantized_vals * local_dec_fp8 * s_dec

        x_deq = deq.reshape(N, D)
        if self.channel_inv_perm is not None:
            x_deq = x_deq[:, self.channel_inv_perm]
        x_deq = x_deq.reshape(orig_shape)

        return x_deq.to(x.dtype)

    @torch.no_grad()
    def quantize_weight(self, weight):
        """Apply fake NVFP4 quantization without STE (for baking at inference)."""
        if not self._enabled:
            return weight.clone()
        return pseudo_nvfp4_quantize_tensor(
            weight.float(), nvfp4_block_size=self.nvfp4_block_size,
        ).to(weight.dtype)

    def extra_repr(self):
        parts = [
            f'block_size={self.nvfp4_block_size}',
            f'enabled={self._enabled}',
            f'learnable_local_scale={self.learnable_local_scale}',
            f'learnable_global_scale={self.learnable_global_scale}',
        ]
        if self.learnable_act_clip != 'none':
            parts.append(f'learnable_act_clip={self.learnable_act_clip}')
        return ', '.join(parts)


class SteNvfp4QuantizerFrozenScale(nn.Module):
    """Differentiable NVFP4 quantizer with a frozen (calibrated) global scale.

    Matches eval-time behavior in ``pseudo_nvfp4_quantize_tensor_with_global_scale``:
    the global scale is pre-computed from calibration data and frozen, while
    local FP8-E4M3 scales per block are computed dynamically.

    STE gradients flow through the local FP8 scale rounding (``FP8Round``)
    and FP4 value rounding (``round_pass_fp4``).

    Args:
        frozen_global_scale: Pre-computed FP32 global scale (scalar tensor).
        nvfp4_block_size: Elements per micro-block (default 16).
        enabled: Whether quantization is active.
    """

    def __init__(self, frozen_global_scale, nvfp4_block_size=16, enabled=True,
                 learnable_global_scale=False, lac=False):
        super().__init__()
        self.nvfp4_block_size = nvfp4_block_size
        self._enabled = enabled
        self.learnable_global_scale = learnable_global_scale

        # Backward compat: old convention stored amax/6 (typically > 1)
        s_dec = frozen_global_scale.float().detach().clone()
        if s_dec.max().item() > 1.0:
            s_dec = s_dec / FP8_E4M3FN_MAX

        if learnable_global_scale:
            # Log-space parameter for stable gradient flow (LSQ-style)
            self._log_global_scale = nn.Parameter(s_dec.log())
        else:
            self.register_buffer('global_scale', s_dec)

        # Learnable Activation Clipping (LAC) for KV cache
        # sigmoid(3.0) = 0.953 -> clips top ~5%
        if lac:
            self.lac_upbound = nn.Parameter(torch.tensor(3.0))
        else:
            self.lac_upbound = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x):
        if not self._enabled:
            return x

        orig_shape = x.shape
        x_f = x.float()

        # LAC: learnable activation clipping before quantization
        if self.lac_upbound is not None:
            with torch.no_grad():
                amax = x_f.abs().amax().clamp(min=1e-12)
            clip_val = torch.sigmoid(self.lac_upbound) * amax
            x_f = x_f.clamp(-clip_val, clip_val)

        # Global decode scale s_dec: learnable (log-space) or frozen
        if self.learnable_global_scale:
            s_dec = torch.exp(self._log_global_scale)
        else:
            s_dec = self.global_scale

        # Reshape last dim into blocks (on original x, not normalized)
        flat = x_f.reshape(-1, x_f.shape[-1])
        N, D = flat.shape
        assert D % self.nvfp4_block_size == 0, \
            f"Last dim ({D}) must be divisible by nvfp4_block_size ({self.nvfp4_block_size})"
        blocked = flat.reshape(N, D // self.nvfp4_block_size, self.nvfp4_block_size)

        # Local decode on original values, scaled into FP8 range
        with torch.no_grad():
            local_amax = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            local_dec = local_amax / FP4_E2M1_MAX

        local_dec_fp8 = FP8Round.apply((local_dec / s_dec).clamp(max=FP8_E4M3FN_MAX))

        # Clamp to min positive FP8 value to prevent division by zero
        fp8_min_pos = FP8_E4M3FN_MIN_POS
        local_dec_fp8 = local_dec_fp8.clamp(min=fp8_min_pos)

        # Encode + FP4 quantize (differentiable via STE)
        scaled_vals = blocked / (local_dec_fp8 * s_dec)
        quantized_vals = round_pass_fp4(scaled_vals)

        # Decode
        deq = quantized_vals * local_dec_fp8 * s_dec
        x_deq = deq.reshape(N, D).reshape(orig_shape)

        return x_deq.to(x.dtype)

    def extra_repr(self):
        if self.learnable_global_scale:
            gs = torch.exp(self._log_global_scale).item()
            return (f'block_size={self.nvfp4_block_size}, enabled={self._enabled}, '
                    f'global_scale={gs:.6g} (learnable)')
        return (
            f'block_size={self.nvfp4_block_size}, enabled={self._enabled}, '
            f'global_scale={self.global_scale.item():.6g}'
        )
