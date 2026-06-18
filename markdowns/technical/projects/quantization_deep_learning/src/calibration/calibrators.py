"""
Calibration techniques for Post-Training Quantization.

Calibration determines the optimal clipping range [min, max] for
quantizing activations. A good range minimizes information loss
while fitting within the discrete INT8 grid.

Implemented calibrators (Strategy pattern — all share a common interface):
  1. MinMaxCalibrator:        range = [min(x), max(x)] — exact, outlier-sensitive
  2. PercentileCalibrator:    range = [p%, (1-p)%] — robust to outliers
  3. MSECalibrator:           alpha-scaling that minimizes MSE
  4. KLDivergenceCalibrator:  minimize KL-divergence between FP32 and INT8 histograms
  5. MovingAverageCalibrator: exponential moving average of min/max across batches

Also implements:
  - BatchNorm folding (fold_bn_into_conv)
  - Cross-Layer Equalization (simple weight range balancing)

All constants from config.yaml.
"""

import abc
import logging
import math
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_PROJ_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibrator interface (Abstract Base Class)
# ---------------------------------------------------------------------------

class BaseCalibrator(abc.ABC):
    """
    Abstract interface for all calibration strategies.

    Each calibrator observes activation tensors from calibration batches,
    then computes the optimal [min_val, max_val] range for INT8 quantization.
    """

    def __init__(self, num_bits: int = 8) -> None:
        self._num_bits = num_bits
        self._collected: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def collect(self, x: torch.Tensor) -> None:
        """
        Collect activation tensors from one calibration batch.

        Args:
            x: Activation tensor (any shape), FP32.
        """
        self._collected.append(x.detach().cpu().float())

    @abc.abstractmethod
    def compute_range(self) -> tuple[float, float]:
        """
        Compute the optimal [min_val, max_val] quantization range.

        Returns:
            Tuple of (min_val, max_val).
        """
        ...

    def compute_scale_zero_point(
        self, symmetric: bool = True
    ) -> tuple[float, int]:
        """
        Compute scale and zero_point from the computed range.

        Args:
            symmetric: Use symmetric quantization (zero_point=0).

        Returns:
            Tuple of (scale, zero_point).
        """
        min_val, max_val = self.compute_range()
        if symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            q_max = 2 ** (self._num_bits - 1) - 1
            scale = abs_max / q_max if abs_max > 0 else 1.0
            zero_point = 0
        else:
            q_range = 2 ** self._num_bits - 1
            scale = (max_val - min_val) / q_range if (max_val - min_val) > 0 else 1.0
            zero_point = int(round(-min_val / scale))
            zero_point = max(0, min(q_range, zero_point))
        return scale, zero_point

    def reset(self) -> None:
        """Clear collected tensors for reuse."""
        self._collected.clear()

    def _get_all_values(self) -> np.ndarray:
        """Concatenate all collected tensors into a flat numpy array."""
        if not self._collected:
            raise RuntimeError(f"{self.name}: No calibration data collected. Call .collect() first.")
        return torch.cat([t.flatten() for t in self._collected]).numpy()


# ---------------------------------------------------------------------------
# Concrete calibrators
# ---------------------------------------------------------------------------

class MinMaxCalibrator(BaseCalibrator):
    """
    Min-Max calibrator: range = [global_min, global_max].

    Pros: Exact range, no information loss within data.
    Cons: Extremely sensitive to outliers — one large outlier wastes most
          of the INT8 range, increasing quantization error for typical values.
    """

    def compute_range(self) -> tuple[float, float]:
        values = self._get_all_values()
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        logger.info(
            "[%s] range=[%.4f, %.4f] | n_values=%d",
            self.name, min_val, max_val, len(values),
        )
        return min_val, max_val


class PercentileCalibrator(BaseCalibrator):
    """
    Percentile calibrator: clips at a configurable percentile.

    range = [p-th percentile, (100-p)-th percentile]

    Common values: p=99.9 (clips top/bottom 0.1%) or p=99.99.

    Pros: Robust to outliers — outliers are clipped, typical values get
          finer quantization granularity.
    Cons: Clips valid data; requires tuning the percentile.
    """

    def __init__(self, percentile: float = 99.9, num_bits: int = 8) -> None:
        super().__init__(num_bits)
        if not (50.0 < percentile < 100.0):
            raise ValueError(f"percentile must be in (50, 100), got {percentile}")
        self._percentile = percentile

    def compute_range(self) -> tuple[float, float]:
        values = self._get_all_values()
        lower = float(np.percentile(values, 100.0 - self._percentile))
        upper = float(np.percentile(values, self._percentile))
        logger.info(
            "[%s] percentile=%.1f | range=[%.4f, %.4f] | clipped=%.2f%% of values",
            self.name, self._percentile, lower, upper,
            float(np.mean((values < lower) | (values > upper)) * 100),
        )
        return lower, upper


class MSECalibrator(BaseCalibrator):
    """
    MSE-minimizing calibrator.

    Searches for the optimal clipping alpha in [alpha_min, alpha_max] * max(|x|)
    such that the MSE between the original and fake-quantized tensor is minimized.

    MSE(alpha) = E[(x - dequant(quant(x, alpha))) ^ 2]

    This directly optimizes the metric we care about (quantization error).

    Pros: Optimal for minimizing MSE distortion.
    Cons: Requires a grid search (search_steps evaluations).
    """

    def __init__(
        self,
        search_steps: int = 100,
        alpha_range: tuple[float, float] = (0.8, 1.0),
        num_bits: int = 8,
    ) -> None:
        super().__init__(num_bits)
        self._search_steps = search_steps
        self._alpha_range = alpha_range

    def _fake_quantize_np(self, values: np.ndarray, abs_max: float) -> np.ndarray:
        """Simulate fake quantization in numpy for fast grid search."""
        q_max = 2 ** (self._num_bits - 1) - 1
        scale = abs_max / q_max if abs_max > 0 else 1.0
        clipped = np.clip(values, -abs_max, abs_max)
        x_int = np.round(clipped / scale)
        return x_int * scale

    def compute_range(self) -> tuple[float, float]:
        values = self._get_all_values()
        global_abs_max = float(np.max(np.abs(values)))

        alpha_min, alpha_max = self._alpha_range
        alphas = np.linspace(alpha_min, alpha_max, self._search_steps)

        best_mse = float("inf")
        best_alpha = alpha_max

        for alpha in alphas:
            clipped_max = alpha * global_abs_max
            quantized = self._fake_quantize_np(values, clipped_max)
            mse = float(np.mean((values - quantized) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        best_abs_max = best_alpha * global_abs_max
        logger.info(
            "[%s] best_alpha=%.4f | range=[%.4f, %.4f] | MSE=%.6e | search_steps=%d",
            self.name, best_alpha, -best_abs_max, best_abs_max,
            best_mse, self._search_steps,
        )
        return -best_abs_max, best_abs_max


class KLDivergenceCalibrator(BaseCalibrator):
    """
    KL-Divergence calibrator (TensorRT-style).

    Finds the clipping threshold T that minimizes KL(P_fp32 || P_int8)
    where P_fp32 is the FP32 histogram and P_int8 is the quantized histogram.

    Algorithm:
      1. Build histogram of FP32 activations with `kl_bins` bins.
      2. For each candidate threshold T (from max/2 to max):
         a. Clip the histogram to [-T, T].
         b. Quantize the clipped histogram to `num_quantized_bins` bins.
         c. Expand back and compute KL divergence.
      3. Choose T with minimum KL divergence.

    This is the standard NVIDIA TensorRT calibration approach.

    Pros: Best for preserving distribution shape, especially for activations.
    Cons: Most complex; requires tuning `kl_bins` and `num_quantized_bins`.
    """

    def __init__(
        self,
        kl_bins: int = 2048,
        num_quantized_bins: int = 256,
        num_bits: int = 8,
    ) -> None:
        super().__init__(num_bits)
        self._kl_bins = kl_bins
        self._num_quantized_bins = num_quantized_bins

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL(p || q), adding epsilon to avoid log(0).

        Args:
            p: Reference distribution (FP32 histogram).
            q: Approximation distribution (quantized histogram).

        Returns:
            KL divergence value.
        """
        eps = 1e-10
        p = p + eps
        q = q + eps
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def compute_range(self) -> tuple[float, float]:
        values = np.abs(self._get_all_values())  # symmetric: use absolute values

        # Build reference histogram
        max_val = float(np.max(values))
        if max_val == 0:
            logger.warning("[%s] All-zero activations, returning range [0, 1]", self.name)
            return -1.0, 1.0

        hist, bin_edges = np.histogram(values, bins=self._kl_bins, range=(0, max_val))

        # Search threshold from half-max to max
        n_candidates = self._kl_bins // 2
        best_kl = float("inf")
        best_threshold = max_val

        for i in range(n_candidates, self._kl_bins):
            threshold = bin_edges[i + 1]
            # Reference: clip histogram to [0, threshold]
            ref_hist = hist[:i + 1].copy().astype(np.float64)
            # Add clipped mass to last bin
            ref_hist[-1] += hist[i + 1:].sum()

            # Quantized: collapse to num_quantized_bins, then expand back
            bin_width = i + 1
            q_bins = self._num_quantized_bins
            quant_hist = np.zeros(bin_width, dtype=np.float64)

            for q_idx in range(q_bins):
                start = int(q_idx * bin_width / q_bins)
                end = int((q_idx + 1) * bin_width / q_bins)
                if end > start:
                    quant_hist[start:end] = ref_hist[start:end].mean()

            kl = self._kl_divergence(ref_hist, quant_hist)
            if kl < best_kl:
                best_kl = kl
                best_threshold = threshold

        logger.info(
            "[%s] best_threshold=%.4f | KL=%.6f | bins=%d | q_bins=%d",
            self.name, best_threshold, best_kl, self._kl_bins, self._num_quantized_bins,
        )
        return -best_threshold, best_threshold


class MovingAverageCalibrator(BaseCalibrator):
    """
    Moving Average calibrator for online / streaming calibration.

    Updates min/max using exponential moving average across batches:
        min_ema = (1 - alpha) * min_ema + alpha * batch_min
        max_ema = (1 - alpha) * max_ema + alpha * batch_max

    Pros: Memory-efficient (no stored tensors), online update.
    Cons: Initial batches have high influence; requires careful alpha tuning.
    """

    def __init__(self, alpha: float = 0.01, num_bits: int = 8) -> None:
        super().__init__(num_bits)
        self._alpha = alpha
        self._ema_min: float | None = None
        self._ema_max: float | None = None

    def collect(self, x: torch.Tensor) -> None:
        """Override collect to update EMA instead of storing tensors."""
        batch_min = float(x.min().item())
        batch_max = float(x.max().item())

        if self._ema_min is None:
            self._ema_min = batch_min
            self._ema_max = batch_max
        else:
            self._ema_min = (1 - self._alpha) * self._ema_min + self._alpha * batch_min
            self._ema_max = (1 - self._alpha) * self._ema_max + self._alpha * batch_max

        logger.debug(
            "[%s] batch=[%.4f, %.4f] | EMA=[%.4f, %.4f]",
            self.name, batch_min, batch_max, self._ema_min, self._ema_max,
        )

    def compute_range(self) -> tuple[float, float]:
        if self._ema_min is None or self._ema_max is None:
            raise RuntimeError(f"{self.name}: No data collected via .collect().")
        logger.info(
            "[%s] EMA range=[%.4f, %.4f] | alpha=%.4f",
            self.name, self._ema_min, self._ema_max, self._alpha,
        )
        return self._ema_min, self._ema_max

    def reset(self) -> None:
        """Reset EMA state."""
        self._ema_min = None
        self._ema_max = None


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> nn.Conv2d:
    """
    Fold BatchNorm parameters into the preceding Conv2d layer.

    After folding, the Conv2d performs:
        y = (W * gamma/std) * x + (beta - gamma * mean / std)

    This is equivalent to Conv2d → BatchNorm2d, but removes BN overhead
    and improves quantization (fewer quantized operations, fewer rounding steps).

    Mathematical derivation:
        BN: y = gamma * (x - mean) / std + beta
        Conv: z = W * x + b
        Combined: z = (W * gamma / std) * x + (b * gamma / std - gamma * mean / std + beta)

    Args:
        conv: Conv2d layer (bias may or may not be present).
        bn: BatchNorm2d layer (must be after the conv).

    Returns:
        New Conv2d with folded BatchNorm parameters (no bias term originally needed).
    """
    assert bn.running_mean is not None and bn.running_var is not None, \
        "BatchNorm must have running statistics (needs at least one forward pass)"

    # Extract BN parameters
    gamma = bn.weight.data  # scale
    beta = bn.bias.data     # shift
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps
    std = torch.sqrt(var + eps)

    # New convolution weights
    W = conv.weight.data  # shape: [out_ch, in_ch, kH, kW]
    W_folded = W * (gamma / std).view(-1, 1, 1, 1)

    # New bias
    if conv.bias is not None:
        b = conv.bias.data
        b_folded = (b - mean) * (gamma / std) + beta
    else:
        b_folded = beta - mean * (gamma / std)

    # Create new Conv2d with folded parameters
    folded_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    folded_conv.weight.data = W_folded
    folded_conv.bias.data = b_folded

    logger.debug(
        "Folded BN into Conv | out_channels=%d | gamma_range=[%.4f, %.4f]",
        conv.out_channels, float(gamma.min()), float(gamma.max()),
    )
    return folded_conv


# ---------------------------------------------------------------------------
# Cross-Layer Equalization
# ---------------------------------------------------------------------------

class CrossLayerEqualizer:
    """
    Simple Cross-Layer Equalization (CLE) for balancing weight ranges.

    CLE (Nagel et al., 2019): For a sequence Conv1 → ReLU → Conv2,
    a scaling factor s can be applied such that:
        Conv1_new = Conv1 * diag(s)^{-1}  (scale Conv1 output channels down)
        Conv2_new = diag(s) * Conv2        (scale Conv2 input channels up)

    The activation passes through ReLU unchanged (ReLU is scale-equivariant:
    ReLU(s*x) = s*ReLU(x) for s > 0).

    This balances the per-channel weight ranges across layers, improving
    INT8 quantization accuracy — especially for per-tensor quantization.

    Reference: Data-Free Quantization Through Weight Equalization and Bias Correction
               (Nagel et al., ICCV 2019)
    """

    def __init__(self) -> None:
        pass

    def equalize_two_layers(
        self,
        conv1: nn.Conv2d,
        conv2: nn.Conv2d,
    ) -> tuple[nn.Conv2d, nn.Conv2d]:
        """
        Equalize weight ranges between two consecutive Conv2d layers.

        Scaling: s[i] = sqrt(max(|W1[i, :, :, :]|) * max(|W2[:, i, :, :]|))

        Args:
            conv1: First conv layer (output channels indexed by i).
            conv2: Second conv layer (input channels indexed by i).

        Returns:
            Tuple of (equalized_conv1, equalized_conv2).
        """
        W1 = conv1.weight.data  # [out1, in1, kH, kW]
        W2 = conv2.weight.data  # [out2, in2, kH, kW]

        assert W1.shape[0] == W2.shape[1], (
            f"Conv1 out_channels ({W1.shape[0]}) must equal Conv2 in_channels ({W2.shape[1]})"
        )

        num_channels = W1.shape[0]
        scale = torch.zeros(num_channels)

        for i in range(num_channels):
            r1 = float(W1[i].abs().max())  # range of conv1 channel i output
            r2 = float(W2[:, i, :, :].abs().max())  # range of conv2 channel i input
            scale[i] = math.sqrt(r1 * r2) if r1 * r2 > 0 else 1.0

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Scale conv1 output channels down by 1/scale[i]
        W1_new = W1 / scale.view(-1, 1, 1, 1)

        # Scale conv2 input channels up by scale[i]
        W2_new = W2 * scale.view(1, -1, 1, 1)

        # Apply to new conv modules
        conv1_eq = self._clone_conv_with_weight(conv1, W1_new)
        conv2_eq = self._clone_conv_with_weight(conv2, W2_new)

        # Log range improvement
        r1_before = float(W1.abs().max())
        r2_before = float(W2.abs().max())
        r1_after = float(W1_new.abs().max())
        r2_after = float(W2_new.abs().max())

        logger.info(
            "CLE applied | Conv1 range: %.4f → %.4f | Conv2 range: %.4f → %.4f",
            r1_before, r1_after, r2_before, r2_after,
        )
        logger.info(
            "CLE balance: ratio before=%.2f | ratio after=%.2f",
            r1_before / (r2_before + 1e-8),
            r1_after / (r2_after + 1e-8),
        )

        return conv1_eq, conv2_eq

    @staticmethod
    def _clone_conv_with_weight(conv: nn.Conv2d, new_weight: torch.Tensor) -> nn.Conv2d:
        """Create a new Conv2d with updated weight, preserving all other params."""
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        new_conv.weight.data = new_weight.clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        return new_conv


# ---------------------------------------------------------------------------
# Calibrator factory
# ---------------------------------------------------------------------------

def build_calibrator_from_config(cfg: dict[str, Any], method: str, num_bits: int = 8) -> BaseCalibrator:
    """
    Build a calibrator instance from config parameters.

    Args:
        cfg: Full project config dict.
        method: Calibration method name (minmax, percentile, mse, kl, movingavg).
        num_bits: Quantization bit-width.

    Returns:
        Instantiated calibrator.
    """
    calib_cfg = cfg["calibration"]
    method = method.lower()

    if method == "minmax":
        calibrator = MinMaxCalibrator(num_bits=num_bits)
    elif method == "percentile":
        calibrator = PercentileCalibrator(
            percentile=calib_cfg["percentile"], num_bits=num_bits
        )
    elif method == "mse":
        calibrator = MSECalibrator(
            search_steps=calib_cfg["mse_search_steps"],
            alpha_range=tuple(calib_cfg["mse_alpha_range"]),
            num_bits=num_bits,
        )
    elif method == "kl":
        calibrator = KLDivergenceCalibrator(
            kl_bins=calib_cfg["kl_bins"],
            num_quantized_bins=calib_cfg["kl_num_quantized_bins"],
            num_bits=num_bits,
        )
    elif method == "movingavg":
        calibrator = MovingAverageCalibrator(
            alpha=calib_cfg["moving_average_constant"], num_bits=num_bits
        )
    else:
        raise ValueError(f"Unknown calibration method: '{method}'. "
                         "Choose from: minmax, percentile, mse, kl, movingavg")

    logger.info("Built calibrator: %s", calibrator.name)
    return calibrator


# ---------------------------------------------------------------------------
# Entry point — demo all calibrators on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    logger.info("=" * 60)
    logger.info("Calibration Methods Demo")
    logger.info("=" * 60)

    # Generate synthetic activation data with outliers
    torch.manual_seed(42)
    # Gaussian activations with rare large outliers (common in practice)
    base_activations = torch.randn(5000) * 0.5  # typical activations
    outliers = torch.tensor([-3.5, 3.8, -4.1, 4.5])  # a few outliers
    activations = torch.cat([base_activations, outliers])

    methods = ["minmax", "percentile", "mse", "kl", "movingavg"]

    logger.info("Activation data | n=%d | true_min=%.4f | true_max=%.4f | std=%.4f",
                len(activations), float(activations.min()), float(activations.max()),
                float(activations.std()))
    logger.info("")

    results = {}
    for method in methods:
        calibrator = build_calibrator_from_config(_cfg, method)
        calibrator.collect(activations)
        min_val, max_val = calibrator.compute_range()
        scale, zp = calibrator.compute_scale_zero_point(symmetric=True)
        results[method] = {
            "min": min_val, "max": max_val,
            "scale": scale, "zero_point": zp,
        }

    logger.info("")
    logger.info("%-20s | %-12s | %-12s | %-12s | %-10s", "Calibrator", "Min", "Max", "Scale", "ZeroPoint")
    logger.info("-" * 75)
    for method, r in results.items():
        logger.info(
            "%-20s | %-12.4f | %-12.4f | %-12.6f | %-10d",
            method.capitalize(), r["min"], r["max"], r["scale"], r["zero_point"],
        )

    # Demo: BN folding
    logger.info("")
    logger.info("--- BatchNorm Folding Demo ---")
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(16)
    # Initialize BN running stats by running a forward pass
    with torch.no_grad():
        dummy = torch.randn(4, 3, 8, 8)
        bn.eval()
        bn(conv(dummy))

    folded = fold_bn_into_conv(conv, bn)
    logger.info("Folded conv bias shape: %s", tuple(folded.bias.shape))
    logger.info("Folded conv weight shape: %s", tuple(folded.weight.shape))
    logger.info("BN folding complete — BN layer eliminated")

    # Demo: CLE
    logger.info("")
    logger.info("--- Cross-Layer Equalization Demo ---")
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    conv2 = nn.Conv2d(16, 32, 3, padding=1)
    # Artificially imbalance the weights
    with torch.no_grad():
        conv1.weight.data *= 10.0
        conv2.weight.data *= 0.1

    cle = CrossLayerEqualizer()
    conv1_eq, conv2_eq = cle.equalize_two_layers(conv1, conv2)
    logger.info("CLE complete — weight ranges balanced")
