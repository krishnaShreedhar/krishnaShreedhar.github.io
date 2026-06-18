"""
Fake Quantization demonstration.

Fake quantization simulates the effect of integer quantization while keeping
values in floating-point — allowing gradients to flow via the STE.

This module demonstrates:
  1. The quantize → dequantize cycle mathematically.
  2. Rounding error visualization for different bit-widths.
  3. How scale and zero_point are computed.
  4. Comparison of FP32 vs fake-quantized values.

All constants from config.yaml.
"""

import logging
import pathlib
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_PROJ_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake quantization math
# ---------------------------------------------------------------------------

class FakeQuantizeFunction(torch.autograd.Function):
    """
    Custom fake quantization with Straight-Through Estimator.

    Forward:  quantize → dequantize in FP32 domain.
    Backward: STE — gradient passes through unchanged (clipped to in-range).

    This is equivalent to torch.quantization.FakeQuantize but explicit
    for educational purposes.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
    ) -> torch.Tensor:
        # Save mask of in-range elements for backward STE clipping
        x_int = torch.round(x / scale) + zero_point
        x_int_clipped = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_int_clipped - zero_point) * scale

        # STE mask: gradient is 1 where x is in range, 0 where clipped
        in_range = (x_int >= quant_min) & (x_int <= quant_max)
        ctx.save_for_backward(in_range)

        return x_dequant

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple:
        (in_range,) = ctx.saved_tensors
        # STE: pass gradient through for in-range, zero for out-of-range (clipped)
        grad_input = grad_output * in_range.float()
        return grad_input, None, None, None, None


def fake_quantize(
    x: torch.Tensor,
    scale: float,
    zero_point: int,
    num_bits: int = 8,
    signed: bool = True,
) -> torch.Tensor:
    """
    Apply fake quantization to a tensor.

    Args:
        x: Input tensor (FP32).
        scale: Quantization scale factor.
        zero_point: Integer zero point.
        num_bits: Quantization bit-width.
        signed: If True, use signed integer range.

    Returns:
        Fake-quantized tensor (FP32 values on INT8 grid, same dtype as x).
    """
    if signed:
        quant_min = -(2 ** (num_bits - 1))
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2 ** num_bits - 1

    return FakeQuantizeFunction.apply(x, scale, zero_point, quant_min, quant_max)


def compute_scale_zero_point(
    x_min: float,
    x_max: float,
    num_bits: int = 8,
    symmetric: bool = True,
) -> tuple[float, int]:
    """
    Compute scale and zero_point from observed range.

    Symmetric:
        scale = max(|x_min|, |x_max|) / (2^(bits-1) - 1)
        zero_point = 0

    Asymmetric:
        scale = (x_max - x_min) / (2^bits - 1)
        zero_point = round(-x_min / scale)

    Args:
        x_min: Minimum observed value.
        x_max: Maximum observed value.
        num_bits: Target bit-width.
        symmetric: Use symmetric (zero_point=0) or asymmetric.

    Returns:
        Tuple of (scale, zero_point).
    """
    if symmetric:
        abs_max = max(abs(x_min), abs(x_max))
        q_max = 2 ** (num_bits - 1) - 1
        scale = abs_max / q_max if abs_max > 0 else 1.0
        zero_point = 0
    else:
        q_max = 2 ** num_bits - 1
        scale = (x_max - x_min) / q_max if (x_max - x_min) > 0 else 1.0
        zero_point = int(round(-x_min / scale))
        zero_point = max(0, min(q_max, zero_point))

    return scale, zero_point


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

class FakeQuantizationVisualizer:
    """
    Creates visualizations of fake quantization effects.

    Shows:
      - FP32 signal vs quantized signal (quantize-dequantize cycle).
      - Quantization error / rounding noise.
      - Effect of different bit-widths (INT8 vs INT4 vs INT2).
    """

    def __init__(self, output_dir: pathlib.Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot_quantize_dequantize_cycle(self, num_bits: int = 8) -> None:
        """
        Plot a 1D signal and its quantized version to visualize rounding error.

        Args:
            num_bits: Quantization bit-width to demonstrate.
        """
        # Generate a clean sine signal
        x_vals = np.linspace(-1.0, 1.0, 500)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32)

        scale, zp = compute_scale_zero_point(float(x_vals.min()), float(x_vals.max()), num_bits, symmetric=True)
        x_fake_quant = fake_quantize(x_tensor, scale, zp, num_bits).numpy()
        error = x_vals - x_fake_quant

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Top: original vs quantized
        ax = axes[0]
        ax.plot(x_vals, x_vals, "b-", linewidth=2, label="FP32 (original)", alpha=0.8)
        ax.step(x_vals, x_fake_quant, "r-", linewidth=1.5, label=f"Fake-quantized INT{num_bits}", alpha=0.9)
        ax.set_xlabel("Input value")
        ax.set_ylabel("Output value")
        ax.set_title(
            f"Fake Quantization (INT{num_bits}): quantize → dequantize cycle\n"
            f"scale={scale:.5f}, zero_point={zp}, levels={2**num_bits}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom: quantization error
        ax2 = axes[1]
        ax2.plot(x_vals, error, "g-", linewidth=1.5, alpha=0.9)
        ax2.axhline(scale / 2, color="red", linestyle="--", linewidth=1, label=f"±scale/2 = ±{scale/2:.5f}")
        ax2.axhline(-scale / 2, color="red", linestyle="--", linewidth=1)
        ax2.fill_between(x_vals, -scale / 2, scale / 2, alpha=0.1, color="red")
        ax2.set_xlabel("Input value")
        ax2.set_ylabel("Rounding error (FP32 - dequant)")
        ax2.set_title(f"Quantization Error | MSE={np.mean(error**2):.2e} | Max={np.max(np.abs(error)):.5f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self._output_dir / f"fake_quant_cycle_int{num_bits}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Fake quant cycle plot saved: %s", path)

    def plot_bitwidth_comparison(self) -> None:
        """
        Compare quantization error for INT8, INT4, and INT2.

        Shows how lower bit-width causes larger discrete steps and more error.
        """
        x_vals = np.linspace(-1.0, 1.0, 1000)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32)

        bit_widths = [8, 4, 2]
        colors = ["steelblue", "darkorange", "firebrick"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax_signal = axes[0]
        ax_error = axes[1]

        ax_signal.plot(x_vals, x_vals, "k--", linewidth=1, alpha=0.5, label="FP32 (ideal)")

        for bits, color in zip(bit_widths, colors):
            scale, zp = compute_scale_zero_point(
                float(x_vals.min()), float(x_vals.max()), bits, symmetric=True
            )
            x_fq = fake_quantize(x_tensor, scale, zp, bits).numpy()
            error = x_vals - x_fq
            mse = np.mean(error ** 2)

            ax_signal.step(x_vals, x_fq, color=color, linewidth=1.5,
                           alpha=0.85, label=f"INT{bits} ({2**bits} levels)")
            ax_error.plot(x_vals, error, color=color, linewidth=1,
                          alpha=0.85, label=f"INT{bits} | MSE={mse:.2e}")

        ax_signal.set_title("Quantized Signal by Bit-Width")
        ax_signal.set_xlabel("Input value")
        ax_signal.set_ylabel("Quantized value")
        ax_signal.legend(fontsize=9)
        ax_signal.grid(True, alpha=0.3)

        ax_error.set_title("Quantization Error by Bit-Width")
        ax_error.set_xlabel("Input value")
        ax_error.set_ylabel("Error (FP32 - dequant)")
        ax_error.legend(fontsize=9)
        ax_error.axhline(0, color="black", linewidth=0.5)
        ax_error.grid(True, alpha=0.3)

        plt.suptitle("Effect of Bit-Width on Quantization Error", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = self._output_dir / "bitwidth_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Bit-width comparison plot saved: %s", path)

    def plot_symmetric_vs_asymmetric(self) -> None:
        """
        Compare symmetric vs asymmetric quantization on a shifted distribution.

        Asymmetric is better when data doesn't span a symmetric range around 0.
        """
        # Distribution shifted to [0, 1] — typical for post-ReLU activations
        x_vals = np.linspace(0.0, 1.0, 500)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32)

        scale_sym, zp_sym = compute_scale_zero_point(0.0, 1.0, num_bits=8, symmetric=True)
        scale_asym, zp_asym = compute_scale_zero_point(0.0, 1.0, num_bits=8, symmetric=False)

        x_sym = fake_quantize(x_tensor, scale_sym, zp_sym, 8, signed=True).numpy()
        x_asym = fake_quantize(x_tensor, scale_asym, zp_asym, 8, signed=False).numpy()

        err_sym = x_vals - x_sym
        err_asym = x_vals - x_asym

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(x_vals, x_vals, "k--", alpha=0.5, label="FP32")
        axes[0].step(x_vals, x_sym, "b-", alpha=0.8, label=f"Symmetric (scale={scale_sym:.4f}, zp={zp_sym})")
        axes[0].step(x_vals, x_asym, "r-", alpha=0.8, label=f"Asymmetric (scale={scale_asym:.4f}, zp={zp_asym})")
        axes[0].set_title("Symmetric vs Asymmetric Quantization (post-ReLU activations)")
        axes[0].set_xlabel("Input value [0, 1]")
        axes[0].set_ylabel("Quantized value")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_vals, err_sym, "b-", alpha=0.8,
                     label=f"Symmetric | MSE={np.mean(err_sym**2):.2e}")
        axes[1].plot(x_vals, err_asym, "r-", alpha=0.8,
                     label=f"Asymmetric | MSE={np.mean(err_asym**2):.2e}")
        axes[1].set_title("Quantization Error: Symmetric vs Asymmetric")
        axes[1].set_xlabel("Input value [0, 1]")
        axes[1].set_ylabel("Error")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            "INT8: Symmetric wastes half the range for ReLU activations; Asymmetric is more efficient",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        path = self._output_dir / "symmetric_vs_asymmetric.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Symmetric vs asymmetric plot saved: %s", path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class FakeQuantizationDemo:
    """Orchestrates all fake quantization demonstrations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        out_dir = pathlib.Path(cfg["model"]["output_dir"]) / "fake_quant"
        self._visualizer = FakeQuantizationVisualizer(out_dir)

    def run(self) -> None:
        """Run all fake quantization demonstrations."""
        logger.info("=" * 60)
        logger.info("Fake Quantization Demo — Start")
        logger.info("=" * 60)

        logger.info("--- Demo 1: INT8 quantize-dequantize cycle ---")
        self._visualizer.plot_quantize_dequantize_cycle(num_bits=8)

        logger.info("--- Demo 2: Bit-width comparison (INT8 vs INT4 vs INT2) ---")
        self._visualizer.plot_bitwidth_comparison()

        logger.info("--- Demo 3: Symmetric vs Asymmetric quantization ---")
        self._visualizer.plot_symmetric_vs_asymmetric()

        # Numeric demonstration
        logger.info("--- Numeric demo: manual fake quantization ---")
        x = torch.tensor([0.1234, 0.5678, -0.3456, 0.9012, -0.7890])
        scale, zp = compute_scale_zero_point(float(x.min()), float(x.max()), 8, symmetric=True)
        x_fq = fake_quantize(x, scale, zp, 8)
        error = x - x_fq

        logger.info("Input FP32:       %s", x.tolist())
        logger.info("Scale=%.6f, ZP=%d", scale, zp)
        logger.info("INT8 integers:    %s", torch.round(x / scale).int().tolist())
        logger.info("Dequantized:      %s", [f"{v:.6f}" for v in x_fq.tolist()])
        logger.info("Rounding error:   %s", [f"{v:.6f}" for v in error.tolist()])
        logger.info("MSE:              %.2e", float(torch.mean(error ** 2)))

        logger.info("=" * 60)
        logger.info("Fake Quantization Demo — Complete")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    demo = FakeQuantizationDemo(_cfg)
    demo.run()
