"""
Straight-Through Estimator (STE) demonstration.

The STE is the key that makes QAT work. Because the round() function has
zero gradient almost everywhere (gradient = 0 at non-integers, undefined
at integers), naive backprop through quantization would kill all gradients.

STE approximation (Bengio et al., 2013):
    Forward:  x_q = quantize(x)       (actual discrete rounding)
    Backward: ∂L/∂x ≈ ∂L/∂x_q        (gradient passes through as if identity)

Clipped STE (recommended):
    Backward: ∂L/∂x = ∂L/∂x_q  if  quant_min ≤ x ≤ quant_max
              ∂L/∂x = 0          otherwise (clipped values get no gradient)

This module:
  1. Demonstrates STE numerically — shows gradient flowing through round().
  2. Compares vanilla round() (zero gradient) vs STE (gradient passes through).
  3. Shows a toy gradient descent step with and without STE.
  4. Visualizes STE gradient approximation.

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
from src.qat.fake_quantization import FakeQuantizeFunction, compute_scale_zero_point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STE gradient analysis
# ---------------------------------------------------------------------------

class STEGradientAnalyzer:
    """
    Analyzes and visualizes the Straight-Through Estimator gradient behavior.

    Demonstrates:
        - Why round() breaks backprop (zero/undefined gradient).
        - How STE approximates gradient as identity within range.
        - Gradient flow comparison: no STE vs with STE.
    """

    def __init__(self, output_dir: pathlib.Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def demonstrate_round_gradient_problem(self) -> None:
        """
        Show that torch.round() has zero gradient — killing backprop.

        Without STE, no gradient flows through quantization, so the model
        cannot learn to accommodate quantization error.
        """
        logger.info("--- Demonstrating gradient problem with plain round() ---")

        x = torch.tensor([0.3, 0.7, 1.2, 1.8, 2.1], requires_grad=True)

        # Plain round: gradient is zero almost everywhere
        y_round = torch.round(x)
        loss_round = y_round.sum()

        # Gradient will be NaN or zero due to round's zero derivative
        try:
            loss_round.backward()
            grad_round = x.grad.clone()
            logger.info("round() gradient: %s", grad_round.tolist())
            logger.info(
                "All gradients zero (or near-zero)? %s",
                torch.all(torch.abs(grad_round) < 1e-6).item(),
            )
        except Exception as exc:
            logger.warning("round() gradient computation failed: %s", exc)
            grad_round = torch.zeros_like(x)

        x_ste = torch.tensor([0.3, 0.7, 1.2, 1.8, 2.1], requires_grad=True)

        # STE: forward uses round, backward passes gradient through
        scale = 1.0
        zp = 0
        y_ste = FakeQuantizeFunction.apply(x_ste, scale, zp, -128, 127)
        loss_ste = y_ste.sum()
        loss_ste.backward()
        grad_ste = x_ste.grad.clone()

        logger.info("Input values:      %s", x_ste.tolist())
        logger.info("After round():     %s", y_ste.detach().tolist())
        logger.info("STE gradient:      %s", grad_ste.tolist())
        logger.info(
            "STE gradients non-zero? %s",
            torch.any(torch.abs(grad_ste) > 1e-6).item(),
        )
        logger.info(
            "Key: STE gradient = 1.0 for in-range inputs (gradient passes through unchanged)"
        )

    def demonstrate_gradient_flow_comparison(self) -> None:
        """
        Compare parameter update with vs without STE in a tiny network.

        Shows that a linear layer after fake quantization receives useful
        gradients via STE, whereas round() would produce no learning.
        """
        logger.info("--- Gradient flow: plain round() vs STE ---")

        torch.manual_seed(42)

        class TinyNetNoSTE(nn.Module):
            """Demonstrates broken gradient flow through plain round."""
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 1, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc(x)
                # Plain round — zero gradient
                x = torch.round(x * 127) / 127
                return x

        class TinyNetWithSTE(nn.Module):
            """Demonstrates working gradient flow through STE fake quant."""
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 1, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc(x)
                # STE fake quantization
                scale = float(x.abs().max().detach() / 127 + 1e-8)
                x = FakeQuantizeFunction.apply(x, scale, 0, -128, 127)
                return x

        inp = torch.randn(8, 4)
        target = torch.randn(8, 1)

        net_no_ste = TinyNetNoSTE()
        net_ste = TinyNetWithSTE()

        # Initialize with same weights for fair comparison
        with torch.no_grad():
            net_ste.fc.weight.copy_(net_no_ste.fc.weight)

        criterion = nn.MSELoss()

        # One backward pass for each
        out_no_ste = net_no_ste(inp)
        loss_no_ste = criterion(out_no_ste, target)
        loss_no_ste.backward()
        grad_no_ste = net_no_ste.fc.weight.grad

        out_ste = net_ste(inp)
        loss_ste = criterion(out_ste, target)
        loss_ste.backward()
        grad_ste = net_ste.fc.weight.grad

        logger.info("No STE: weight gradient norm = %.6f", float(grad_no_ste.norm()) if grad_no_ste is not None else 0.0)
        logger.info("With STE: weight gradient norm = %.6f", float(grad_ste.norm()) if grad_ste is not None else 0.0)
        logger.info(
            "Result: STE produces %.1fx larger gradient magnitude",
            float(grad_ste.norm() / (grad_no_ste.norm() + 1e-10)) if grad_no_ste is not None else float("inf"),
        )

    def plot_ste_gradient_visualization(self) -> None:
        """
        Plot the STE gradient approximation vs true round() gradient.

        Shows the piecewise-constant true gradient of round() (all zeros)
        vs the STE approximation (identity within range).
        """
        x_vals = np.linspace(-2.5, 2.5, 1000)

        # True gradient of round(): 0 everywhere (Dirac deltas at integers, practically 0)
        true_grad = np.zeros_like(x_vals)

        # STE gradient (clipped): 1 in range, 0 outside
        quant_min, quant_max = -127, 127
        scale = 1.0  # scale=1 means x values correspond directly to integer grid
        x_int = x_vals / scale
        ste_grad = np.where((x_int >= quant_min) & (x_int <= quant_max), 1.0, 0.0)

        # Quantized (rounded) values for reference
        x_quantized = np.round(np.clip(x_vals / scale, quant_min, quant_max)) * scale

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Forward pass: input vs quantized
        ax = axes[0, 0]
        ax.plot(x_vals, x_vals, "b-", linewidth=2, alpha=0.7, label="Identity (FP32)")
        ax.step(x_vals, x_quantized, "r-", linewidth=2, alpha=0.8, label="round(x) — INT8 grid")
        ax.set_title("Forward Pass: FP32 vs Quantized (scale=1.0)")
        ax.set_xlabel("x (input)")
        ax.set_ylabel("y (output)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. True gradient of round()
        ax = axes[0, 1]
        ax.plot(x_vals, true_grad, "r-", linewidth=2)
        ax.set_ylim(-0.2, 1.5)
        ax.set_title("True Gradient of round(x) — almost everywhere zero")
        ax.set_xlabel("x")
        ax.set_ylabel("d(round(x))/dx")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.7, "Gradient = 0 everywhere\n(except undefined at integers)\n→ No learning possible!",
                transform=ax.transAxes, ha="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        # 3. STE gradient approximation
        ax = axes[1, 0]
        ax.plot(x_vals, ste_grad, "g-", linewidth=2.5)
        ax.fill_between(x_vals, 0, ste_grad, where=ste_grad > 0, alpha=0.2, color="green")
        ax.set_ylim(-0.2, 1.5)
        ax.set_title("STE Gradient Approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("STE: d(round(x))/dx ≈ 1 (in range)")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.7, "STE approximates gradient as 1\nwithin quantization range\n→ Learning proceeds normally!",
                transform=ax.transAxes, ha="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

        # 4. Conceptual: loss landscape effect
        ax = axes[1, 1]
        # Simulate how STE smooths the quantized loss landscape
        w_vals = np.linspace(-1.5, 1.5, 300)
        # Loss with quantization (jagged due to discrete steps)
        loss_quantized = (np.round(w_vals * 4) / 4 - 0.7) ** 2  # target=0.7
        # Loss in FP32 (smooth parabola)
        loss_fp32 = (w_vals - 0.7) ** 2
        # STE-guided loss (smooth gradient but quantized forward values)
        loss_ste_guided = (np.round(w_vals * 4) / 4 - 0.7) ** 2

        ax.plot(w_vals, loss_fp32, "b-", linewidth=2, alpha=0.7, label="FP32 loss (smooth)")
        ax.plot(w_vals, loss_quantized, "r-", linewidth=1.5, alpha=0.8, label="Quantized loss (jagged)")
        ax.axvline(0.7, color="green", linestyle="--", linewidth=1.5, label="Optimal w=0.7")
        ax.set_title("Loss Landscape: FP32 vs Quantized\n(STE uses FP32 gradient on quantized loss)")
        ax.set_xlabel("Weight value w")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Straight-Through Estimator (STE): Enabling Gradient Flow Through Quantization",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        path = self._output_dir / "ste_gradient_visualization.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("STE visualization saved: %s", path)

    def demonstrate_ste_gradient_math(self) -> None:
        """
        Show STE gradient math numerically with explicit forward/backward values.
        """
        logger.info("--- STE Mathematical Demonstration ---")
        logger.info("Quantization: x → round(x/scale) * scale (fake quant, scale=0.1)")

        scale = 0.1
        x_vals = torch.tensor(
            [0.14, 0.27, 0.51, 0.73, -0.12, -0.35],
            requires_grad=True,
            dtype=torch.float32,
        )

        # Forward through STE fake quant
        x_q = FakeQuantizeFunction.apply(x_vals, scale, 0, -128, 127)

        # Simulate downstream loss: squared error to a target
        target = torch.zeros_like(x_q)
        loss = ((x_q - target) ** 2).sum()

        # Backward
        loss.backward()
        grads = x_vals.grad

        logger.info("%-15s | %-15s | %-15s | %-15s | %-15s", "x (FP32)", "x_q (fake)", "error", "dL/dx_q", "STE dL/dx")
        logger.info("-" * 80)
        for i in range(len(x_vals)):
            x_i = x_vals[i].item()
            xq_i = x_q[i].item()
            err_i = xq_i - target[i].item()
            dl_dxq = 2 * err_i  # dL/dx_q for MSE loss
            dl_dx = grads[i].item() if grads is not None else float("nan")
            logger.info(
                "%-15.4f | %-15.4f | %-15.4f | %-15.4f | %-15.4f",
                x_i, xq_i, err_i, dl_dxq, dl_dx,
            )

        logger.info("")
        logger.info("Key insight: STE sets dL/dx = dL/dx_q (gradient passes through)")
        logger.info("This allows the optimizer to update x even though forward used round()")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class STEDemo:
    """Orchestrates all STE demonstrations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        out_dir = pathlib.Path(cfg["model"]["output_dir"]) / "ste_demo"
        self._analyzer = STEGradientAnalyzer(out_dir)

    def run(self) -> None:
        """Run all STE demonstrations."""
        logger.info("=" * 60)
        logger.info("STE (Straight-Through Estimator) Demo — Start")
        logger.info("=" * 60)

        self._analyzer.demonstrate_round_gradient_problem()
        self._analyzer.demonstrate_gradient_flow_comparison()
        self._analyzer.plot_ste_gradient_visualization()
        self._analyzer.demonstrate_ste_gradient_math()

        logger.info("=" * 60)
        logger.info("STE Demo — Complete")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    demo = STEDemo(_cfg)
    demo.run()
