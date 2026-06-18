"""
Weight distribution analysis before and after quantization.

Demonstrates how quantization compresses weight ranges:
  - FP32: continuous Gaussian-like distribution
  - INT8: discrete levels at intervals of (scale)
  - Quantization error: difference between FP32 and dequantized weights

Produces matplotlib histogram plots saved to outputs/.
All constants read from config.yaml.
"""

import copy
import logging
import pathlib
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_PROJ_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.utils import (
    QuantizableLeNetCNN,
    build_dataloaders,
    get_model_size_mb,
    load_config,
    setup_logging,
    train_one_epoch,
    evaluate_accuracy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------

def extract_weight_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Extract all weight tensors from a model.

    For quantized models, dequantize INT8 weights back to float for comparison.

    Args:
        model: FP32 or quantized model.

    Returns:
        Dict mapping layer_name -> weight tensor (float32).
    """
    weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight
            if w.is_quantized:
                w = w.dequantize()
            weights[name] = w.detach().cpu().float()
            logger.debug("Extracted weight | layer=%s | shape=%s", name, tuple(w.shape))
    return weights


def compute_weight_statistics(weights: dict[str, torch.Tensor]) -> dict[str, dict]:
    """
    Compute descriptive statistics for each weight tensor.

    Args:
        weights: Dict of layer_name -> weight tensor.

    Returns:
        Dict of layer_name -> {mean, std, min, max, abs_max, num_params}.
    """
    stats = {}
    for name, w in weights.items():
        flat = w.flatten().numpy()
        stats[name] = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "abs_max": float(np.max(np.abs(flat))),
            "num_params": flat.size,
        }
        logger.info(
            "Weight stats [%s] | mean=%.4f | std=%.4f | range=[%.4f, %.4f]",
            name, stats[name]["mean"], stats[name]["std"],
            stats[name]["min"], stats[name]["max"],
        )
    return stats


def compute_quantization_error(
    fp32_weights: dict[str, torch.Tensor],
    quant_weights: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """
    Compute per-layer quantization error metrics.

    Metrics:
        - MSE: Mean Squared Error between FP32 and dequantized weights.
        - SQNR: Signal-to-Quantization-Noise Ratio in dB.
          SQNR = 10 * log10(E[x^2] / E[(x - x_q)^2])

    Args:
        fp32_weights: FP32 weight tensors.
        quant_weights: Dequantized INT8 weight tensors (same layer keys).

    Returns:
        Dict of layer_name -> {mse, sqnr_db}.
    """
    errors: dict[str, dict[str, float]] = {}
    for name in fp32_weights:
        if name not in quant_weights:
            continue
        fp32 = fp32_weights[name].flatten().numpy()
        quant = quant_weights[name].flatten().numpy()
        error = fp32 - quant
        mse = float(np.mean(error ** 2))
        signal_power = float(np.mean(fp32 ** 2))
        noise_power = mse if mse > 1e-15 else 1e-15
        sqnr_db = 10 * np.log10(signal_power / noise_power) if signal_power > 1e-15 else float("inf")

        errors[name] = {"mse": mse, "sqnr_db": sqnr_db}
        logger.info(
            "Quant error [%s] | MSE=%.6f | SQNR=%.2f dB", name, mse, sqnr_db
        )
    return errors


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class WeightDistributionPlotter:
    """
    Plots weight histograms before and after quantization.

    Follows Single Responsibility: only handles plotting, not model logic.
    """

    def __init__(self, output_dir: pathlib.Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot_weight_comparison(
        self,
        fp32_weights: dict[str, torch.Tensor],
        quant_weights: dict[str, torch.Tensor],
        max_layers: int = 4,
    ) -> None:
        """
        Plot side-by-side weight histograms for FP32 vs dequantized INT8.

        Args:
            fp32_weights: FP32 weight tensors per layer.
            quant_weights: Dequantized INT8 weight tensors per layer.
            max_layers: Maximum number of layers to plot.
        """
        layer_names = [k for k in fp32_weights if k in quant_weights][:max_layers]
        if not layer_names:
            logger.warning("No common layers found for comparison plot")
            return

        n_layers = len(layer_names)
        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 4 * n_layers))
        if n_layers == 1:
            axes = [axes]

        fig.suptitle("Weight Distribution: FP32 vs Dequantized INT8", fontsize=14, fontweight="bold")

        for row, name in enumerate(layer_names):
            fp32_flat = fp32_weights[name].flatten().numpy()
            quant_flat = quant_weights[name].flatten().numpy()
            error_flat = fp32_flat - quant_flat

            ax_fp32, ax_int8, ax_err = axes[row]

            # FP32 distribution
            ax_fp32.hist(fp32_flat, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
            ax_fp32.set_title(f"FP32 — {name}", fontsize=9)
            ax_fp32.set_xlabel("Weight value")
            ax_fp32.set_ylabel("Count")
            ax_fp32.axvline(fp32_flat.mean(), color="red", linestyle="--", linewidth=1.5, label="mean")
            ax_fp32.legend(fontsize=8)

            # Dequantized INT8 distribution (shows discrete quantization levels)
            ax_int8.hist(quant_flat, bins=100, color="darkorange", alpha=0.8, edgecolor="none")
            ax_int8.set_title(f"Dequantized INT8 — {name}", fontsize=9)
            ax_int8.set_xlabel("Weight value")
            ax_int8.axvline(quant_flat.mean(), color="red", linestyle="--", linewidth=1.5, label="mean")
            ax_int8.legend(fontsize=8)

            # Quantization error distribution
            ax_err.hist(error_flat, bins=100, color="firebrick", alpha=0.8, edgecolor="none")
            ax_err.set_title(f"Quantization Error — {name}", fontsize=9)
            ax_err.set_xlabel("FP32 - Dequant(INT8)")
            mse = np.mean(error_flat ** 2)
            sqnr = 10 * np.log10(np.mean(fp32_flat**2) / max(mse, 1e-15))
            ax_err.text(
                0.98, 0.95,
                f"MSE={mse:.2e}\nSQNR={sqnr:.1f} dB",
                transform=ax_err.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        out_path = self._output_dir / "weight_distribution_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Weight distribution plot saved to %s", out_path)

    def plot_sqnr_by_layer(
        self,
        errors: dict[str, dict[str, float]],
    ) -> None:
        """
        Plot SQNR (dB) per layer as a horizontal bar chart.

        Args:
            errors: Dict from compute_quantization_error().
        """
        names = list(errors.keys())
        sqnr_values = [errors[n]["sqnr_db"] for n in names]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))
        colors = ["green" if v > 40 else "orange" if v > 30 else "red" for v in sqnr_values]
        bars = ax.barh(names, sqnr_values, color=colors, alpha=0.8)
        ax.axvline(40, color="green", linestyle="--", linewidth=1, label="Good (>40 dB)")
        ax.axvline(30, color="orange", linestyle="--", linewidth=1, label="Acceptable (>30 dB)")
        ax.set_xlabel("SQNR (dB)")
        ax.set_title("Signal-to-Quantization-Noise Ratio per Layer")
        ax.legend()

        for bar, val in zip(bars, sqnr_values):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8,
            )

        plt.tight_layout()
        out_path = self._output_dir / "sqnr_per_layer.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SQNR bar chart saved to %s", out_path)


# ---------------------------------------------------------------------------
# Weight Analysis runner
# ---------------------------------------------------------------------------

class WeightAnalysisPipeline:
    """
    Orchestrates weight analysis before and after quantization.

    Single responsibility: coordinate weight extraction, statistics computation,
    error analysis, and visualization.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._model_cfg = cfg["model"]
        self._ptq_cfg = cfg["ptq"]
        self._device = torch.device("cpu")

    def _train_fp32_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> QuantizableLeNetCNN:
        """Train a fresh FP32 model for 3 epochs."""
        model = QuantizableLeNetCNN(
            in_channels=self._model_cfg["in_channels"],
            num_classes=self._model_cfg["num_classes"],
            hidden_dim=self._model_cfg["hidden_dim"],
        ).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 4):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, self._device)
            acc = evaluate_accuracy(model, test_loader, self._device)
            logger.info("Train epoch %d | loss=%.4f | acc=%.4f", epoch, loss, acc)

        return model

    def _quantize_model(
        self,
        fp32_model: QuantizableLeNetCNN,
        calib_loader: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """Apply static PTQ and return INT8 model."""
        backend = self._ptq_cfg["backend"]
        torch.backends.quantized.engine = backend

        prepared = copy.deepcopy(fp32_model)
        prepared.eval()
        prepared.fuse_modules()
        prepared.qconfig = torch.quantization.get_default_qconfig(backend)
        prepared = torch.quantization.prepare(prepared, inplace=True)

        with torch.no_grad():
            for images, _ in calib_loader:
                prepared(images)

        int8_model = torch.quantization.convert(prepared, inplace=False)
        logger.info("INT8 model ready for weight analysis")
        return int8_model

    def run(self) -> None:
        """Execute full weight analysis pipeline."""
        logger.info("=" * 60)
        logger.info("Weight Analysis Pipeline — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        fp32_model = self._train_fp32_model(train_loader, test_loader)
        int8_model = self._quantize_model(fp32_model, calib_loader)

        # Extract weights
        fp32_weights = extract_weight_tensors(fp32_model)
        quant_weights = extract_weight_tensors(int8_model)

        logger.info("FP32 layers found: %s", list(fp32_weights.keys()))
        logger.info("INT8 layers found: %s", list(quant_weights.keys()))

        # Statistics
        logger.info("--- FP32 Weight Statistics ---")
        compute_weight_statistics(fp32_weights)

        logger.info("--- INT8 Dequantized Weight Statistics ---")
        compute_weight_statistics(quant_weights)

        # Quantization error
        errors = compute_quantization_error(fp32_weights, quant_weights)

        # Visualization
        out_dir = pathlib.Path(self._cfg["model"]["output_dir"]) / "weight_analysis"
        plotter = WeightDistributionPlotter(out_dir)
        plotter.plot_weight_comparison(fp32_weights, quant_weights)
        plotter.plot_sqnr_by_layer(errors)

        logger.info("Weight analysis complete. Outputs saved to %s", out_dir)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    pipeline = WeightAnalysisPipeline(_cfg)
    pipeline.run()
