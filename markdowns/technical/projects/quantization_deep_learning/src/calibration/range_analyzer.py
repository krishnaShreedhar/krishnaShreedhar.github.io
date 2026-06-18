"""
Calibration range analysis and visualization.

Compares all calibration methods side-by-side on the same activation
distribution, showing:
  - Range selection for each method.
  - Quantization error (MSE) for each method's range.
  - Histogram with range indicators overlaid.
  - Scale and zero_point comparison table.

All constants from config.yaml.
"""

import logging
import pathlib
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

_PROJ_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.utils import load_config, setup_logging
from src.calibration.calibrators import (
    BaseCalibrator,
    MinMaxCalibrator,
    PercentileCalibrator,
    MSECalibrator,
    KLDivergenceCalibrator,
    MovingAverageCalibrator,
    build_calibrator_from_config,
)

logger = logging.getLogger(__name__)

# Color palette for each calibration method
METHOD_COLORS = {
    "MinMax": "steelblue",
    "Percentile": "darkorange",
    "MSE": "firebrick",
    "KL-Divergence": "green",
    "MovingAvg": "purple",
}


# ---------------------------------------------------------------------------
# Range comparison engine
# ---------------------------------------------------------------------------

class CalibrationRangeAnalyzer:
    """
    Evaluates and compares all calibration methods on a given activation tensor.

    Responsibilities:
        - Run all calibrators on the same data.
        - Compute MSE for each method's chosen range.
        - Produce comparison visualizations.
    """

    def __init__(self, cfg: dict[str, Any], output_dir: pathlib.Path) -> None:
        self._cfg = cfg
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._calib_cfg = cfg["calibration"]

    def _build_all_calibrators(self) -> dict[str, BaseCalibrator]:
        """Instantiate all calibrators from config."""
        return {
            "MinMax": MinMaxCalibrator(num_bits=8),
            "Percentile": PercentileCalibrator(
                percentile=self._calib_cfg["percentile"], num_bits=8
            ),
            "MSE": MSECalibrator(
                search_steps=self._calib_cfg["mse_search_steps"],
                alpha_range=tuple(self._calib_cfg["mse_alpha_range"]),
                num_bits=8,
            ),
            "KL-Divergence": KLDivergenceCalibrator(
                kl_bins=self._calib_cfg["kl_bins"],
                num_quantized_bins=self._calib_cfg["kl_num_quantized_bins"],
                num_bits=8,
            ),
            "MovingAvg": MovingAverageCalibrator(
                alpha=self._calib_cfg["moving_average_constant"], num_bits=8
            ),
        }

    def _compute_mse_for_range(
        self, values: np.ndarray, min_val: float, max_val: float
    ) -> float:
        """
        Compute MSE between original values and fake-quantized values for a given range.

        Args:
            values: Original FP32 activation values.
            min_val: Chosen min of quantization range.
            max_val: Chosen max of quantization range.

        Returns:
            MSE value.
        """
        abs_max = max(abs(min_val), abs(max_val))
        q_max = 127  # INT8 symmetric
        scale = abs_max / q_max if abs_max > 0 else 1.0
        clipped = np.clip(values, min_val, max_val)
        quantized = np.round(clipped / scale) * scale
        return float(np.mean((values - quantized) ** 2))

    def analyze(
        self, activations: torch.Tensor, label: str = "activations"
    ) -> dict[str, dict[str, float]]:
        """
        Run all calibrators on the given activation tensor and collect results.

        Args:
            activations: FP32 activation tensor.
            label: Descriptive label for logging.

        Returns:
            Dict of method_name -> {min, max, scale, zero_point, mse}.
        """
        logger.info("Analyzing calibration ranges for: %s", label)
        logger.info(
            "Data stats | n=%d | mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
            len(activations), float(activations.mean()), float(activations.std()),
            float(activations.min()), float(activations.max()),
        )

        calibrators = self._build_all_calibrators()
        values = activations.detach().cpu().float()
        values_np = values.numpy().flatten()

        results: dict[str, dict[str, float]] = {}

        for name, calibrator in calibrators.items():
            calibrator.collect(values)
            min_val, max_val = calibrator.compute_range()
            scale, zp = calibrator.compute_scale_zero_point(symmetric=True)
            mse = self._compute_mse_for_range(values_np, min_val, max_val)

            results[name] = {
                "min": min_val,
                "max": max_val,
                "scale": scale,
                "zero_point": zp,
                "mse": mse,
            }

            logger.info(
                "%-20s | range=[%+.4f, %+.4f] | scale=%.6f | MSE=%.2e",
                name, min_val, max_val, scale, mse,
            )

        return results

    def plot_range_comparison(
        self,
        activations: torch.Tensor,
        results: dict[str, dict[str, float]],
        label: str = "activations",
    ) -> None:
        """
        Plot histogram of activation distribution with calibration ranges overlaid.

        Args:
            activations: FP32 activation tensor.
            results: Output of analyze().
            label: Title label.
        """
        values_np = activations.detach().cpu().float().numpy().flatten()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # --- Top: histogram with range indicators ---
        ax = axes[0]
        ax.hist(
            values_np, bins=200, color="lightgray", edgecolor="none",
            alpha=0.9, label="Activation distribution", zorder=1,
        )

        y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1

        legend_patches = []
        for name, r in results.items():
            color = METHOD_COLORS.get(name, "black")
            ax.axvline(r["min"], color=color, linewidth=2, linestyle="--", alpha=0.85)
            ax.axvline(r["max"], color=color, linewidth=2, linestyle="--", alpha=0.85)
            ax.axvspan(r["min"], r["max"], alpha=0.05, color=color)
            patch = mpatches.Patch(color=color, alpha=0.7,
                                   label=f"{name}: [{r['min']:.3f}, {r['max']:.3f}]")
            legend_patches.append(patch)

        ax.set_xlabel("Activation value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            f"Calibration Range Comparison — {label}\n"
            "(dashed vertical lines show selected [min, max] range per method)",
            fontsize=12,
        )
        ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.2)

        # --- Bottom: MSE and scale comparison bar charts ---
        names = list(results.keys())
        mse_vals = [results[n]["mse"] for n in names]
        scale_vals = [results[n]["scale"] for n in names]
        colors = [METHOD_COLORS.get(n, "gray") for n in names]

        ax2 = axes[1]
        x_pos = np.arange(len(names))
        width = 0.35

        bars1 = ax2.bar(x_pos - width / 2, mse_vals, width, label="MSE", color=colors, alpha=0.7)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x_pos + width / 2, scale_vals, width, label="Scale",
                             color=colors, alpha=0.4, hatch="///")

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=15, ha="right")
        ax2.set_ylabel("MSE (lower is better)", color="black")
        ax2_twin.set_ylabel("Quantization Scale", color="gray")
        ax2.set_title("MSE and Scale per Calibration Method")

        # Add value labels
        for bar, val in zip(bars1, mse_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{val:.2e}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, scale_vals):
            ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                          f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        path = self._output_dir / f"calibration_range_comparison_{label.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Range comparison plot saved: %s", path)

    def plot_mse_vs_clipping_alpha(self, activations: torch.Tensor) -> None:
        """
        Plot MSE as a function of clipping alpha to show MSE calibrator's search.

        Helps visualize why the optimal clipping threshold is NOT always max(|x|).

        Args:
            activations: FP32 activation tensor.
        """
        values_np = activations.detach().cpu().float().numpy().flatten()
        global_abs_max = float(np.max(np.abs(values_np)))
        q_max = 127

        alphas = np.linspace(0.5, 1.0, 200)
        mse_values = []

        for alpha in alphas:
            clip_max = alpha * global_abs_max
            scale = clip_max / q_max if clip_max > 0 else 1.0
            clipped = np.clip(values_np, -clip_max, clip_max)
            quantized = np.round(clipped / scale) * scale
            mse = float(np.mean((values_np - quantized) ** 2))
            mse_values.append(mse)

        best_idx = int(np.argmin(mse_values))
        best_alpha = alphas[best_idx]
        best_mse = mse_values[best_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(alphas, mse_values, "b-", linewidth=2, label="MSE vs alpha")
        ax.axvline(best_alpha, color="red", linestyle="--", linewidth=2,
                   label=f"Optimal alpha={best_alpha:.3f} (MSE={best_mse:.2e})")
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.5, label="alpha=1.0 (Min-Max)")
        ax.fill_between(alphas, mse_values, alpha=0.1, color="blue")
        ax.set_xlabel("Clipping alpha (fraction of max(|x|))")
        ax.set_ylabel("MSE (FP32 vs dequantized INT8)")
        ax.set_title(
            "MSE Calibrator: Effect of Clipping Range on Quantization Error\n"
            "(optimal range < max(|x|) because clipping outliers reduces rounding error)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self._output_dir / "mse_vs_clipping_alpha.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("MSE vs alpha plot saved: %s", path)

    def print_summary_table(self, results: dict[str, dict[str, float]]) -> None:
        """Log a formatted comparison table of all calibrator results."""
        logger.info("")
        logger.info("=" * 75)
        logger.info("Calibration Method Comparison Table")
        logger.info("=" * 75)
        logger.info(
            "%-22s | %-10s | %-10s | %-10s | %-12s | %-10s",
            "Method", "Min", "Max", "Scale", "MSE", "ZeroPoint",
        )
        logger.info("-" * 75)
        for name, r in results.items():
            logger.info(
                "%-22s | %-10.4f | %-10.4f | %-10.6f | %-12.2e | %-10d",
                name, r["min"], r["max"], r["scale"], r["mse"], int(r["zero_point"]),
            )
        logger.info("=" * 75)

        # Rank by MSE
        sorted_by_mse = sorted(results.items(), key=lambda kv: kv[1]["mse"])
        logger.info("Ranking by MSE (lower is better):")
        for rank, (name, r) in enumerate(sorted_by_mse, 1):
            logger.info("  %d. %-22s | MSE=%.2e", rank, name, r["mse"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class RangeAnalysisRunner:
    """Orchestrates range analysis on multiple activation distributions."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        out_dir = pathlib.Path(cfg["model"]["output_dir"]) / "calibration"
        self._analyzer = CalibrationRangeAnalyzer(cfg, out_dir)

    def run(self) -> None:
        """Run range analysis on multiple synthetic distributions."""
        logger.info("=" * 60)
        logger.info("Calibration Range Analysis — Start")
        logger.info("=" * 60)

        torch.manual_seed(42)

        # Case 1: Gaussian activations with a few outliers (typical post-conv)
        logger.info("\n--- Case 1: Gaussian + outliers (typical post-conv activations) ---")
        base = torch.randn(10000) * 0.5
        outliers = torch.tensor([-3.2, 3.7, -4.0, 4.8, -5.1])
        activations_with_outliers = torch.cat([base, outliers])

        results1 = self._analyzer.analyze(activations_with_outliers, "gaussian_with_outliers")
        self._analyzer.plot_range_comparison(activations_with_outliers, results1, "gaussian_with_outliers")
        self._analyzer.print_summary_table(results1)
        self._analyzer.plot_mse_vs_clipping_alpha(activations_with_outliers)

        # Case 2: Skewed post-ReLU activations (non-negative, tail on right)
        logger.info("\n--- Case 2: Post-ReLU activations (non-negative skewed) ---")
        relu_activations = torch.relu(torch.randn(10000) * 0.3 + 0.1)
        # Add a few large activations
        relu_activations = torch.cat([relu_activations, torch.tensor([1.5, 2.0, 2.5, 3.0])])

        results2 = self._analyzer.analyze(relu_activations, "relu_activations")
        self._analyzer.plot_range_comparison(relu_activations, results2, "relu_activations")
        self._analyzer.print_summary_table(results2)

        # Case 3: Bimodal activations
        logger.info("\n--- Case 3: Bimodal activations ---")
        bimodal = torch.cat([
            torch.randn(5000) * 0.2 - 0.6,
            torch.randn(5000) * 0.2 + 0.6,
        ])
        results3 = self._analyzer.analyze(bimodal, "bimodal")
        self._analyzer.plot_range_comparison(bimodal, results3, "bimodal")
        self._analyzer.print_summary_table(results3)

        logger.info("=" * 60)
        logger.info("Calibration Range Analysis — Complete")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    runner = RangeAnalysisRunner(_cfg)
    runner.run()
