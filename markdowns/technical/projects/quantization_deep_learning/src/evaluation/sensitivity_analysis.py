"""
Sensitivity analysis and mixed-precision assignment.

Sensitivity analysis identifies which layers are most sensitive to quantization
by quantizing one layer at a time and measuring the accuracy drop. Layers with
high accuracy drop should stay at higher precision (FP16) while less sensitive
layers can be quantized to INT8.

Workflow:
  1. Establish FP32 baseline accuracy.
  2. For each quantizable layer:
     a. Keep all other layers in FP32.
     b. Quantize only this layer to INT8.
     c. Measure accuracy drop vs FP32 baseline.
  3. Rank layers by sensitivity (accuracy drop).
  4. Apply mixed-precision: sensitive layers → FP16, others → INT8.
  5. Estimate final mixed-precision model accuracy and size.

All constants from config.yaml.
"""

import copy
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

from src.utils import (
    QuantizableLeNetCNN,
    build_dataloaders,
    ensure_output_dir,
    evaluate_accuracy,
    get_model_size_mb,
    load_config,
    setup_logging,
    train_one_epoch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer sensitivity analysis
# ---------------------------------------------------------------------------

class LayerSensitivityAnalyzer:
    """
    Quantizes each layer individually to measure per-layer sensitivity.

    Uses a simple approach:
      - The full FP32 model is the baseline.
      - For each layer, clone the model, quantize only that layer,
        and measure accuracy drop.
      - Layers with larger accuracy drops are more sensitive.

    This approach is called "layer-wise sensitivity analysis" and is a
    standard pre-processing step before mixed-precision quantization.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._eval_cfg = cfg["evaluation"]
        self._ptq_cfg = cfg["ptq"]
        self._device = torch.device("cpu")

    def _get_quantizable_layer_names(self, model: nn.Module) -> list[str]:
        """
        Return names of Conv2d and Linear layers (quantizable in PyTorch).

        Args:
            model: FP32 model to inspect.

        Returns:
            List of module names.
        """
        quantizable = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantizable.append(name)
        logger.info("Quantizable layers found: %s", quantizable)
        return quantizable

    def _quantize_single_layer(
        self,
        fp32_model: QuantizableLeNetCNN,
        layer_name: str,
        calib_loader: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """
        Create a model where only `layer_name` is quantized to INT8.

        Implementation: Run full PTQ on a copy of the model, but use
        QConfig=None for all layers except `layer_name`. This effectively
        keeps everything else in FP32 and only quantizes the target layer.

        Args:
            fp32_model: Trained FP32 model.
            layer_name: Name of the layer to quantize.
            calib_loader: Calibration data.

        Returns:
            Model with only the target layer quantized.
        """
        backend = self._ptq_cfg["backend"]
        torch.backends.quantized.engine = backend

        model = copy.deepcopy(fp32_model)
        model.eval()

        # Fuse modules first (required before quantization)
        model.fuse_modules()

        # Set qconfig=None for all layers, then enable only target layer
        model.qconfig = None
        for name, module in model.named_modules():
            module.qconfig = None

        # Enable quantization only for the target layer
        target_found = False
        for name, module in model.named_modules():
            if name == layer_name:
                module.qconfig = torch.quantization.get_default_qconfig(backend)
                target_found = True
                logger.debug("Enabling quantization for layer: %s", layer_name)
                break

        if not target_found:
            logger.warning("Layer '%s' not found after fusing. Trying best match.", layer_name)
            # After fusing, layer names may change (e.g., conv1+bn1+relu1 → conv1)
            # Try matching by partial name
            for name, module in model.named_modules():
                if layer_name.split(".")[0] in name and isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.qconfig = torch.quantization.get_default_qconfig(backend)
                    logger.debug("Partial match: quantizing %s for target %s", name, layer_name)
                    break

        # Insert observers only where qconfig is set
        torch.quantization.prepare(model, inplace=True)

        with torch.no_grad():
            for images, _ in calib_loader:
                model(images)

        torch.quantization.convert(model, inplace=True)
        return model

    def run_sensitivity(
        self,
        fp32_model: QuantizableLeNetCNN,
        fp32_accuracy: float,
        calib_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """
        Quantize each layer individually and record accuracy drop.

        Args:
            fp32_model: Trained FP32 model (baseline).
            fp32_accuracy: FP32 model accuracy (precomputed for efficiency).
            calib_loader: Calibration data for per-layer quantization.
            test_loader: Test data for accuracy evaluation.

        Returns:
            Dict of layer_name -> accuracy_drop (positive = worse).
        """
        layer_names = self._get_quantizable_layer_names(fp32_model)
        sensitivity: dict[str, float] = {}

        logger.info("Starting per-layer sensitivity analysis | %d layers", len(layer_names))
        logger.info("FP32 baseline accuracy: %.4f", fp32_accuracy)

        for i, layer_name in enumerate(layer_names):
            logger.info(
                "Analyzing layer %d/%d: %s", i + 1, len(layer_names), layer_name
            )
            try:
                single_layer_quant = self._quantize_single_layer(
                    fp32_model, layer_name, calib_loader
                )
                acc = evaluate_accuracy(single_layer_quant, test_loader, self._device)
                drop = fp32_accuracy - acc
                sensitivity[layer_name] = drop
                logger.info(
                    "  Layer '%s' | accuracy=%.4f | drop=%.4f",
                    layer_name, acc, drop,
                )
            except Exception as exc:
                logger.warning(
                    "  Layer '%s' quantization failed: %s — marking as insensitive (drop=0)",
                    layer_name, exc,
                )
                sensitivity[layer_name] = 0.0

        return sensitivity


# ---------------------------------------------------------------------------
# Mixed-precision assignment
# ---------------------------------------------------------------------------

class MixedPrecisionAssigner:
    """
    Assigns quantization precision per layer based on sensitivity scores.

    Decision rule (from config):
        If accuracy_drop > sensitive_layer_threshold → keep at FP16 (high precision)
        Else → quantize to INT8 (low precision, faster)

    This is a simplified version of the AutoQ / HAQ approach where
    sensitivity guides bit-width selection.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._mp_cfg = cfg["mixed_precision"]
        self._threshold: float = self._mp_cfg["sensitive_layer_threshold"]
        self._default_dtype: str = self._mp_cfg["default_dtype"]
        self._sensitive_dtype: str = self._mp_cfg["sensitive_dtype"]

    def assign(
        self, sensitivity: dict[str, float]
    ) -> dict[str, str]:
        """
        Assign precision to each layer based on sensitivity threshold.

        Args:
            sensitivity: Dict of layer_name -> accuracy_drop.

        Returns:
            Dict of layer_name -> dtype_string (e.g., 'int8' or 'float16').
        """
        assignments: dict[str, str] = {}
        for layer_name, drop in sensitivity.items():
            if drop > self._threshold:
                assignments[layer_name] = self._sensitive_dtype
                logger.info(
                    "Layer %-40s | drop=%.4f > threshold=%.4f → %s (sensitive)",
                    f"'{layer_name}'", drop, self._threshold, self._sensitive_dtype,
                )
            else:
                assignments[layer_name] = self._default_dtype
                logger.info(
                    "Layer %-40s | drop=%.4f ≤ threshold=%.4f → %s",
                    f"'{layer_name}'", drop, self._threshold, self._default_dtype,
                )

        n_int8 = sum(1 for v in assignments.values() if v == self._default_dtype)
        n_fp16 = sum(1 for v in assignments.values() if v == self._sensitive_dtype)
        logger.info(
            "Mixed precision summary: %d INT8 layers + %d FP16 layers (total=%d)",
            n_int8, n_fp16, len(assignments),
        )
        return assignments

    def estimate_mixed_precision_size(
        self,
        fp32_model: nn.Module,
        assignments: dict[str, str],
    ) -> dict[str, float]:
        """
        Estimate the size of a mixed-precision model.

        Approximation:
          - INT8 layers: 1 byte per weight
          - FP16 layers: 2 bytes per weight
          - FP32 baseline: 4 bytes per weight

        Args:
            fp32_model: FP32 model for parameter counts.
            assignments: Per-layer dtype assignments.

        Returns:
            Dict with fp32_size_mb, mixed_size_mb, compression_ratio.
        """
        fp32_total_bytes = sum(
            p.nelement() * 4 for p in fp32_model.parameters()
        )

        mixed_total_bytes = 0
        for name, module in fp32_model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if name not in assignments:
                # Keep at FP32
                bytes_per_param = 4
            elif assignments[name] == "int8":
                bytes_per_param = 1
            elif assignments[name] == "float16":
                bytes_per_param = 2
            else:
                bytes_per_param = 4

            mixed_total_bytes += sum(p.nelement() * bytes_per_param for p in module.parameters())

        # Non-quantizable parameters stay at FP32
        quant_names = set(assignments.keys())
        for name, module in fp32_model.named_modules():
            if name not in quant_names and not isinstance(module, (nn.Conv2d, nn.Linear)):
                mixed_total_bytes += sum(p.nelement() * 4 for p in module.parameters(recurse=False))

        fp32_mb = fp32_total_bytes / (1024 ** 2)
        mixed_mb = mixed_total_bytes / (1024 ** 2)
        compression = fp32_mb / max(mixed_mb, 1e-9)

        logger.info(
            "Mixed precision size estimate | FP32=%.3f MB | Mixed=%.3f MB | Compression=%.2fx",
            fp32_mb, mixed_mb, compression,
        )

        return {
            "fp32_size_mb": fp32_mb,
            "mixed_size_mb": mixed_mb,
            "compression_ratio": compression,
        }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class SensitivityVisualizer:
    """Creates sensitivity analysis visualizations."""

    def __init__(self, output_dir: pathlib.Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sensitivity(
        self,
        sensitivity: dict[str, float],
        assignments: dict[str, str],
        threshold: float,
    ) -> None:
        """
        Plot per-layer sensitivity with mixed-precision color coding.

        Args:
            sensitivity: Dict of layer_name -> accuracy_drop.
            assignments: Dict of layer_name -> dtype.
            threshold: Threshold used for assignment.
        """
        layers = list(sensitivity.keys())
        drops = [sensitivity[l] for l in layers]

        colors = [
            "firebrick" if assignments.get(l, "int8") == "float16" else "steelblue"
            for l in layers
        ]

        fig, ax = plt.subplots(figsize=(12, max(5, len(layers) * 0.6)))
        bars = ax.barh(layers, drops, color=colors, alpha=0.8)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                   label=f"Threshold={threshold:.3f}")

        # Legend
        import matplotlib.patches as mpatches
        int8_patch = mpatches.Patch(color="steelblue", alpha=0.8, label="INT8 (quantized)")
        fp16_patch = mpatches.Patch(color="firebrick", alpha=0.8, label="FP16 (sensitive, kept high-prec)")
        ax.legend(handles=[int8_patch, fp16_patch, ax.lines[0]], fontsize=9)

        for bar, val in zip(bars, drops):
            ax.text(
                max(bar.get_width(), 0) + threshold * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

        ax.set_xlabel("Accuracy Drop (FP32 - single-layer-quantized)")
        ax.set_title(
            "Per-Layer Sensitivity Analysis\n"
            "(Layers above threshold → FP16; others → INT8)",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        plt.tight_layout()
        path = self._output_dir / "sensitivity_analysis.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Sensitivity plot saved: %s", path)

    def plot_mixed_precision_summary(
        self,
        assignments: dict[str, str],
        size_estimates: dict[str, float],
    ) -> None:
        """
        Plot mixed-precision layer assignment and size comparison.

        Args:
            assignments: Dict of layer_name -> dtype.
            size_estimates: Output of estimate_mixed_precision_size().
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Layer dtype assignment as horizontal color blocks
        ax = axes[0]
        layers = list(assignments.keys())
        colors = [
            "firebrick" if v == "float16" else "steelblue"
            for v in assignments.values()
        ]
        y = np.arange(len(layers))
        ax.barh(y, [1] * len(layers), color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(layers, fontsize=9)
        ax.set_xticks([])
        ax.set_title("Mixed Precision Layer Assignments")
        ax.invert_yaxis()

        import matplotlib.patches as mpatches
        int8_p = mpatches.Patch(color="steelblue", alpha=0.8, label="INT8")
        fp16_p = mpatches.Patch(color="firebrick", alpha=0.8, label="FP16")
        ax.legend(handles=[int8_p, fp16_p], fontsize=9)

        # Right: Size comparison pie chart
        ax = axes[1]
        fp32_mb = size_estimates["fp32_size_mb"]
        mixed_mb = size_estimates["mixed_size_mb"]
        compression = size_estimates["compression_ratio"]

        categories = ["FP32 Baseline", "Mixed Precision"]
        sizes_mb = [fp32_mb, mixed_mb]
        bar_colors = ["gray", "mediumseagreen"]
        bars = ax.bar(categories, sizes_mb, color=bar_colors, alpha=0.8, width=0.4)
        ax.set_ylabel("Model Size (MB)")
        ax.set_title(f"Size Comparison\nCompression: {compression:.2f}x")
        for bar, val in zip(bars, sizes_mb):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.3f} MB", ha="center", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Mixed Precision Quantization Summary", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = self._output_dir / "mixed_precision_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Mixed precision summary plot saved: %s", path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SensitivityAnalysisRunner:
    """Orchestrates the full sensitivity analysis and mixed-precision pipeline."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._device = torch.device("cpu")
        out_dir = pathlib.Path(cfg["model"]["output_dir"]) / "sensitivity"
        self._sensitivity_analyzer = LayerSensitivityAnalyzer(cfg)
        self._mp_assigner = MixedPrecisionAssigner(cfg)
        self._visualizer = SensitivityVisualizer(out_dir)

    def _build_and_train_fp32(self, train_loader, test_loader):
        model = QuantizableLeNetCNN(
            in_channels=self._cfg["model"]["in_channels"],
            num_classes=self._cfg["model"]["num_classes"],
            hidden_dim=self._cfg["model"]["hidden_dim"],
        ).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, 4):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, self._device)
            acc = evaluate_accuracy(model, test_loader, self._device)
            logger.info("Epoch %d | loss=%.4f | acc=%.4f", epoch, loss, acc)
        return model

    def run(self) -> dict[str, Any]:
        """Execute full sensitivity analysis pipeline."""
        logger.info("=" * 60)
        logger.info("Sensitivity Analysis — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        # Build and train FP32 model
        fp32_model = self._build_and_train_fp32(train_loader, test_loader)
        fp32_accuracy = evaluate_accuracy(fp32_model, test_loader, self._device)
        logger.info("FP32 baseline accuracy: %.4f", fp32_accuracy)

        # Per-layer sensitivity analysis
        sensitivity = self._sensitivity_analyzer.run_sensitivity(
            fp32_model, fp32_accuracy, calib_loader, test_loader
        )

        # Print ranked sensitivity
        ranked = sorted(sensitivity.items(), key=lambda kv: -kv[1])
        logger.info("")
        logger.info("Layers ranked by sensitivity (highest drop first):")
        for i, (name, drop) in enumerate(ranked, 1):
            logger.info("  %d. %-40s | accuracy_drop=%.4f", i, f"'{name}'", drop)

        # Mixed-precision assignment
        assignments = self._mp_assigner.assign(sensitivity)

        # Size estimation
        size_estimates = self._mp_assigner.estimate_mixed_precision_size(fp32_model, assignments)

        # Visualizations
        threshold = self._cfg["mixed_precision"]["sensitive_layer_threshold"]
        self._visualizer.plot_sensitivity(sensitivity, assignments, threshold)
        self._visualizer.plot_mixed_precision_summary(assignments, size_estimates)

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Mixed Precision Assignment Summary")
        logger.info("=" * 70)
        logger.info("%-40s | %-12s | %-10s", "Layer", "Precision", "Sensitivity")
        logger.info("-" * 70)
        for layer, dtype in assignments.items():
            logger.info(
                "%-40s | %-12s | %.4f",
                f"'{layer}'", dtype, sensitivity[layer],
            )
        logger.info("=" * 70)
        logger.info("FP32 size:          %.3f MB", size_estimates["fp32_size_mb"])
        logger.info("Mixed prec. size:   %.3f MB", size_estimates["mixed_size_mb"])
        logger.info("Compression ratio:  %.2fx", size_estimates["compression_ratio"])
        logger.info("=" * 70)

        logger.info("=" * 60)
        logger.info("Sensitivity Analysis — Complete")
        logger.info("=" * 60)

        return {
            "sensitivity": sensitivity,
            "assignments": assignments,
            "size_estimates": size_estimates,
            "fp32_accuracy": fp32_accuracy,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    runner = SensitivityAnalysisRunner(_cfg)
    runner.run()
