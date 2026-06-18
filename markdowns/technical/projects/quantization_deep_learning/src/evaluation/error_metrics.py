"""
Quantization error metrics module.

Implements layer-by-layer error analysis between FP32 and quantized models:

  - SQNR: Signal-to-Quantization-Noise Ratio (dB)
          SQNR = 10 * log10(E[x^2] / E[(x - x_q)^2])
          Higher is better: >40 dB is excellent, <30 dB is problematic.

  - Cosine Similarity: alignment of output vector directions
          cos_sim(x, x_q) = (x · x_q) / (||x|| * ||x_q||)
          Range: [-1, 1], closer to 1.0 is better.

  - MSE: Mean Squared Error between FP32 and quantized activations.
          Lower is better.

  - SNR: Traditional signal-to-noise ratio (not log scale).

All metrics are computed per-layer by hooking into model forward passes.
All constants from config.yaml.
"""

import logging
import pathlib
import sys
from collections import defaultdict
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
    load_config,
    setup_logging,
    train_one_epoch,
    evaluate_accuracy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

class ActivationCollector:
    """
    Collects intermediate layer activations via forward hooks.

    Uses context manager pattern to automatically register/deregister hooks.
    """

    def __init__(self) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self.activations: dict[str, list[torch.Tensor]] = defaultdict(list)

    def register(self, model: nn.Module) -> "ActivationCollector":
        """Register hooks on all leaf modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf module
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
        logger.debug("Registered %d forward hooks", len(self._hooks))
        return self

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name].append(output.detach().cpu().float())
        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("All hooks removed")

    def get_concatenated(self) -> dict[str, torch.Tensor]:
        """Concatenate all collected activations per layer into single tensors."""
        return {
            name: torch.cat(tensors, dim=0)
            for name, tensors in self.activations.items()
        }


# ---------------------------------------------------------------------------
# Error metric computation
# ---------------------------------------------------------------------------

class LayerErrorMetrics:
    """
    Computes SQNR, cosine similarity, and MSE between two sets of activations.

    Designed for comparing FP32 vs quantized model layer outputs.
    """

    @staticmethod
    def sqnr(signal: torch.Tensor, noisy: torch.Tensor) -> float:
        """
        Signal-to-Quantization-Noise Ratio in dB.

        SQNR = 10 * log10( E[signal^2] / E[(signal - noisy)^2] )

        Args:
            signal: FP32 reference tensor.
            noisy: Quantized (or approximate) tensor.

        Returns:
            SQNR in dB. Returns float('inf') if noise is zero.
        """
        signal_power = float(torch.mean(signal.float() ** 2))
        noise = signal.float() - noisy.float()
        noise_power = float(torch.mean(noise ** 2))

        if noise_power < 1e-15:
            return float("inf")
        if signal_power < 1e-15:
            return 0.0

        return float(10 * np.log10(signal_power / noise_power))

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Compute cosine similarity between two flattened tensors.

        Args:
            a: First tensor (FP32 reference).
            b: Second tensor (quantized approximation).

        Returns:
            Scalar cosine similarity in [-1, 1].
        """
        a_flat = a.float().flatten()
        b_flat = b.float().flatten()
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))

    @staticmethod
    def mse(signal: torch.Tensor, noisy: torch.Tensor) -> float:
        """
        Mean Squared Error between signal and noisy tensors.

        Args:
            signal: FP32 reference.
            noisy: Quantized approximation.

        Returns:
            MSE value (lower is better).
        """
        return float(torch.mean((signal.float() - noisy.float()) ** 2))

    @staticmethod
    def max_abs_error(signal: torch.Tensor, noisy: torch.Tensor) -> float:
        """Maximum absolute error between signal and noisy tensors."""
        return float(torch.max(torch.abs(signal.float() - noisy.float())))

    def compute_all(
        self, signal: torch.Tensor, noisy: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute all error metrics between signal and noisy tensors.

        Args:
            signal: FP32 reference tensor.
            noisy: Quantized tensor.

        Returns:
            Dict with sqnr_db, cosine_similarity, mse, max_abs_error.
        """
        return {
            "sqnr_db": self.sqnr(signal, noisy),
            "cosine_similarity": self.cosine_similarity(signal, noisy),
            "mse": self.mse(signal, noisy),
            "max_abs_error": self.max_abs_error(signal, noisy),
        }


# ---------------------------------------------------------------------------
# Per-layer error analysis
# ---------------------------------------------------------------------------

class PerLayerErrorAnalyzer:
    """
    Runs per-layer error analysis comparing FP32 vs quantized model activations.

    Workflow:
      1. Register hooks on both FP32 and quantized models.
      2. Run the same calibration data through both.
      3. Compare activations layer by layer.
      4. Produce SQNR/MSE/cosine-sim plots per layer.
    """

    def __init__(self, cfg: dict[str, Any], output_dir: pathlib.Path) -> None:
        self._cfg = cfg
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = LayerErrorMetrics()

    def analyze(
        self,
        fp32_model: nn.Module,
        quant_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> dict[str, dict[str, float]]:
        """
        Collect activations from both models and compute per-layer error metrics.

        Args:
            fp32_model: Reference FP32 model.
            quant_model: Quantized model (PTQ or QAT INT8).
            data_loader: DataLoader for the analysis data.

        Returns:
            Dict of layer_name -> error metric dict.
        """
        fp32_collector = ActivationCollector()
        quant_collector = ActivationCollector()

        fp32_collector.register(fp32_model)
        quant_collector.register(quant_model)

        fp32_model.eval()
        quant_model.eval()

        logger.info("Running data through models to collect activations...")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(data_loader):
                fp32_model(images)
                quant_model(images)
                if batch_idx >= 5:  # Limit to 5 batches for speed
                    break

        fp32_collector.remove_hooks()
        quant_collector.remove_hooks()

        fp32_acts = fp32_collector.get_concatenated()
        quant_acts = quant_collector.get_concatenated()

        logger.info("FP32 layers collected: %d", len(fp32_acts))
        logger.info("Quant layers collected: %d", len(quant_acts))

        # Find common layers
        common_layers = set(fp32_acts.keys()) & set(quant_acts.keys())
        logger.info("Common layers for comparison: %d", len(common_layers))

        results: dict[str, dict[str, float]] = {}
        for name in sorted(common_layers):
            fp32_act = fp32_acts[name]
            quant_act = quant_acts[name]

            if fp32_act.shape != quant_act.shape:
                logger.warning(
                    "Shape mismatch at layer '%s': FP32=%s, Quant=%s — skipping",
                    name, fp32_act.shape, quant_act.shape,
                )
                continue

            metrics = self._metrics.compute_all(fp32_act, quant_act)
            results[name] = metrics

            logger.info(
                "Layer %-40s | SQNR=%6.2f dB | cos_sim=%.4f | MSE=%.2e",
                f"'{name}'", metrics["sqnr_db"], metrics["cosine_similarity"], metrics["mse"],
            )

        return results

    def plot_metrics(self, results: dict[str, dict[str, float]], title_suffix: str = "") -> None:
        """
        Plot SQNR, cosine similarity, and MSE per layer.

        Args:
            results: Output of analyze().
            title_suffix: Extra label for plot titles.
        """
        layers = list(results.keys())
        sqnr_vals = [results[l]["sqnr_db"] for l in layers]
        cos_vals = [results[l]["cosine_similarity"] for l in layers]
        mse_vals = [results[l]["mse"] for l in layers]

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(
            f"Per-Layer Error Metrics: FP32 vs Quantized {title_suffix}",
            fontsize=13, fontweight="bold",
        )

        x = np.arange(len(layers))

        # SQNR
        ax = axes[0]
        bars = ax.bar(x, sqnr_vals, color=["green" if v > 40 else "orange" if v > 30 else "red"
                                             for v in sqnr_vals], alpha=0.8)
        ax.axhline(40, color="green", linestyle="--", linewidth=1, label="Good (>40 dB)")
        ax.axhline(30, color="orange", linestyle="--", linewidth=1, label="Acceptable (>30 dB)")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("SQNR (dB)")
        ax.set_title("Signal-to-Quantization-Noise Ratio per Layer")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Cosine similarity
        ax = axes[1]
        bars = ax.bar(x, cos_vals, color=["green" if v > 0.99 else "orange" if v > 0.95 else "red"
                                            for v in cos_vals], alpha=0.8)
        ax.axhline(0.99, color="green", linestyle="--", linewidth=1, label="Excellent (>0.99)")
        ax.axhline(0.95, color="orange", linestyle="--", linewidth=1, label="Acceptable (>0.95)")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cosine Similarity Between FP32 and Quantized Activations")
        ax.legend(fontsize=9)
        ax.set_ylim(min(cos_vals) - 0.05, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        # MSE
        ax = axes[2]
        ax.bar(x, mse_vals, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("MSE")
        ax.set_title("Mean Squared Error per Layer")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        safe_suffix = title_suffix.replace(" ", "_").replace("/", "_")
        path = self._output_dir / f"per_layer_error_{safe_suffix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Per-layer error plot saved: %s", path)

    def print_ranked_table(self, results: dict[str, dict[str, float]]) -> None:
        """Print layers ranked from worst to best SQNR."""
        ranked = sorted(results.items(), key=lambda kv: kv[1]["sqnr_db"])

        logger.info("")
        logger.info("=" * 80)
        logger.info("Layers ranked by SQNR (worst first — most sensitive to quantization)")
        logger.info("=" * 80)
        logger.info("%-45s | %-10s | %-12s | %-12s", "Layer", "SQNR (dB)", "Cos Sim", "MSE")
        logger.info("-" * 80)
        for name, m in ranked:
            logger.info(
                "%-45s | %-10.2f | %-12.4f | %-12.2e",
                name, m["sqnr_db"], m["cosine_similarity"], m["mse"],
            )
        logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ErrorMetricsRunner:
    """Orchestrates per-layer error analysis."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._device = torch.device("cpu")
        out_dir = pathlib.Path(cfg["model"]["output_dir"]) / "error_metrics"
        self._analyzer = PerLayerErrorAnalyzer(cfg, out_dir)

    def _build_and_train_fp32(self, train_loader, test_loader):
        import copy
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

    def _quantize(self, fp32_model, calib_loader):
        import copy
        backend = self._cfg["ptq"]["backend"]
        torch.backends.quantized.engine = backend
        q_model = copy.deepcopy(fp32_model)
        q_model.eval()
        q_model.fuse_modules()
        q_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(q_model, inplace=True)
        with torch.no_grad():
            for images, _ in calib_loader:
                q_model(images)
        torch.quantization.convert(q_model, inplace=True)
        return q_model

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("Error Metrics Analysis — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        fp32_model = self._build_and_train_fp32(train_loader, test_loader)
        ptq_model = self._quantize(fp32_model, calib_loader)

        results = self._analyzer.analyze(fp32_model, ptq_model, calib_loader)
        self._analyzer.plot_metrics(results, "PTQ-INT8")
        self._analyzer.print_ranked_table(results)

        logger.info("=" * 60)
        logger.info("Error Metrics Analysis — Complete")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    runner = ErrorMetricsRunner(_cfg)
    runner.run()
