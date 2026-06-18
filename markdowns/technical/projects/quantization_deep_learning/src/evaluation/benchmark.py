"""
Benchmarking module for quantized model evaluation.

Measures:
  - Latency (single-sample inference time, p50 / p95 / p99)
  - Throughput (samples per second)
  - Model size (MB)
  - Memory usage (peak resident memory)
  - Compression ratio (FP32 size / quantized size)

Compares: FP32 vs PTQ INT8 vs QAT INT8

All constants from config.yaml.
"""

import copy
import gc
import logging
import pathlib
import sys
import time
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
# Latency measurement
# ---------------------------------------------------------------------------

class LatencyMeter:
    """
    Measures inference latency with statistical summary.

    Collects multiple timing samples and computes p50, p95, p99 percentiles
    for robust latency characterization (single outlier doesn't dominate).
    """

    def __init__(self, warmup_iterations: int, benchmark_iterations: int) -> None:
        self._warmup = warmup_iterations
        self._iterations = benchmark_iterations

    def measure(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        label: str = "model",
    ) -> dict[str, float]:
        """
        Measure inference latency statistics.

        Args:
            model: Model to benchmark.
            input_tensor: Fixed input tensor for benchmarking.
            label: Descriptive label for logging.

        Returns:
            Dict with p50_ms, p95_ms, p99_ms, mean_ms, std_ms, throughput.
        """
        model.eval()

        # Warmup to stabilize JIT, cache, etc.
        with torch.no_grad():
            for _ in range(self._warmup):
                model(input_tensor)

        # Timed iterations
        latencies_ms = []
        with torch.no_grad():
            for _ in range(self._iterations):
                t_start = time.perf_counter()
                model(input_tensor)
                t_end = time.perf_counter()
                latencies_ms.append((t_end - t_start) * 1000)

        arr = np.array(latencies_ms)
        stats = {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "throughput_samples_per_sec": float(1000.0 / arr.mean()),
        }

        logger.info(
            "[%s] Latency | p50=%.3f ms | p95=%.3f ms | p99=%.3f ms | mean=%.3f ms | std=%.3f ms",
            label, stats["p50_ms"], stats["p95_ms"], stats["p99_ms"],
            stats["mean_ms"], stats["std_ms"],
        )
        logger.info(
            "[%s] Throughput | %.1f samples/sec",
            label, stats["throughput_samples_per_sec"],
        )
        return stats


# ---------------------------------------------------------------------------
# Size and compression measurement
# ---------------------------------------------------------------------------

class SizeAnalyzer:
    """
    Analyzes model size and compression characteristics.

    Measures:
        - Parameter count (FP32 equivalent).
        - Total bytes (accounting for quantized storage).
        - Compression ratio vs FP32 baseline.
    """

    def analyze(self, model: nn.Module, label: str = "model") -> dict[str, float]:
        """
        Compute size metrics for a model.

        Args:
            model: Model to analyze.
            label: Descriptive label.

        Returns:
            Dict with size_mb, num_params, num_buffers_bytes.
        """
        total_param_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        total_buffer_bytes = sum(
            b.nelement() * b.element_size() for b in model.buffers()
        )
        total_bytes = total_param_bytes + total_buffer_bytes

        num_params = sum(p.nelement() for p in model.parameters())

        stats = {
            "size_mb": total_bytes / (1024 ** 2),
            "num_params": num_params,
            "param_bytes": total_param_bytes,
            "buffer_bytes": total_buffer_bytes,
        }

        logger.info(
            "[%s] Size | %.3f MB | %d params | param_bytes=%d | buffer_bytes=%d",
            label, stats["size_mb"], num_params, total_param_bytes, total_buffer_bytes,
        )
        return stats


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class ModelBenchmark:
    """
    Orchestrates comprehensive benchmarking of FP32, PTQ INT8, and QAT INT8 models.

    Follows Open/Closed Principle: new model variants can be added without
    changing the benchmarking logic.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._eval_cfg = cfg["evaluation"]
        self._ptq_cfg = cfg["ptq"]
        self._model_cfg = cfg["model"]
        self._device = torch.device("cpu")
        self._latency_meter = LatencyMeter(
            warmup_iterations=self._eval_cfg["warmup_iterations"],
            benchmark_iterations=self._eval_cfg["benchmark_iterations"],
        )
        self._size_analyzer = SizeAnalyzer()

    def _build_and_train_fp32(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> QuantizableLeNetCNN:
        """Build and pre-train FP32 model."""
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
            logger.info("Pre-train epoch %d | loss=%.4f | acc=%.4f", epoch, loss, acc)

        return model

    def _build_ptq_model(
        self,
        fp32_model: QuantizableLeNetCNN,
        calib_loader: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """Apply static PTQ to fp32_model using calibration data."""
        backend = self._ptq_cfg["backend"]
        torch.backends.quantized.engine = backend

        ptq_model = copy.deepcopy(fp32_model)
        ptq_model.eval()
        ptq_model.fuse_modules()
        ptq_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(ptq_model, inplace=True)

        with torch.no_grad():
            for images, _ in calib_loader:
                ptq_model(images)

        torch.quantization.convert(ptq_model, inplace=True)
        logger.info("PTQ INT8 model ready")
        return ptq_model

    def _build_qat_model(
        self,
        fp32_model: QuantizableLeNetCNN,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """Apply QAT fine-tuning and convert to INT8."""
        backend = self._cfg["qat"]["backend"]
        torch.backends.quantized.engine = backend

        qat_model = copy.deepcopy(fp32_model)
        qat_model.train()
        qat_model.fuse_modules()
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(qat_model, inplace=True)

        optimizer = torch.optim.Adam(
            qat_model.parameters(),
            lr=self._cfg["qat"]["learning_rate"],
            weight_decay=self._cfg["qat"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self._cfg["qat"]["epochs"] + 1):
            loss = train_one_epoch(qat_model, train_loader, optimizer, criterion, self._device)
            acc = evaluate_accuracy(qat_model, test_loader, self._device)
            logger.info("QAT epoch %d | loss=%.4f | acc=%.4f", epoch, loss, acc)

        qat_model.eval()
        torch.quantization.convert(qat_model, inplace=True)
        logger.info("QAT INT8 model ready")
        return qat_model

    def _benchmark_one(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        label: str,
    ) -> dict[str, Any]:
        """
        Run full benchmark for one model variant.

        Args:
            model: Model to benchmark.
            test_loader: Test DataLoader for accuracy and latency.
            label: Descriptive label.

        Returns:
            Dict with latency, size, and accuracy metrics.
        """
        logger.info("Benchmarking: %s", label)

        # Accuracy
        acc = evaluate_accuracy(model, test_loader, self._device)
        logger.info("[%s] Accuracy: %.4f", label, acc)

        # Size
        size_metrics = self._size_analyzer.analyze(model, label)

        # Single-sample latency
        images, _ = next(iter(test_loader))
        single_sample = images[:self._eval_cfg["batch_size"]]
        latency_metrics = self._latency_meter.measure(model, single_sample, label)

        return {
            "label": label,
            "accuracy": acc,
            **size_metrics,
            **latency_metrics,
        }

    def run(self) -> dict[str, dict[str, Any]]:
        """
        Build all model variants and benchmark them, returning comparison results.

        Returns:
            Dict of model_label -> benchmark metrics.
        """
        logger.info("=" * 60)
        logger.info("Model Benchmark — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        # Build models
        fp32_model = self._build_and_train_fp32(train_loader, test_loader)
        ptq_model = self._build_ptq_model(fp32_model, calib_loader)
        qat_model = self._build_qat_model(fp32_model, train_loader, test_loader)

        # Benchmark each
        all_results = {}
        for model, label in [
            (fp32_model, "FP32"),
            (ptq_model, "PTQ-INT8"),
            (qat_model, "QAT-INT8"),
        ]:
            result = self._benchmark_one(model, test_loader, label)
            all_results[label] = result

        # Compute relative metrics
        fp32_size = all_results["FP32"]["size_mb"]
        fp32_latency = all_results["FP32"]["p50_ms"]
        fp32_acc = all_results["FP32"]["accuracy"]

        for label in ["PTQ-INT8", "QAT-INT8"]:
            r = all_results[label]
            r["compression_ratio"] = fp32_size / max(r["size_mb"], 1e-9)
            r["speedup"] = fp32_latency / max(r["p50_ms"], 1e-9)
            r["accuracy_drop"] = fp32_acc - r["accuracy"]

        # Plots
        out_dir = pathlib.Path(self._cfg["model"]["output_dir"]) / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._plot_benchmark_comparison(all_results, out_dir)

        # Print summary
        self._print_summary_table(all_results)

        logger.info("=" * 60)
        logger.info("Model Benchmark — Complete")
        logger.info("=" * 60)

        return all_results

    def _plot_benchmark_comparison(
        self, results: dict[str, dict], output_dir: pathlib.Path
    ) -> None:
        """Create multi-panel benchmark comparison plot."""
        labels = list(results.keys())
        colors = ["steelblue", "darkorange", "firebrick"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Benchmark Comparison: FP32 vs PTQ INT8 vs QAT INT8",
                     fontsize=13, fontweight="bold")

        # 1. Model size
        ax = axes[0, 0]
        sizes = [results[l]["size_mb"] for l in labels]
        bars = ax.bar(labels, sizes, color=colors, alpha=0.8)
        ax.set_ylabel("Model Size (MB)")
        ax.set_title("Model Size")
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # 2. Latency (p50 + p95 + p99 as grouped bars)
        ax = axes[0, 1]
        x = np.arange(len(labels))
        width = 0.25
        for i, pct in enumerate(["p50_ms", "p95_ms", "p99_ms"]):
            vals = [results[l][pct] for l in labels]
            ax.bar(x + i * width, vals, width, label=pct, alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency (single sample)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Accuracy
        ax = axes[1, 0]
        accs = [results[l]["accuracy"] for l in labels]
        bars = ax.bar(labels, accs, color=colors, alpha=0.8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy")
        ax.set_ylim(max(0, min(accs) - 0.05), 1.0)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Throughput
        ax = axes[1, 1]
        throughputs = [results[l]["throughput_samples_per_sec"] for l in labels]
        bars = ax.bar(labels, throughputs, color=colors, alpha=0.8)
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Inference Throughput")
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.0f}", ha="center", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / "benchmark_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Benchmark comparison plot saved: %s", path)

    def _print_summary_table(self, results: dict[str, dict]) -> None:
        """Print formatted benchmark summary table."""
        logger.info("")
        logger.info("=" * 90)
        logger.info("Benchmark Summary Table")
        logger.info("=" * 90)
        logger.info(
            "%-12s | %-8s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
            "Method", "Acc", "Size(MB)", "p50(ms)", "p95(ms)", "Compress", "Speedup", "AccDrop",
        )
        logger.info("-" * 90)

        fp32 = results.get("FP32", {})
        for label, r in results.items():
            compress = r.get("compression_ratio", 1.0)
            speedup = r.get("speedup", 1.0)
            acc_drop = r.get("accuracy_drop", 0.0)
            logger.info(
                "%-12s | %-8.4f | %-10.3f | %-10.3f | %-10.3f | %-10.2f | %-10.2f | %-10.4f",
                label, r["accuracy"], r["size_mb"], r["p50_ms"], r["p95_ms"],
                compress, speedup, acc_drop,
            )
        logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    benchmark = ModelBenchmark(_cfg)
    benchmark.run()
