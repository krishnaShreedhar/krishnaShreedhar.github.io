"""
Static Post-Training Quantization (PTQ) demonstration.

Workflow:
  1. Train a small CNN on synthetic data (FP32).
  2. Set quantization backend and qconfig.
  3. Fuse Conv-BN-ReLU modules.
  4. Prepare model with observer insertion (torch.quantization.prepare).
  5. Run calibration data through the prepared model.
  6. Convert to INT8 (torch.quantization.convert).
  7. Compare FP32 vs INT8 model size and inference time.

All constants are read from config.yaml.
"""

import copy
import logging
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import yaml

# Allow running as script from any directory
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
# Static PTQ pipeline
# ---------------------------------------------------------------------------

class StaticPTQPipeline:
    """
    Encapsulates the complete static PTQ workflow.

    Responsibilities:
        - Build and pre-train FP32 model.
        - Fuse modules for quantization.
        - Run calibration.
        - Convert to INT8.
        - Benchmark and compare.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._model_cfg = cfg["model"]
        self._ptq_cfg = cfg["ptq"]
        self._data_cfg = cfg["data"]
        self._device = torch.device("cpu")  # PTQ static runs on CPU

    def build_fp32_model(self) -> QuantizableLeNetCNN:
        """Instantiate a fresh FP32 QuantizableLeNetCNN."""
        model = QuantizableLeNetCNN(
            in_channels=self._model_cfg["in_channels"],
            num_classes=self._model_cfg["num_classes"],
            hidden_dim=self._model_cfg["hidden_dim"],
        )
        logger.info(
            "FP32 model created | params=%.3f M", get_model_size_mb(model)
        )
        return model

    def pretrain(
        self,
        model: QuantizableLeNetCNN,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
    ) -> QuantizableLeNetCNN:
        """
        Pre-train the FP32 model for a few epochs on synthetic data.

        Args:
            model: FP32 model to train.
            train_loader: Training data loader.
            test_loader: Test data loader.
            epochs: Number of training epochs.

        Returns:
            Trained FP32 model.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.to(self._device)

        logger.info("Starting FP32 pre-training for %d epochs", epochs)
        for epoch in range(1, epochs + 1):
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, self._device
            )
            acc = evaluate_accuracy(model, test_loader, self._device)
            logger.info(
                "Epoch %d/%d | loss=%.4f | test_acc=%.4f", epoch, epochs, avg_loss, acc
            )

        final_acc = evaluate_accuracy(model, test_loader, self._device)
        logger.info("Pre-training complete | FP32 accuracy=%.4f", final_acc)
        return model

    def prepare_for_quantization(self, model: QuantizableLeNetCNN) -> nn.Module:
        """
        Fuse Conv-BN-ReLU modules and insert observers for calibration.

        Args:
            model: Trained FP32 model.

        Returns:
            Prepared model with observer hooks.
        """
        backend: str = self._ptq_cfg["backend"]
        torch.backends.quantized.engine = backend
        logger.info("Quantization backend set to '%s'", backend)

        # Deep copy to keep FP32 model intact
        prepared_model = copy.deepcopy(model)
        prepared_model.eval()

        # Step 1: Fuse Conv-BN-ReLU for correct quantization
        prepared_model.fuse_modules()
        logger.debug("Conv-BN-ReLU modules fused")

        # Step 2: Assign qconfig
        prepared_model.qconfig = torch.quantization.get_default_qconfig(backend)
        logger.debug("qconfig assigned: %s", prepared_model.qconfig)

        # Step 3: Insert observer hooks
        prepared_model = torch.quantization.prepare(prepared_model, inplace=True)
        logger.info("Model prepared with observer hooks (ready for calibration)")

        return prepared_model

    def calibrate(
        self,
        prepared_model: nn.Module,
        calib_loader: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """
        Run calibration data through the prepared model to collect activation statistics.

        Args:
            prepared_model: Model with observer hooks.
            calib_loader: DataLoader with calibration samples.

        Returns:
            Calibrated prepared model (observers have recorded statistics).
        """
        logger.info(
            "Running calibration with %d batches", len(calib_loader)
        )
        prepared_model.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calib_loader):
                prepared_model(images)
                logger.debug("Calibration batch %d processed", batch_idx)

        logger.info("Calibration complete — observers have recorded activation ranges")
        return prepared_model

    def convert_to_int8(self, calibrated_model: nn.Module) -> nn.Module:
        """
        Convert calibrated model to INT8 using recorded observer statistics.

        Args:
            calibrated_model: Calibrated prepared model.

        Returns:
            INT8 quantized model.
        """
        int8_model = torch.quantization.convert(calibrated_model, inplace=False)
        logger.info("Model converted to INT8")
        return int8_model

    def benchmark_inference(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        iterations: int = 100,
        warmup: int = 10,
        label: str = "model",
    ) -> dict[str, float]:
        """
        Measure inference latency and throughput for a model.

        Args:
            model: Model to benchmark.
            loader: DataLoader to draw input batches from.
            iterations: Number of timing iterations.
            warmup: Number of warmup iterations (not timed).
            label: Label for logging.

        Returns:
            Dict with 'avg_latency_ms', 'throughput_samples_per_sec', 'model_size_mb'.
        """
        model.eval()
        # Get a single batch for fixed benchmarking
        images, _ = next(iter(loader))
        single_image = images[:1]

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(single_image)

        # Timed iterations
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(single_image)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000
        throughput = iterations / elapsed

        size_mb = get_model_size_mb(model)

        logger.info(
            "[%s] size=%.3f MB | avg_latency=%.3f ms | throughput=%.1f samples/s",
            label, size_mb, avg_latency_ms, throughput,
        )

        return {
            "avg_latency_ms": avg_latency_ms,
            "throughput_samples_per_sec": throughput,
            "model_size_mb": size_mb,
        }

    def run(self) -> dict[str, Any]:
        """
        Execute the complete static PTQ pipeline and return comparison results.

        Returns:
            Dict with fp32 and int8 benchmark metrics.
        """
        logger.info("=" * 60)
        logger.info("Static PTQ Pipeline — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        # Build and pre-train FP32 model
        fp32_model = self.build_fp32_model()
        fp32_model = self.pretrain(fp32_model, train_loader, test_loader, epochs=3)

        fp32_acc = evaluate_accuracy(fp32_model, test_loader, self._device)
        logger.info("FP32 test accuracy: %.4f", fp32_acc)

        # Prepare, calibrate, and convert
        prepared_model = self.prepare_for_quantization(fp32_model)
        calibrated_model = self.calibrate(prepared_model, calib_loader)
        int8_model = self.convert_to_int8(calibrated_model)

        # Evaluate INT8 accuracy
        int8_acc = evaluate_accuracy(int8_model, test_loader, self._device)
        logger.info("INT8 test accuracy: %.4f | accuracy drop: %.4f", int8_acc, fp32_acc - int8_acc)

        # Benchmark both models
        fp32_metrics = self.benchmark_inference(fp32_model, test_loader, label="FP32")
        int8_metrics = self.benchmark_inference(int8_model, test_loader, label="INT8")

        # Compression ratio
        compression_ratio = fp32_metrics["model_size_mb"] / max(int8_metrics["model_size_mb"], 1e-9)
        speedup = fp32_metrics["avg_latency_ms"] / max(int8_metrics["avg_latency_ms"], 1e-9)

        logger.info(
            "Compression ratio: %.2fx | Speedup: %.2fx",
            compression_ratio, speedup,
        )

        # Save models
        out_dir = ensure_output_dir(self._cfg)
        fp32_path = out_dir / "fp32_lenet.pth"
        int8_path = out_dir / "int8_static_ptq_lenet.pth"
        torch.save(fp32_model.state_dict(), fp32_path)
        torch.save(int8_model.state_dict(), int8_path)
        logger.info("Models saved | fp32=%s | int8=%s", fp32_path, int8_path)

        results = {
            "fp32": {**fp32_metrics, "accuracy": fp32_acc},
            "int8": {**int8_metrics, "accuracy": int8_acc},
            "compression_ratio": compression_ratio,
            "speedup": speedup,
            "accuracy_drop": fp32_acc - int8_acc,
        }

        logger.info("=" * 60)
        logger.info("Static PTQ Results Summary")
        logger.info("  FP32 size:      %.3f MB", fp32_metrics["model_size_mb"])
        logger.info("  INT8 size:      %.3f MB", int8_metrics["model_size_mb"])
        logger.info("  Compression:    %.2fx", compression_ratio)
        logger.info("  Speedup:        %.2fx", speedup)
        logger.info("  FP32 accuracy:  %.4f", fp32_acc)
        logger.info("  INT8 accuracy:  %.4f", int8_acc)
        logger.info("  Accuracy drop:  %.4f", fp32_acc - int8_acc)
        logger.info("=" * 60)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    pipeline = StaticPTQPipeline(_cfg)
    results = pipeline.run()
