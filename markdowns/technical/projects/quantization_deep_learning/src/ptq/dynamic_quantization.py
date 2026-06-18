"""
Dynamic Post-Training Quantization (PTQ) demonstration.

Dynamic quantization quantizes weights statically at conversion time while
activations are quantized dynamically at runtime (per-batch, per-token).
This makes it ideal for LSTM and Linear layers where activation ranges vary.

Workflow:
  1. Build a SimpleLSTM model (FP32).
  2. Apply torch.quantization.quantize_dynamic() on LSTM + Linear layers.
  3. Compare FP32 vs INT8 model size and latency.
  4. Show which layer types were quantized.

All constants read from config.yaml.
"""

import copy
import logging
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn

_PROJ_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.utils import (
    SimpleLSTM,
    get_model_size_mb,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic PTQ pipeline
# ---------------------------------------------------------------------------

class DynamicPTQPipeline:
    """
    Demonstrates dynamic quantization on LSTM and Linear layers.

    Dynamic PTQ:
        - Weights: pre-quantized to INT8 at conversion time.
        - Activations: quantized dynamically during each forward pass.
        - No calibration data required.
        - Best for RNN/LSTM, Transformer, and FC-heavy models.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._model_cfg = cfg["model"]
        self._ptq_cfg = cfg["ptq"]

    def build_lstm_model(self) -> SimpleLSTM:
        """Create a SimpleLSTM model for dynamic quantization demo."""
        hidden_dim = self._model_cfg["hidden_dim"]
        num_classes = self._model_cfg["num_classes"]
        input_size = self._model_cfg["in_channels"] * self._model_cfg["input_height"]

        model = SimpleLSTM(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        logger.info(
            "SimpleLSTM created | input_size=%d | hidden=%d | classes=%d | size=%.3f MB",
            input_size, hidden_dim, num_classes, get_model_size_mb(model),
        )
        return model

    def apply_dynamic_quantization(self, model: SimpleLSTM) -> nn.Module:
        """
        Apply dynamic quantization to LSTM and Linear layers.

        torch.quantization.quantize_dynamic():
            - Replaces nn.LSTM with DynamicQuantizedLSTM.
            - Replaces nn.Linear with DynamicQuantizedLinear.
            - dtype=torch.qint8 means INT8 weight quantization.
            - Activations are cast to float16 or int8 per-invocation.

        Args:
            model: FP32 LSTM model.

        Returns:
            Dynamically quantized model.
        """
        quantized_model = torch.quantization.quantize_dynamic(
            copy.deepcopy(model),
            qconfig_spec={nn.LSTM, nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Dynamic quantization applied | layers: {nn.LSTM, nn.Linear}")
        logger.info(
            "INT8 dynamic model size: %.3f MB", get_model_size_mb(quantized_model)
        )
        return quantized_model

    def _make_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """
        Create a dummy LSTM input tensor of shape (batch, seq_len, input_size).

        Using input_height as sequence length, input_size = in_channels * input_width.
        """
        seq_len = self._model_cfg["input_height"]
        input_size = self._model_cfg["in_channels"] * self._model_cfg["input_height"]
        return torch.randn(batch_size, seq_len, input_size)

    def benchmark_latency(
        self,
        model: nn.Module,
        iterations: int = 200,
        warmup: int = 20,
        label: str = "model",
    ) -> dict[str, float]:
        """
        Measure single-sample inference latency.

        Args:
            model: Model to benchmark.
            iterations: Number of timed iterations.
            warmup: Number of warmup passes.
            label: Identifier for logging.

        Returns:
            Dict with avg_latency_ms and model_size_mb.
        """
        model.eval()
        dummy_input = self._make_dummy_input(batch_size=1)

        with torch.no_grad():
            for _ in range(warmup):
                model(dummy_input)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000
        size_mb = get_model_size_mb(model)

        logger.info(
            "[%s] size=%.3f MB | avg_latency=%.3f ms | throughput=%.1f inf/s",
            label, size_mb, avg_latency_ms, 1000 / avg_latency_ms,
        )
        return {"avg_latency_ms": avg_latency_ms, "model_size_mb": size_mb}

    def show_layer_types(self, model: nn.Module, label: str) -> None:
        """Log the type of each named module for comparison."""
        logger.info("--- Layer types in [%s] ---", label)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                logger.info("  %-40s %s", name, type(module).__name__)

    def run(self) -> dict[str, Any]:
        """
        Execute dynamic PTQ pipeline and return comparison results.

        Returns:
            Dict with fp32 and int8 benchmark metrics plus compression ratio.
        """
        logger.info("=" * 60)
        logger.info("Dynamic PTQ Pipeline — Start")
        logger.info("=" * 60)

        fp32_model = self.build_lstm_model()
        int8_model = self.apply_dynamic_quantization(fp32_model)

        self.show_layer_types(fp32_model, "FP32-LSTM")
        self.show_layer_types(int8_model, "INT8-Dynamic-LSTM")

        fp32_metrics = self.benchmark_latency(fp32_model, label="FP32-LSTM")
        int8_metrics = self.benchmark_latency(int8_model, label="INT8-Dynamic-LSTM")

        compression = fp32_metrics["model_size_mb"] / max(int8_metrics["model_size_mb"], 1e-9)
        speedup = fp32_metrics["avg_latency_ms"] / max(int8_metrics["avg_latency_ms"], 1e-9)

        logger.info("=" * 60)
        logger.info("Dynamic PTQ Results Summary")
        logger.info("  FP32 size:    %.3f MB", fp32_metrics["model_size_mb"])
        logger.info("  INT8 size:    %.3f MB", int8_metrics["model_size_mb"])
        logger.info("  Compression:  %.2fx", compression)
        logger.info("  Speedup:      %.2fx", speedup)
        logger.info("=" * 60)

        logger.info(
            "Key insight: Dynamic PTQ requires NO calibration data. "
            "Weights are statically quantized; activations quantized at runtime. "
            "Best for variable-length sequence models (LSTM, Transformer)."
        )

        return {
            "fp32": fp32_metrics,
            "int8": int8_metrics,
            "compression_ratio": compression,
            "speedup": speedup,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    pipeline = DynamicPTQPipeline(_cfg)
    pipeline.run()
