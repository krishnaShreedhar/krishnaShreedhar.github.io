"""
Quantization-Aware Training (QAT) trainer.

QAT inserts FakeQuantize nodes into the model graph during training.
The model learns to compensate for quantization error while still in FP32.
After training, the model is converted to INT8 — achieving better accuracy
than PTQ because the weights adapt to quantization noise.

Workflow:
  1. Build and optionally pre-train FP32 model.
  2. Prepare for QAT: fuse modules, set qconfig, insert FakeQuantize nodes.
  3. Fine-tune with QAT for a few epochs (STE gradients flow through FakeQuantize).
  4. Convert to INT8 for deployment.
  5. Compare: PTQ INT8 vs QAT INT8 accuracy.

All constants from config.yaml.
"""

import copy
import logging
import pathlib
import sys
from typing import Any

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
# PTQ baseline (for comparison)
# ---------------------------------------------------------------------------

def run_static_ptq_baseline(
    fp32_model: QuantizableLeNetCNN,
    calib_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    backend: str,
) -> tuple[nn.Module, float]:
    """
    Build a PTQ INT8 model from the same FP32 checkpoint for fair comparison.

    Args:
        fp32_model: Trained FP32 model.
        calib_loader: Calibration data loader.
        test_loader: Test data loader.
        backend: Quantization backend (qnnpack or fbgemm).

    Returns:
        Tuple of (int8_model, test_accuracy).
    """
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
    acc = evaluate_accuracy(ptq_model, test_loader, torch.device("cpu"))
    logger.info("PTQ baseline INT8 accuracy: %.4f", acc)
    return ptq_model, acc


# ---------------------------------------------------------------------------
# QAT trainer
# ---------------------------------------------------------------------------

class QATTrainer:
    """
    Manages the Quantization-Aware Training lifecycle.

    Steps:
        1. Pre-train FP32 model.
        2. prepare_qat(): fuse + insert FakeQuantize.
        3. Fine-tune (STE gradients).
        4. Convert to INT8.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._model_cfg = cfg["model"]
        self._qat_cfg = cfg["qat"]
        self._ptq_cfg = cfg["ptq"]
        self._device = torch.device("cpu")

    def _build_model(self) -> QuantizableLeNetCNN:
        return QuantizableLeNetCNN(
            in_channels=self._model_cfg["in_channels"],
            num_classes=self._model_cfg["num_classes"],
            hidden_dim=self._model_cfg["hidden_dim"],
        )

    def pretrain_fp32(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
    ) -> QuantizableLeNetCNN:
        """
        Pre-train FP32 model before QAT fine-tuning.

        Args:
            train_loader: Training data loader.
            test_loader: Test data loader.
            epochs: Pre-training epochs.

        Returns:
            Trained FP32 model.
        """
        model = self._build_model().to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        logger.info("FP32 pre-training | epochs=%d", epochs)
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, self._device)
            acc = evaluate_accuracy(model, test_loader, self._device)
            logger.info("Pre-train epoch %d/%d | loss=%.4f | acc=%.4f", epoch, epochs, loss, acc)

        fp32_acc = evaluate_accuracy(model, test_loader, self._device)
        logger.info("FP32 model ready | accuracy=%.4f | size=%.3f MB", fp32_acc, get_model_size_mb(model))
        return model

    def prepare_qat_model(self, fp32_model: QuantizableLeNetCNN) -> QuantizableLeNetCNN:
        """
        Prepare model for QAT by:
          1. Fusing Conv-BN-ReLU patterns.
          2. Setting qconfig with FakeQuantize observers.
          3. Inserting FakeQuantize nodes via prepare_qat().

        FakeQuantize nodes:
          - Forward: quantize → dequantize (INT8 grid, but in FP32 space)
          - Backward: Straight-Through Estimator (gradient passes through)

        Args:
            fp32_model: Trained FP32 model.

        Returns:
            QAT-prepared model (still runs in FP32, but simulates INT8).
        """
        backend = self._qat_cfg["backend"]
        torch.backends.quantized.engine = backend

        qat_model = copy.deepcopy(fp32_model)
        qat_model.train()

        # Fuse before QAT preparation
        qat_model.fuse_modules()

        # QAT qconfig uses FakeQuantize instead of MinMaxObserver
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        logger.info("QAT qconfig set | backend=%s", backend)
        logger.info("QAT activation FakeQuantize: %s", qat_model.qconfig.activation)
        logger.info("QAT weight FakeQuantize:     %s", qat_model.qconfig.weight)

        # Insert FakeQuantize nodes
        torch.quantization.prepare_qat(qat_model, inplace=True)
        logger.info("FakeQuantize nodes inserted via prepare_qat()")

        return qat_model

    def fine_tune_qat(
        self,
        qat_model: QuantizableLeNetCNN,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> QuantizableLeNetCNN:
        """
        Fine-tune the QAT model for cfg[qat][epochs] epochs.

        During fine-tuning:
          - Forward pass uses simulated INT8 values (quantize → dequantize).
          - Backward pass uses STE: gradient flows through rounding unchanged.
          - Model weights adapt to minimize loss under simulated quantization.

        Args:
            qat_model: Model with FakeQuantize nodes.
            train_loader: Training data loader.
            test_loader: Test data loader.

        Returns:
            Fine-tuned QAT model.
        """
        epochs: int = self._qat_cfg["epochs"]
        lr: float = self._qat_cfg["learning_rate"]
        wd: float = self._qat_cfg["weight_decay"]

        optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        logger.info("QAT fine-tuning | epochs=%d | lr=%.6f | wd=%.6f", epochs, lr, wd)

        for epoch in range(1, epochs + 1):
            qat_model.train()
            loss = train_one_epoch(qat_model, train_loader, optimizer, criterion, self._device)
            acc = evaluate_accuracy(qat_model, test_loader, self._device)
            scheduler.step()

            # Log FakeQuantize observer statistics after each epoch
            self._log_fake_quant_stats(qat_model, epoch)
            logger.info(
                "QAT epoch %d/%d | loss=%.4f | acc=%.4f (FP32 mode with simulated INT8)",
                epoch, epochs, loss, acc,
            )

        logger.info("QAT fine-tuning complete")
        return qat_model

    def _log_fake_quant_stats(self, model: nn.Module, epoch: int) -> None:
        """Log scale and zero_point of FakeQuantize observers for selected layers."""
        for name, module in model.named_modules():
            if isinstance(module, torch.quantization.FakeQuantize):
                if hasattr(module, "scale") and module.scale is not None:
                    logger.debug(
                        "Epoch %d | FakeQuant [%s] | scale=%.6f | zero_point=%d",
                        epoch, name,
                        module.scale.item() if module.scale.numel() == 1 else module.scale.mean().item(),
                        module.zero_point.item() if module.zero_point.numel() == 1 else int(module.zero_point.mean().item()),
                    )
                    break  # Only log first FakeQuant node to avoid spam

    def convert_to_int8(self, qat_model: QuantizableLeNetCNN) -> nn.Module:
        """
        Convert QAT model to true INT8 model for deployment.

        This removes FakeQuantize nodes and replaces operations with
        actual integer arithmetic (same as static PTQ convert step).

        Args:
            qat_model: Fine-tuned QAT model.

        Returns:
            INT8 quantized model ready for deployment.
        """
        qat_model.eval()
        int8_model = torch.quantization.convert(qat_model, inplace=False)
        logger.info(
            "QAT model converted to INT8 | size=%.3f MB", get_model_size_mb(int8_model)
        )
        return int8_model

    def run(self) -> dict[str, Any]:
        """
        Execute the full QAT pipeline with PTQ baseline comparison.

        Returns:
            Dict with fp32, ptq, and qat accuracy and size metrics.
        """
        logger.info("=" * 60)
        logger.info("QAT Training Pipeline — Start")
        logger.info("=" * 60)

        train_loader, calib_loader, test_loader = build_dataloaders(self._cfg)

        # Step 1: Pre-train FP32
        fp32_model = self.pretrain_fp32(train_loader, test_loader, epochs=3)
        fp32_acc = evaluate_accuracy(fp32_model, test_loader, self._device)

        # Step 2: PTQ baseline from same FP32 checkpoint
        ptq_int8_model, ptq_acc = run_static_ptq_baseline(
            fp32_model, calib_loader, test_loader, self._ptq_cfg["backend"]
        )

        # Step 3: Prepare QAT
        qat_model = self.prepare_qat_model(fp32_model)

        # Step 4: Fine-tune with QAT
        qat_model = self.fine_tune_qat(qat_model, train_loader, test_loader)

        # Step 5: Convert to INT8
        qat_int8_model = self.convert_to_int8(qat_model)
        qat_acc = evaluate_accuracy(qat_int8_model, test_loader, self._device)

        # Sizes
        fp32_size = get_model_size_mb(fp32_model)
        ptq_size = get_model_size_mb(ptq_int8_model)
        qat_size = get_model_size_mb(qat_int8_model)

        # Save models
        out_dir = ensure_output_dir(self._cfg)
        torch.save(qat_int8_model.state_dict(), out_dir / "int8_qat_lenet.pth")
        logger.info("QAT INT8 model saved to %s", out_dir / "int8_qat_lenet.pth")

        logger.info("=" * 60)
        logger.info("QAT vs PTQ Comparison")
        logger.info("  %-20s | %10s | %10s", "Method", "Accuracy", "Size (MB)")
        logger.info("  %-20s | %10.4f | %10.3f", "FP32", fp32_acc, fp32_size)
        logger.info("  %-20s | %10.4f | %10.3f", "PTQ INT8", ptq_acc, ptq_size)
        logger.info("  %-20s | %10.4f | %10.3f", "QAT INT8", qat_acc, qat_size)
        logger.info("")
        logger.info("  PTQ accuracy drop: %.4f", fp32_acc - ptq_acc)
        logger.info("  QAT accuracy drop: %.4f", fp32_acc - qat_acc)
        logger.info("  QAT advantage over PTQ: %.4f", ptq_acc - qat_acc if ptq_acc < qat_acc else qat_acc - ptq_acc)
        logger.info("=" * 60)

        return {
            "fp32": {"accuracy": fp32_acc, "size_mb": fp32_size},
            "ptq_int8": {"accuracy": ptq_acc, "size_mb": ptq_size},
            "qat_int8": {"accuracy": qat_acc, "size_mb": qat_size},
            "ptq_accuracy_drop": fp32_acc - ptq_acc,
            "qat_accuracy_drop": fp32_acc - qat_acc,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _config_path = _PROJ_ROOT / "config.yaml"
    _cfg = load_config(str(_config_path))
    setup_logging(_cfg)

    trainer = QATTrainer(_cfg)
    trainer.run()
