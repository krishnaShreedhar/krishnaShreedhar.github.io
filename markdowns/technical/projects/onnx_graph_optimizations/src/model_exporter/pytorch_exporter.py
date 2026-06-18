"""
pytorch_exporter.py
-------------------
Exports a minimal CNN (Conv → BatchNorm → ReLU → Linear) to ONNX format.

Design principles (SOLID):
  - Single Responsibility : only handles PyTorch → ONNX export.
  - Open/Closed           : new model types can be added by subclassing BaseCNN.
  - Liskov                : BaseCNN contract is fulfilled by SimpleCNN.
  - Interface Segregation : only export-related methods exposed.
  - Dependency Inversion  : configuration injected at construction time.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import onnx
import onnx.shape_inference
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pytorch_exporter")
    logger.setLevel(getattr(logging, log_cfg["level"].upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_path,
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CNN model definition
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Minimal CNN: Conv2d → BatchNorm2d → ReLU → AdaptiveAvgPool → Linear.

    This architecture is intentionally small so that ONNX export, graph
    inspection, and optimization can be demonstrated quickly without GPU
    requirements or large datasets.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),   # Conv
            nn.BatchNorm2d(32),                                       # BN
            nn.ReLU(inplace=True),                                    # ReLU
            nn.Conv2d(32, 64, kernel_size=3, padding=1),             # Conv
            nn.BatchNorm2d(64),                                       # BN
            nn.ReLU(inplace=True),                                    # ReLU
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class PyTorchExporter:
    """
    Exports a PyTorch model to ONNX and validates the exported graph.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration tree.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._model_cfg = config["pytorch_model"]
        self._export_cfg = config["model"]
        self._output_dir = Path(self._export_cfg["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_model(self) -> SimpleCNN:
        """Instantiate the CNN and set it to eval mode."""
        model = SimpleCNN(
            in_channels=self._model_cfg["in_channels"],
            num_classes=self._model_cfg["num_classes"],
            hidden_dim=self._model_cfg["hidden_dim"],
        )
        model.eval()
        self._logger.info(
            "SimpleCNN built | in_channels=%d | num_classes=%d | hidden_dim=%d",
            self._model_cfg["in_channels"],
            self._model_cfg["num_classes"],
            self._model_cfg["hidden_dim"],
        )
        total_params = sum(p.numel() for p in model.parameters())
        self._logger.info("Total parameters: %d", total_params)
        return model

    def export(self, model: nn.Module, onnx_filename: str = "cnn_model.onnx") -> Path:
        """
        Export *model* to ONNX format.

        Parameters
        ----------
        model : nn.Module
            PyTorch model in eval mode.
        onnx_filename : str
            Output filename (relative to ``model.output_dir``).

        Returns
        -------
        Path
            Absolute path to the exported ONNX file.
        """
        onnx_path = self._output_dir / onnx_filename
        h = self._model_cfg["input_height"]
        w = self._model_cfg["input_width"]
        c = self._model_cfg["in_channels"]
        bs = self._model_cfg["batch_size"]

        dummy_input = torch.randn(bs, c, h, w)
        self._logger.info(
            "Starting ONNX export | dummy_input_shape=%s | opset=%d | path=%s",
            list(dummy_input.shape),
            self._export_cfg["opset_version"],
            onnx_path,
        )

        # Build dynamic_axes mapping from config
        dynamic_axes: dict[str, dict[int, str]] = {}
        raw_axes = self._export_cfg.get("dynamic_axes", {})
        for tensor_name, axis_map in raw_axes.items():
            dynamic_axes[tensor_name] = {int(k): v for k, v in axis_map.items()}

        t0 = time.perf_counter()
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=self._export_cfg["input_names"],
            output_names=self._export_cfg["output_names"],
            opset_version=self._export_cfg["opset_version"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        self._logger.info("ONNX export completed in %.3f s | file=%s", elapsed, onnx_path)
        return onnx_path

    def validate(self, onnx_path: Path) -> onnx.ModelProto:
        """
        Load and validate an ONNX model.

        Runs ``onnx.checker.check_model`` and ``onnx.shape_inference.infer_shapes``.
        Logs opset version, node count, and I/O shapes.

        Parameters
        ----------
        onnx_path : Path
            Path to the ONNX file to validate.

        Returns
        -------
        onnx.ModelProto
            The validated (and shape-inferred) model proto.
        """
        self._logger.info("Loading ONNX model from %s", onnx_path)
        model_proto = onnx.load(str(onnx_path))

        self._logger.info("Running onnx.checker.check_model …")
        onnx.checker.check_model(model_proto)
        self._logger.info("Model check passed (no structural errors)")

        self._logger.info("Running shape inference …")
        model_proto = onnx.shape_inference.infer_shapes(model_proto)
        self._logger.info("Shape inference completed")

        # Log metadata
        opset = model_proto.opset_import[0].version
        graph = model_proto.graph
        node_count = len(graph.node)
        self._logger.info("Opset version : %d", opset)
        self._logger.info("Number of nodes: %d", node_count)

        for inp in graph.input:
            shape_str = _format_type_proto_shape(inp.type)
            self._logger.info("Input  | name=%-20s | shape=%s", inp.name, shape_str)

        for out in graph.output:
            shape_str = _format_type_proto_shape(out.type)
            self._logger.info("Output | name=%-20s | shape=%s", out.name, shape_str)

        return model_proto

    def run(self) -> Path:
        """Convenience entry point: build → export → validate."""
        model = self.build_model()
        onnx_path = self.export(model)
        self.validate(onnx_path)
        return onnx_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_type_proto_shape(type_proto: onnx.TypeProto) -> str:
    """Return a human-readable shape string from an ONNX TypeProto."""
    try:
        shape = type_proto.tensor_type.shape
        dims = []
        for d in shape.dim:
            if d.HasField("dim_param"):
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(str(d.dim_value))
            else:
                dims.append("?")
        return "[" + ", ".join(dims) + "]"
    except Exception:
        return "unknown"


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from *config_path*."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(cfg_path)
    exporter = PyTorchExporter(config)
    out = exporter.run()
    print(f"Exported ONNX model → {out}")
