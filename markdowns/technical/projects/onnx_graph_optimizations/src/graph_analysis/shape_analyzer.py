"""
shape_analyzer.py
-----------------
Propagates and prints tensor shapes through every node of an ONNX graph.

Shape propagation relies on ``onnx.shape_inference.infer_shapes``, which
annotates value_info tensors with their shapes.  This module reads those
annotations and presents them in a human-readable form, making it easy to:

  - Verify input/output shapes match expectations.
  - Identify where dynamic dimensions appear.
  - Spot shape mismatches after model modification.

Design principles (SOLID):
  - Single Responsibility : shape extraction and reporting only.
  - Interface Segregation : provides discrete methods per concern.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import onnx
import onnx.shape_inference
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("shape_analyzer")
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
# Shape Analyzer
# ---------------------------------------------------------------------------

class ShapeAnalyzer:
    """
    Extracts and reports tensor shapes from an ONNX graph after shape
    inference.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(self, model_path: Path) -> dict[str, list[Any]]:
        """
        Run shape inference and build a tensor-name → shape mapping.

        Parameters
        ----------
        model_path : Path
            ONNX model file.

        Returns
        -------
        dict[str, list]
            Mapping of tensor name to its shape (list of int / str).
        """
        self._logger.info("Running shape analysis on %s", model_path)
        model = onnx.load(str(model_path))
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph

        shape_map: dict[str, list[Any]] = {}

        # Graph-level inputs
        for vi in graph.input:
            shape_map[vi.name] = _extract_shape(vi.type)

        # Intermediate tensors (value_info)
        for vi in graph.value_info:
            shape_map[vi.name] = _extract_shape(vi.type)

        # Graph-level outputs
        for vi in graph.output:
            shape_map[vi.name] = _extract_shape(vi.type)

        # Initializers (weights)
        for init in graph.initializer:
            shape_map[init.name] = list(init.dims)

        self._logger.info("Shape map built | %d tensors resolved", len(shape_map))
        return shape_map

    def print_propagation(self, model_path: Path) -> None:
        """
        Print the shape of each node's output tensors, effectively showing
        how shapes propagate through the graph.
        """
        model = onnx.load(str(model_path))
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
        shape_map = self.analyse(model_path)

        max_nodes = (
            len(graph.node)
            if self._config["graph_analysis"]["print_all_nodes"]
            else self._config["graph_analysis"]["max_nodes_to_print"]
        )

        self._logger.info("=" * 70)
        self._logger.info("Shape propagation through graph (first %d nodes)", max_nodes)
        self._logger.info("-" * 70)

        for idx, node in enumerate(graph.node[:max_nodes]):
            input_shapes = [
                f"{inp}:{shape_map.get(inp, '?')}"
                for inp in node.input
                if inp
            ]
            output_shapes = [
                f"{out}:{shape_map.get(out, '?')}"
                for out in node.output
                if out
            ]
            self._logger.info(
                "[%3d] %-22s  in=%s  →  out=%s",
                idx,
                node.op_type,
                _compact(input_shapes),
                _compact(output_shapes),
            )

        self._logger.info("=" * 70)

    def find_dynamic_dimensions(self, model_path: Path) -> dict[str, list[str]]:
        """
        Identify tensors with symbolic (dynamic) dimensions.

        Returns
        -------
        dict[str, list[str]]
            Mapping of tensor name to list of symbolic dimension names.
        """
        model = onnx.load(str(model_path))
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph

        dynamic_tensors: dict[str, list[str]] = {}

        all_vi = list(graph.input) + list(graph.value_info) + list(graph.output)
        for vi in all_vi:
            try:
                syms = []
                for d in vi.type.tensor_type.shape.dim:
                    if d.HasField("dim_param"):
                        syms.append(d.dim_param)
                if syms:
                    dynamic_tensors[vi.name] = syms
            except Exception:
                pass

        self._logger.info("Dynamic dimensions found in %d tensors:", len(dynamic_tensors))
        for name, syms in dynamic_tensors.items():
            self._logger.info("  %-40s → %s", name, syms)

        return dynamic_tensors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_shape(type_proto: onnx.TypeProto) -> list[Any]:
    try:
        dims: list[Any] = []
        for d in type_proto.tensor_type.shape.dim:
            if d.HasField("dim_param"):
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append("?")
        return dims
    except Exception:
        return []


def _compact(items: list[str], max_len: int = 80) -> str:
    s = ", ".join(items)
    return s[:max_len] + "…" if len(s) > max_len else s


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/cnn_model.onnx"

    config = load_config(cfg_path)
    analyzer = ShapeAnalyzer(config)
    analyzer.print_propagation(Path(model_arg))
    analyzer.find_dynamic_dimensions(Path(model_arg))
