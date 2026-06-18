"""
optimization_pipeline.py
------------------------
Applies ONNX Runtime's four optimization levels to an ONNX model and
records per-level graph statistics (node counts, saved model paths).

Optimization levels (in ascending aggressiveness):
  ORT_DISABLE_ALL    – no optimizations applied
  ORT_ENABLE_BASIC   – constant folding, redundant node elimination
  ORT_ENABLE_EXTENDED– operator fusion (Conv+BN, Gelu, etc.)
  ORT_ENABLE_ALL     – layout optimizations + all above

Design principles (SOLID):
  - Single Responsibility : pipeline orchestration only.
  - Open/Closed           : new levels can be added without touching
                            existing logic.
  - Dependency Inversion  : config injected; no hard-coded paths.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import yaml


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Holds per-level optimization statistics."""

    level_name: str
    level_value: ort.GraphOptimizationLevel
    node_count: int
    optimized_model_path: Path
    elapsed_session_init_s: float
    op_type_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("optimization_pipeline")
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
# Optimization level map
# ---------------------------------------------------------------------------

_LEVEL_MAP: dict[str, ort.GraphOptimizationLevel] = {
    "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OptimizationPipeline:
    """
    Applies each configured ORT optimization level to the given ONNX model,
    saves the optimized graph to disk, and returns structured results.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._opt_cfg = config["optimization"]
        self._model_cfg = config["model"]
        self._output_dir = Path(self._model_cfg["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, onnx_model_path: Path) -> list[OptimizationResult]:
        """
        Apply all configured optimization levels sequentially.

        Parameters
        ----------
        onnx_model_path : Path
            Source ONNX model (FP32, un-optimized).

        Returns
        -------
        list[OptimizationResult]
            One result per optimization level, in config order.
        """
        self._logger.info(
            "Starting optimization pipeline | source=%s | levels=%s",
            onnx_model_path,
            self._opt_cfg["levels"],
        )

        baseline_count = self._count_nodes(onnx_model_path)
        self._logger.info("Baseline node count (raw ONNX): %d", baseline_count)

        results: list[OptimizationResult] = []
        for level_name in self._opt_cfg["levels"]:
            result = self._apply_level(onnx_model_path, level_name)
            results.append(result)
            reduction = baseline_count - result.node_count
            self._logger.info(
                "Level=%-25s | nodes=%d | reduction=%+d | init_time=%.3f s",
                level_name,
                result.node_count,
                reduction,
                result.elapsed_session_init_s,
            )

        self._log_summary(baseline_count, results)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_level(
        self, onnx_path: Path, level_name: str
    ) -> OptimizationResult:
        """Create an ORT session at the specified optimization level and
        optionally save the optimized model to disk."""

        level_value = _LEVEL_MAP[level_name]
        opt_model_path = self._output_dir / f"optimized_{level_name}.onnx"

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = level_value

        if self._opt_cfg["save_optimized"]:
            sess_opts.optimized_model_filepath = str(opt_model_path)
            self._logger.debug(
                "Optimized model will be saved to %s", opt_model_path
            )

        self._logger.info("Initializing ORT session | level=%s …", level_name)
        t0 = time.perf_counter()
        # CPUExecutionProvider is always available
        _ = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        elapsed = time.perf_counter() - t0

        # Count nodes in the optimized file (if saved), else fall back
        # to the source model node count (ORT_DISABLE_ALL case)
        if self._opt_cfg["save_optimized"] and opt_model_path.exists():
            node_count = self._count_nodes(opt_model_path)
            op_counts = self._op_type_counts(opt_model_path)
        else:
            node_count = self._count_nodes(onnx_path)
            op_counts = self._op_type_counts(onnx_path)

        return OptimizationResult(
            level_name=level_name,
            level_value=level_value,
            node_count=node_count,
            optimized_model_path=opt_model_path,
            elapsed_session_init_s=elapsed,
            op_type_counts=op_counts,
        )

    @staticmethod
    def _count_nodes(onnx_path: Path) -> int:
        model = onnx.load(str(onnx_path))
        return len(model.graph.node)

    @staticmethod
    def _op_type_counts(onnx_path: Path) -> dict[str, int]:
        model = onnx.load(str(onnx_path))
        counts: dict[str, int] = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts

    def _log_summary(
        self, baseline: int, results: list[OptimizationResult]
    ) -> None:
        self._logger.info("=" * 70)
        self._logger.info("Optimization pipeline summary")
        self._logger.info("%-30s %8s %10s", "Level", "Nodes", "BatchNorm")
        self._logger.info("-" * 70)
        for r in results:
            bn_count = r.op_type_counts.get("BatchNormalization", 0)
            self._logger.info(
                "%-30s %8d %10d", r.level_name, r.node_count, bn_count
            )
        self._logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    onnx_path_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/cnn_model.onnx"

    config = load_config(cfg_path)
    pipeline = OptimizationPipeline(config)
    pipeline.run(Path(onnx_path_arg))
