"""
fusion_analyzer.py
------------------
Detects operator-fusion patterns in an ONNX graph and compares
fused vs un-fused representations.

Fusion combines multiple consecutive operators into a single optimised
kernel, reducing memory bandwidth and kernel-launch overhead.  Common
patterns include:

  Conv  → BatchNorm → Relu  →  ConvBatchNormRelu  (EXTENDED)
  Matmul → Add               →  GemmFusion         (BASIC)
  LayerNorm sub-graph        →  LayerNormalization  (EXTENDED)
  Attention sub-graph        →  MultiHeadAttention  (EXTENDED, Transformer)
  Gelu sub-graph             →  FastGelu            (EXTENDED, Transformer)

Design principles (SOLID):
  - Single Responsibility : only detects and reports fusion patterns.
  - Open/Closed           : new patterns can be added as FusionPattern
                            objects without touching existing detection logic.
"""

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import onnx
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fusion_analyzer")
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
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FusionPattern:
    """Describes a multi-op sequence that can be fused."""

    name: str
    op_sequence: list[str]
    fused_op_name: str
    required_opt_level: str


@dataclass
class PatternMatch:
    """A single occurrence of a FusionPattern found in the graph."""

    pattern: FusionPattern
    start_node_index: int
    node_names: list[str]


# ---------------------------------------------------------------------------
# Known fusion patterns
# ---------------------------------------------------------------------------

KNOWN_PATTERNS: list[FusionPattern] = [
    FusionPattern(
        name="Conv+BN+Relu",
        op_sequence=["Conv", "BatchNormalization", "Relu"],
        fused_op_name="ConvBatchNormRelu (ORT fused)",
        required_opt_level="ORT_ENABLE_EXTENDED",
    ),
    FusionPattern(
        name="Conv+BN",
        op_sequence=["Conv", "BatchNormalization"],
        fused_op_name="ConvBatchNorm (ORT fused)",
        required_opt_level="ORT_ENABLE_EXTENDED",
    ),
    FusionPattern(
        name="MatMul+Add",
        op_sequence=["MatMul", "Add"],
        fused_op_name="Gemm (or FusedMatMul)",
        required_opt_level="ORT_ENABLE_BASIC",
    ),
    FusionPattern(
        name="Gelu sub-graph",
        op_sequence=["Div", "Erf", "Add", "Mul", "Mul"],
        fused_op_name="FastGelu",
        required_opt_level="ORT_ENABLE_EXTENDED",
    ),
    FusionPattern(
        name="LayerNorm sub-graph",
        op_sequence=["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"],
        fused_op_name="LayerNormalization",
        required_opt_level="ORT_ENABLE_EXTENDED",
    ),
]


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class FusionAnalyzer:
    """
    Scans an ONNX graph for known fusion-candidate patterns and
    compares fused vs un-fused graphs.

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

    def find_patterns(
        self, model_path: Path, patterns: list[FusionPattern] | None = None
    ) -> list[PatternMatch]:
        """
        Scan *model_path* for all known (or supplied) fusion patterns.

        Parameters
        ----------
        model_path : Path
            ONNX model to inspect.
        patterns : list[FusionPattern] | None
            Patterns to look for.  Defaults to ``KNOWN_PATTERNS``.

        Returns
        -------
        list[PatternMatch]
            All found matches across the graph.
        """
        if patterns is None:
            patterns = KNOWN_PATTERNS

        self._logger.info("Scanning for fusion patterns in %s", model_path)
        model = onnx.load(str(model_path))
        graph = model.graph
        op_sequence = [node.op_type for node in graph.node]

        self._logger.info(
            "Graph has %d nodes | operator sequence (first 30): %s",
            len(op_sequence),
            op_sequence[:30],
        )

        matches: list[PatternMatch] = []
        for pattern in patterns:
            found = self._scan_sequence(graph, op_sequence, pattern)
            matches.extend(found)
            self._logger.info(
                "Pattern '%-20s' | found=%d | fuses_to=%s",
                pattern.name,
                len(found),
                pattern.fused_op_name,
            )

        self._logger.info("Total fusion opportunities detected: %d", len(matches))
        return matches

    def compare_fusion(self, before_path: Path, after_path: Path) -> None:
        """
        Compare fusion state between two ONNX models (before/after optimisation).

        Parameters
        ----------
        before_path : Path
            Un-optimised model.
        after_path : Path
            Optimised model.
        """
        self._logger.info("Fusion comparison:")
        self._logger.info("  before: %s", before_path)
        self._logger.info("  after : %s", after_path)

        before = onnx.load(str(before_path))
        after = onnx.load(str(after_path))

        b_types = _count_op_types(before)
        a_types = _count_op_types(after)

        all_ops = sorted(set(b_types) | set(a_types))
        self._logger.info("%-30s %8s %8s %8s", "Operator", "Before", "After", "Delta")
        self._logger.info("-" * 60)
        for op in all_ops:
            b = b_types.get(op, 0)
            a = a_types.get(op, 0)
            delta = a - b
            self._logger.info("%-30s %8d %8d %8+d", op, b, a, delta)

        # Summarise BatchNorm folding
        bn_before = b_types.get("BatchNormalization", 0)
        bn_after = a_types.get("BatchNormalization", 0)
        if bn_before > 0 and bn_after < bn_before:
            self._logger.info(
                "BatchNorm folding: %d/%d BN nodes absorbed into Conv weights",
                bn_before - bn_after,
                bn_before,
            )

    def report_transformer_fusions(self) -> None:
        """
        Log the transformer-specific fusion patterns available in ORT
        (requires model_type-specific optimiser from onnxruntime-tools).
        """
        transformer_type = self._config["optimization"]["transformer_model_type"]
        self._logger.info(
            "Transformer-specific optimizations (model_type=%s)", transformer_type
        )
        fusions = [
            "MultiHeadAttention (QKV matmuls + softmax + projection)",
            "FastGelu (tanh-based Gelu approximation)",
            "LayerNormalization (mean/variance sub-graph)",
            "EmbedLayerNormalization (embedding + position + LayerNorm)",
            "SkipLayerNormalization (residual add + LayerNorm)",
            "RotaryEmbedding (RoPE pattern for LLMs)",
            "GemmFastGelu (Gemm + FastGelu combined)",
        ]
        for f in fusions:
            self._logger.info("  • %s", f)
        self._logger.info(
            "Enable via: onnxruntime.transformers.optimizer.optimize_model('%s')",
            transformer_type,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_sequence(
        self,
        graph: onnx.GraphProto,
        op_types: list[str],
        pattern: FusionPattern,
    ) -> list[PatternMatch]:
        """Slide a window over the op_types list to find pattern sequences."""
        seq = pattern.op_sequence
        n = len(seq)
        matches: list[PatternMatch] = []

        for i in range(len(op_types) - n + 1):
            window = op_types[i : i + n]
            if window == seq:
                node_names = [
                    graph.node[i + j].name or f"node_{i+j}" for j in range(n)
                ]
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        start_node_index=i,
                        node_names=node_names,
                    )
                )
                self._logger.debug(
                    "  Found '%s' at index %d | nodes=%s",
                    pattern.name,
                    i,
                    node_names,
                )
        return matches


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _count_op_types(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


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
    analyzer = FusionAnalyzer(config)
    matches = analyzer.find_patterns(Path(onnx_path_arg))
    analyzer.report_transformer_fusions()

    opt_path = Path("outputs/models/optimized_ORT_ENABLE_ALL.onnx")
    if opt_path.exists():
        analyzer.compare_fusion(Path(onnx_path_arg), opt_path)
