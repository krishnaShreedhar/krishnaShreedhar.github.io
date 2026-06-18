"""
constant_folder.py
------------------
Demonstrates and analyses constant folding in ONNX graphs.

Constant folding replaces sub-graphs whose inputs are all constants with
a single pre-computed constant node.  This eliminates redundant computation
at inference time (e.g., weight reshaping ops, constant arithmetic).

After ORT_ENABLE_BASIC optimisation the following patterns disappear:
  - Constant → Cast → downstream ops  (constant type-cast folded)
  - Constant → Reshape → ...          (reshape of fixed weights folded)
  - Shape → Gather → ...              (static shape queries folded)

Design principles (SOLID):
  - Single Responsibility : only analyses and reports constant folding.
  - Open/Closed           : new constant-pattern detectors can be added
                            without changing existing ones.
"""

import logging
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

    logger = logging.getLogger("constant_folder")
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
# Constant Folder Analyser
# ---------------------------------------------------------------------------

class ConstantFolder:
    """
    Analyses an ONNX graph to identify and report constant nodes and
    quantify the reduction achieved by constant folding.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    # Operator types whose outputs are statically known at compile time
    _CONSTANT_OPS = frozenset(
        ["Constant", "ConstantOfShape", "Shape", "Size", "Cast", "Reshape"]
    )

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(self, model_path: Path) -> dict[str, Any]:
        """
        Load the ONNX model and return a dictionary of constant-folding
        statistics.

        Parameters
        ----------
        model_path : Path
            ONNX file to inspect.

        Returns
        -------
        dict
            ``{total_nodes, constant_candidate_nodes, initializer_count,
               constant_op_breakdown, foldable_fraction}``
        """
        self._logger.info("Analysing constant folding candidates in %s", model_path)
        model = onnx.load(str(model_path))
        graph = model.graph

        total_nodes = len(graph.node)
        initializer_names = {init.name for init in graph.initializer}
        initializer_count = len(initializer_names)

        constant_nodes: list[onnx.NodeProto] = []
        op_breakdown: dict[str, int] = {}

        for node in graph.node:
            if self._is_constant_candidate(node, initializer_names, graph):
                constant_nodes.append(node)
                op_breakdown[node.op_type] = op_breakdown.get(node.op_type, 0) + 1

        foldable = len(constant_nodes)
        fraction = foldable / total_nodes if total_nodes else 0.0

        self._logger.info("Total nodes            : %d", total_nodes)
        self._logger.info("Initializers           : %d", initializer_count)
        self._logger.info("Constant-foldable nodes: %d  (%.1f%%)", foldable, fraction * 100)
        self._logger.info("Op breakdown           : %s", op_breakdown)

        return {
            "total_nodes": total_nodes,
            "constant_candidate_nodes": foldable,
            "initializer_count": initializer_count,
            "constant_op_breakdown": op_breakdown,
            "foldable_fraction": fraction,
        }

    def compare(self, before_path: Path, after_path: Path) -> None:
        """
        Compare two ONNX graphs (before/after constant folding) and log
        the reduction in node count and BatchNorm presence.

        Parameters
        ----------
        before_path : Path
            Original un-optimized ONNX model.
        after_path : Path
            Optimized ONNX model (post constant folding).
        """
        self._logger.info("Comparing graphs:")
        self._logger.info("  before: %s", before_path)
        self._logger.info("  after : %s", after_path)

        before = onnx.load(str(before_path))
        after = onnx.load(str(after_path))

        b_nodes = len(before.graph.node)
        a_nodes = len(after.graph.node)

        b_bn = sum(1 for n in before.graph.node if n.op_type == "BatchNormalization")
        a_bn = sum(1 for n in after.graph.node if n.op_type == "BatchNormalization")

        b_const = sum(1 for n in before.graph.node if n.op_type in self._CONSTANT_OPS)
        a_const = sum(1 for n in after.graph.node if n.op_type in self._CONSTANT_OPS)

        self._logger.info(
            "Node count        : %d → %d  (Δ %+d)", b_nodes, a_nodes, a_nodes - b_nodes
        )
        self._logger.info(
            "BatchNorm nodes   : %d → %d  (Δ %+d)", b_bn, a_bn, a_bn - b_bn
        )
        self._logger.info(
            "Constant-type ops : %d → %d  (Δ %+d)", b_const, a_const, a_const - b_const
        )

        if b_bn > a_bn:
            self._logger.info(
                "BatchNorm folding detected: %d BN nodes absorbed into Conv weights",
                b_bn - a_bn,
            )
        else:
            self._logger.info(
                "No BatchNorm folding at this optimization level "
                "(BN folding requires ORT_ENABLE_EXTENDED or higher)"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_constant_candidate(
        self,
        node: onnx.NodeProto,
        initializer_names: set[str],
        graph: onnx.GraphProto,
    ) -> bool:
        """Return True if all inputs of *node* are constants/initializers."""
        if node.op_type == "Constant":
            return True
        # All inputs must be either initializers or graph-level constants
        for inp in node.input:
            if inp == "":
                continue
            if inp not in initializer_names and not self._is_graph_constant(inp, graph):
                return False
        return node.op_type in self._CONSTANT_OPS

    @staticmethod
    def _is_graph_constant(name: str, graph: onnx.GraphProto) -> bool:
        """Check if *name* is produced by a Constant node."""
        for node in graph.node:
            if node.op_type == "Constant" and name in node.output:
                return True
        return False


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
    before_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/cnn_model.onnx"
    after_arg = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "outputs/models/optimized_ORT_ENABLE_ALL.onnx"
    )

    config = load_config(cfg_path)
    folder = ConstantFolder(config)
    folder.analyse(Path(before_arg))
    if Path(after_arg).exists():
        folder.compare(Path(before_arg), Path(after_arg))
