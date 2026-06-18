"""
graph_inspector.py
------------------
Loads and introspects an ONNX computational graph.

Provides structured inspection of:
  - Nodes      : operator type, name, inputs, outputs, attributes
  - Edges      : data flow between nodes
  - Initializers: learnable parameters (weights / biases)
  - Graph-level inputs and outputs with tensor shapes

Design principles (SOLID):
  - Single Responsibility : graph loading and structural inspection only.
  - Open/Closed           : new report sections can be added without
                            touching the core loading logic.
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

    logger = logging.getLogger("graph_inspector")
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
# Inspector
# ---------------------------------------------------------------------------

class GraphInspector:
    """
    Loads and introspects an ONNX model's computational graph.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._analysis_cfg = config["graph_analysis"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> onnx.ModelProto:
        """Load an ONNX model and run shape inference."""
        self._logger.info("Loading ONNX model: %s", model_path)
        model = onnx.load(str(model_path))
        model = onnx.shape_inference.infer_shapes(model)
        self._logger.info(
            "Model loaded | ir_version=%d | opset=%d | domain='%s'",
            model.ir_version,
            model.opset_import[0].version,
            model.opset_import[0].domain or "ai.onnx",
        )
        return model

    def inspect(self, model: onnx.ModelProto) -> dict[str, Any]:
        """
        Full graph inspection: nodes, initializers, I/O, metadata.

        Returns
        -------
        dict
            Structured summary of the graph.
        """
        graph = model.graph
        summary: dict[str, Any] = {}

        # ---- Metadata ----
        summary["model_doc"] = model.doc_string
        summary["ir_version"] = model.ir_version
        summary["opset_version"] = model.opset_import[0].version
        summary["graph_name"] = graph.name

        # ---- Inputs ----
        inputs = []
        for inp in graph.input:
            inputs.append({"name": inp.name, "shape": _shape_str(inp.type)})
        summary["inputs"] = inputs
        self._logger.info("Graph inputs (%d):", len(inputs))
        for i in inputs:
            self._logger.info("  %s  shape=%s", i["name"], i["shape"])

        # ---- Outputs ----
        outputs = []
        for out in graph.output:
            outputs.append({"name": out.name, "shape": _shape_str(out.type)})
        summary["outputs"] = outputs
        self._logger.info("Graph outputs (%d):", len(outputs))
        for o in outputs:
            self._logger.info("  %s  shape=%s", o["name"], o["shape"])

        # ---- Initializers ----
        inits = []
        total_params = 0
        for init in graph.initializer:
            numel = 1
            for d in init.dims:
                numel *= d
            total_params += numel
            inits.append(
                {
                    "name": init.name,
                    "shape": list(init.dims),
                    "dtype": onnx.TensorProto.DataType.Name(init.data_type),
                    "numel": numel,
                }
            )
        summary["initializers"] = inits
        summary["total_parameters"] = total_params
        self._logger.info(
            "Initializers: %d tensors | total parameters: %d", len(inits), total_params
        )

        # ---- Nodes ----
        nodes = self._inspect_nodes(graph)
        summary["nodes"] = nodes
        summary["node_count"] = len(nodes)

        return summary

    def print_graph_structure(self, model: onnx.ModelProto) -> None:
        """Print a readable representation of the computation graph."""
        graph = model.graph
        initializer_names = {init.name for init in graph.initializer}
        max_nodes = (
            len(graph.node)
            if self._analysis_cfg["print_all_nodes"]
            else self._analysis_cfg["max_nodes_to_print"]
        )

        self._logger.info("=" * 70)
        self._logger.info("Computation graph structure (first %d nodes)", max_nodes)
        self._logger.info("-" * 70)

        for idx, node in enumerate(graph.node[:max_nodes]):
            inputs = [i for i in node.input if i and i not in initializer_names]
            weights = [i for i in node.input if i and i in initializer_names]
            self._logger.info(
                "[%3d] %-25s | name=%-20s",
                idx,
                node.op_type,
                node.name or "(unnamed)",
            )
            self._logger.info(
                "       inputs=%s | weights=%s | outputs=%s",
                inputs or "—",
                weights or "—",
                list(node.output) or "—",
            )

        if len(graph.node) > max_nodes:
            self._logger.info(
                "… and %d more nodes (set print_all_nodes=true to see all)",
                len(graph.node) - max_nodes,
            )
        self._logger.info("=" * 70)

    def detect_optimization_opportunities(self, model: onnx.ModelProto) -> list[str]:
        """
        Scan for patterns that are amenable to ORT optimisation.

        Returns a list of human-readable opportunity descriptions.
        """
        graph = model.graph
        op_types = [n.op_type for n in graph.node]
        opportunities: list[str] = []

        # Conv → BN (foldable with ORT_ENABLE_EXTENDED)
        for i in range(len(op_types) - 1):
            if op_types[i] == "Conv" and op_types[i + 1] == "BatchNormalization":
                opportunities.append(
                    f"Node[{i}] Conv → BatchNormalization: "
                    "foldable into single Conv (ORT_ENABLE_EXTENDED)"
                )

        # Conv → BN → Relu (fully fuseable)
        for i in range(len(op_types) - 2):
            if (
                op_types[i] == "Conv"
                and op_types[i + 1] == "BatchNormalization"
                and op_types[i + 2] == "Relu"
            ):
                opportunities.append(
                    f"Node[{i}] Conv → BN → Relu: "
                    "fuseable into ConvBatchNormRelu (ORT_ENABLE_EXTENDED)"
                )

        # Standalone Relu that can become in-place
        relu_count = op_types.count("Relu")
        if relu_count > 0:
            opportunities.append(
                f"{relu_count}× Relu: may be converted to in-place (basic)"
            )

        # Dropout at inference: always identity
        dropout_count = op_types.count("Dropout")
        if dropout_count > 0:
            opportunities.append(
                f"{dropout_count}× Dropout: should be eliminated at inference "
                "(handled by export with model.eval())"
            )

        for opp in opportunities:
            self._logger.info("Opportunity: %s", opp)

        if not opportunities:
            self._logger.info("No obvious un-fused patterns detected (already optimized?)")

        return opportunities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inspect_nodes(self, graph: onnx.GraphProto) -> list[dict[str, Any]]:
        nodes = []
        for idx, node in enumerate(graph.node):
            attrs = {}
            for attr in node.attribute:
                try:
                    if attr.type == onnx.AttributeProto.INT:
                        attrs[attr.name] = attr.i
                    elif attr.type == onnx.AttributeProto.FLOAT:
                        attrs[attr.name] = attr.f
                    elif attr.type == onnx.AttributeProto.STRING:
                        attrs[attr.name] = attr.s.decode("utf-8")
                    elif attr.type == onnx.AttributeProto.INTS:
                        attrs[attr.name] = list(attr.ints)
                    elif attr.type == onnx.AttributeProto.FLOATS:
                        attrs[attr.name] = list(attr.floats)
                except Exception:
                    attrs[attr.name] = "<unparseable>"

            nodes.append(
                {
                    "index": idx,
                    "op_type": node.op_type,
                    "name": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": attrs,
                }
            )
        self._logger.info("Inspected %d nodes", len(nodes))
        return nodes


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _shape_str(type_proto: onnx.TypeProto) -> str:
    try:
        dims = []
        for d in type_proto.tensor_type.shape.dim:
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
    inspector = GraphInspector(config)
    model = inspector.load(Path(model_arg))
    summary = inspector.inspect(model)
    inspector.print_graph_structure(model)
    inspector.detect_optimization_opportunities(model)
