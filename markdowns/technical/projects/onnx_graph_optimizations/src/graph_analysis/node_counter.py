"""
node_counter.py
---------------
Counts and visualises the distribution of operator types in an ONNX graph.

Provides:
  - Per-operator-type node counts.
  - Sorted frequency table.
  - Matplotlib bar chart of operator distribution (saved to disk).
  - Comparison across multiple models (before/after optimisation).

Design principles (SOLID):
  - Single Responsibility : counting and visualisation only.
  - Open/Closed           : new chart styles can be added without altering
                            the counting logic.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import onnx
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("node_counter")
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
# Node Counter
# ---------------------------------------------------------------------------

class NodeCounter:
    """
    Counts operator occurrences in an ONNX graph and produces charts.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._output_dir = Path(self._config["model"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def count(self, model_path: Path) -> dict[str, int]:
        """
        Count operator types in the graph.

        Parameters
        ----------
        model_path : Path
            ONNX model to analyse.

        Returns
        -------
        dict[str, int]
            Mapping of operator type → count, sorted descending.
        """
        model = onnx.load(str(model_path))
        counts: dict[str, int] = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1

        # Sort descending
        counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
        total = sum(counts.values())

        self._logger.info("Operator distribution in %s (total=%d):", model_path.name, total)
        self._logger.info("%-30s %8s %8s", "Operator", "Count", "Pct%")
        self._logger.info("-" * 50)
        for op, cnt in counts.items():
            self._logger.info("%-30s %8d %7.1f%%", op, cnt, 100.0 * cnt / total)

        return counts

    def plot_distribution(
        self,
        counts: dict[str, int],
        title: str = "Operator Distribution",
        filename: str = "operator_distribution.png",
    ) -> Path:
        """
        Plot a horizontal bar chart of operator counts and save to disk.

        Parameters
        ----------
        counts : dict[str, int]
            Output of ``count()``.
        title : str
            Chart title.
        filename : str
            Output image filename (relative to model output_dir).

        Returns
        -------
        Path
            Absolute path to the saved figure.
        """
        out_path = self._output_dir / filename

        ops = list(counts.keys())
        vals = list(counts.values())

        fig, ax = plt.subplots(figsize=(10, max(4, len(ops) * 0.5)))
        bars = ax.barh(ops[::-1], vals[::-1], color="steelblue", edgecolor="white")

        for bar, val in zip(bars, vals[::-1]):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Node Count")
        ax.set_title(title)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)

        self._logger.info("Operator distribution chart saved → %s", out_path)
        return out_path

    def compare_counts(
        self,
        models: dict[str, Path],
        chart_filename: str = "optimization_comparison.png",
    ) -> Path:
        """
        Compare node counts across multiple models (e.g., per optimisation
        level) and produce a grouped bar chart.

        Parameters
        ----------
        models : dict[str, Path]
            Mapping of label → ONNX model path.
        chart_filename : str
            Output filename.

        Returns
        -------
        Path
            Absolute path to the saved figure.
        """
        all_counts: dict[str, dict[str, int]] = {}
        total_counts: dict[str, int] = {}

        for label, path in models.items():
            if not path.exists():
                self._logger.warning("Model not found, skipping: %s", path)
                continue
            cnts = self.count(path)
            all_counts[label] = cnts
            total_counts[label] = sum(cnts.values())

        if not total_counts:
            self._logger.error("No valid models to compare.")
            raise ValueError("No valid models provided for comparison.")

        self._logger.info("Total node counts across models:")
        for label, total in total_counts.items():
            self._logger.info("  %-30s : %d nodes", label, total)

        # Chart: total node counts per optimisation level
        out_path = self._output_dir / chart_filename
        labels = list(total_counts.keys())
        values = list(total_counts.values())
        colors = plt.cm.Blues_r(  # type: ignore[attr-defined]
            [i / max(len(labels), 1) for i in range(len(labels))]
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel("Total Node Count")
        ax.set_title("Node Count Reduction Across Optimization Levels")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)

        self._logger.info("Comparison chart saved → %s", out_path)
        return out_path


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
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/cnn_model.onnx"

    config = load_config(cfg_path)
    counter = NodeCounter(config)
    counts = counter.count(Path(model_arg))
    counter.plot_distribution(counts, title="Baseline Model - Operator Distribution")

    # Multi-model comparison
    output_dir = Path(config["model"]["output_dir"])
    models_to_compare = {
        label: output_dir / f"optimized_{label}.onnx"
        for label in config["optimization"]["levels"]
    }
    # Add baseline
    models_to_compare = {"Baseline (FP32)": Path(model_arg)} | models_to_compare
    try:
        counter.compare_counts(models_to_compare)
    except ValueError:
        pass  # Some optimised models may not exist yet
