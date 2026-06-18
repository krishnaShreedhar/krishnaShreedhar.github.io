"""
sklearn_exporter.py
-------------------
Exports a scikit-learn pipeline (StandardScaler → LogisticRegression) to ONNX
using the skl2onnx library.

Design principles (SOLID):
  - Single Responsibility : only handles sklearn → ONNX export.
  - Open/Closed           : additional sklearn steps can be chained without
                            modifying this class.
  - Dependency Inversion  : config injected; no hard-coded constants.
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.shape_inference
import yaml
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sklearn_exporter")
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
# Exporter
# ---------------------------------------------------------------------------

class SklearnExporter:
    """
    Trains a sklearn pipeline and exports it to ONNX.

    The pipeline is: StandardScaler → LogisticRegression.
    Training data is generated synthetically via ``make_classification``.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration tree.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._sk_cfg = config["sklearn_model"]
        self._model_cfg = config["model"]
        self._output_dir = Path(self._model_cfg["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data as configured."""
        n_features = self._sk_cfg["n_features"]
        n_classes = self._sk_cfg["n_classes"]
        n_samples = self._sk_cfg["n_samples"]

        self._logger.info(
            "Generating synthetic data | n_samples=%d | n_features=%d | n_classes=%d",
            n_samples,
            n_features,
            n_classes,
        )
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=max(n_features // 2, n_classes),
            n_redundant=1,
            random_state=42,
        )
        self._logger.info("Data generated | X.shape=%s | y.shape=%s", X.shape, y.shape)
        return X.astype(np.float32), y

    def build_pipeline(self) -> Pipeline:
        """Construct and return the sklearn pipeline (unfitted)."""
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=500, random_state=42)),
            ]
        )
        self._logger.info("Pipeline constructed: %s", " → ".join(s for s, _ in pipeline.steps))
        return pipeline

    def train(self, pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """Fit the pipeline on training data."""
        self._logger.info("Fitting pipeline …")
        t0 = time.perf_counter()
        pipeline.fit(X, y)
        elapsed = time.perf_counter() - t0
        self._logger.info("Fitting completed in %.3f s", elapsed)
        score = pipeline.score(X, y)
        self._logger.info("Training accuracy: %.4f", score)
        return pipeline

    def export(
        self,
        pipeline: Pipeline,
        onnx_filename: str = "sklearn_pipeline.onnx",
    ) -> Path:
        """
        Convert the fitted sklearn pipeline to ONNX.

        Uses ``skl2onnx`` for conversion.  The function will raise an
        ``ImportError`` with a clear message if skl2onnx is not installed.

        Parameters
        ----------
        pipeline : Pipeline
            A fitted sklearn Pipeline.
        onnx_filename : str
            Output filename under ``model.output_dir``.

        Returns
        -------
        Path
            Absolute path of the written ONNX file.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise ImportError(
                "skl2onnx is required for sklearn → ONNX export.\n"
                "Install it with:  pip install skl2onnx\n"
                "Documentation: https://onnx.ai/sklearn-onnx/"
            ) from exc

        n_features = self._sk_cfg["n_features"]
        opset = self._model_cfg["opset_version"]
        onnx_path = self._output_dir / onnx_filename

        # Initial type: batch of float32 feature vectors
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        self._logger.info(
            "Converting sklearn pipeline to ONNX | opset=%d | initial_type=%s",
            opset,
            initial_type,
        )

        t0 = time.perf_counter()
        model_proto = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=opset,
        )
        elapsed = time.perf_counter() - t0
        self._logger.info("skl2onnx conversion completed in %.3f s", elapsed)

        with open(onnx_path, "wb") as fh:
            fh.write(model_proto.SerializeToString())
        self._logger.info("ONNX model saved → %s", onnx_path)
        return onnx_path

    def validate(self, onnx_path: Path) -> onnx.ModelProto:
        """
        Load, structural-check and shape-infer the ONNX model.

        Logs opset version, node count, and I/O tensor names/shapes.
        """
        self._logger.info("Loading ONNX model from %s", onnx_path)
        model_proto = onnx.load(str(onnx_path))

        self._logger.info("Running onnx.checker.check_model …")
        onnx.checker.check_model(model_proto)
        self._logger.info("Model check passed")

        self._logger.info("Running shape inference …")
        model_proto = onnx.shape_inference.infer_shapes(model_proto)

        graph = model_proto.graph
        opset = model_proto.opset_import[0].version
        node_count = len(graph.node)

        self._logger.info("Opset version : %d", opset)
        self._logger.info("Number of nodes: %d", node_count)

        op_types = [n.op_type for n in graph.node]
        self._logger.info("Operator types: %s", op_types)

        for inp in graph.input:
            self._logger.info("Input  | name=%s", inp.name)
        for out in graph.output:
            self._logger.info("Output | name=%s", out.name)

        return model_proto

    def run(self) -> Path:
        """Convenience entry point: generate → train → export → validate."""
        X, y = self.generate_data()
        pipeline = self.build_pipeline()
        pipeline = self.train(pipeline, X, y)
        onnx_path = self.export(pipeline)
        self.validate(onnx_path)
        return onnx_path


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
    config = load_config(cfg_path)
    exporter = SklearnExporter(config)
    out = exporter.run()
    print(f"Exported ONNX model → {out}")
