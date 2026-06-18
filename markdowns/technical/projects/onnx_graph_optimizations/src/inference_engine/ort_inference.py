"""
ort_inference.py
----------------
Creates an ONNX Runtime InferenceSession and exposes a clean run interface.

Responsibilities:
  - Build an ORT session with configured optimization level and providers.
  - Run single and batched inference.
  - Optionally enable the ORT inference profiler.

Design principles (SOLID):
  - Single Responsibility : session creation and inference only.
  - Dependency Inversion  : configuration injected; providers resolved
                            via ExecutionProviderSelector.
"""

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(config: dict[str, Any]) -> logging.Logger:
    log_cfg = config["logging"]
    log_path = Path(log_cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ort_inference")
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
# Session factory
# ---------------------------------------------------------------------------

class OrtInferenceSession:
    """
    Wraps an ORT InferenceSession with logging and profiling support.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    providers : list[str] | None
        Execution providers to use.  If None, uses ``inference.providers``
        from config.
    optimization_level : ort.GraphOptimizationLevel | None
        Overrides the default (ORT_ENABLE_ALL) if provided.
    """

    def __init__(
        self,
        config: dict[str, Any],
        providers: list[str] | None = None,
        optimization_level: ort.GraphOptimizationLevel | None = None,
    ) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._inf_cfg = config["inference"]
        self._providers = providers or self._inf_cfg["providers"]
        self._opt_level = optimization_level or ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session: ort.InferenceSession | None = None
        self._model_path: Path | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> "OrtInferenceSession":
        """
        Load and initialise ORT session from *model_path*.

        Parameters
        ----------
        model_path : Path
            Path to the ONNX model file.

        Returns
        -------
        self
            Fluent interface.
        """
        self._model_path = model_path
        sess_opts = self._build_session_options()

        self._logger.info(
            "Loading ORT session | model=%s | providers=%s | opt_level=%s",
            model_path,
            self._providers,
            self._opt_level.name if hasattr(self._opt_level, "name") else self._opt_level,
        )
        t0 = time.perf_counter()
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=self._providers,
        )
        elapsed = time.perf_counter() - t0
        self._logger.info("Session loaded in %.4f s", elapsed)

        self._log_session_info()
        return self

    def run(
        self, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """
        Execute inference.

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            Mapping of input name → numpy array.

        Returns
        -------
        list[np.ndarray]
            Model outputs.
        """
        if self._session is None:
            raise RuntimeError("Session not loaded. Call load() first.")

        output_names = [o.name for o in self._session.get_outputs()]
        t0 = time.perf_counter()
        outputs = self._session.run(output_names, inputs)
        elapsed = time.perf_counter() - t0
        self._logger.debug(
            "Inference completed | elapsed=%.4f s | output_shapes=%s",
            elapsed,
            [o.shape for o in outputs],
        )
        return outputs

    def run_batch(
        self,
        input_data: np.ndarray,
        input_name: str,
        batch_size: int,
    ) -> list[np.ndarray]:
        """
        Run inference on *input_data* in chunks of *batch_size*.

        Parameters
        ----------
        input_data : np.ndarray
            Full dataset, shape ``[N, ...]``.
        input_name : str
            ONNX model input tensor name.
        batch_size : int
            Number of samples per forward pass.

        Returns
        -------
        list[np.ndarray]
            Concatenated outputs for all batches.
        """
        if self._session is None:
            raise RuntimeError("Session not loaded. Call load() first.")

        n_samples = input_data.shape[0]
        self._logger.info(
            "Batch inference | n_samples=%d | batch_size=%d", n_samples, batch_size
        )

        all_outputs: list[np.ndarray] = []
        for start in range(0, n_samples, batch_size):
            batch = input_data[start : start + batch_size]
            result = self.run({input_name: batch})
            all_outputs.append(result[0])

        combined = np.concatenate(all_outputs, axis=0)
        self._logger.info(
            "Batch inference done | combined_shape=%s", combined.shape
        )
        return [combined]

    def get_session(self) -> ort.InferenceSession:
        """Return the underlying ORT session."""
        if self._session is None:
            raise RuntimeError("Session not loaded. Call load() first.")
        return self._session

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = self._opt_level

        enable_profiling = self._inf_cfg.get("enable_profiling", False)
        if enable_profiling:
            log_dir = Path(self._config["logging"]["log_file"]).parent
            opts.enable_profiling = True
            opts.profile_file_prefix = str(log_dir / "ort_profile")
            self._logger.info("ORT profiling enabled | prefix=%s", opts.profile_file_prefix)

        opts.inter_op_num_threads = 0   # let ORT auto-detect
        opts.intra_op_num_threads = 0
        return opts

    def _log_session_info(self) -> None:
        session = self._session
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        providers_active = session.get_providers()

        self._logger.info("Active providers  : %s", providers_active)
        for inp in inputs:
            self._logger.info(
                "Session input  | name=%-20s | shape=%s | dtype=%s",
                inp.name,
                inp.shape,
                inp.type,
            )
        for out in outputs:
            self._logger.info(
                "Session output | name=%-20s | shape=%s | dtype=%s",
                out.name,
                out.shape,
                out.type,
            )


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
    m_cfg = config["pytorch_model"]

    session = OrtInferenceSession(config)
    session.load(Path(model_arg))

    dummy = np.random.randn(
        1,
        m_cfg["in_channels"],
        m_cfg["input_height"],
        m_cfg["input_width"],
    ).astype(np.float32)

    outputs = session.run({"input": dummy})
    print(f"Output shape: {outputs[0].shape}")
