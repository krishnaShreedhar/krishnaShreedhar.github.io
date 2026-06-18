"""
execution_providers.py
----------------------
Selects and validates available ONNX Runtime Execution Providers (EPs).

Execution Providers are the backends ORT uses to run operators.  Each EP
targets specific hardware:

  CPUExecutionProvider     – always available, reference implementation
  CUDAExecutionProvider    – NVIDIA GPU via CUDA
  TensorrtExecutionProvider– NVIDIA GPU via TensorRT (additional fusion)
  ROCmExecutionProvider    – AMD GPU via ROCm
  CoreMLExecutionProvider  – Apple Silicon / macOS
  OpenVINOExecutionProvider– Intel CPUs, iGPUs, VPUs

ORT falls through the provider list in order, delegating ops it cannot
execute on the preferred EP to the next in the list.

Design principles (SOLID):
  - Single Responsibility : EP selection and availability checks only.
  - Open/Closed           : new providers can be registered without
                            modifying existing ones.
"""

import logging
from dataclasses import dataclass
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

    logger = logging.getLogger("execution_providers")
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
# Provider registry
# ---------------------------------------------------------------------------

@dataclass
class ProviderInfo:
    """Describes an Execution Provider."""

    name: str
    hardware: str
    notes: str
    requires_package: str | None = None


_KNOWN_PROVIDERS: list[ProviderInfo] = [
    ProviderInfo(
        name="CPUExecutionProvider",
        hardware="CPU (x86 / ARM)",
        notes="Always available. Reference EP. Supports MLAS kernel library.",
    ),
    ProviderInfo(
        name="CUDAExecutionProvider",
        hardware="NVIDIA GPU (CUDA)",
        notes="Requires onnxruntime-gpu package and CUDA toolkit.",
        requires_package="onnxruntime-gpu",
    ),
    ProviderInfo(
        name="TensorrtExecutionProvider",
        hardware="NVIDIA GPU (TensorRT)",
        notes="Requires TensorRT installed and onnxruntime-gpu. Best for throughput.",
        requires_package="onnxruntime-gpu",
    ),
    ProviderInfo(
        name="ROCmExecutionProvider",
        hardware="AMD GPU (ROCm)",
        notes="Requires onnxruntime-rocm package.",
        requires_package="onnxruntime-rocm",
    ),
    ProviderInfo(
        name="CoreMLExecutionProvider",
        hardware="Apple Silicon / macOS",
        notes="Available on macOS. Uses Core ML delegate.",
    ),
    ProviderInfo(
        name="OpenVINOExecutionProvider",
        hardware="Intel CPU / iGPU / VPU",
        notes="Requires OpenVINO toolkit and onnxruntime-openvino.",
        requires_package="onnxruntime-openvino",
    ),
    ProviderInfo(
        name="DirectMLExecutionProvider",
        hardware="Windows GPU (DirectX 12)",
        notes="Windows only. Requires onnxruntime-directml.",
        requires_package="onnxruntime-directml",
    ),
]


# ---------------------------------------------------------------------------
# EP Selector
# ---------------------------------------------------------------------------

class ExecutionProviderSelector:
    """
    Identifies available ORT Execution Providers and guides selection.

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

    def list_available(self) -> list[str]:
        """
        Return providers available in the current ORT installation.

        Logs a structured table for clarity.
        """
        available = ort.get_available_providers()
        self._logger.info("ORT version: %s", ort.__version__)
        self._logger.info("Available Execution Providers (%d):", len(available))
        for ep in available:
            self._logger.info("  • %s", ep)
        return available

    def report_all_providers(self) -> None:
        """Log full registry of known providers and their availability."""
        available = set(ort.get_available_providers())
        self._logger.info("=" * 70)
        self._logger.info(
            "%-35s %-8s %s", "Provider", "Status", "Hardware"
        )
        self._logger.info("-" * 70)
        for pinfo in _KNOWN_PROVIDERS:
            status = "AVAILABLE" if pinfo.name in available else "not installed"
            self._logger.info(
                "%-35s %-12s %s", pinfo.name, status, pinfo.hardware
            )
            if pinfo.name not in available and pinfo.requires_package:
                self._logger.info(
                    "    Install: pip install %s", pinfo.requires_package
                )
        self._logger.info("=" * 70)

    def build_provider_list(self, preferred: list[str]) -> list[str]:
        """
        Build a safe provider list by filtering *preferred* down to what
        is actually available, always appending CPUExecutionProvider as
        the final fallback within ORT's EP chain.

        Parameters
        ----------
        preferred : list[str]
            Requested providers in priority order.

        Returns
        -------
        list[str]
            Providers that are both requested and available.
        """
        available = set(ort.get_available_providers())
        result: list[str] = []
        for ep in preferred:
            if ep in available:
                result.append(ep)
                self._logger.info("EP selected  : %s", ep)
            else:
                self._logger.warning(
                    "EP not available: %s — skipping", ep
                )

        # Guarantee CPU is always in the list (ORT requires it)
        if "CPUExecutionProvider" not in result:
            result.append("CPUExecutionProvider")
            self._logger.info("CPUExecutionProvider appended as safety fallback")

        return result

    def demo_cuda_switch(
        self,
        model_path: Path,
        dummy_input: np.ndarray,
        input_name: str = "input",
    ) -> str:
        """
        Attempt to create a CUDA session.  If CUDA is unavailable, log a
        clear message and return 'CPUExecutionProvider'.

        This method demonstrates how to write portable code that handles
        GPU absence gracefully.

        Parameters
        ----------
        model_path : Path
            ONNX model to load.
        dummy_input : np.ndarray
            Input tensor to verify the session works.
        input_name : str
            ONNX input tensor name.

        Returns
        -------
        str
            The provider that was actually used.
        """
        available = ort.get_available_providers()
        self._logger.info("Attempting CUDA session …")

        if "CUDAExecutionProvider" not in available:
            self._logger.warning(
                "CUDAExecutionProvider is NOT available in this ORT build. "
                "Falling back to CPUExecutionProvider for this demo. "
                "To enable CUDA: pip install onnxruntime-gpu and ensure "
                "CUDA drivers are installed."
            )
            return "CPUExecutionProvider"

        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = session.get_providers()[0]
            session.run(None, {input_name: dummy_input})
            self._logger.info("CUDA session active | provider=%s", active)
            return active
        except Exception as exc:
            self._logger.warning(
                "CUDA session failed (%s). "
                "Continuing with CPUExecutionProvider.",
                exc,
            )
            return "CPUExecutionProvider"


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
    selector = ExecutionProviderSelector(config)
    selector.list_available()
    selector.report_all_providers()

    m_cfg = config["pytorch_model"]
    dummy = np.random.randn(
        1,
        m_cfg["in_channels"],
        m_cfg["input_height"],
        m_cfg["input_width"],
    ).astype(np.float32)
    selector.demo_cuda_switch(Path(model_arg), dummy)
