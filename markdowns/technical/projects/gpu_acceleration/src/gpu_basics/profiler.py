"""
GPU Utilization Profiler — Tutorial Module 1c

Demonstrates:
  - Sampling GPU utilization and memory usage via pynvml
  - Using torch.cuda.Event for kernel-level timing
  - PyTorch profiler integration for op-level traces

Run: python -m src.gpu_basics.profiler
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_utils import setup_logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gpu_basics.yaml"

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class GPUProfiler:
    """Wraps pynvml and PyTorch profiler for GPU utilisation and kernel timing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._log = setup_logger("gpu_basics.profiler", config)
        device_id = config["gpu"]["primary_device"]
        self._device = torch.device(f"cuda:{device_id}")
        self._device_id = device_id

        if _NVML_AVAILABLE:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self._log.info("pynvml initialised for GPU %d", device_id)
        else:
            self._log.warning("pynvml not available — NVML sampling disabled")

    def sample_utilization(self) -> dict[str, float]:
        """Return current GPU utilization and memory via NVML."""
        if not _NVML_AVAILABLE:
            return {}
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        return {
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "mem_used_mb": mem_info.used / 1e6,
            "mem_total_mb": mem_info.total / 1e6,
        }

    def time_kernel(self, fn: Any, *args: Any, warmup: int = 3, iters: int = 20) -> float:
        """
        Measure kernel execution time using CUDA Events (GPU-side timer, no CPU sync overhead).
        Returns average latency in milliseconds.
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize(self._device)

        start_event.record()
        for _ in range(iters):
            fn(*args)
        end_event.record()
        torch.cuda.synchronize(self._device)

        return start_event.elapsed_time(end_event) / iters

    def profile_matmul(self) -> None:
        """Show GPU utilization during a heavy matmul and report kernel time."""
        self._log.info("=== Matrix multiplication kernel profiling ===")
        N = 8192
        A = torch.randn(N, N, device=self._device, dtype=torch.float16)
        B = torch.randn(N, N, device=self._device, dtype=torch.float16)

        def matmul() -> torch.Tensor:
            return torch.mm(A, B)

        avg_ms = self.time_kernel(matmul, warmup=5, iters=20)
        tflops = (2 * N ** 3) / (avg_ms * 1e-3) / 1e12
        self._log.info(
            "matmul %dx%d (fp16): avg=%.3f ms  throughput=%.1f TFLOPS", N, N, avg_ms, tflops
        )

        util = self.sample_utilization()
        if util:
            self._log.info(
                "NVML sample — gpu_util=%d%%  mem_util=%d%%  mem_used=%.0f MB",
                util["gpu_util_pct"], util["mem_util_pct"], util["mem_used_mb"],
            )

    def profile_with_torch_profiler(self) -> None:
        """Trace ops with torch.profiler and log the top 5 by CUDA time."""
        self._log.info("=== torch.profiler trace ===")
        N = 4096
        A = torch.randn(N, N, device=self._device)
        B = torch.randn(N, N, device=self._device)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                C = torch.mm(A, B)
                _ = torch.relu(C)
            torch.cuda.synchronize(self._device)

        top_events = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=5
        )
        self._log.info("Top 5 ops by CUDA time:\n%s", top_events)

    def run(self) -> None:
        self.profile_matmul()
        self.profile_with_torch_profiler()

    def __del__(self) -> None:
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    profiler = GPUProfiler(config)
    profiler.run()


if __name__ == "__main__":
    main()
