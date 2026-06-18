"""
GPU Data Transfer Benchmark — Tutorial Module 1b

Demonstrates:
  - Synchronous H2D and D2H transfers
  - Asynchronous transfers with CUDA streams
  - Overlapping compute with data transfer (double-buffering pattern)
  - Measuring effective PCIe bandwidth

Run: python -m src.gpu_basics.data_transfer
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


class DataTransferBenchmark:
    """Benchmarks CPU↔GPU data transfer under different strategies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._log = setup_logger("gpu_basics.data_transfer", config)
        device_id = config["gpu"]["primary_device"]
        self._device = torch.device(f"cuda:{device_id}")
        self._warmup = config["transfer"]["num_warmup_iters"]
        self._iters = config["transfer"]["num_benchmark_iters"]
        self._log.info(
            "DataTransferBenchmark on %s  warmup=%d  iters=%d",
            self._device, self._warmup, self._iters,
        )

    def _make_tensors(self, shape: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        pinned = torch.randn(*shape, pin_memory=True)
        gpu = torch.empty(*shape, device=self._device)
        return pinned, gpu

    def _bandwidth_gb_s(self, size_bytes: int, elapsed_s: float) -> float:
        return size_bytes / elapsed_s / 1e9

    def benchmark_sync(self, shape: list[int]) -> None:
        """Baseline: synchronous (blocking) host-to-device transfer."""
        pinned, _ = self._make_tensors(shape)
        size_bytes = pinned.numel() * pinned.element_size()
        label = f"sync H2D {shape}"

        for _ in range(self._warmup):
            gpu = pinned.to(self._device)
            torch.cuda.synchronize(self._device)

        t0 = time.perf_counter()
        for _ in range(self._iters):
            gpu = pinned.to(self._device)
            torch.cuda.synchronize(self._device)
        elapsed = (time.perf_counter() - t0) / self._iters
        bw = self._bandwidth_gb_s(size_bytes, elapsed)
        self._log.info("%-30s  lat=%7.3f ms  BW=%6.2f GB/s", label, elapsed * 1e3, bw)

    def benchmark_async_stream(self, shape: list[int]) -> None:
        """Async H2D using a dedicated CUDA stream — non-blocking."""
        num_streams = self._cfg["transfer"]["num_streams"]
        streams = [torch.cuda.Stream(device=self._device) for _ in range(num_streams)]
        pinned, _ = self._make_tensors(shape)
        size_bytes = pinned.numel() * pinned.element_size()
        label = f"async H2D {shape} ({num_streams} streams)"

        for _ in range(self._warmup):
            with torch.cuda.stream(streams[0]):
                gpu = pinned.to(self._device, non_blocking=True)
            torch.cuda.synchronize(self._device)

        t0 = time.perf_counter()
        for i in range(self._iters):
            stream = streams[i % num_streams]
            with torch.cuda.stream(stream):
                gpu = pinned.to(self._device, non_blocking=True)
        torch.cuda.synchronize(self._device)
        elapsed = (time.perf_counter() - t0) / self._iters
        bw = self._bandwidth_gb_s(size_bytes, elapsed)
        self._log.info("%-30s  lat=%7.3f ms  BW=%6.2f GB/s", label, elapsed * 1e3, bw)

    def benchmark_d2h(self, shape: list[int]) -> None:
        """Device-to-host transfer benchmark."""
        gpu = torch.randn(*shape, device=self._device)
        size_bytes = gpu.numel() * gpu.element_size()
        label = f"sync D2H {shape}"

        for _ in range(self._warmup):
            cpu = gpu.cpu()

        t0 = time.perf_counter()
        for _ in range(self._iters):
            cpu = gpu.cpu()
        elapsed = (time.perf_counter() - t0) / self._iters
        bw = self._bandwidth_gb_s(size_bytes, elapsed)
        self._log.info("%-30s  lat=%7.3f ms  BW=%6.2f GB/s", label, elapsed * 1e3, bw)

    def demonstrate_double_buffering(self, shape: list[int]) -> None:
        """
        Double-buffering: overlap PCIe transfer of batch N+1 with compute on batch N.
        This is the key pattern for hiding data transfer latency during training.
        """
        self._log.info("=== Double-buffering pattern (compute + transfer overlap) ===")
        stream_compute = torch.cuda.Stream(device=self._device)
        stream_transfer = torch.cuda.Stream(device=self._device)

        batches = [torch.randn(*shape, pin_memory=True) for _ in range(4)]

        t0 = time.perf_counter()
        prev_gpu: torch.Tensor | None = None

        for i, batch in enumerate(batches):
            # Transfer current batch on transfer stream
            with torch.cuda.stream(stream_transfer):
                curr_gpu = batch.to(self._device, non_blocking=True)

            # Compute on previous batch on compute stream (after transfer is done)
            if prev_gpu is not None:
                stream_compute.wait_stream(stream_transfer)
                with torch.cuda.stream(stream_compute):
                    result = prev_gpu * 2.0 + 1.0  # stand-in for a real forward pass

            prev_gpu = curr_gpu

        torch.cuda.synchronize(self._device)
        elapsed = (time.perf_counter() - t0) * 1e3
        self._log.info(
            "Processed %d batches with double-buffering in %.2f ms", len(batches), elapsed
        )

    def run(self) -> None:
        shapes = self._cfg["transfer"]["tensor_shapes"]
        self._log.info("=== Transfer benchmarks ===")
        for shape in shapes:
            self.benchmark_sync(shape)
            self.benchmark_async_stream(shape)
            self.benchmark_d2h(shape)
            self._log.info("")
        self.demonstrate_double_buffering(shapes[0])


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    bench = DataTransferBenchmark(config)
    bench.run()


if __name__ == "__main__":
    main()
