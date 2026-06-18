"""
GPU Memory Manager — Tutorial Module 1a

Demonstrates:
  - GPU memory allocation and deallocation
  - Pinned (page-locked) host memory for faster transfers
  - GPU memory pool queries and fragmentation awareness
  - Explicit memory cleanup with torch.cuda.empty_cache()

Run: python -m src.gpu_basics.memory_manager
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_utils import setup_logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gpu_basics.yaml"


@dataclass
class MemoryStats:
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float


class GPUMemoryManager:
    """Manages GPU memory allocation and provides diagnostic utilities."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._log = setup_logger("gpu_basics.memory_manager", config)
        device_id = config["gpu"]["primary_device"]
        self._device = torch.device(f"cuda:{device_id}")
        self._allocated: list[torch.Tensor] = []
        self._log.info("GPUMemoryManager initialised on %s", self._device)

    def stats(self) -> MemoryStats:
        props = torch.cuda.get_device_properties(self._device)
        allocated = torch.cuda.memory_allocated(self._device) / 1e6
        reserved = torch.cuda.memory_reserved(self._device) / 1e6
        total = props.total_memory / 1e6
        free = total - reserved
        return MemoryStats(allocated, reserved, free, total)

    def _log_stats(self, label: str) -> None:
        s = self.stats()
        self._log.debug(
            "[%s] allocated=%.1f MB  reserved=%.1f MB  free=%.1f MB  total=%.1f MB",
            label, s.allocated_mb, s.reserved_mb, s.free_mb, s.total_mb,
        )

    def allocate(self, size_mb: float) -> torch.Tensor:
        """Allocate a float32 tensor of approximately size_mb on the GPU."""
        num_floats = int(size_mb * 1e6 / 4)
        self._log.debug("Allocating %.1f MB on %s ...", size_mb, self._device)
        t = torch.empty(num_floats, dtype=torch.float32, device=self._device)
        self._allocated.append(t)
        self._log_stats(f"after alloc {size_mb} MB")
        return t

    def allocate_pinned(self, size_mb: float) -> torch.Tensor:
        """Allocate pinned (page-locked) host memory — faster for PCIe transfers."""
        num_floats = int(size_mb * 1e6 / 4)
        self._log.debug("Allocating %.1f MB of pinned host memory ...", size_mb)
        t = torch.empty(num_floats, dtype=torch.float32, pin_memory=True)
        return t

    def free_all(self) -> None:
        self._log.info("Freeing %d tensors and clearing cache ...", len(self._allocated))
        self._allocated.clear()
        torch.cuda.empty_cache()
        self._log_stats("after free_all")

    def run_allocation_benchmark(self) -> None:
        sizes = self._cfg["memory"]["allocation_sizes_mb"]
        self._log.info("=== Allocation benchmark: sizes %s MB ===", sizes)
        self._log_stats("baseline")
        for mb in sizes:
            t0 = time.perf_counter()
            tensor = self.allocate(mb)
            elapsed = (time.perf_counter() - t0) * 1000
            self._log.info("Allocated %5.0f MB in %6.2f ms — tensor.shape=%s", mb, elapsed, tensor.shape)
        self._log_stats("peak")
        self.free_all()

    def demonstrate_pinned_memory(self) -> None:
        self._log.info("=== Pinned vs. pageable host memory ===")
        size_mb = 512
        num_floats = int(size_mb * 1e6 / 4)

        for label, pin in [("pageable", False), ("pinned", True)]:
            if pin:
                host = torch.empty(num_floats, dtype=torch.float32, pin_memory=True)
            else:
                host = torch.empty(num_floats, dtype=torch.float32)

            # warmup
            _ = host.to(self._device, non_blocking=pin)
            torch.cuda.synchronize(self._device)

            t0 = time.perf_counter()
            gpu = host.to(self._device, non_blocking=pin)
            torch.cuda.synchronize(self._device)
            elapsed = (time.perf_counter() - t0) * 1000
            bw_gb_s = (size_mb / 1e3) / (elapsed / 1e3)
            self._log.info(
                "%-10s → GPU: %6.2f ms  bandwidth=%.1f GB/s", label, elapsed, bw_gb_s
            )
            del gpu
            torch.cuda.empty_cache()


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    mgr = GPUMemoryManager(config)
    mgr.run_allocation_benchmark()
    mgr.demonstrate_pinned_memory()


if __name__ == "__main__":
    main()
