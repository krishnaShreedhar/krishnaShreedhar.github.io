"""
benchmark.py
------------
Benchmarks ONNX Runtime inference latency and throughput.

Measurements:
  - Wall-clock time for ``session.run()`` averaged over N iterations.
  - Warm-up phase to prime the CPU caches.
  - Comparison: FP32 baseline vs optimised model.
  - Batch-size sweep: latency and throughput across batch sizes.

All timings use ``time.perf_counter`` for highest resolution.

Design principles (SOLID):
  - Single Responsibility : benchmarking and statistics only.
  - Open/Closed           : new metric collectors can be added without
                            touching the core timing loop.
"""

import logging
import time
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

    logger = logging.getLogger("benchmark")
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
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Statistics from a single benchmarking run."""

    model_label: str
    batch_size: int
    n_iterations: int
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_s: float


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class InferenceBenchmark:
    """
    Measures ORT inference performance for one or more ONNX models.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._logger = _build_logger(config)
        self._inf_cfg = config["inference"]
        self._m_cfg = config["pytorch_model"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def benchmark_model(
        self,
        model_path: Path,
        batch_size: int,
        label: str | None = None,
        providers: list[str] | None = None,
    ) -> BenchmarkResult:
        """
        Benchmark a single ONNX model at the given batch size.

        Parameters
        ----------
        model_path : Path
            ONNX file to benchmark.
        batch_size : int
            Number of samples per forward pass.
        label : str | None
            Human-readable name for logging.
        providers : list[str] | None
            ORT Execution Providers.

        Returns
        -------
        BenchmarkResult
        """
        label = label or model_path.stem
        providers = providers or self._inf_cfg["providers"]
        n_iters = self._inf_cfg["benchmark_iterations"]
        n_warmup = self._inf_cfg["warmup_iterations"]

        self._logger.info(
            "Benchmarking | model=%-40s | batch_size=%d | iters=%d | warmup=%d",
            label,
            batch_size,
            n_iters,
            n_warmup,
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers,
        )
        input_name = session.get_inputs()[0].name

        # Build a fixed random input
        dummy = self._build_dummy_input(batch_size)

        # Warm-up
        self._logger.debug("Running %d warm-up iterations …", n_warmup)
        for _ in range(n_warmup):
            session.run(None, {input_name: dummy})

        # Timed loop
        latencies_ms: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies_ms)
        result = BenchmarkResult(
            model_label=label,
            batch_size=batch_size,
            n_iterations=n_iters,
            mean_latency_ms=float(arr.mean()),
            std_latency_ms=float(arr.std()),
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            throughput_samples_per_s=batch_size / (arr.mean() / 1000.0),
        )

        self._logger.info(
            "  mean=%.3f ms | p50=%.3f ms | p95=%.3f ms | p99=%.3f ms | "
            "throughput=%.1f samples/s",
            result.mean_latency_ms,
            result.p50_latency_ms,
            result.p95_latency_ms,
            result.p99_latency_ms,
            result.throughput_samples_per_s,
        )
        return result

    def compare_models(
        self,
        baseline_path: Path,
        optimized_path: Path,
        batch_size: int = 1,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Compare FP32 baseline vs optimised model side-by-side.

        Returns
        -------
        tuple[BenchmarkResult, BenchmarkResult]
            (baseline_result, optimized_result)
        """
        self._logger.info("=" * 70)
        self._logger.info("FP32 baseline vs optimised comparison | batch_size=%d", batch_size)

        baseline = self.benchmark_model(baseline_path, batch_size, label="baseline (FP32)")
        optimized = self.benchmark_model(optimized_path, batch_size, label="optimized (ORT_ALL)")

        speedup = baseline.mean_latency_ms / optimized.mean_latency_ms
        self._logger.info("-" * 70)
        self._logger.info(
            "Speedup: %.2fx  (baseline=%.3f ms → optimized=%.3f ms)",
            speedup,
            baseline.mean_latency_ms,
            optimized.mean_latency_ms,
        )
        self._logger.info("=" * 70)
        return baseline, optimized

    def batch_size_sweep(
        self,
        model_path: Path,
        label: str | None = None,
    ) -> list[BenchmarkResult]:
        """
        Run the benchmark across all configured batch sizes.

        Returns
        -------
        list[BenchmarkResult]
        """
        batch_sizes = self._inf_cfg["batch_sizes"]
        self._logger.info(
            "Batch-size sweep | model=%s | batch_sizes=%s",
            model_path.name,
            batch_sizes,
        )
        results: list[BenchmarkResult] = []
        for bs in batch_sizes:
            r = self.benchmark_model(model_path, bs, label=label)
            results.append(r)

        self._log_sweep_table(results)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dummy_input(self, batch_size: int) -> np.ndarray:
        c = self._m_cfg["in_channels"]
        h = self._m_cfg["input_height"]
        w = self._m_cfg["input_width"]
        return np.random.randn(batch_size, c, h, w).astype(np.float32)

    def _log_sweep_table(self, results: list[BenchmarkResult]) -> None:
        self._logger.info("=" * 70)
        self._logger.info("Batch-size sweep results")
        self._logger.info(
            "%-10s %12s %12s %12s %18s",
            "Batch", "Mean (ms)", "P95 (ms)", "P99 (ms)", "Throughput (s/s)",
        )
        self._logger.info("-" * 70)
        for r in results:
            self._logger.info(
                "%-10d %12.3f %12.3f %12.3f %18.1f",
                r.batch_size,
                r.mean_latency_ms,
                r.p95_latency_ms,
                r.p99_latency_ms,
                r.throughput_samples_per_s,
            )
        self._logger.info("=" * 70)


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
    opt_arg = (
        sys.argv[3]
        if len(sys.argv) > 3
        else "outputs/models/optimized_ORT_ENABLE_ALL.onnx"
    )

    config = load_config(cfg_path)
    bench = InferenceBenchmark(config)

    # Batch-size sweep on baseline
    bench.batch_size_sweep(Path(model_arg), label="baseline")

    # Compare baseline vs optimised
    if Path(opt_arg).exists():
        bench.compare_models(Path(model_arg), Path(opt_arg), batch_size=1)
