"""
Chunked CSV Processing
======================
Demonstrates:
  - Writing a large synthetic CSV to disk
  - Processing it in chunks with pd.read_csv(chunksize=)
  - Accumulating partial aggregations across chunks
  - Concatenating chunked results safely

All constants loaded from config.yaml.

Run:
    python src/pandas_optimization/chunked_processing.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

class SyntheticCsvWriter:
    """Generates and writes synthetic data to a CSV file."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._seed = cfg["data"]["random_seed"]
        self._output_dir = Path(cfg["data"]["output_dir"])
        self._rng = np.random.default_rng(self._seed)

    def write(self) -> Path:
        """Write CSV and return its path."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "synthetic_sales.csv"

        dates = pd.date_range("2022-01-01", periods=self._n, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": dates.astype(str),
                "region": self._rng.choice(["North", "South", "East", "West"], self._n),
                "product": self._rng.choice(["Widget", "Gadget", "Doohickey"], self._n),
                "salesperson_id": self._rng.integers(1, 21, self._n),
                "revenue": self._rng.exponential(500, self._n).round(2),
                "units": self._rng.integers(1, 100, self._n),
            }
        )
        df.to_csv(path, index=False)
        size_mb = path.stat().st_size / 1024 ** 2
        logger.info("SyntheticCsvWriter: wrote %s (%.2f MB)", path, size_mb)
        return path


# ---------------------------------------------------------------------------
# Partial aggregation accumulator
# ---------------------------------------------------------------------------

class ChunkAggregator:
    """
    Accumulates partial group-level statistics across chunks.

    Uses the algebraic property:
      - sum(sum_i) = total_sum
      - count(count_i) = total_count
      - mean = total_sum / total_count
    """

    def __init__(self) -> None:
        self._partials: List[pd.DataFrame] = []
        self._chunks_processed = 0

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        """Compute per-group sum/count for this chunk and store."""
        partial = (
            chunk.groupby(["region", "product"])
            .agg(
                revenue_sum=("revenue", "sum"),
                revenue_count=("revenue", "count"),
                units_sum=("units", "sum"),
            )
            .reset_index()
        )
        self._partials.append(partial)
        self._chunks_processed += 1

    def finalise(self) -> pd.DataFrame:
        """Combine all partial results into a final aggregation."""
        combined = pd.concat(self._partials, ignore_index=True)
        final = (
            combined.groupby(["region", "product"])
            .agg(
                revenue_sum=("revenue_sum", "sum"),
                revenue_count=("revenue_count", "sum"),
                units_sum=("units_sum", "sum"),
            )
            .reset_index()
        )
        final["mean_revenue"] = (final["revenue_sum"] / final["revenue_count"]).round(2)
        logger.info(
            "ChunkAggregator.finalise(): %d chunks processed, final shape=%s",
            self._chunks_processed,
            final.shape,
        )
        return final


# ---------------------------------------------------------------------------
# Chunk processor
# ---------------------------------------------------------------------------

class CsvChunkProcessor:
    """Reads a large CSV in chunks and applies transformations."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._chunksize = cfg["pandas"]["chunksize"]
        self._low_memory = cfg["pandas"]["low_memory"]

    def process(self, path: Path) -> pd.DataFrame:
        """Stream through CSV and return aggregated result."""
        logger.info(
            "CsvChunkProcessor.process(): path=%s chunksize=%d",
            path,
            self._chunksize,
        )
        aggregator = ChunkAggregator()

        reader = pd.read_csv(
            path,
            chunksize=self._chunksize,
            low_memory=self._low_memory,
            parse_dates=["timestamp"],
        )

        for i, chunk in enumerate(reader):
            # Per-chunk transformation: filter out negative revenue (data quality)
            chunk = chunk[chunk["revenue"] > 0]
            aggregator.add_chunk(chunk)
            logger.debug("  chunk %d: %d rows", i, len(chunk))

        result = aggregator.finalise()
        logger.info("Final aggregation:\n%s", result.to_string(index=False))
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ChunkedProcessingRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def run(self) -> None:
        logger.info("=== Chunked Processing START ===")

        csv_path = SyntheticCsvWriter(self._cfg).write()
        result = CsvChunkProcessor(self._cfg).process(csv_path)

        output_path = Path(self._cfg["data"]["output_dir"]) / "chunked_aggregation.csv"
        result.to_csv(output_path, index=False)
        logger.info("Saved aggregation to %s", output_path)

        logger.info("=== Chunked Processing END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    ChunkedProcessingRunner(_cfg).run()
