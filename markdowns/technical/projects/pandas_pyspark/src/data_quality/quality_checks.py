"""
Data Quality Checks
====================
Implements Great-Expectations-style checks using pure Pandas:
  - Null / missing value thresholds
  - Range validation (min/max bounds)
  - Uniqueness checks
  - Schema validation (expected dtypes, required columns)
  - Custom predicate checks
  - Summary report generation

All constants loaded from config.yaml.

Run:
    python src/data_quality/quality_checks.py
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import load_config, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check result model
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Immutable result of a single quality check."""

    check_name: str
    column: Optional[str]
    passed: bool
    actual_value: Any
    expected: str
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        col = f"[{self.column}]" if self.column else ""
        return f"{status} | {self.check_name}{col} | actual={self.actual_value} | expected={self.expected} | {self.message}"


# ---------------------------------------------------------------------------
# Individual check implementations (Open/Closed: extend by adding new classes)
# ---------------------------------------------------------------------------

class NullCheck:
    """Fail if fraction of nulls exceeds threshold."""

    def __init__(self, column: str, threshold: float) -> None:
        self._col = column
        self._threshold = threshold

    def run(self, df: pd.DataFrame) -> CheckResult:
        null_frac = df[self._col].isna().mean()
        passed = null_frac <= self._threshold
        return CheckResult(
            check_name="NullCheck",
            column=self._col,
            passed=passed,
            actual_value=round(null_frac, 6),
            expected=f"<= {self._threshold}",
            message=f"null_fraction={null_frac:.4%}",
        )


class RangeCheck:
    """Fail if any value falls outside [min_val, max_val]."""

    def __init__(self, column: str, min_val: float, max_val: float) -> None:
        self._col = column
        self._min = min_val
        self._max = max_val

    def run(self, df: pd.DataFrame) -> CheckResult:
        series = df[self._col].dropna()
        out_of_range = ((series < self._min) | (series > self._max)).sum()
        passed = out_of_range == 0
        return CheckResult(
            check_name="RangeCheck",
            column=self._col,
            passed=passed,
            actual_value=int(out_of_range),
            expected=f"0 violations in [{self._min}, {self._max}]",
            message=f"out_of_range_count={out_of_range}",
        )


class UniquenessCheck:
    """Fail if any duplicate values exist in the column."""

    def __init__(self, column: str) -> None:
        self._col = column

    def run(self, df: pd.DataFrame) -> CheckResult:
        n_duplicates = df[self._col].duplicated().sum()
        passed = n_duplicates == 0
        return CheckResult(
            check_name="UniquenessCheck",
            column=self._col,
            passed=passed,
            actual_value=int(n_duplicates),
            expected="0 duplicates",
            message=f"duplicate_count={n_duplicates}",
        )


class SchemaCheck:
    """Validate presence and dtype of required columns."""

    def __init__(self, expected_schema: Dict[str, type]) -> None:
        self._schema = expected_schema

    def run(self, df: pd.DataFrame) -> List[CheckResult]:
        results = []
        for col, expected_dtype in self._schema.items():
            if col not in df.columns:
                results.append(
                    CheckResult(
                        check_name="SchemaCheck",
                        column=col,
                        passed=False,
                        actual_value="MISSING",
                        expected=str(expected_dtype),
                        message=f"Column '{col}' not found in DataFrame",
                    )
                )
            else:
                actual_kind = df[col].dtype.kind
                type_map = {
                    "float": "f",
                    "int": "i",
                    "str": "O",
                    "object": "O",
                    "bool": "b",
                }
                expected_kind = type_map.get(expected_dtype.__name__, "O")
                passed = actual_kind == expected_kind
                results.append(
                    CheckResult(
                        check_name="SchemaCheck",
                        column=col,
                        passed=passed,
                        actual_value=str(df[col].dtype),
                        expected=f"kind='{expected_kind}'",
                        message=(
                            "dtype matches" if passed
                            else f"expected kind '{expected_kind}', got '{actual_kind}'"
                        ),
                    )
                )
        return results


class PredicateCheck:
    """Fail if predicate returns False for any row."""

    def __init__(self, name: str, column: Optional[str], predicate: Callable[[pd.DataFrame], pd.Series]) -> None:
        self._name = name
        self._col = column
        self._predicate = predicate

    def run(self, df: pd.DataFrame) -> CheckResult:
        mask = self._predicate(df)
        failures = (~mask).sum()
        passed = failures == 0
        return CheckResult(
            check_name=f"PredicateCheck:{self._name}",
            column=self._col,
            passed=passed,
            actual_value=int(failures),
            expected="0 predicate failures",
            message=f"failure_count={failures}",
        )


# ---------------------------------------------------------------------------
# Quality suite runner
# ---------------------------------------------------------------------------

class DataQualitySuite:
    """
    Registers and executes a collection of checks against a DataFrame.

    Interface Segregation: accepts any callable with a .run(df) method.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._checks: List[Any] = []
        logger.info("DataQualitySuite '%s' initialised", name)

    def add_check(self, check) -> "DataQualitySuite":
        self._checks.append(check)
        return self

    def run(self, df: pd.DataFrame) -> List[CheckResult]:
        logger.info("Running %d checks on suite '%s'", len(self._checks), self._name)
        results: List[CheckResult] = []
        for check in self._checks:
            partial = check.run(df)
            if isinstance(partial, list):
                results.extend(partial)
            else:
                results.append(partial)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        logger.info(
            "Suite '%s': %d total checks | %d passed | %d failed",
            self._name,
            len(results),
            passed,
            failed,
        )
        for result in results:
            log_fn = logger.info if result.passed else logger.warning
            log_fn("  %s", result)

        return results


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class QualityReportGenerator:
    """Converts check results into a Pandas DataFrame report and saves to CSV."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def generate(self, results: List[CheckResult], suite_name: str) -> pd.DataFrame:
        rows = [
            {
                "suite": suite_name,
                "check_name": r.check_name,
                "column": r.column or "",
                "passed": r.passed,
                "actual_value": str(r.actual_value),
                "expected": r.expected,
                "message": r.message,
            }
            for r in results
        ]
        report_df = pd.DataFrame(rows)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{suite_name}_quality_report.csv"
        report_df.to_csv(path, index=False)
        logger.info("Quality report saved: %s (%d rows)", path, len(report_df))
        return report_df


# ---------------------------------------------------------------------------
# Data generator with intentional quality issues
# ---------------------------------------------------------------------------

class QualityDataGenerator:
    """Generates data with deliberate quality problems to exercise checks."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._n = cfg["data"]["num_rows"]
        self._seed = cfg["data"]["random_seed"]
        self._rng = np.random.default_rng(self._seed)

    def build_clean(self) -> pd.DataFrame:
        age = self._rng.integers(0, 100, self._n).astype(float)
        revenue = self._rng.exponential(500, self._n).round(2)
        region = self._rng.choice(["North", "South", "East", "West"], self._n)
        order_id = [f"ORD-{i:08d}" for i in range(self._n)]
        return pd.DataFrame(
            {"order_id": order_id, "region": region, "age": age, "revenue": revenue}
        )

    def build_dirty(self) -> pd.DataFrame:
        """Add nulls, out-of-range values, and duplicates."""
        df = self.build_clean()
        rng = self._rng

        # Inject ~8% nulls in age (exceeds 5% threshold)
        null_indices = rng.choice(self._n, size=int(self._n * 0.08), replace=False)
        df.loc[null_indices, "age"] = np.nan

        # Inject out-of-range age values
        bad_indices = rng.choice(self._n, size=50, replace=False)
        df.loc[bad_indices, "age"] = rng.choice([-5, 130, 200], size=50)

        # Inject negative revenue
        neg_indices = rng.choice(self._n, size=30, replace=False)
        df.loc[neg_indices, "revenue"] = -100.0

        # Introduce duplicate order_ids
        dup_indices = rng.choice(self._n, size=20, replace=False)
        df.loc[dup_indices, "order_id"] = "ORD-00000001"

        logger.info("Built dirty dataset: %d rows with injected quality issues", len(df))
        return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataQualityRunner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg

    def _build_suite(self) -> DataQualitySuite:
        dq_cfg = self._cfg["data_quality"]
        null_thresh = dq_cfg["null_threshold"]
        min_age = dq_cfg["min_age"]
        max_age = dq_cfg["max_age"]
        min_revenue = dq_cfg["min_revenue"]

        schema = {"order_id": str, "region": str, "age": float, "revenue": float}

        suite = DataQualitySuite("sales_quality")
        suite.add_check(NullCheck("age", threshold=null_thresh))
        suite.add_check(NullCheck("revenue", threshold=null_thresh))
        suite.add_check(NullCheck("region", threshold=null_thresh))
        suite.add_check(RangeCheck("age", min_val=min_age, max_val=max_age))
        suite.add_check(RangeCheck("revenue", min_val=min_revenue, max_val=float("inf")))
        suite.add_check(UniquenessCheck("order_id"))
        suite.add_check(SchemaCheck(schema))
        suite.add_check(
            PredicateCheck(
                "no_empty_region",
                column="region",
                predicate=lambda df: df["region"].str.strip() != "",
            )
        )
        return suite

    def run(self) -> None:
        logger.info("=== Data Quality Checks START ===")
        gen = QualityDataGenerator(self._cfg)
        suite = self._build_suite()
        reporter = QualityReportGenerator(Path(self._cfg["data"]["output_dir"]))

        # Run on clean data
        clean_df = gen.build_clean()
        logger.info("--- Quality checks on CLEAN data ---")
        clean_results = suite.run(clean_df)
        reporter.generate(clean_results, "clean_data")

        # Run on dirty data
        dirty_df = gen.build_dirty()
        logger.info("--- Quality checks on DIRTY data ---")
        dirty_results = suite.run(dirty_df)
        reporter.generate(dirty_results, "dirty_data")

        clean_pass = sum(1 for r in clean_results if r.passed)
        dirty_pass = sum(1 for r in dirty_results if r.passed)
        logger.info(
            "Pass rates: clean=%d/%d  dirty=%d/%d",
            clean_pass, len(clean_results),
            dirty_pass, len(dirty_results),
        )
        logger.info("=== Data Quality Checks END ===")


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    DataQualityRunner(_cfg).run()
