"""
sql_statistics.py

SQL-based statistical analysis using DuckDB as an in-process analytical engine.

Concepts illustrated:
  - Descriptive statistics via SQL aggregates (STDDEV_SAMP, PERCENTILE_CONT, etc.)
  - Window functions: PERCENT_RANK, CUME_DIST, NTILE, ROW_NUMBER
  - Z-score normalisation in SQL
  - IQR-based outlier detection in SQL
  - Histogram buckets with WIDTH_BUCKET
  - Funnel analysis: step-by-step conversion rates
  - Cohort retention analysis: weekly/monthly retention
  - Linear regression metrics: REGR_SLOPE, REGR_INTERCEPT, REGR_R2

All SQL queries are clearly commented with the statistical concept they implement.
"""

from __future__ import annotations

import logging
import logging.handlers
import pathlib
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(cfg: dict[str, Any]) -> logging.Logger:
    log_cfg = cfg["logging"]
    log_file = pathlib.Path(log_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sql_analytics.sql_statistics")
    logger.setLevel(getattr(logging, log_cfg["level"].upper()))

    formatter = logging.Formatter(
        fmt=(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def load_config(config_path: str | pathlib.Path) -> dict[str, Any]:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class SyntheticEcommerceDataGenerator:
    """
    Generate synthetic e-commerce datasets suitable for SQL analytics demos.
    SRP: Only responsible for data generation.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._n = cfg["data"]["sample_size"]
        self._seed = cfg["data"]["random_seed"]
        self._rng = np.random.default_rng(self._seed)
        self._logger.info(
            f"SyntheticEcommerceDataGenerator | n={self._n}, seed={self._seed}"
        )

    def transactions_df(self) -> pd.DataFrame:
        """
        User transaction data with columns:
          user_id, transaction_date, revenue, category, user_age
        """
        user_ids = self._rng.integers(1, self._n // 5, self._n)
        revenues = np.abs(self._rng.lognormal(mean=3.5, sigma=1.2, size=self._n))
        # Add some outliers
        outlier_idx = self._rng.choice(self._n, size=int(self._n * 0.02), replace=False)
        revenues[outlier_idx] *= 10

        categories = self._rng.choice(
            ["Electronics", "Clothing", "Books", "Home", "Sports"],
            size=self._n,
            p=[0.25, 0.30, 0.15, 0.20, 0.10],
        )
        ages = self._rng.integers(18, 75, self._n)

        base_date = pd.Timestamp("2024-01-01")
        days = self._rng.integers(0, 365, self._n)
        dates = [base_date + pd.Timedelta(days=int(d)) for d in days]

        df = pd.DataFrame(
            {
                "user_id": user_ids,
                "transaction_date": dates,
                "revenue": revenues,
                "category": categories,
                "user_age": ages,
            }
        )
        self._logger.info(
            f"Generated transactions_df | shape={df.shape}, "
            f"revenue_range=[{revenues.min():.2f}, {revenues.max():.2f}]"
        )
        return df

    def funnel_events_df(self) -> pd.DataFrame:
        """
        Funnel event data: user journey through 4 steps.
        Simulates realistic drop-off rates.
        """
        n_users = self._n
        user_ids = np.arange(1, n_users + 1)

        # Retention at each funnel step
        funnel_rates = [1.0, 0.60, 0.35, 0.15, 0.07]
        step_names = ["visit", "product_view", "add_to_cart", "checkout", "purchase"]

        rows = []
        for uid in user_ids:
            for step, (name, rate) in enumerate(zip(step_names, funnel_rates)):
                if self._rng.random() < rate:
                    rows.append({"user_id": uid, "step": step + 1, "step_name": name})
                else:
                    break

        df = pd.DataFrame(rows)
        self._logger.info(
            f"Generated funnel_events_df | "
            f"users={n_users}, rows={len(df)}"
        )
        return df

    def cohort_df(self) -> pd.DataFrame:
        """
        User cohort data: cohort_week, user_id, activity_week.
        Simulates weekly retention with exponential decay.
        """
        rows = []
        n_cohorts = 12
        cohort_size = self._n // n_cohorts

        for cohort_week in range(n_cohorts):
            for uid_offset in range(cohort_size):
                uid = cohort_week * cohort_size + uid_offset + 1
                # User always active in week 0 (acquisition)
                rows.append(
                    {"cohort_week": cohort_week, "user_id": uid, "activity_week": cohort_week}
                )
                # Retention decay
                for w in range(1, n_cohorts - cohort_week):
                    retention_prob = 0.5 * np.exp(-0.3 * w)
                    if self._rng.random() < retention_prob:
                        rows.append(
                            {
                                "cohort_week": cohort_week,
                                "user_id": uid,
                                "activity_week": cohort_week + w,
                            }
                        )

        df = pd.DataFrame(rows).drop_duplicates()
        self._logger.info(
            f"Generated cohort_df | cohorts={n_cohorts}, rows={len(df)}"
        )
        return df

    def regression_df(self) -> pd.DataFrame:
        """
        Data for SQL linear regression: advertising spend vs revenue.
        revenue = 500 + 3.5 * spend + noise
        """
        spend = self._rng.uniform(100, 2000, self._n // 2)
        noise = self._rng.normal(0, 200, self._n // 2)
        revenue = 500 + 3.5 * spend + noise
        df = pd.DataFrame({"ad_spend": spend, "revenue": revenue})
        self._logger.info(
            f"Generated regression_df | n={len(df)}"
        )
        return df


# ---------------------------------------------------------------------------
# SQLStatisticsAnalyzer
# ---------------------------------------------------------------------------

class SQLStatisticsAnalyzer:
    """
    Executes statistical SQL queries using DuckDB.

    SRP: Only runs SQL queries and returns DataFrames.
    OCP: New analyses can be added as new methods without modifying existing ones.
    DIP: Depends on DuckDB connection injected or created from config.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._sql_cfg = cfg["sql_analytics"]
        db_path = self._sql_cfg["database"]
        self._con = duckdb.connect(db_path)
        self._histogram_bins = self._sql_cfg["histogram_bins"]
        self._iqr_multiplier = self._sql_cfg["iqr_multiplier"]
        self._logger.info(
            f"SQLStatisticsAnalyzer | db={db_path}, "
            f"histogram_bins={self._histogram_bins}, "
            f"iqr_multiplier={self._iqr_multiplier}"
        )

    def register_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Register a Pandas DataFrame as a DuckDB view."""
        self._con.register(name, df)
        self._logger.info(f"Registered DataFrame as DuckDB view '{name}' | shape={df.shape}")

    def _execute(self, sql: str, label: str) -> pd.DataFrame:
        """Execute SQL and return result as DataFrame."""
        self._logger.debug(f"Executing [{label}]:\n{sql}")
        result = self._con.execute(sql).df()
        self._logger.info(
            f"[{label}] returned {len(result)} rows x {len(result.columns)} cols"
        )
        return result

    # ------------------------------------------------------------------
    # 1. Descriptive Statistics via SQL aggregates
    # ------------------------------------------------------------------

    def descriptive_statistics(self, table: str, column: str) -> pd.DataFrame:
        """
        Compute comprehensive descriptive statistics using SQL.

        SQL functions used:
          - COUNT, AVG, STDDEV_SAMP, VAR_SAMP: basic aggregates
          - PERCENTILE_CONT: exact percentile calculation
          - MIN, MAX: range
          - SKEWNESS, KURTOSIS: DuckDB built-in shape statistics
        """
        sql = f"""
        SELECT
            COUNT({column})                                 AS n,
            AVG({column})                                   AS mean,
            STDDEV_SAMP({column})                           AS std_dev,
            VAR_SAMP({column})                              AS variance,
            MIN({column})                                   AS min_val,
            MAX({column})                                   AS max_val,
            MAX({column}) - MIN({column})                   AS range_val,

            -- Percentiles via PERCENTILE_CONT (exact, linear interpolation)
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY {column}) AS p10,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {column}) AS p50_median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) AS p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY {column}) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {column}) AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {column}) AS p99,

            -- IQR = Q3 - Q1
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) -
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column})  AS iqr,

            -- Shape statistics
            SKEWNESS({column})                              AS skewness,
            KURTOSIS({column})                              AS kurtosis

        FROM {table}
        WHERE {column} IS NOT NULL
        """
        return self._execute(sql, label=f"descriptive_stats({table}.{column})")

    # ------------------------------------------------------------------
    # 2. Window functions: PERCENT_RANK, CUME_DIST, NTILE
    # ------------------------------------------------------------------

    def window_functions_demo(self, table: str, column: str, n_tiles: int = 4) -> pd.DataFrame:
        """
        Illustrate window ranking functions.

        PERCENT_RANK: (rank - 1) / (N - 1) in [0, 1]
        CUME_DIST:    rank / N in (0, 1] — fraction of rows <= current
        NTILE(k):     divide into k equal-size buckets (quartiles when k=4)
        ROW_NUMBER:   unique sequential rank (no ties)
        RANK:         same rank for ties, gaps after ties
        DENSE_RANK:   same rank for ties, no gaps
        """
        sql = f"""
        SELECT
            {column},
            PERCENT_RANK()  OVER (ORDER BY {column}) AS percent_rank,
            CUME_DIST()     OVER (ORDER BY {column}) AS cumulative_dist,
            NTILE({n_tiles}) OVER (ORDER BY {column}) AS ntile_{n_tiles},
            ROW_NUMBER()    OVER (ORDER BY {column}) AS row_num,
            RANK()          OVER (ORDER BY {column}) AS rank_with_gaps,
            DENSE_RANK()    OVER (ORDER BY {column}) AS dense_rank
        FROM {table}
        WHERE {column} IS NOT NULL
        ORDER BY {column}
        LIMIT 20
        """
        return self._execute(sql, label=f"window_functions({table}.{column})")

    # ------------------------------------------------------------------
    # 3. Z-score normalisation in SQL
    # ------------------------------------------------------------------

    def zscore_normalisation(self, table: str, column: str) -> pd.DataFrame:
        """
        Compute z-scores: z = (x - mean) / std_dev.

        Uses a CTE for the global mean/std, then applies to each row.
        Z-scores > 3 or < -3 are flagged as potential outliers.
        """
        sql = f"""
        WITH stats AS (
            SELECT
                AVG({column})        AS global_mean,
                STDDEV_SAMP({column}) AS global_std
            FROM {table}
            WHERE {column} IS NOT NULL
        )
        SELECT
            {column}                                        AS original_value,
            ({column} - stats.global_mean)
                / NULLIF(stats.global_std, 0)              AS z_score,
            CASE
                WHEN ABS(({column} - stats.global_mean)
                    / NULLIF(stats.global_std, 0)) > 3
                THEN TRUE
                ELSE FALSE
            END                                            AS is_outlier_3sigma
        FROM {table}, stats
        WHERE {column} IS NOT NULL
        ORDER BY ABS(({column} - stats.global_mean)
            / NULLIF(stats.global_std, 0)) DESC
        LIMIT 30
        """
        return self._execute(sql, label=f"zscore({table}.{column})")

    # ------------------------------------------------------------------
    # 4. IQR-based outlier detection in SQL
    # ------------------------------------------------------------------

    def iqr_outlier_detection(self, table: str, column: str) -> pd.DataFrame:
        """
        Tukey's fences for outlier detection.

        Lower fence = Q1 - k * IQR
        Upper fence = Q3 + k * IQR
        (k = 1.5 by default from config)

        More robust than z-score for skewed distributions.
        """
        k = self._iqr_multiplier
        sql = f"""
        WITH iqr_stats AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) AS q3,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) -
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS iqr
            FROM {table}
            WHERE {column} IS NOT NULL
        )
        SELECT
            t.{column},
            s.q1,
            s.q3,
            s.iqr,
            s.q1 - {k} * s.iqr                    AS lower_fence,
            s.q3 + {k} * s.iqr                    AS upper_fence,
            CASE
                WHEN t.{column} < s.q1 - {k} * s.iqr
                  OR t.{column} > s.q3 + {k} * s.iqr
                THEN 'outlier'
                ELSE 'normal'
            END                                    AS classification
        FROM {table} t, iqr_stats s
        WHERE t.{column} IS NOT NULL
        ORDER BY t.{column} DESC
        LIMIT 50
        """
        return self._execute(sql, label=f"iqr_outliers({table}.{column})")

    def iqr_outlier_summary(self, table: str, column: str) -> pd.DataFrame:
        """Summary count of outliers vs normal points."""
        k = self._iqr_multiplier
        sql = f"""
        WITH iqr_stats AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) AS q3,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) -
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS iqr
            FROM {table}
            WHERE {column} IS NOT NULL
        ),
        classified AS (
            SELECT
                CASE
                    WHEN t.{column} < s.q1 - {k} * s.iqr
                      OR t.{column} > s.q3 + {k} * s.iqr
                    THEN 'outlier'
                    ELSE 'normal'
                END AS classification
            FROM {table} t, iqr_stats s
            WHERE t.{column} IS NOT NULL
        )
        SELECT
            classification,
            COUNT(*) AS n,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
        FROM classified
        GROUP BY classification
        ORDER BY classification
        """
        return self._execute(sql, label=f"iqr_outlier_summary({table}.{column})")

    # ------------------------------------------------------------------
    # 5. Histogram with WIDTH_BUCKET
    # ------------------------------------------------------------------

    def histogram_buckets(self, table: str, column: str) -> pd.DataFrame:
        """
        Create histogram buckets using WIDTH_BUCKET.

        WIDTH_BUCKET(val, min, max, n_buckets) assigns each value to a
        bucket from 1..n_buckets based on equal-width intervals.
        Values outside [min, max] go to bucket 0 or n+1.
        """
        bins = self._histogram_bins
        sql = f"""
        WITH range_stats AS (
            SELECT
                MIN({column}) AS lo,
                MAX({column}) AS hi
            FROM {table}
            WHERE {column} IS NOT NULL
        ),
        bucketed AS (
            SELECT
                WIDTH_BUCKET({column}, s.lo, s.hi + 1e-10, {bins}) AS bucket,
                s.lo + (WIDTH_BUCKET({column}, s.lo, s.hi + 1e-10, {bins}) - 1)
                    * (s.hi - s.lo + 1e-10) / {bins}                AS bucket_low,
                s.lo + WIDTH_BUCKET({column}, s.lo, s.hi + 1e-10, {bins})
                    * (s.hi - s.lo + 1e-10) / {bins}                AS bucket_high
            FROM {table}, range_stats s
            WHERE {column} IS NOT NULL
        )
        SELECT
            bucket,
            ROUND(AVG(bucket_low), 2)  AS bucket_low,
            ROUND(AVG(bucket_high), 2) AS bucket_high,
            COUNT(*) AS frequency
        FROM bucketed
        GROUP BY bucket
        ORDER BY bucket
        """
        return self._execute(sql, label=f"histogram({table}.{column}, bins={bins})")

    # ------------------------------------------------------------------
    # 6. Funnel analysis
    # ------------------------------------------------------------------

    def funnel_analysis(self, funnel_table: str) -> pd.DataFrame:
        """
        Calculate conversion rates at each funnel step.

        Uses MAX step reached per user to determine which users completed
        each step. Measures step-to-step drop-off and cumulative conversion.
        """
        sql = f"""
        WITH user_max_step AS (
            -- Find the furthest step each user reached
            SELECT
                user_id,
                MAX(step) AS max_step,
                MAX(step_name) AS final_step_name
            FROM {funnel_table}
            GROUP BY user_id
        ),
        step_counts AS (
            -- Count users who reached each step or beyond
            SELECT
                step,
                step_name,
                COUNT(DISTINCT user_id) AS users_at_step
            FROM {funnel_table}
            GROUP BY step, step_name
        ),
        total_users AS (
            SELECT COUNT(DISTINCT user_id) AS total FROM {funnel_table}
        )
        SELECT
            sc.step,
            sc.step_name,
            sc.users_at_step,
            t.total                                              AS total_users,
            ROUND(sc.users_at_step * 100.0 / t.total, 2)        AS cumulative_conversion_pct,
            ROUND(sc.users_at_step * 100.0 /
                LAG(sc.users_at_step, 1, sc.users_at_step)
                    OVER (ORDER BY sc.step), 2)                  AS step_conversion_pct,
            t.total - sc.users_at_step                           AS dropped_off
        FROM step_counts sc, total_users t
        ORDER BY sc.step
        """
        return self._execute(sql, label="funnel_analysis")

    # ------------------------------------------------------------------
    # 7. Cohort retention analysis
    # ------------------------------------------------------------------

    def cohort_retention(self, cohort_table: str) -> pd.DataFrame:
        """
        Weekly cohort retention matrix.

        For each cohort (defined by their acquisition week), computes
        what fraction of users returned in weeks 1, 2, 3, ... after
        their acquisition week.

        This is a standard product analytics technique for measuring
        user retention and lifetime value trends.
        """
        sql = f"""
        WITH cohort_base AS (
            -- Get acquisition week and size for each cohort
            SELECT
                cohort_week,
                user_id,
                MIN(activity_week) AS acquired_week
            FROM {cohort_table}
            GROUP BY cohort_week, user_id
        ),
        cohort_sizes AS (
            SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
            FROM cohort_base
            GROUP BY cohort_week
        ),
        retention AS (
            SELECT
                cb.cohort_week,
                ct.activity_week - cb.cohort_week AS weeks_since_acquisition,
                COUNT(DISTINCT ct.user_id) AS retained_users
            FROM cohort_base cb
            JOIN {cohort_table} ct
                ON cb.user_id = ct.user_id
            GROUP BY cb.cohort_week, weeks_since_acquisition
        )
        SELECT
            r.cohort_week,
            r.weeks_since_acquisition,
            r.retained_users,
            cs.cohort_size,
            ROUND(r.retained_users * 100.0 / cs.cohort_size, 2) AS retention_rate_pct
        FROM retention r
        JOIN cohort_sizes cs ON r.cohort_week = cs.cohort_week
        WHERE r.weeks_since_acquisition >= 0
        ORDER BY r.cohort_week, r.weeks_since_acquisition
        """
        return self._execute(sql, label="cohort_retention")

    # ------------------------------------------------------------------
    # 8. SQL linear regression (REGR_ functions)
    # ------------------------------------------------------------------

    def sql_linear_regression(
        self, table: str, y_col: str, x_col: str
    ) -> pd.DataFrame:
        """
        Compute linear regression y = a + b*x entirely in SQL.

        DuckDB / SQL standard regression aggregate functions:
          REGR_SLOPE(y, x)      : OLS slope (b)
          REGR_INTERCEPT(y, x)  : OLS intercept (a)
          REGR_R2(y, x)         : coefficient of determination R²
          REGR_COUNT(y, x)      : number of non-null pairs
          REGR_AVGX, REGR_AVGY  : means of x and y
          REGR_SXX, REGR_SYY    : sum of squares for x and y
          REGR_SXY              : sum of cross products
        """
        sql = f"""
        SELECT
            REGR_SLOPE({y_col}, {x_col})      AS slope,
            REGR_INTERCEPT({y_col}, {x_col})  AS intercept,
            REGR_R2({y_col}, {x_col})         AS r_squared,
            SQRT(1 - REGR_R2({y_col}, {x_col}))   AS rmse_proxy,
            REGR_COUNT({y_col}, {x_col})      AS n_pairs,
            REGR_AVGX({y_col}, {x_col})       AS mean_x,
            REGR_AVGY({y_col}, {x_col})       AS mean_y,
            CORR({y_col}, {x_col})            AS pearson_r,
            -- Residual std approximation
            STDDEV_SAMP({y_col} - (
                REGR_SLOPE({y_col}, {x_col}) * {x_col}
                + REGR_INTERCEPT({y_col}, {x_col})
            ))                                AS residual_std
        FROM {table}
        WHERE {y_col} IS NOT NULL AND {x_col} IS NOT NULL
        """
        return self._execute(sql, label=f"regression({y_col} ~ {x_col})")

    # ------------------------------------------------------------------
    # 9. Category-level descriptive statistics
    # ------------------------------------------------------------------

    def grouped_statistics(
        self, table: str, group_col: str, value_col: str
    ) -> pd.DataFrame:
        """
        Per-group descriptive statistics using SQL.
        Useful for comparing distribution across categories.
        """
        sql = f"""
        SELECT
            {group_col},
            COUNT({value_col})                                           AS n,
            ROUND(AVG({value_col}), 2)                                   AS mean,
            ROUND(STDDEV_SAMP({value_col}), 2)                           AS std_dev,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                (ORDER BY {value_col}), 2)                               AS median,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                (ORDER BY {value_col}), 2)                               AS p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                (ORDER BY {value_col}), 2)                               AS p75,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {value_col})
                - PERCENTILE_CONT(0.25) WITHIN GROUP
                (ORDER BY {value_col}), 2)                               AS iqr,
            ROUND(MIN({value_col}), 2)                                   AS min_val,
            ROUND(MAX({value_col}), 2)                                   AS max_val,
            ROUND(SUM({value_col}), 2)                                   AS total
        FROM {table}
        WHERE {value_col} IS NOT NULL
        GROUP BY {group_col}
        ORDER BY mean DESC
        """
        return self._execute(
            sql, label=f"grouped_stats({table}, {group_col}, {value_col})"
        )

    def close(self) -> None:
        """Close DuckDB connection."""
        self._con.close()
        self._logger.info("DuckDB connection closed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== SQLStatisticsAnalyzer demo start ===")

    gen = SyntheticEcommerceDataGenerator(cfg, logger)
    analyzer = SQLStatisticsAnalyzer(cfg, logger)

    # --- Transactions ---
    txn_df = gen.transactions_df()
    analyzer.register_dataframe(txn_df, "transactions")

    print("\n=== 1. Descriptive Statistics (SQL) ===")
    desc = analyzer.descriptive_statistics("transactions", "revenue")
    print(desc.to_string(index=False))

    print("\n=== 2. Window Functions ===")
    win = analyzer.window_functions_demo("transactions", "revenue", n_tiles=4)
    print(win.head(10).to_string(index=False))

    print("\n=== 3. Z-Score Normalisation ===")
    zscores = analyzer.zscore_normalisation("transactions", "revenue")
    print(zscores.head(10).to_string(index=False))

    print("\n=== 4. IQR Outlier Detection Summary ===")
    outlier_summary = analyzer.iqr_outlier_summary("transactions", "revenue")
    print(outlier_summary.to_string(index=False))

    print("\n=== 5. Histogram Buckets ===")
    hist = analyzer.histogram_buckets("transactions", "revenue")
    print(hist.head(10).to_string(index=False))

    print("\n=== 6. Grouped Statistics by Category ===")
    grouped = analyzer.grouped_statistics("transactions", "category", "revenue")
    print(grouped.to_string(index=False))

    # --- Funnel ---
    funnel_df = gen.funnel_events_df()
    analyzer.register_dataframe(funnel_df, "funnel_events")

    print("\n=== 7. Funnel Analysis ===")
    funnel = analyzer.funnel_analysis("funnel_events")
    print(funnel.to_string(index=False))

    # --- Cohort ---
    cohort_df = gen.cohort_df()
    analyzer.register_dataframe(cohort_df, "cohort_data")

    print("\n=== 8. Cohort Retention (first 5 cohorts, first 5 weeks) ===")
    retention = analyzer.cohort_retention("cohort_data")
    pivot = retention[
        (retention["cohort_week"] < 5) & (retention["weeks_since_acquisition"] < 5)
    ].pivot(
        index="cohort_week",
        columns="weeks_since_acquisition",
        values="retention_rate_pct",
    )
    print(pivot.to_string())

    # --- Regression ---
    reg_df = gen.regression_df()
    analyzer.register_dataframe(reg_df, "ad_data")

    print("\n=== 9. SQL Linear Regression ===")
    reg_result = analyzer.sql_linear_regression("ad_data", "revenue", "ad_spend")
    print(reg_result.to_string(index=False))

    analyzer.close()
    logger.info("=== SQLStatisticsAnalyzer demo complete ===")


if __name__ == "__main__":
    main()
