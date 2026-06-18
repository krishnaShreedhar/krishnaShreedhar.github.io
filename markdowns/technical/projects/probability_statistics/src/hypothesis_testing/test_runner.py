"""
test_runner.py

Hypothesis testing module: parametric and non-parametric tests, A/B testing,
bootstrap confidence intervals, and multiple testing correction.

Concepts illustrated:
  - One-sample t-test (test a sample mean against a known value)
  - Two-sample Welch's t-test (unequal variance)
  - Chi-squared test of independence (categorical data)
  - Mann-Whitney U test (non-parametric alternative to t-test)
  - Two-proportion z-test (A/B testing for conversion rates)
  - Bootstrap confidence intervals (percentile method)
  - Multiple testing correction: Bonferroni and Benjamini-Hochberg (BH/FDR)
"""

from __future__ import annotations

import logging
import logging.handlers
import pathlib
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(cfg: dict[str, Any]) -> logging.Logger:
    log_cfg = cfg["logging"]
    log_file = pathlib.Path(log_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hypothesis_testing.test_runner")
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Structured result from any hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    degrees_of_freedom: float | None = None
    extra_info: dict[str, Any] | None = None

    def __str__(self) -> str:
        decision = "REJECT H0" if self.reject_null else "FAIL TO REJECT H0"
        parts = [
            f"[{self.test_name}] statistic={self.statistic:.4f}, "
            f"p={self.p_value:.6f}, alpha={self.alpha} => {decision}"
        ]
        if self.effect_size is not None:
            parts.append(f" | effect_size={self.effect_size:.4f}")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            parts.append(f" | CI=({lo:.4f}, {hi:.4f})")
        return "".join(parts)


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class TestRunner:
    """
    Runs hypothesis tests on provided data arrays.

    SRP: Performs statistical testing only.
    OCP: Each test is an independent method; new tests can be added without
         modifying existing ones.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ht_cfg = cfg["hypothesis_testing"]
        self._alpha = self._ht_cfg["alpha"]
        self._bootstrap_iters = self._ht_cfg["bootstrap_iterations"]
        self._alternative = self._ht_cfg["alternative"]
        self._output_dir = pathlib.Path(cfg["data"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"TestRunner initialised | alpha={self._alpha}, "
            f"bootstrap_iters={self._bootstrap_iters}, "
            f"alternative={self._alternative}"
        )

    # ------------------------------------------------------------------
    # One-sample t-test
    # ------------------------------------------------------------------

    def one_sample_ttest(
        self, data: np.ndarray, popmean: float
    ) -> TestResult:
        """
        H0: The sample mean equals *popmean*.
        H1: The sample mean != popmean (two-sided by default).

        Assumes data is approximately normally distributed.
        Uses Cohen's d as effect size: d = (x_bar - mu) / s
        """
        self._logger.info(
            f"One-sample t-test | n={len(data)}, popmean={popmean}"
        )
        t_stat, p_val = stats.ttest_1samp(data, popmean)
        d = (np.mean(data) - popmean) / np.std(data, ddof=1)
        df = len(data) - 1
        result = TestResult(
            test_name="One-sample t-test",
            statistic=t_stat,
            p_value=p_val,
            alpha=self._alpha,
            reject_null=p_val < self._alpha,
            effect_size=d,
            degrees_of_freedom=df,
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Two-sample Welch's t-test
    # ------------------------------------------------------------------

    def two_sample_welch_ttest(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> TestResult:
        """
        H0: mean(A) == mean(B).
        H1: mean(A) != mean(B).

        Welch's t-test does NOT assume equal variances (more robust than
        Student's t-test in practice). Uses Cohen's d for effect size.
        """
        self._logger.info(
            f"Welch's t-test | n_A={len(group_a)}, n_B={len(group_b)}"
        )
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
        # Pooled std for Cohen's d (Welch variant uses pooled sd)
        pooled_std = np.sqrt(
            (np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2
        )
        d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
        result = TestResult(
            test_name="Two-sample Welch t-test",
            statistic=t_stat,
            p_value=p_val,
            alpha=self._alpha,
            reject_null=p_val < self._alpha,
            effect_size=d,
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Chi-squared test of independence
    # ------------------------------------------------------------------

    def chi_squared_independence(
        self,
        contingency_table: np.ndarray,
    ) -> TestResult:
        """
        H0: Row and column variables are independent.
        H1: They are associated.

        Cramer's V used as effect size:
            V = sqrt(chi2 / (n * min(rows-1, cols-1)))
        """
        self._logger.info(
            f"Chi-squared independence test | "
            f"table shape={contingency_table.shape}"
        )
        chi2, p_val, df, expected = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum()
        k = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
        result = TestResult(
            test_name="Chi-squared independence",
            statistic=chi2,
            p_value=p_val,
            alpha=self._alpha,
            reject_null=p_val < self._alpha,
            effect_size=cramers_v,
            degrees_of_freedom=df,
            extra_info={"expected_frequencies": expected.tolist()},
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Mann-Whitney U test (non-parametric)
    # ------------------------------------------------------------------

    def mann_whitney_u(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> TestResult:
        """
        Non-parametric alternative to the two-sample t-test.
        H0: Distributions of A and B are equal.
        H1: One distribution is stochastically greater.

        Rank-biserial correlation used as effect size:
            r = 1 - 2U / (n_A * n_B)
        """
        self._logger.info(
            f"Mann-Whitney U | n_A={len(group_a)}, n_B={len(group_b)}"
        )
        u_stat, p_val = stats.mannwhitneyu(
            group_a, group_b, alternative="two-sided"
        )
        n_a, n_b = len(group_a), len(group_b)
        r = 1 - 2 * u_stat / (n_a * n_b)  # rank-biserial correlation
        result = TestResult(
            test_name="Mann-Whitney U",
            statistic=u_stat,
            p_value=p_val,
            alpha=self._alpha,
            reject_null=p_val < self._alpha,
            effect_size=r,
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Two-proportion z-test (A/B for rates)
    # ------------------------------------------------------------------

    def two_proportion_ztest(
        self,
        conversions_a: int,
        n_a: int,
        conversions_b: int,
        n_b: int,
    ) -> TestResult:
        """
        Test whether two conversion rates (proportions) differ.
        H0: p_A == p_B
        H1: p_A != p_B

        Pooled proportion estimate under H0.
        Cohen's h used as effect size:
            h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        """
        self._logger.info(
            f"Two-proportion z-test | "
            f"A={conversions_a}/{n_a}, B={conversions_b}/{n_b}"
        )
        p_a = conversions_a / n_a
        p_b = conversions_b / n_b
        p_pool = (conversions_a + conversions_b) / (n_a + n_b)

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        z_stat = (p_a - p_b) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Cohen's h
        h = 2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b))

        result = TestResult(
            test_name="Two-proportion z-test",
            statistic=z_stat,
            p_value=p_val,
            alpha=self._alpha,
            reject_null=p_val < self._alpha,
            effect_size=h,
            extra_info={
                "rate_a": p_a,
                "rate_b": p_b,
                "pooled_rate": p_pool,
                "absolute_lift": p_b - p_a,
                "relative_lift_pct": (p_b - p_a) / p_a * 100,
            },
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Bootstrap confidence interval
    # ------------------------------------------------------------------

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_fn: Any = np.mean,
        seed: int | None = None,
    ) -> tuple[float, float, float]:
        """
        Percentile bootstrap CI for any statistic.

        Returns (observed_stat, lower_bound, upper_bound).
        The percentile method is simple and effective for large samples.
        """
        rng = np.random.default_rng(seed or self._cfg["data"]["random_seed"])
        observed = statistic_fn(data)
        n = len(data)
        boot_stats = np.empty(self._bootstrap_iters)
        for i in range(self._bootstrap_iters):
            resample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic_fn(resample)

        alpha = self._alpha
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        self._logger.info(
            f"Bootstrap CI (B={self._bootstrap_iters}, alpha={alpha}) | "
            f"observed={observed:.4f}, CI=({lower:.4f}, {upper:.4f})"
        )
        return float(observed), lower, upper

    # ------------------------------------------------------------------
    # Multiple testing correction
    # ------------------------------------------------------------------

    def bonferroni_correction(
        self,
        p_values: list[float],
    ) -> list[bool]:
        """
        Bonferroni correction: adjusted alpha = alpha / m.
        Controls Family-Wise Error Rate (FWER).
        Conservative - suitable when any false positive is unacceptable.
        """
        m = len(p_values)
        adjusted_alpha = self._alpha / m
        rejected = [p < adjusted_alpha for p in p_values]
        self._logger.info(
            f"Bonferroni | m={m}, adjusted_alpha={adjusted_alpha:.6f}, "
            f"rejections={sum(rejected)}/{m}"
        )
        return rejected

    def benjamini_hochberg(
        self,
        p_values: list[float],
    ) -> tuple[list[bool], list[float]]:
        """
        Benjamini-Hochberg FDR procedure.
        Controls the expected proportion of false discoveries.
        More powerful than Bonferroni when many tests are performed.

        BH procedure:
          1. Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)
          2. Find largest k s.t. p_(k) <= k/m * alpha
          3. Reject hypotheses 1..k
        Returns (rejected_flags, adjusted_p_values).
        """
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        rejected = [False] * m
        adj_p = [0.0] * m

        # Compute adjusted p-values (BH-adjusted)
        # adj_p[i] = min(p_i * m / rank, 1)
        # Apply cumulative minimum from the right for monotonicity
        adj_sorted = [p * m / (rank + 1) for rank, (_, p) in enumerate(indexed)]
        # Ensure monotone: scan from right
        for i in range(len(adj_sorted) - 2, -1, -1):
            adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
        adj_sorted = [min(v, 1.0) for v in adj_sorted]

        # Map back to original order
        for rank, (orig_idx, _) in enumerate(indexed):
            adj_p[orig_idx] = adj_sorted[rank]
            rejected[orig_idx] = adj_p[orig_idx] < self._alpha

        self._logger.info(
            f"Benjamini-Hochberg | m={m}, alpha={self._alpha}, "
            f"rejections={sum(rejected)}/{m}"
        )
        return rejected, adj_p

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"Figure saved | {path}")
        return path

    def plot_pvalue_distribution(
        self,
        p_values: list[float],
        rejected_bh: list[bool],
        label: str = "tests",
    ) -> pathlib.Path:
        """
        Histogram of p-values coloured by BH rejection status.
        A uniform distribution under H0 should look flat.
        """
        p_arr = np.array(p_values)
        rejected_arr = np.array(rejected_bh)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(p_arr[~rejected_arr], bins=20, color="steelblue",
                alpha=0.7, label="Not rejected")
        ax.hist(p_arr[rejected_arr], bins=20, color="firebrick",
                alpha=0.7, label="Rejected (BH)")
        ax.axvline(self._alpha, color="black", linestyle="--",
                   label=f"alpha={self._alpha}")
        ax.set_title(f"P-value Distribution: {label}")
        ax.set_xlabel("p-value")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"pvalue_dist_{label}")

    def plot_bootstrap_distribution(
        self,
        boot_stats: np.ndarray,
        observed: float,
        ci_lower: float,
        ci_upper: float,
        label: str = "statistic",
    ) -> pathlib.Path:
        """
        Plot bootstrap sampling distribution with confidence interval.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(boot_stats, bins=50, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
        ax.axvline(observed, color="black", linewidth=2,
                   label=f"Observed = {observed:.4f}")
        ax.axvline(ci_lower, color="firebrick", linestyle="--",
                   label=f"CI lower = {ci_lower:.4f}")
        ax.axvline(ci_upper, color="firebrick", linestyle="--",
                   label=f"CI upper = {ci_upper:.4f}")
        ax.set_title(f"Bootstrap Distribution: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"bootstrap_{label}")


# ---------------------------------------------------------------------------
# Synthetic data for demonstration
# ---------------------------------------------------------------------------

class HypothesisTestDataGenerator:
    """Generate synthetic datasets tailored to each test type."""

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._n = cfg["data"]["sample_size"]
        self._seed = cfg["data"]["random_seed"]
        self._rng = np.random.default_rng(self._seed)

    def control_treatment_continuous(
        self,
        control_mean: float = 50.0,
        treatment_effect: float = 2.0,
        std: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two groups: control and treatment (continuous outcome)."""
        control = self._rng.normal(control_mean, std, self._n // 2)
        treatment = self._rng.normal(control_mean + treatment_effect, std, self._n // 2)
        self._logger.info(
            f"Generated control/treatment | effect={treatment_effect}, "
            f"std={std}, n={self._n // 2} each"
        )
        return control, treatment

    def contingency_table(self) -> np.ndarray:
        """2x2 contingency table for chi-squared test."""
        # Example: ad campaign (shown/not) x conversion (yes/no)
        table = np.array([[200, 150], [80, 220]])
        self._logger.info(f"Contingency table:\n{table}")
        return table

    def ab_test_conversions(
        self,
        rate_a: float = 0.10,
        lift: float = 0.02,
    ) -> tuple[int, int, int, int]:
        """Simulate A/B test conversion counts."""
        n_a = self._n // 2
        n_b = self._n // 2
        conv_a = int(self._rng.binomial(n_a, rate_a))
        conv_b = int(self._rng.binomial(n_b, rate_a + lift))
        self._logger.info(
            f"A/B conversions | A={conv_a}/{n_a}={conv_a/n_a:.4f}, "
            f"B={conv_b}/{n_b}={conv_b/n_b:.4f} (true lift={lift})"
        )
        return conv_a, n_a, conv_b, n_b

    def multiple_pvalues(
        self, m: int = 50, n_true_effects: int = 10
    ) -> list[float]:
        """
        Simulate m p-values: n_true_effects come from true effects
        (Beta(0.5, 5)), rest are uniform [0,1] under H0.
        """
        null_p = self._rng.uniform(0, 1, m - n_true_effects).tolist()
        effect_p = self._rng.beta(0.5, 5, n_true_effects).tolist()
        p_values = null_p + effect_p
        self._rng.shuffle(p_values)
        self._logger.info(
            f"Generated {m} p-values | {n_true_effects} true effects"
        )
        return p_values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== TestRunner demo start ===")

    runner = TestRunner(cfg, logger)
    gen = HypothesisTestDataGenerator(cfg, logger)

    # 1. One-sample t-test
    data = np.random.default_rng(42).normal(52.0, 10.0, 200)
    r1 = runner.one_sample_ttest(data, popmean=50.0)
    print(f"\n{r1}")

    # 2. Two-sample Welch's t-test
    control, treatment = gen.control_treatment_continuous(
        control_mean=50.0, treatment_effect=2.5
    )
    r2 = runner.two_sample_welch_ttest(control, treatment)
    print(f"\n{r2}")

    # 3. Chi-squared test
    table = gen.contingency_table()
    r3 = runner.chi_squared_independence(table)
    print(f"\n{r3}")

    # 4. Mann-Whitney U
    r4 = runner.mann_whitney_u(control, treatment)
    print(f"\n{r4}")

    # 5. Two-proportion z-test
    conv_a, n_a, conv_b, n_b = gen.ab_test_conversions(rate_a=0.10, lift=0.02)
    r5 = runner.two_proportion_ztest(conv_a, n_a, conv_b, n_b)
    print(f"\n{r5}")
    if r5.extra_info:
        print(
            f"  Rate A={r5.extra_info['rate_a']:.4f}, "
            f"Rate B={r5.extra_info['rate_b']:.4f}, "
            f"Absolute lift={r5.extra_info['absolute_lift']:.4f}, "
            f"Relative lift={r5.extra_info['relative_lift_pct']:.2f}%"
        )

    # 6. Bootstrap CI for mean
    observed, ci_lo, ci_hi = runner.bootstrap_confidence_interval(
        control, statistic_fn=np.mean
    )
    print(
        f"\nBootstrap CI (mean): observed={observed:.4f}, "
        f"95% CI=({ci_lo:.4f}, {ci_hi:.4f})"
    )

    # 7. Multiple testing correction
    p_values = gen.multiple_pvalues(m=50, n_true_effects=10)
    bonf_rejected = runner.bonferroni_correction(p_values)
    bh_rejected, adj_p = runner.benjamini_hochberg(p_values)
    print(
        f"\nMultiple testing (m=50): "
        f"Bonferroni rejections={sum(bonf_rejected)}, "
        f"BH rejections={sum(bh_rejected)}"
    )
    runner.plot_pvalue_distribution(p_values, bh_rejected, label="multiple_tests")

    logger.info("=== TestRunner demo complete ===")


if __name__ == "__main__":
    main()
