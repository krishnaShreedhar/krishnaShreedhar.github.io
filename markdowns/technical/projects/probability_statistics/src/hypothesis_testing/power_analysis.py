"""
power_analysis.py

Statistical power analysis and sample size calculation.

Concepts illustrated:
  - Statistical power = P(reject H0 | H1 is true) = 1 - beta
  - Type I error (alpha): false positive rate
  - Type II error (beta): false negative rate
  - Effect size: Cohen's d (continuous), Cohen's h (proportions)
  - Sample size determination for t-test, z-test, chi-squared
  - Power curves (power vs sample size, power vs effect size)
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
import yaml
from scipy import stats


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(cfg: dict[str, Any]) -> logging.Logger:
    log_cfg = cfg["logging"]
    log_file = pathlib.Path(log_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hypothesis_testing.power_analysis")
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
class PowerAnalysisResult:
    """Result of a power / sample-size calculation."""
    test_name: str
    alpha: float
    power: float
    effect_size: float
    sample_size_per_group: int
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.test_name}] alpha={self.alpha}, power={self.power}, "
            f"effect_size={self.effect_size:.4f} => "
            f"n_per_group={self.sample_size_per_group}"
            + (f" | {self.notes}" if self.notes else "")
        )


# ---------------------------------------------------------------------------
# PowerAnalyzer
# ---------------------------------------------------------------------------

class PowerAnalyzer:
    """
    Calculates required sample sizes and power for common statistical tests.

    SRP: Only responsible for power analysis calculations and visualisation.
    DIP: Accepts configuration via injected cfg dict.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ht_cfg = cfg["hypothesis_testing"]
        self._alpha = self._ht_cfg["alpha"]
        self._power = self._ht_cfg["power"]
        self._effect_size = self._ht_cfg["effect_size"]
        self._baseline_rate = self._ht_cfg["baseline_rate"]
        self._output_dir = pathlib.Path(cfg["data"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"PowerAnalyzer initialised | alpha={self._alpha}, "
            f"power={self._power}, effect_size={self._effect_size}"
        )

    # ------------------------------------------------------------------
    # Core analytical formulas
    # ------------------------------------------------------------------

    def _z_alpha(self, two_sided: bool = True) -> float:
        """Critical value for given alpha."""
        a = self._alpha / 2 if two_sided else self._alpha
        return float(stats.norm.ppf(1 - a))

    def _z_beta(self) -> float:
        """Critical value for given power (1 - beta)."""
        return float(stats.norm.ppf(self._power))

    # ------------------------------------------------------------------
    # Two-sample t-test sample size
    # ------------------------------------------------------------------

    def sample_size_ttest(
        self,
        cohens_d: float | None = None,
    ) -> PowerAnalysisResult:
        """
        Calculate required n per group for a two-sample t-test.

        Formula (approximate, large n):
            n = (z_alpha/2 + z_beta)^2 * 2 / d^2

        Cohen's d benchmarks:
            Small=0.2, Medium=0.5, Large=0.8
        """
        d = cohens_d if cohens_d is not None else self._effect_size
        z_a = self._z_alpha(two_sided=True)
        z_b = self._z_beta()
        n = int(np.ceil(2 * (z_a + z_b) ** 2 / d ** 2))
        result = PowerAnalysisResult(
            test_name="Two-sample t-test",
            alpha=self._alpha,
            power=self._power,
            effect_size=d,
            sample_size_per_group=n,
            notes="Cohen's d — small=0.2, medium=0.5, large=0.8",
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Two-proportion z-test sample size
    # ------------------------------------------------------------------

    def sample_size_proportion_ztest(
        self,
        baseline_rate: float | None = None,
        min_detectable_effect: float | None = None,
    ) -> PowerAnalysisResult:
        """
        Required n per group for a two-proportion z-test (A/B test).

        Formula:
            n = (z_alpha/2 * sqrt(2*p_bar*(1-p_bar))
                 + z_beta * sqrt(p1*(1-p1) + p2*(1-p2)))^2
                / (p2 - p1)^2

        where p_bar = (p1 + p2) / 2.
        """
        p1 = baseline_rate if baseline_rate is not None else self._baseline_rate
        mde = min_detectable_effect if min_detectable_effect is not None else self._effect_size
        p2 = p1 + mde

        if not (0 < p1 < 1 and 0 < p2 < 1):
            raise ValueError(
                f"Rates must be in (0,1); got p1={p1:.4f}, p2={p2:.4f}"
            )

        p_bar = (p1 + p2) / 2
        z_a = self._z_alpha(two_sided=True)
        z_b = self._z_beta()

        numerator = (
            z_a * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        n = int(np.ceil(numerator / (p2 - p1) ** 2))

        # Cohen's h for reference
        h = abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))

        result = PowerAnalysisResult(
            test_name="Two-proportion z-test",
            alpha=self._alpha,
            power=self._power,
            effect_size=h,
            sample_size_per_group=n,
            notes=f"baseline={p1:.4f}, p2={p2:.4f}, MDE={mde:.4f}",
        )
        self._logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Compute achieved power for given n
    # ------------------------------------------------------------------

    def achieved_power_ttest(self, n_per_group: int, cohens_d: float) -> float:
        """
        Compute statistical power for a given sample size and effect size.

        Power = Phi(|d| * sqrt(n/2) - z_alpha/2)   (two-sided)
        where Phi is the normal CDF.
        """
        z_a = self._z_alpha(two_sided=True)
        ncp = abs(cohens_d) * np.sqrt(n_per_group / 2)
        power = float(stats.norm.cdf(ncp - z_a))
        self._logger.debug(
            f"Achieved power | n={n_per_group}, d={cohens_d:.4f} => power={power:.4f}"
        )
        return power

    def achieved_power_proportion(
        self, n_per_group: int, p1: float, p2: float
    ) -> float:
        """Achieved power for two-proportion z-test."""
        p_bar = (p1 + p2) / 2
        z_a = self._z_alpha(two_sided=True)
        se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
        se_alt = np.sqrt(
            (p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group
        )
        ncp = abs(p2 - p1) / se_alt
        power = float(stats.norm.cdf(ncp - z_a * se_null / se_alt))
        self._logger.debug(
            f"Achieved power (proportion) | n={n_per_group}, "
            f"p1={p1:.4f}, p2={p2:.4f} => power={power:.4f}"
        )
        return power

    # ------------------------------------------------------------------
    # Power curves
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"Figure saved | {path}")
        return path

    def plot_power_vs_n(
        self,
        effect_sizes: list[float],
        n_range: tuple[int, int] = (20, 500),
    ) -> pathlib.Path:
        """
        Power curves: power as a function of n for multiple effect sizes.
        Horizontal dashed line at target power.
        """
        n_vals = np.arange(n_range[0], n_range[1] + 1, 5)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(effect_sizes)))

        for d, color in zip(effect_sizes, colors):
            powers = [self.achieved_power_ttest(n, d) for n in n_vals]
            ax.plot(n_vals, powers, color=color, linewidth=2,
                    label=f"d={d:.2f}")

        ax.axhline(
            self._power, color="firebrick", linestyle="--", linewidth=1.5,
            label=f"Target power={self._power}"
        )
        ax.axhline(
            self._alpha, color="grey", linestyle=":", linewidth=1.2,
            label=f"alpha={self._alpha}"
        )
        ax.set_title("Power Curves: Two-Sample t-Test")
        ax.set_xlabel("Sample Size per Group (n)")
        ax.set_ylabel("Statistical Power (1 - β)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, "power_vs_n_ttest")

    def plot_power_vs_effect_size(
        self,
        n_values: list[int],
        d_range: tuple[float, float] = (0.05, 1.5),
    ) -> pathlib.Path:
        """Power as a function of effect size for multiple sample sizes."""
        d_vals = np.linspace(d_range[0], d_range[1], 200)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_values)))

        for n, color in zip(n_values, colors):
            powers = [self.achieved_power_ttest(n, d) for d in d_vals]
            ax.plot(d_vals, powers, color=color, linewidth=2, label=f"n={n}")

        ax.axhline(
            self._power, color="firebrick", linestyle="--", linewidth=1.5,
            label=f"Target power={self._power}"
        )
        ax.set_title("Power vs. Effect Size: Two-Sample t-Test")
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_ylabel("Statistical Power (1 - β)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_figure(fig, "power_vs_effect_size_ttest")

    def plot_sample_size_heatmap(
        self,
        alpha_range: list[float],
        power_range: list[float],
        cohens_d: float = 0.5,
    ) -> pathlib.Path:
        """
        Heatmap of required sample size for a grid of (alpha, power) values.
        Shows how sensitivity choices affect data requirements.
        """
        n_matrix = np.zeros((len(power_range), len(alpha_range)))
        for i, pwr in enumerate(power_range):
            for j, alp in enumerate(alpha_range):
                z_a = stats.norm.ppf(1 - alp / 2)
                z_b = stats.norm.ppf(pwr)
                n_matrix[i, j] = int(np.ceil(2 * (z_a + z_b) ** 2 / cohens_d ** 2))

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(n_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(alpha_range)))
        ax.set_xticklabels([f"{a:.3f}" for a in alpha_range])
        ax.set_yticks(range(len(power_range)))
        ax.set_yticklabels([f"{p:.2f}" for p in power_range])
        ax.set_xlabel("Alpha (Type I error rate)")
        ax.set_ylabel("Power (1 - Type II error rate)")
        ax.set_title(
            f"Required n per Group | Cohen's d = {cohens_d}\n"
            "(Two-sample t-test)"
        )
        fig.colorbar(im, ax=ax, label="n per group")
        # Annotate cells
        for i in range(len(power_range)):
            for j in range(len(alpha_range)):
                ax.text(
                    j, i, f"{int(n_matrix[i, j])}",
                    ha="center", va="center", fontsize=8,
                    color="black" if n_matrix[i, j] < n_matrix.max() * 0.7 else "white",
                )
        fig.tight_layout()
        return self._save_figure(fig, "sample_size_heatmap")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== PowerAnalyzer demo start ===")
    analyzer = PowerAnalyzer(cfg, logger)

    # Sample size for t-test under various effect sizes
    print("\n=== Sample Size Requirements (Two-sample t-test) ===")
    for d in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        result = analyzer.sample_size_ttest(cohens_d=d)
        print(f"  d={d:.1f}: n_per_group={result.sample_size_per_group}")

    # Sample size for A/B proportion test
    print("\n=== Sample Size Requirements (Two-proportion z-test) ===")
    for mde in [0.01, 0.02, 0.05, 0.10]:
        result = analyzer.sample_size_proportion_ztest(
            baseline_rate=0.10, min_detectable_effect=mde
        )
        print(
            f"  baseline=10%, MDE={mde:.2f}: "
            f"n_per_group={result.sample_size_per_group}"
        )

    # Power curves
    analyzer.plot_power_vs_n(
        effect_sizes=[0.2, 0.3, 0.5, 0.8],
        n_range=(20, 400),
    )
    analyzer.plot_power_vs_effect_size(
        n_values=[50, 100, 200, 500],
        d_range=(0.05, 1.5),
    )
    analyzer.plot_sample_size_heatmap(
        alpha_range=[0.001, 0.005, 0.01, 0.05, 0.10],
        power_range=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        cohens_d=0.5,
    )

    logger.info("=== PowerAnalyzer demo complete ===")


if __name__ == "__main__":
    main()
