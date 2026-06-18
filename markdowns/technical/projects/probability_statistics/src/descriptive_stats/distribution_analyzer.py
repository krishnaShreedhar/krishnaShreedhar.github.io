"""
distribution_analyzer.py

Fit and visualize statistical distributions (normal, log-normal, Poisson).
Saves plots to the configured output directory.

Concepts illustrated:
  - Distribution fitting via MLE (scipy.stats.fit)
  - Goodness-of-fit: Kolmogorov-Smirnov test
  - PDF / PMF overlay on histogram
  - QQ-plot (normality assessment)
  - Comparison of fitted vs empirical CDF
"""

from __future__ import annotations

import logging
import logging.handlers
import pathlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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

    logger = logging.getLogger("descriptive_stats.distribution_analyzer")
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
class FitResult:
    """Stores result of a single distribution fit."""
    distribution: str
    params: tuple
    ks_statistic: float
    ks_pvalue: float
    aic: float = field(default=0.0)

    def __str__(self) -> str:
        return (
            f"{self.distribution}: params={self.params}, "
            f"KS={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f}, "
            f"AIC={self.aic:.2f}"
        )


# ---------------------------------------------------------------------------
# DistributionAnalyzer
# ---------------------------------------------------------------------------

class DistributionAnalyzer:
    """
    Fits statistical distributions to empirical data and produces
    diagnostic visualizations.

    SRP: Handles fitting and visualisation only - no data generation.
    OCP: New distributions can be added by extending _DISTRIBUTIONS.
    """

    # Candidate continuous distributions to try
    _CONTINUOUS_DISTRIBUTIONS = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "exponential": stats.expon,
        "gamma": stats.gamma,
        "beta": stats.beta,
    }

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ds_cfg = cfg["descriptive_stats"]
        self._bins = self._ds_cfg["distribution_bins"]
        self._output_dir = pathlib.Path(cfg["data"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"DistributionAnalyzer initialised | bins={self._bins} | "
            f"output_dir={self._output_dir}"
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_distribution(
        self,
        data: np.ndarray,
        dist_name: str,
    ) -> FitResult:
        """
        Fit a named scipy distribution to *data* using MLE.
        Computes KS goodness-of-fit and AIC.

        AIC = 2k - 2*log(L)  where k = number of free parameters.
        """
        if dist_name not in self._CONTINUOUS_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{dist_name}'. "
                f"Choose from: {list(self._CONTINUOUS_DISTRIBUTIONS)}"
            )
        dist = self._CONTINUOUS_DISTRIBUTIONS[dist_name]
        self._logger.info(f"Fitting '{dist_name}' to data (n={len(data)})")

        params = dist.fit(data)
        k = len(params)  # number of parameters

        # Log-likelihood
        log_like = np.sum(dist.logpdf(data, *params))
        aic = 2 * k - 2 * log_like

        # KS test
        ks_stat, ks_p = stats.kstest(data, dist_name, args=params)

        result = FitResult(
            distribution=dist_name,
            params=params,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            aic=aic,
        )
        self._logger.info(f"Fit result | {result}")
        return result

    def fit_all_continuous(self, data: np.ndarray) -> list[FitResult]:
        """Fit all candidate continuous distributions and rank by AIC."""
        results = []
        for name in self._CONTINUOUS_DISTRIBUTIONS:
            try:
                r = self.fit_distribution(data, name)
                results.append(r)
            except Exception as exc:
                self._logger.warning(f"Could not fit '{name}': {exc}")
        results.sort(key=lambda r: r.aic)
        self._logger.info(
            "Distribution ranking by AIC: "
            + " | ".join(r.distribution for r in results)
        )
        return results

    def fit_poisson(self, count_data: np.ndarray) -> FitResult:
        """
        Fit a Poisson distribution (discrete) to count data.
        MLE for Poisson: lambda_hat = sample mean.
        """
        lam = float(np.mean(count_data))
        self._logger.info(
            f"Fitting Poisson | lambda_hat={lam:.4f} (n={len(count_data)})"
        )
        ks_stat, ks_p = stats.kstest(
            count_data, "poisson", args=(lam,)
        )
        k = 1
        log_like = np.sum(stats.poisson.logpmf(count_data.astype(int), lam))
        aic = 2 * k - 2 * log_like
        result = FitResult(
            distribution="poisson",
            params=(lam,),
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            aic=aic,
        )
        self._logger.info(f"Poisson fit | {result}")
        return result

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"Figure saved | {path}")
        return path

    def plot_histogram_with_fit(
        self,
        data: np.ndarray,
        fit_result: FitResult,
        label: str = "data",
    ) -> pathlib.Path:
        """
        Plot histogram overlaid with fitted PDF.
        """
        dist_name = fit_result.distribution
        dist = self._CONTINUOUS_DISTRIBUTIONS[dist_name]
        params = fit_result.params

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            data,
            bins=self._bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            label="Empirical",
            edgecolor="white",
        )
        x = np.linspace(data.min(), data.max(), 500)
        ax.plot(
            x,
            dist.pdf(x, *params),
            color="firebrick",
            linewidth=2,
            label=f"Fitted {dist_name}\n(KS p={fit_result.ks_pvalue:.3f}, AIC={fit_result.aic:.1f})",
        )
        ax.set_title(f"Distribution Fit: {label} — {dist_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"histogram_fit_{label}_{dist_name}")

    def plot_qq(self, data: np.ndarray, label: str = "data") -> pathlib.Path:
        """
        Normal QQ-plot.
        Points on the diagonal line imply normality.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        ax.plot(osm, osr, "o", color="steelblue", markersize=3, alpha=0.6, label="Data")
        x_line = np.array([osm[0], osm[-1]])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="firebrick",
            linewidth=2,
            label=f"Normal reference (R²={r**2:.4f})",
        )
        ax.set_title(f"Normal QQ-Plot: {label}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"qq_plot_{label}")

    def plot_empirical_cdf(
        self,
        data: np.ndarray,
        fit_result: FitResult,
        label: str = "data",
    ) -> pathlib.Path:
        """
        Empirical CDF vs. fitted theoretical CDF.
        """
        dist_name = fit_result.distribution
        dist = self._CONTINUOUS_DISTRIBUTIONS[dist_name]
        params = fit_result.params

        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        tcdf = dist.cdf(sorted_data, *params)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.step(sorted_data, ecdf, where="post", color="steelblue",
                label="ECDF", linewidth=2)
        ax.plot(sorted_data, tcdf, color="firebrick", linewidth=2,
                linestyle="--", label=f"Fitted CDF ({dist_name})")
        ax.set_title(f"Empirical vs. Theoretical CDF: {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"ecdf_{label}_{dist_name}")

    def plot_multi_distribution_comparison(
        self,
        data: np.ndarray,
        fits: list[FitResult],
        label: str = "data",
        top_n: int = 3,
    ) -> pathlib.Path:
        """
        Overlay historgam with the top-N fitted PDFs ranked by AIC.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            data,
            bins=self._bins,
            density=True,
            alpha=0.35,
            color="grey",
            label="Empirical",
            edgecolor="white",
        )
        colors = ["firebrick", "darkorange", "seagreen", "purple", "navy"]
        x = np.linspace(data.min(), data.max(), 500)
        for idx, fr in enumerate(fits[:top_n]):
            dist = self._CONTINUOUS_DISTRIBUTIONS[fr.distribution]
            try:
                pdf_vals = dist.pdf(x, *fr.params)
                ax.plot(
                    x,
                    pdf_vals,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    label=f"{fr.distribution} (AIC={fr.aic:.1f})",
                )
            except Exception as exc:
                self._logger.warning(
                    f"Could not plot PDF for {fr.distribution}: {exc}"
                )

        ax.set_title(f"Top-{top_n} Distribution Fits: {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"multi_dist_{label}")

    def plot_poisson_pmf(
        self,
        count_data: np.ndarray,
        fit_result: FitResult,
        label: str = "count_data",
    ) -> pathlib.Path:
        """Bar chart of observed counts vs fitted Poisson PMF."""
        lam = fit_result.params[0]
        max_val = int(count_data.max()) + 1
        k_vals = np.arange(0, max_val + 1)

        observed_freq = np.bincount(count_data.astype(int), minlength=max_val + 1)
        observed_prob = observed_freq / len(count_data)
        fitted_prob = stats.poisson.pmf(k_vals, lam)

        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.35
        ax.bar(k_vals - width / 2, observed_prob[:len(k_vals)], width,
               label="Observed", color="steelblue", alpha=0.7)
        ax.bar(k_vals + width / 2, fitted_prob, width,
               label=f"Poisson(λ={lam:.2f})", color="firebrick", alpha=0.7)
        ax.set_title(f"Poisson Fit: {label}")
        ax.set_xlabel("Count")
        ax.set_ylabel("Probability")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"poisson_pmf_{label}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== DistributionAnalyzer demo start ===")

    rng = np.random.default_rng(cfg["data"]["random_seed"])
    n = cfg["data"]["sample_size"]
    analyzer = DistributionAnalyzer(cfg, logger)

    # --- Normal data ---
    normal_data = rng.normal(loc=5.0, scale=1.5, size=n)
    fits_normal = analyzer.fit_all_continuous(normal_data)
    best_fit = fits_normal[0]
    analyzer.plot_histogram_with_fit(normal_data, best_fit, label="normal_data")
    analyzer.plot_qq(normal_data, label="normal_data")
    analyzer.plot_empirical_cdf(normal_data, best_fit, label="normal_data")
    analyzer.plot_multi_distribution_comparison(normal_data, fits_normal, label="normal_data")

    # --- Log-normal data ---
    lognormal_data = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    fits_ln = analyzer.fit_all_continuous(lognormal_data)
    analyzer.plot_histogram_with_fit(lognormal_data, fits_ln[0], label="lognormal_data")
    analyzer.plot_qq(lognormal_data, label="lognormal_data")

    # --- Poisson count data ---
    count_data = rng.poisson(lam=4.0, size=n)
    poisson_fit = analyzer.fit_poisson(count_data)
    analyzer.plot_poisson_pmf(count_data, poisson_fit, label="poisson_data")

    logger.info("=== DistributionAnalyzer demo complete ===")

    print("\n=== Distribution Fitting Results (Normal data) ===")
    for fr in fits_normal:
        print(f"  {fr}")

    print("\n=== Poisson Fit ===")
    print(f"  {poisson_fit}")


if __name__ == "__main__":
    main()
