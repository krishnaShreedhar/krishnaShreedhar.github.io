"""
bayesian_ab_test.py

Bayesian A/B testing using the Beta-Binomial conjugate model.

Concepts illustrated:
  - Beta distribution as prior over conversion rates
  - Conjugate update: posterior Beta(alpha + successes, beta + failures)
  - Monte Carlo estimation of P(B > A) and P(B > A + threshold)
  - Expected Loss (risk-minimisation decision criterion)
  - Credible intervals vs. frequentist confidence intervals
  - Sequential testing: no fixed sample size required
  - Comparison with frequentist two-proportion z-test
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

    logger = logging.getLogger("bayesian_analysis.bayesian_ab_test")
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
class BayesianTestResult:
    """Results from a Bayesian A/B test."""
    # Observed data
    n_a: int
    conversions_a: int
    n_b: int
    conversions_b: int
    # Posterior parameters
    posterior_alpha_a: float
    posterior_beta_a: float
    posterior_alpha_b: float
    posterior_beta_b: float
    # Posterior statistics
    mean_a: float
    mean_b: float
    credible_lower_a: float
    credible_upper_a: float
    credible_lower_b: float
    credible_upper_b: float
    # Decision metrics
    prob_b_beats_a: float
    expected_loss_a: float
    expected_loss_b: float
    # Frequentist comparison
    frequentist_z: float
    frequentist_p: float

    def __str__(self) -> str:
        lines = [
            f"=== Bayesian A/B Test Result ===",
            f"  Group A: {self.conversions_a}/{self.n_a} = {self.conversions_a/self.n_a:.4f}",
            f"  Group B: {self.conversions_b}/{self.n_b} = {self.conversions_b/self.n_b:.4f}",
            f"  Posterior mean A: {self.mean_a:.4f}, "
            f"  {int(100*(self.credible_upper_a - self.credible_lower_a)*100/100):.0f}% CI: "
            f"({self.credible_lower_a:.4f}, {self.credible_upper_a:.4f})",
            f"  Posterior mean B: {self.mean_b:.4f}, "
            f"  CI: ({self.credible_lower_b:.4f}, {self.credible_upper_b:.4f})",
            f"  P(B > A)             = {self.prob_b_beats_a:.4f}",
            f"  Expected loss(A)     = {self.expected_loss_a:.6f}",
            f"  Expected loss(B)     = {self.expected_loss_b:.6f}",
            f"  Frequentist: z={self.frequentist_z:.4f}, p={self.frequentist_p:.6f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BayesianABTester
# ---------------------------------------------------------------------------

class BayesianABTester:
    """
    Implements Bayesian A/B testing using the Beta-Binomial conjugate model.

    The Beta distribution is the conjugate prior for the Binomial likelihood.
    Prior: p ~ Beta(alpha_0, beta_0)
    Likelihood: X | p ~ Binomial(n, p)
    Posterior: p | X ~ Beta(alpha_0 + X, beta_0 + n - X)

    SRP: Handles only Bayesian A/B test logic and visualisation.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ab_cfg = cfg["bayesian_ab"]
        self._prior_alpha = self._ab_cfg["prior_alpha"]
        self._prior_beta = self._ab_cfg["prior_beta"]
        self._mc_samples = self._ab_cfg["monte_carlo_samples"]
        self._ci_level = self._ab_cfg["credible_interval"]
        self._output_dir = pathlib.Path(cfg["data"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])
        self._logger.info(
            f"BayesianABTester | prior=Beta({self._prior_alpha}, {self._prior_beta}), "
            f"MC_samples={self._mc_samples}, CI={self._ci_level}"
        )

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------

    def posterior_params(
        self,
        conversions: int,
        n: int,
    ) -> tuple[float, float]:
        """
        Compute posterior Beta parameters after observing data.
        posterior_alpha = prior_alpha + conversions
        posterior_beta  = prior_beta  + (n - conversions)
        """
        alpha_post = self._prior_alpha + conversions
        beta_post = self._prior_beta + (n - conversions)
        self._logger.debug(
            f"Posterior | {conversions}/{n} => "
            f"Beta({alpha_post}, {beta_post})"
        )
        return alpha_post, beta_post

    def credible_interval(
        self, alpha_post: float, beta_post: float
    ) -> tuple[float, float]:
        """
        Compute the highest-density credible interval at the configured level.
        For Beta, this is the equal-tailed interval (HDI approximation).
        """
        lo = (1 - self._ci_level) / 2
        hi = 1 - lo
        lower = float(stats.beta.ppf(lo, alpha_post, beta_post))
        upper = float(stats.beta.ppf(hi, alpha_post, beta_post))
        return lower, upper

    # ------------------------------------------------------------------
    # Monte Carlo estimation
    # ------------------------------------------------------------------

    def prob_b_beats_a(
        self,
        alpha_a: float,
        beta_a: float,
        alpha_b: float,
        beta_b: float,
        threshold: float = 0.0,
    ) -> float:
        """
        Estimate P(p_B > p_A + threshold) via Monte Carlo sampling.

        This is the probability that B's true rate exceeds A's by at least
        *threshold* (e.g., threshold=0.01 for "B is at least 1pp better").
        """
        samples_a = self._rng.beta(alpha_a, beta_a, self._mc_samples)
        samples_b = self._rng.beta(alpha_b, beta_b, self._mc_samples)
        prob = float(np.mean(samples_b > samples_a + threshold))
        self._logger.info(
            f"P(B > A + {threshold:.4f}) = {prob:.4f} "
            f"(MC samples={self._mc_samples})"
        )
        return prob

    def expected_loss(
        self,
        alpha_a: float,
        beta_a: float,
        alpha_b: float,
        beta_b: float,
    ) -> tuple[float, float]:
        """
        Expected loss for choosing A over B and B over A.

        E[loss(choose A)] = E[max(0, p_B - p_A)]
        E[loss(choose B)] = E[max(0, p_A - p_B)]

        A good decision rule: choose the option with lower expected loss.
        """
        samples_a = self._rng.beta(alpha_a, beta_a, self._mc_samples)
        samples_b = self._rng.beta(alpha_b, beta_b, self._mc_samples)
        loss_a = float(np.mean(np.maximum(0, samples_b - samples_a)))
        loss_b = float(np.mean(np.maximum(0, samples_a - samples_b)))
        self._logger.info(
            f"Expected loss | E[loss(A)]={loss_a:.6f}, "
            f"E[loss(B)]={loss_b:.6f}"
        )
        return loss_a, loss_b

    # ------------------------------------------------------------------
    # Full test
    # ------------------------------------------------------------------

    def run_test(
        self,
        conversions_a: int,
        n_a: int,
        conversions_b: int,
        n_b: int,
    ) -> BayesianTestResult:
        """
        Run a complete Bayesian A/B test and return structured results.
        Also computes frequentist two-proportion z-test for comparison.
        """
        self._logger.info(
            f"Running Bayesian A/B test | "
            f"A={conversions_a}/{n_a}, B={conversions_b}/{n_b}"
        )

        # Posterior parameters
        al_a, be_a = self.posterior_params(conversions_a, n_a)
        al_b, be_b = self.posterior_params(conversions_b, n_b)

        # Posterior means = alpha / (alpha + beta)
        mean_a = al_a / (al_a + be_a)
        mean_b = al_b / (al_b + be_b)

        # Credible intervals
        ci_lo_a, ci_hi_a = self.credible_interval(al_a, be_a)
        ci_lo_b, ci_hi_b = self.credible_interval(al_b, be_b)

        # Decision metrics
        p_b_beats_a = self.prob_b_beats_a(al_a, be_a, al_b, be_b)
        loss_a, loss_b = self.expected_loss(al_a, be_a, al_b, be_b)

        # Frequentist comparison
        p1 = conversions_a / n_a
        p2 = conversions_b / n_b
        p_pool = (conversions_a + conversions_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        z_freq = (p2 - p1) / se if se > 0 else 0.0
        p_freq = 2 * (1 - stats.norm.cdf(abs(z_freq)))

        result = BayesianTestResult(
            n_a=n_a,
            conversions_a=conversions_a,
            n_b=n_b,
            conversions_b=conversions_b,
            posterior_alpha_a=al_a,
            posterior_beta_a=be_a,
            posterior_alpha_b=al_b,
            posterior_beta_b=be_b,
            mean_a=mean_a,
            mean_b=mean_b,
            credible_lower_a=ci_lo_a,
            credible_upper_a=ci_hi_a,
            credible_lower_b=ci_lo_b,
            credible_upper_b=ci_hi_b,
            prob_b_beats_a=p_b_beats_a,
            expected_loss_a=loss_a,
            expected_loss_b=loss_b,
            frequentist_z=z_freq,
            frequentist_p=p_freq,
        )
        self._logger.info(
            f"Test complete | P(B>A)={p_b_beats_a:.4f}, "
            f"freq_p={p_freq:.6f}"
        )
        return result

    # ------------------------------------------------------------------
    # Sequential testing
    # ------------------------------------------------------------------

    def sequential_analysis(
        self,
        true_rate_a: float,
        true_rate_b: float,
        stopping_threshold: float = 0.95,
        max_n_per_group: int = 2000,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        """
        Simulate sequential Bayesian A/B test.
        Stop when P(B > A) > stopping_threshold or P(A > B) > stopping_threshold.

        Returns history of P(B>A) and sample sizes at each step.
        """
        self._logger.info(
            f"Sequential analysis | true_rate_A={true_rate_a}, "
            f"true_rate_B={true_rate_b}, threshold={stopping_threshold}"
        )
        history = []
        conv_a, total_a = 0, 0
        conv_b, total_b = 0, 0
        stopped_at = None

        step_rng = np.random.default_rng(self._cfg["data"]["random_seed"] + 1)

        while total_a < max_n_per_group:
            new_a = step_rng.binomial(batch_size, true_rate_a)
            new_b = step_rng.binomial(batch_size, true_rate_b)
            conv_a += new_a
            conv_b += new_b
            total_a += batch_size
            total_b += batch_size

            al_a, be_a = self.posterior_params(conv_a, total_a)
            al_b, be_b = self.posterior_params(conv_b, total_b)
            p = self.prob_b_beats_a(al_a, be_a, al_b, be_b)
            history.append(
                {"n_per_group": total_a, "prob_b_beats_a": p,
                 "rate_a": conv_a / total_a, "rate_b": conv_b / total_b}
            )
            if (p > stopping_threshold or p < (1 - stopping_threshold)) and stopped_at is None:
                stopped_at = total_a
                self._logger.info(
                    f"Sequential: stopped at n={total_a} | P(B>A)={p:.4f}"
                )
                break

        return {"history": history, "stopped_at": stopped_at}

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"Figure saved | {path}")
        return path

    def plot_posteriors(
        self, result: BayesianTestResult, label: str = "ab_test"
    ) -> pathlib.Path:
        """
        Plot posterior Beta distributions for A and B with credible intervals.
        """
        x = np.linspace(0, 0.5, 1000)
        pdf_a = stats.beta.pdf(
            x, result.posterior_alpha_a, result.posterior_beta_a
        )
        pdf_b = stats.beta.pdf(
            x, result.posterior_alpha_b, result.posterior_beta_b
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, pdf_a, color="steelblue", linewidth=2, label="Posterior A")
        ax.fill_between(x, pdf_a, alpha=0.2, color="steelblue")
        ax.plot(x, pdf_b, color="firebrick", linewidth=2, label="Posterior B")
        ax.fill_between(x, pdf_b, alpha=0.2, color="firebrick")

        # Credible interval spans
        ax.axvspan(
            result.credible_lower_a, result.credible_upper_a,
            alpha=0.08, color="steelblue", label=f"{int(self._ci_level*100)}% CI A"
        )
        ax.axvspan(
            result.credible_lower_b, result.credible_upper_b,
            alpha=0.08, color="firebrick", label=f"{int(self._ci_level*100)}% CI B"
        )

        ax.axvline(result.mean_a, color="steelblue", linestyle="--", alpha=0.8)
        ax.axvline(result.mean_b, color="firebrick", linestyle="--", alpha=0.8)

        ax.set_title(
            f"Posterior Distributions (Beta-Binomial)\n"
            f"P(B > A) = {result.prob_b_beats_a:.4f}"
        )
        ax.set_xlabel("Conversion Rate")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"bayesian_posteriors_{label}")

    def plot_sequential(
        self,
        history: list[dict[str, Any]],
        stopping_threshold: float = 0.95,
        label: str = "sequential",
    ) -> pathlib.Path:
        """Plot P(B > A) over sequential observations."""
        ns = [h["n_per_group"] for h in history]
        probs = [h["prob_b_beats_a"] for h in history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ns, probs, color="steelblue", linewidth=2)
        ax.axhline(
            stopping_threshold, color="firebrick", linestyle="--",
            label=f"Stop threshold ({stopping_threshold})"
        )
        ax.axhline(
            1 - stopping_threshold, color="darkorange", linestyle="--",
            label=f"Stop threshold ({1-stopping_threshold:.2f})"
        )
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_title("Sequential Bayesian A/B Test: P(B > A) Over Time")
        ax.set_xlabel("Sample Size per Group")
        ax.set_ylabel("P(B > A)")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"sequential_ab_{label}")

    def plot_mc_distribution(
        self, result: BayesianTestResult, label: str = "ab_test"
    ) -> pathlib.Path:
        """
        Distribution of (rate_B - rate_A) from Monte Carlo samples.
        Shows the lift distribution and its uncertainty.
        """
        samples_a = self._rng.beta(
            result.posterior_alpha_a, result.posterior_beta_a, 50000
        )
        samples_b = self._rng.beta(
            result.posterior_alpha_b, result.posterior_beta_b, 50000
        )
        lift = samples_b - samples_a

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(lift, bins=80, density=True, color="steelblue",
                alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--",
                   label="No effect")
        ax.axvline(np.mean(lift), color="firebrick", linewidth=2,
                   label=f"Mean lift = {np.mean(lift):.4f}")
        lo, hi = np.percentile(lift, [2.5, 97.5])
        ax.axvline(lo, color="darkorange", linestyle=":", label=f"2.5% = {lo:.4f}")
        ax.axvline(hi, color="darkorange", linestyle=":", label=f"97.5% = {hi:.4f}")
        ax.set_title(
            f"Distribution of Lift (rate_B - rate_A)\n"
            f"P(B > A) = {result.prob_b_beats_a:.4f}"
        )
        ax.set_xlabel("Lift (rate_B - rate_A)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, f"mc_lift_dist_{label}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== BayesianABTester demo start ===")

    tester = BayesianABTester(cfg, logger)

    # Scenario 1: Clear winner
    print("\n--- Scenario 1: Clear winner (B much better) ---")
    result1 = tester.run_test(
        conversions_a=100, n_a=1000,
        conversions_b=130, n_b=1000,
    )
    print(result1)
    tester.plot_posteriors(result1, label="clear_winner")
    tester.plot_mc_distribution(result1, label="clear_winner")

    # Scenario 2: Close call
    print("\n--- Scenario 2: Close call ---")
    result2 = tester.run_test(
        conversions_a=100, n_a=1000,
        conversions_b=104, n_b=1000,
    )
    print(result2)
    tester.plot_posteriors(result2, label="close_call")

    # Scenario 3: Sequential analysis
    print("\n--- Scenario 3: Sequential analysis ---")
    seq = tester.sequential_analysis(
        true_rate_a=0.10,
        true_rate_b=0.13,
        stopping_threshold=0.95,
        max_n_per_group=2000,
        batch_size=25,
    )
    print(f"  Stopped at n={seq['stopped_at']} per group")
    tester.plot_sequential(
        seq["history"], stopping_threshold=0.95, label="scenario3"
    )

    logger.info("=== BayesianABTester demo complete ===")


if __name__ == "__main__":
    main()
