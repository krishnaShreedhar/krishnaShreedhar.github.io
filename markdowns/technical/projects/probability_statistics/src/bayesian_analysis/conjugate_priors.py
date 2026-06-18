"""
conjugate_priors.py

Demonstration of conjugate prior-posterior pairs.

Concepts illustrated:
  - Beta-Binomial conjugacy (binary outcomes)
  - Normal-Normal conjugacy (known variance, unknown mean)
  - Gamma-Poisson conjugacy (count data, unknown rate)
  - Bayesian updating: how the posterior shifts with more data
  - Prior sensitivity: effect of informative vs. vague priors
  - Bayesian vs. frequentist credible/confidence interval comparison
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

    logger = logging.getLogger("bayesian_analysis.conjugate_priors")
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
class BetaBinomialState:
    """State of a Beta-Binomial Bayesian update."""
    prior_alpha: float
    prior_beta: float
    successes: int
    n: int

    @property
    def posterior_alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def posterior_beta(self) -> float:
        return self.prior_beta + (self.n - self.successes)

    @property
    def posterior_mean(self) -> float:
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)


@dataclass
class NormalNormalState:
    """State of a Normal-Normal Bayesian update (known variance)."""
    prior_mu: float
    prior_sigma2: float   # prior variance
    obs_sigma2: float     # known observation variance
    data_mean: float
    n: int

    @property
    def posterior_sigma2(self) -> float:
        return 1 / (1 / self.prior_sigma2 + self.n / self.obs_sigma2)

    @property
    def posterior_mu(self) -> float:
        return self.posterior_sigma2 * (
            self.prior_mu / self.prior_sigma2
            + self.n * self.data_mean / self.obs_sigma2
        )


@dataclass
class GammaPoissonState:
    """State of a Gamma-Poisson Bayesian update."""
    prior_shape: float   # alpha
    prior_rate: float    # beta (rate parameterisation)
    total_counts: int
    n_intervals: int

    @property
    def posterior_shape(self) -> float:
        return self.prior_shape + self.total_counts

    @property
    def posterior_rate(self) -> float:
        return self.prior_rate + self.n_intervals

    @property
    def posterior_mean(self) -> float:
        return self.posterior_shape / self.posterior_rate


# ---------------------------------------------------------------------------
# ConjugatePriorAnalyzer
# ---------------------------------------------------------------------------

class ConjugatePriorAnalyzer:
    """
    Demonstrates conjugate prior-posterior relationships.

    SRP: Responsible only for conjugate-prior analysis and visualisation.
    OCP: Each conjugate family is an independent method pair.
    """

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger) -> None:
        self._cfg = cfg
        self._logger = logger
        self._ab_cfg = cfg["bayesian_ab"]
        self._ci_level = self._ab_cfg["credible_interval"]
        self._output_dir = pathlib.Path(cfg["data"]["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(cfg["data"]["random_seed"])
        self._logger.info("ConjugatePriorAnalyzer initialised")

    # ------------------------------------------------------------------
    # Beta-Binomial
    # ------------------------------------------------------------------

    def update_beta_binomial(
        self,
        prior_alpha: float,
        prior_beta: float,
        successes: int,
        n: int,
    ) -> BetaBinomialState:
        """
        Update Beta prior with Binomial observations.

        Prior:       p ~ Beta(alpha, beta)
        Likelihood:  X ~ Binomial(n, p)
        Posterior:   p | X ~ Beta(alpha + X, beta + n - X)

        The posterior mean is a weighted average of prior mean and
        observed frequency, shrinking towards the prior as alpha+beta grows.
        """
        state = BetaBinomialState(prior_alpha, prior_beta, successes, n)
        self._logger.info(
            f"Beta-Binomial update | prior=Beta({prior_alpha},{prior_beta}), "
            f"data={successes}/{n} | "
            f"posterior=Beta({state.posterior_alpha},{state.posterior_beta}), "
            f"posterior_mean={state.posterior_mean:.4f}"
        )
        return state

    # ------------------------------------------------------------------
    # Normal-Normal (known variance)
    # ------------------------------------------------------------------

    def update_normal_normal(
        self,
        prior_mu: float,
        prior_sigma2: float,
        obs_sigma2: float,
        data: np.ndarray,
    ) -> NormalNormalState:
        """
        Update Normal prior (mean) with Normal data (known variance).

        Prior:       mu ~ Normal(mu_0, sigma_0^2)
        Likelihood:  X_i ~ Normal(mu, sigma^2)  [sigma known]
        Posterior:   mu | X ~ Normal(mu_n, sigma_n^2)
          where:
            1/sigma_n^2 = 1/sigma_0^2 + n/sigma^2
            mu_n = sigma_n^2 * (mu_0/sigma_0^2 + n*x_bar/sigma^2)

        The posterior precision is the sum of prior and data precisions.
        """
        state = NormalNormalState(
            prior_mu=prior_mu,
            prior_sigma2=prior_sigma2,
            obs_sigma2=obs_sigma2,
            data_mean=float(np.mean(data)),
            n=len(data),
        )
        self._logger.info(
            f"Normal-Normal update | prior=N({prior_mu},{prior_sigma2:.2f}), "
            f"n={len(data)}, x_bar={state.data_mean:.4f} | "
            f"posterior=N({state.posterior_mu:.4f},{state.posterior_sigma2:.4f})"
        )
        return state

    # ------------------------------------------------------------------
    # Gamma-Poisson
    # ------------------------------------------------------------------

    def update_gamma_poisson(
        self,
        prior_shape: float,
        prior_rate: float,
        counts: np.ndarray,
    ) -> GammaPoissonState:
        """
        Update Gamma prior with Poisson observations.

        Prior:       lambda ~ Gamma(alpha, beta)  [rate parameterisation]
        Likelihood:  X_i ~ Poisson(lambda)
        Posterior:   lambda | X ~ Gamma(alpha + sum(X), beta + n)

        The posterior mean = (alpha + sum_X) / (beta + n) shrinks
        towards prior mean = alpha/beta for small n.
        """
        total = int(np.sum(counts))
        state = GammaPoissonState(
            prior_shape=prior_shape,
            prior_rate=prior_rate,
            total_counts=total,
            n_intervals=len(counts),
        )
        self._logger.info(
            f"Gamma-Poisson update | prior=Gamma({prior_shape},{prior_rate}), "
            f"n={len(counts)}, total_counts={total} | "
            f"posterior=Gamma({state.posterior_shape},{state.posterior_rate}), "
            f"posterior_mean={state.posterior_mean:.4f}"
        )
        return state

    # ------------------------------------------------------------------
    # Sequential updating
    # ------------------------------------------------------------------

    def sequential_beta_binomial_update(
        self,
        prior_alpha: float,
        prior_beta: float,
        true_rate: float,
        n_steps: int = 20,
    ) -> list[BetaBinomialState]:
        """
        Simulate incremental Bayesian updating as data arrives one-by-one.
        Shows how the posterior converges to the true rate.
        """
        alpha, beta = prior_alpha, prior_beta
        states = []
        total_n = 0
        total_s = 0
        for _ in range(n_steps):
            obs = int(self._rng.binomial(1, true_rate))
            total_s += obs
            total_n += 1
            state = BetaBinomialState(
                prior_alpha=alpha,
                prior_beta=beta,
                successes=total_s,
                n=total_n,
            )
            alpha = state.posterior_alpha
            beta = state.posterior_beta
            states.append(state)
        self._logger.info(
            f"Sequential Beta-Binomial | {n_steps} steps, "
            f"true_rate={true_rate}, final_posterior_mean={states[-1].posterior_mean:.4f}"
        )
        return states

    # ------------------------------------------------------------------
    # Credible vs. Confidence interval comparison
    # ------------------------------------------------------------------

    def compare_credible_confidence(
        self,
        true_rate: float = 0.15,
        n: int = 100,
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """
        Simulate n_simulations experiments and compare:
          - Frequentist 95% CI coverage (proportion containing true rate)
          - Bayesian 95% credible interval coverage

        The frequentist CI has exact frequentist coverage in repeated sampling.
        The Bayesian credible interval expresses posterior probability
        that the true parameter lies in the interval.
        """
        self._logger.info(
            f"Credible vs CI comparison | true_rate={true_rate}, "
            f"n={n}, sims={n_simulations}"
        )
        freq_coverage = 0
        bayes_coverage = 0

        for _ in range(n_simulations):
            obs = int(self._rng.binomial(n, true_rate))
            p_hat = obs / n

            # Frequentist Wilson CI
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            freq_lo = max(0, p_hat - 1.96 * se)
            freq_hi = min(1, p_hat + 1.96 * se)
            if freq_lo <= true_rate <= freq_hi:
                freq_coverage += 1

            # Bayesian credible interval (uniform prior)
            al_post = 1 + obs
            be_post = 1 + (n - obs)
            lo = (1 - self._ci_level) / 2
            hi = 1 - lo
            bayes_lo = stats.beta.ppf(lo, al_post, be_post)
            bayes_hi = stats.beta.ppf(hi, al_post, be_post)
            if bayes_lo <= true_rate <= bayes_hi:
                bayes_coverage += 1

        results = {
            "true_rate": true_rate,
            "n": n,
            "n_simulations": n_simulations,
            "frequentist_coverage": freq_coverage / n_simulations,
            "bayesian_coverage": bayes_coverage / n_simulations,
            "target_coverage": self._ci_level,
        }
        self._logger.info(
            f"Coverage | Frequentist={results['frequentist_coverage']:.3f}, "
            f"Bayesian={results['bayesian_coverage']:.3f}"
        )
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._logger.info(f"Figure saved | {path}")
        return path

    def plot_beta_updating(
        self,
        true_rate: float,
        prior_specs: list[tuple[float, float]],
        n_obs: int = 200,
        label: str = "beta_updating",
    ) -> pathlib.Path:
        """
        Plot Beta prior, likelihood, and posterior for multiple priors.
        Illustrates how informative priors vs. vague priors affect the posterior.
        """
        x = np.linspace(0.001, 0.999, 1000)
        n_successes = int(self._rng.binomial(n_obs, true_rate))
        n_failures = n_obs - n_successes

        fig, axes = plt.subplots(
            1, len(prior_specs), figsize=(5 * len(prior_specs), 5), sharey=False
        )
        if len(prior_specs) == 1:
            axes = [axes]

        colors = {"prior": "grey", "likelihood": "darkorange", "posterior": "steelblue"}

        for ax, (al, be) in zip(axes, prior_specs):
            # Prior
            prior_pdf = stats.beta.pdf(x, al, be)
            ax.plot(x, prior_pdf, color=colors["prior"], linewidth=2,
                    linestyle="--", label=f"Prior Beta({al},{be})")

            # Likelihood (up to proportionality: x^s * (1-x)^f)
            log_lik = n_successes * np.log(x) + n_failures * np.log(1 - x)
            lik = np.exp(log_lik - log_lik.max())
            lik_scaled = lik / np.trapz(lik, x)
            ax.plot(x, lik_scaled, color=colors["likelihood"], linewidth=2,
                    linestyle="-.", label="Likelihood (scaled)")

            # Posterior
            al_post = al + n_successes
            be_post = be + n_failures
            post_pdf = stats.beta.pdf(x, al_post, be_post)
            ax.plot(x, post_pdf, color=colors["posterior"], linewidth=2,
                    label=f"Posterior Beta({al_post:.0f},{be_post:.0f})")
            ax.axvline(true_rate, color="firebrick", linestyle=":",
                       linewidth=2, label=f"True rate = {true_rate}")

            ax.set_title(
                f"Prior Beta({al},{be})\n"
                f"Data: {n_successes}/{n_obs} = {n_successes/n_obs:.3f}"
            )
            ax.set_xlabel("p")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 0.5)

        fig.suptitle("Bayesian Updating: Beta-Binomial", fontsize=13)
        fig.tight_layout()
        return self._save_figure(fig, label)

    def plot_sequential_updating(
        self,
        states: list[BetaBinomialState],
        true_rate: float,
        label: str = "sequential_beta",
    ) -> pathlib.Path:
        """Plot posterior mean and credible interval over sequential updates."""
        ns = list(range(1, len(states) + 1))
        means = [s.posterior_mean for s in states]
        lows = [
            stats.beta.ppf(
                (1 - self._ci_level) / 2,
                s.posterior_alpha, s.posterior_beta
            )
            for s in states
        ]
        highs = [
            stats.beta.ppf(
                1 - (1 - self._ci_level) / 2,
                s.posterior_alpha, s.posterior_beta
            )
            for s in states
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ns, means, color="steelblue", linewidth=2, label="Posterior mean")
        ax.fill_between(ns, lows, highs, color="steelblue", alpha=0.2,
                        label=f"{int(self._ci_level*100)}% Credible Interval")
        ax.axhline(true_rate, color="firebrick", linestyle="--",
                   linewidth=2, label=f"True rate = {true_rate}")
        ax.set_title("Sequential Bayesian Updating: Posterior Mean Over Time")
        ax.set_xlabel("Number of Observations")
        ax.set_ylabel("Estimated Rate")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return self._save_figure(fig, label)

    def plot_gamma_poisson_updating(
        self,
        state: GammaPoissonState,
        true_lambda: float,
        label: str = "gamma_poisson",
    ) -> pathlib.Path:
        """Plot Gamma prior and posterior for Poisson rate estimation."""
        max_lam = true_lambda * 3
        x = np.linspace(0.001, max_lam, 1000)

        prior_pdf = stats.gamma.pdf(
            x, a=state.prior_shape, scale=1 / state.prior_rate
        )
        post_pdf = stats.gamma.pdf(
            x, a=state.posterior_shape, scale=1 / state.posterior_rate
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, prior_pdf, color="grey", linewidth=2, linestyle="--",
                label=f"Prior Gamma({state.prior_shape},{state.prior_rate})")
        ax.plot(x, post_pdf, color="steelblue", linewidth=2,
                label=f"Posterior Gamma({state.posterior_shape:.0f},{state.posterior_rate:.0f})")
        ax.axvline(true_lambda, color="firebrick", linestyle=":",
                   linewidth=2, label=f"True λ = {true_lambda}")
        ax.axvline(state.posterior_mean, color="darkorange", linestyle="--",
                   linewidth=2, label=f"Posterior mean = {state.posterior_mean:.3f}")

        ax.set_title("Bayesian Updating: Gamma-Poisson")
        ax.set_xlabel("λ (Poisson rate)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        return self._save_figure(fig, label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = pathlib.Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    logger = _build_logger(cfg)

    logger.info("=== ConjugatePriorAnalyzer demo start ===")

    analyzer = ConjugatePriorAnalyzer(cfg, logger)
    rng = np.random.default_rng(cfg["data"]["random_seed"])

    # 1. Beta-Binomial with multiple priors
    true_rate = 0.15
    prior_specs = [
        (1, 1),     # Uniform (vague)
        (2, 10),    # Weakly informative (biased low)
        (15, 85),   # Strongly informative (centred near truth)
    ]
    analyzer.plot_beta_updating(
        true_rate=true_rate,
        prior_specs=prior_specs,
        n_obs=100,
        label="beta_binomial_priors",
    )
    print("\n=== Beta-Binomial Updates ===")
    for al, be in prior_specs:
        state = analyzer.update_beta_binomial(al, be, successes=15, n=100)
        print(
            f"  Prior Beta({al},{be}) + 15/100 => "
            f"Posterior Beta({state.posterior_alpha},{state.posterior_beta}), "
            f"mean={state.posterior_mean:.4f}"
        )

    # 2. Sequential updating
    states = analyzer.sequential_beta_binomial_update(
        prior_alpha=1, prior_beta=1, true_rate=0.15, n_steps=100
    )
    analyzer.plot_sequential_updating(states, true_rate=0.15, label="sequential_updating")

    # 3. Normal-Normal
    print("\n=== Normal-Normal Update ===")
    data = rng.normal(loc=5.0, scale=2.0, size=50)
    nn_state = analyzer.update_normal_normal(
        prior_mu=0.0, prior_sigma2=100.0, obs_sigma2=4.0, data=data
    )
    print(
        f"  Posterior mu = {nn_state.posterior_mu:.4f}, "
        f"sigma^2 = {nn_state.posterior_sigma2:.4f}"
    )

    # 4. Gamma-Poisson
    print("\n=== Gamma-Poisson Update ===")
    true_lambda = 4.0
    counts = rng.poisson(true_lambda, size=50)
    gp_state = analyzer.update_gamma_poisson(
        prior_shape=1.0, prior_rate=0.25, counts=counts
    )
    print(
        f"  Posterior lambda mean = {gp_state.posterior_mean:.4f} "
        f"(true = {true_lambda})"
    )
    analyzer.plot_gamma_poisson_updating(gp_state, true_lambda=true_lambda)

    # 5. Credible vs Confidence interval coverage
    print("\n=== Credible vs Confidence Interval Coverage ===")
    cov = analyzer.compare_credible_confidence(true_rate=0.15, n=50, n_simulations=5000)
    print(
        f"  Frequentist 95% CI coverage: {cov['frequentist_coverage']:.3f}"
    )
    print(
        f"  Bayesian {int(cov['target_coverage']*100)}% credible interval coverage: "
        f"{cov['bayesian_coverage']:.3f}"
    )

    logger.info("=== ConjugatePriorAnalyzer demo complete ===")


if __name__ == "__main__":
    main()
