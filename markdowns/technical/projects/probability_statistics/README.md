---
title: "Probability & Statistics"
subtitle: "A self-contained educational project illustrating core Probability and Statistics concepts through minimal, runnable Python code and SQL analytics. Designed so a practitioner can read the source, understand the..."
category: technical
project: probability_statistics
project_title: "Probability & Statistics"
date: 2025-02-14
reading_time: 4
tags:
  - probability-statistics
author: "Shreedhar Kodate"
output: "blogs/technical/posts/probability_statistics/index.html"
---
A self-contained educational project illustrating core Probability and Statistics concepts through minimal, runnable Python code and SQL analytics. Designed so a practitioner can read the source, understand the algorithm, and immediately apply it to real data.

---

## Concepts Covered

### Descriptive Statistics (`src/descriptive_stats/`)
- **Central tendency**: arithmetic mean, median, mode, trimmed mean, geometric mean, harmonic mean
- **Spread**: variance, standard deviation, IQR, MAD, coefficient of variation
- **Shape**: skewness (Pearson), excess kurtosis
- **Percentiles**: P10, P25, P50, P75, P90, P95, P99
- **Distribution fitting**: Normal, Log-Normal, Gamma, Exponential, Beta — ranked by AIC
- **Goodness-of-fit**: Kolmogorov-Smirnov test
- **Visualisations**: histograms with fitted PDFs, QQ plots, empirical vs. theoretical CDFs

### Hypothesis Testing (`src/hypothesis_testing/`)
- **One-sample t-test**: compare sample mean to known value
- **Two-sample Welch's t-test**: compare two groups without equal-variance assumption
- **Chi-squared test of independence**: categorical association
- **Mann-Whitney U test**: non-parametric two-sample test
- **Two-proportion z-test**: A/B testing for conversion rates
- **Bootstrap confidence intervals**: percentile method, B=10,000 iterations
- **Power analysis**: sample size calculation for t-test and z-test
- **Multiple testing correction**: Bonferroni (FWER) and Benjamini-Hochberg (FDR)
- **Effect sizes**: Cohen's d, Cohen's h, Cramér's V, rank-biserial correlation

### Bayesian Analysis (`src/bayesian_analysis/`)
- **Beta-Binomial model**: conjugate prior for conversion rate A/B testing
- **Bayesian updating**: sequential posterior updates with each observation
- **Monte Carlo sampling**: P(B > A) and lift distribution with 100,000 samples
- **Expected loss**: risk-minimisation decision criterion
- **Credible intervals**: vs. frequentist confidence intervals (coverage comparison)
- **Prior sensitivity**: effect of vague vs. informative priors
- **Normal-Normal and Gamma-Poisson conjugacy**: for continuous and count data

### SQL Analytics (`src/sql_analytics/`)
- **Descriptive stats SQL**: `PERCENTILE_CONT`, `STDDEV_SAMP`, `VAR_SAMP`, `SKEWNESS`, `KURTOSIS`
- **Window functions**: `PERCENT_RANK`, `CUME_DIST`, `NTILE`, `RANK`, `DENSE_RANK`
- **Z-score normalisation**: in SQL with outlier flagging
- **IQR outlier detection**: Tukey's fences entirely in SQL
- **Histogram buckets**: `WIDTH_BUCKET` for equal-width binning
- **Grouped statistics**: per-category descriptive stats
- **Funnel analysis**: step-by-step conversion and drop-off rates
- **Cohort retention**: weekly retention matrix
- **Linear regression**: `REGR_SLOPE`, `REGR_INTERCEPT`, `REGR_R2`, `CORR`

---

## Project Structure

```
probability_statistics/
    src/
        notebooks/
            statistics_demo.ipynb       # Full workflow demo
        descriptive_stats/
            statistics_calculator.py    # Mean, median, spread, shape
            distribution_analyzer.py    # Distribution fitting and visualisation
        hypothesis_testing/
            test_runner.py              # t-tests, chi-squared, z-test, bootstrap, MTC
            power_analysis.py           # Sample size and power curves
        bayesian_analysis/
            bayesian_ab_test.py         # Beta-Binomial A/B testing
            conjugate_priors.py         # Conjugate prior families
        sql_analytics/
            sql_statistics.py           # DuckDB SQL statistical queries
    docs/
        concepts.md                     # Theory + mermaid diagrams
        flow_diagrams.md                # Pipeline flow diagrams
    docker/
        Dockerfile
        docker-compose.yml
        requirements.txt
    outputs/                            # Generated plots (auto-created)
    logs/                               # Log files (auto-created)
    config.yaml                         # All constants and hyperparameters
    pyproject.toml
    README.md
```

---

## Usage

All configuration is in `config.yaml`. No command-line arguments.

### Run individual modules directly

```bash
# From the project root (probability_statistics/)

# Descriptive statistics
python src/descriptive_stats/statistics_calculator.py
python src/descriptive_stats/distribution_analyzer.py

# Hypothesis testing
python src/hypothesis_testing/test_runner.py
python src/hypothesis_testing/power_analysis.py

# Bayesian analysis
python src/bayesian_analysis/bayesian_ab_test.py
python src/bayesian_analysis/conjugate_priors.py

# SQL analytics (DuckDB)
python src/sql_analytics/sql_statistics.py
```

### Run via Jupyter notebook

```bash
cd src/notebooks
jupyter notebook statistics_demo.ipynb
```

### Run via Docker

```bash
cd docker

# Build image
docker-compose build

# Run SQL analytics demo
docker-compose run probability_statistics

# Run descriptive stats
docker-compose run descriptive_stats

# Run hypothesis testing
docker-compose run hypothesis_testing

# Run Bayesian A/B test
docker-compose run bayesian_ab

# Start Jupyter server (access at http://localhost:8888)
docker-compose up jupyter
```

---

## Configuration (`config.yaml`)

All hyperparameters and constants live in `config.yaml`. Key sections:

| Section | Key Parameters |
|---|---|
| `logging` | `level`, `log_file`, `max_bytes`, `backup_count` |
| `data` | `sample_size`, `random_seed`, `output_dir` |
| `descriptive_stats` | `trimmed_mean_pct`, `percentiles`, `distribution_bins` |
| `hypothesis_testing` | `alpha`, `power`, `bootstrap_iterations`, `effect_size`, `baseline_rate` |
| `bayesian_ab` | `prior_alpha`, `prior_beta`, `monte_carlo_samples`, `credible_interval` |
| `sql_analytics` | `database`, `histogram_bins`, `iqr_multiplier` |

---

## Outputs

All plots are saved to `outputs/` as PNG files:

| File | Description |
|---|---|
| `histogram_fit_*.png` | Histogram with best-fit distribution PDF |
| `qq_plot_*.png` | Normal QQ plot |
| `ecdf_*.png` | Empirical vs. theoretical CDF |
| `multi_dist_*.png` | Top-N distribution fits comparison |
| `power_vs_n_ttest.png` | Power curves by sample size |
| `power_vs_effect_size_ttest.png` | Power curves by effect size |
| `sample_size_heatmap.png` | Required n for (alpha, power) grid |
| `bayesian_posteriors_*.png` | Beta posterior distributions |
| `mc_lift_dist_*.png` | Monte Carlo lift distribution |
| `sequential_ab_*.png` | Sequential P(B>A) over time |
| `beta_binomial_priors.png` | Prior sensitivity analysis |
| `sequential_updating.png` | Sequential Bayesian updating |
| `notebook_*.png` | Plots from the Jupyter notebook |

---

## Key References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.)
- VanderPlas, J. (2016). *Python Data Science Handbook* (NumPy, SciPy, Pandas)
- Kohavi, R., et al. (2020). *Trustworthy Online Controlled Experiments* (A/B testing)
- DuckDB Documentation: https://duckdb.org/docs/sql/functions/aggregates