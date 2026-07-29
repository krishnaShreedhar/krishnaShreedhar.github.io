---
title: "Flow Diagrams: Statistical Analysis Pipelines"
subtitle: "This document contains workflow diagrams for the three main analysis patterns in this project."
category: technical
project: probability_statistics
project_title: "Probability & Statistics"
date: 2025-03-26
reading_time: 5
tags:
  - probability-statistics
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/probability_statistics/docs/flow_diagrams.html"
---
This document contains workflow diagrams for the three main analysis patterns
in this project.

---

## 1. Statistical Analysis Pipeline

The full end-to-end pipeline from raw data to actionable insights.

```mermaid
flowchart TD
    RAW["Raw Data\n(CSV, DB, API, stream)"]

    EDA["Exploratory Data Analysis\n\n- Shape and dtypes\n- Missing value audit\n- Duplicate check\n- Value distributions"]

    DESC["Descriptive Statistics\n\nstatistics_calculator.py\n\n- Central tendency: mean, median, mode\n- Spread: std, IQR, MAD, CV\n- Shape: skewness, kurtosis\n- Percentiles: P25, P50, P75, P99"]

    DIST["Distribution Analysis\n\ndistribution_analyzer.py\n\n- Fit: Normal, Log-Normal,\n  Gamma, Exponential, Beta\n- Goodness-of-fit: KS test\n- Model selection: AIC\n- QQ plot for normality check"]

    NORM{Data\nnormally\ndistributed?}

    PARAM["Parametric Methods\n\n- t-tests (Welch's)\n- ANOVA\n- Pearson correlation\n- Linear regression"]

    NONPAR["Non-parametric Methods\n\n- Mann-Whitney U\n- Kruskal-Wallis\n- Spearman correlation\n- Bootstrap CI"]

    HYPO["Hypothesis Testing\n\ntest_runner.py\n\n- Formulate H0 and H1\n- Choose test statistic\n- Compute p-value\n- Check power"]

    MULTI{Multiple\nhypotheses?}
    MTC["Multiple Testing Correction\n\n- Bonferroni (FWER)\n- Benjamini-Hochberg (FDR)"]

    EFFECT["Effect Size Analysis\n\n- Cohen's d (continuous)\n- Cohen's h (proportions)\n- Cramér's V (categorical)\n- Rank-biserial r (non-param)"]

    BAYES["Bayesian Analysis\n\nbayesian_ab_test.py\nconjugate_priors.py\n\n- Posterior distribution\n- Credible intervals\n- P(effect exists)\n- Expected loss"]

    SQL["SQL Analytics\n\nsql_statistics.py\n\n- Scalable aggregations\n- Window functions\n- Outlier detection\n- Cohort analysis"]

    VIZ["Visualisation\n\nmatplotlib / seaborn\n\n- Histograms + fitted PDFs\n- QQ plots\n- Power curves\n- Posterior distributions\n- Heatmaps"]

    REPORT["Report & Decision\n\n- Statistical significance\n- Practical significance\n- Confidence/credible intervals\n- Recommendations"]

    RAW --> EDA --> DESC --> DIST
    DIST --> NORM
    NORM -->|Yes| PARAM
    NORM -->|No| NONPAR
    PARAM --> HYPO
    NONPAR --> HYPO
    HYPO --> MULTI
    MULTI -->|Yes| MTC --> EFFECT
    MULTI -->|No| EFFECT
    EFFECT --> BAYES
    EFFECT --> SQL
    BAYES --> VIZ
    SQL --> VIZ
    VIZ --> REPORT

    style RAW fill:#34495e,color:#fff
    style REPORT fill:#2d6a4f,color:#fff
    style HYPO fill:#4a90d9,color:#fff
    style BAYES fill:#e06c1a,color:#fff
    style SQL fill:#6a3d9a,color:#fff
```

---

## 2. Hypothesis Testing Decision Flow

Step-by-step process for running a principled hypothesis test.

```mermaid
flowchart TD
    START(["State the Research Question\nExample: Does email subject line B\nincrease open rate over A?"])

    H0["Define Hypotheses\n\nH0 (Null): pA = pB\nH1 (Alt): pA ≠ pB\n(two-sided)"]

    ALPHA["Set Significance Level\nalpha = 0.05\n(acceptable Type I error rate)"]

    POWER["Power Analysis\nSet target power = 0.80\nEstimate effect size (MDE)\nCompute required n"]

    COLLECT["Collect Data\n- Random assignment\n- Full exposure period\n- No peeking (sequential testing requires\n  different stopping rules)"]

    ASSUME["Check Assumptions\n\nFor t-test:\n  - Independence\n  - Normality (or n > 30 by CLT)\n\nFor z-test (proportions):\n  - np ≥ 10 and n(1-p) ≥ 10\n\nFor chi-squared:\n  - Expected cell counts ≥ 5"]

    ASSUME_OK{Assumptions\nmet?}
    ASSUME_OK -->|No| TRANSFORM["Transform Data or\nUse Non-Parametric Test"]
    ASSUME_OK -->|Yes| COMPUTE

    COMPUTE["Compute Test Statistic\n\nt = (x̄A - x̄B) / SE\nz = (p̂A - p̂B) / SE_pool\nchi2 = Σ(O-E)²/E"]

    PVAL["Compute P-value\n\np = P(|T| ≥ |t_obs| | H0)\n\nTwo-sided: multiply by 2\nOne-sided: use tail directly"]

    CI["Compute Confidence Interval\n\nCI = estimate ± z_{alpha/2} · SE\n\nExample for difference in means:\n95% CI = (x̄B - x̄A) ± 1.96 · SE"]

    EFFECT_SZ["Compute Effect Size\n\nContinuous: Cohen's d = Δμ / σ_pooled\nProportions: Cohen's h\nCategorical: Cramér's V"]

    DECISION{"Decision Rule\np < alpha AND\nCI excludes 0 AND\neffect_size > practical threshold"}

    REJECT["Reject H0\nStatistically significant\nREPORT effect size and CI\nDo not claim 'proof'"]
    FAIL["Fail to Reject H0\nInsufficient evidence\nDo NOT say 'H0 is proven'\nConsider power and effect size"]

    INTERPRET["Interpret Results\n\n- Statistical significance ≠ practical significance\n- Large n can make tiny effects 'significant'\n- Always report effect size\n- Replicate before acting on findings"]

    START --> H0 --> ALPHA --> POWER --> COLLECT --> ASSUME
    TRANSFORM --> COMPUTE
    COMPUTE --> PVAL --> CI --> EFFECT_SZ --> DECISION
    DECISION -->|"p < alpha"| REJECT --> INTERPRET
    DECISION -->|"p ≥ alpha"| FAIL --> INTERPRET

    style START fill:#2d6a4f,color:#fff
    style REJECT fill:#c0392b,color:#fff
    style FAIL fill:#f39c12,color:#000
    style INTERPRET fill:#1a535c,color:#fff
```

---

## 3. Bayesian A/B Test Workflow

End-to-end Bayesian A/B testing from prior specification to decision.

```mermaid
flowchart TD
    DEFINE(["Define A/B Test\nControl A: current version\nVariant B: new version\nMetric: conversion rate"])

    PRIOR["Specify Prior\n\nChoose Beta(alpha0, beta0)\n\nVague prior: Beta(1, 1)\n- No prior knowledge\n- Uniform over [0,1]\n\nInformative prior: Beta(10, 90)\n- Believe ~10% base rate\n- Strong prior certainty\n\nPrior mean = alpha0 / (alpha0 + beta0)"]

    DEPLOY["Deploy Experiment\n- Randomise users to A/B\n- Track conversions\n- No fixed stopping rule needed"]

    DATA["Observe Data\n\nGroup A: nA users, sA conversions\nGroup B: nB users, sB conversions"]

    UPDATE["Bayesian Update\n\nPosterior A: Beta(alpha0 + sA, beta0 + fA)\nPosterior B: Beta(alpha0 + sB, beta0 + fB)\n\nwhere fA = nA - sA, fB = nB - sB\n\nConjugate: posterior is also Beta!"]

    SUMMARIES["Posterior Summaries\n\nMean A = (alpha0+sA)/(alpha0+beta0+nA)\nMean B = (alpha0+sB)/(alpha0+beta0+nB)\n\n95% Credible Interval:\n[Beta_PPF(0.025), Beta_PPF(0.975)]"]

    MC["Monte Carlo Estimation\nDraw M=100,000 samples from each posterior\n\nsamples_A ~ Beta(posterior_A)\nsamples_B ~ Beta(posterior_B)\n\nlift_samples = samples_B - samples_A"]

    METRICS["Compute Decision Metrics\n\nP(B > A) = mean(samples_B > samples_A)\n\nExpected Loss(A) = mean(max(0, samples_B - samples_A))\nExpected Loss(B) = mean(max(0, samples_A - samples_B))\n\nCredible interval for lift:\n[P2.5(lift_samples), P97.5(lift_samples)]"]

    STOP{"Stopping Rule\n\nP(B > A) > 0.95\nOR\nP(A > B) > 0.95\nOR\nMax sample size reached"}

    CONTINUE["Continue Collecting Data\nSequential updating: no fixed n!"]

    DECIDE{"Decision\n\nWhich variant has\nlower expected loss?"}

    SHIP_B["Ship B\nExpected loss(A) > threshold\nP(B > A) > 0.95"]

    SHIP_A["Keep A\nExpected loss(B) > threshold\nP(A > B) > 0.95"]

    INCONCLUSIVE["Inconclusive\nCollect more data or\nReduce prior uncertainty"]

    COMPARE["Compare with Frequentist\n\nTwo-proportion z-test:\nz = (p̂B - p̂A) / SE_pool\np-value = 2*(1 - Phi(|z|))\n\nTypically agrees with Bayesian\nbut different interpretation"]

    DEFINE --> PRIOR --> DEPLOY --> DATA
    DATA --> UPDATE --> SUMMARIES --> MC --> METRICS --> STOP
    STOP -->|Not reached| CONTINUE --> DATA
    STOP -->|Reached| DECIDE
    DECIDE -->|"loss(A) minimal"| SHIP_B
    DECIDE -->|"loss(B) minimal"| SHIP_A
    DECIDE -->|"Both losses\nequally small"| INCONCLUSIVE
    SHIP_B --> COMPARE
    SHIP_A --> COMPARE

    style DEFINE fill:#2d6a4f,color:#fff
    style UPDATE fill:#27ae60,color:#fff
    style MC fill:#4a90d9,color:#fff
    style SHIP_B fill:#c0392b,color:#fff
    style SHIP_A fill:#f39c12,color:#000
    style INCONCLUSIVE fill:#6a3d9a,color:#fff
```

---

## 4. SQL Statistical Analysis Pipeline

How SQL-based statistics flows from raw tables to insights.

```mermaid
flowchart LR
    RAW_TBL["Raw Tables\nin DuckDB / Warehouse\n\ntransactions\nevents\nusers"]

    CTE["Common Table Expressions\n(CTEs)\n\nCalculate intermediate\nstatistics reusably\n\nWITH stats AS (...)"]

    WINDOW["Window Functions\n\nPERCENT_RANK()\nCUME_DIST()\nNTILE(k)\nROW_NUMBER()\nRANK()\nLAG() / LEAD()"]

    AGG["Aggregate Functions\n\nAVG, STDDEV_SAMP\nVAR_SAMP, COUNT\nPERCENTILE_CONT\nSKEWNESS, KURTOSIS\nCORR, REGR_SLOPE"]

    BUCKET["Histogram Bucketing\n\nWIDTH_BUCKET(\n  value, lo, hi, n_bins\n)\n\nEqual-width bins"]

    ZSCORE["Z-Score Normalisation\n\n(x - AVG(x)) /\nSTDDEV_SAMP(x)\n\nFlagged: |z| > 3"]

    IQR["IQR Outlier Detection\n\nQ1 - 1.5*IQR → lower fence\nQ3 + 1.5*IQR → upper fence\n\nRobust to non-normality"]

    FUNNEL["Funnel Analysis\n\nCOUNT(DISTINCT user_id)\nby step\n\nStep-to-step conversion\nand cumulative rates"]

    COHORT["Cohort Retention\n\nJOIN users to events\non cohort_week\n\nRetention = users_active_week_k /\ncohort_size"]

    REGR["SQL Regression\n\nREGR_SLOPE(y, x)\nREGR_INTERCEPT(y, x)\nREGR_R2(y, x)\nCORR(y, x)"]

    RESULT["Results DataFrame\n\nvia .df() to Pandas\n\nFurther analysis\nVisualisation\nExport"]

    RAW_TBL --> CTE
    CTE --> WINDOW
    CTE --> AGG
    CTE --> BUCKET
    CTE --> ZSCORE
    CTE --> IQR
    CTE --> FUNNEL
    CTE --> COHORT
    CTE --> REGR

    WINDOW --> RESULT
    AGG --> RESULT
    BUCKET --> RESULT
    ZSCORE --> RESULT
    IQR --> RESULT
    FUNNEL --> RESULT
    COHORT --> RESULT
    REGR --> RESULT

    style RAW_TBL fill:#34495e,color:#fff
    style RESULT fill:#2d6a4f,color:#fff
    style AGG fill:#4a90d9,color:#fff
    style WINDOW fill:#6a3d9a,color:#fff
    style REGR fill:#e06c1a,color:#fff
```