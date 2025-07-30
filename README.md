# ivory_demand_UCT_SEM

This repository contains the code used to estimate a hierarchical Bayesian model of **sensitive behaviour** related to ivory consumption, using the **Unmatched Count Technique (UCT)**. It incorporates **latent variable modelling** (SEM) for psychological and normative constructs (e.g., ivory aesthetics, legality, environmental concern) and **covariance structures** for important socio-demographic covariates.

---

## Contents

* `Full_UCT.stan`: Stan model file containing the full SEM + UCT model.
* `data_for_UCT_analysis.csv`: Survey data used as input.
* `README.md`: This file.
* `R script` (included inline in your workflow): Loads data, prepares inputs, runs the Stan model, and produces diagnostics and plots.

---

## Dependencies

Make sure to install the following R packages:

```r
install.packages(c("rstan", "grid"))
```

---

## Overview of the Model

The Stan model estimates:

* A **binomial-logit model** for both UCT control and treatment group responses.
* A **latent probability** of endorsing the sensitive behaviour (e.g., ivory product ownership).
* **Latent variables** for:

  * **Ivory aesthetics**
  * **Legal perspectives**
  * **Environmental perspectives**
* **Measurement models** using multiple observed proxy variables for each latent construct.
* **Regression models** for additional predictors (e.g., cultural significance, social status).
* **Multivariate covariance structures** to model dependencies between demographic and attitudinal variables.

---

## Data Inputs

Your R script prepares a named list `ivory_own_stan`, which includes:

| Name             | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| `N`              | Number of observations                                                       |
| `J`              | Number of items in the control group (UCT)                                   |
| `K`              | Number of base covariates (age, income, gender, location)                    |
| `K1`             | Number of additional covariates (e.g., social status, cultural significance) |
| `Pi`, `Pl`, `Pe` | Number of proxies for aesthetics, legal, and environmental latent constructs |
| `Y`              | UCT count outcome variable                                                   |
| `treat`          | 0 = control group, 1 = treatment group                                       |
| `X`, `X1`        | Covariate matrices for base and sensitive models                             |
| `Zi`, `Zl`, `Ze` | Proxy matrices for latent variable measurement models                        |

---

## Running the Model

Your R script compiles and runs the model using `rstan::stan()`, specifying 3 chains and 2000 iterations (1000 warmup). Initial values are specified using `init_list()` to promote stable sampling.

```r
ivory_own_fit <- stan(
  file = "Full_UCT.stan",
  data = ivory_own_stan,
  chains = 3,
  warmup = 1000,
  iter = 2000,
  cores = 1,
  init = init_list,
  control = list(adapt_delta = 0.85)
)
```

---

## Post-Estimation and Plots

Your script includes:

* Posterior plots for selected regression coefficients and latent effects:

```r
plot(ivory_own_fit, pars = c("beta_cult[2]", "beta_aes[2]", "gamma_aes", ...))
```

* Diagnostics for latent variables, variances, and covariance terms:

```r
plot(ivory_own_fit, pars = c("theta_aes", "Sigma_cov_age_inc[1,2]", ...))
```

* Traceplots to visually inspect convergence:

```r
traceplot(ivory_own_fit, pars = c("beta_base", "gamma_aes", "theta_env"))
```

* Summarisation:

```r
summary(ivory_own_fit, pars = c("beta_base", "gamma_aes", "beta_cult", ...))
```

---

## Model Output

The model object can be saved and reused:

```r
saveRDS(ivory_own_fit, file = "FINAL_Ivory_own_fit.rds")
```

Or loaded later:

```r
ivory_own_fit <- readRDS("APRIL_Ivory_own_fit.rds")
```

---

## Interpretation

* **`gamma_*` coefficients** quantify the influence of latent constructs (e.g., aesthetics, legality, environment) on the latent sensitive behaviour.
* **`theta_*` parameters** represent factor loadings from observed proxies onto the latent variables.
* **Covariance matrices** offer insight into shared variance between constructs (e.g., aesthetics and social status).

---

## Citation

If using this model or code, please cite:

> Molly \[Brown], \[2025]. Strategic behavioural interventions are required to tackle Chinese aesthetic, cultural and social values driving ivory demand. 
