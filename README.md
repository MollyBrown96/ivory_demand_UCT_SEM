# ivory_demand_UCT_SEM

This repository contains the code used to estimate a hierarchical Bayesian model of sensitive behaviours (here, related to ivory consumption) using the Unmatched Count Technique (UCT) and Structural Equation Modelling (SEM). 

---

## Contents

* `Full_UCT.stan`: Stan model file containing the full SEM + UCT model.
* `data_for_UCT_analysis.csv`: questionnaire data used as input.
* `R script`: Loads data, prepares inputs, runs the Stan model, and produces diagnostics and plots.

---

## Dependencies

Make sure to install the following R packages:

```r
install.packages(c("rstan", "grid"))
```

---

## Overview of the Model

* A binomial-logit model for both UCT control and treatment group responses.
* Latent variables for aesthetic values, environmental attitudes, and legal attitudes.

---

## Running the Model

Compile and run the model using `rstan::stan()`, specifying 3 chains and 2000 iterations (1000 warmup). Initial values are specified using `init_list()` to promote stable sampling.

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

---

## Citation

If using this model or code, please cite:

> Brown, M.R.C., et al., 2025. Strategic behavioural interventions are required to tackle Chinese aesthetic, cultural and social values driving ivory demand. 
