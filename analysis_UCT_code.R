library(rstan)   ## for help read:https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html
library(list)
library(grid)

## Stan SEM Code:
cat('data {
  int<lower=0> N;                            // Total number of observations
  int<lower=0> J;                            // Maximum value of Y in base dataset
  int<lower=0> K;                            // Number of covariates in base model
  int<lower=0> K1;                           // Number of additional covariates in sensitive model
  int<lower=0> Pi;                           // Number of proxies for "aes_p" model
  int<lower=0> Pl;                           // Number of proxies for "legal_p" model
  int<lower=0> Pe;                           // Number of proxies for "env_p" model
  int<lower=0,upper=1> treat[N];             // Indicator of dataset (0 = control, 1 = treatment)
  int<lower=0,upper=J+1> Y[N];               // Outcome variable
  matrix[N, K] X;                            // Covariate matrix for base model
  matrix[N, K1] X1;                          // Additional covariate matrix for sensitive model
  matrix[N, Pi] Zi;                          // Proxy matrix for "ivory perspectives" model
  matrix[N, Pl] Zl;                          // Proxy matrix for "legal perspectives" model
  matrix[N, Pe] Ze;                          // Proxy matrix for "environmental perspectives" model

}

parameters {
  vector[K] beta_base;                       // Coefficients for base logistic regression
  vector[K1] beta_add;                       // Coefficients for additional covariates in sensitive binomial model
  vector[K] beta_sens;                       // Coefficients for base covariates in sensitive binary model
  vector[K] beta_soc;                        // Coefficients for base covariates in model for social status
  vector[K] beta_cult;                       // Coefficients for base covariates in model for cultural significance
  vector[K] beta_aes;                        // Coefficients for base covariates in latent model for aesthetics
  vector[K] beta_legal;                      // Coefficients for base covariates in latent model for legal
  vector[K] beta_env;                        // Coefficients for base covariates in latent model for environmental perceptions
  vector[Pi] theta_aes;                      // Loading for latent variables in "ivory aesthetics" model
  vector[Pl] theta_legal;                    // Loading for latent variables in "legal perspectives" model
  vector[Pe] theta_env;                      // Loading for latent variables in "elephant perspectives" model
  vector[N] ivory_aesthetics;                // Latent "ivory aesthetics" variable
  vector[N] legal_perspectives;              // Latent "legal perspectives" variable
  vector[N] env_perspectives;                // Latent "environmental perspectives" variable
  real<lower = -5,upper = 5> gamma_aes;      // Coefficients for aesthetics impact on binary model
  real<lower = -5,upper = 5> gamma_legal;    // Coefficients for legal perspectives impact on binary model
  real<lower = -5,upper = 5> gamma_env;      // Coefficients for environmental perspectives impact on binary model
  real<lower=0> sigma_cult;                  // Variance of base regression for cultural significance 
  real<lower=0> sigma_soc;                   // Variance of base regression for social status 
  
  // Define separate covariance structures
  cholesky_factor_corr[2] L_cov_age_inc;     // Covariance structure between age and income
  cholesky_factor_corr[2] L_cov_social_cult; // Covariance structure between social status and cultural significance
  cholesky_factor_corr[2] L_cov_aes_cult;    // Covariance structure between aesthetics and cultural significance
  cholesky_factor_corr[2] L_cov_social_aes;  // Covariance structure between social status and aesthetics
}

transformed parameters {
  // Covariance matrices based on Cholesky decomposition
  matrix[2,2] Sigma_cov_age_inc = multiply_lower_tri_self_transpose(L_cov_age_inc);           // Covariance between age and income
  matrix[2,2] Sigma_cov_social_cult = multiply_lower_tri_self_transpose(L_cov_social_cult);   // Covariance between social status and cultural significance
  matrix[2,2] Sigma_cov_aes_cult = multiply_lower_tri_self_transpose(L_cov_aes_cult);         // Covariance between aesthetics and cultural significance
  matrix[2,2] Sigma_cov_social_aes = multiply_lower_tri_self_transpose(L_cov_social_aes);     // Covariance between social status and aesthetics

  // Continuous latent probability (sensitive variable)
  vector<lower=0,upper=1>[N] sensitive;  // Continuous latent probability that depends on covariates and latent variables
  for (n in 1:N) {
  // The following equation calculates the continuous latent probability for each observation (n)
  // The probability is determined by a logistic function (inv_logit), which ensures the output is constrained between 0 and 1.
  
    sensitive[n] = inv_logit(X[n] * beta_sens + X1[n] * beta_add 
                   + gamma_aes * ivory_aesthetics[n]
                   + gamma_legal * legal_perspectives[n]
                   + gamma_env * env_perspectives[n]);

  }
}

model {
  // Priors for coefficients (normal prior with mean 0 and standard deviation 5)
  beta_base   ~ normal(0, 5);    // base model parameters
  beta_sens   ~ normal(0, 5);    // sensitive model parameters
  beta_add    ~ normal(0, 5);    // additional sensitive model parameters
  beta_soc    ~ normal(0, 5);    // base model parameters for social 
  beta_cult   ~ normal(0, 5);    // base model parameters for cultural
  beta_aes    ~ normal(0, 5);    // coefficients for latent variable parameters (SEM component)
  beta_legal  ~ normal(0, 5);    // coefficients for latent variable parameters (SEM component)
  beta_env    ~ normal(0, 5);    // coefficients for latent variable parameters (SEM component)
  gamma_aes   ~ normal(0, 5);    // coefficients for aesthetics in sensitive model 
  gamma_legal ~ normal(0, 5);    // coefficients for legal perspectives in sensitive model
  gamma_env   ~ normal(0, 5);    // coefficients for environmental perspectives in sensitive model
  sigma_cult  ~ lognormal(0, 1); // variances for cultural significance model (normal regression model)
  sigma_soc   ~ lognormal(0, 1); // variances for social status model (normal regression model)
  theta_aes   ~ gamma(2, 2);     // loadings for latent variable (ivory aesthetics) 
  theta_legal ~ gamma(2, 2);     // loadings for latent variable (legal perspectives)
  theta_env   ~ gamma(2, 2);     // loadings for latent variable (environmental perspectives)

  // Priors for covariance structures using LKJ correlation prior
  L_cov_age_inc ~ lkj_corr_cholesky(2);      // Covariance between age and income
  L_cov_social_cult ~ lkj_corr_cholesky(2);  // Covariance between social status and cultural significance
  L_cov_aes_cult ~ lkj_corr_cholesky(2);     // Covariance between aesthetics and cultural significance
  L_cov_social_aes ~ lkj_corr_cholesky(2);   // Covariance between social status and aesthetics

  // Multi-normal distributions for the covariates (age, income, social status, cultural significance, etc.)
  for (n in 1:N) {
    [X[n, 2], X[n, 3]]   ~ multi_normal(rep_vector(0, 2), Sigma_cov_age_inc);                // X[,2] = age, X[,3] = income
    [X1[n, 2], X1[n, 1]] ~ multi_normal(rep_vector(0, 2), Sigma_cov_social_cult);            // X1[,1] = Cultural Significance, X1[,2] = Social Status
    [ivory_aesthetics[n], X1[n, 1]] ~ multi_normal(rep_vector(0, 2), Sigma_cov_aes_cult);    // X1[,1] = Cultural Significance
    [ivory_aesthetics[n], X1[n, 2]] ~ multi_normal(rep_vector(0, 2), Sigma_cov_social_aes);  // X1[,2] = Social Status
  }

  // Main regression models for control and treatment groups

  for (n in 1:N) {
    if (treat[n] == 0) {
      // Base dataset: Logistic regression
      Y[n] ~ binomial_logit(J, X[n] * beta_base);                                              // Logistic regression for control group
    } else {
      // Treatment dataset, using continuous latent probability
      Y[n] ~ binomial_logit(J+1, X[n] * beta_base + X1[n] * beta_add + logit(sensitive[n]));   // Logistic regression for treatment group
    }
  }

  // Latent variable models (normal priors for aesthetics, legal, and environmental perspectives)
  ivory_aesthetics   ~ normal(X * beta_aes, 1);
  legal_perspectives ~ normal(X * beta_legal, 1);
  env_perspectives   ~ normal(X * beta_env, 1);
  
  // Simple regression models for cultural significance and social status
  X1[,1]   ~ normal(X * beta_cult, sigma_cult); // simple model for cultural significance
  X1[,2]   ~ normal(X * beta_soc, sigma_soc);   // simple model for social status

  // Measurement models for latent variables
  for (pi in 1:Pi) {
    Zi[, pi] ~ normal(theta_aes[pi] * ivory_aesthetics, 1);       // Measurement model for ivory aesthetics
  }
  for (pl in 1:Pl) {
    Zl[, pl] ~ normal(theta_legal[pl] * legal_perspectives, 1);   // Measurement model for legal perspectives
  }
  for (pe in 1:Pe) {
    Ze[, pe] ~ normal(theta_env[pe] * env_perspectives, 1);       // Measurement model for environmental perspectives
  }
}


', file = "Full_UCT.stan")

# Load data:
ivory <- read.csv("data_for_UCT_analysis.csv")
colnames(ivory)
ivory$IV_LocationProvince <- ivory$IV_LocationProvince - 1
ivory$IV_Gender <- ivory$IV_Gender - 1

### MODELS ####

### ivory ownership products ----
ivory_own_stan <- list(N = NROW(ivory),
                   J =  6,  # number of questions in control group   (6 in control, 7 in treatment for UCT_own_products_score)
                   K =  5,  # number of parameters in the base model for control questions (including intercept)
                   K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                   Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                   Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                   Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                   Y  = ivory$UCT_Own_Products_score,      # dependent variable
                   treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                   X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                   X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                   Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                   Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                   Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)

init_list <- function() {
  list(
    beta_base = rnorm(ivory_own_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_own_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_own_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_own_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_own_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_own_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_own_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_own_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_own_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_own_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_own_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_own_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_own_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_own_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}

## fit the model: 
ivory_own_fit <- stan(
  file = "Full_UCT.stan", # Stan program
  data = ivory_own_stan,      # named list of data
  chains = 3,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain 
  cores = 1,              # number of cores (could use one per chain)
  init = init_list,       # initial values function
  control = list(adapt_delta = 0.85)
  )                         


plot(ivory_own_fit,  pars = c(#"beta_sens", "beta_add", #"beta_base", 
                           #"gamma_aes", "gamma_legal", "gamma_env",
  "beta_cult[2]", "beta_cult[3]", "beta_cult[4]", "beta_cult[5]",                        
  "beta_aes[2]", "beta_aes[5]", 
  "beta_legal[4]", "beta_legal[5]",
  "beta_env[3]", "beta_env[5]",
  "gamma_aes"
                           #"beta_soc", "beta_cult", "theta_aes", "theta_legal", "theta_env",
                           #"sigma_cult", "sigma_soc",
                           #"Sigma_cov_age_inc[1,2]", 
                           #"Sigma_cov_social_cult[1,2]",
                           #"Sigma_cov_aes_cult[1,2]", 
                           #"Sigma_cov_social_aes[1,2]")
  )
)
grid.text("Ivory Ownership (Products)", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))


plot(ivory_own_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                              "sigma_cult", "sigma_soc",
                            "Sigma_cov_age_inc[1,2]", 
                            "Sigma_cov_social_cult[1,2]",
                            "Sigma_cov_aes_cult[1,2]", 
                            "Sigma_cov_social_aes[1,2]"
))

traceplot(ivory_own_fit, pars = c("beta_base", "beta_sens", "beta_add",
                               "gamma_aes", "gamma_legal", "gamma_env",
                               "beta_aes", "beta_legal", "beta_env",
                               "beta_soc", "beta_cult"),
          inc_warmup = FALSE, nrow = 3)

traceplot(ivory_own_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                            "Sigma_cov_age_inc[1,2]", 
                            "Sigma_cov_social_cult[1,2]",
                            "Sigma_cov_aes_cult[1,2]", 
                            "Sigma_cov_social_aes[1,2]"
                          ),
          inc_warmup = FALSE, nrow = 3)


# summary:
ivory_own_fit_summary <- summary(ivory_own_fit, c("beta_base", "beta_sens", "beta_add",
                      "gamma_aes", "gamma_legal", "gamma_env",
                      "beta_aes", "beta_legal", "beta_env",
                      "beta_soc", "beta_cult"))
## save things:   
#ivory_own_fit@stanmodel@dso <- new("cxxdso")
#saveRDS(ivory_own_fit, file = "FINAL_Ivory_own_fit.rds")
#ivory_own_fit <- readRDS("FINAL_Ivory_own_fit.rds")

ivory_own_fit <- readRDS("APRIL_Ivory_own_fit.rds")


### own art ### ------


### ivory ownership artwork
ivory_own2_stan <- list(N = NROW(ivory),
                       J =  5,  # number of questions in control group   (5 in control, 6 in treatment for UCT_own_ART_score)
                       K =  5,  # number of parameters in the base model for control questions (including intercept)
                       K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                       Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                       Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                       Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                       Y  = ivory$UCT_Own_Art_score,      # dependent variable
                       treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                       X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                       X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                       Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                    Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                       Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                       Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)


init_list <- function() {
  list(
    beta_base = rnorm(ivory_own2_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_own2_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_own2_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_own2_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_own2_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_own2_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_own2_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_own2_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_own2_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_own2_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_own2_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_own2_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_own2_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_own2_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}


## fit the model: 
ivory_own2_fit <- stan(
  file = "Full_UCT.stan", # Stan program
  data = ivory_own2_stan,      # named list of data
  chains = 3,             # number of Markov chains
  warmup = 1000,          # number of warmup iterations per chain
  iter = 2000,            # total number of iterations per chain 
  cores = 1,              # number of cores (could use one per chain)
  init = init_list,       # initial values function
  control = list(adapt_delta = 0.85)
)                         


plot(ivory_own2_fit,  pars = c("beta_sens", "beta_add", # "beta_base", 
                               "gamma_aes", "gamma_legal", "gamma_env",
                               "beta_aes", "beta_legal", "beta_env",
                               "beta_soc", "beta_cult",
                               "theta_aes", "theta_legal", "theta_env",
                               "Sigma_cov_age_inc[1,2]", 
                               "Sigma_cov_social_cult[1,2]",
                               "Sigma_cov_aes_cult[1,2]", 
                               "Sigma_cov_social_aes[1,2]"))
grid.text("Ivory Ownership (Artwork)", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))


plot(ivory_own2_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                              "Sigma_cov_age_inc[1,2]", 
                              "Sigma_cov_social_cult[1,2]",
                              "Sigma_cov_aes_cult[1,2]", 
                              "Sigma_cov_social_aes[1,2]"
))
plot(ivory_own2_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                              "Sigma_cov_age_inc", 
                              "Sigma_cov_social_cult",
                              "Sigma_cov_aes_cult", 
                              "Sigma_cov_social_aes"
))

traceplot(ivory_own2_fit, pars = c("beta_base", "beta_sens", "beta_add",
                                  "gamma_aes", "gamma_legal", "gamma_env",
                                  "beta_aes", "beta_legal", "beta_env"),
          inc_warmup = FALSE, nrow = 3)

traceplot(ivory_own2_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                                  "Sigma_cov_age_inc[1,2]", 
                                  "Sigma_cov_social_cult[1,2]",
                                  "Sigma_cov_aes_cult[1,2]", 
                                  "Sigma_cov_social_aes[1,2]"
),
inc_warmup = FALSE, nrow = 3)


# summary:
summary(ivory_own2_fit, c("beta_sens", "beta_add",
                         "gamma_aes", "gamma_legal", "gamma_env",
                         "beta_aes", "beta_legal", "beta_env"))


## save things:   
ivory_own2_fit@stanmodel@dso <- new("cxxdso")
saveRDS(ivory_own2_fit, file = "FINAL_Ivory_own_art_fit.rds")
ivory_own2_fit <- readRDS("FINAL_Ivory_own_art_fit.rds")

###########
# GIFTING #

## gifting accessories model ----
ivory_gift1_stan <- list(N = NROW(ivory),
                   J =  5,  # number of questions in control group   (6 in control, 7 in treatment for log_UCT_own_products)
                   K =  5,  # number of parameters in the base model for control questions (including intercept)
                   K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                   Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                   Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                   Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                   Y  = ivory$UCT_GiftGiving_Acc_score,      # dependent variable
                   treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                   X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                   X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                   Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                   Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                   Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)

init_list <- function() {
  list(
    beta_base = rnorm(ivory_gift1_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_gift1_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_gift1_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_gift1_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_gift1_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_gift1_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_gift1_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_gift1_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_gift1_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_gift1_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_gift1_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_gift1_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_gift1_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_gift1_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}



## fit the model: 
ivory_gift1_fit <- stan(
  file = "Full_UCT.stan",   # Stan program
  data = ivory_gift1_stan,  # named list of data
  chains = 3,               # number of Markov chains
  warmup = 1000,            # number of warmup iterations per chain
  iter = 2000,              # total number of iterations per chain 
  cores = 1,                # number of cores (could use one per chain)
  init = init_list,         # initial values function
  control = list(adapt_delta = 0.85)
)                         


plot(ivory_gift1_fit,  pars = c("beta_sens", "beta_add", "beta_base",
                                "gamma_aes", "gamma_legal", "gamma_env",
                                "beta_aes", "beta_legal", "beta_env",  "beta_soc", "beta_cult",
                                "theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]")
)
grid.text("Ivory Gifting Accessories", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))


plot(ivory_gift1_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]"
))
plot(ivory_gift1_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc", 
                                "Sigma_cov_social_cult",
                                "Sigma_cov_aes_cult", 
                                "Sigma_cov_social_aes"
))

traceplot(ivory_gift1_fit, pars = c("beta_base", "beta_sens", "beta_add",
                                    "gamma_aes", "gamma_legal", "gamma_env",
                                    "beta_aes", "beta_legal", "beta_env"),
          inc_warmup = FALSE, nrow = 3)
traceplot(ivory_gift1_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                                    "Sigma_cov_age_inc[1,2]", 
                                    "Sigma_cov_social_cult[1,2]",
                                    "Sigma_cov_aes_cult[1,2]", 
                                    "Sigma_cov_social_aes[1,2]"
),
inc_warmup = FALSE, nrow = 3)


# summary:
ivory_gift1_fit_summary <- summary(ivory_gift1_fit, c("beta_base", "beta_sens", "beta_add",
                           "gamma_aes", "gamma_legal", "gamma_env",
                           "beta_aes", "beta_legal", "beta_env"))

ivory_gift1_fit@stanmodel@dso <- new("cxxdso")
saveRDS(ivory_gift1_fit, file = "APRIL_ivory_gift1_fit.rds")
ivory_gift1_fit <- readRDS("APRIL_ivory_gift1_fit.rds") 


##### gift giving art #### ----

## gifting art model
ivory_gift2_stan <- list(N = NROW(ivory),
                         J =  5,  # number of questions in control group   (6 in control, 7 in treatment for log_UCT_own_products)
                         K =  5,  # number of parameters in the base model for control questions (including intercept)
                         K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                         Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                         Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                         Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                         Y  = ivory$UCT_GiftGiving_Art_score,      # dependent variable
                         treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                         X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                         X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                         Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                      Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                         Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                         Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)

init_list <- function() {
  list(
    beta_base = rnorm(ivory_gift2_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_gift2_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_gift2_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_gift2_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_gift2_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_gift2_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_gift2_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_gift2_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_gift2_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_gift2_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_gift2_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_gift2_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_gift2_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_gift2_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}

## fit the model: 
ivory_gift2_fit <- stan(
  file = "Full_UCT.stan",   # Stan program
  data = ivory_gift2_stan,  # named list of data
  chains = 3,               # number of Markov chains
  warmup = 1000,            # number of warmup iterations per chain
  iter = 2000,              # total number of iterations per chain 
  cores = 1,                # number of cores (could use one per chain)
  init = init_list,         # initial values function
  control = list(adapt_delta = 0.85)
)                         

plot(ivory_gift2_fit,  pars = c("beta_base", "beta_sens", "beta_add",
                                "gamma_aes", "gamma_legal", "gamma_env",
                                "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc",
                                "theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]")
)
grid.text("Ivory Gifting Artwork", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))

plot(ivory_gift1_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]"
))
plot(ivory_gift1_fit,  pars = c("theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc", 
                                "Sigma_cov_social_cult",
                                "Sigma_cov_aes_cult", 
                                "Sigma_cov_social_aes"
))

traceplot(ivory_gift1_fit, pars = c("beta_base", "beta_sens", "beta_add",
                                    "gamma_aes", "gamma_legal", "gamma_env",
                                    "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc"),
          inc_warmup = FALSE, nrow = 3)
traceplot(ivory_gift1_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                                    "Sigma_cov_age_inc[1,2]", 
                                    "Sigma_cov_social_cult[1,2]",
                                    "Sigma_cov_aes_cult[1,2]", 
                                    "Sigma_cov_social_aes[1,2]"
),
inc_warmup = FALSE, nrow = 3)


# summary:
summary(ivory_gift2_fit, c("beta_base", "beta_sens", "beta_add",
                           "gamma_aes", "gamma_legal", "gamma_env",
                           "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc"))

ivory_gift2_fit@stanmodel@dso <- new("cxxdso")
saveRDS(ivory_gift2_fit, file = "APRIL_ivory_gift2_fit.rds")
ivory_gift2_fit <- readRDS("APRIL_ivory_gift2_fit.rds")


### inheritance models ----

## inherit accessories model ----
colnames(ivory)
ivory_inherit1_stan <- list(N = NROW(ivory),
                         J =  4,  # number of questions in control group   (4 in control, 5 in treatment for UCT_Inheriting_Acc_score)
                         K =  5,  # number of parameters in the base model for control questions (including intercept)
                         K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                         Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                         Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                         Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                         Y  = ivory$UCT_Inheriting_Acc_score,      # dependent variable
                         treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                         X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                         X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                         Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                      Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                         Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                         Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)

init_list <- function() {
  list(
    beta_base = rnorm(ivory_inherit1_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_inherit1_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_inherit1_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_inherit1_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_inherit1_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_inherit1_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_inherit1_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_inherit1_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_inherit1_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}


## fit the model: 
ivory_inherit1_fit <- stan(
  file = "Full_UCT.stan",   # Stan program
  data = ivory_inherit1_stan,  # named list of data
  chains = 3,               # number of Markov chains
  warmup = 1000,            # number of warmup iterations per chain
  iter = 2000,              # total number of iterations per chain 
  cores = 1,                # number of cores (could use one per chain)
  init = init_list,         # initial values function
  control = list(adapt_delta = 0.85)
)                         


plot(ivory_inherit1_fit,  pars = c("beta_base", "beta_sens", "beta_add",
                                "gamma_aes", "gamma_legal", "gamma_env",
                                "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc",
                                "theta_aes", "theta_legal", "theta_env",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]")
)
grid.text("Ivory Accessories Inheritance", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))


plot(ivory_inherit1_fit,  pars = c("theta_aes", "theta_legal", "theta_env", "sigma_cult", "sigma_soc",
                                "Sigma_cov_age_inc[1,2]", 
                                "Sigma_cov_social_cult[1,2]",
                                "Sigma_cov_aes_cult[1,2]", 
                                "Sigma_cov_social_aes[1,2]"
))

traceplot(ivory_inherit1_fit, pars = c("beta_base", "beta_sens", "beta_add",
                                    "gamma_aes", "gamma_legal", "gamma_env",
                                    "beta_aes", "beta_legal", "beta_env"),
          inc_warmup = FALSE, nrow = 3)
traceplot(ivory_inherit1_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                                    "Sigma_cov_age_inc[1,2]", 
                                    "Sigma_cov_social_cult[1,2]",
                                    "Sigma_cov_aes_cult[1,2]", 
                                    "Sigma_cov_social_aes[1,2]"
),
inc_warmup = FALSE, nrow = 3)


# summary:
summary(ivory_inherit1_fit, c("beta_base", "beta_sens", "beta_add",
                           "gamma_aes", "gamma_legal", "gamma_env",
                           "beta_aes", "beta_legal", "beta_env"))

ivory_inherit1_fit@stanmodel@dso <- new("cxxdso")
saveRDS(ivory_inherit1_fit, file = "APRIL_ivory_inherit1_fit.rds")
ivory_inherit1_fit <- readRDS("ivory_inherit1_fit.rds")


## inherit artwork model -----
colnames(ivory)
ivory_inherit2_stan <- list(N = NROW(ivory),
                            J =  5,  # number of questions in control group   (5 in control, 6 in treatment for UCT_Inheriting_Art_score)
                            K =  5,  # number of parameters in the base model for control questions (including intercept)
                            K1 = 2,  # number of parameters in the addition model for sensitive questions (including intercept)
                            Pi = 3,  # number of proxies for the latent variable model for ivory aesthetics  
                            Pl = 2,  # number of proxies for in the latent variable model for legal perspectives  
                            Pe = 2,  # number of proxies for in the latent variable model for environmental perspectives  
                            Y  = ivory$UCT_Inheriting_Art_score,      # dependent variable
                            treat = ivory$TreatmentGroup , # treatment indicator - a numeric vector of 0 (control) and 1 (treatment)
                            X  = matrix(c(model.matrix(~ Age_mid + Income_mid + IV_LocationProvince + IV_Gender, data = ivory)), ncol = 5),   # matrix of covariates (including intercept) for base model. ncol = K
                            X1 = matrix(c(model.matrix(~ -1 + IV_CulturalSignificance + IV_SocialStatusPC , data = ivory)), ncol = 2),   # matrix of covariates (including intercept) for base model. ncol = K
                            Zi = matrix(c(model.matrix(~ -1 + Perception_beauty_scaled + Perception_works_of_art_scaled +
                                                         Perception_unique_scaled, data = ivory)), ncol = 3),   # model matrix for ivory perspectives. ncol = Ki
                            Zl = matrix(c(scale(model.matrix(~ -1 +  Legal_1 + Legal_2, data = ivory))), ncol = 2),   # model matrix for legal perspectives. ncol = Kl ## add in original scores
                            Ze = matrix(c(scale(model.matrix(~ -1 + Env_1 + Env_2, data = ivory))), ncol = 2)    # model matrix for env perspective. ncol = Ke
)


init_list <- function() {
  list(
    beta_base = rnorm(ivory_inherit2_stan$K, 0, 0.1), # Smaller initial values
    beta_add = rnorm(ivory_inherit2_stan$K1, 0, 0.1),
    beta_sens = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    beta_aes = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    beta_legal = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    beta_env = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    beta_cult = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    beta_soc = rnorm(ivory_inherit2_stan$K, 0, 0.1),
    theta_aes = rgamma(ivory_inherit2_stan$Pi, 2, 2), # Gamma(2,2)
    theta_legal = rgamma(ivory_inherit2_stan$Pl, 2, 2),
    theta_env = rgamma(ivory_inherit2_stan$Pe, 2, 2),
    ivory_aesthetics = rnorm(ivory_inherit2_stan$N, 0, 0.1), # Smaller initial values
    legal_perspectives = rnorm(ivory_inherit2_stan$N, 0, 0.1),
    env_perspectives = rnorm(ivory_inherit2_stan$N, 0, 0.1),
    gamma_aes = rnorm(1, 0, 0.1), # Smaller initial values
    gamma_legal = rnorm(1, 0, 0.1),
    gamma_env = rnorm(1, 0, 0.1),
    Sigma_cov_age_inc = diag(2),
    Sigma_cov_social_cult = diag(2),
    Sigma_cov_aes_cult = diag(2),
    Sigma_cov_social_aes = diag(2),
    sigma_cult = rgamma(1, 10, 4),
    sigma_soc = rgamma(1, 10, 4)
  )
}


## fit the model: 
ivory_inherit2_fit <- stan(
  file = "Full_UCT.stan",   # Stan program
  data = ivory_inherit2_stan,  # named list of data
  chains = 3,               # number of Markov chains
  warmup = 1000,            # number of warmup iterations per chain
  iter = 2000,              # total number of iterations per chain 
  cores = 1,                # number of cores (could use one per chain)
  init = init_list,         # initial values function
  control = list(adapt_delta = 0.85)
)                         

plot(ivory_inherit2_fit, 
     pars = c("beta_base", "beta_sens", "beta_add",
              "gamma_aes", "gamma_legal", "gamma_env",
              "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc",
              "theta_aes", "theta_legal", "theta_env", "sigma_cult", "sigma_soc",
              "Sigma_cov_age_inc[1,2]", 
              "Sigma_cov_social_cult[1,2]",
              "Sigma_cov_aes_cult[1,2]", 
              "Sigma_cov_social_aes[1,2]"))
grid.text("Inheriting Ivory Accessories", y = unit(1, "npc") - unit(0.5, "lines"), gp = gpar(fontsize = 16))

plot(ivory_inherit2_fit,  pars = c("theta_aes", "theta_legal", "theta_env", "sigma_cult", "sigma_soc",
                                   "Sigma_cov_age_inc[1,2]", 
                                   "Sigma_cov_social_cult[1,2]",
                                   "Sigma_cov_aes_cult[1,2]", 
                                   "Sigma_cov_social_aes[1,2]"))

traceplot(ivory_inherit2_fit, pars = c("beta_base", "beta_sens", "beta_add",
                                       "gamma_aes", "gamma_legal", "gamma_env",
                                       "beta_aes", "beta_legal", "beta_env"), inc_warmup = FALSE, nrow = 3)
traceplot(ivory_inherit2_fit, pars = c("theta_aes", "theta_legal", "theta_env",
                                       "Sigma_cov_age_inc[1,2]", 
                                       "Sigma_cov_social_cult[1,2]",
                                       "Sigma_cov_aes_cult[1,2]", 
                                       "Sigma_cov_social_aes[1,2]"), inc_warmup = FALSE, nrow = 3)


# summary:
summary(ivory_inherit2_fit, c("beta_base", "beta_sens", "beta_add",
                              "gamma_aes", "gamma_legal", "gamma_env",
                              "beta_aes", "beta_legal", "beta_env", "beta_cult", "beta_soc"))

ivory_inherit2_fit@stanmodel@dso <- new("cxxdso")
saveRDS(ivory_inherit2_fit, file = "APRIL_ivory_inherit2_fit.rds")
ivory_inherit2_fit <- readRDS("APRIL_ivory_inherit2_fit.rds")

