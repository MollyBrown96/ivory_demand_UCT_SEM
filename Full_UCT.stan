data {
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


