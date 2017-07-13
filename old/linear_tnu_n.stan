data {
  int<lower=0> p;
  int<lower=0> N;
  vector[N] y;
  matrix[N,p] x;
  int<lower=0> Nt;
  vector[Nt] yt;
  matrix[Nt,p] xt;
} 
parameters {
  real beta0;
  vector[p] beta;
  real<lower=0> sigma;
  real<lower=2> nu;
}
model {
  beta0 ~ normal(0, 10);
  beta ~ normal(0, 2.5);
  sigma ~ cauchy(0, 1);
  nu ~ gamma(2, .1);
  if (p==0) {
    y ~ student_t(nu, beta0, sigma);
  } else {
    y ~ student_t(nu, beta0 + x * beta, sigma);
  }
} 
generated quantities {
  vector[N] log_lik;
  vector[Nt] log_likt;
  vector[N] mu;
  vector[Nt] mut;
  if (p==0) {
    mu = rep_vector(beta0, N);
    mut = rep_vector(beta0, Nt);
    for (i in 1:N)
      log_lik[i] = student_t_lpdf(y[i] | nu, beta0, sigma);
    for (i in 1:Nt)
      log_likt[i] = student_t_lpdf(yt[i] | nu, beta0, sigma);
  } else {
    mu = beta0 + x*beta;
    mut = beta0 + xt*beta;
    for (i in 1:N)
      log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
    for (i in 1:Nt)
      log_likt[i] = student_t_lpdf(yt[i] | nu, mut[i], sigma);
  }
}
