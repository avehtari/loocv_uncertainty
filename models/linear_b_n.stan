data {
  int<lower=0> p;
  int<lower=0> N;
  int<lower=0,upper=1> y[N*2];
  matrix[N*2,p] x;
  int<lower=0> Nt;
  int<lower=0,upper=1> yt[Nt*2];
  matrix[Nt*2,p] xt;
}
parameters {
  real beta0;
  vector[p] beta;
}
model {
  beta0 ~ normal(0, 10);
  beta ~ normal(0, 2.5);
  if (p==0) {
    y ~ bernoulli_logit(beta0);
  } else {
    y ~ bernoulli_logit(beta0 + x * beta);
  }
}
generated quantities {
  vector[N] log_lik;
  vector[Nt] log_likt;
  vector[N*2] mu;
  vector[Nt*2] mut;
  if (p==0) {
    mu = rep_vector(beta0, N);
    mut = rep_vector(beta0, Nt);
    for (i in 1:N)
      log_lik[i] = bernoulli_logit_lpmf(y[i*2-1] | beta0) +
                   bernoulli_logit_lpmf(y[i*2] | beta0);
    for (i in 1:Nt)
      log_likt[i] = bernoulli_logit_lpmf(yt[i*2-1] | beta0) +
                    bernoulli_logit_lpmf(yt[i*2] | beta0);
  } else {
    mu = beta0 + x*beta;
    mut = beta0 + xt*beta;
    for (i in 1:N)
      log_lik[i] = bernoulli_logit_lpmf(y[i*2-1] | mu[i*2-1]) +
                   bernoulli_logit_lpmf(y[i*2] | mu[i*2]);
    for (i in 1:Nt)
      log_likt[i] = bernoulli_logit_lpmf(yt[i*2-1] | mut[i*2-1]) +
                    bernoulli_logit_lpmf(yt[i*2] | mut[i*2]);
  }
}
