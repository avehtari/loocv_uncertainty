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
  if (p==0) {
    for (i in 1:N)
      log_lik[i] = bernoulli_logit_lpmf(y[i*2-1] | beta0) +
                   bernoulli_logit_lpmf(y[i*2] | beta0);
    for (i in 1:Nt)
      log_likt[i] = bernoulli_logit_lpmf(yt[i*2-1] | beta0) +
                    bernoulli_logit_lpmf(yt[i*2] | beta0);
  } else {
    for (i in 1:N)
      log_lik[i] = bernoulli_logit_lpmf(y[i*2-1] | beta0 + x[i*2-1] * beta) +
                   bernoulli_logit_lpmf(y[i*2] | beta0 + x[i*2] * beta);
    for (i in 1:Nt)
      log_likt[i] = bernoulli_logit_lpmf(yt[i*2-1] | beta0 + xt[i*2-1] * beta) +
                    bernoulli_logit_lpmf(yt[i*2] | beta0 + xt[i*2] * beta);
  }
}
