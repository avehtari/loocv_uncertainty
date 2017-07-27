functions {
  real sgt_log(vector x, real mu, real s, real l, real p, real q) {
    // Skewed generalised t
    int N;
    real lz1;
    real lz2;
    real v;
    real m;
    real r;
    real out;
    N = dims(x)[1];
    lz1 = lbeta(1.0/p,q);
    lz2 = lbeta(2.0/p,q-1.0/p);
    v = q^(-1.0/p)*((3*l^2+1)*exp(lbeta(3.0/p,q-2.0/p)-lz1)-4*l^2*exp(lz2-lz1)^2)^(-0.5);
    m = 2*v*s*l*q^(1.0/p)*exp(lz2-lz1);
    out = 0;
    for (n in 1:N) {
      r = x[n]-mu+m;
      // out = out+log(p)-log(2*v*s*q^(1.0/p)*exp(lz1)*(fabs(r)^p /(q*(v*s)^p*(l*((r<0) ? -1 : 1)+1)^p)+1)^(1.0/p+q));
      if (r<0)
      	     out = out+log(p)-log(2*v*s*q^(1.0/p)*exp(lz1)*(fabs(r)^p /(q*(v*s)^p*(l*(-1)+1)^p)+1)^(1.0/p+q));
      else
      	     out = out+log(p)-log(2*v*s*q^(1.0/p)*exp(lz1)*(fabs(r)^p /(q*(v*s)^p*(l*(1)+1)^p)+1)^(1.0/p+q));
    }
    return out;
  }
}
data {
  int<lower=0> N;
  vector[N] x;
}
parameters {
  real mu;
  real<lower=0> sigma;
  real<lower=-.99,upper=0> l;
  real<lower=1,upper=10> p;
  real<lower=3.0/p,upper=p*50> q;
}
model {
  mu ~ normal(0,9.0/sqrt(N));
  l ~ normal(0,.5);
  p ~ lognormal(log(2),1);
  q ~ gamma(2,0.1);
  x ~ sgt(mu, sigma, l, p, q);
}
