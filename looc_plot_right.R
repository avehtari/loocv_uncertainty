
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(ggExtra)
library(gridExtra)

library(sn)
# library(emg)

source('sn_fit.R')


SAVE_FIGURE = FALSE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 1

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

ni = 8
n = Ns[ni]

# load data in variable out
load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
    truedist, modeldist, priordist, p0, n))
# drop singleton dimensions
for (name in names(out)) {
    out[[name]] = drop(out[[name]])
}
Niter = dim(out$loos)[2]



# ==============================================================================
# select measure

# M1-M2
loos = out$loos[,,1] - out$loos[,,2]
tls = out$tls[,1] - out$tls[,2]
g2s = out$g2s_d
g3s = out$g3s_d
# g2s = out$g2s_nod_d
# g3s = out$g3s_nod_d


# ==============================================================================
# calc

# loo point estimates
loop_sums = colSums(loos)
loop_means = colMeans(loos)
loop_sds = colSds(loos)
loop_vars = loop_sds**2
loop_skews = apply(loos, 2, skewness)

# loo sum mean estimate
loo_means = loop_sums

# loo sum var estimates
loo_vars_1 = n*loop_vars
loo_vars_2 = 2*loo_vars_1
loo_vars_3 = loo_vars_1 + n*g2s
loo_vars_4 = loo_vars_1 + (n^2)*g3s
# pack them
# var_estim_names = list('naive', 'x2', 'g2', 'g3')
# var_estims = list(loo_vars_1, loo_vars_2, loo_vars_3, loo_vars_4)
var_estim_names = list('naive', 'x2', 'g3')
var_estims = list(loo_vars_1, loo_vars_2, loo_vars_4)

# loo sum skew 3rd moment
loo_3moment = (
    n*(loop_means**3 + 3*loop_means*loop_vars + loop_sds**3*loop_skews) +
    3*n*(n-1)*(loop_means**2 + loop_vars)*loop_means + n*(n-1)*(n-2)*loop_means**3
)
loo_skews = (
    (loo_3moment - 3*loo_means*loo_vars_1 - loo_means**3) / (n**(3/2)*loop_sds**3)
)

# estimate p(loo<elpd_t), normal approx. using all var estimates
test_p_n = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {
    test_p_n[[var_e_i]] = pnorm(
        tls, mean=loo_means, sd=sqrt(var_estims[[var_e_i]]))
}

# estimate p(loo<elpd_t), skew normal approx. using all var estimates
test_p_sn = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {
    sn_param = sn_from_moments(
        loo_means, sqrt(var_estims[[var_e_i]]), loo_skews)
    test_p_sn[[var_e_i]] = psn(
        tls, xi=sn_param$xi, omega=sn_param$omega, alpha=sn_param$alpha)
}


# required uncertainty sigma multipliers
idx_wrong = sign(loo_means) != sign(tls)
n_wrong = sum(idx_wrong)
n_correct = Niter - n_wrong
uncertain_prob = seq(from=n_correct, to=Niter)/Niter
uncertain_multip_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {
    uncertain_multip_s[[var_e_i]] = c(
        0.0,
        sort(abs(loo_means[idx_wrong])/sqrt(var_estims[[var_e_i]][idx_wrong]))
    )
}


# ==============================================================================
# Plot error multiplier

g = ggplot()
for (var_e_i in 1:length(var_estims)) {
    g = g +
        geom_step(
            data=data.frame(
                x=uncertain_multip_s[[var_e_i]],
                y=uncertain_prob
            ),
            aes(x=x, y=y),

        )
}
g

ggplot() + geom_step(
    data=data.frame(
        x=uncertain_multip_s[[3]],
        y=uncertain_prob
    ),
    aes(x=x, y=y)
)
