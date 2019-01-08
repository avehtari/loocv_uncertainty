
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
COMPARISON = FALSE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 1

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

ni = 1
n = Ns[ni]

# load data in variable out
load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
    truedist, modeldist, priordist, p0, n))
# remove some last trials (to make binning easier)
# also drop singleton dimensions
Niter_full = dim(out$loos)[2]
Niter = Niter_full - 1  # drop one sample to get 1999
for (name in names(out)) {
    dims = dim(out[[name]])
    if (length(dims) == 2) {
        out[[name]] = out[[name]][,1:Niter]
    } else if (length(dims) == 3) {
        out[[name]] = out[[name]][,1:Niter,]
    } else if (length(dims) == 4) {
        out[[name]] = out[[name]][,,1:Niter,]
    } else {
        stop('Unrecognised name in out part')
    }
}



# ==============================================================================
# select measure

if (COMPARISON) {
    # M1-M2
    loo_name = 'M1-M2'
    loos = out$loos[,,1] - out$loos[,,2]
    tls = out$tls[,1] - out$tls[,2]
    g2s = out$g2s_d
    g3s = out$g3s_d
    # g2s = out$g2s_nod_d
    # g3s = out$g3s_nod_d
} else {
    # Mi
    m_i = 1
    loo_name = sprintf('M%d', m_i)
    loos = out$loos[,,m_i]
    tls = out$tls[,m_i]
    g2s = out$g2s[,m_i]
    g3s = out$g3s[,m_i]
    # g2s = out$g2s_nod[,m_i]
    # g3s = out$g3s_nod[,m_i]
}

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
var_estim_names = list('naive', 'x2', 'g2', 'g3')
var_estims = list(loo_vars_1, loo_vars_2, loo_vars_3, loo_vars_4)

# loo sum skew 3rd moment
loo_3moment = (
    n*(loop_means**3 + 3*loop_means*loop_vars + loop_sds**3*loop_skews) +
    3*n*(n-1)*(loop_means**2 + loop_vars)*loop_means + n*(n-1)*(n-2)*loop_means**3
)
loo_skews = (
    (loo_3moment - 3*loo_means*loo_vars_1 - loo_means**3) / (n**(3/2)*loop_sds**3)
)


# p(elpd<elpd_t)
# run looc_plot_dist instead for this
# ref_p = sapply(tls, function(x) mean(tls < x))

# p(loo<elpd_t)
ref_p = sapply(tls, function(x) mean(loop_sums < x))


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

# ==============================================================================
# plot ranks

g_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {

    # test_p = test_p_n[[var_e_i]]
    # test_p_name = sprintf('normal, %s var', var_estim_names[[var_e_i]])

    test_p = test_p_sn[[var_e_i]]
    test_p_name = sprintf('skew-normal, %s var', var_estim_names[[var_e_i]])

    # calc
    Ntg = Niter
    ranks = sapply(test_p, function(x) sum(x > ref_p))

    # combine bins
    comb = 200
    B = (Ntg+1)/comb
    if (comb > 1) {
        ranks_comb = array(NA, Niter)
        for (bi in 0:(B-1)) {
            selected = array(FALSE, Niter)
            for (ci in 0:(comb-1)) {
                selected = selected | (ranks == (bi*comb+ci))
            }
            ranks_comb[selected] = bi
        }
    } else {
        ranks_comb = ranks
    }

    # plot
    g = ggplot()
    g = g + geom_bar(
            data=data.frame(
                x=ranks_comb
            ),
            aes(x=x)
        )
    g = g +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
        geom_hline(yintercept=Niter/B) +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
        xlab("") +
        ylab("") +
        ggtitle(sprintf("%s", test_p_name))
    g_s[[var_e_i]] = g
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/rank_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "%s, %s_%s_%s, p0=%g n=%g",
    loo_name, truedist, modeldist, priordist, p0, n
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()
