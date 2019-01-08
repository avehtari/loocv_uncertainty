
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

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

ni = 3
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
ref_p = sapply(tls, function(x) mean(tls < x))  # uniform
# p(loo<elpd_t)
# ref_p = sapply(tls, function(x) mean(loop_sums < x))

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
# plot pairwise plots

dev.new()
g = ggplot()
g = g + geom_abline(intercept=0, slope=1, colour='red')
g = g + geom_point(aes(x=tls, y=loop_sums))
g = g +
    xlab("elpd") +
    ylab("loo") +
    ggtitle(sprintf(
        "%s, %s_%s_%s, p0=%g n=%g",
        loo_name, truedist, modeldist, priordist, p0, n
    ))
g = ggMarginal(g, type='histogram')
print(g)
if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=8, height=5,
        filename = sprintf(
            "figs/elpd-loo_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        )
    )
}

g_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {

    # test_p = test_p_n[[var_e_i]]
    # test_p_name = 'normal'

    test_p = test_p_sn[[var_e_i]]
    test_p_name = 'skew-normal'

    var_estim_name = var_estim_names[[var_e_i]]

    g = ggplot()
    g = g + geom_abline(intercept=0, slope=1, colour='red')
    g = g + geom_point(
            data=data.frame(
                x=ref_p,
                y=test_p
            ),
            aes(x=x, y=y)
        )
    g = g +
        xlab("p(elpd<elpd_t)") +
        ylab(sprintf("p(loo<elpd_t), %s", var_estim_name))
    g = ggMarginal(g, type='histogram')
    g_s[[var_e_i]] = g
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/pelpd-ploo_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "%s, %s_%s_%s, p0=%g, n=%g, %s",
    loo_name, truedist, modeldist, priordist, p0, n, test_p_name
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()


# ==============================================================================
# plot comparison normal vs skew-normal

g_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {

    diff_p = test_p_n[[var_e_i]] - test_p_sn[[var_e_i]]
    var_estim_name = var_estim_names[[var_e_i]]

    g = ggplot()
    g = g + geom_point(
            data=data.frame(
                x=ref_p,
                y=diff_p
            ),
            aes(x=x, y=y)
        )
    g = g +
        xlab("p(elpd<elpd_t)") +
        ylab("delta p(loo<elpd_t)") +
        ggtitle(var_estim_name)
    g = ggMarginal(g, type='histogram', margins='y')
    g_s[[var_e_i]] = g
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/delta-pelpd-ploo_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "%s, %s_%s_%s, p0=%g, n=%g",
    loo_name, truedist, modeldist, priordist, p0, n
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()


# ==============================================================================
# plot uniformity check histograms

g_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {

    # test_p = test_p_n[[var_e_i]]
    # test_p_name = 'normal'

    test_p = test_p_sn[[var_e_i]]
    test_p_name = 'skew-normal'

    var_estim_name = var_estim_names[[var_e_i]]

    # bins
    B = 8

    # plot
    g = ggplot()
    g = g + geom_histogram(
            data=data.frame(
                x=test_p
            ),
            aes(x=x),
            breaks=seq(from=0, by=1/B, length=B+1)
        )
    g = g +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
        geom_hline(yintercept=Niter/B) +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
        xlab("") +
        ylab("") +
        ggtitle(sprintf("%s", var_estim_name))
    g_s[[var_e_i]] = g
}
# share ymax
max_bin = max(sapply(g_s, function(g) max(ggplot_build(g)$data[[1]]$y)))
for (var_e_i in 1:length(var_estims)) {
    g_s[[var_e_i]] = g_s[[var_e_i]] + ylim(0, max_bin)
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/dist_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "%s, %s_%s_%s, p0=%g, n=%g, %s",
    loo_name, truedist, modeldist, priordist, p0, n, test_p_name
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()



# ==============================================================================
# plot uniformity check histograms grouped

g_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {

    # test_p = test_p_n[[var_e_i]]
    # test_p_name = 'normal'

    test_p = test_p_sn[[var_e_i]]
    test_p_name = 'skew-normal'

    var_estim_name = var_estim_names[[var_e_i]]

    # bins
    B = 8

    # plot
    g = ggplot()
    g = g + geom_histogram(
            data=data.frame(
                x=test_p
            ),
            aes(x=x),
            breaks=seq(from=0, by=1/B, length=B+1)
        )
    g = g +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
        geom_hline(yintercept=Niter/B) +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
        xlab("") +
        ylab("") +
        ggtitle(sprintf("%s", var_estim_name))
    g_s[[var_e_i]] = g
}
# share ymax
max_bin = max(sapply(g_s, function(g) max(ggplot_build(g)$data[[1]]$y)))
for (var_e_i in 1:length(var_estims)) {
    g_s[[var_e_i]] = g_s[[var_e_i]] + ylim(0, max_bin)
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/dist_%s_%s_%s_%i_%i_%s.pdf",
            truedist, modeldist, priordist, p0, n, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "%s, %s_%s_%s, p0=%g, n=%g, %s",
    loo_name, truedist, modeldist, priordist, p0, n, test_p_name
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()
