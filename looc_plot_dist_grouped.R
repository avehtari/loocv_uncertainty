
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
FORCE_NONNEGATIVE_G3S = TRUE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 0

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'



bins = 8

# ==============================================================================
# For all n

# initialise empty result plot data
data_out = data.frame(
    n_str=character(),
    bin=character(),
    var_estim=character(),
    count=integer()
)

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

    # load data in variable out
    load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
        truedist, modeldist, priordist, p0, n))
    # drop singleton dimensions
    for (name in names(out)) {
        out[[name]] = drop(out[[name]])
    }
    Niter = dim(out$loos)[2]

    # ==========================================================================
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

    if (FORCE_NONNEGATIVE_G3S) {
        # force g3s nonnegative
        g3s[g3s<0] = 0.0
    }

    # ==========================================================================
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
    var_estim_names = list('naive', 'corr', 'x2')
    var_estims = list(loo_vars_1, loo_vars_4, loo_vars_2)

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


    # ==========================================================================
    # gather plot data

    limits = seq(from=0, by=1/bins, length=bins+1)
    rownames = sapply(
        1:bins, function(bin) sprintf('%g-%g', limits[bin], limits[bin+1]))
    for (var_e_i in 1:length(var_estims)) {
        # normal approx
        data_out = rbind(
            data_out,
            data.frame(
                n_str=sprintf('n=%g', n),
                bin=rownames,
                var_estim=sprintf(
                    '%s, %s',
                    var_estim_names[[var_e_i]],
                    'normal approx.'
                ),
                count=binCounts(test_p_n[[var_e_i]], bx=limits)
            )
        )
        # skew-normal approx
        data_out = rbind(
            data_out,
            data.frame(
                n_str=sprintf('n=%g', n),
                bin=rownames,
                var_estim=sprintf(
                    '%s, %s',
                    var_estim_names[[var_e_i]],
                    'skew-normal approx.'
                ),
                count=binCounts(test_p_sn[[var_e_i]], bx=limits)
            )
        )
    }

}

# get fill colours
library(RColorBrewer)
colours = brewer.pal(6,"Paired")
# swap 3,4 (greens) <> 5,6 (reds)
colours = c(colours[1:2], colours[5:6], colours[3:4])
names(colours) = levels(data_out$var_estim)


g = ggplot(data_out, aes(factor(bin), count, fill=var_estim)) +
    geom_bar(stat="identity", width=0.7, position=position_dodge(width=0.7)) +
    # geom_bar(stat="identity", position='dodge') +
    # scale_fill_brewer(palette = "Set1") +
    # scale_fill_brewer(palette = "Paired") +
    scale_fill_manual(values=colours) +
    geom_hline(yintercept=qbinom(0.005, Niter, 1/bins)) +
    geom_hline(yintercept=Niter/bins) +
    geom_hline(yintercept=qbinom(0.995, Niter, 1/bins)) +
    xlab("") +
    ylab("") +
    ggtitle(
        sprintf(
            "Histogram of p(loo<elpd_t), model: %s_%s_%s, p0=%g, %s",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
g = g + facet_wrap(~n_str, ncol=2)
# g = g + facet_grid(n_str ~ approx_dist)
print(g)

if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=20, height=14,
        filename = sprintf(
            "figs/unif_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
}
