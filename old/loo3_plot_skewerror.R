
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(gridExtra)

library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))

source('sn_fit.R')


SAVE_FIGURE = TRUE
MEASURE = 5  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2

Ns = c(10, 20, 50, 130, 250, 400)
p0 = 1

# beta0 = 0.25
# beta0 = 0.5
# beta0 = 1
# beta0 = 2
beta0 = 3
# beta0 = 4
# beta0 = 8
beta0s = c(0.25, 0.5, 1, 2, 4, 8)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

Niter = 2000

# bayes bootstrap samples
bbn = 2000
bb_alpha = 1


# output arrays
res = data.frame(
    measure=character(),
    n_str=character(),
    beta0_str=character(),
    elpd=double(),
    loo=double(),
    skew=double(),
    looerror=double(),
    correct=logical()
)

# # =====================================================================
# # These are for running them all (also uncimment `}`s at the bottom)
# for (beta0 in beta0s) {
for (MEASURE in c(1,4,5)) {
# # =====================================================================


# ==============================================================================

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

    # load data in variable out
    load(sprintf('res_loo3/%s_%s_%s_%g_%d.RData',
        truedist, modeldist, priordist, beta0, n))
    # drop singleton dimensions
    for (name in names(out)) {
        out[[name]] = drop(out[[name]])
    }

    # ==========================================================================
    # select measure

    if (MEASURE <= 3) {
        m_i = MEASURE
        loo_name = sprintf('M%d', m_i-1)
        loos = out$loos[,,m_i]
        tls = out$tls[,m_i]
    } else if (MEASURE == 4) {
        # M0-M1
        loo_name = 'M0-M1'
        loos = out$loos[,,1] - out$loos[,,2]
        tls = out$tls[,1] - out$tls[,2]
    } else if (MEASURE == 5) {
        # M0-M2
        loo_name = 'M0-M2'
        loos = out$loos[,,1] - out$loos[,,3]
        tls = out$tls[,1] - out$tls[,3]
    } else {
        # M1-M2
        loo_name = 'M1-M2'
        loos = out$loos[,,2] - out$loos[,,3]
        tls = out$tls[,2] - out$tls[,3]
    }

    # ==================================================

    loo = colSums(loos)
    skew = apply(loos, 2, skewness)
    looerror = tls-loo
    correct = sign(loo) == sign(tls)

    res = rbind(
        res,
        data.frame(
            n_str=sprintf('n=%g', n),
            beta0_str=sprintf('beta0=%g', beta0),
            measure=loo_name,
            elpd=tls,
            loo=loo,
            skew=skew,
            looerror=looerror,
            correct=correct
        )
    )
}
cat('\ndone processing\n')

} # for MEASURE in ...

rm(loo,skew,looerror,correct)


## plot ------------------------------------------------------

dev.new()
g = ggplot(res)
g = g + geom_point(aes(x=skew, y=looerror))
g = g + facet_grid(n_str~measure, scales='free_x')
g = g+ geom_hline(yintercept=0, colour='orangered1')
g = g + xlab('skewness') + ylab('loo-elpd')
print(g)

dev.new()
g = ggplot(res[res$measure='M0-M2'])
g = g + geom_histogram(aes(x=skew, fill=correct))
g = g + facet_grid(n_str~measure, scales='free_x')
g = g + xlab('skewness') + ylab(NULL)
print(g)


# ----------------------------------------------------------
# calc bin success probabilities

beta_prior_alpha=1
beta_prior_beta=1

# output arrays
res_prob = data.frame(
    measure=character(),
    n_str=character(),
    skew_bin_min=double(),
    skew_bin=double(),
    skew_bin_max=double(),
    prob_025=double(),
    prob_500=double(),
    prob_975=double()
)
for (measure in c('M0-M1', 'M0-M2')) {
    res_m = res[res$measure==measure,]
    bins = 12
    for (n_i in 1:length(Ns)) {
        n = Ns[n_i]
        n_str=sprintf('n=%g', n)
        res_m_n = res_m[res_m$n_str==n_str,]
        bin_edges = seq(
            from=min(res_m_n$skew), to=max(res_m_n$skew)+0.00001, length=bins+1)
        for (bin in 1:bins) {
            res_cur = res_m_n[res_m_n$skew<bin_edges[bin+1],]
            res_cur = res_cur[res_cur$skew>=bin_edges[bin],]
            y_success = sum(res_cur$correct)
            n_success = length(res_cur$correct)
            quantiles = qbeta(
                c(0.025, 0.5, 0.975),
                y_success+beta_prior_alpha,
                n_success-y_success+beta_prior_beta
            )
            res_prob = rbind(
                res_prob,
                data.frame(
                    measure=measure,
                    n_str=n_str,
                    skew_bin_min=bin_edges[bin],
                    skew_bin=(bin_edges[bin]+bin_edges[bin+1])/2,
                    skew_bin_max=bin_edges[bin+1],
                    prob_025=quantiles[1],
                    prob_500=quantiles[2],
                    prob_975=quantiles[3]
                )
            )

        }
    }
}

dev.new()
g = ggplot(res_prob)
g = g + geom_vline(xintercept=0, colour='gray')
g = g + geom_line(aes(x=skew_bin, y=prob_500))
g = g + geom_line(aes(x=skew_bin, y=prob_025), colour='tomato1')
g = g + geom_line(aes(x=skew_bin, y=prob_975), colour='tomato1')
g = g + facet_grid(n_str~measure, scales='free_x')
g = g + xlab('skewness') + ylab('Pr(sign(loo)=sign(elpd))')
print(g)

# save figure
if (SAVE_FIGURE) {
    beta0_name = sub('\\.', '', sprintf('%g', beta0))  # remove .
    ggsave(
        plot=g, width=7, height=8,
        filename = sprintf(
            "figs/skewerror_%s_%s_%s_%s.pdf",
            truedist, modeldist, priordist, beta0_name
        )
    )
}
