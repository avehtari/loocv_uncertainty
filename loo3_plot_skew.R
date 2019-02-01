
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(gridExtra)

source('sn_fit.R')


SAVE_FIGURE = TRUE
MEASURE = 5  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2

Ns = c(10, 20, 50, 130, 250, 400)
p0 = 1

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

Niter = 2000

# bayes bootstrap samples
bbn = 2000
bb_alpha = 0.5

# ==============================================================================

# output arrays
skew_loo_i = array(NA, c(Niter, length(Ns)))
skew_loosum_bb = array(NA, c(bbn, length(Ns)))
skew_elpd_bb = array(NA, c(bbn, length(Ns)))
skew_looerror_bb = array(NA, c(bbn, length(Ns)))
skew_loosum = array(NA, length(Ns))
skew_elpd = array(NA, length(Ns))
skew_looerror = array(NA, length(Ns))
rat_s = array(NA, c(Niter, length(Ns)))

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

    # load data in variable out
    load(sprintf('res_loo3/%s_%s_%s_%g_%g.RData',
        truedist, modeldist, priordist, p0, n))
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

    loosums = colSums(loos)
    loovars = colVars(loos)*n
    looerror = tls-loosums

    target_sd = sd(looerror)  # sd of loo error

    skew_loo_i[,ni] = apply(loos, 2, skewness)
    skew_loosum[ni] = skewness(loosums)
    skew_elpd[ni] = skewness(tls)
    skew_looerror[ni] = skewness(looerror)

    # bb
    skew_loosum_bb[,ni] = apply(
        rdirichlet(bbn, rep(bb_alpha, length(loosums))), 1,
        function(w) skewness_weighted(loosums, w)
    )
    skew_elpd_bb[,ni] = apply(
        rdirichlet(bbn, rep(bb_alpha, length(tls))), 1,
        function(w) skewness_weighted(tls, w)
    )
    skew_looerror_bb[,ni] = apply(
        rdirichlet(bbn, rep(bb_alpha, length(looerror))), 1,
        function(w) skewness_weighted(looerror, w)
    )


    rat_s[,ni] = sqrt(loovars) / target_sd

}
cat('\ndone processing\n')


## plot ------------------------------------------------------

dev.new()
g = ggplot()
for (ni in 1:length(Ns)) {
    g = g + geom_violin(
        data = data.frame(
            y=skew_loo_i[,ni],
            x=Ns[ni],
            fill='loo_i'
        ),
        aes(x=x, y=y),
        width = 16,
        alpha = 0.5
    )
    g = g + geom_violin(
        data = data.frame(
            y=skew_loosum_bb[,ni],
            x=Ns[ni],
            fill='loosum'
        ),
        aes(x=x, y=y),
        width = 16,
        alpha = 0.5
    )
    g = g + geom_violin(
        data = data.frame(
            y=skew_elpd_bb[,ni],
            x=Ns[ni],
            fill='elpd'
        ),
        aes(x=x, y=y),
        width = 16,
        alpha = 0.5
    )
    g = g + geom_violin(
        data = data.frame(
            y=skew_looerror_bb[,ni],
            x=Ns[ni],
            fill='loosum-elpd'
        ),
        aes(x=x, y=y),
        width = 16,
        alpha = 0.5
    )
}
g = g + facet_wrap(~fill, nrow=4)
g = g +
    xlab("number of samples") +
    ggtitle(sprintf(
        "skewness, model: %s_%s_%s, p0=%g, %s",
        truedist, modeldist, priordist, p0, loo_name
    ))
print(g)

# save figure
if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=8, height=5,
        filename = sprintf(
            "figs/skew_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
}
