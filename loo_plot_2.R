
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(gridExtra)


SAVE_FIGURE = FALSE
COMPARISON = FALSE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 0

Niter = 2000

# ==============================================================================
# ratio peformance of variance terms

var_estim_names = list('naive', 'g2', 'g3', 'x2')

# output arrays
skew_loo_i = array(NA, c(Niter, length(Ns)))
skew_loosum = array(NA, length(Ns))
skew_elpd = array(NA, length(Ns))
skew_looerror = array(NA, length(Ns))
kurt_loo_i = array(NA, c(Niter, length(Ns)))
kurt_loosum = array(NA, length(Ns))
kurt_elpd = array(NA, length(Ns))
kurt_looerror = array(NA, length(Ns))
rat_s = array(NA, c(Niter, length(Ns)))

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

    # ==========================================================================
    # select measure

    if (COMPARISON) {
        # M1-M2
        loo_name = 'M1-M2'
        loos = out$loos[,,1] - out$loos[,,2]
        tls = out$tls[,1] - out$tls[,2]
    } else {
        # Mi
        m_i = 1
        loo_name = sprintf('M%d', m_i)
        loos = out$loos[,,m_i]
        tls = out$tls[,m_i]
    }

    # ==================================================

    loosums = colSums(loos)
    loovars = colVars(loos)*n

    target_sd = sd(tls-loosums)  # sd of loo error

    skew_loo_i[,ni] = apply(loos, 2, skewness)
    skew_loosum[ni] = skewness(loosums)
    skew_elpd[ni] = skewness(tls)
    skew_looerror[ni] = skewness(tls-loosums)
    kurt_loo_i[,ni] = apply(loos, 2, kurtosis)
    kurt_loosum[ni] = kurtosis(loosums)
    kurt_elpd[ni] = kurtosis(tls)
    kurt_looerror[ni] = kurtosis(tls-loosums)

    rat_s[,ni] = sqrt(loovars) / target_sd

}
cat('\ndone processing\n')


## plot ------------------------------------------------------

## skewness
dev.new()
g = ggplot()
for (ni in 1:length(Ns)) {
    g = g + geom_violin(
        data = data.frame(
            y=skew_loo_i[,ni],
            x=Ns[ni],
            fill='loo_i'
        ),
        aes(x=x, y=y, fill=fill),
        width = 8,
        alpha = 0.5
    )
}
g = g +
    geom_line(aes(x=Ns, y=skew_loosum, colour='loosum')) +
    geom_line(aes(x=Ns, y=skew_elpd, colour='elpd')) +
    geom_line(aes(x=Ns, y=skew_looerror, colour='loosum-elpd')) +
    scale_colour_manual(
        "",
        breaks = c("loo_i", "loosum", "elpd", "loosum-elpd"),
        values = c("red", "green", "blue")
    )
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


## kurtosi
dev.new()
g = ggplot()
for (ni in 1:length(Ns)) {
    g = g + geom_violin(
        data = data.frame(
            y=kurt_loo_i[,ni],
            x=Ns[ni],
            fill='loo_i'
        ),
        aes(x=x, y=y, fill=fill),
        width = 8,
        alpha = 0.5
    )
}
g = g +
    geom_line(aes(x=Ns, y=kurt_loosum, colour='loosum')) +
    geom_line(aes(x=Ns, y=kurt_elpd, colour='elpd')) +
    geom_line(aes(x=Ns, y=kurt_looerror, colour='loosum-elpd')) +
    scale_colour_manual(
        "",
        breaks = c("loo_i", "loosum", "elpd", "loosum-elpd"),
        values = c("red", "green", "blue")
    )
g = g +
    xlab("number of samples") +
    ggtitle(sprintf(
        "kurtosis, model: %s_%s_%s, p0=%g, %s",
        truedist, modeldist, priordist, p0, loo_name
    ))
print(g)
if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=8, height=5,
        filename = sprintf(
            "figs/kurt_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
}


## pairwise skewness
g_s = vector("list", length(Ns))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    g = ggplot()
    g = g +
        geom_point(
            data=data.frame(
                x=skew_loo_i[,ni],
                y=rat_s[,ni]
            ),
            aes(x=x, y=y),
            alpha=0.2
        )
    g = g +
        geom_hline(yintercept=1) +
        xlab("") +
        ylab("") +
        ggtitle(sprintf("n=%g", n))
    g_s[[ni]] = g
}
# add labels
g_s[[1]] = g_s[[1]] +
    ylab("hat{sd(loo_i)} / sd(elpd_i-loo_i)")
g_s[[1+ceiling(length(Ns)/2)]] = g_s[[1+ceiling(length(Ns)/2)]] +
    ylab("hat{sd(loo_i)} / sd(elpd_i-loo_i)")
for (ni in (1+ceiling(length(Ns)/2)):length(Ns)) {
    g_s[[ni]] = g_s[[ni]] + xlab("skewness")
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/skewpair_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "model: %s_%s_%s, p0=%g, %s",
    truedist, modeldist, priordist, p0, loo_name
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()



## pairwise kurtosis
g_s = vector("list", length(Ns))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    g = ggplot()
    g = g +
        geom_point(
            data=data.frame(
                x=kurt_loo_i[,ni],
                y=rat_s[,ni]
            ),
            aes(x=x, y=y),
            alpha=0.2
        )
    g = g +
        geom_hline(yintercept=1) +
        xlab("") +
        ylab("") +
        ggtitle(sprintf("n=%g", n))
    g_s[[ni]] = g
}
# add labels
g_s[[1]] = g_s[[1]] +
    ylab("hat{sd(loo_i)} / sd(elpd_i-loo_i)")
g_s[[1+ceiling(length(Ns)/2)]] = g_s[[1+ceiling(length(Ns)/2)]] +
    ylab("hat{sd(loo_i)} / sd(elpd_i-loo_i)")
for (ni in (1+ceiling(length(Ns)/2)):length(Ns)) {
    g_s[[ni]] = g_s[[ni]] + xlab("kurtosis")
}
# plot (and save)
if (SAVE_FIGURE) {
    pdf(
        file=sprintf(
            "figs/kurtpair_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        ),
        width=12,
        height=8
    )
} else {
    dev.new()
}
top_str = sprintf(
    "model: %s_%s_%s, p0=%g, %s",
    truedist, modeldist, priordist, p0, loo_name
)
do.call("grid.arrange", c(g_s, nrow=2, top=top_str))
if (SAVE_FIGURE) dev.off()
