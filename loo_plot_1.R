
library(ggplot2)
library(matrixStats)
library(extraDistr)


SAVE_FIGURE = FALSE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 0

# ==============================================================================
# ratio peformance of variance terms

# settings for bayesian bootstrap
bbn = 2000
bba = 1

var_estim_names = list('naive', 'g2', 'g3', 'x2')

# output arrays
rat_s = array(0, c(length(var_estim_names), length(Ns)))
rat_bb_s = array(0, c(length(var_estim_names), length(Ns), bbn))

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

    # load data in variable out
    load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
        truedist, modeldist, priordist, p0, n))
    # drop singleton dimensions
    Niter = dim(out$loos)[2]
    for (name in names(out)) {
        out[[name]] = drop(out[[name]])
    }

    # ==================================================
    # select data

    # # Mi
    # m_i = 1
    # loo_name = sprintf('M%d', m_i)
    # loos = out$loos[,,m_i]
    # tls = out$tls[,m_i]
    # g2s = out$g2s[,m_i]
    # g3s = out$g3s[,m_i]
    # # g2s = out$g2s_nod[,m_i]
    # # g3s = out$g3s_nod[,m_i]

    # M1-M2
    loo_name = 'M1-M2'
    loos = out$loos[,,1] - out$loos[,,2]
    tls = out$tls[,1] - out$tls[,2]
    g2s = out$g2s_d
    g3s = out$g3s_d
    # g2s = out$g2s_nod_d
    # g3s = out$g3s_nod_d

    # ==================================================

    target_sd = sd(tls-colSums(loos))  # sd of loo error
    # target_sd = sd(colSums(loos))    # sd of loo

    loovars_naive = colVars(loos)*n

    # basic
    estims = loovars_naive
    rat_s[1,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[1,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # g2s
    estims = loovars_naive + n*g2s
    rat_s[2,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[2,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # g3s
    estims = loovars_naive + (n^2)*g3s
    rat_s[3,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[3,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # x2
    estims = 2*loovars_naive
    rat_s[4,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[4,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd
}
cat('\ndone processing\n')


## plot ------------------------------------------------------

## Plot all Ns
g = ggplot()
for (fi in 1:length(var_estim_names)) {
    for (ni in 1:length(Ns)) {
        g = g + geom_violin(
            data = data.frame(
                y=rat_bb_s[fi,ni,],
                x=Ns[ni],
                fill=var_estim_names[[fi]]
            ),
            aes(x=x, y=y, fill=fill),
            width = 8,
            alpha = 0.5
        )
    }
}
g = g +
    geom_hline(yintercept=1) +
    scale_fill_manual(values=c("green", "red", "blue", "gray")) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "hat{sd(loo_i)} / sd(elpd_i-loo_i), model: %s_%s_%s, p0=%g, %s",
        truedist, modeldist, priordist, p0, loo_name
    ))
g
# save figure
if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=8, height=5,
        filename = sprintf(
            "figs/ratio_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
}
