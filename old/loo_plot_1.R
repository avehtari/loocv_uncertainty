
library(matrixStats)
library(extraDistr)
library(ggplot2)
library(RColorBrewer)


SAVE_FIGURE = FALSE
COMPARISON = FALSE
FORCE_NONNEGATIVE_G3S = TRUE
FORCE_G3S_MAX_X2 = TRUE

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 0

# # =====================================================================
# # These are for running them all (also uncimment `}`s at the bottom)
# for (p0 in c(0,1)) {
# for (COMPARISON in c(FALSE, TRUE)) {
# for (temp_i in c(1,2)) {
# if (temp_i == 1) {
#     truedist = 'n'; modeldist = 'n'; priordist = 'n'
# } else {
#     truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# }
# # =====================================================================



# ==============================================================================
# ratio peformance of variance terms

# settings for bayesian bootstrap
bbn = 2000
bba = 1

var_estim_names = list('naive', 'corr', 'x2')

# output arrays
rat_s = array(NA, c(length(var_estim_names), length(Ns)))
rat_bb_s = array(NA, c(length(var_estim_names), length(Ns), bbn))
# rat_loo_target_s = array(NA, length(Ns))

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

    # ==================================================

    loosums = colSums(loos)
    loovars = colVars(loos)*n

    target_sd = sd(tls-loosums)  # sd of loo error

    # rat_loo_target_s[ni] = sd(loosums) / target_sd

    # basic
    estims = loovars
    rat_s[1,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[1,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # # g2s
    # estims = loovars + n*g2s
    # rat_s[2,ni] = sqrt(mean(estims)) / target_sd
    # rat_bb_s[2,ni,] = sqrt(
    #     rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # g3s
    estims = loovars + (n^2)*g3s
    if (FORCE_G3S_MAX_X2) {
        g3s_too_big_idxs = estims > 2*loovars
        estims[g3s_too_big_idxs] = 2*loovars[g3s_too_big_idxs]
    }
    rat_s[2,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[2,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # x2
    estims = 2*loovars
    rat_s[3,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[3,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd
}
cat('\ndone processing\n')


## plot ------------------------------------------------------

# get fill colours
colours = brewer.pal(6,"Paired")
colours = c(colours[2], colours[6], colours[4])
names(colours) = var_estim_names

## Plot all Ns
dev.new()
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
            alpha = 0.6
        )
    }
}
# g = g +
#     geom_point(aes(x=Ns, y=rat_loo_target_s[ni]))
g = g +
    geom_hline(yintercept=1) +
    # scale_fill_manual(values=c("green", "red", "blue", "gray")) +
    # scale_fill_brewer(palette = "Set1")
    scale_fill_manual(breaks=var_estim_names, values=colours) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "hat{sd(loo_i)} / sd(elpd_i-loo_i), model: %s_%s_%s, p0=%g, %s",
        truedist, modeldist, priordist, p0, loo_name
    ))
print(g)
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


# # =====================================================================
# # There are for running them all
# }
# }
# }
# # =====================================================================
