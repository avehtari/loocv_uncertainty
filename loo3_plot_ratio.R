
library(matrixStats)
library(extraDistr)
library(ggplot2)
library(RColorBrewer)

library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))

SAVE_FIGURE = TRUE
MEASURE = 1  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2
FORCE_NONNEGATIVE_G3S = TRUE
FORCE_G3S_MAX_X2 = FALSE

Ns = c(10, 20, 50, 130, 250, 400)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 1

# beta0 = 0.25
# beta0 = 0.5
# beta0 = 1
# beta0 = 2
beta0 = 3
# beta0 = 4
beta0s = c(0.25, 0.5, 1, 2, 4)

# # =====================================================================
# # These are for running them all (also uncimment `}`s at the bottom)
for (beta0 in beta0s) {
for (MEASURE in c(1,4,5)) {
# # =====================================================================



# ==============================================================================
# ratio peformance of variance terms

# settings for bayesian bootstrap
bbn = 2000
bba = 1

var_estim_names = list('naive', 'improved', 'conservative')

# output arrays
rat_s = array(NA, c(length(var_estim_names), length(Ns)))
rat_bb_s = array(NA, c(length(var_estim_names), length(Ns), bbn))
# rat_loo_target_s = array(NA, length(Ns))

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

    # load data in variable out
    load(sprintf('res_loo3/%s_%s_%s_%g_%d.RData',
        truedist, modeldist, priordist, beta0, n))
    # drop singleton dimensions
    Niter = dim(out$loos)[2]
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
        # g2s = out$g2s[,m_i]
        # g3s = out$g3s[,m_i]
        g2s = out$g2s_nod[,m_i]
        g3s = out$g3s_nod[,m_i]
    } else if (MEASURE == 4) {
        # M0-M1
        loo_name = 'M0-M1'
        loos = out$loos[,,1] - out$loos[,,2]
        tls = out$tls[,1] - out$tls[,2]
        # g2s = out$g2s_d01
        # g3s = out$g3s_d01
        g2s = out$g2s_nod_d01
        g3s = out$g3s_nod_d01
    } else if (MEASURE == 5) {
        # M0-M2
        loo_name = 'M0-M2'
        loos = out$loos[,,1] - out$loos[,,3]
        tls = out$tls[,1] - out$tls[,3]
        # g2s = out$g2s_d02
        # g3s = out$g3s_d02
        g2s = out$g2s_nod_d02
        g3s = out$g3s_nod_d02
    } else {
        # M1-M2
        loo_name = 'M1-M2'
        loos = out$loos[,,2] - out$loos[,,3]
        tls = out$tls[,2] - out$tls[,3]
        # g2s = out$g2s_d12
        # g3s = out$g3s_d12
        g2s = out$g2s_nod_d12
        g3s = out$g3s_nod_d12
    }

    # fix ddof error in loo3_fun g3s
    num_of_pairs = (n^2-n)/2
    g3s = g3s*((num_of_pairs-1)*(2/(n*(n-2))))

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
        g3s_too_big_idxs = estims > 2*(n-1)/(n-2)*loovars
        estims[g3s_too_big_idxs] = 2*(n-1)/(n-2)*loovars[g3s_too_big_idxs]
    }
    rat_s[2,ni] = sqrt(mean(estims)) / target_sd
    rat_bb_s[2,ni,] = sqrt(
        rdirichlet(bbn, rep(bba, length(estims))) %*% estims) / target_sd

    # x2
    estims = 2*(n-1)/(n-2)*loovars
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
g = g + scale_x_sqrt(breaks=Ns)
g = g + geom_hline(yintercept=1, colour='gray')
for (fi in 1:length(var_estim_names)) {
    for (ni in 1:length(Ns)) {
        g = g + geom_violin(
            data = data.frame(
                y=rat_bb_s[fi,ni,],
                x=Ns[ni],
                fill=var_estim_names[[fi]]
            ),
            aes(x=x, y=y, fill=fill),
            width = 2,
            alpha = 0.6
        )
    }
}
# g = g +
#     geom_point(aes(x=Ns, y=rat_loo_target_s[ni]))
g = g +
    # scale_fill_manual(values=c("green", "red", "blue", "gray")) +
    # scale_fill_brewer(palette = "Set1")
    scale_fill_manual(breaks=var_estim_names, values=colours) +
    xlab("data size") +
    ylab(NULL)

# g = g +
#     ggtitle(sprintf(
#         "hat{sd(loo_i)} / sd(elpd_i-loo_i), model: %s_%s_%s, beta0=%g, %s",
#         truedist, modeldist, priordist, beta0, loo_name
#     ))

if (MEASURE == 1) {
    g = g + ylab("sqrt{hat{sigma}^2} / sd(elpd_i-loo_i)")
} else {
    g = g + ylab(NULL)
}

if (MEASURE != 5) {
    g = g + theme(legend.position="none")
    fig_width = 4
} else {
    fig_width = 5.5
}


print(g)
# save figure
if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=fig_width, height=5,
        filename = sprintf(
            "figs/ratio_%s_%s_%s_%g_%s.pdf",
            truedist, modeldist, priordist, beta0, loo_name
        )
    )
}


# # =====================================================================
# # There are for running them all
}
}
# # =====================================================================
