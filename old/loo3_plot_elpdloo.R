
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(grid)
library(gridExtra)
library(ggExtra)

library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))


SAVE_FIGURE = TRUE
MEASURE = 5  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2


bins = 50

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
    loo=double()
)

# # =====================================================================
# for (beta0 in beta0s) {
for (MEASURE in c(1,4,5,6)) {

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

    loosums = colSums(loos)
    # loovars = colVars(loos)*n
    # looerror = tls-loosums

    res = rbind(
        res,
        data.frame(
            n_str=sprintf('n=%g', n),
            beta0_str=sprintf('beta0=%g', beta0),
            measure=loo_name,
            elpd=tls,
            loo=loosums
        )
    )

    # plot

    # dev.new()

    # first plot 2d hist
    g = ggplot()
    g = g + geom_bin2d(aes(x=loosums, y=tls), bins=bins, show.legend=FALSE)
    g = g + scale_fill_gradient(low="gray", high="black")
    g = g + geom_abline(intercept=0, slope=1, colour='#ce7e7e')
    if (MEASURE >= 4) {
        g = g + geom_hline(yintercept=0, colour='#ce7e7e')
        g = g + geom_vline(xintercept=0, colour='#ce7e7e')
    }
    # add mean
    g = g + geom_point(
        aes(x=mean(loosums), y=mean(tls)),
        # colour='olivedrab',
        # colour='royalblue',
        # colour='dodgerblue4',
        colour='orangered1',
        # colour='black',
        shape=10,
        size=6)

    # then plot scatter + ggMarginal hists
    g2 = ggplot()
    # first plot scatterplot and ggMarginal
    g2 = g2 + geom_point(aes(x=loosums, y=tls))
    g2 = g2 + geom_abline(intercept=0, slope=1, colour='#ce7e7e')
    if (MEASURE >= 4) {
        g2 = g2 + geom_hline(yintercept=0, colour='#ce7e7e')
        g2 = g2 + geom_vline(xintercept=0, colour='#ce7e7e')
    }
    # g2 = g2 + xlab(NULL) + ylab(NULL)
    g2 = g2 + xlab('elpd_loo') + ylab('elpd_target')
    g2_gtable = ggMarginal(g2, type='histogram', bins=bins)

    # convert g to gTable
    g_gtable = ggplotGrob(g)
    # replace the main grob in g2_gtable with the one in g_gtable
    g2_gtable$grobs[[6]] = g_gtable$grobs[[6]]

    print(g2_gtable)

    beta0_name = sub('\\.', '', sprintf('%g', beta0))  # remove .
    ggsave(
        plot=g2_gtable, width=5, height=4,
        filename = sprintf(
            "elpdloo_%s_%s_%s_%s_%s_%d.pdf",
            truedist, modeldist, priordist, beta0_name, loo_name, n
        )
    )

    # save figure
    if (SAVE_FIGURE) {
        beta0_name = sub('\\.', '', sprintf('%g', beta0))  # remove .
        ggsave(
            plot=g2_gtable, width=5, height=4,
            filename = sprintf(
                "figs/elpdloo_%s_%s_%s_%s_%s_%d.pdf",
                truedist, modeldist, priordist, beta0_name, loo_name, n
            )
        )
    }

}
cat('\ndone processing\n')

# } # for beta0 in ...
} # for MEASURE in ...


## plot all ------------------------------------------------------
