
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(ggExtra)
library(gridExtra)
library(RColorBrewer)

library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans"))

library(sn)
# library(emg)

source('sn_fit.R')


SAVE_FIGURE = TRUE
MEASURE = 4  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2
FORCE_NONNEGATIVE_G3S = TRUE
FORCE_G3S_MAX_X2 = FALSE

# Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
Ns = c(10, 20, 50, 130, 250, 400)
p0 = 1

# beta0 = 0.25
# beta0 = 0.5
# beta0 = 1
# beta0 = 2
beta0 = 3
# beta0 = 4
# beta0 = 8
# beta0s = c(0.25, 0.5, 1, 2, 3, 4, 8)
beta0s = c(0.25, 1, 3, 8)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'



bins = 6


# =============00
for (MEASURE in c(1,4,5,6)) {
# =============00


# initialise empty result plot data
data_out = data.frame(
    n_str=character(),
    bin=character(),
    var_estim=character(),
    count=integer(),
    beta0_str=character()
)


# ==============================================================================
for (beta0 in beta0s) {
cat(sprintf('\nbeta0=%g,', beta0))

# ==============================================================================
# For all n

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
    Niter = dim(out$loos)[2]

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
    loo_vars_2 = 2*(n-1)/(n-2)*loo_vars_1
    loo_vars_4 = loo_vars_1 + (n^2)*g3s
    if (FORCE_G3S_MAX_X2) {
        g3s_too_big_idxs = loo_vars_4 > loo_vars_2
        loo_vars_4[g3s_too_big_idxs] = loo_vars_2[g3s_too_big_idxs]
    }
    # pack them
    var_estim_names = list('naive', 'improved', 'conservative')
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
    # rownames = sapply(
    #     1:bins, function(bin) sprintf('%.1f-%.1f', limits[bin], limits[bin+1]))
    rownames = sapply(
        1:bins, function(bin) sprintf('%d/%d-%d/%d', bin-1, bins, bin, bins))
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
                count=binCounts(test_p_n[[var_e_i]], bx=limits),
                beta0_str=sprintf('beta0=%g', beta0)
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
                count=binCounts(test_p_sn[[var_e_i]], bx=limits),
                beta0_str=sprintf('beta0=%g', beta0)
            )
        )
    }

}

} # for beta0 = ...

# get fill colours
colours = brewer.pal(6,"Paired")
# swap 3,4 (greens) <> 5,6 (reds)
colours = c(colours[1:2], colours[5:6], colours[3:4])
names(colours) = levels(data_out$var_estim)

if (TRUE) {
    # plot grouped

    dev.new()
    g = ggplot(data_out, aes(factor(bin), count, fill=var_estim)) +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/bins)) +
        geom_hline(yintercept=Niter/bins, colour='gray') +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/bins)) +
        geom_bar(stat="identity", width=0.7, position=position_dodge(width=0.7)) +
        # geom_bar(stat="identity", position='dodge') +
        # scale_fill_brewer(palette = "Set1") +
        # scale_fill_brewer(palette = "Paired") +
        scale_fill_manual(values=colours) +
        xlab('quantile') +
        ylab(NULL)

    g = g + facet_grid(n_str ~ beta0_str)
    g = g + theme(legend.position="bottom")
    g = g + theme(legend.title=element_blank())
    g = g + theme(axis.text.x = element_text(angle=45, hjust=1))

    print(g)

    if (SAVE_FIGURE) {
        ggsave(
            plot=g, width=12, height=16,
            filename = sprintf(
                "figs/dist_%s_%s_%s_%s.pdf",
                truedist, modeldist, priordist, loo_name
            )
        )
    }

} else {
    # save individual
    for (n in Ns) {
    for (beta0 in beta0s) {

    df_cur = data_out[data_out$n_str==sprintf('n=%d', n),]
    df_cur = df_cur[df_cur$beta0_str==sprintf('beta0=%g', beta0),]

    # dev.new()
    g = ggplot(df_cur, aes(factor(bin), count, fill=var_estim)) +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/bins)) +
        geom_hline(yintercept=Niter/bins, colour='gray') +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/bins)) +
        geom_bar(stat="identity", width=0.7, position=position_dodge(width=0.7)) +
        # geom_bar(stat="identity", position='dodge') +
        # scale_fill_brewer(palette = "Set1") +
        # scale_fill_brewer(palette = "Paired") +
        scale_fill_manual(values=colours) +
        xlab(NULL) +
        ylab(NULL)

    g = g + ylim(0,max(data_out[,'count']))

    g = g + theme(legend.position="none")

    # print(g)

    if (SAVE_FIGURE) {
        beta0_name = sub('\\.', '', sprintf('%g', beta0))  # remove .
        ggsave(
            plot=g, width=8, height=6,
            filename = sprintf(
                "figs/dist_%s_%s_%s_%s_%s_n%d.pdf",
                truedist, modeldist, priordist, beta0_name, loo_name, n
            )
        )
    }

    }}
}


# =============00
}
# =============00
