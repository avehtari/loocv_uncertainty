
library(matrixStats)
library(extraDistr)
library(moments)
library(ggplot2)
library(ggExtra)
library(gridExtra)
library(RColorBrewer)

library(sn)
# library(emg)

source('sn_fit.R')


SAVE_FIGURE = FALSE

FORCE_NONNEGATIVE_G3S = TRUE
FORCE_G3S_MAX_X2 = TRUE

Ns = c(10, 20, 50, 130, 250, 400)
p0 = 1

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'


# define min number of selected trials
# min_trials = 400
min_trials = 1

# thin the plot
plot_thin = 200
# plot_thin = FALSE

# prior for binomil model
beta_prior_alpha = 1
beta_prior_beta = 1


# # =====================================================================
# # These are for running them all (also uncimment `}`s at the bottom)
# for (p0 in c(0,1)) {
# for (temp_i in c(1,2)) {
# if (temp_i == 1) {
#     truedist = 'n'; modeldist = 'n'; priordist = 'n'
# } else {
#     truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# }
# # =====================================================================


# ==============================================================================
# For all n

# initialise empty result plot data
data_out = data.frame(
    n_str=character(),
    var_estim=character(),
    multiplier=double(),
    success_median=double(),
    success_025=double(),
    success_975=double()
)

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
    Niter = dim(out$loos)[2]



    # ==========================================================================
    # select measure

    # # M0-M1
    # loo_name = 'M0-M1'
    # loos = out$loos[,,1] - out$loos[,,2]
    # tls = out$tls[,1] - out$tls[,2]
    # g2s = out$g2s_d01
    # g3s = out$g3s_d01
    # # g2s = out$g2s_nod_d01
    # # g3s = out$g3s_nod_d01

    # M0-M2
    loo_name = 'M0-M2'
    loos = out$loos[,,1] - out$loos[,,3]
    tls = out$tls[,1] - out$tls[,3]
    g2s = out$g2s_d02
    g3s = out$g3s_d02
    # g2s = out$g2s_nod_d02
    # g3s = out$g3s_nod_d02

    # # M1-M2
    # loo_name = 'M1-M2'
    # loos = out$loos[,,2] - out$loos[,,3]
    # tls = out$tls[,2] - out$tls[,3]
    # g2s = out$g2s_d12
    # g3s = out$g3s_d12
    # # g2s = out$g2s_nod_d12
    # # g3s = out$g3s_nod_d12

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
    if (FORCE_G3S_MAX_X2) {
        g3s_too_big_idxs = loo_vars_4 > loo_vars_2
        loo_vars_4[g3s_too_big_idxs] = loo_vars_2[g3s_too_big_idxs]
    }
    # pack them
    # var_estim_names = list('naive', 'x2', 'g2', 'g3')
    # var_estims = list(loo_vars_1, loo_vars_2, loo_vars_3, loo_vars_4)
    var_estim_names = list('naive', 'corr')
    var_estims = list(loo_vars_1, loo_vars_4)


    # required uncertainty sigma multipliers
    multiplier = vector("list", length(var_estims))
    success_median = vector("list", length(var_estims))
    success_025 = vector("list", length(var_estims))
    success_975 = vector("list", length(var_estims))
    for (var_e_i in 1:length(var_estims)) {
        multips = abs(loo_means)/sqrt(var_estims[[var_e_i]])
        multips_order = order(multips)
        if (plot_thin) {
            # calculate with thinned multipliers
            thinned_x = seq(
                from=(multips[multips_order[1]])/2,
                to=(multips[multips_order[Niter-min_trials-1]] +
                    multips[multips_order[Niter-min_trials]]
                )/2,
                length=plot_thin
            )
            success_median_i = array(NA, plot_thin)
            success_025_i = array(NA, plot_thin)
            success_975_i = array(NA, plot_thin)
            for (x_i in 1:plot_thin) {
                selected = (
                    abs(loo_means) >
                    thinned_x[x_i]*sqrt(var_estims[[var_e_i]])
                )
                n_success = sum(selected)
                y_success = sum(
                    sign(loo_means[selected]) == sign(tls[selected]))
                quantiles = qbeta(
                    c(0.025, 0.5, 0.975),
                    y_success+beta_prior_alpha,
                    n_success-y_success+beta_prior_beta
                )
                success_median_i[x_i] = quantiles[2]
                success_025_i[x_i] = quantiles[1]
                success_975_i[x_i] = quantiles[3]
            }
            multiplier[[var_e_i]] = thinned_x
            success_median[[var_e_i]] = success_median_i
            success_025[[var_e_i]] = success_025_i
            success_975[[var_e_i]] = success_975_i
        } else {
            # calculate with all multipliers
            success_median_i = array(NA, (Niter-min_trials))
            success_025_i = array(NA, (Niter-min_trials))
            success_975_i = array(NA, (Niter-min_trials))
            for (x_i in 1:(Niter-min_trials)) {
                selected = multips_order[x_i:Niter]
                n_success = length(selected)
                y_success = sum(
                    sign(loo_means[selected]) == sign(tls[selected]))
                quantiles = qbeta(
                    c(0.025, 0.5, 0.975),
                    y_success+beta_prior_alpha,
                    n_success-y_success+beta_prior_beta
                )
                success_median_i[x_i] = quantiles[2]
                success_025_i[x_i] = quantiles[1]
                success_975_i[x_i] = quantiles[3]
            }
            multiplier[[var_e_i]] = c(
                0.0, multips[multips_order[1:(Niter-min_trials)]])
            success_median[[var_e_i]] = c(success_median_i[1], success_median_i)
            success_025[[var_e_i]] = c(success_025_i[1], success_025_i)
            success_975[[var_e_i]] = c(success_975_i[1], success_975_i)
        }
    }

    # store data
    for (var_e_i in 1:length(var_estims)) {
        data_out = rbind(
            data_out,
            data.frame(
                n_str=sprintf('n=%g', n),
                var_estim=var_estim_names[[var_e_i]],
                multiplier=multiplier[[var_e_i]],
                success_median=success_median[[var_e_i]],
                success_025=success_025[[var_e_i]],
                success_975=success_975[[var_e_i]]
            )
        )
    }

}


# ==============================================================================
# Plot var multipliers

# get colours
colours = brewer.pal(6,"Paired")
colours = c(colours[2], colours[6], colours[4])
names(colours) = levels(data_out$var_estim)

if (plot_thin) {
    dev.new()
    g = ggplot(
            data_out,
            aes(
                x=multiplier, y=success_median, colour=var_estim,
                ymin=success_025, ymax=success_975
            )
        ) +
        facet_wrap(~n_str, ncol=2) +
        geom_ribbon(
            aes(x=multiplier, ymin=success_025, ymax=success_975,
                fill=var_estim),
            alpha=0.4,
            colour=NA
        ) +
        geom_line(aes(x=multiplier, y=success_median, colour=var_estim)) +
        scale_colour_manual(values=colours) +
        scale_fill_manual(values=colours)
} else {
    dev.new()
    g = ggplot(data_out) +
        facet_wrap(~n_str, ncol=2) +
        geom_step(
            aes(x=multiplier, y=success_025, colour=var_estim),
            direction="vh", alpha=0.4
        ) +
        geom_step(
            aes(x=multiplier, y=success_975, colour=var_estim),
            direction="vh", alpha=0.4
        ) +
        geom_step(
            aes(x=multiplier, y=success_median, colour=var_estim),
            direction="vh"
        ) +
        scale_colour_manual(values=colours)
}

# add labels
g = g + xlab("x") + ylab("Pr(sign(elpd)=sign(loo) | abs(loo)>x*sd(loo))")
g = g + ggtitle(sprintf(
    "model: %s_%s_%s, p0=%g, %s",
    truedist, modeldist, priordist, p0, loo_name
))

print(g)

if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=16, height=12,
        filename = sprintf(
            "figs/multip_%s_%s_%s_%i_%s.pdf",
            truedist, modeldist, priordist, p0, loo_name
        )
    )
}


# # =====================================================================
# # There are for running them all
# }
# }
# # =====================================================================
