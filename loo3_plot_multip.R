
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
MEASURE = 4  # 1:M0, 2:M1, 3:M2, 4:M0-M1, 5:M0-M2, 6:M1-M2
FORCE_NONNEGATIVE_G3S = TRUE
FORCE_G3S_MAX_X2 = FALSE

Ns = c(10, 20, 50, 130, 250, 400)
p0 = 1

beta0 = 0.25
# beta0 = 0.5
# beta0 = 1
# beta0 = 2
# beta0 = 3
# beta0 = 4
beta0s = c(0.25, 0.5, 1, 2, 4)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'


# define min number of selected trials
# min_trials = 400
min_trials = 10

# thin the plot
plot_thin = 200
# plot_thin = FALSE

# prior for binomil model
beta_prior_alpha = 1
beta_prior_beta = 1


# # =====================================================================
# # These are for running them all (also uncimment `}`s at the bottom)
# for (beta0 in beta0s) {
# for (MEASURE in 4:6) {
# # =====================================================================


# =============================================================================
# For all n

# initialise empty result plot data
data_out = data.frame(
    n_str=character(),
    loosign=character(),
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
    load(sprintf('res_loo3/%s_%s_%s_%g_%d.RData',
        truedist, modeldist, priordist, beta0, n))
    # drop singleton dimensions
    for (name in names(out)) {
        out[[name]] = drop(out[[name]])
    }
    Niter = dim(out$loos)[2]



    # ==========================================================================
    # select measure

    if (MEASURE == 4) {
        # M0-M1
        loo_name = 'M0-M1'
        loo_pos_chosen = 'M0 chosen'
        loo_neg_chosen = 'M1 chosen'
        loos = out$loos[,,1] - out$loos[,,2]
        tls = out$tls[,1] - out$tls[,2]
        g2s = out$g2s_d01
        g3s = out$g3s_d01
        # g2s = out$g2s_nod_d01
        # g3s = out$g3s_nod_d01
    } else if (MEASURE == 5) {
        # M0-M2
        loo_name = 'M0-M2'
        loo_pos_chosen = 'M0 chosen'
        loo_neg_chosen = 'M2 chosen'
        loos = out$loos[,,1] - out$loos[,,3]
        tls = out$tls[,1] - out$tls[,3]
        g2s = out$g2s_d02
        g3s = out$g3s_d02
        # g2s = out$g2s_nod_d02
        # g3s = out$g3s_nod_d02
    } else if (MEASURE == 6) {
        # M1-M2
        loo_name = 'M1-M2'
        loo_pos_chosen = 'M1 chosen'
        loo_neg_chosen = 'M2 chosen'
        loos = out$loos[,,2] - out$loos[,,3]
        tls = out$tls[,2] - out$tls[,3]
        g2s = out$g2s_d12
        g3s = out$g3s_d12
        # g2s = out$g2s_nod_d12
        # g3s = out$g3s_nod_d12
    } else {
        stop('Invalid measure')
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
    loop_vars = colVars(loos)
    loop_sds = sqrt(loop_vars)
    loop_skews = apply(loos, 2, skewness)

    # loo sum mean estimate
    loo_means = loop_sums

    # loo sum var estimates
    loo_vars_1 = n*loop_vars
    loo_vars_2 = 2*(n-1)/(n-2)*loo_vars_1
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
    multiplier = vector("list", length(var_estims)*2)
    success_median = vector("list", length(var_estims)*2)
    success_025 = vector("list", length(var_estims)*2)
    success_975 = vector("list", length(var_estims)*2)
    for (loosign_i in 1:2) {
        loosign_idx = sign(loo_means) == c(-1,1)[loosign_i]
        loo_means_cur = loo_means[loosign_idx]
        tls_cur = tls[loosign_idx]
        len_cur = length(loo_means_cur)
        if (len_cur <= min_trials)
            next
        for (var_e_i in 1:length(var_estims)) {
            var_estims_cur = var_estims[[var_e_i]][loosign_idx]

            multips = abs(loo_means_cur)/sqrt(var_estims_cur)
            multips_order = order(multips)
            if (plot_thin) {
                # calculate with thinned multipliers
                thinned_x = seq(
                    from=0,
                    to=(multips[multips_order[len_cur-min_trials]] +
                        multips[multips_order[len_cur-min_trials+1]]
                    )/2,
                    length=plot_thin
                )
                success_median_i = array(NA, plot_thin)
                success_025_i = array(NA, plot_thin)
                success_975_i = array(NA, plot_thin)
                for (x_i in 1:plot_thin) {
                    selected = (
                        abs(loo_means_cur) >
                        thinned_x[x_i]*sqrt(var_estims_cur)
                    )
                    n_success = sum(selected)
                    y_success = sum(
                        sign(loo_means_cur[selected]) ==
                        sign(tls_cur[selected])
                    )
                    quantiles = qbeta(
                        c(0.025, 0.5, 0.975),
                        y_success+beta_prior_alpha,
                        n_success-y_success+beta_prior_beta
                    )
                    success_median_i[x_i] = quantiles[2]
                    success_025_i[x_i] = quantiles[1]
                    success_975_i[x_i] = quantiles[3]
                }
                list_idx = var_e_i+(loosign_i-1)*length(var_estims)
                multiplier[[list_idx]] = thinned_x
                success_median[[list_idx]] = success_median_i
                success_025[[list_idx]] = success_025_i
                success_975[[list_idx]] = success_975_i
            } else {
                # calculate with all multipliers
                success_median_i = array(NA, (len_cur-min_trials+1))
                success_025_i = array(NA, (len_cur-min_trials+1))
                success_975_i = array(NA, (len_cur-min_trials+1))
                for (x_i in 1:(len_cur-min_trials+1)) {
                    selected = multips_order[x_i:len_cur]
                    n_success = length(selected)
                    y_success = sum(
                        sign(loo_means_cur[selected]) ==
                        sign(tls_cur[selected])
                    )
                    quantiles = qbeta(
                        c(0.025, 0.5, 0.975),
                        y_success+beta_prior_alpha,
                        n_success-y_success+beta_prior_beta
                    )
                    success_median_i[x_i] = quantiles[2]
                    success_025_i[x_i] = quantiles[1]
                    success_975_i[x_i] = quantiles[3]
                }
                list_idx = var_e_i+(loosign_i-1)*length(var_estims)
                multiplier[[list_idx]] = c(
                    0.0, multips[multips_order[1:(len_cur-min_trials+1)]])
                success_median[[list_idx]] = c(
                    success_median_i[1], success_median_i)
                success_025[[list_idx]] = c(
                    success_025_i[1], success_025_i)
                success_975[[list_idx]] = c(
                    success_975_i[1], success_975_i)
            }
        }
    }

    # store data
    for (loosign_i in 1:2) {
        if (loosign_i == 1)
            loosign_str = loo_neg_chosen
        else
            loosign_str = loo_pos_chosen
        for (var_e_i in 1:length(var_estims)) {
            list_idx = var_e_i+(loosign_i-1)*length(var_estims)
            if (is.null(multiplier[[list_idx]])) next
            data_out = rbind(
                data_out,
                data.frame(
                    n_str=sprintf('n=%g', n),
                    loosign=loosign_str,
                    var_estim=var_estim_names[[var_e_i]],
                    multiplier=multiplier[[list_idx]],
                    success_median=success_median[[list_idx]],
                    success_025=success_025[[list_idx]],
                    success_975=success_975[[list_idx]]
                )
            )
        }
    }

}

# order loosign
data_out$loosign <- factor(
    data_out$loosign, levels = c(loo_pos_chosen, loo_neg_chosen))


# ==============================================================================
# Plot var multipliers

# get colours
colours = brewer.pal(6,"Paired")
# colours = c(colours[2], colours[6], colours[4])
colours = c(colours[2], colours[6])
names(colours) = levels(data_out$var_estim)

dev.new()
if (plot_thin) {
    g = ggplot(data_out) +
        facet_grid(n_str~loosign) +
        # facet_grid(n_str~loosign, scales='free_x') +
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
    g = ggplot(data_out) +
        facet_grid(n_str~loosign, scales='free_x') +
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

g = g + ylim(0,1)

# add labels
g = g + xlab("x") + ylab("Pr(sign(elpd)=sign(loo) | abs(loo)>x*sd(loo))")
g = g + ggtitle(sprintf(
    "model: %s_%s_%s, beta0=%g, %s",
    truedist, modeldist, priordist, beta0, loo_name
))

print(g)

if (SAVE_FIGURE) {
    ggsave(
        plot=g, width=16, height=12,
        filename = sprintf(
            "figs/multip_%s_%s_%s_%g_%s.pdf",
            truedist, modeldist, priordist, beta0, loo_name
        )
    )
}


# # =====================================================================
# # These are for running them all
# }
# }
# # =====================================================================
