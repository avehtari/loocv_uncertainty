library(ggplot2)
library(matrixStats)
library(grid)
library(gridExtra)
library(reshape)
library(extraDistr)
source('plot_known_viol.R')

# library(loo)
# options(loo.cores=1)


# trials per run
Niter = 800
# num of test samples
Nt = 10000
# array of slope coefficients beta_{p+1}
betas = 2^seq(-6, 2)

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# possible parameters
Ps = c(0, 1, 2, 5, 10, 20)
#Ns = c(10, 20, 40, 60, 100, 140, 200)
Ns = c(10, 100)



# ==============================================================================
# ==============================================================================
## normal approximation accuracy with some quantiles as a function of n

p_i = 2
beta_i = 6

quantiles = c(0.05, 0.10, 0.90, 0.95)
viol_grid_na = 50
accs_q_na = array(0, c(2, length(Ns), length(quantiles), 5))
accs_m_na = array(0, c(2, length(Ns), length(quantiles)))
accs_viol_na = array(0, c(2, length(Ns), length(quantiles), 2, viol_grid_na))

for (ni in 1:length(Ns)) {
    n = Ns[ni]

    # load data in variable out
    load(sprintf('res_comp/%s_%s_%s_%d_%d_%d.RData',
        truedist, modeldist, priordist, Ps[p_i], n, beta_i))
    # populate local environment with the named stored variables in selected out
    list2env(out, envir=environment())
    rm(out)

    tls_diff = tls[,2] - tls[,1]
    loos_diff = loos[,,2] - loos[,,1]

    colvars_loos = colVars(loos_diff)
    tls_loos = tls_diff - colSums(loos_diff)

    t_basic = pnorm(
        tls_loos,
        mean = 0,
        sd = sqrt(n*colvars_loos)
    )
    t_x2 = pnorm(
        tls_loos,
        mean = 0,
        sd = sqrt(2*n*colvars_loos)
    )
    for (qi in 1:length(quantiles)) {
        # g2s
        c = sum(t_basic < quantiles[qi])
        b_alpha = c+1
        b_beta = Niter-c+1
        accs_m_na[1,ni,qi] = b_alpha / (b_alpha + b_beta)
        accs_q_na[1,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
        accs_viol_na[1,ni,qi,1,] = seq(
            qbeta(0.01, b_alpha, b_beta),
            qbeta(0.99, b_alpha, b_beta),
            length=viol_grid_na)
        accs_viol_na[1,ni,qi,2,] = dbeta(
            accs_viol_na[1,ni,qi,1,], b_alpha, b_beta)
        # x2
        c = sum(t_x2 < quantiles[qi])
        b_alpha = c+1
        b_beta = Niter-c+1
        accs_m_na[2,ni,qi] = b_alpha / (b_alpha + b_beta)
        accs_q_na[2,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
        accs_viol_na[2,ni,qi,1,] = seq(
            qbeta(0.01, b_alpha, b_beta),
            qbeta(0.99, b_alpha, b_beta),
            length=viol_grid_na)
        accs_viol_na[2,ni,qi,2,] = dbeta(
            accs_viol_na[2,ni,qi,1,], b_alpha, b_beta)

    }
}

# find limits
ylimits = array(0, c(length(quantiles), 2))
for (qi in 1:length(quantiles)) {
    if (quantiles[qi] < 0.5) {
        ylimits[qi,1] = 0
        ylimits[qi,2] = max(
            max(accs_viol_na[,,qi,1,]),
            #max(accs_viol_calbb[,,qi,1,]),
            quantiles[qi]
        )
    } else {
        ylimits[qi,2] = 1
        ylimits[qi,1] = min(
            min(accs_viol_na[,,qi,1,]),
            #min(accs_viol_calbb[,,qi,1,]),
            quantiles[qi]
        )
    }
}


gis_strs = c("basic", "x2")
# >>>>>> -------------------------------------
# for (gis in 1:length(gis_strs)) {
# >>>>>> -------------------------------------
gis = 1
# -------------------------------------

gs = vector('list', length(quantiles))
for (qi in 1:length(quantiles)) {
    g = plot_known_viol(
        Ns, accs_viol_na[gis,,qi,1,], accs_viol_na[gis,,qi,2,],
        # range1 = accs_q_na[gis,,qi,c(1,5)],  # 95 % interval
        # range2 = accs_q_na[gis,,qi,c(2,4)],  # 50 % interval
        line = accs_q_na[gis,,qi,3],  # median
        # point = accs_m_na[gis,,qi],  # mean
        colors='blue'
    )
    g = g +
        geom_hline(yintercept=quantiles[qi]) +
        coord_cartesian(ylim=ylimits[qi,]) +
        ggtitle(sprintf("quantile %g", quantiles[qi])) +
        xlab("") +
        ylab("")
    gs[[qi]] = g
}
# pdf(
#     file = sprintf(
#         "figs/p4_%s_%s_%s_%i--%i.pdf",
#         truedist, modeldist, priordist, Ps[p_i], gis
#     ),
#     width=15, height=5
# )
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top = sprintf(
        "norm approx accuracy with quantiles (model: %s_%s_%s_%i)\n%s",
        truedist, modeldist, priordist, Ps[p_i],
        gis_strs[gis]),
    bottom = "number of samples")
# dev.off()

# <<<<<< -------------------------------------
# }
# <<<<<< -------------------------------------

# clear
rm(gs)
