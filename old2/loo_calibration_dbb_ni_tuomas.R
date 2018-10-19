
library(ggplot2)
library(matrixStats)
library(grid)
library(gridExtra)
library(reshape)
library(extraDistr)
library(bayesboot)
source('gg_qq.R')
source('plot_known_viol.R')

# in addition to loo_calibration_tuomas.R
library(loo)
options(loo.cores=1)



# possible parameters
dists = list(
    c('n', 'n', 'n'),
    c('t4', 'tnu', 'n'),
    c('b', 'b', 'n'),
    c('n', 'tnu', 'n'),
    c('t4', 'n', 'n')
)
Ns<-c(10, 20, 40, 60, 100, 140, 200, 260)
Ps<-c(1, 2, 5, 10)

Niter = 2000

# select params (cmd arg)
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi == 0) {
    p_i = 1
    truedist = 'n'; modeldist = 'n'; priordist = 'n'
} else if (jobi == 1) {
    p_i = 2
    truedist = 'n'; modeldist = 'n'; priordist = 'n'
} else if (jobi == 2) {
    p_i = 3
    truedist = 'n'; modeldist = 'n'; priordist = 'n'
} else if (jobi == 3) {
    p_i = 4
    truedist = 'n'; modeldist = 'n'; priordist = 'n'
} else {
    stop(sprintf("Invalid jobi"))
}


# ====================
# output allocations

bbsamples = 2000

alphas = c(function(n) 1, function(n) 1 - 6/n)
alphas_str = c("1", "1 - 6/n")

# dws = array(vector('list'), c(length(alphas), length(Ns)))
# qws = array(vector('list'), c(length(alphas), length(Ns)))

qpp = array(0, c(length(alphas), length(Ns), Niter))

quantiles = c(0.05, 0.10, 0.90, 0.95)
viol_grid_caldbb = 50
accs_q_caldbb = array(NaN, c(length(alphas), length(Ns), length(quantiles), 5))
accs_m_caldbb = array(NaN, c(length(alphas), length(Ns), length(quantiles)))
accs_viol_caldbb = array(
    NaN, c(length(alphas), length(Ns), length(quantiles), 2, viol_grid_caldbb))

# ====================




for (ni in 1:length(Ns)) {
    n = Ns[ni]
    print(sprintf('n=%d', n))
    # load data in variable out
    load(sprintf('res/%s_%s_%s_%d_%d.RData',
        truedist, modeldist, priordist, Ps[p_i], n))
    # modify 1d matrices into vectors in out
    out$peff = out$peff[1,]
    out$tls = out$tls[1,]
    out$ets = out$ets[1,]
    out$es = out$es[1,]
    out$tes = out$tes[1,]
    out$bs = out$bs[1,]
    out$gs = out$gs[1,]
    out$gms = out$gms[1,]
    out$g2s = out$g2s[1,]
    out$gm2s = out$gm2s[1,]
    out$g3s = out$g3s[1,]

    # populate local environment with the named stored variables in selected out
    list2env(out, envir=environment())


    # # Get bb samples
    # for (alpha_i in 1:length(alphas)) {
    #     alpha = alphas[[alpha_i]](n)
    #
    #     dw = rdirichlet(bbsamples, rep(alpha, n))
    #     dws[[alpha_i, ni]] = dw
    #
    #     qws[[alpha_i, ni]] = (dw%*%loos)*n
    # }


    # preallocate temp arrays
    qww = numeric(bbsamples)

    for (i1 in 1:Niter) {

        qq = tqq[,,i1]
        qqw = qq
        for (cvi in 1:n) {
            qqw[-cvi,cvi] = (qq[-cvi,cvi]-mean(qq[-cvi,cvi])) + qq[cvi,cvi]
        }

        for (alpha_i in 1:length(alphas)) {
            alpha = alphas[[alpha_i]](n)

            for (i2 in 1:bbsamples) {
                qww[i2] = sum(
                    t(t(rdirichlet(n, rep(alpha, n))) *
                      as.vector(rdirichlet(1, rep(alpha, n))))
                    * qqw) * n
            }
            qpp[alpha_i, ni, i1] = mean(qww <= tls[i1])

        }
    }


    # calc viols for some test quantiles
    for (alpha_i in 1:length(alphas)) {

        for (qi in 1:length(quantiles)) {

            c = sum(qpp[alpha_i, ni, ] < quantiles[qi])
            b_alpha = c+1
            b_beta = Niter-c+1
            accs_m_caldbb[alpha_i,ni,qi] = b_alpha / (b_alpha + b_beta)
            accs_q_caldbb[alpha_i,ni,qi,] = qbeta(
                c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
            accs_viol_caldbb[alpha_i,ni,qi,1,] = seq(
                qbeta(0.01, b_alpha, b_beta),
                qbeta(0.99, b_alpha, b_beta),
                length=viol_grid_caldbb)
            accs_viol_caldbb[alpha_i,ni,qi,2,] = dbeta(
                accs_viol_caldbb[alpha_i,ni,qi,1,], b_alpha, b_beta)

        }

    }

    # free memory
    rm(
        out,
        ltrs,
        loos,
        looks,
        peff,
        pks,
        tls,
        ets,
        es,
        tes,
        mutrs,
        muloos,
        mulooks,
        bs,
        gs,
        gms,
        g2s,
        gm2s,
        g3s,
        tqq
    )
    gc()

}





# find limits
ylimits = array(0, c(length(quantiles), 2))
for (qi in 1:length(quantiles)) {
    if (quantiles[qi] < 0.5) {
        ylimits[qi,1] = 0
        ylimits[qi,2] = max(
            # max(accs_viol_na[,,qi,1,]),
            # max(accs_viol_calbb[,,qi,1,]),
            max(accs_viol_caldbb[,,qi,1,]),
            quantiles[qi]
        )
    } else {
        ylimits[qi,2] = 1
        ylimits[qi,1] = min(
            # min(accs_viol_na[,,qi,1,]),
            # min(accs_viol_calbb[,,qi,1,]),
            min(accs_viol_caldbb[,,qi,1,]),
            quantiles[qi]
        )
    }
}


for (alpha_i in 1:length(alphas)) {   # =======================================

# alpha_i = 1


# plot
gs = vector('list', length(quantiles))
for (qi in 1:length(quantiles)) {
    g = plot_known_viol(
        Ns, accs_viol_caldbb[alpha_i,,qi,1,], accs_viol_caldbb[alpha_i,,qi,2,],
        # range1 = accs_q_caldbb[alpha_i,,qi,c(1,5)],  # 95 % interval
        # range2 = accs_q_caldbb[alpha_i,,qi,c(2,4)],  # 50 % interval
        line = accs_q_caldbb[alpha_i,,qi,3],  # median
        # point = accs_m_caldbb[alpha_i,,qi],  # mean
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
pdf(
    file = sprintf(
        "figs/p6_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    ),
    width=15, height=5
)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top = sprintf(
        "Calibrated double BB accuracy (model: %s_%s_%s_%i)\nalpha=%s, bbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        alphas_str[alpha_i],
        bbsamples
    ),
    bottom = "number of samples")
dev.off()


} # =======================================
