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


# possible parameters
dists = list(
    c('n', 'n', 'n'),
    c('t4', 'tnu', 'n'),
    c('b', 'b', 'n'),
    c('n', 'tnu', 'n'),
    c('t4', 'n', 'n')
)
Ns<-c(10, 20, 40, 60, 100, 140, 200, 260)
Ps<-c(0, 1, 2, 5, 10, 20)

n_included = c(1,5)
p_included = c(1,2,5)

truedist = 'n'; modeldist = 'n'; priordist = 'n'


# compute
comp_plot_count = array(NaN, c(length(betas), length(Ns), length(Ps)))
comp_plot_count_t = array(NaN, c(length(betas), length(Ns), length(Ps)))
for (p_i in p_included) {
    for (ni in n_included) {
        n = Ns[ni]
        for (beta_i in 1:length(betas)) {
            beta = betas[beta_i]
            # load data in variable out
            load(sprintf('res_comp/%s_%s_%s_%d_%d_%d.RData',
                truedist, modeldist, priordist, Ps[p_i], n, beta_i))
            loos = out$loos
            tls = out$tls
            comp_plot_count[beta_i, ni, p_i] = (
                sum(colSums(loos[,,2]) > colSums(loos[,,1])))
            comp_plot_count_t[beta_i, ni, p_i] = (
                sum(tls[,2] > tls[,1]))
        }
        rm(loos, tls)
    }
}

# ==========================================
# ns
p_i = 1
g = ggplot()
strNs = sapply(Ns, function(n) sprintf("%d", n))
for (ni in n_included) {
    g = g + geom_line(
        data = data.frame(
            y = comp_plot_count[,ni,p_i] / Niter,
            x = betas,
            group = strNs[ni],
            colour = strNs[ni]
        ),
        aes(x=x, y=y, group=group, colour=colour)
    )
}
g = g +
    scale_colour_discrete(name="n")
g = g +
    scale_x_continuous(trans="log2") +
    xlab("beta") +
    ylab(NULL) +
    coord_cartesian(ylim=c(0,1)) +
    ggtitle(sprintf(
        "proportion of looM1 > looM0, %s_%s_%s, p=%d",
        truedist, modeldist, priordist, Ps[p_i]
    ))


# ==========================================
# ps
ni = 1
g = ggplot()
strPs = sapply(Ps, function(p) sprintf("%d", p))
for (p_i in p_included) {
    g = g + geom_line(
        data = data.frame(
            y = comp_plot_count[,ni,p_i] / Niter,
            x = betas,
            group = strPs[p_i],
            colour = strPs[p_i]
        ),
        aes(x=x, y=y, group=group, colour=colour)
    )
}
g = g +
    scale_colour_discrete(name="p")
g = g +
    scale_x_continuous(trans="log2") +
    xlab("beta") +
    ylab(NULL) +
    coord_cartesian(ylim=c(0,1)) +
    ggtitle(sprintf(
        "proportion of looM1 > looM0, %s_%s_%s, n=%d",
        truedist, modeldist, priordist, Ns[ni]
    ))

# ==========================================
# mean errorlines
ni = 1
p_i = 1
quantiles = array(NaN, c(length(betas), 3))
for (beta_i in 1:length(betas)) {
    c = comp_plot_count[beta_i,ni,p_i]
    b_alpha = c+1
    b_beta = Niter-c+1
    quantiles[beta_i,] = qbeta(
        c(0.025, 0.5, 0.975), b_alpha, b_beta)
}
g = ggplot()
g = g + geom_pointrange(
    data = data.frame(
        x = betas,
        y = quantiles[,2],
        ymin = quantiles[,1],
        ymax = quantiles[,3]
    ),
    aes(x=x, y=y, ymax=ymax, ymin=ymin)
)
g = g +
    scale_x_continuous(trans="log2") +
    xlab("beta") +
    ylab(NULL) +
    coord_cartesian(ylim=c(0,1)) +
    ggtitle(sprintf(
        "proportion of looM1 > looM0, %s_%s_%s, n=%d",
        truedist, modeldist, priordist, Ns[ni]
    ))


# ==========================================
# mean bb violins
bbsamples = 4000
bbalpha = 1.0
bbs = rdirichlet(bbsamples, rep(bbalpha, Niter))

gs = vector('list', length(n_included)*length(p_included))
for (p_ii in 1:length(p_included)) {
    p_i = p_included[p_ii]
    for (nii in 1:length(n_included)) {
        ni = n_included[nii]
        n = Ns[ni]
        g = ggplot() + scale_x_continuous(trans="log2")
        for (beta_i in 1:length(betas)) {
            beta = betas[beta_i]
            # load data in variable out
            load(sprintf('res_comp/%s_%s_%s_%d_%d_%d.RData',
                truedist, modeldist, priordist, Ps[p_i], n, beta_i))
            loos = out$loos
            comps = colSums(loos[,,2]) > colSums(loos[,,1])
            c = bbs %*% comps
            g = g + geom_violin(
                data = data.frame(
                    y = c,
                    x = beta
                ),
                aes(x=x, y=y),
                width = 0.5
            )
        }
        g = g +
            xlab("beta") +
            ylab(NULL) +
            coord_cartesian(ylim=c(0,1)) +
            ggtitle(sprintf("n=%d, p=%g", Ns[ni], Ps[p_i]))
        gs[[(nii-1)*length(p_included) + p_ii]] = g
    }
}
dev.new(height=10, width=16)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], gs[[5]], gs[[6]], nrow=2,
    top = sprintf(
        "proportion of looM1 > looM0, %s_%s_%s",
        truedist, modeldist, priordist
    )
)


# ==========================================
# bb violins
gs = vector('list', length(n_included)*length(p_included))
for (p_ii in 1:length(p_included)) {
    p_i = p_included[p_ii]
    for (nii in 1:length(n_included)) {
        ni = n_included[nii]
        n = Ns[ni]
        g = ggplot() + scale_x_continuous(trans="log2")
        quantiles = array(NaN, c(length(betas), 3))
        for (beta_i in 1:length(betas)) {
            beta = betas[beta_i]
            # load data in variable out
            load(sprintf('res_comp/%s_%s_%s_%d_%d_%d.RData',
                truedist, modeldist, priordist, Ps[p_i], n, beta_i))
            g = g + geom_violin(
                data = data.frame(
                    y = out$comp,  # LOO
                    # y = out$compt,  # test
                    x = beta
                ),
                aes(x=x, y=y),
                width = 0.5
            )
            # calc point range quantiles
            count_true = sum(colSums(out$loos[,,2]) > colSums(out$loos[,,1]))
            # count_true = sum(out$tls[,2] > out$tls[,1])
            b_alpha = count_true + 1
            b_beta = Niter - count_true + 1
            quantiles[beta_i,] = qbeta(c(0.025, 0.5, 0.975), b_alpha, b_beta)
        }
        # add point range
        g = g + geom_pointrange(
            data = data.frame(
                x = betas,
                y = quantiles[,2],
                ymin = quantiles[,1],
                ymax = quantiles[,3]
            ),
            aes(x=x, y=y, ymax=ymax, ymin=ymin),
            color='red'
        )
        # decorate
        g = g +
            xlab("beta") +
            ylab(NULL) +
            coord_cartesian(ylim=c(0,1)) +
            ggtitle(sprintf("n=%d, p=%g", Ns[ni], Ps[p_i]))
        gs[[(nii-1)*length(p_included) + p_ii]] = g
    }
}
dev.new(height=10, width=16)
# pdf(
#     file = sprintf(
#         "figs_comp/bb1_%s_%s_%s.pdf",
#         truedist, modeldist, priordist
#     ),
#     height=10, width=16
# )
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], gs[[5]], gs[[6]], nrow=2,
    top = sprintf(
        "Pr(z > 0), %s_%s_%s",
        truedist, modeldist, priordist
    )
)
# dev.off()


# ==========================================
# posterior coef positive
gs = vector('list', length(n_included)*length(p_included))
for (p_ii in 1:length(p_included)) {
    p_i = p_included[p_ii]
    for (nii in 1:length(n_included)) {
        ni = n_included[nii]
        n = Ns[ni]
        g = ggplot() + scale_x_continuous(trans="log2")
        for (beta_i in 1:length(betas)) {
            beta = betas[beta_i]
            # load data in variable out
            load(sprintf('res_comp/%s_%s_%s_%d_%d_%d.RData',
                truedist, modeldist, priordist, Ps[p_i], n, beta_i))
            g = g + geom_violin(
                data = data.frame(
                    y = out$beta_pos,
                    x = beta
                ),
                aes(x=x, y=y),
                width = 0.5
            )
        }
        g = g +
            xlab("beta") +
            ylab(NULL) +
            coord_cartesian(ylim=c(0,1)) +
            ggtitle(sprintf("n=%d, p=%g", Ns[ni], Ps[p_i]))
        gs[[(nii-1)*length(p_included) + p_ii]] = g
    }
}
dev.new(height=10, width=16)
# pdf(
#     file = sprintf(
#         "figs_comp/betapos_%s_%s_%s.pdf",
#         truedist, modeldist, priordist
#     ),
#     height=10, width=16
# )
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], gs[[5]], gs[[6]], nrow=2,
    top = sprintf(
        "Pr(beta_p+1 > 0), %s_%s_%s",
        truedist, modeldist, priordist
    )
)
# dev.off()



# # ==========================================
# # violin
# ni = 1
# p_i = 1
# viol_gridn = 50
# viol = array(NaN, c(viol_gridn, length(betas), 2))
# viol_mean = numeric(length(betas))
# for (beta_i in 1:length(betas)) {
#     c = comp_plot_count[beta_i,ni,p_i]
#     b_alpha = c+1
#     b_beta = Niter-c+1
#     viol_mean[beta_i] = b_alpha / (b_alpha + b_beta)
#     viol[,beta_i,1] = seq(
#         qbeta(0.01, b_alpha, b_beta),
#         qbeta(0.99, b_alpha, b_beta),
#         length=viol_gridn
#     )
#     viol[,beta_i,2] = dbeta(viol[,beta_i,1], b_alpha, b_beta)
# }
# g = plot_known_viol(
#     betas, t(viol[,,1]), t(viol[,,2]),
#     line = viol_mean,
#     colors='blue'
# )
# g = g +
#     coord_cartesian(ylim=c(0,1)) +
#     xlab("") +
#     ylab("")
