
# ==============================================================================
## Find alpha for BB calibration

bbsamples = 2000

# n = 200
ni = 7
n = Ns[ni]
# populate local environment with the named stored variables in selected out
list2env(outs[[ni]], envir=environment())


a_grid = seq(0.05, 1, 0.05)


quantiles = c(0.05, 0.10, 0.90, 0.95)
viol_grid_n = 50
accs_q = array(0, c(length(a_grid), length(quantiles), 5))
accs_m = array(0, c(length(a_grid), length(quantiles)))
accs_viol = array(0, c(length(a_grid), length(quantiles), 2, viol_grid_n))

## calc
# qps = array(0, c(length(Ns), Niter))
for (ai in 1:length(a_grid)) {
    alpha = a_grid[ai]

    dw = rdirichlet(bbsamples, rep(alpha, n))
    qw = (dw%*%loos)*n

    qp = colMeans(t(t(qw) <= tls))

    for (qi in 1:length(quantiles)) {
        c = sum(qp < quantiles[qi])
        b_alpha = c+1
        b_beta = Niter-c+1
        accs_m[ai,qi] = b_alpha / (b_alpha + b_beta)
        accs_q[ai,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
        accs_viol[ai,qi,1,] = seq(
            qbeta(0.01, b_alpha, b_beta),
            qbeta(0.99, b_alpha, b_beta),
            length=viol_grid_n)
        accs_viol[ai,qi,2,] = dbeta(accs_viol[ai,qi,1,], b_alpha, b_beta)

    }

}


# plot
gs = vector('list', length(Ns))
for (qi in 1:length(quantiles)) {
    g = plot_known_viol(
        a_grid, accs_viol[,qi,1,], accs_viol[,qi,2,],
        # range1 = accs_q[,qi,c(1,5)],  # 95 % interval
        # range2 = accs_q[,qi,c(2,4)],  # 50 % interval
        line = accs_q[,qi,3],  # median
        # point = accs_m[,qi],  # mean
        colors='blue'
    )
    g = g + geom_hline(yintercept=quantiles[qi])
    if (quantiles[qi] < 0.5) {
        g = g + coord_cartesian(
            ylim=c(0, max(quantiles[qi], max(accs_viol[,qi,1,]))))
    } else {
        g = g + coord_cartesian(
            ylim=c(min(quantiles[qi], min(accs_viol[,qi,1,])), 1))
    }
    g = g +
        ggtitle(sprintf("quantile %g", quantiles[qi])) +
        xlab("") +
        ylab("")
    gs[[qi]] = g
}
pdf(
    file = sprintf(
        "figs/find_a_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], n
    ),
    width=15, height=5
)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top = sprintf(
        "Find alpha for BB (model: %s_%s_%s_%i)\nn=%i, bbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        n, bbsamples
    ),
    bottom = "alpha")
dev.off()
