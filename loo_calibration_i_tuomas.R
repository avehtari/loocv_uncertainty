
# ==============================================================================
## normal approximation accuracy with some quantiles as a function of n

quantiles = c(0.05, 0.10, 0.90, 0.95)
viol_grid_n = 50
accs_q = array(0, c(2, length(Ns), length(quantiles), 5))
accs_m = array(0, c(2, length(Ns), length(quantiles)))
accs_viol = array(0, c(2, length(Ns), length(quantiles), 2, viol_grid_n))

for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    t_g2s = pnorm(
        tls-colSums(loos),
        mean = 0,
        sd = sqrt(colVars(loos) + n*g2s)*sqrt(n)
    )
    t_g3s = pnorm(
        tls-colSums(loos),
        mean = 0,
        sd = sqrt(colVars(loos) + n*0*g3s)*sqrt(n)
    )
    for (qi in 1:length(quantiles)) {
        # g2s
        q = quantiles[qi]
        c = sum(t_g2s < quantiles[qi])
        alpha = c+1
        beta = Niter-c+1
        accs_m[1,ni,qi] = alpha / (alpha + beta)
        accs_q[1,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), alpha, beta)
        accs_viol[1,ni,qi,1,] = seq(
            qbeta(0.01, alpha, beta),
            qbeta(0.99, alpha, beta),
            length=viol_grid_n)
        accs_viol[1,ni,qi,2,] = dbeta(accs_viol[1,ni,qi,1,], alpha, beta)
        # g3s
        c = sum(t_g3s < quantiles[qi])
        alpha = c+1
        beta = Niter-c+1
        accs_m[2,ni,qi] = alpha / (alpha + beta)
        accs_q[2,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), alpha, beta)
        accs_viol[2,ni,qi,1,] = seq(
            qbeta(0.01, alpha, beta),
            qbeta(0.99, alpha, beta),
            length=viol_grid_n)
        accs_viol[2,ni,qi,2,] = dbeta(accs_viol[2,ni,qi,1,], alpha, beta)

    }
}

# plot

# >>>>>> -------------------------------------
for (gis in c(1,2)) {
# >>>>>> -------------------------------------

# 1 -> g2s, 2 -> g3s
#gis = 1
# -------------------------------------

gs = vector('list', length(Ns))
for (qi in 1:length(quantiles)) {
    g = plot_known_viol(
        Ns, accs_viol[gis,,qi,1,], accs_viol[gis,,qi,2,],
        # range1 = accs_q[gis,,qi,c(1,5)],
        # range2 = accs_q[gis,,qi,c(2,4)],
        line = accs_q[gis,,qi,3],
        point = accs_m[gis,,qi],
        colors='blue'
    )
    g = g +
        geom_hline(yintercept=quantiles[qi]) +
        coord_cartesian(ylim=c(
            min(quantiles[qi], min(accs_viol[gis,,qi,1,])),
            max(quantiles[qi], max(accs_viol[gis,,qi,1,]))
        )) +
        ggtitle(sprintf("quantile %g", quantiles[qi])) +
        xlab("") +
        ylab("")
    gs[[qi]] = g
}
pdf(
    file = sprintf(
        "figs/p1_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], gis
    ),
    width=15, height=5
)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top=sprintf(
        "norm approx accuracy with quantiles (model: %s_%s_%s_%i)\ng%is",
        truedist, modeldist, priordist, Ps[p_i],
        if (gis == 1) 2 else 3),
    bottom="number of samples")
dev.off()

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(accs_q, accs_m, accs_viol, quantiles, gs)


# ==============================================================================
# peformance of mean and variance terms

bbsamples = 4000

# >>>>>> -------------------------------------
for (alpha_i in c(1,2)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
alphas = c(1, 1 - 6/n)
alphas_str = c("1", "1 - 6/n")
alpha = alphas[alpha_i]
pvs = array(0, c(3, length(Ns)))
bbs = array(0, c(3, length(Ns), bbsamples))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    # # mean ##### for MSEs, need to be fixed !!!!!!!!!!!!!!
    # pmz = sd(t(tls-colSums(muloos)))
    # colvars_muloos = colVars(muloos)
    # pm[ni,1] = sqrt(mean(colvars_muloos*n + 0*gms*n*n)) / pmz
    # pm[ni,2] = sqrt(mean(colvars_muloos*n + gms*n*n)) / pmz
    # pm[ni,3] = sqrt(mean(colvars_muloos*n + gm2s*n*n)) / pmz
    # pm[ni,1:3] = pm[ni,1:3]*n/(n-1)
    # pm[ni,4] = n
    # pmq[ni,1:4] = sqrt(quantile(colvars_muloos, c(0.05, 0.1, 0.9, 0.95))) / pmz
    # pmq[ni,5] = n


    ## variance
    pvz = sd(t(tls-colSums(loos)))
    colvars_loos_n = colVars(loos)*n
    # gs
    t = colvars_loos_n + 0*gs*n*n
    pvs[1,ni] = sqrt(mean(t)) / pvz
    # bbs[1,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples, use.weights=TRUE)$V1) / pvz
    bbs[1,ni,] = sqrt(rdirichlet(bbsamples, rep(alpha, length(t))) %*% t) / pvz
    # g2s
    t = colvars_loos_n + g2s*n*n
    pvs[2,ni] = sqrt(mean(t)) / pvz
    # bbs[2,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples, use.weights=TRUE)$V1) / pvz
    bbs[2,ni,] = sqrt(rdirichlet(bbsamples, rep(alpha, length(t))) %*% t) / pvz
    # g3s
    t = colvars_loos_n + g3s*n*n
    pvs[3,ni] = sqrt(mean(t)) / pvz
    # bbs[3,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples, use.weights=TRUE)$V1) / pvz
    bbs[3,ni,] = sqrt(rdirichlet(bbsamples, rep(alpha, length(t))) %*% t) / pvz

}

## plot
g = ggplot()
fillcats = c("gs", "g2s", "g3s")
for (fi in 1:3) {
    for (ni in 1:length(Ns)) {
        g = g + geom_violin(
            data = data.frame(
                y = bbs[fi,ni,],
                x = Ns[ni],
                fill = fillcats[fi]
            ),
            aes(x=x, y=y, fill=fill),
            width = 8,
            alpha = 0.5
        )
    }
}
g = g +
    geom_hline(yintercept=1) +
    scale_fill_manual(values=c("green", "red", "blue")) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "peformance of variance terms (model: %s_%s_%s_%i)\nalpha=%s, bbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        alphas_str[alpha_i],
        bbsamples
    ))
# g
ggsave(
    plot=g, width=8, height=5,
    filename = sprintf(
        "figs/p2_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    )
)

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(pvs, bbs)


# ==============================================================================
## Analytic variance for Bayesian bootstrap (thanks to Juho)

# >>>>>> -------------------------------------
for (alpha_i in c(1,2)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
alphas = c(1, 1 - 6/n)
alphas_str = c("1", "1 - 6/n")
alpha = alphas[alpha_i]
avs = array(0, c(2, length(Ns), Niter))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    avs[1,ni,] = sqrt(1/(n*(n*alpha+1))*(colSums(loos^2) - n*colMeans(loos)^2))*n
    ## Gaussian approximation
    colVarsLoos = colVars(loos)
    # avs[2,ni,] = sqrt((colVarsLoos*n + g2s*n*n)) * n/n*n/(n-1)
    # avs[2,ni,] = sqrt((colVarsLoos*n + 0*g2s*n*n))
    avs[2,ni,] = sqrt((colVarsLoos*n + g2s*n*n)) # extra variance

}

## plot
g = ggplot()
fillcats = c("analytic", "gaussian approx")
for (fi in 1:2) {
    for (ni in 1:length(Ns)) {
        g = g + geom_violin(
            data = data.frame(
                y = avs[fi,ni,],
                x = Ns[ni],
                fill = fillcats[fi]
            ),
            aes(x=x, y=y, fill=fill),
            width = 8,
            alpha = 0.5
        )
    }
}
g = g +
    scale_fill_manual(values=c("green", "red")) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "analytic var for BB (model: %s_%s_%s_%i)\nalpha=%s",
        truedist, modeldist, priordist, Ps[p_i],
        alphas_str[alpha_i]
    ))
# g
ggsave(
    plot=g, width=8, height=5,
    filename = sprintf(
        "figs/p3_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    )
)

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(avs)


# ==============================================================================
## Empirical variance for Bayesian bootstrap

bbsamples = 4000

# >>>>>> -------------------------------------
for (alpha_i in c(1,2)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
alphas = c(1, 1 - 6/n)
alphas_str = c("1", "1 - 6/n")
alpha = alphas[alpha_i]
evs = array(0, c(length(Ns), Niter))
dws = vector('list', length(Ns))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    dw = rdirichlet(bbsamples, rep(alpha, n))
    dws[[ni]] = dw
    # evs[ni,] = apply(loos, 2, function(loos_i) sd(rowSums(t(t(dw)*loos_i))*n))
    evs[ni,] = colSds((dw%*%loos)*n)
}

## plot
g = ggplot()
for (ni in 1:length(Ns)) {
    g = g + geom_violin(
        data = data.frame(
            y = evs[ni,],
            x = Ns[ni]
        ),
        aes(x=x, y=y),
        width = 8
    )
}
g = g +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "empirical var for BB (model: %s_%s_%s_%i)\nalpha=%s, bbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        alphas_str[alpha_i],
        bbsamples
    ))
# g
ggsave(
    plot=g, width=8, height=5,
    filename = sprintf(
        "figs/p4_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    )
)

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(evs)
