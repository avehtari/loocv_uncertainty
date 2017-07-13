
# ==============================================================================
## Get bb samples for each ni

bbsamples = 2000
alphas = c(function(n) 1, function(n) 1 - 6/n)
alphas_str = c("1", "1 - 6/n")
# dws = array(vector('list'), c(length(alphas), length(Ns)))
qws = array(vector('list'), c(length(alphas), length(Ns)))
for (alpha_i in 1:length(alphas)) {
    for (ni in 1:length(Ns)) {
        n = Ns[ni]
        alpha = alphas[[alpha_i]](n)
        # dws[[alpha_i, ni]] = rdirichlet(bbsamples, rep(alpha, n))
        dw = rdirichlet(bbsamples, rep(alpha, n))

        # populate local environment with the named stored variables in selected out
        list2env(outs[[ni]], envir=environment())

        qws[[alpha_i, ni]] = (dw%*%loos)*n
    }
}


# ==============================================================================
# peformance of mean and variance terms

## calc
bbsamples_perf = 2000
bbalpha = 1

pvs = array(0, c(4, length(Ns)))
bbs = array(0, c(4, length(Ns), bbsamples_perf))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    # # MSE ##### for MSEs, need to be fixed !!!!!!!!!!!!!!
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
    # bbs[1,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[1,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha, length(t))) %*% t) / pvz
    # g2s
    t = colvars_loos_n + g2s*n*n
    pvs[2,ni] = sqrt(mean(t)) / pvz
    # bbs[2,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[2,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha, length(t))) %*% t) / pvz
    # g3s
    t = colvars_loos_n + g3s*n*n
    pvs[3,ni] = sqrt(mean(t)) / pvz
    # bbs[3,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[3,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha, length(t))) %*% t) / pvz
    # x2
    t = 2*colvars_loos_n
    pvs[4,ni] = sqrt(mean(t)) / pvz
    # bbs[3,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[4,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha, length(t))) %*% t) / pvz

}

## plot
g = ggplot()
fillcats = c("gs", "g2s", "g3s", "x2")
for (fi in 1:length(fillcats)) {
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
    scale_fill_manual(values=c("green", "red", "blue", "gray")) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "peformance of variance terms (model: %s_%s_%s_%i)\nbbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        bbsamples_perf
    ))
# g
ggsave(
    plot=g, width=8, height=5,
    filename = sprintf(
        "figs/p1_%s_%s_%s_%i.pdf",
        truedist, modeldist, priordist, Ps[p_i]
    )
)

# clear
rm(pvs, bbs)


# ==============================================================================
## Analytic variance for Bayesian bootstrap (thanks to Juho)

# >>>>>> -------------------------------------
for (alpha_i in 1:length(alphas)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
avs = array(0, c(3, length(Ns), Niter))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    alpha = alphas[[alpha_i]](n)
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    avs[1,ni,] = sqrt(1/(n*(n*alpha+1))*(colSums(loos^2) - colSums(loos)^2/n))*n
    ## Gaussian approximation
    colVarsLoos = colVars(loos)
    # avs[2,ni,] = sqrt((colVarsLoos*n + g2s*n*n)) * n/n*n/(n-1)
    # avs[2,ni,] = sqrt((colVarsLoos*n + 0*g2s*n*n))
    avs[2,ni,] = sqrt((colVarsLoos*n + g2s*n*n)) # extra variance
    avs[3,ni,] = sqrt(2*n*colVarsLoos) # x2 var

}

## plot
g = ggplot()
fillcats = c("analytic", "gaussian approx g2s", "gaussian approx x2")
for (fi in 1:3) {
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
    scale_fill_manual(values=c("green", "red", 'gray')) +
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
        "figs/p2_%s_%s_%s_%i--%i.pdf",
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

# >>>>>> -------------------------------------
for (alpha_i in 1:length(alphas)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
evs = array(0, c(length(Ns), Niter))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    alpha = alphas[[alpha_i]](n)
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    # dw = dws[[alpha_i, ni]]
    # # evs[ni,] = apply(loos, 2, function(loos_i) sd(rowSums(t(t(dw)*loos_i))*n))
    # evs[ni,] = colSds((dw%*%loos)*n)

    qw = qws[[alpha_i, ni]]
    evs[ni,] = colSds(qw)
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
        "figs/p3_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    )
)

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(evs)


# ==============================================================================
# ==============================================================================
## normal approximation accuracy with some quantiles as a function of n

quantiles = c(0.05, 0.10, 0.90, 0.95)
viol_grid_na = 50
accs_q_na = array(0, c(3, length(Ns), length(quantiles), 5))
accs_m_na = array(0, c(3, length(Ns), length(quantiles)))
accs_viol_na = array(0, c(3, length(Ns), length(quantiles), 2, viol_grid_na))

for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    colvars_loos = colVars(loos)

    t_g2s = pnorm(
        tls-colSums(loos),
        mean = 0,
        sd = sqrt(colvars_loos + n*g2s)*sqrt(n)
    )
    t_g3s = pnorm(
        tls-colSums(loos),
        mean = 0,
        sd = sqrt(colvars_loos + n*0*g3s)*sqrt(n)
    )
    t_x2 = pnorm(
        tls-colSums(loos),
        mean = 0,
        sd = sqrt(2*n*colvars_loos)
    )
    for (qi in 1:length(quantiles)) {
        # g2s
        c = sum(t_g2s < quantiles[qi])
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
        # g3s
        c = sum(t_g3s < quantiles[qi])
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
        # x2
        c = sum(t_x2 < quantiles[qi])
        b_alpha = c+1
        b_beta = Niter-c+1
        accs_m_na[3,ni,qi] = b_alpha / (b_alpha + b_beta)
        accs_q_na[3,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
        accs_viol_na[3,ni,qi,1,] = seq(
            qbeta(0.01, b_alpha, b_beta),
            qbeta(0.99, b_alpha, b_beta),
            length=viol_grid_na)
        accs_viol_na[3,ni,qi,2,] = dbeta(
            accs_viol_na[3,ni,qi,1,], b_alpha, b_beta)

    }
}


# ==============================================================================
## Calibration for Bayesian bootstrap

viol_grid_calbb = 50
accs_q_calbb = array(0, c(length(alphas), length(Ns), length(quantiles), 5))
accs_m_calbb = array(0, c(length(alphas), length(Ns), length(quantiles)))
accs_viol_calbb = array(
    0, c(length(alphas), length(Ns), length(quantiles), 2, viol_grid_calbb))

# >>>>>> -------------------------------------
for (alpha_i in 1:length(alphas)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

## calc
# qps = array(0, c(length(Ns), Niter))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    alpha = alphas[[alpha_i]](n)
    # populate local environment with the named stored variables in selected out
    list2env(outs[[ni]], envir=environment())

    qw = qws[[alpha_i, ni]]
    qp = colMeans(t(t(qw) <= tls))
    # qps[ni,] = qp

    for (qi in 1:length(quantiles)) {
        c = sum(qp < quantiles[qi])
        b_alpha = c+1
        b_beta = Niter-c+1
        accs_m_calbb[alpha_i,ni,qi] = b_alpha / (b_alpha + b_beta)
        accs_q_calbb[alpha_i,ni,qi,] = qbeta(
            c(0.025, 0.25, 0.5, 0.75, 0.975), b_alpha, b_beta)
        accs_viol_calbb[alpha_i,ni,qi,1,] = seq(
            qbeta(0.01, b_alpha, b_beta),
            qbeta(0.99, b_alpha, b_beta),
            length=viol_grid_calbb)
        accs_viol_calbb[alpha_i,ni,qi,2,] = dbeta(
            accs_viol_calbb[alpha_i,ni,qi,1,], b_alpha, b_beta)

    }

}

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------



# ==============================================================================
## PLOT both normal and BB accuracies

# find limits
ylimits = array(0, c(length(quantiles), 2))
for (qi in 1:length(quantiles)) {
    if (quantiles[qi] < 0.5) {
        ylimits[qi,1] = 0
        ylimits[qi,2] = max(
            max(accs_viol_na[,,qi,1,]),
            max(accs_viol_calbb[,,qi,1,]),
            quantiles[qi]
        )
    } else {
        ylimits[qi,2] = 1
        ylimits[qi,1] = min(
            min(accs_viol_na[,,qi,1,]),
            min(accs_viol_calbb[,,qi,1,]),
            quantiles[qi]
        )
    }
}


gis_strs = c("g2s", "g3s", "x2")
# >>>>>> -------------------------------------
for (gis in c(1,2,3)) {
# >>>>>> -------------------------------------

# 1 -> g2s, 2 -> g3s
#gis = 1
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
pdf(
    file = sprintf(
        "figs/p4_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], gis
    ),
    width=15, height=5
)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top = sprintf(
        "norm approx accuracy with quantiles (model: %s_%s_%s_%i)\n%s",
        truedist, modeldist, priordist, Ps[p_i],
        gis_strs[gis]),
    bottom = "number of samples")
dev.off()

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(gs)


# >>>>>> -------------------------------------
for (alpha_i in 1:length(alphas)) {
# >>>>>> -------------------------------------

# alpha_i = 1
# -------------------------------------

# plot
gs = vector('list', length(quantiles))
for (qi in 1:length(quantiles)) {
    g = plot_known_viol(
        Ns, accs_viol_calbb[alpha_i,,qi,1,], accs_viol_calbb[alpha_i,,qi,2,],
        # range1 = accs_q_calbb[alpha_i,,qi,c(1,5)],  # 95 % interval
        # range2 = accs_q_calbb[alpha_i,,qi,c(2,4)],  # 50 % interval
        line = accs_q_calbb[alpha_i,,qi,3],  # median
        # point = accs_m_calbb[alpha_i,,qi],  # mean
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
        "figs/p5_%s_%s_%s_%i--%i.pdf",
        truedist, modeldist, priordist, Ps[p_i], alpha_i
    ),
    width=15, height=5
)
grid.arrange(
    gs[[1]], gs[[2]], gs[[3]], gs[[4]], nrow=1,
    top = sprintf(
        "Calibrated BB accuracy (model: %s_%s_%s_%i)\nalpha=%s, bbsamples=%i",
        truedist, modeldist, priordist, Ps[p_i],
        alphas_str[alpha_i],
        bbsamples
    ),
    bottom = "number of samples")
dev.off()

# <<<<<< -------------------------------------
}
# <<<<<< -------------------------------------

# clear
rm(gs)
rm(accs_q_na, accs_m_na, accs_viol_na)
rm(accs_q_calbb, accs_m_calbb, accs_viol_calbb)
rm(quantiles)
