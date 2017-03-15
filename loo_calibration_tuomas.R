library(ggplot2)
library(matrixStats)
library(gridExtra)
library(reshape)
library(extraDistr)
library(bayesboot)
source('gg_qq.R')

# possible parameters
dists = list(
    c('n', 'n', 'n'),
    c('t4', 'tnu', 'n'),
    c('b', 'b', 'n'),
    c('n', 'tnu', 'n'),
    c('t4', 'n', 'n')
)
Ns<-c(10, 20, 40, 60, 100, 140, 200)
Ps<-c(1, 2, 5, 10)

# ---------------------------------------------------
# select params

Niter = 1000
p = 2

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# ---------------------------------------------------

# load data for all n
outs = vector('list', length(Ns))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # load data in variable out
    load(sprintf('res/%s_%s_%s_%d_%d.RData',
        truedist, modeldist, priordist, p, n))
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
    # store out into list outs
    outs[[ni]] = out
}


# ==============================================================================

# select n
ni = 5
n = Ns[ni]
# populate local environment with the named stored variables in selected out
list2env(outs[[ni]], envir=environment())

# normal approximation accuracy
qplot(
    colSds(loos)*sqrt(n),
    sqrt(colVars(loos)+n*g2s)*sqrt(n)
) +
geom_abline(intercept=0, slope=1)
c(mean((sqrt(colVars(loos)+n*g2s)*sqrt(n))), sd(tls-colSums(loos)))

# normal approximation accuracy
qplot(
    tls-colSums(loos),
    sqrt(colVars(loos)+n*g2s)*sqrt(n)
)

# normal approximation accuracy with qq-plot
gg_qq(qnorm(pnorm(
    tls-colSums(loos),
    mean = 0,
    sd = sqrt(colVars(loos)+n*1*g2s)*sqrt(n)
)))

# normal approximation accuracy with some quantiles for some n
mean(pnorm(
    tls-colSums(loos),
    mean = 0,
    sd = (sqrt(colVars(loos)+n*g2s)*sqrt(n))
) > 0.95)
mean(pnorm(
    tls-colSums(loos),
    mean = 0,
    sd = (sqrt(colVars(loos)+n*0*g3s)*sqrt(n))
) > 0.95)
mean(pnorm(
    tls-colSums(loos),
    mean = 0,
    sd = (sqrt(colVars(loos)+n*g2s)*sqrt(n))
) < 0.05)
mean(pnorm(
    tls-colSums(loos),
    mean = 0,
    sd = (sqrt(colVars(loos)+n*0*g3s)*sqrt(n))
) < 0.05)

# ==============================================================================
## accuracy as a function of n

quantiles = c(0.05, 0.10, 0.90, 0.95)
accs_g2s = matrix(0, length(Ns), length(quantiles))
accs_g3s = matrix(0, length(Ns), length(quantiles))
accs_g2s_05 = matrix(0, length(Ns), length(quantiles))
accs_g2s_95 = matrix(0, length(Ns), length(quantiles))
accs_g3s_05 = matrix(0, length(Ns), length(quantiles))
accs_g3s_95 = matrix(0, length(Ns), length(quantiles))
# manual violin
vgrid_n = 50
accs_g2s_viol = array(0, c(length(Ns), length(quantiles), 2, vgrid_n))
accs_g3s_viol = array(0, c(length(Ns), length(quantiles), 2, vgrid_n))

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
        c = sum(t_g2s < quantiles[qi])
        accs_g2s[ni,qi] = c/Niter
        accs_g2s_05[ni,qi] = qbeta(0.05, c+1, Niter-c+1)
        accs_g2s_95[ni,qi] = qbeta(0.95, c+1, Niter-c+1)
        accs_g2s_viol[ni,qi,1,] = seq(
            qbeta(0.01, c+1, Niter-c+1),
            qbeta(0.99, c+1, Niter-c+1),
            length=vgrid_n)
        accs_g2s_viol[ni,qi,2,] = dbeta(accs_g2s_viol[ni,qi,1,], c+1, Niter-c+1)
        # g3s
        c = sum(t_g3s < quantiles[qi])
        accs_g3s[ni,qi] = c/Niter
        accs_g3s_05[ni,qi] = qbeta(0.05, c+1, Niter-c+1)
        accs_g3s_95[ni,qi] = qbeta(0.95, c+1, Niter-c+1)
        accs_g3s_viol[ni,qi,1,] = seq(
            qbeta(0.01, c+1, Niter-c+1),
            qbeta(0.99, c+1, Niter-c+1),
            length=vgrid_n)
        accs_g3s_viol[ni,qi,2,] = dbeta(accs_g3s_viol[ni,qi,1,], c+1, Niter-c+1)
    }
}

# g2s
vscale = 0.1
qi = 2
plotd = data.frame(
    x=Ns, y=accs_g2s[,qi], ymin=accs_g2s_05[,qi], ymax=accs_g2s_95[,qi]
)
# setup violin dataframes
plotdv = vector("list", length(Ns))
for (ni in 1:length(Ns)) {
    plotdv[[ni]] = data.frame(
        x = Ns[ni] + vscale*c(
            -accs_g2s_viol[ni,qi,2,], rev(accs_g2s_viol[ni,qi,2,])),
        y = c(accs_g2s_viol[ni,qi,1,], rev(accs_g2s_viol[ni,qi,1,]))
    )
}
# construct plot
g = ggplot(plotd)
for (ni in 1:length(Ns)) {
    g = g + geom_polygon(data=plotdv[[ni]], aes(x=x, y=y))
}
g = g +
    geom_point(aes(x=x, y=y), color='red') +
    geom_abline(intercept=quantiles[qi], slope=0) +
    coord_cartesian(ylim=c(
        min(quantiles[qi], 0.99*min(accs_g2s_05[,qi])),
        max(quantiles[qi], 1.01*max(accs_g2s_95[,qi]))
    )) +
    ggtitle(sprintf("g2s %g", quantiles[qi]))
g

# g3s
qi = 2
plotd = data.frame(
    x=Ns, y=accs_g3s[,qi], ymin=accs_g3s_05[,qi], ymax=accs_g3s_95[,qi]
)
# setup violin dataframes
plotdv = vector("list", length(Ns))
for (ni in 1:length(Ns)) {
    plotdv[[ni]] = data.frame(
        x = Ns[ni] + vscale*c(
            -accs_g3s_viol[ni,qi,2,], rev(accs_g3s_viol[ni,qi,2,])),
        y = c(accs_g3s_viol[ni,qi,1,], rev(accs_g3s_viol[ni,qi,1,]))
    )
}
# construct plot
g = ggplot(plotd)
for (ni in 1:length(Ns)) {
    g = g + geom_polygon(data=plotdv[[ni]], aes(x=x, y=y))
}
g = g +
    geom_point(aes(x=x, y=y), color='red') +
    geom_abline(intercept=quantiles[qi], slope=0) +
    coord_cartesian(ylim=c(
        min(quantiles[qi], 0.99*min(accs_g3s_05[,qi])),
        max(quantiles[qi], 1.01*max(accs_g3s_95[,qi]))
    )) +
    ggtitle(sprintf("g3s %g", quantiles[qi]))
g

# ==============================================================================

# plot peformance of mean and variance terms
# pm1 = matrix(0, length(Ns), 3)
# pm2 = matrix(0, length(Ns), 3)
# pm3 = matrix(0, length(Ns), 3)
pv1 = matrix(0, length(Ns), 3)
pv2 = matrix(0, length(Ns), 3)
pv3 = matrix(0, length(Ns), 3)
bbsamples = 1000
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
    pv1[ni,1] = sqrt(mean(t)) / pvz
    tbb = bayesboot(t, mean, R=bbsamples)
    pv1[ni,2:3] = sqrt(quantile(tbb$V1, c(0.05, 0.95))) / pvz # quantiles
    # g2s
    t = colvars_loos_n + g2s*n*n
    pv2[ni,1] = sqrt(mean(t)) / pvz
    tbb = bayesboot(t, mean, R=bbsamples)
    pv2[ni,2:3] = sqrt(quantile(tbb$V1, c(0.05, 0.95))) / pvz # quantiles
    # g3s
    t = colvars_loos_n + g3s*n*n
    pv3[ni,1] = sqrt(mean(t)) / pvz
    tbb = bayesboot(t, mean, R=bbsamples)
    pv3[ni,2:3] = sqrt(quantile(tbb$V1, c(0.05, 0.95))) / pvz # quantiles

}


# # var
# pv = data.frame(pv)
# colnames(pv)[4] = "n"
# pvm = melt(pv, id = "n")

## var
plotd = data.frame(
    x=Ns,
    y1=pv1[,1], ymin1=pv1[,2], ymax1=pv1[,3],
    y2=pv2[,1], ymin2=pv2[,2], ymax2=pv2[,3],
    y3=pv3[,1], ymin3=pv3[,2], ymax3=pv3[,3])
ggplot(plotd) +
    geom_pointrange(aes(x=x, y=y1, ymin=ymin1, ymax=ymax1), color='red') +
    geom_pointrange(aes(x=x, y=y2, ymin=ymin2, ymax=ymax2), color='green') +
    geom_pointrange(aes(x=x, y=y3, ymin=ymin3, ymax=ymax3), color='blue') +
    geom_hline(yintercept=1)


# ==============================================================================
## Analytic variance for Bayesian bootstrap (thanks to Juho)

# select n
ni = 4
n = Ns[ni]
# populate local environment with the named stored variables in selected out
list2env(outs[[ni]], envir=environment())


alpha = 1
# alpha = 1 - 6/n
qplot(sqrt(1/(n*(n*alpha+1))*(colSums(loos^2) - n*colMeans(loos)^2))*n)

## Gaussian approximation
qplot(
    sqrt((colVars(loos)*n + g2s*n*n)) * n/n*n/(n-1)
)
## Gaussian approximation
qplot(
    sqrt((colVars(loos)*n + 0*g2s*n*n))
)
## Gaussian approximation with extra variance
qplot(
    sqrt((colVars(loos)*n + g2s*n*n))
)


# ==============================================================================
## Empirical variance for Bayesian bootstrap

# select n
ni = 4
n = Ns[ni]
# populate local environment with the named stored variables in selected out
list2env(outs[[ni]], envir=environment())

n_emp = 80

alpha=1
# alpha = 1 - 6/n
dw = rdirichlet(n_emp, rep(alpha, n))
sds = apply(loos, 2, function(loos_i) sd(rowSums(t(t(dw)*loos_i))*n))
qplot(sds)


# ==============================================================================
## Calibration for Bayesian bootstrap

dw=rdirichlet(n_emp,matrix(1-6/n,1,n))
#dw=rdirichlet(n_emp,matrix(1,1,n))
#dw=rdirichlet(n_emp,matrix(1-6/20,1,n))
qp=matrix(0,100,1)
qw=matrix(0,n_emp,100)
for (i1 in 1:100) {
    qw[,i1]=rowSums(t(t(as.matrix(dw))*as.vector(loos[,i1])))*n
    qp[i1]=mean(qw[,i1]<=tls[i1])
}
qplot(qp)
c(mean(qp<0.05), mean(qp<0.1), mean(qp<0.9), mean(qp<0.95))

c(mean(colSums(loos)),mean(t(tls)))
qplot(x=colSums(loos),y=t(tls))+geom_abline(intercept=0,slope=1)

## Calibration for Bayesian double bootstrap
ni=4
dw=rdirichlet(n_emp,matrix(1,1,n))
qww=matrix(0,n_emp,100)
n=n
for (i1 in 1:100) {
    log_lik1=lls[[i1]]
    psis1=psislw(-log_lik1)
    qq<-matrix(nrow=n,ncol=n)
    for (cvi in 1:n) {
        qq[cvi,]<-log(colSums(exp(log_lik1+psis1$lw_smooth[,cvi])))
    }
    qqw<-qq
    for (cvi in 1:n) {
        qqw[-cvi,cvi]<-(qq[-cvi,cvi]-mean(qq[-cvi,cvi]))+qq[cvi,cvi]
    }
    for (i2 in 1:n_emp) {
        qww[i2,i1]=sum(   t(t(rdirichlet(n,matrix(1,1,n)))*as.vector(rdirichlet(1,matrix(1,1,n))))*qqw)*n
    }
}
qpp=matrix(0,100,1)
for (i1 in 1:100) {
    qpp[i1]=mean(qww[,i1]<=tls[i1])
}
qplot(qpp)
c(mean(qpp<0.05), mean(qpp<0.1), mean(qpp<0.9), mean(qpp<0.95))

qplot(sort(rowSums(t(t(as.matrix(dw))*as.vector(loos[,i1])))*n),sort(qww[,1]))+geom_abline()

##RMSE test code
ni=5
dw=rdirichlet(n_emp,matrix(1-6/n,1,n))
#dw=rdirichlet(n_emp,matrix(1,1,n))
#dw=rdirichlet(n_emp,matrix(1-6/20,1,n))
qmp=matrix(0,100,1)
qmw=matrix(0,n_emp,100)
    truedist="n"
for (i1 in 1:100) {
    psis1=psislw(-lls[[i1]])
    loomu=colSums(mus[[i1]]*exp(psis1$lw_smooth))
    set.seed(i1)

                if (truedist=="n") {
                    y <- as.array(rnorm(n))
                } else if (truedist=="t4") {
                    y <- as.array(rt(n,4))
                } else if (truedist=="b") {
                    y <- as.array(as.double(kronecker(matrix(1,1,n),t(c(0,1)))))
                } else {
                    stop("Unknown true distribution")
                }
    qmw[,i1]=1-rowSums(t(t(as.matrix(dw))*as.vector(y-loomu)^2))/var(y)
    qmp[i1]=mean(qw[,i1]>0)
}
qplot(qp)
c(mean(qp<0.05), mean(qp<0.1), mean(qp<0.9), mean(qp<0.95))
