library(ggplot2)
library(matrixStats)
library(grid)
library(gridExtra)
library(reshape)
library(extraDistr)
library(bayesboot)
source('gg_qq.R')
source('plot_known_viol.R')

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

Niter = 100

# ==============================================================================
# load results data

# select params
p_i = 2
truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# load data for all n
outs = vector('list', length(Ns))
for (ni in 1:length(Ns)) {
    n = Ns[ni]
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
    # store out into list outs
    outs[[ni]] = out
}


# ==============================================================================
# some plots for selected n

# select n
ni = 4
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
## normal approximation accuracy with some quantiles as a function of n
# ... in loo_calibration_i

# ==============================================================================
# peformance of mean and variance terms
# ... in loo_calibration_i

# ==============================================================================
## Analytic variance for Bayesian bootstrap (thanks to Juho)
# ... in loo_calibration_i

# ==============================================================================
## Empirical variance for Bayesian bootstrap
# ... in loo_calibration_i


# ==============================================================================
# run loo_calibration_i to save all the plots for current results data
# source("loo_calibration_i_tuomas.R")










##RMSE test code
ni=5
dw=rdirichlet(bbsamples,matrix(1-6/n,1,n))
#dw=rdirichlet(bbsamples,matrix(1,1,n))
#dw=rdirichlet(bbsamples,matrix(1-6/20,1,n))
qmp=matrix(0,100,1)
qmw=matrix(0,bbsamples,100)
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
