
library(matrixStats)
library(extraDistr)


Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
Ps = c(1, 2, 5, 10)

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

p_i = 1

# ==============================================================================
# peformance of variance terms

# settings for bayesian bootstrap
bbsamples_perf = 2000
bbalpha_perf = 1

# output arrays
pvs = array(0, c(4, length(Ns)))
bbs = array(0, c(4, length(Ns), bbsamples_perf))

for (ni in 1:length(Ns)) {
    n = Ns[ni]
    # load data in variable out
    load(sprintf('res_loo/%s_%s_%s_%d_%d.RData',
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
    rm(out)
    Niter = dim(loos)[2]

    pvz = sd(t(tls-colSums(loos)))
    colvars_loos_n = colVars(loos)*n

    # basic
    t = colvars_loos_n
    pvs[1,ni] = sqrt(mean(t)) / pvz
    bbs[1,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # g2s
    t = colvars_loos_n + g2s*n*n
    pvs[2,ni] = sqrt(mean(t)) / pvz
    # bbs[2,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[2,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # skewed generalised t fit
    # load data
    load(sprintf('res_loo_sgt/%s_%s_%s_%d_%d.RData',
        truedist, modeldist, priordist, Ps[p_i], n))
    list2env(out, envir=environment())
    rm(out)
    pvs[3,ni] = sqrt(mean(vm_iter)) / pvz
    bbs[3,ni,] = sqrt(vm_samp) / pvz

    # x2
    t = 2*colvars_loos_n
    pvs[4,ni] = sqrt(mean(t)) / pvz
    # bbs[3,] = sqrt(bayesboot(t, function(d, w) sum(d*w), R=bbsamples_perf, use.weights=TRUE)$V1) / pvz
    bbs[4,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz
