library(ggplot2)
library(matrixStats)
library(grid)
library(gridExtra)
library(reshape)
library(extraDistr)
# library(bayesboot)
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
Ns<-c(10, 20, 40, 60, 100, 140, 200, 260)
Ps<-c(1, 2, 5, 10)

Niter = 2000

# ==============================================================================
# select model

# number of jobs (20)
num_job = length(dists) * length(Ps)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
p_i = jobi %% length(Ps)
dist_i = jobi %/% length(Ps)

truedist = dists[[dist_i+1]][1]
modeldist = dists[[dist_i+1]][2]
priordist = dists[[dist_i+1]][3]
p_i = p_i + 1

sprintf('jobi=%d', jobi)
sprintf('%s, %s, %s, %d', truedist, modeldist, priordist, Ps[p_i])

# ==============================================================================
# load results data

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
# run loo_calibration_i to save all the plots for current results data
source("loo_calibration_i_tuomas.R")
