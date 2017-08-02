
library(matrixStats)

library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores=1, loo.cores=1)


# ==============================================================================
# setup

# possible parameters

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# ---- variables
Ps = c(1, 2, 5, 10)
Ns = c(10, 20, 40, 60, 100, 140, 200)

# number of jobs (28)
num_job = length(Ps) * length(Ns)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
n_i = (jobi %% length(Ns)) + 1
p_i = (jobi %/% length(Ns)) + 1

n = Ns[n_i]

cat(sprintf('jobi=%d\n', jobi))
cat(sprintf('%s_%s_%s_%d_%d\n', truedist, modeldist, priordist, Ps[p_i], n))


# ==============================================================================
# run the content

stansamples = 2000

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
Niter = dim(loos)[2]

# sample array
sigma2s = array(0, c(stansamples, Niter))

# ---- progress bar
progressbar_length = 40
cat(sprintf('Niter: %d\n', Niter))
cat('sampling...\n')
cat(sprintf(
    '0%% |%s| 100%%\n   |',
    paste(rep('-', progressbar_length), collapse='')
))
progressbar_cur = 0
progressbar_scale = progressbar_length / Niter
# ---- time estimator
time_estim_start_time = Sys.time()
time_estim_activate_at = c(1, 10, 20, 100)
time_estim_i = 1

# iterate for all trials
for (i in 1:Niter) {

    # skewed generalised t fit
    output <- capture.output(
        sgt_loo_fit <- stan(
            'models/sgt.stan',
            data=list(N=n, x=loos[,i]),
            iter=1000, refresh=-1, save_warmup=FALSE, open_progress=FALSE
        )
    )
    sigma2s[,i] = (extract(sgt_loo_fit, 'sigma')$sigma)^2

    # ---- progress bar
    progressbar_new = floor(i * progressbar_scale)
    if (progressbar_cur < progressbar_new) {
        cat(paste(rep('#', progressbar_new - progressbar_cur), collapse=''))
        progressbar_cur = progressbar_new
    }
    # ---- time estimator
    if (time_estim_i <= length(time_estim_activate_at) &&
            i == time_estim_activate_at[time_estim_i]) {
        time_estim_i = time_estim_i + 1
        cat(sprintf(
            '\n   | based on %d iters, estimated total runtime is %.2g h\n',
            i,
            ((Sys.time() - time_estim_start_time) * Niter / i) / 3600
        ))
        # resume progressbar
        cat('   |')
        cat(paste(rep('#', progressbar_cur), collapse=''))
    }
}
cat('|\nsampling done\n')

# calc results
vm_iter = n*colMeans(sigma2s)
vm_samp = n*rowMeans(sigma2s)

# save output to file
out = list(vm_iter=vm_iter, vm_samp=vm_samp)
filename = sprintf(
    "res_loo_sgt/%s_%s_%s_%d_%d.RData",
    truedist, modeldist, priordist, Ps[p_i], n
)
save(out, file=filename)
