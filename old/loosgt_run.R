
library(matrixStats)

library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores = parallel::detectCores())  # local
# options(mc.cores=1)  # cluster


# ==============================================================================
# setup

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 0

stan_iter = 2000


# ---- variables
Ns = c(10, 20, 40, 60, 100, 140, 200, 260)

# number of jobs
num_job = length(Ns)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
n_i = jobi + 1

n = Ns[n_i]

cat(sprintf('jobi=%d\n', jobi))
cat(sprintf('%s_%s_%s_%d_%d\n', truedist, modeldist, priordist, p0, n))


# ==============================================================================
# run the content

# load data in variable out
load(sprintf('res_looc/%s_%s_%s_%d_%d.RData',
    truedist, modeldist, priordist, p0, n))
# drop singleton dimensions
Niter = dim(out$loos)[2]
for (name in names(out)) {
    out[[name]] = drop(out[[name]])
}

# sample array
mus = array(0, c(stan_iter*2, Niter))
sigmas = array(0, c(stan_iter*2, Niter))
ls = array(0, c(stan_iter*2, Niter))
ps = array(0, c(stan_iter*2, Niter))
qs = array(0, c(stan_iter*2, Niter))

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

    loos_i = out$loos[,i]
    # center data (no scaling)
    loos_i_mean = mean(loos_i)
    loos_i = loos_i - loos_i_mean

    # skewed generalised t fit
    output <- capture.output(
        sgt_loo_fit <- stan(
            'models/sgt.stan',
            data=list(N=n, x=loos_i),
            iter=stan_iter, refresh=-1, save_warmup=FALSE, open_progress=FALSE
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
    "res_sgt/%s_%s_%s_%d_%d.RData",
    truedist, modeldist, priordist, p0, n
)
save(out, file=filename)
