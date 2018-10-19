source('loocomp_fun.R')

# ==============================================================================
# setup

# trials per run
Niter = 200
# num of test samples
Nt = 10000

# bayesian bootstrap
bbsamples = 2000
bbalpha = 1.0

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# ---- variables
Ps = c(0, 1, 2, 5, 10, 20)
Ns = c(10, 20, 40, 60, 100, 140, 200)
# array of slope coefficients beta_{p+1}
betas = 2^seq(-6, 2)

# number of jobs (378)
num_job = length(Ps) * length(Ns) * length(betas)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
beta_i = (jobi %% length(betas)) + 1
quo = jobi %/% length(betas)
n_i = (quo %% length(Ns)) + 1
p_i = (quo %/% length(Ns)) + 1

n = Ns[n_i]


message(sprintf('jobi=%d', jobi))
message(sprintf(
    'truedist=%s, modeldist=%s, priordist=%s,\nNiter=%d, Nt=%d, p=%d, n=%d, beta_i=%d',
    truedist, modeldist, priordist, Niter, Nt, Ps[p_i], n, beta_i))
# sprintf('%s', betas)

# run the function
loocomp_fun_one(truedist, modeldist, priordist, Ps[p_i], n, Niter, Nt, betas, beta_i, bbsamples, bbalpha)
