source('loo_fun.R')

# ==============================================================================
# setup

# trials per run
Niter = 1023
# num of test samples
Nt = 34000
# number of test groups for rank statistics (at most Nt/n)
Ntg = 100
# seed
seed = 11

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# ---- variables
Ps = c(1, 2, 5, 10)
Ns = c(10, 20, 40, 60, 100, 140, 200, 260, 340)

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

# run the function
loo_fun_one(truedist, modeldist, priordist, Niter, Nt, Ps[p_i], n, Ntg, seed)
