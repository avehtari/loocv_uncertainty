source('loo_fun.R')

# ==============================================================================
# setup

# trials total
Niter = 2000
# num of test points is ``Nt = Ntx*n``
Ntx = 2000
# number of test groups for rank statistics
# (Ntg*Ntgs at most Ntx)
Ntg = 9
# number of test samples per group for rank statistics
Ntgs = 222
# number of runs to split the trials
run_tot = 25
# seed
seed = 11
# fall back to not using PSIS with large k
fallback_k = F

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

# ---- variables
Ps = c(1, 2, 5, 10)
Ns = c(10, 20, 40, 60, 100, 140, 200, 260)

# number of jobs
num_job = length(Ps) * length(Ns) * run_tot

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
counter = jobi
run_i = (counter %% run_tot) + 1
counter = counter %/% run_tot
n_i = (counter %% length(Ns)) + 1
counter = counter %/% length(Ns)
p_i = counter + 1

n = Ns[n_i]

cat(sprintf('jobi=%d\n', jobi))
cat(sprintf('%s_%s_%s_%d_%d\n', truedist, modeldist, priordist, Ps[p_i], n))

# run the function
loo_fun_one(
    truedist, modeldist, priordist, Niter, Ntx, Ps[p_i], n, Ntg, Ntgs,
    run_tot, run_i, seed, fallback_k
)
