source('loo3_fun.R')

# ==============================================================================
# setup

# trials total
Niter = 2000
# num of test points
Nt = 20000
# number of runs to split the trials
run_tot = 20
# seed
seed = 11

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

p0 = 1

# ---- variables

# beta_ks = c(0.5, 1, 2, 3, 4)
beta_ks = c(0.5, 1, 2, 4)

Ns = c(10, 20, 50, 130, 250, 400)



# number of jobs
num_job = length(Ns) * length(beta_ks) * run_tot

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
beta_k_i = counter + 1


n = Ns[n_i]

beta_k = beta_ks[beta_k_i]
beta0 = beta_k
beta1 = beta_k
beta2 = 0.0


cat(sprintf('jobi=%d\n', jobi))
cat(sprintf(
    '%s_%s_%s_%g_%d_%d\n', truedist, modeldist, priordist, beta0, n, run_i))

# run the function
loo3_fun_one(
    truedist, modeldist, priordist, Niter, Nt, p0, n, beta0, beta1, beta2,
    run_tot, run_i, seed
)
