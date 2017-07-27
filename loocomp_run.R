source('loocomp_fun.R')

# trials per run
Niter = 200
# num of test samples
Nt = 10000
# array of slope coefficients beta_{p+1}
betas = 2^seq(-6, 2)

# bayesian bootstrap
bbsamples = 4000
bbalpha = 1.0


# possible parameters
dists = list(
    c('n', 'n', 'n'),
    c('t4', 'tnu', 'n'),
    c('b', 'b', 'n'),
    c('n', 'tnu', 'n'),
    c('t4', 'n', 'n')
)
Ns<-c(10, 20, 40, 60, 100, 140, 200, 260)
Ps<-c(0, 1, 2, 5, 10, 20)

# number of jobs (240)
num_job = length(dists) * length(Ns) * length(Ps) * length(betas)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
beta_i = jobi %% length(betas)
quo = jobi %/% length(betas)
p_i = quo %% length(Ps)
quo = quo %/% length(Ps)
n_i = quo %% length(Ns)
dist_i = quo %/% length(Ns)

truedist = dists[[dist_i+1]][1]
modeldist = dists[[dist_i+1]][2]
priordist = dists[[dist_i+1]][3]
n = Ns[n_i+1]
p = Ps[p_i+1]
beta_i = beta_i + 1

message(sprintf('jobi=%d', jobi))
message(sprintf(
    'truedist=%s, modeldist=%s, priordist=%s,\nn=%d, p=%d, Niter=%d, Nt=%d, beta_i=%d',
    truedist, modeldist, priordist, n, p, Niter, Nt, beta_i))
# sprintf('%s', betas)

# run the function
loocomp_fun_one(truedist, modeldist, priordist, n, p, Niter, Nt, betas, beta_i, bbsamples, bbalpha)
