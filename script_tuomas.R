source('paral_fun.R')

# config iterations per run
Niter = 100

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

# number of jobs (140)
num_job = length(dists) * length(Ns) * length(Ps)

# get job number [0, num_job-1] as command line argument
jobi = as.numeric(commandArgs(trailingOnly = TRUE)[1])
if (jobi >= num_job)
    stop(sprintf("Jobi must be smaller than %d", num_job))

# convert jobi to parameters
# (there should be a function to calc quotient and remainder at the same time)
p_i = jobi %% length(Ps)
quo = jobi %/% length(Ps)
n_i = quo %% length(Ns)
dist_i = quo %/% length(Ns)

truedist = dists[[dist_i+1]][1]
modeldist = dists[[dist_i+1]][2]
priordist = dists[[dist_i+1]][3]
n = Ns[n_i+1]
p = Ps[p_i+1]

sprintf('jobi=%d', jobi)
sprintf('%s, %s, %s, %d, %d', truedist, modeldist, priordist, n, p)

# run the function
loocomp_fun_one(truedist, modeldist, priordist, n, p, Niter)
