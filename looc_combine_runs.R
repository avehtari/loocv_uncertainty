
library(ggplot2)
library(matrixStats)
library(extraDistr)

# number of runs to split the trials
run_tot = 20
# trials per run
Niter = 2000

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 1

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'


for (n_i in 1:length(Ns)) {

    n = Ns[n_i]

    print(sprintf('n=%d', n))

    # output array placeholder
    out_all = NULL

    Niter_per_run = rep(Niter %/% run_tot, run_tot)
    if ((Niter %% run_tot) > 0) {
        Niter_per_run[1:(Niter %% run_tot)] = (Niter %/% run_tot) + 1
    }

    for (run_i in 1:run_tot) {

        Niter_cur = Niter_per_run[run_i]
        if (run_i == 1) {
            Niters_before = 0
        } else {
            Niters_before = sum(Niter_per_run[1:(run_i-1)])
        }

        # load data in variable out
        load(sprintf('res_looc/parts/%s_%s_%s_%g_%g_%g',
            truedist, modeldist, priordist, p0, n, run_i))

        # create output array if not created already
        if (is.null(out_all)) {
            out_all = list()
            for (name in names(out)) {
                out_all[[name]] = array(
                    NA, c(dim(out[[name]])[1], Niter, dim(out[[name]])[3]))
            }
        }

        # slice indexes
        start_i = 1 + Niters_before
        end_i = Niters_before + dim(out$ltrs)[2]

        # fill output with the current slice
        for (name in names(out)) {
            out_all[[name]][,start_i:end_i,] = out[[name]]
        }

    }

    # rename out_all to out
    out = out_all
    # save
    filename = sprintf(
        "res_looc/%s_%s_%s_%d_%d.RData",
        truedist, modeldist, priordist, p0, n
    )
    save(out, file=filename)

}
