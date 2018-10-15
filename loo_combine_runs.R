
library(ggplot2)
library(matrixStats)
library(extraDistr)

# number of runs to split the trials
run_tot = 20
# trials per run
Niter = 2000

Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
Ps = c(1, 2, 5, 10)

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

p_i = 1
p = Ps[p_i]


for (n_i in 1:length(Ns)) {

    n = Ns[n_i]

    # output array
    out_all = list(
        ltrs = matrix(nrow=n, ncol=Niter),
        loos = matrix(nrow=n, ncol=Niter),
        loo2s = matrix(nrow=n, ncol=Niter),
        kexeeds = vector("list", Niter),
        looks = vector("list", Niter),
        peff = matrix(nrow=1, ncol=Niter),
        pks = matrix(nrow=n, ncol=Niter),
        tls = matrix(nrow=1, ncol=Niter),
        ets = matrix(nrow=1, ncol=Niter),
        es = matrix(nrow=1, ncol=Niter),
        tes = matrix(nrow=1, ncol=Niter),
        # lls = vector("list", Niter),
        mutrs =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter)
            else matrix(nrow=n, ncol=Niter),
        muloos =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter)
            else matrix(nrow=n, ncol=Niter),
        mulooks = vector("list", Niter),
        g2s = matrix(nrow=1, ncol=Niter),
        g2s_new = matrix(nrow=1, ncol=Niter),
        g2s_new2 = matrix(nrow=1, ncol=Niter),
        loovar1 = matrix(nrow=1, ncol=Niter),
        loovar2 = matrix(nrow=1, ncol=Niter),
        loovar3 = matrix(nrow=1, ncol=Niter),
        loovar4 = matrix(nrow=1, ncol=Niter),
        loovar1_rank = matrix(nrow=1, ncol=Niter),
        loovar2_rank = matrix(nrow=1, ncol=Niter),
        loovar3_rank = matrix(nrow=1, ncol=Niter),
        loovar4_rank = matrix(nrow=1, ncol=Niter),
        test_target = matrix(nrow=1, ncol=Niter)
    )


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
        load(sprintf('res_loo/parts/%s_%s_%s_%g_%g_%g',
            truedist, modeldist, priordist, Ps[p_i], n, run_i))

        start_i = 1 + Niters_before
        end_i = Niters_before + dim(out$ltrs)[2]

        out_all$ltrs[,start_i:end_i] = out$ltrs
        out_all$loos[,start_i:end_i] = out$loos
        out_all$loo2s[,,start_i:end_i] = out$loo2s
        out_all$kexeeds[start_i:end_i] = out$kexeeds
        out_all$looks[start_i:end_i] = out$looks
        out_all$peff[,start_i:end_i] = out$peff
        out_all$pks[,start_i:end_i] = out$pks
        out_all$tls[,start_i:end_i] = out$tls
        out_all$ets[,start_i:end_i] = out$ets
        out_all$es[,start_i:end_i] = out$es
        out_all$tes[,start_i:end_i] = out$tes
        out_all$mutrs[,start_i:end_i] = out$mutrs
        out_all$muloos[,start_i:end_i] = out$muloos
        out_all$mulooks[start_i:end_i] = out$mulooks
        out_all$g2s[,start_i:end_i] = out$g2s
        out_all$g2s_new[,start_i:end_i] = out$g2s_new
        out_all$g2s_new2[,start_i:end_i] = out$g2s_new2
        out_all$loovar1[,start_i:end_i] = out$loovar1
        out_all$loovar2[,start_i:end_i] = out$loovar2
        out_all$loovar3[,start_i:end_i] = out$loovar3
        out_all$loovar4[,start_i:end_i] = out$loovar4
        out_all$loovar1_rank[,start_i:end_i] = out$loovar1_rank
        out_all$loovar2_rank[,start_i:end_i] = out$loovar2_rank
        out_all$loovar3_rank[,start_i:end_i] = out$loovar3_rank
        out_all$loovar4_rank[,start_i:end_i] = out$loovar4_rank
        out_all$test_target[,start_i:end_i] = out$test_target

    }

    # rename out_all to out
    out = out_all
    # save
    filename = sprintf(
        "res_loo/%s_%s_%s_%d_%d.RData",
        truedist, modeldist, priordist, p, n
    )
    save(out, file=filename)

}
