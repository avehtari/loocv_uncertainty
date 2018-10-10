
library(matrixStats)

library(loo)
library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores=1, loo.cores=1)

# truedist = 'n'
# modeldist = 'n'
# priordist = 'n'

# Niter = 10
# Ntx = 200
# p = 2
# n = 10
# Ntg = 9
# Ntgs = 12
# run_tot = 4
# run_i = 3
# seed = 11
# fallback_k = F


loo_fun_one = function(
    truedist, modeldist, priordist, Niter, Ntx, p, n, Ntg, Ntgs, run_tot, run_i,
    seed, fallback_k) {
    #' @param truedist true distribution identifier
    #' @param modeldist model distribution identifier
    #' @param priordist prior distribution identifier
    #' @param Niter number of trials
    #' @param Ntx number of test points is ``Nt = Ntx*n``
    #' @param p number of input dimensions in M_0 (M_1 has p+1 dim)
    #' @param n number of datapoints
    #' @param Ntg number of test groups for rank statistics
    #' @param Ntgs number of test samples for each group for rank statistics
    #' @param run_tot number of runs the iterations are splitted into
    #' @param run_i current run index
    #' @param seed seed to use
    #' @param fallback_k fall back to not using PSIS with large k

    Nt = Ntx*n

    if (Ntg*Ntgs > Ntx) {
        stop("``Ntg*Ntgs`` must be equal or smaller than `Ntx`")
    }

    # config
    modelname = sprintf("models/linear_%s_%s.stan",modeldist,priordist)

    set.seed(seed)
    if (truedist=="n") {
        yt <- as.array(rnorm(Nt))
        xt <- matrix(runif(p*Nt,-1,1),nrow=Nt,ncol=p)
    } else if (truedist=="t4") {
        yt <- as.array(rt(Nt,4))
        xt <- matrix(runif(p*Nt,-1,1),nrow=Nt,ncol=p)
    } else if (truedist=="b") {
        yt <- as.array(as.double(kronecker(matrix(1,1,Nt),t(c(0,1)))))
        xt <- matrix(runif(p*Nt*2,-1,1),nrow=Nt*2,ncol=p)
    } else {
        stop("Unknown true distribution")
    }

    num_of_pairs = (n^2-n)/2

    # splitting the iterations
    if (run_i > run_tot) {
        stop("`run_i` must be equal or smaller than `run_tot`")
    }
    Niter_per_run = rep(Niter %/% run_tot, run_tot)
    if ((Niter %% run_tot) > 0) {
        Niter_per_run[1:(Niter %% run_tot)] = (Niter %/% run_tot) + 1
    }
    Niter_cur = Niter_per_run[run_i]
    if (run_i == 1) {
        Niters_before = 0
    } else {
        Niters_before = sum(Niter_per_run[1:(run_i-1)])
    }


    # allocate output
    out = list(
        ltrs = matrix(nrow=n, ncol=Niter_cur),
        loos = matrix(nrow=n, ncol=Niter_cur),
        kexeeds = vector("list", Niter_cur),
        looks = vector("list", Niter_cur),
        peff = matrix(nrow=1, ncol=Niter_cur),
        pks = matrix(nrow=n, ncol=Niter_cur),
        tls = matrix(nrow=1, ncol=Niter_cur),
        ets = matrix(nrow=1, ncol=Niter_cur),
        es = matrix(nrow=1, ncol=Niter_cur),
        tes = matrix(nrow=1, ncol=Niter_cur),
        # lls = vector("list", length(Niter_cur)),
        mutrs =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter_cur)
            else matrix(nrow=n, ncol=Niter_cur),
        muloos =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter_cur)
            else matrix(nrow=n, ncol=Niter_cur),
        mulooks = vector("list", Niter_cur),
        bs = matrix(nrow=1, ncol=Niter_cur),
        gs = matrix(nrow=1, ncol=Niter_cur),
        gms = matrix(nrow=1, ncol=Niter_cur),
        g2s = matrix(nrow=1, ncol=Niter_cur),
        gm2s = matrix(nrow=1, ncol=Niter_cur),
        g2s_new = matrix(nrow=1, ncol=Niter_cur),
        g2s_new2 = matrix(nrow=1, ncol=Niter_cur),
        loovar1 = matrix(nrow=1, ncol=Niter_cur),
        loovar2 = matrix(nrow=1, ncol=Niter_cur),
        loovar3 = matrix(nrow=1, ncol=Niter_cur),
        loovar4 = matrix(nrow=1, ncol=Niter_cur),
        loovar5 = matrix(nrow=1, ncol=Niter_cur),
        loovar1_rank = matrix(nrow=1, ncol=Niter_cur),
        loovar2_rank = matrix(nrow=1, ncol=Niter_cur),
        loovar3_rank = matrix(nrow=1, ncol=Niter_cur),
        loovar4_rank = matrix(nrow=1, ncol=Niter_cur),
        loovar5_rank = matrix(nrow=1, ncol=Niter_cur),
        test_target = matrix(nrow=1, ncol=Niter_cur)
    )

    # iterate
    for (i1 in 1:Niter_cur) {
        print(sprintf(
            '%s%s%s p=%d n=%d iter=%d',
            truedist, modeldist, priordist, p, n, Niters_before+i1))
        set.seed(seed+Niters_before+i1)
        if (truedist=="n") {
            y <- as.array(rnorm(n))
            x <- matrix(runif(p*n,-1,1),nrow=n,ncol=p)
        } else if (truedist=="t4") {
            y <- as.array(rt(n,4))
            x <- matrix(runif(p*n,-1,1),nrow=n,ncol=p)
        } else if (truedist=="b") {
            y <- as.array(as.double(kronecker(matrix(1,1,n),t(c(0,1)))))
            x <- matrix(runif(p*n*2,-1,1),nrow=n*2,ncol=p)
        } else {
            stop("Unknown true distribution")
        }
        data = list(N=n, p=ncol(x), x=x, y=y, Nt=Nt, xt=xt, yt=yt)
        output <- capture.output(
            model1 <- stan(
                modelname, data=data, iter=1000, refresh=-1, save_warmup=FALSE,
                open_progress=FALSE)
        )
        log_lik1 = extract_log_lik(model1)
        # out$lls[[i1]] = log_lik1
        psis1 = psis(-log_lik1)
        psis1_lw = weights(psis1, normalize = TRUE, log = TRUE)
        mu1 = extract_log_lik(model1, parameter_name="mu")
        out$mutrs[,i1] = colMeans(mu1)
        if (truedist=="b") {
            out$muloos[seq(1,n*2,2),i1] = colSums(
                mu1[,seq(1,n*2,2)]*exp(psis1_lw))
            out$muloos[seq(2,n*2,2),i1] = colSums(
                mu1[,seq(2,n*2,2)]*exp(psis1_lw))
        } else {
            out$muloos[,i1] = colSums(mu1*exp(psis1_lw))
        }
        out$ltrs[,i1] = log(colMeans(exp(log_lik1)))
        # loo1 = loo(log_lik1)  # broken in loo 2.0.0 ???
        # out$loos[,i1] = loo1$pointwise[,1]
        out$loos[,i1] = colLogSumExps(log_lik1 + psis1_lw)
        out$peff[,i1] = sum(out$ltrs[,i1]) - sum(out$loos[,i1])
        out$pks[,i1] = psis1$diagnostics$pareto_k
        log_likt_samp = extract_log_lik(model1, parameter_name="log_likt")
        log_likt = log(colMeans(exp(log_likt_samp)))
        out$tls[,i1] = mean(log_likt)*n
        mut = extract_log_lik(model1, parameter_name="mut")
        if (truedist=="b") {
            out$ets[,i1] = mean(xor(out$mutrs[,i1]>0,(y>0)))
            out$es[,i1] = mean(xor(out$muloos[,i1]>0,(y>0)))
            out$tes[,i1] = mean(xor(colMeans(mut)>0,(yt>0)))
        } else {
            out$ets[,i1] = mean((y-out$mutrs[,i1])^2)/var(y)
            out$es[,i1] = mean((y-out$muloos[,i1])^2)/var(y)
            out$tes[,i1] = mean((yt-colMeans(mut))^2)/var(yt)
        }

        # find in which points pareto k exeeds treshold
        kexeeds = which(psis1$diagnostics$pareto_k > 0.7)
        out$kexeeds[[i1]] = kexeeds

        # free memory
        rm(model1, output, mut, log_likt_samp)
        # rm(loo1)
        gc()

        qq = matrix(nrow=n, ncol=n)
        qqm = matrix(nrow=n, ncol=n)
        for (cvi in 1:n) {
            qq[cvi,] = colLogSumExps(log_lik1+psis1_lw[,cvi])
            if (truedist=="b") {
                qqm[cvi,] = as.numeric(xor(
                    colSums(mu1[,seq(1,n*2,2)]*exp(psis1_lw[,cvi])) > 0,
                    y[seq(1,n*2,2)]
                ))
                qqm[cvi,] = qqm[cvi,] + as.numeric(xor(
                    colSums(mu1[,seq(2,n*2,2)]*exp(psis1_lw[,cvi])) > 0,
                    y[seq(2,n*2,2)]
                ))
                qqm[cvi,] = 0.5 * qqm[cvi,]
            } else {
                qqm[cvi,] = (y-colSums(mu1*exp(psis1_lw[,cvi])))^2
            }
        }
        gammas = matrix(nrow=1, ncol=n)
        gammams = matrix(nrow=1, ncol=n)
        for (cvi in 1:n) {
            gammas[,cvi] = var(qq[-cvi,cvi]);
            gammams[,cvi] = var(qqm[-cvi,cvi]);
        }
        betas = matrix(nrow=1, ncol=n)
        mbetas = matrix(nrow=1, ncol=n)
        for (cvi in 1:n) {
            betas[,cvi] = mean(qq[-cvi,cvi]);
            mbetas[,cvi] = mean(qqm[-cvi,cvi]);
        }
        gamma = mean(gammas);
        gammam = mean(gammams);
        gamma2 = mean(colVars(qq));
        gammam2 = mean(colVars(qqm));
        out$bs[,i1] = sum(betas)
        out$gs[,i1] = gamma
        out$gms[,i1] = gammam
        out$g2s[,i1] = gamma2
        out$gm2s[,i1] = gammam2

        # g2s_new
        g2s_s = array(0, num_of_pairs)
        # center columns
        qq_c = qq - rep(colMeans(qq), rep.int(nrow(qq), ncol(qq)))
        cur_pair_i = 1
        for (xi1 in 1:(n-1)) {
            for (xi2 in (xi1+1):n) {
                g2s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                cur_pair_i = cur_pair_i + 1
            }
        }
        out$g2s_new[,i1] = sum(g2s_s)/(num_of_pairs-1)

        # qq diag <- nan
        diag(qq) = NA

        # g2s_new2
        g2s_s = array(0, num_of_pairs)
        # center columns (exclude diagonal)
        qq_c = qq - rep(
            apply(qq, 2, function(col) mean(na.omit(col))),
            rep.int(nrow(qq), ncol(qq))
        )
        cur_pair_i = 1
        for (xi1 in 1:(n-1)) {
            for (xi2 in (xi1+1):n) {
                g2s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                cur_pair_i = cur_pair_i + 1
            }
        }
        out$g2s_new2[,i1] = sum(g2s_s)/(num_of_pairs-1)

        # loo estimates
        # -------------

        # test target (from all the test samples)
        out$test_target[,i1] = var(colSums(array(log_likt, c(n, Ntx))))

        # naive
        out$loovar1[,i1] = n*var(out$loos[,i1])
        # conservative 2x
        out$loovar2[,i1] = 2*out$loovar1[,i1]
        # g2s
        out$loovar3[,i1] = out$loovar1[,i1] + n*out$g2s[,i1]
        out$loovar4[,i1] = out$loovar1[,i1] + (n^2)*out$g2s_new[,i1]
        out$loovar5[,i1] = out$loovar1[,i1] + (n^2)*out$g2s_new2[,i1]

        # ranks
        # form test set variances
        rank_test_nvars = colVars(colSums(array(log_likt, c(n, Ntgs, Ntg))))
        # calc rank for each estimate
        out$loovar1_rank[,i1] = sum(rank_test_nvars < out$loovar1[,i1])
        out$loovar2_rank[,i1] = sum(rank_test_nvars < out$loovar2[,i1])
        out$loovar3_rank[,i1] = sum(rank_test_nvars < out$loovar3[,i1])
        out$loovar4_rank[,i1] = sum(rank_test_nvars < out$loovar4[,i1])
        out$loovar5_rank[,i1] = sum(rank_test_nvars < out$loovar5[,i1])

        # free memory
        rm(log_lik1, log_likt, psis1, psis1_lw, mu1)
        gc()

        # k_exeeds
        # --------

        n_kexeeds = length(kexeeds)
        if (n_kexeeds > 0 && fallback_k) {
            print("k>0.7")
            print(kexeeds)
            # reprocess probelmatic points
            out$looks[[i1]] = array(NaN, n_kexeeds)
            out$mulooks[[i1]] = (
                if (truedist=="b") array(NaN, 2*n_kexeeds)
                else array(NaN, n_kexeeds)
            )
            for (ki in 1:n_kexeeds) {
                cvi = kexeeds[ki]
                if (truedist == "b") {
                    data <- list(
                        N = n-1,
                        p = ncol(x),
                        x = as.matrix(x[-c(cvi*2-1,cvi*2),]),
                        y = y[-c(cvi*2-1,cvi*2)],
                        Nt = 1,
                        xt = matrix(data=x[c(cvi*2-1,cvi*2),],nrow=2,ncol=p),
                        yt = as.array(y[c(cvi*2-1,cvi*2)])
                    )
                } else {
                    data = list(
                        N = n-1,
                        p = ncol(x),
                        x = as.matrix(x[-cvi,]),
                        y = y[-cvi],
                        Nt = 1,
                        xt = matrix(data=x[cvi,],nrow=1,ncol=p),
                        yt = as.array(y[cvi])
                    )
                }
                output <- capture.output(
                    modelcv <- stan(
                        modelname, data=data, iter=1000, refresh=-1,
                        save_warmup = FALSE, open_progress = FALSE)
                )
                out$looks[[i1]][ki] = log(colMeans(exp(
                    extract_log_lik(modelcv, parameter_name="log_likt"))))
                if (truedist=="b") {
                    out$mulooks[[i1]][((ki-1)*2+1):((ki-1)*2+2)] = colMeans(
                        extract_log_lik(modelcv, parameter_name="mut"))
                } else {
                    out$mulooks[[i1]][ki] = colMeans(
                        extract_log_lik(modelcv, parameter_name="mut"))
                }
                # free memory
                rm(modelcv, output)
                gc()
            }
        }

    }

    # save to .rdata
    if (TRUE) {
        filename = sprintf(
            "res_loo/parts/%s_%s_%s_%d_%d_%d",
            truedist, modeldist, priordist, p, n, run_i
        )
        save(out, file=filename)
    }

    # save with feather
    if (FALSE) {
        saved_names = c(
            'ltrs',
            'loos',
            # 'looks',
            'peff',
            'pks',
            'tls',
            'ets',
            'es',
            'tes',
            # 'lls',
            'mutrs',
            'muloos',
            # 'mulooks',
            'bs',
            'gs',
            'gms',
            'g2s',
            'gm2s',
            'g2s_new',
            'g2s_new2',
            'loovar1',
            'loovar2',
            'loovar3',
            'loovar4',
            'loovar5',
            'loovar1_rank',
            'loovar2_rank',
            'loovar3_rank',
            'loovar4_rank',
            'loovar5_rank',
            'test_target'
        )
        res_dir = sprintf(
            "res_loo/parts/%s_%s_%s_%d_%d_%d",
            truedist, modeldist, priordist, p, n, run_i
        )
        dir.create(res_dir)
        for (saved_name in saved_names) {
            write_feather(
                as.data.frame(`$`(out, saved_name)),
                file.path(res_dir, paste(saved_name, '.feather', sep=''))
            )
        }
    }

}
