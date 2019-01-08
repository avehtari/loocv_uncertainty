
library(matrixStats)

library(loo)
library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores=1, loo.cores=1)

# truedist = 'n'
# modeldist = 'n'
# priordist = 'n'

# Niter = 2000
# Nt = 20000
# p0 = 0
# n = 20
# run_tot = 20
# run_i = 3
# seed = 11
# save_loo2=FALSE


looc_fun_one = function(
    truedist, modeldist, priordist, Niter, Nt, p0, n, run_tot, run_i, seed,
    save_loo2=FALSE, stan_iter=4000, stan_chains=4) {
    #' @param truedist true distribution identifier
    #' @param modeldist model distribution identifier
    #' @param priordist prior distribution identifier
    #' @param Niter number of trials
    #' @param Nt number of test points
    #' @param p0 number of input dimensions in M_0 (M_1 has p0+1 dim)
    #' @param n number of datapoints
    #' @param run_tot number of runs the iterations are splitted into
    #' @param run_i current run index
    #' @param seed seed to use
    #' @param save_loo2 bool if loo2 should be saved to out
    #' @param stan_iter number of stan iterations per chain
    #' @param stan_chains number of stan chains

    # config
    modelname = sprintf("models/linear_%s_%s.stan",modeldist,priordist)

    p1 = p0 + 1
    num_of_pairs = (n^2-n)/2

    set.seed(seed)
    if (truedist=="n") {
        yt <- as.array(rnorm(Nt))
        xt <- matrix(runif(p1*Nt,-1,1), nrow=Nt, ncol=p1)
    } else if (truedist=="t4") {
        yt <- as.array(rt(Nt,4))
        xt <- matrix(runif(p1*Nt,-1,1), nrow=Nt, ncol=p1)
    } else {
        stop("Unknown true distribution")
    }

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
        ltrs = array(NA, c(n, Niter_cur, 2)),
        loos = array(NA, c(n, Niter_cur, 2)),
        peff = array(NA, c(1, Niter_cur, 2)),
        pks = array(NA, c(n, Niter_cur, 2)),
        tls = array(NA, c(1, Niter_cur, 2)),
        ets = array(NA, c(1, Niter_cur, 2)),
        es = array(NA, c(1, Niter_cur, 2)),
        tes = array(NA, c(1, Niter_cur, 2)),
        mutrs = array(NA, c(n, Niter_cur, 2)),
        muloos = array(NA, c(n, Niter_cur, 2)),
        g2s = array(NA, c(1, Niter_cur, 2)),
        g2s_nod = array(NA, c(1, Niter_cur, 2)),
        g3s = array(NA, c(1, Niter_cur, 2)),
        g3s_nod = array(NA, c(1, Niter_cur, 2)),
        g2s_d = array(NA, c(1, Niter_cur)),
        g2s_nod_d = array(NA, c(1, Niter_cur)),
        g3s_d = array(NA, c(1, Niter_cur)),
        g3s_nod_d = array(NA, c(1, Niter_cur))
    )
    if (save_loo2) out$loo2 = array(NA, c(n, n, Niter_cur, 2))

    # loo2 placeholder
    loo2 = array(NA, c(n, n, 2))

    # iterate
    for (i1 in 1:Niter_cur) {
        print(sprintf(
            '%s%s%s p0=%d n=%d iter=%d',
            truedist, modeldist, priordist, p0, n, Niters_before+i1))

        # seed used by R for this trial
        trial_seed_r = seed+Niters_before+i1
        set.seed(trial_seed_r)
        # seed used by Stan for this trial
        trial_seed_stan = seed+Niter+Niters_before+i1

        # generate trial data
        if (truedist=="n") {
            y <- as.array(rnorm(n))
            x <- matrix(runif(p1*n,-1,1), nrow=n, ncol=p1)
        } else if (truedist=="t4") {
            y <- as.array(rt(n,4))
            x <- matrix(runif(p1*n,-1,1), nrow=n, ncol=p1)
        } else {
            stop("Unknown true distribution")
        }

        var_y = var(y)
        var_yt = var(yt)

        for (m_i in 1:2) {

            if (m_i == 1) {
                if (p0 == 0) {
                    data = list(
                        N=n, p=p0, x=array(0, c(n, 0)), y=y,
                        Nt=Nt, xt=array(0, c(Nt, 0)), yt=yt
                    )
                } else {
                    data = list(
                        N=n, p=p0, x=x[,1:p0, drop=FALSE], y=y,
                        Nt=Nt, xt=xt[,1:p0, drop=FALSE], yt=yt
                    )
                }
            } else {
                data = list(N=n, p=p1, x=x, y=y, Nt=Nt, xt=xt, yt=yt)
            }

            output <- capture.output(
                model1 <- stan(
                    modelname, data=data, iter=stan_iter, chains=stan_chains,
                    refresh=-1, save_warmup=FALSE, open_progress=FALSE,
                    seed=trial_seed_stan)
            )

            log_lik1 <- extract_log_lik(model1, merge_chains=FALSE)
            r_eff = relative_eff(exp(log_lik1))

            # broken in loo 2.0.0 ???
            # loo1 = loo(log_lik1, r_eff=r_eff, save_psis=TRUE)
            # out$loos[,i1,m_i] = loo1$pointwise[,1]

            # combine chains
            log_lik1 = array(
                log_lik1,
                c(dim(log_lik1)[1]*dim(log_lik1)[2], dim(log_lik1)[3])
            )

            out$ltrs[,i1,m_i] = log(colMeans(exp(log_lik1)))

            psis1 = psis(-log_lik1, r_eff=r_eff)
            psis1_lw = weights(psis1, normalize=TRUE, log=TRUE)
            out$loos[,i1,m_i] = colLogSumExps(log_lik1 + psis1_lw)

            mu1 = extract_log_lik(model1, parameter_name="mu")
            out$mutrs[,i1,m_i] = colMeans(mu1)
            out$muloos[,i1,m_i] = colSums(mu1*exp(psis1_lw))

            out$peff[,i1,m_i] = sum(out$ltrs[,i1,m_i]) - sum(out$loos[,i1,m_i])
            out$pks[,i1,m_i] = psis1$diagnostics$pareto_k

            log_likt_samp = extract_log_lik(model1, parameter_name="log_likt")
            log_likt = log(colMeans(exp(log_likt_samp)))
            out$tls[,i1,m_i] = mean(log_likt)*n
            mut = extract_log_lik(model1, parameter_name="mut")
            out$ets[,i1,m_i] = mean((y-out$mutrs[,i1,m_i])^2)/var_y
            out$es[,i1,m_i] = mean((y-out$muloos[,i1,m_i])^2)/var_y
            out$tes[,i1,m_i] = mean((yt-colMeans(mut))^2)/var_yt

            # free memory
            rm(model1, output, mut, log_likt_samp, mu1, log_likt, psis1)
            gc()


            # ====== loo2
            for (cvi in 1:n) {
                loo2[cvi,,m_i] = colLogSumExps(log_lik1+psis1_lw[,cvi])
            }
            if (save_loo2) out$loo2[,,i1,m_i] = loo2[,,m_i]
            qq = loo2[,,m_i]

            # g2s
            out$g2s[,i1,m_i] = mean(colVars(qq))

            # g3s
            g3s_s = array(0, num_of_pairs)
            # center columns
            qq_c = qq - rep(colMeans(qq), rep.int(nrow(qq), ncol(qq)))
            cur_pair_i = 1
            for (xi1 in 1:(n-1)) {
                for (xi2 in (xi1+1):n) {
                    g3s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                    cur_pair_i = cur_pair_i + 1
                }
            }
            out$g3s[,i1,m_i] = sum(g3s_s)/(num_of_pairs-1)

            # qq diag <- NA
            diag(qq) = NA

            # g2s_nod
            out$g2s_nod[,i1,m_i] = mean(
                apply(qq, 2, function(col) var(na.omit(col))))

            # g3s_nod
            g3s_s = array(0, num_of_pairs)
            # center columns (exclude diagonal)
            qq_c = qq - rep(
                apply(qq, 2, function(col) mean(na.omit(col))),
                rep.int(nrow(qq), ncol(qq))
            )
            cur_pair_i = 1
            for (xi1 in 1:(n-1)) {
                for (xi2 in (xi1+1):n) {
                    g3s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                    cur_pair_i = cur_pair_i + 1
                }
            }
            out$g3s_nod[,i1,m_i] = sum(g3s_s)/(num_of_pairs-1)

            # free memory
            rm(log_lik1, psis1_lw)
            gc()

        }

        # ====== loo2 diff
        qq = loo2[,,1] - loo2[,,2]

        # g2s
        out$g2s_d[,i1] = mean(colVars(qq))

        # g3s
        g3s_s = array(0, num_of_pairs)
        # center columns
        qq_c = qq - rep(colMeans(qq), rep.int(nrow(qq), ncol(qq)))
        cur_pair_i = 1
        for (xi1 in 1:(n-1)) {
            for (xi2 in (xi1+1):n) {
                g3s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                cur_pair_i = cur_pair_i + 1
            }
        }
        out$g3s_d[,i1] = sum(g3s_s)/(num_of_pairs-1)

        # qq diag <- NA
        diag(qq) = NA

        # g2s_nod
        out$g2s_nod_d[,i1] = mean(
            apply(qq, 2, function(col) var(na.omit(col))))

        # g3s_nod
        g3s_s = array(0, num_of_pairs)
        # center columns (exclude diagonal)
        qq_c = qq - rep(
            apply(qq, 2, function(col) mean(na.omit(col))),
            rep.int(nrow(qq), ncol(qq))
        )
        cur_pair_i = 1
        for (xi1 in 1:(n-1)) {
            for (xi2 in (xi1+1):n) {
                g3s_s[cur_pair_i] = qq_c[xi1,xi2]*qq_c[xi2,xi1]
                cur_pair_i = cur_pair_i + 1
            }
        }
        out$g3s_nod_d[,i1] = sum(g3s_s)/(num_of_pairs-1)

    }

    # save
    filename = sprintf(
        "res_looc/parts/%s_%s_%s_%d_%d_%d",
        truedist, modeldist, priordist, p0, n, run_i
    )
    save(out, file=filename)

}
