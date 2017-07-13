library(loo)
library(matrixStats)
library(extraDistr)

library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores=1, loo.cores=1)

# truedist = 'n'
# modeldist = 'n'
# priordist = 'n'

# truedist = 'b'
# modeldist = 'b'
# priordist = 'n'

# n = 5
# p = 2
# Niter = 3
# Nt = 7
# betas = 2^seq(-6, 3)
# beta_i = 6
# bbsamples = 123
# bbalpha = 1.0


loocomp_fun_one = function(
        truedist, modeldist, priordist, n, p, Niter, Nt, betas, beta_i,
        bbsamples, bbalpha) {
    #' @param n number of datapoints
    #' @param p number of input dimensions in M_0 (M_1 has p+1 dim)
    #' @param Niter number of trials
    #' @param Nt number of test points
    #' @param betas array of slope coefficients beta_{p+1}
    #' @param beta_i the index of the current beta
    #' @param bbsamples number of bayesian bootstrap samples
    #' @param bbalpha parameter alpha for bayesian bootstrap

    p1 = p + 1
    modelname = sprintf("linear_%s_%s.stan", modeldist, priordist)
    beta = betas[beta_i]

    # allocate output
    out = list(
        # ==== shared general
        beta = beta,
        bbsamples = bbsamples,
        bbalpha = bbalpha,
        # ==== for model 0 and 1 and for each beta
        ltrs = array(NaN, c(n, Niter, 2)),
        loos = array(NaN, c(n, Niter, 2)),
        peff = array(NaN, c(Niter, 2)),
        pks = array(NaN, c(n, Niter, 2)),
        tls = array(NaN, c(Niter, 2)),
        mutrs =
            if (truedist=="b") array(NaN, c(2*n, Niter, 2))
            else array(NaN, c(n, Niter, 2)),
        muloos =
            if (truedist=="b") array(NaN, c(2*n, Niter, 2))
            else array(NaN, c(n, Niter, 2)),
        ets = array(NaN, c(Niter, 2)),
        es = array(NaN, c(Niter, 2)),
        tes = array(NaN, c(Niter, 2)),
        # ==== for model 1
        beta_pos = array(NaN, c(Niter)),
        # ==== compares
        # comp = array(NaN, c(Niter)),  # added in the end
        compt = array(NaN, c(Niter))
    )

    set.seed(1)
    if (truedist=="n") {
        xt = matrix(runif(p1*Nt, -1, 1), nrow=Nt, ncol=p1)
        yt = as.array(rnorm(Nt)) + beta*xt[,p1]
    } else if (truedist=="t4") {
        xt = matrix(runif(p1*Nt, -1, 1), nrow=Nt, ncol=p1)
        yt = as.array(rt(Nt, 4)) + beta*xt[,p1]
    } else if (truedist=="b") {
        stop("Not implemented")
        # xt = matrix(runif(p1*Nt*2, -1, 1), nrow=Nt*2, ncol=p)
        # yt = as.array(as.double(kronecker(matrix(1, 1, Nt), t(c(0, 1)))))
    } else {
        stop("Unknown true distribution")
    }

    # work arrays
    log_likts = array(NaN, c(Nt, 2))
    # bb samples for test
    bbst = rdirichlet(bbsamples, rep(bbalpha, Nt))

    # iterate
    for (i1 in 1:Niter) {
        print(sprintf(
            '%s%s%s p=%d n=%d beta=%g iter=%d',
            truedist, modeldist, priordist, p, n, beta, i1))
        set.seed(i1)
        if (truedist=="n") {
            x = matrix(runif(p1*n, -1, 1), nrow=n, ncol=p1)
            y = as.array(rnorm(n)) + beta*x[,p1]
        } else if (truedist=="t4") {
            x = matrix(runif(p1*n, -1, 1), nrow=n, ncol=p1)
            y = as.array(rt(n, 4)) + beta*x[,p1]
        } else if (truedist=="b") {
            stop("Not implemented")
            # x = matrix(runif(p1*n*2, -1, 1), nrow=n*2, ncol=p)
            # y = as.array(as.double(kronecker(matrix(1, 1, n), t(c(0, 1)))))
        } else {
            stop("Unknown true distribution")
        }

        for (m_i in 1:2) {

            if (m_i == 1) {
                if (p == 0) {
                    data = list(
                        N=n, p=p, x=array(0, c(n, 0)), y=y,
                        Nt=Nt, xt=array(0, c(Nt, 0)), yt=yt)
                } else {
                    data = list(
                        N=n, p=p, x=x[,1:p, drop=FALSE], y=y,
                        Nt=Nt, xt=xt[,1:p, drop=FALSE], yt=yt)
                }
            } else {
                data = list(N=n, p=p1, x=x, y=y, Nt=Nt, xt=xt, yt=yt)
            }

            output <- capture.output(
                model <- stan(
                    modelname, data=data, iter=1000, refresh=-1,
                    save_warmup=FALSE, open_progress=FALSE
                )
            )

            log_lik = extract_log_lik(model)
            loo = loo(log_lik)
            out$ltrs[,i1,m_i] = log(colMeans(exp(log_lik)))
            out$loos[,i1,m_i] = loo$pointwise[,1]
            out$peff[i1,m_i] = (
                sum(out$ltrs[,i1,m_i]) -
                sum(out$loos[,i1,m_i]))
            out$pks[,i1,m_i] = loo$pareto_k

            log_likt = extract_log_lik(model, parameter_name="log_likt")
            log_likts[,m_i] = log(colMeans(exp(log_likt)))
            out$tls[i1,m_i] = mean(log_likts[,m_i])*n

            mu = extract_log_lik(model, parameter_name="mu")
            out$mutrs[,i1,m_i] = colMeans(mu)
            psis = psislw(-log_lik)
            if (truedist=="b") {
                stop("Not implemented")
                # out$muloos[seq(1,n*2,2),i1,m_i] = colSums(
                #     mu1[,seq(1,n*2,2)]*exp(psis1$lw_smooth))
                # out$muloos[seq(2,n*2,2),i1,m_i] = colSums(
                #     mu1[,seq(2,n*2,2)]*exp(psis1$lw_smooth))
            } else {
                out$muloos[,i1,m_i] = colSums(mu*exp(psis$lw_smooth))
            }

            mut = extract_log_lik(model, parameter_name="mut")
            if (truedist=="b") {
                stop("Not implemented")
                # out$ets[i1,m_i] = (
                #     mean(xor(out$mutrs[,i1,m_i]>0,(y>0))))
                # out$es[i1,m_i] = (
                #     mean(xor(out$muloos[,i1,m_i]>0,(y>0))))
                # out$tes[i1,m_i] = (
                #     mean(xor(colMeans(mut)>0,(yt>0))))
            } else {
                out$ets[i1,m_i] = (
                    mean((y-out$mutrs[,i1,m_i])^2)/var(y))
                out$es[i1,m_i] = (
                    mean((y-out$muloos[,i1,m_i])^2)/var(y))
                out$tes[i1,m_i] = (
                    mean((yt-colMeans(mut))^2)/var(yt))
            }

            if (m_i == 2) {
                # inspect posterior probability for positive slope of p+1
                samp_b = extract(model, pars='beta')$beta
                out$beta_pos[i1] = mean(samp_b[,p1] > 0)
                rm(samp_b)
            }

            # clear some memory
            rm(model, output, log_lik, loo, mu, psis, log_likt, mut)
            gc()
        }

        # compare test samples
        dtest = log_likts[,2] - log_likts[,1]
        s = bbst %*% dtest
        out$compt[i1] = sum(s > 0) / bbsamples
        rm(dtest, s)
        gc()

    }

    # clear bbst
    rm(bbst)
    gc()

    # ==== compare loos
    # bb samples
    bbs = rdirichlet(bbsamples, rep(bbalpha, n))
    dloos = out$loos[,,2] - out$loos[,,1]
    s = bbs %*% dloos
    out$comp = colSums(s > 0) / bbsamples

    # ==== save results
    filename = sprintf(
        "res_comp/%s_%s_%s_%d_%d_%d.RData",
        truedist, modeldist, priordist, p, n, beta_i)
    save(out, file=filename)
}
