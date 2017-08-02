
library(matrixStats)

library(loo)
library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores=1, loo.cores=1)

# truedist = 'n'
# modeldist = 'n'
# priordist = 'n'

# truedist = 'b'
# modeldist = 'b'
# priordist = 'n'

# n = 10
# p = 2
# Niter = 4

loo_fun_one = function(truedist, modeldist, priordist, Niter, Nt, p, n) {
    #' @param truedist true distribution identifier
    #' @param modeldist model distribution identifier
    #' @param priordist prior distribution identifier
    #' @param Niter number of trials
    #' @param Nt number of test points
    #' @param p number of input dimensions in M_0 (M_1 has p+1 dim)
    #' @param n number of datapoints

    # config
    modelname = sprintf("models/linear_%s_%s.stan",modeldist,priordist)

    set.seed(1)
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

    # allocate output
    out = list(
        ltrs = matrix(nrow=n, ncol=Niter),
        loos = matrix(nrow=n, ncol=Niter),
        # looks = matrix(nrow=n, ncol=Niter),
        peff = matrix(nrow=1, ncol=Niter),
        pks = matrix(nrow=n, ncol=Niter),
        tls = matrix(nrow=1, ncol=Niter),
        ets = matrix(nrow=1, ncol=Niter),
        es = matrix(nrow=1, ncol=Niter),
        tes = matrix(nrow=1, ncol=Niter),
        # lls = vector("list", length(Niter)),
        mutrs =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter)
            else matrix(nrow=n, ncol=Niter),
        muloos =
            if (truedist=="b") matrix(nrow=n*2, ncol=Niter)
            else matrix(nrow=n, ncol=Niter),
        # mulooks =
        #     if (truedist=="b") matrix(nrow=n*2, ncol=Niter)
        #     else matrix(nrow=n, ncol=Niter),
        bs = matrix(nrow=1, ncol=Niter),
        gs = matrix(nrow=1, ncol=Niter),
        gms = matrix(nrow=1, ncol=Niter),
        g2s = matrix(nrow=1, ncol=Niter),
        gm2s = matrix(nrow=1, ncol=Niter),
        g3s = matrix(nrow=1, ncol=Niter),
    )

    # iterate
    for (i1 in 1:Niter) {
        print(sprintf(
            '%s%s%s p=%d n=%d iter=%d',
            truedist, modeldist, priordist, p, n, i1))
        set.seed(i1)
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
        psis1 = psislw(-log_lik1)
        mu1 = extract_log_lik(model1, parameter_name="mu")
        out$mutrs[,i1] = colMeans(mu1)
        if (truedist=="b") {
            out$muloos[seq(1,n*2,2),i1] = colSums(
                mu1[,seq(1,n*2,2)]*exp(psis1$lw_smooth))
            out$muloos[seq(2,n*2,2),i1] = colSums(
                mu1[,seq(2,n*2,2)]*exp(psis1$lw_smooth))
        } else {
            out$muloos[,i1] = colSums(mu1*exp(psis1$lw_smooth))
        }
        out$mulooks[,i1] = out$muloos[,i1]
        loo1 = loo(log_lik1)
        out$ltrs[,i1] = log(colMeans(exp(log_lik1)))
        out$loos[,i1] = loo1$pointwise[,1]
        out$looks[,i1] = out$loos[,i1]
        out$peff[,i1] = sum(out$ltrs[,i1]) - sum(out$loos[,i1])
        out$pks[,i1] = loo1$pareto_k
        log_lik2 = extract_log_lik(model1, parameter_name="log_likt")
        out$tls[,i1] = mean(log(colMeans(exp(log_lik2))))*n
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
        kexeeds = which(loo1$pareto_k > 0.7)
        out$kexeeds = kexeeds

        # free memory
        rm(model1, output, loo1, log_lik2, mut)
        gc()

        qq = matrix(nrow=n, ncol=n)
        qqm = matrix(nrow=n, ncol=n)
        for (cvi in 1:n) {
            qq[cvi,] = log(colSums(exp(log_lik1+psis1$lw_smooth[,cvi])))
            if (truedist=="b") {
                qqm[cvi,] = as.numeric(xor(
                    colSums(mu1[,seq(1,n*2,2)]*exp(psis1$lw_smooth[,cvi])) > 0,
                    y[seq(1,n*2,2)]
                ))
                qqm[cvi,] = qqm[cvi,] + as.numeric(xor(
                    colSums(mu1[,seq(2,n*2,2)]*exp(psis1$lw_smooth[,cvi])) > 0,
                    y[seq(2,n*2,2)]
                ))
                qqm[cvi,] = 0.5 * qqm[cvi,]
            } else {
                qqm[cvi,] = (y-colSums(mu1*exp(psis1$lw_smooth[,cvi])))^2
            }
        }
        qqq = matrix(nrow=n,ncol=n)
        for (ii1 in 1:n) {
            for (ii2 in 1:n) {
                if (ii1 == ii2)
                    ll2 = log_lik1[,ii1]
                else
                    ll2 = log_lik1[,ii1] + log_lik1[,ii2]
                ps2 = psislw(-ll2)
                qqq[ii2,ii1] = log(sum(exp(log_lik1[,ii1]+ps2$lw_smooth)))
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
        gamma3 = mean(colVars(qqq));
        out$bs[,i1] = sum(betas)
        out$gs[,i1] = gamma
        out$gms[,i1] = gammam
        out$g2s[,i1] = gamma2
        out$gm2s[,i1] = gammam2
        out$g3s[,i1] = gamma3

        # free memory
        rm(log_lik1, psis1, mu1)
        gc()

        n_kexeeds = length(kexeeds)
        if (n_kexeeds > 0) {
            print("k>0.7")
            print(kexeeds)
            # reprocess probelmatic points
            out$looks = matrix(nrow=n_kexeeds, ncol=Niter)
            out$mulooks =
                if (truedist=="b") matrix(nrow=n_kexeeds*2, ncol=Niter)
                else matrix(nrow=n_kexeeds, ncol=Niter),
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
                out$looks[ki, i1] = log(colMeans(exp(
                    extract_log_lik(modelcv, parameter_name="log_likt"))))
                if (truedist=="b") {
                    out$mulooks[((ki-1)*2+1):((ki-1)*2+2), i1] = colMeans(
                        extract_log_lik(modelcv, parameter_name="mut"))
                } else {
                    out$mulooks[ki, i1] = colMeans(
                        extract_log_lik(modelcv, parameter_name="mut"))
                }
                # free memory
                rm(modelcv, output)
                gc()
            }
        }

    }

    filename = sprintf(
        "res_loo/%s_%s_%s_%d_%d.RData", truedist, modeldist, priordist, p, n)
    save(out, file=filename)
}
