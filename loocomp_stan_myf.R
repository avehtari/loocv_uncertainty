library(rstan)
library(loo)
library(matrixStats)
rstan_options(auto_write = TRUE)
##stanmodel<-stan_model("linear_n_n.stan", verbose = FALSE)
##stanmodel<-stan_model("linear_tnu_n.stan", verbose = FALSE)

myf<-function(truedist,modeldist,priordist) {
    Niter=1000
    modelname<-sprintf("linear_%s_%s.stan",modeldist,priordist)
    Ns<-c(10, 20, 30, 40, 60, 80, 100)
    Ps<-c(0, 1, 2, 5, 10)
    if (priordist=="hs") {
        Ps<-c(10,100)
    }
    Nt<-10000
    ## truedist<-"n"
    ## modeldist<-"n"
    ## priordist<-"n"
    for (ppi in 2:length(Ps)) {
        p<-Ps[ppi]
        ltrs<-vector("list",length(Ns))
        loos<-vector("list",length(Ns))
        looks<-vector("list",length(Ns))
        peff<-vector("list",length(Ns))
        pks<-vector("list",length(Ns))
        tls<-vector("list",length(Ns))
        ets<-vector("list",length(Ns))
        es<-vector("list",length(Ns))
        tes<-vector("list",length(Ns))
        lls<-vector("list",length(Ns))
        mutrs<-vector("list",length(Ns))
        muloos<-vector("list",length(Ns))
        mulooks<-vector("list",length(Ns))
        bs<-vector("list",length(Ns))
        gs<-vector("list",length(Ns))
        gms<-vector("list",length(Ns))
        g2s<-vector("list",length(Ns))
        gm2s<-vector("list",length(Ns))
        g3s<-vector("list",length(Ns))
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
        for (ni in 1:length(Ns)) {
            n<-Ns[ni]
            ltrs[[ni]]<-matrix(nrow=n,ncol=Niter)
            loos[[ni]]<-matrix(nrow=n,ncol=Niter)
            looks[[ni]]<-matrix(nrow=n,ncol=Niter)
            peff[[ni]]<-matrix(nrow=1,ncol=Niter)
            pks[[ni]]<-matrix(nrow=n,ncol=Niter)
            tls[[ni]]<-matrix(nrow=1,ncol=Niter)
            ets[[ni]]<-matrix(nrow=1,ncol=Niter)
            es[[ni]]<-matrix(nrow=1,ncol=Niter)
            tes[[ni]]<-matrix(nrow=1,ncol=Niter)
            lls[[ni]]<-vector("list",length(Niter))
            if (truedist=="b") {
                mutrs[[ni]]<-matrix(nrow=n*2,ncol=Niter)
                muloos[[ni]]<-matrix(nrow=n*2,ncol=Niter)
                mulooks[[ni]]<-matrix(nrow=n*2,ncol=Niter)
            } else {
                mutrs[[ni]]<-matrix(nrow=n,ncol=Niter)
                muloos[[ni]]<-matrix(nrow=n,ncol=Niter)
                mulooks[[ni]]<-matrix(nrow=n,ncol=Niter)
            }
            bs[[ni]]<-matrix(nrow=1,ncol=Niter)
            gs[[ni]]<-matrix(nrow=1,ncol=Niter)
            gms[[ni]]<-matrix(nrow=1,ncol=Niter)
            g2s[[ni]]<-matrix(nrow=1,ncol=Niter)
            gm2s[[ni]]<-matrix(nrow=1,ncol=Niter)
            g3s[[ni]]<-matrix(nrow=1,ncol=Niter)
            for (i1 in 1:Niter) {
                print(sprintf('%s%s%s p=%d n=%d i1=%d',
                              truedist,modeldist,priordist,p,n,i1))
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
                data<-list(N = n, p = ncol(x), x = x, y = y,
                           Nt = Nt, xt = xt, yt = yt)
                output <- capture.output(
                    model1<-stan(modelname,data=data,iter=1000,refresh=-1,
                                 save_warmup = FALSE,
                                 cores = 1, open_progress = FALSE))
                log_lik1 <- extract_log_lik(model1)
                psis1=psislw(-log_lik1)
                #lls[[ni]][[i1]]=log_lik1
                mu1 <- extract_log_lik(model1,parameter_name="mu")
                mutrs[[ni]][,i1]=colMeans(mu1)
                if (truedist=="b") {
                    muloos[[ni]][seq(1,n*2,2),i1]=colSums(mu1[,seq(1,n*2,2)]*exp(psis1$lw_smooth))
                    muloos[[ni]][seq(2,n*2,2),i1]=colSums(mu1[,seq(2,n*2,2)]*exp(psis1$lw_smooth))
                } else {
                    muloos[[ni]][,i1]=colSums(mu1*exp(psis1$lw_smooth))
                }
                mulooks[[ni]][,i1]<-muloos[[ni]][,i1]
                loo1<-loo(log_lik1)
                ltrs[[ni]][,i1]<-log(colMeans(exp(log_lik1)))
                loos[[ni]][,i1]<-loo1$pointwise[,1]
                looks[[ni]][,i1]<-loos[[ni]][,i1]
                peff[[ni]][,i1]<-sum(ltrs[[ni]][,i1])-sum(loos[[ni]][,i1])
                pks[[ni]][,i1]<-loo1$pareto_k
                log_lik2 <- extract_log_lik(model1,parameter_name="log_likt")
                tls[[ni]][,i1] <- mean(log(colMeans(exp(log_lik2))))*n
                mut <- extract_log_lik(model1,parameter_name="mut")
                if (truedist=="b") {
                    ets[[ni]][,i1] <- mean(xor(mutrs[[ni]][,i1]>0,(y>0)))
                    es[[ni]][,i1] <- mean(xor(muloos[[ni]][,i1]>0,(y>0)))
                    tes[[ni]][,i1] <- mean(xor(colMeans(mut)>0,(yt>0)))
                } else {
                    ets[[ni]][,i1] <- mean((y-mutrs[[ni]][,i1])^2)/var(y)
                    es[[ni]][,i1] <- mean((y-muloos[[ni]][,i1])^2)/var(y)
                    tes[[ni]][,i1] <- mean((yt-colMeans(mut))^2)/var(yt)
                }
                {
                    qq<-matrix(nrow=n,ncol=n)
                    qqm<-matrix(nrow=n,ncol=n)
                    for (cvi in 1:n) {
                        qq[cvi,]<-log(colSums(exp(log_lik1+psis1$lw_smooth[,cvi])))
                        if (truedist=="b") {
                            qqm[cvi,]<-as.numeric(xor(colSums(mu1[,seq(1,n*2,2)]*exp(psis1$lw_smooth[,cvi]))>0,y[seq(1,n*2,2)]))
                            qqm[cvi,]<-(qqm[cvi,]+as.numeric(xor(colSums(mu1[,seq(2,n*2,2)]*exp(psis1$lw_smooth[,cvi]))>0,y[seq(2,n*2,2)])))/2
                        } else {
                            qqm[cvi,]<-(y-colSums(mu1*exp(psis1$lw_smooth[,cvi])))^2
                        }
                    }
                    qqq<-matrix(nrow=n,ncol=n)
                    for (ii1 in 1:n) {
                        for (ii2 in 1:n) {
                            if (ii1 == ii2)
                                ll2<-log_lik1[,ii1]
                            else
                                ll2<-log_lik1[,ii1]+log_lik1[,ii2]
                            ps2=psislw(-ll2)
                            qqq[ii2,ii1]<-log(sum(exp(log_lik1[,ii1]+ps2$lw_smooth)))
                        }
                    }
                    gammas<-matrix(nrow=1,ncol=n)
                    gammams<-matrix(nrow=1,ncol=n)
                    for (cvi in 1:n) {
                        gammas[,cvi]=var(qq[-cvi,cvi]);
                        gammams[,cvi]=var(qqm[-cvi,cvi]);
                    }
                    betas<-matrix(nrow=1,ncol=n)
                    mbetas<-matrix(nrow=1,ncol=n)
                    for (cvi in 1:n) {
                        betas[,cvi]=mean(qq[-cvi,cvi]);
                        mbetas[,cvi]=mean(qqm[-cvi,cvi]);
                    }
                    gamma=mean(gammas);
                    gammam=mean(gammas);
                    gamma2=mean(colVars(qq));
                    gammam2=mean(colVars(qq));
                    gamma3=mean(colVars(qqq));
                    bs[[ni]][,i1]=sum(betas)
                    gs[[ni]][,i1]=gamma
                    gms[[ni]][,i1]=gammam
                    g2s[[ni]][,i1]=gamma2
                    gm2s[[ni]][,i1]=gammam2
                    g3s[[ni]][,i1]=gamma3
                }
                if (length(which(loo1$pareto_k>0.7))>0) {
                    print("k>0.7")
                    print(which(loo1$pareto_k>0.7))
                }
                for (cvi in which(loo1$pareto_k>0.7)) {
                    if (truedist=="b") {
                        data<-list(N = n-1, p = ncol(x),
                                   x = as.matrix(x[-c(cvi*2-1,cvi*2),]), y = y[-c(cvi*2-1,cvi*2)],
                                   Nt = 1, xt = matrix(data=x[c(cvi*2-1,cvi*2),],nrow=2,ncol=p),
                                   yt= as.array(y[c(cvi*2-1,cvi*2)]))
                    } else {
                        data<-list(N = n-1, p = ncol(x),
                                   x = as.matrix(x[-cvi,]), y = y[-cvi],
                                   Nt = 1, xt = matrix(data=x[cvi,],nrow=1,ncol=p),
                                   yt= as.array(y[cvi]))
                    }
                    output <- capture.output(
                        modelcv<-stan(modelname,data=data,iter=1000,refresh=-1,
                                      save_warmup = FALSE,
                                      cores = 1, open_progress = FALSE))
                    log_likcv <- extract_log_lik(modelcv,parameter_name="log_likt")
                    looks[[ni]][cvi,i1] <- log(colMeans(exp(log_likcv)))
                    mucv <- extract_log_lik(modelcv,parameter_name="mut")
                    mulooks[[ni]][cvi,i1]=colMeans(mucv)
                }
            }
            save(truedist,modeldist,priordist, loos, looks, pks, tls, tes, lls, muloos, mutrs, mulooks, muts, gs, g2s, g3s, ltrs, bs, file=sprintf("%s%s%s%d.RData",truedist,modeldist,priordist,p))
        }
    }
}
