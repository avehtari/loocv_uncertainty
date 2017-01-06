library("rstan")
library("loo")
rstan_options(auto_write = TRUE)
##stanmodel<-stan_model("linear_n_n.stan", verbose = FALSE)
##stanmodel<-stan_model("linear_tnu_n.stan", verbose = FALSE)

myf<-function(truedist,modeldist,priordist) {
    modelname<-sprintf("linear_%s_%s.stan",modeldist,priordist)
    Ns<-c(10, 20, 30, 40, 60, 80, 100)
    Ps<-c(0, 1, 2, 5, 10)
    Nt<-10000
    ## truedist<-"n"
    ## modeldist<-"n"
    ## priordist<-"n"
    for (ppi in 1:length(Ps)) {
        p<-Ps[ppi]
        ltrs<-vector("list",length(Ns))
        loos<-vector("list",length(Ns))
        pks<-vector("list",length(Ns))
        tls<-vector("list",length(Ns))
        sds<-vector("list",length(Ns))
        gs<-vector("list",length(Ns))
        g2s<-vector("list",length(Ns))
        g3s<-vector("list",length(Ns))
        set.seed(1)
        if (truedist=="n")
            yt <- as.array(rnorm(Nt))
        else if (truedist=="nt")
            yt <- as.array(rt(Nt,4))
        else
            stop("Unknown true distribution")
        xt <- matrix(runif(p*Nt)*2-1,nrow=Nt,ncol=p)
        for (ni in 1:length(Ns)) {
            n<-Ns[ni]
            ltrs[[ni]]<-matrix(nrow=n,ncol=1000)
            loos[[ni]]<-matrix(nrow=n,ncol=1000)
            pks[[ni]]<-matrix(nrow=n,ncol=1000)
            tls[[ni]]<-matrix(nrow=1,ncol=1000)
            sds[[ni]]<-matrix(nrow=1,ncol=1000)
            gs[[ni]]<-matrix(nrow=1,ncol=1000)
            g2s[[ni]]<-matrix(nrow=1,ncol=1000)
            g3s[[ni]]<-matrix(nrow=1,ncol=1000)
            for (i1 in 1:1000) {
                print(sprintf('%s%s%s p=%d n=%d i1=%d',
                              truedist,modeldist,priordist,p,n,i1))
                set.seed(i1)
                if (truedist=="n")
                    y <- as.array(rnorm(n))
                else if (truedist=="nt")
                    y <- as.array(rt(n,4))
                else
                    stop("Unknown true distribution")
                x <- matrix(runif(p*n),nrow=n,ncol=p)
                data<-list(N = length(y), p = ncol(x), x = x, y = y,
                           Nt = length(yt), xt = xt, yt = yt)
                output <- capture.output(
                    model1<-stan(modelname,data=data,iter=1000,refresh=-1,
                                 save_warmup = FALSE,
                                 cores = 1, open_progress = FALSE))
                log_lik1 <- extract_log_lik(model1)
                loo1<-loo(log_lik1)
                ltrs[[ni]][,i1]<-log(colMeans(exp(log_lik1)))
                loos[[ni]][,i1]<-loo1$pointwise[,1]
                pks[[ni]][,i1]<-loo1$pareto_k
                log_lik2 <- extract_log_lik(model1,parameter_name="log_likt")
                tls[[ni]][,i1] <- mean(log(colMeans(exp(log_lik2))))*n
                {
                    psis1=psislw(-log_lik1)
                    qq<-matrix(nrow=n,ncol=n)
                    for (cvi in 1:n) {
                        qq[cvi,]<-log(colSums(exp(log_lik1+psis1$lw_smooth[,cvi])))
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
                    for (cvi in 1:n) {
                        gammas[,cvi]=var(qq[-cvi,cvi]);
                    }
                    gamma=mean(gammas);
                    gamma2=mean(colVars(qq));
                    gamma3=mean(colVars(qqq));
                    varloo=var(loos[[ni]][,i1])
                    gs[[ni]][,i1]=gamma
                    g2s[[ni]][,i1]=gamma2
                    g3s[[ni]][,i1]=gamma3
                    sds[[ni]][,i1]=sqrt(n*varloo*(1+n*gamma3))
                }
                if (length(which(loo1$pareto_k>0.7))>0) {
                    print("k>0.7")
                    print(which(loo1$pareto_k>0.7))
                }
                for (cvi in which(loo1$pareto_k>0.7)) {
                    data<-list(N = length(y)-1, p = ncol(x),
                               x = as.matrix(x[-cvi,]), y = y[-cvi],
                               Nt = 1, xt = matrix(data=x[cvi,],nrow=1,ncol=p),
                               yt= as.array(y[cvi]))
                    output <- capture.output(
                        modelcv<-stan(modelname,data=data,iter=1000,refresh=-1,
                                      save_warmup = FALSE,
                                      cores = 1, open_progress = FALSE))
                    log_likcv <- extract_log_lik(modelcv,parameter_name="log_likt")
                    loos[[ni]][cvi,i1] <- log(colMeans(exp(log_likcv)))
                }
            }
            save(loos, pks, tls, sds, gs, g2s, g3s, ltrs, file=sprintf("%s%s%s%d.RData",truedist,modeldist,priordist,p))
        }
    }
}
