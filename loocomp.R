library("rstanarm")
library("loo")
library("ggplot2")

Ns<-c(10, 20, 30, 40, 60, 80, 100)

loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
pb<-txtProgressBar(1,1000)
yt <- rnorm(10000)
datat <- data.frame(y=yt)
for (ni in 1:length(Ns)) {
    print(ni)
    loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    tls[[ni]]<-matrix(nrow=1,ncol=1000)
    for (i1 in 1:1000) {
        print(i1)
        y <- rnorm(Ns[ni])
        data<-data.frame(y)
        output <- capture.output(
            model1 <- stan_glm(y ~ 1, data=data,
                               family = gaussian(link = "identity"), 
                               cores = 4, seed = i1, iter=500, refresh=-1,
                               open_progress = FALSE))
        loo1<-loo(model1)
        loos[[ni]][,i1]<-loo1$pointwise[,1]
        pks[[ni]][,i1]<-loo1$pareto_k
        ll2 <- log_lik(model1, newdata = datat)
        tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
    }
    save(loos, pks, tls, file="N0.RData")
}

Ns<-c(10, 20, 30, 40, 60, 80, 100)
loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
yt <- rnorm(10000)
xt <- runif(10000)
datat<-data.frame(x=xt,y=yt)
for (ni in 1:length(Ns)) {
    print(ni)
    loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    tls[[ni]]<-matrix(nrow=1,ncol=1000)
    for (i1 in 1:1000) {
        print(i1)
        y <- rnorm(Ns[ni])
        x <- runif(Ns[ni])
        data <- data.frame(x,y)
        output <- capture.output(
            model1 <- stan_glm(y ~ x, data=data,
                               family = gaussian(link = "identity"), 
                               cores = 4, seed = i1, iter=500, refresh=-1,
                               open_progress = FALSE))

        loo1<-loo(model1)
        loos[[ni]][,i1]<-loo1$pointwise[,1]
        pks[[ni]][,i1]<-loo1$pareto_k
        ll2 <- log_lik(model1, newdata = datat)
        tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
    }
    save(loos, pks, tls, file="N1.RData")
}

Ns<-c(10, 20, 30, 40, 60, 80, 100)
loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
yt <- rnorm(10000)
xt <- matrix(runif(2*10000),ncol=2)
datat<-data.frame(x=xt,y=yt)
for (ni in 1:length(Ns)) {
    print(ni)
    loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
    tls[[ni]]<-matrix(nrow=1,ncol=1000)
    for (i1 in 1:1000) {
        print(i1)
        y <- rnorm(Ns[ni])
        x <- matrix(runif(Ns[ni]),ncol=2)
        data <- data.frame(x,y)
        output <- capture.output(
            model1 <- stan_glm(y ~ x, data=data,
                               family = gaussian(link = "identity"), 
                               cores = 4, seed = i1, iter=500, refresh=-1,
                               open_progress = FALSE))

        loo1<-loo(model1)
        loos[[ni]][,i1]<-loo1$pointwise[,1]
        pks[[ni]][,i1]<-loo1$pareto_k
        ll2 <- log_lik(model1, newdata = datat)
        tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
    }
    save(loos, pks, tls, file="N2.RData")
}

Ns<-c(10, 20, 30, 40, 60, 80, 100)
ps<-c(1,2,5,10)
loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
yt <- rnorm(10000)
for (pi in 1:4) {
    p<-ps[pi]
    xt <- matrix(runif(p*10000),ncol=p)
    datat<-data.frame(x=xt,y=yt)
    for (ni in 1:length(Ns)) {
        print(ni)
        loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        tls[[ni]]<-matrix(nrow=1,ncol=1000)
        for (i1 in 1:1000) {
            print(i1)
            y <- rnorm(Ns[ni])
            x <- matrix(runif(Ns[ni]),ncol=p)
            data <- data.frame(x,y)
            output <- capture.output(
                model1 <- stan_glm(y ~ x, data=data,
                                   family = gaussian(link = "identity"), 
                                   cores = 4, seed = i1, iter=500, refresh=-1,
                                   open_progress = FALSE))

            loo1<-loo(model1)
            loos[[ni]][,i1]<-loo1$pointwise[,1]
            pks[[ni]][,i1]<-loo1$pareto_k
            ll2 <- log_lik(model1, newdata = datat)
            tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
        }
        save(loos, pks, tls, file=sprintf("N%d.RData",p))
    }
}

Ns<-c(10, 20, 30, 40, 60, 80, 100)
ps<-c(1,2,5,10)
loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
yt <- rt(10000,4)
pi<-0
p<-0

#    xt <- matrix(runif(p*10000),ncol=2)
#    datat<-data.frame(x=xt,y=yt)
    datat<-data.frame(y=yt)
    for (ni in 4:length(Ns)) {
        loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        tls[[ni]]<-matrix(nrow=1,ncol=1000)
        for (i1 in 1:1000) {
            print(sprintf('p=%d, n=%d, i1=%d',p,Ns[ni],i1)) 
            y <- rt(Ns[ni],4)
#            x <- matrix(runif(p*Ns[ni]),ncol=p)
#            data <- data.frame(x,y)
            data <- data.frame(y)
            output <- capture.output(
                model1 <- stan_glm(y ~ 1, data=data,
                                   family = gaussian(link = "identity"), 
                                   cores = 1, seed = i1, iter=500, refresh=-1,
                                   open_progress = FALSE))

            loo1<-loo(model1)
            loos[[ni]][,i1]<-loo1$pointwise[,1]
            pks[[ni]][,i1]<-loo1$pareto_k
            ll2 <- log_lik(model1, newdata = datat)
            tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
        }
        save(loos, pks, tls, file=sprintf("tN%d.RData",p))
    }
}

Ns<-c(10, 20, 30, 40, 60, 80, 100)
ps<-c(1,2,5,10)
loos<-vector("list",7)
pks<-vector("list",7)
tls<-vector("list",7)
yt <- rt(10000,4)
for (pi in 2:5) {
    p<-ps[pi]
    xt <- matrix(runif(p*10000),ncol=p)
    datat<-data.frame(x=xt,y=yt)
    for (ni in 1:length(Ns)) {
        print(ni)
        loos[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        pks[[ni]]<-matrix(nrow=Ns[ni],ncol=1000)
        tls[[ni]]<-matrix(nrow=1,ncol=1000)
        for (i1 in 1:1000) {
            print(i1)
            y <- rt(Ns[ni],4)
            x <- matrix(runif(p*Ns[ni]),ncol=p)
            data <- data.frame(x=x,y=y)
            output <- capture.output(
                model1 <- stan_glm(y ~ ., data=data,
                                   family = gaussian(link = "identity"), 
                                   cores = 1, seed = i1, iter=500, refresh=-1,
                                   open_progress = FALSE))

            loo1<-loo(model1)
            loos[[ni]][,i1]<-loo1$pointwise[,1]
            pks[[ni]][,i1]<-loo1$pareto_k
            ll2 <- log_lik(model1, newdata = datat)
            tls[[ni]][,i1] <- mean(log(colMeans(exp(ll2))))*Ns[ni]
        }
        save(loos, pks, tls, file=sprintf("tN%d.RData",p))
    }
}
