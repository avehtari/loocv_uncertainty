## Load one of the datasets
load("nnn1.RData")
## load("t4tnun10.RData")
## load("t4nn10.RData")
## load("nnn0.RData")
## load("nnn5.RData")
## load("nnn10.RData")

# normal approximation accuracy
ni=2;qplot(colSds(loos[[ni]][,1:100])*sqrt(Ns[ni]),sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*g2s[[ni]][,1:100])*sqrt(Ns[ni]))+geom_abline(intercept=0,slope=1);c(mean((sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*g2s[[ni]][,1:100])*sqrt(Ns[ni]))),sd(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100])))

# normal approximation accuracy
ni=7;qplot(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*g2s[[ni]][,1:100])*sqrt(Ns[ni])))

# normal approximation accuracy with qq-plot
library(gridExtra)
ni=4;q1<-gg_qq(qnorm(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*1*g2s[[ni]][,1:100])*sqrt(Ns[ni])))))
ni=1;q2<-gg_qq(qnorm(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100]))*sqrt(Ns[ni])))))
grid.arrange(q1, q2, ncol=2, nrow =1)

# normal approximation accuracy with some quantiles
ni=4
mean(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*g2s[[ni]][,1:100])*sqrt(Ns[ni])))>0.95)
mean(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*0*g3s[[ni]][,1:100])*sqrt(Ns[ni])))>0.95)
mean(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*g2s[[ni]][,1:100])*sqrt(Ns[ni])))<0.05)
mean(pnorm(tls[[ni]][,1:100]-colSums(loos[[ni]][,1:100]),0,(sqrt(colVars(loos[[ni]][,1:100])+(Ns[ni])*0*g3s[[ni]][,1:100])*sqrt(Ns[ni])))<0.05)

##library(matrixStats)
## plot peformance of variance terms
Ns=c(10, 20, 50, 100)
qz=matrix(0,4,4)
for (ni in 1:4) {
    qzz=sd(t(tls[[ni]]-colSums(loos[[ni]])))
    qz[ni,1]=sqrt(mean(colVars(loos[[ni]])*Ns[ni]+0*gs[[ni]]*Ns[ni]*Ns[ni]))/qzz
    qz[ni,2]=sqrt(mean(colVars(loos[[ni]])*Ns[ni]+g2s[[ni]]*Ns[ni]*Ns[ni]))/qzz
    qz[ni,3]=sqrt(mean(colVars(loos[[ni]])*Ns[ni]+g3s[[ni]]*Ns[ni]*Ns[ni]))/qzz
    qz[ni,1:3]=qz[ni,1:3]*Ns[ni]/(Ns[ni]-1)
    qz[ni,4]=ni
}
qz=data.frame(qz)
colnames(qz)[4]="ni"
library(reshape)
qzm=melt(qz, id = "ni")
ggplot(data = qzm, aes(x = ni, y = value, color = variable)) + geom_point() + geom_hline(yintercept=1)

## for MSEs, need to be fixed
## Ns=c(10, 20, 50, 100)
## qz=matrix(0,4,4)
## for (ni in 1:4) {
##     qzz=sd(t(tes[[ni]]-colSums(muloos[[ni]])))
##     qz[ni,1]=sqrt(mean(colVars(muloos[[ni]])*Ns[ni]+0*gms[[ni]]*Ns[ni]*Ns[ni]))/qzz
##     qz[ni,2]=sqrt(mean(colVars(muloos[[ni]])*Ns[ni]+gms[[ni]]*Ns[ni]*Ns[ni]))/qzz
##     qz[ni,3]=sqrt(mean(colVars(muloos[[ni]])*Ns[ni]+gm2s[[ni]]*Ns[ni]*Ns[ni]))/qzz
##     qz[ni,1:3]=qz[ni,1:3]*Ns[ni]/(Ns[ni]-1)
##     qz[ni,4]=ni
## }
## qz=data.frame(qz)
## colnames(qz)[4]="ni"
## library(reshape)
## qzm=melt(qz, id = "ni")
## ggplot(data = qzm, aes(x = ni, y = value, color = variable)) + geom_point() + geom_hline(yintercept=1)

## Analytic variance for Bayesian bootstrap (thanks to Juho)
ni=5
i1=2
ql=loos[[ni]][,i1]
## alpha=1
alpha=1;sqrt(1/(Ns[ni]*(Ns[ni]*alpha+1))*(sum(ql^2) - Ns[ni]*mean(ql)^2))*Ns[ni]
## alpha=1/6
alpha=1-6/Ns[ni];sqrt(1/(Ns[ni]*(Ns[ni]*alpha+1))*(sum(ql^2) - Ns[ni]*mean(ql)^2))*Ns[ni]
## Gaussian approximation
sqrt((var(loos[[ni]][,i1])*Ns[ni]+g2s[[ni]][,i1]*Ns[ni]*Ns[ni]))*Ns[ni]/(Ns[ni])*Ns[ni]/(Ns[ni]-1)

## Gaussian approximation
sqrt((colVars(loos[[ni]])*Ns[ni]+0*g2s[[ni]]*Ns[ni]*Ns[ni]))[1]
## Gaussian approximation with extra variance
sqrt((colVars(loos[[ni]])*Ns[ni]+g2s[[ni]]*Ns[ni]*Ns[ni]))[1]

## Empirical variance for Bayesian bootstrap
##library(extraDistr)
## alpha=1
dw=rdirichlet(100000,matrix(1,1,Ns[ni]))
qw=rowSums(t(t(as.matrix(dw))*as.vector(loos[[ni]][,i1])))*Ns[ni]
sd(qw)
## alpha=1/6
dw=rdirichlet(100000,matrix(1-6/Ns[ni],1,Ns[ni]))
qw=rowSums(t(t(as.matrix(dw))*as.vector(loos[[ni]][,i1])))*Ns[ni]
sd(qw)

## Calibration for Bayesian bootstrap
ni=4
dw=rdirichlet(10000,matrix(1-6/Ns[ni],1,Ns[ni]))
#dw=rdirichlet(10000,matrix(1,1,Ns[ni]))
#dw=rdirichlet(10000,matrix(1-6/20,1,Ns[ni]))
qp=matrix(0,100,1)
qw=matrix(0,10000,100)
for (i1 in 1:100) {
    qw[,i1]=rowSums(t(t(as.matrix(dw))*as.vector(loos[[ni]][,i1])))*Ns[ni]
    qp[i1]=mean(qw[,i1]<=tls[[ni]][i1])
}
qplot(qp)
c(mean(qp<0.05), mean(qp<0.1), mean(qp<0.9), mean(qp<0.95))

c(mean(colSums(loos[[ni]])),mean(t(tls[[ni]])))
qplot(x=colSums(loos[[ni]]),y=t(tls[[ni]]))+geom_abline(intercept=0,slope=1)

## Calibration for Bayesian double bootstrap
ni=4
dw=rdirichlet(10000,matrix(1,1,Ns[ni]))
qww=matrix(0,10000,100)
n=Ns[ni]
for (i1 in 1:100) {
    log_lik1=lls[[ni]][[i1]]
    psis1=psislw(-log_lik1)
    qq<-matrix(nrow=n,ncol=n)
    for (cvi in 1:n) {
        qq[cvi,]<-log(colSums(exp(log_lik1+psis1$lw_smooth[,cvi])))
    }
    qqw<-qq
    for (cvi in 1:n) {
        qqw[-cvi,cvi]<-(qq[-cvi,cvi]-mean(qq[-cvi,cvi]))+qq[cvi,cvi]
    }
    for (i2 in 1:10000) {
        qww[i2,i1]=sum(   t(t(rdirichlet(Ns[ni],matrix(1,1,Ns[ni])))*as.vector(rdirichlet(1,matrix(1,1,Ns[ni]))))*qqw)*Ns[ni]
    }
}
qpp=matrix(0,100,1)
for (i1 in 1:100) {
    qpp[i1]=mean(qww[,i1]<=tls[[ni]][i1])
}
qplot(qpp)
c(mean(qpp<0.05), mean(qpp<0.1), mean(qpp<0.9), mean(qpp<0.95))

qplot(sort(rowSums(t(t(as.matrix(dw))*as.vector(loos[[ni]][,i1])))*Ns[ni]),sort(qww[,1]))+geom_abline()

##RMSE test code
ni=5
dw=rdirichlet(10000,matrix(1-6/Ns[ni],1,Ns[ni]))
#dw=rdirichlet(10000,matrix(1,1,Ns[ni]))
#dw=rdirichlet(10000,matrix(1-6/20,1,Ns[ni]))
qmp=matrix(0,100,1)
qmw=matrix(0,10000,100)
    truedist="n"
for (i1 in 1:100) {
    psis1=psislw(-lls[[ni]][[i1]])
    loomu=colSums(mus[[ni]][[i1]]*exp(psis1$lw_smooth))
    set.seed(i1)
    
                if (truedist=="n") {
                    y <- as.array(rnorm(n))
                } else if (truedist=="t4") {
                    y <- as.array(rt(n,4))
                } else if (truedist=="b") {
                    y <- as.array(as.double(kronecker(matrix(1,1,n),t(c(0,1)))))
                } else {
                    stop("Unknown true distribution")
                }
    qmw[,i1]=1-rowSums(t(t(as.matrix(dw))*as.vector(y-loomu)^2))/var(y)
    qmp[i1]=mean(qw[,i1]>0)
}
qplot(qp)
c(mean(qp<0.05), mean(qp<0.1), mean(qp<0.9), mean(qp<0.95))
