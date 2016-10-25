%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-10-24 22:07:59 EDT


clear
N=[10 20 30 50 70 100];

%% unknown variance %%
for ni=1:numel(N)
    n=N(ni);
    fprintf('N=%d\n',n)
    %load(sprintf('normal_n%d',n),'y','yt')
    y=randn(1000,n);
    yt=randn(1000,n);
    mu=mean(y,2);
    sh=std(y,0,2);
    s=1;
    ps=sqrt(1+1/n);
    ltrs=t_lpdfs(y,n-1,mu,sh.*ps);
    ltr=sum(ltrs,2);
    ltsts=t_lpdfs(yt,n-1,mu,sh.*ps);
    ltst=sum(ltsts,2);
    for i1=1:1000
        ltst0(i1,1)=n*integral(@(wy) norm_pdfs(wy,0,1).*t_lpdfs(wy,n-1,mu(i1),sh(i1).*ps),-6,6);
    end
    ss=zeros(1000);
    smu=zeros(1000);
    ss=bsxfun(@times,sqrt(sinvchi2rand(n-1,1,1000,1000)),sh);
    smu=bsxfun(@plus,mu,randn(1000,1000).*ss.*sqrt(1/n));
    clear loos pks
    for i1=1:1000
        logliks=norm_lpdfs(y(i1,:),smu(i1,:)',ss(i1,:)');
        %pwaics(i1,:)=var(logliks);
        [~,loos(i1,:),pks(i1,:)]=psisloo(logliks);
    end
    % waics=ltrs-pwaics;
    % waic=sum(waics,2);
    % loo=sum(loos,2);
    ploo(:,ni)=sum(ltrs-loos,2);
    for i1=1:1000
        setrandstream(1);
        qbbm(i1,:,1)=wmean(loos(i1,:)',dirrnd(1*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,2)=wmean(loos(i1,:)',dirrnd(.9*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,3)=wmean(loos(i1,:)',dirrnd(.8*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,4)=wmean(loos(i1,:)',dirrnd(.7*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,5)=wmean(loos(i1,:)',dirrnd(.6*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,6)=wmean(loos(i1,:)',dirrnd(.5*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,7)=wmean(loos(i1,:)',dirrnd(.4*ones(1,n),10000)')*n;
    end
    for i2=1:7
        qps(ni,:,i2)=[mean(mean(bsxfun(@ge,ltst0,qbbm(:,:,i2)),2)<0.05)...
                      mean(mean(bsxfun(@ge,ltst0,qbbm(:,:,i2)),2)>0.95)];
        qmss(ni,i2)=mean(std(qbbm(:,:,i2),[],2));
    end
    disp(squeeze(qps(ni,:,:)))
end

clrs=get(gca,'colororder');
clf
hold on
for ni=1:numel(N)
h=plot(1./[1 .9 .8 .7 .6 .5],squeeze(qps(ni,1,:))','-v');set(h,'color',clrs(ni,:))
h=plot(1./[1 .9 .8 .7 .6 .5],squeeze(qps(ni,2,:))','-^');set(h,'color',clrs(ni,:))
end
line(xlim,0.05*[1 1],'linestyle','--','color','k')
xlabel('1/alpha')
clf,hold on
for i2=1:6
h=plot(N,squeeze(qps(:,1,i2))','-v');set(h,'color',clrs(i2,:))
h=plot(N,squeeze(qps(:,2,i2))','-^');set(h,'color',clrs(i2,:))
end
line(xlim,0.05*[1 1],'linestyle','--','color','k')


mean(std(qbbm(:,:,3),[],2)./std(qbbm(:,:,1),[],2))
%1.2125

hold on
plot(1./[1 3/4 2/3 1/2],squeeze(qps(2,:,:))','-')
line(xlim,0.05*[1 1],'linestyle','--','color','k')
xlabel('1/alpha')

q=loos(i1,:)-mean(loos(1,:));
qq=bsxfun(@times,q,q');qq=qq./(sqrt(bsxfun(@times,diag(qq),diag(qq)')));
(sum(qq(:)-sum(diag(qq))))

% gp=gp_set('meanf',gpmf_constant('prior_cov',1000^2),'cf',gpcf_constant('constSigma2_prior',prior_loggaussian('s2',1)),'lik',lik_gaussian('sigma2_prior',prior_gaussian('s2',10^2)));
% % gp=gp_set('cf',gpcf_constant('constSigma2_prior',prior_loggaussian('s2',1)),'lik',lik_gaussian('sigma2_prior',prior_gaussian('s2',10^2)));
% qx=[1:n]';
% qy=loos(i1,:)';%-mean(loos(i1,:));
% qy=randn(n*20,1);
% qx=ones(size(qy));
% gp=gp_optim(gp,qx,qy);
% [K,C]=gp_trcov(gp,qx);
% C(1:2,1:2)
% CC=C./sqrt(bsxfun(@times,diag(C),diag(C)'));
% CC(1:2,1:2)
% 0.7792
% sqrt(1.66)
% %1.3339
% std(qy)


10  0.5
20  0.6
30  0.7
40  0.75
50  0.8
70  0.9
100 1.0
%%%

clear
N=[10 20 30 50 70 100];

%% known variance %%
for ni=2:numel(N)
    n=N(ni);
    fprintf('N=%d\n',n)
    %load(sprintf('normal_n%d',n),'y','yt')
 y=randn(1000,n);
 yt=randn(1000,n);
 ytt=randn(100000,n);
mu=mean(y,2);
sh=std(y,0,2);
s=1;
ps=sqrt(1+1/n);
ltrs=norm_lpdfs(y,mu,ps);
ltr=sum(ltrs,2);
ltsts=norm_lpdfs(yt,mu,ps);
ltst=sum(ltsts,2);
for i1=1:1000
  ltst0(i1,1)=n*integral(@(wy) norm_pdfs(wy,0,1).*norm_lpdfs(wy,mu(i1),ps),-6,6);
end
smu=zeros(1000);
smu=bsxfun(@plus,mu,bsxfun(@times,randn(1000,1000),sh.*sqrt(1/n)));
    clear loos pks
for i1=1:1000
  logliks=norm_lpdfs(y(i1,:),smu(i1,:)',sh(i1));
  [~,loos(i1,:),pks(i1,:)]=psisloo(logliks);
end
loo=sum(loos,2);
    ploo(:,ni)=sum(ltrs-loos,2);
    for i1=1:1000
        setrandstream(1);
        qbbm(i1,:,1)=wmean(loos(i1,:)',dirrnd(1*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,2)=wmean(loos(i1,:)',dirrnd(.9*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,3)=wmean(loos(i1,:)',dirrnd(.8*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,4)=wmean(loos(i1,:)',dirrnd(.7*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,5)=wmean(loos(i1,:)',dirrnd(.6*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,6)=wmean(loos(i1,:)',dirrnd(.5*ones(1,n),10000)')*n;
        setrandstream(1);
        qbbm(i1,:,7)=wmean(loos(i1,:)',dirrnd(.4*ones(1,n),10000)')*n;
    end
    for i2=1:7
        qps(ni,:,i2)=[mean(mean(bsxfun(@ge,ltst0,qbbm(:,:,i2)),2)<0.05)...
                      mean(mean(bsxfun(@ge,ltst0,qbbm(:,:,i2)),2)>0.95)];
        qmss(ni,i2)=mean(std(qbbm(:,:,i2),[],2));
    end
    disp(squeeze(qps(ni,:,:)))
end

clrs=get(gca,'colororder');
clf
hold on
for ni=1:numel(N)
h=plot(1./[1 3/4 2/3 1/2],squeeze(qps(ni,1,:))','-v');set(h,'color',clrs(ni,:))
h=plot(1./[1 3/4 2/3 1/2],squeeze(qps(ni,2,:))','-^');set(h,'color',clrs(ni,:))
end
line(xlim,0.05*[1 1],'linestyle','--','color','k')
xlabel('1/alpha')
clf,hold on
for i2=1:4
h=plot(N,squeeze(qps(:,1,i2))','-v');set(h,'color',clrs(i2,:))
h=plot(N,squeeze(qps(:,2,i2))','-^');set(h,'color',clrs(i2,:))
end

mean(std(qbbm(:,:,3),[],2)./std(qbbm(:,:,1),[],2))
%1.2125

hold on
plot(1./[1 3/4 2/3 1/2],squeeze(qps(2,:,:))','-')
line(xlim,0.05*[1 1],'linestyle','--','color','k')
xlabel('1/alpha')

q=loos(i1,:)-mean(loos(1,:));
qq=bsxfun(@times,q,q');qq=qq./(sqrt(bsxfun(@times,diag(qq),diag(qq)')));
(sum(qq(:)-sum(diag(qq))))

% gp=gp_set('meanf',gpmf_constant('prior_cov',1000^2),'cf',gpcf_constant('constSigma2_prior',prior_loggaussian('s2',1)),'lik',lik_gaussian('sigma2_prior',prior_gaussian('s2',10^2)));
% % gp=gp_set('cf',gpcf_constant('constSigma2_prior',prior_loggaussian('s2',1)),'lik',lik_gaussian('sigma2_prior',prior_gaussian('s2',10^2)));
% qx=[1:n]';
% qy=loos(i1,:)';%-mean(loos(i1,:));
% qy=randn(n*20,1);
% qx=ones(size(qy));
% gp=gp_optim(gp,qx,qy);
% [K,C]=gp_trcov(gp,qx);
% C(1:2,1:2)
% CC=C./sqrt(bsxfun(@times,diag(C),diag(C)'));
% CC(1:2,1:2)
% 0.7792
% sqrt(1.66)
% %1.3339
% std(qy)


10  0.5
20  0.6
30  0.7
40  0.75
50  0.8
70  0.9
100 1.0





    bbm=bbmean(loos',1000)'*n;
    %save(sprintf('m2_normalz_n%d',n),'ltsts','ltst','ltst0','pwaics','waic','waics','bbm')

%for i1=1:10000;qb=(-2*bbm(i1,:));qb=qb-min(qb);qb=qb+min(qb(qb>0));qbd=min(-2*bbm(i1,:))-min(qb);qqqq(i1,:)=gamfit(qb);qgp(i1,1)=1-gamcdf(ltst0(i1)*-2-qbd,3,mean(qb)/2);end
%for i1=1:10000;qb=(-2*bbm(i1,:));qb=qb-min(qb);qb=qb+min(qb(qb>0));qbd=min(-2*bbm(i1,:))-min(qb);qgp(i1,1)=1-gamcdf(ltst0(i1)*-2-qbd,1.5,mean(qb));end

mi=1;n=10;
load(sprintf('m%d_normalz_n%d',mi,n))
a4(.33,.5)
plot(ltst0,loo,'.')
xlabel('elppd(y)')
ylabel('LOO(y)')
print('-depsc2',sprintf('m%d_normalz_%d_elppd_vs_LOO.eps',mi,n))
plot(std(loos,0,2)*sqrt(n),(ltst0-loo),'.')
xlabel('std(LOO)')
ylabel('elppd(y)-LOO(y)')
print('-depsc2',sprintf('m%d_normalz_%d_stdLOO_vs_error.eps',mi,n))
histogram(ltst0-loo,50)
xlabel('elppd(y)-LOO(y)')
print('-depsc2',sprintf('m%d_normalz_%d_error.eps',mi,n))
% std(loos)
hist(normcdf(ltst0-loo,0,std(loos,0,2)*sqrt(n)),50)
hist(tcdf((ltst0-loo)./(std(loos,0,2)*sqrt(n)),n-1),50)
hist(tcdf((ltst0-loo)./(std(loos,0,2)*sqrt(n))*(0.8),n-1-1.6),50)
qqplot(tcdf((ltst0-loo)./(std(loos,0,2)*sqrt(n))*(1),n-1-1.7),rand(10000,1)),diagline
mean(tcdf((ltst0-loo)./(std(loos,0,2)*sqrt(n))*(.7),n-1-1.6)<0.05)
mean(tcdf((ltst0-loo)./(std(loos,0,2)*sqrt(n))*(1),n-1-2)>0.95)
i1=1;clf;plot(loos(i1,:),loos(i1,:),'o',mean(loos(i1,:)),mean(loos(i1,:)),'*',ltst0(i1,:)/n,ltst0(i1,:)/n,'d')
for i1=1:1000
   bbm1(i1,:)=wmean(loos(i1,:)',dirrnd(1*ones(1,n),1000)')*n;
   %qbbm(i1,:)=wmean(loos(i1,:)',dirrnd((n-2-ploo(i1)*2)/n*ones(1,n),10000)')*n;
   qbbm(i1,:)=wmean(loos(i1,:)',dirrnd(.4*ones(1,n),10000)')*n;
   %qbbm(i1,:)=wmean(((loos(i1,:)'-max(loos(i1,:)'))*sqrt(1.3)+max(loos(i1,:)')),dirrnd(1*ones(1,n),100000)')*n;
end
for i1=1:50;
    clf
    %[qp,~,qxt]=lgpdens(bbmean(loos(i1,:)',10000));
    [qp,~,qxt]=lgpdens(bbm(i1,:)');
    plot(qxt,qp)
hold on
    [qp,~,qxt]=lgpdens(qbbm(i1,:)');
    plot(qxt,qp)
% histogram(bbm(i1,:)','normalization','pdf')
% histogram(qbbm(i1,:)','normalization','pdf')
line(ltst0(i1,:)*[1 1],ylim,'color','r')
pause
end
qpn=qp./sum(qp);sum(qpn(1:find(qxt>ltst0(i1,:)/n,1)))
mean(bbmean(loos(i1,:)',10000)<ltst0(i1,:)/n)

xlabel('p(elppd(y)>LOO(y)) using Gaussian approximation')
print('-depsc2',sprintf('m%d_normalz_%d_pLOO.eps',mi,n))
qqplot(rand(10000,1),normcdf(ltst0-loo,0,std(loos,0,2)*sqrt(n)))
ylim([0 1])
diagline
xlabel('Uniform quantiles')
ylabel('p(elppd(y)>LOO(y)) (Gaussian) quantiles')
print('-depsc2',sprintf('m%d_normalz_%d_pLOOqplot.eps',mi,n))

% hist(normcdf(mean(ltst0)-waic,0,std(waics,0,2)*sqrt(n)),50)
% hist(tcdf((ltst0-waic)./(std(waics,0,2)*sqrt(n)),4),50)
% dlta=(ltst0-waic);
% dlta=(mean(ltst0)-waic);
% qii=(ltst0-waic)>-2;
% qii=std(waics,0,2)*sqrt(n)>1.05;
% qii=Ef>-.2;
% qii=(kurtosis(waics,1,2)-3*skewness(waics,1,2))<8;
% hist(normcdf(dlta(qii),0,std(waics(qii,:),0,2)*sqrt(n)),50)
% hist(normcdf(dlta(~qii),0,std(waics(~qii,:),0,2)*sqrt(n)),50)
% % corrected? std(waics)
% hist(normcdf(ltst0-waic,0,n./(n-p_waic).*std(waics,0,2)*sqrt(n)),50)
% bbmean
histogram(mean(bsxfun(@ge,ltst0,bbm),2),50)
histogram(mean(bsxfun(@ge,ltst0,bbm2),2),50)
xlabel('p(elppd(y)>WAIC(y)) using BB approximation')
print('-depsc2',sprintf('m%d_normalz_%d_bbpWAIC.eps',mi,n))
qqplot(rand(10000,1),mean(bsxfun(@ge,ltst0,bbm),2))
ylim([0 1])
diagline
xlabel('Uniform quantiles')
ylabel('p(elppd(y)>WAIC(y)) (BB) quantiles')
print('-depsc2',sprintf('m%d_normalz_%d_bbpWAICqplot.eps',mi,n))

% for i1=1:10000;q=-waics(i1,:)+max(waics(i1,:));q2p(i1,1)=chi2cdf(n,(-(ltst0(i1)-1/(2*n+2))-(n*(+.000-max(waics(i1,:)))))*2);end
% hist(q2p,100)
% hist(mean(bsxfun(@ge,ltst0(dlta>0),bbm(dlta>0,:)),2),50)
% theoretical with true sigma
hist(normcdf(ltst0-waic,-1*(n-1)/2/n/(n+1),sqrt((n+3+4/n)/(2+4/n))),50)
xlabel('p(elppd(y)>WAIC(y)) theoretical with true sigma')
print('-depsc2',sprintf('m%d_normalz_%d_theorpWAIC.eps',mi,n))
% theoretical with hat sigma
hist(normcdf(ltst0-waic,-1*(n-1)/2/n/(n+1),sqrt((n+3+4/n)/(2+4/n)).*sh.^2),50)
xlabel('p(elppd(y)>WAIC(y)) theoretical with hat sigma')
print('-depsc2',sprintf('m%d_normalz_%d_theorhatpWAIC.eps',mi,n))
% n future points
qqplot(std(ltsts,0,2)*sqrt(n),std(waics,0,2)*sqrt(n))
xlabel('std(n independent test points) quantiles')
ylabel('std(waic) quantiles')
print('-depsc2',sprintf('m%d_normalz_%d_tstWAIC.eps',mi,n))
% hist(ltst-waic,50)
% hist(std(ltrs(:,randperm(n))-waics,0,2),50)
% sqrt(var(ltst-waic)-mean(var(waics,0,2))*n)
% hist(normcdf(ltst-waic,0,sqrt(2)*std(waics,0,2)*sqrt(n)),50)

sqrt((n+3+4/n)/(2+4/n))
% 2.3629
std(ltst0-waic)
% 2.4066
std(mean(ltst0)-waic)
% 2.3197
mean(std(waics,0,2)*sqrt(n))
% 1.9367
mean(n./(n-p_waic).*std(waics,0,2)*sqrt(n))
% 2.1924

% for i1=1:10000
%   qp(i1,1)=prctile(sqrt(sinvchi2rand(n-1,var(waics(i1,:)),10000,1)),95);
% end

% smu=bsxfun(@plus,mu,randn(1,10000)*sqrt(1+1/n));
% for i1=1:10000
%   logls=norm_lpdfs(y(i1,:),smu(i1,:)',s);
% end

% for i1=1:10000
% for i2=1:n
%   q1(i2)=integral(@(wmu) t_pdfs(wmu,n-1,mu(i1),sh(i1).*sqrt(1/n)).*norm_lpdfs(y(i1,i2)',wmu,s),mu(i1)-6,mu(i1)+6);
%   q2(i2)=integral(@(wmu) norm_pdfs(wmu,mu(i1),sqrt(1/n)).*(q1(i2)-norm_lpdfs(y(i1,i2),wmu,s)).^2,mu(i1)-6,mu(i1)+6);
% end
% p_waics(i1,:)=q2;
% p_waic(i1,1)=sum(q2);
% end
% waics=ltrs-p_waics;
% waic=ltr-p_waic;
%
hist(ltst0-waic,50)
% std(waics)
hist(normcdf(ltst0-waic,-1*(n-1)/2/n/(n+1),std(waics,0,2)*sqrt(n)),50)
hist(normcdf(mean(ltst0)-waic,0,std(waics,0,2)*sqrt(n)),50)
hist(tcdf((ltst0-waic)./(std(waics,0,2)*sqrt(n)),n-1),50)
qii=ones(size(waic));
qii=(ltst0-waic)>-2;
qii=std(waics,0,2)*sqrt(n)>1.05;
qii=Ef>-.2;
qii=(kurtosis(waics,1,2)-3*skewness(waics,1,2))<8;
hist(normcdf(dlta(qii),0,std(waics(qii,:),0,2)*sqrt(n)),50)
hist(normcdf(dlta(~qii),0,std(waics(~qii,:),0,2)*sqrt(n)),50)
% corrected? std(waics)
hist(normcdf(ltst0-waic,0,n./(n-p_waic).*std(waics,0,2)*sqrt(n)),50)
% bbmean
hist(mean(bsxfun(@ge,ltst0,bbm),2),50)
hist(mean(bsxfun(@ge,ltst0(dlta>0),bbm(dlta>0,:)),2),50)
% theoretical with true sigma
hist(normcdf(ltst0-waic,-1*(n-1)/2/n/(n+1),sqrt((n+3+4/n)/(2+4/n))),50)
% theoretical with hat sigma
hist(normcdf(ltst0-waic,-1*(n-1)/2/n/(n+1),sqrt((n+3+4/n)/(2+4/n)).*sh.^2),50)
% n future points
qqplot(std(ltsts,0,2)*sqrt(n),std(waics,0,2)*sqrt(n))
%
hist(ltst-waic,50)
sqrt(var(ltst-waic)-mean(var(waics,0,2))*n)
%
hist(normcdf(ltst-waic,0,sqrt(2)*std(waics,0,2)*sqrt(n)),50)
hist(normcdf(ltst-waic,0,sqrt(2)*std(waics,0,2)*sqrt(n)),50)

plot(std(waics,0,2),(ltst0-waic),'.')
plot((ltst0),waic,'.')

 for i1=1:10000
%   qp(i1,1)=prctile(sqrt(sinvchi2rand(n-1,var(waics(i1,:)),10000,1)),95);
% end

% smu=bsxfun(@plus,mu,randn(1,10000)*sqrt(1+1/n));
% for i1=1:10000
%   logls=norm_lpdfs(y(i1,:),smu(i1,:)',s);
% end

%% comparison
n=10;
q1=load(sprintf('m1_normalz_n%d',n),'ltsts','ltst','ltst0','pwaics','waic','waics','bbm');
q2=load(sprintf('m2_normalz_n%d',n),'ltsts','ltst','ltst0','pwaics','waic','waics','bbm');
figure(14)
a4(.33,.5)
plot(q1.ltst0,q2.ltst0,'.'),diagline
xlabel('elppd(y,M_1)')
ylabel('elppd(y,M_2)')
print('-depsc2',sprintf('normald_n%d_elppd.eps',n))
plot(q1.waic,q2.waic,'.'),diagline
xlabel('WAIC(M_1)')
ylabel('WAIC(M_2)')
print('-depsc2',sprintf('normald_n%d_waic.eps',n))
plot(q1.ltst0-q2.ltst0,q1.waic-q2.waic,'.')
xlabel('elppd(y,M_1)-elppd(y,M_2)')
ylabel('WAIC(M_1)-WAIC(M_2)')
print('-depsc2',sprintf('normald_n%d_elppd_vs_waic.eps',n))
plot(std(q1.waics-q2.waics,0,2)*sqrt(n),(q1.ltst0-q2.ltst0)-(q1.waic-q2.waic),'.')
xlabel('std(WAIC(y,M_1)-WAIC(y,M_2))')
ylabel('(elppd(y,M_1)-elppd(y,M_2))-(WAIC(M_1)-WAIC(M_2))')
print('-depsc2',sprintf('normald_n%d_stdWAIC_vs_error.eps',n))

hist(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)),50)
xlabel('p(elppd(y,M_1)>elppd(y,M_2)) using Gaussian approximation')
print('-depsc2',sprintf('normald_n%d_pWAIC.eps',n))
hist(mean(bsxfun(@ge,q1.ltst0-q2.ltst0,q1.bbm-q2.bbm),2),50)
xlim([0 1])
xlabel('p(elppd(y,M_1)>elppd(y,M_2)) using BB approximation')
print('-depsc2',sprintf('normald_n%d_bbpWAIC.eps',n))

%%
mean(std(qn10.waics,0,2)*sqrt(10))
%1.9
mean(std(q2n10.waics,0,2)*sqrt(10))
%2.15
mean(std(qn10.waics-q2n10.waics,0,2)*sqrt(10))
%0.79
mean(std(qn100.waics,0,2)*sqrt(100))
%6.95
mean(std(q2n100.waics,0,2)*sqrt(100))
%7.09
mean(std(qn100.waics-q2n100.waics,0,2)*sqrt(100))
%0.81

%sqrt(1/2)
subplot(211),hist(normcdf(qn10.ltst0-q2n10.ltst0,qn10.waic-q2n10.waic,std(qn10.waics-q2n10.waics,0,2)*sqrt(n)),50)
subplot(212),hist(mean(bsxfun(@ge,qn10.ltst0-q2n10.ltst0,qn10.bbm-q2n10.bbm),2),50)

figure(2)
subplot(311),hist(qn10.ltst0-q2n10.ltst0,50)
subplot(312),hist(qn10.waic-q2n10.waic,50)
subplot(313),hist(qn10.ltst0-q2n10.ltst0-(qn10.waic-q2n10.waic),50)

figure(1)
subplot(311),plot(qn100.ltst0,q2n100.ltst0,'.')
subplot(312),plot(qn100.waic,q2n100.waic,'.')
subplot(313),plot(qn100.ltst0-q2n100.ltst0,qn100.waic-q2n100.waic,'.')

figure(2)
subplot(311),hist(qn100.ltst0-q2n100.ltst0,50)
subplot(312),hist(qn100.waic-q2n100.waic,50)
subplot(313),hist(qn100.ltst0-q2n100.ltst0-(qn100.waic-q2n100.waic),50)

%%%% zero mean

clear
n=40;
MU0=[0 .5 1 1.5 2 2.5 3];
for mui=1:numel(MU0);
  fprintf('mui=%d\n',mui)
  clear('ltsts','ltst','ltst0','pwaics','waic','waics','bbm')
  %data
  setrandstream(1);
  y=randn(1000,n);
  yt=randn(1000,n);
  mu0=MU0(mui);
  y=y+mu0;
  yt=yt+mu0;
  %N(0,sigma^2)
  mu1=0*mean(y,2);
  sh1=sqrt(mean(y.^2,2));
  s=1;
  ps1=sqrt(1);
  ltrs=t_lpdfs(y,n,mu1,sh1.*ps1);
  ltr=sum(ltrs,2);
  ltsts=t_lpdfs(yt,n,mu1,sh1.*ps1);
  ltst=sum(ltsts,2);
  for i1=1:1000
    ltst0(i1,1)=n*integral(@(wy) norm_pdfs(wy,mu0,1).*t_lpdfs(wy,n,mu1(i1),sh1(i1).*ps1),mu0+-6,mu0+6);
  end
  ss=bsxfun(@times,sqrt(sinvchi2rand(n,1,1000,1000)),sh1);
  smu=0;
  for i1=1:1000
    logliks=norm_lpdfs(y(i1,:),smu,ss(i1,:)');
    pwaics(i1,:)=var(logliks);
  end
  clear ss
  waics=ltrs-pwaics;
  waic=sum(waics,2);
  setrandstream(1);
  bbm=bbmean(waics',1000)'*n;
  save(sprintf('m1_normalnz_n%d_mu%d',n,mui),'ltsts','ltst','ltst0','pwaics','waic','waics','bbm');
  %N(theta,sigma^2)
  mu2=mean(y,2);
  sh2=std(y,0,2);
  s=1;
  ps2=sqrt(1+1/n);
  ltrs=t_lpdfs(y,n-1,mu2,sh2.*ps2);
  ltr=sum(ltrs,2);
  ltsts=t_lpdfs(yt,n-1,mu2,sh2.*ps2);
  ltst=sum(ltsts,2);
  for i1=1:1000
    ltst0(i1,1)=n*integral(@(wy) norm_pdfs(wy,mu0,1).*t_lpdfs(wy,n-1,mu2(i1),sh2(i1).*ps2),mu0-6,mu0+6);
  end
  ss=bsxfun(@times,sqrt(sinvchi2rand(n-1,1,1000,1000)),sh2);
  smu=bsxfun(@plus,mu2,randn(1000,1000).*ss.*sqrt(1/n));
  for i1=1:1000
    logliks=norm_lpdfs(y(i1,:),smu(i1,:)',ss(i1,:)');
    pwaics(i1,:)=var(logliks);
  end
  clear ss smu
  waics=ltrs-pwaics;
  waic=sum(waics,2);
  setrandstream(1);
  bbm=bbmean(waics',1000)'*n;
  save(sprintf('m2_normalnz_n%d_mu%d',n,mui),'ltsts','ltst','ltst0','pwaics','waic','waics','bbm');
  tic
  for i1=1:1000
    setrandstream(i1);
    yr1=trnd(n,1000,1).*sh1(i1)*ps1;
    setrandstream(i1);
    yr2=trnd(n-1,1000,1).*sh2(i1)*ps2+mu2(i1);
    lrs11=t_lpdfs(yr1,n,0,sh1(i1));lrs12=t_lpdfs(yr1,n-1,mu2(i1),sh2(i1).*sqrt(1+1/n));
    lrs21=t_lpdfs(yr2,n,0,sh1(i1));lrs22=t_lpdfs(yr2,n-1,mu2(i1),sh2(i1).*sqrt(1+1/n));
    rstd1112(i1,1)=std(lrs11-lrs12);
    rstd1122(i1,1)=std(lrs11-lrs22);
    rstd2122(i1,1)=std(lrs21-lrs22);
  end
  toc
  save(sprintf('m12_normalnz_n%d_mu%d',n,mui),'rstd1112','rstd1122','rstd2122');
end

n=100;
MU0=[0 .5 1 1.5 2 2.5 3];
for mui=1:numel(MU0);
  load(sprintf('normal_n%d',n),'y','yt')
  mu0=MU0(mui);
  y=y+mu0;
  yt=yt+mu0;
  pth=tcdf(-mu2./sh2*sqrt(n),n-1);
  mu2=mean(y,2);
  sh2=std(y,0,2);
end

% for mui=1:5
%   fprintf('mui=%d\n',mui)
%   q1=load(sprintf('m1_normalnz_n10_mu%d',mui));
%   setrandstream(1);
%   bbm=bbmean(q1.waics',10000)'*n;
%   save(sprintf('m1_normalnz_n10_mu%d',mui),'bbm','-append');
%   q2=load(sprintf('m2_normalnz_n10_mu%d',mui));
%   setrandstream(1);
%   bbm=bbmean(q2.waics',10000)'*n;
%   save(sprintf('m2_normalnz_n10_mu%d',mui),'bbm','-append');
% end
figure(8)
a4(.33,.5)
qp=[0.0060 0.1230 0.7350 0.9840 0.9970 1 1];
plot(MU0,qpn10,MU0,qpn20,MU0,qpn40,MU0,qpn100)
h=plot(MU0,qpnn10,'--',MU0,qpnn20,'--',MU0,qpnn40,'--',MU0,qpnn100,'--')
xlabel('mu')
ylabel('p(M2>M1|WAIC comparison)')

n=100;
for mui=1:7
    q1=load(sprintf('m1_normalnz_n%d_mu%d',n,mui));
    q2=load(sprintf('m2_normalnz_n%d_mu%d',n,mui));
    err1=(q1.ltst0);
    err2=(q2.ltst0);
    qw=[exp(q1.waic) exp(q2.waic)];
    qw=bsxfun(@rdivide,qw,sum(qw,2));
    mean(err1)
    mean(err2)
    mean(err1.*qw(:,1)+err2.*qw(:,2))
    qii=((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))<0.05);
    qii=((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))<0.95);
    mean([err1(~qii); err2(qii)])
    mean(err1)
    mean(err2)
    mean(err1.*qw(:,1)+err2.*qw(:,2))
    %qpn10(mui)=mean((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))<0.05);
    %qpnn100(mui)=mean((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))<0.95);
end
    

n=100;
for mui=1:7
  setrandstream(1);
  y=randn(1000,n);
  yt=randn(1000,n);
  mu0=MU0(mui);
  y=y+mu0;
  yt=yt+mu0;
  pth=tcdf(-mu2./sh2*sqrt(n),n-1);
  mu2=mean(y,2);
  sh2=std(y,0,2);
  q1=load(sprintf('m1_normalnz_n%d_mu%d',n,mui));
  q2=load(sprintf('m2_normalnz_n%d_mu%d',n,mui));
  %load(sprintf('m12_normalnz_n%d_mu%d',n,mui));
  err=(q1.ltst0-q2.ltst0)-(q1.waic-q2.waic);
  figure(1)
  a4(.33,.5)
  scatter(q1.ltst0,q2.ltst0,12,err,'filled'),diagline
  mean(q1.ltst0>q2.ltst0)
  xlabel('elppd(y,M_1)')
  ylabel('elppd(y,M_2)')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_elppd.eps',n,mui))
  figure(2)
  a4(.33,.5)
  scatter(q1.waic,q2.waic,12,err,'filled'),diagline
  mean(q1.waic>q2.waic)
  xlabel('WAIC(M_1)')
  ylabel('WAIC(M_2)')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_waic.eps',n,mui))
  figure(3)
  a4(.33,.5)
  scatter(q1.ltst0-q2.ltst0,q1.waic-q2.waic,12,err,'filled'),diagline
  xlabel('elppd(y,M_1)-elppd(y,M_2)')
  ylabel('WAIC(M_1)-WAIC(M_2)')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_elppd_vs_waic.eps',n,mui))
  figure(4)
  a4(.33,.5)
  scatter(std(q1.waics-q2.waics,0,2)*sqrt(n),(q1.ltst0-q2.ltst0)-(q1.waic-q2.waic),12,err,'filled')
  xlabel('std(WAIC(y,M_1)-WAIC(y,M_2))*sqrt(n)')
  ylabel('(elppd(y,M_1)-elppd(y,M_2))-(WAIC(M_1)-WAIC(M_2))')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_stdWAIC_vs_error.eps',n,mui))
  figure(5)
  a4(.33,.5)
  hist(1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)),50)
  %mean(1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))
  %mean((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))>0.05)
  mean((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))<0.05)
  xlabel('p(WAIC(M_1)>WAIC(M_2)) using Gaussian approximation')
  xlim([0 1])
  figure(7)
  a4(.33,.5)
  plot([0:.01:1],mean(bsxfun(@lt,(1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n))),[0:.01:1]),1),[0:.01:1],mean(bsxfun(@lt,1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2),[0:.01:1]),1));
  ylim([0 1])
  mean((1-normcdf(0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))>0.05)
  diagline
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_pWAIC.eps',n,mui))
  figure(6)
  a4(.33,.5)
  hist(1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2),50)
  %mean(1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2))
  %mean((1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2))>0.05)
  %mean((1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2))>0.5)
  xlim([0 1])
  xlabel('p(WAIC(M_1)>WAIC(M_2)) using BB approximation')
  % h=line(repmat(normcdf(-MU0(mui)) ,2,1),ylim,'color','r');
  % text(normcdf(-MU0(mui))+.02,max(ylim)*0.95,sprintf('\\Phi(%.1f)',-MU0(mui)))
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_bbpWAIC.eps',n,mui))
end
  figure(7)
  a4(.33,.5)
  hist(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)),50)
  mean(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))
  xlabel('p(delta(elppd(y))>delta(WAIC)) using Gaussian approximation')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_pdWAIC.eps',n,mui))
  yl7=ylim;
  figure(8)
  a4(.33,.5)
  hist(mean(bsxfun(@ge,q1.ltst0-q2.ltst0,q1.bbm-q2.bbm),2),50)
  %mean(mean(bsxfun(@ge,q1.ltst0-q2.ltst0,q1.bbm-q2.bbm),2))
  xlim([0 1])
  xlabel('p(delta(elppd(y))>delta(WAIC)) using BB approximation')
  % h=line(repmat(normcdf(-MU0(mui)) ,2,1),ylim,'color','r');
  % text(normcdf(-MU0(mui))+.02,max(ylim)*0.95,sprintf('\\Phi(%.1f)',-MU0(mui)))
  yl8=ylim;
  ylim([0 max([yl7(2) yl8(2)])])
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_bbpdWAIC.eps',n,mui))
  figure(7);
  ylim([0 max([yl7(2) yl8(2)])])
  figure(9)
  a4(.33,.5)
  %qii=(1-mean(bsxfun(@ge,0,q1.bbm-q2.bbm),2))>0.05;
  %hist(q1.waic(~qii)-q2.waic(~qii),20)
  hist(q1.waic-q2.waic,50)
  xlabel('WAIC(M_1)-WAIC(M_2)')
  prctile([q1.waic-q2.waic],[1 5 50 95 99])
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_histdWAIC.eps',n,mui))
   figure(10)
   a4(.33,.5)
   scatter(sh2,err,12,err,'filled')
   xlabel('Sample std')
   ylabel('Error')
  % xlabel('\hat{\sigma_2}|y,M_2')
  % ylabel('err')
  figure(11)
  a4(.33,.5)
  scatter(q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n),12,err,'filled')
   xlabel('WAIC(M_1)-WAIC(M_2)')
   ylabel('std(WAIC(M_1)-WAIC(M_2))*sqrt(n)')
  figure(12)
  a4(.33,.5)
  %qii=abs(q1.waic-q2.waic)>2;
  %hist(normcdf(q1.ltst0(qii)-q2.ltst0(qii),q1.waic(qii)-q2.waic(qii),std(q1.waics(qii,:)-q2.waics(qii,:),0,2)*sqrt(n)),50)
  hist(normcdf(mean(q1.ltst0-q2.ltst0),q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)),50)
  mean(normcdf(mean(q1.ltst0-q2.ltst0),q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n)))
  xlabel('p(delta(elppd(y))>delta(WAIC)) using Gaussian approximation')
  %print('-depsc2',sprintf('normalnzd_n%d_mu%d_pedWAIC.eps',n,mui))
  figure(13)
  a4(.33,.5)
  hist(mean(bsxfun(@ge,mean(q1.ltst0-q2.ltst0),q1.bbm-q2.bbm),2),50)
  mean(mean(bsxfun(@ge,mean(q1.ltst0-q2.ltst0),q1.bbm-q2.bbm),2))
  xlim([0 1])
  xlabel('p(delta(elppd(y))>delta(WAIC)) using BB approximation')
  print('-depsc2',sprintf('normalnzd_n%d_mu%d_bbpedWAIC.eps',n,mui))
  % figure(10)
  % a4(.33,.5)
  % hist(1-normcdf(0,q1.waic-q2.waic,rstd2122*sqrt(n)),50)
  % xlabel('p(WAIC(M_1)>WAIC(M_2)) using Gaussian approximation')
  % xlim([0 1])
  % figure(11)
  % a4(.33,.5)
  % hist(1-normcdf(0,q1.waic-q2.waic,max(rstd2122,std(q1.waics-q2.waics,0,2))*sqrt(n)),50)
  % mean((1-normcdf(0,q1.waic-q2.waic,max(rstd2122,std(q1.waics-q2.waics,0,2))*sqrt(n)))>0.5)
  % xlim([0 1])
  % xlabel('p(WAIC(M_1)>WAIC(M_2)) using BB approximation')
  % figure(12)
  % a4(.33,.5)
  % hist(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,rstd2122*sqrt(n)),50)
  % hist(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n*2)),50)
  % xlabel('p(delta(elppd(y))>delta(WAIC)) using Gaussian approximation')
  % figure(13)
  % a4(.33,.5)
  % hist(normcdf(q1.ltst0-q2.ltst0,q1.waic-q2.waic,max(rstd2122,std(q1.waics-q2.waics,0,2))*sqrt(n)),50)
  % xlim([0 1])
  % xlabel('p(delta(elppd(y))>delta(WAIC)) using BB approximation')
   fprintf('paused\n')
   keyboard
   fprintf('continued\n')
end
%n10
%0.0: 0.8332 / 0.8181
%0.5: 0.1275 / 0.4346
%1.0: 0.0135 / 0.0537
%1.5: 0.0045 / 0.0013
%2.0: 0.0015 / 0.0001

%  n2s(prctile([q1.waic-q2.waic],[1 5 50 95 99]),'%4.2f ')
  %0.0:  -2.8299   -1.2757    0.8283    1.1390    1.2095
  %0.5:  -5.7896   -3.7263   -0.2430    1.0928    1.1682
  %1.0:  -9.1996   -6.9276   -2.7421    0.0514    0.8242
  %1.5: -11.9470   -9.6627   -5.2492   -2.0626   -0.9085
  %2.0: -14.2050  -11.9357   -7.4109   -4.0659   -2.8662

%0.0: 0.8413
%0.5: 1.0000e-04
%1.0: 0
%1.5: 0
%2.0: 0

  %  prctile([q1.waic-q2.waic],[1 5 50 95 99])
  %0.0:  -2.1991   -0.9199    0.7814    1.0239    1.0493
  %0.5: -22.1320  -18.1273  -10.2399   -4.3254   -2.3965
  %1.0:  -9.1996   -6.9276   -2.7421    0.0514    0.8242
  %1.5: -11.9470   -9.6627   -5.2492   -2.0626   -0.9085
  %2.0: -14.2050  -11.9357   -7.4109   -4.0659   -2.8662
  figure(16)
scatter(q1.waic-q2.waic,std(q1.waics-q2.waics,0,2)*sqrt(n),12,err,'filled')

cov([q1.waics(i1,:)' q2.waics(i1,:)'])