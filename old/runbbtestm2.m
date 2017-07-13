%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-10-24 21:53:24 EDT

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
save bbtestm2 qps qmss
