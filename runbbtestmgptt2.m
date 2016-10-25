%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-10-25 17:49:32 EDT

clear
N=[10 20 30 50 70 100];
gp=gp_set('cf',{gpcf_constant('constSigma2',1,'constSigma2_prior',[]) gpcf_linear('coeffSigma2',1,'coeffSigma2_prior',[])},'lik',lik_t);
gp.cf{1}

%% unknown variance %%
for ni=1:numel(N)
    n=N(ni);
    fprintf('N=%d\n',n)
    %load(sprintf('normal_n%d',n),'y','yt')
    y=randn(1000,n);
    x=randn(1000,n);
    yt=randn(1000,n);
    ytt=randn(100000,1);
    xtt=randn(100000,1);
    clear loos pks ltrs
    for i1=1:1000
        gpia=gp_ia(gp,x(i1,:)',y(i1,:)','int_method','grid','display','off');
        [~,~,ltst00]=gp_pred(gpia,x(i1,:)',y(i1,:)',xtt,'yt',ytt);
        ltst0(i1,:)=mean(ltst00)*n;
        [~,~,ltrs(i1,:)]=gp_pred(gpia,x(i1,:)',y(i1,:)');
        [~,~,loos(i1,:)]=gp_loopred(gpia,x(i1,:)',y(i1,:)');
    end
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
save bbtestmgp2 qps qmss
