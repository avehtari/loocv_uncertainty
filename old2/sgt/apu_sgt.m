%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-11-03 09:42:42 EET

opt=sls_opt;
q=randn(10,1);
opt.nsamples=10000;
opt.nomit=10;
opt.mmlimits=[-2 -1 -1 -1 0; ...
               2 1 1 2 8];
opt.method='minmax';
q=loos(1,:)';
qr=sls(@sgte, [0, 0, 0, log(2), log(50)], opt, [], q);
qr=thin(qr,100,10);
% subplot(311)
% cla
clear qp
for qi=1:size(qr,1)
    qp(qi,:)=sgt_pdf(qx,qr(qi,1),exp(qr(qi,2)),qr(qi,3),exp(qr(qi,4)),exp(qr(qi,5))*2/exp(qr(qi,4)));
    % plot(qx,qp(qi,:));
    % hold on
end
%subplot(212)
clf
plot(qx,mean(qp))
%subplot(212)
hold on
[qpp,~,qxx]=lgpdens(qr(:,1));
plot(qxx,qpp)

ni=1;

0, 0, 0, log(2), log(50)

qx=linspace(-4,4,100);plot(qx,sgt_pdf(qx,1,1,1,2,100))

addpath ~ave/matlab/MatlabProcessManager
addpath ~ave/matlab/MatlabStan
setenv('LD_LIBRARY_PATH','/usr/lib')
n=100;
dat=struct('N',n,'x',x-mean(x));
init=struct('mu',0,'sigma',std(x),'l',-.1,'p',2,'q',50);
% go nuts
fit = stan('file','sgt.stan','data',dat,'init',init,'sample_file','sgt','file_overwrite',true,'verbose',true,'iter',8000);
fit.block()
s = fit.extract('permuted',true)
s = fit.extract('permuted',false);
plot(s.mu)
plot(s.sigma)
plot(s.l)
plot(s.p)
plot(s.q)

subplot(211)
histogram(qbbm(1,:,1),50)
subplot(212)
histogram((s.mu+mean(x))*n,50)
qqplot(qbbm(1,:,1),(s.mu+mean(x))*n)
clf;hold on;for i1=1:4, plot(s(i1).mu), end
clf;hold on;for i1=1:4, plot(s(i1).sigma), end
clf;hold on;for i1=1:4, plot(s(i1).l), end
clf;hold on;for i1=1:4, plot(s(i1).p), end
clf;hold on;for i1=1:4, plot(s(i1).q), end
