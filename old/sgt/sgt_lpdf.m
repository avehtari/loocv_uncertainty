function ld = sgt_lpdf(x, mu, s, l, p, q)
% SGT_PDF - 
%   

%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-11-02 20:39:52 EET

lz1=lbeta(1/p,q);
lz2=lbeta(2/p,q-1/p);
v=q^(-1/p)*((3*l^2+1)*exp(lbeta(3/p,q-2/p)-lz1)-4*l^2*exp(lz2-lz1)^2)^(-1/2);
m=2*v*s*l*q^(1/p)*exp(lz2-lz1);
r=x-mu+m;
ld=log(p)-log(2*v*s*q^(1/p)*exp(lz1)*(abs(r).^p./(q*(v*s).^p*(l*sign(r)+1).^p)+1).^(1/p+q));

function lb = lbeta(a,b)
lb=log(beta(a,b));
