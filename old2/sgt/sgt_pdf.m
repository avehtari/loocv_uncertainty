function d = sgt_pdf(x, mu, s, l, p, q)
% SGT_PDF - 
%   

%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-11-02 20:45:57 EET

z1=beta(1./p,q);
z2=beta(2./p,q-1./p);
v=q.^(-1./p).*((3*l.^2+1).*(beta(3./p,q-2./p)./z1)-4*l.^2.*(z2./z1).^2).^(-1/2);
m=2*v.*s.*l.*q.^(1/p).*z2./z1;
d=p./(2*v.*s.*q.^(1/p).*z1.*(abs(x-mu+m).^p./(q.*(v.*s).^p.*(l.*sign(x-mu+m)+1).^p)+1).^(1./p+q));
