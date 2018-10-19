function e = sgte(w, x)
%   Author: Aki Vehtari <Aki.Vehtari@aalto.fi>
%   Last modified: 2016-10-25 18:21:14 EDT

mu=w(1);
s=exp(w(2));
l=w(3);
p=exp(w(4));
q=exp(w(5))*2/p;
e=-sum(log(sgt_pdf(x, mu, s, l, p, q)));
