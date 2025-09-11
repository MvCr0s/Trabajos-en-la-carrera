%q(X,f(X)).

rr(f(X,Y),g(Z)):- rr(Y,Z).
rr(a,a).

p(X,Y,f(a)):-p(X,X,Y).
p(X,X,X).
