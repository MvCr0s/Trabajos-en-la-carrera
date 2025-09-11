up(s1).
down(s2).
up(s3).
live(outside):-!.
delay(ok(_)).

ligth(l1).
ligth(l2).
enchufe(p1).
enchufe(p2).

conectado(outside,w5).
conectado(outside,w3):-ok(cb1).

conectado(w5,w1):-ok(cb1),ok(s1),up(s1).
conectado(w5,w2):-ok(cb1),ok(s1),down(s1).

conectado(w1,w0):-ok(s2),up(s2).
conectado(w2,w0):-ok(s2),down(s2).
conectado(w0,l1).

conectado(w3,w4):-ok(s3),up(s3).
conectado(w4,l2).
conectado(w3,p1).

conectado(w5,w6):-ok(cb2).
conectado(w6,p2).


live(X):- conectado(Y,X),live(Y).
lit(X):- ok(X),live(X),ligth(X),!.
enchufePotencia(X):-enchufe(X),ok(X),live(X),!.


dsolve(true,D,D):- !.
dsolve((A,B), D1,D2) :- !, dsolve(A,D1,D3), dsolve(B,D3,D2).
dsolve(A,D,D):-delay(A), member(A,D),!.
%para el permiso
dsolve(!,D,D):-!.

dsolve(A,D,[A|D]) :- delay(A).
dsolve(A,D1,D2) :- clause(A, B), dsolve(B,D1,D2).



dsolve_traza(true,D,D):- !.
dsolve_traza((A,B), D1,D2) :- !, dsolve_traza(A,D1,D3), dsolve_traza(B,D3,D2).
%para el permiso
dsolve_traza(!,D,D):-!.

dsolve_traza(A,D,[A|D]) :- delay(A).
dsolve_traza(A,D1,D2) :- clause(A, B), dsolve_traza(B,D1,D2).
