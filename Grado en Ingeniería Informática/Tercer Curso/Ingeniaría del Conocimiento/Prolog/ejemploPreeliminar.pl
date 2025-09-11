dedo(indice):- magia(activada).
luce(X) :- dedo(X),magia(activada).
delay(magia(activada)).
dsolve(true,D,D):- !.




% Base case: solve "true".
dsolve(true, D, D) :- !.

% Solve a conjunction (A, B), solve A first, then B.
dsolve((A, B), D1, D2) :- !,
    dsolve(A, D1, D3),
    dsolve(B, D3, D2).

dsolve(A,D,D):-delay(A), member(A, D),!.

% If A should be delayed, check if it's already in the list of delayed goals.
% If not, delay it and add it to the list.
dsolve(A, D, [A|D]) :-
    delay(A).

% General case for solving a goal A at depth D.
dsolve(A, D1, D2) :-
    clause(A, B),
    dsolve(B, D1, D2).




dsolve_traza(true,D,D):- !.
dsolve_traza((A,B), D1,D2) :- !, dsolve_traza(A,D1,D3), dsolve_traza(B,D3,D2).
dsolve_traza(A,D,[A|D]) :- delay(A).
dsolve_traza(A,D1,D2) :- clause(A, B), dsolve_traza(B,D1,D2).
