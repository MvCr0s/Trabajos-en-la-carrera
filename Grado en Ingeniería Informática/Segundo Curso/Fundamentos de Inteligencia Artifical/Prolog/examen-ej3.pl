burbuja(L,L):-ordenada(L).

burbuja(L,Lord) :- permuta(L,L1),burbuja(L1,Lord),!.

ordenada([]).
ordenada([_]).
ordenada([X,Y|Cola]):- X=<Y,ordenada([Y|Cola]).


permuta([X, Y|R], [Y, X|R]):- X>Y.
permuta([X,Y|R], [X|R1]):-X=<Y,permuta([Y|R], R1).
