unalista([],[]):-!.

unalista([X|Cola],[X|R]):-not(lista(X)),unalista(Cola,R),!.

unalista([X|Cola],R):-lista(X),unalista(X,R2) ,unalista(Cola,R1),
    juntar_listas(R2,R1,R).


juntar_listas([X|L1],L2,[X|F]):-juntar_listas(L1,L2,F).
juntar_listas([],L2,L2):-!.


lista([]):-!.
lista([_|Cola]):-lista(Cola).
