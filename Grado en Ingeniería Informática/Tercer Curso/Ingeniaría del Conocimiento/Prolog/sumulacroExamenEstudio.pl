%Comprobamos que a=b
caso(e0, a, [], e0, [a]).
caso(e0, a, Pila, e0, [a|Pila]).
caso(e0, b , [a|Pila], e1, Pila).
caso(e1, b , [a|Pila], e1, Pila).
caso(e1,c,Pila,e2,Pila).
caso(e2,c,Pila,e2,Pila).

%Comprobamos que b=c
caso(e3, a, [], e3, []).
caso(e3, b, [], e4, [b]).
caso(e4, b , Pila, e4, [b|Pila]).
caso(e4, c , [b|Pila], e5, Pila).
caso(e5 ,c , [b|Pila], e5,Pila).





mueve(e2, [], [], eF).
mueve(e5, [], [], eF).

mueve(Ei, [Li|L], Pi, Ef):-
	caso(Ei, Li, Pi, En, Pn),
	mueve(En, L, Pn, Ef).

comprueba(L, Res):-
	mueve(e0, L, [], Ef),
	Ef = eF,
	Res is 1,
	!.


solve_traza(A):-solve_traza(A,1).

solve_traza(true, _).
solve_traza((A,B), Nivel):-
    solve_traza(A,Nivel),
    solve_traza(B,Nivel).

solve_traza(A,Nivel):-
    predicate_property(A, built_in),!,
    Tab is Nivel * 3,
    format("~t~*|llamada: ~w~n",[Tab, A]), call(A).

solve_traza(A,Nivel):-
    clause(A,B),
    Tab is Nivel * 3,
    format("~t~*|llamada: ~w~n",[Tab, A]),
    NivelNuevo is Nivel+1,
    solve_traza(B, NivelNuevo).

