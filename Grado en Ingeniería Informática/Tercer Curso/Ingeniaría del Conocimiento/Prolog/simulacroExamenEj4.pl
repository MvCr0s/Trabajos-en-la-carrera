caso(e0, a, [], e0, [a]).
caso(e0, a, Pila, e0, [a|Pila]).
caso(e0, b , [a|Pila], e1, Pila).
caso(e1, b , [a|Pila], e1, Pila).
caso(e1, b , [], e1, [b]).
caso(e1, b , Pila, e1, [b|Pila]).
caso(e1, c , [b|Pila], e2, Pila).
caso(e2, c , [b|Pila], e2, Pila).




mueve(e2, [], [], eF).
mueve(Ei, [Li|L], Pi, Ef):-
	caso(Ei, Li, Pi, En, Pn),
	mueve(En, L, Pn, Ef).

comprueba(L, Res):-
	mueve(e0, L, [], Ef),
	Ef = eF,
	Res is 1,
	!.


solve_pmaxInv(A,Prof):-solve_pmax(A,1,Prof).

solve_pmaxInv(true,_,_).

solve_pmaxInv((A,B),Nivel,Prof):-
    solve_pmaxInv(B,Nivel,Prof),
    solve_pmaxInv(A,Nivel,Prof).

solve_pmaxInv(A,Nivel,_):-
    predicate_property(A, built_in),!,
    Tab is Nivel * 3,
    format("~t~*|LLamada: ~w~n", [Tab,A]),
    call(A).
solve_pmaxInv(A,Nivel,Prof):-
    Prof>0,clause(A,B),
    Tab is Nivel*3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    NivelNuevo is Nivel+1,
    ProfNuevo is Prof-1,
    solve_pmaxInv(B,NivelNuevo,ProfNuevo).

solve_pmaxInv(_,_,0):-write('Limite de profundidad alcanzado'),nl,abort.
