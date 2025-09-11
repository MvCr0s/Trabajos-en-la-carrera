
%Comprobamos que a=b
caso(q0, a, [], q0, [a]).
caso(q0, a, Pila, q0, [a|Pila]).
caso(q0, b , [a|Pila], q1, Pila).
caso(q1, b , [a|Pila], q1, Pila).
caso(q1,c,Pila,q2,Pila).
caso(q2,c,Pila,q2,Pila).

%Comprobamos que b=c
caso(q3, a, [], q3, []).
caso(q3, b, [], q4, [b]).
caso(q4, b , Pila, q4, [b|Pila]).
caso(q4, c , [b|Pila], q5, Pila).
caso(q5 ,c , [b|Pila], q5,Pila).

caso(_, [], _, _):-
	!,
	abort.



transita(q5,[],[],qf):-!.
transita(q2,[],[],qf):-!.
transita(Qi,[X|Y],Pi,Q):-caso(Qi,X,Pi,Qf,Pn),transita(Qf,Y,Pn,Q).

acepta(X,Resultado):-transita(q0,X,[],Q1), Q1=qf,
    transita(q0,X,[],Q2), Q2=qf,
    Resultado is 1,!.


solve_traza(true, _).

solve_traza((A, B), Nivel) :-
    solve_traza(A, Nivel),
    solve_traza(B, Nivel).

solve_traza(A, Nivel) :-
    predicate_property(A, built_in),
    !,
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    call(A).

solve_traza(A, Nivel) :-
    clause(A, B),
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    NivelNuevo is Nivel + 1,
    solve_traza(B, NivelNuevo).


solve_pmaxInv(A,Prof):-solve_pmaxInv(A,1,Prof).

solve_pmaxInv(true,_,_).


solve_pmaxInv((A,B),Nivel,Prof):-
    solve_pmaxInv(B,Nivel,Prof),
    solve_pmaxInv(A,Nivel,Prof).


solve_pmaxInv(A,Nivel,_):-
    Tab is Nivel * 3,
    format("~t~*|Llamada : ~w~n",[Tab,A]),call(A).

solve_pmaxInv(A,Nivel,Prof):-
    Prof>0, clause(A,B),
    Tab is Nivel * 3,
    format("~t~*|Llamada : ~w~n",[Tab,A]),call(A),
    Nivel2 is Nivel +1,
    Prof2 is Prof-1,
    solve_pmaxInv(B,Nivel2,Prof2).

solve_pmaxInv(_,_,0):-write('Limite de profundidad alcanzado'),nl,abort.
