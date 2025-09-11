
caso(q0,a,[],q0,[a]).
caso(q0,b,[],q0,[b]).
caso(q0,a,Pila,q0,[a|Pila]).
caso(q0,b,Pila,q0,[b|Pila]).
caso(q0,e,Pila,q1,Pila).
caso(q1,a,[b|Pila],q1,Pila).
caso(q1,b,[a|Pila], q1,Pila).






transita(q1,[],[],qf):-!.
transita(Qi,[X|Y],Pi,Q):-caso(Qi,X,Pi,Q2,Pila2),transita(Q2,Y,Pila2,Q).


acepta(X,R):-X\=[], transita(q0,X,[],Qf), Qf = qf, R is 1.


solve_traza_simple(A):-solve_traza(A,1).
solve_traza(true,_):-!.

solve_traza((A,B),Nivel):-
	solve_traza(A, Nivel),
	solve_traza(B, Nivel).


solve_traza(A,Nivel):-
    predicate_property(A,built_in), !,
    Tab is Nivel*3,
    format("~t~*| Llamada: ~w~n",[Tab,A]),call(A).



solve_traza(A,Nivel):-
    clause(A,B),
    Tab is Nivel*3,
    format("~t~*| Llamada: ~w~n",[Tab,A]),
    Nivel2 is Nivel +1,
    solve_traza(B,Nivel2).


solve_pmax(A,Prof):-solve_pmax(A,1,Prof).

solve_pmax(true,_,_):-!.

solve_pmax((A,B),Nivel,Prof):-
    solve_pmax(A,Nivel,Prof),
    solve_pmax(B,Nivel,Prof).

solve_pmax(A,Nivel,Prof):-
    Prof>0,
    predicate_property(A,built_in), !,
    Tab is Nivel*3,
    format("~t~*|LLamada: ~w~n",[Tab,A]),call(A).


solve_pmax(A,Nivel,Prof):-
    Prof>0,clause(A,B),
    Tab is Nivel*3,
    format("~t~*|LLamada: ~w~n",[Tab,A]),
    Nivel2 is Nivel +1,
    Prof2 is Prof -1,
    solve_pmax(B,Nivel2,Prof2).

solve_pmax(_,_,0):-write("Limite de profundidad alcanczado"), nl, !,abort.



solve_pmaxInv(A,Prof):-solve_pmaxInv(A,1,Prof).

solve_pmaxInv(true,_,_):-!.

solve_pmaxInv((A,B),Nivel,Prof):-
    solve_pmaxInv(B,Nivel,Prof),
    solve_pmaxInv(A,Nivel,Prof).

solve_pmaxInv(A,Nivel,Prof):-
    Prof>0,
    predicate_property(A,built_in), !,
    Tab is Nivel*3,
    format("~t~*|LLamada: ~w~n",[Tab,A]),call(A).


solve_pmaxInv(A,Nivel,Prof):-
    Prof>0,clause(A,B),
    Tab is Nivel*3,
    format("~t~*|LLamada: ~w~n",[Tab,A]),
    Nivel2 is Nivel +1,
    Prof2 is Prof -1,
    solve_pmaxInv(B,Nivel2,Prof2).

solve_pmaxInv(_,_,0):-write("Limite de profundidad alcanczado"), nl, !,abort.


