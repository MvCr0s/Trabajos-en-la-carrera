
mueve(q0, a, [], q1, [a]).

mueve(q1, a, H, q1, [a|H]).
mueve(q1, b, [a|H], q2, H).

mueve(q2, b, [a|H], q2, H).
mueve(q2, b, [a], q2, []).
mueve(q2, c, [], q2, []).


mueve(q3, a, [], q4,[a]).

mueve(q4, a, H, q4, [a|H]).
mueve(q4, b, H, q4, H).
mueve(q4, c, [a|H], q5, H).


mueve(q5, c, [a|H], q5, H).
mueve(q5, c, [a], qf, []).


transita(q5,[],[],qf):-!.
transita(q2,[],[],qf):-!.
transita(Qi,[X|Y],S, Qf):-mueve(Qi, X, S, Qt, H), transita(Qt,Y,H,Qf).


acepta(C,R):- transita(q0,C,_,Q1),transita(q3,C,_,Q2),Q1=qf,Q2=qf, R is 1.


solve_traza(true,_,_):-!.
solve_traza((A,B)):-!,solve_traza(A), solve_traza(B).
solve_traza(A) :- predicate_property(A,built_in), !, call(A).
solve_traza(A):- write("Call: "),write(A), nl, clause(A,B),solve_traza(B),
    write("Exit: "), write(B), nl.

solve_pmax(A,Prof):-solve_pmax(A,1,Prof).
solve_pmax(true,_,_):-!.
solve_pmax((A,B),Nivel,Prof):-solve_pmax(A,Nivel,Prof),solve_pmax(B,Nivel,Prof).

solve_pmax(A,Nivel,Prof):-Prof>0, predicate_property(A,built_in), !,
    Tab is Nivel*3, format("~t~*|Call: ~w~n",[Tab,A]), call(A).

solve_pmax(A,Nivel,Prof):-Prof>0,clause(A,B),Tab is Nivel*3,
    format("~t~*|Call: ~w~n",[Tab,A]), Nivel2 is Nivel+1, Prof2 is Prof-1,
    solve_pmax(B,Nivel2,Prof2).

solve_pmax(_,_,0):-write("Limite Alcanzado"), nl,!,abort.



solve_pmaxInv(A,Prof):-solve_pmaxInv(A,1,Prof).
solve_pmaxInv(true,_,_):-!.
solve_pmaxInv((A,B),Nivel,Prof):-solve_pmaxInv(B,Nivel,Prof),
    solve_pmaxInv(A,Nivel,Prof).

solve_pmaxInv(A,Nivel,Prof):-Prof>0, predicate_property(A,built_in), !,
    Tab is Nivel*3, format("~t~*|Call: ~w~n", [Tab,A]),call(A).

solve_pmaxInv(A,Nivel,Prof):- Prof>0,clause(A,B), Tab is Nivel*3,
     format("~t~*|Call: ~w~n", [Tab,A]),Nivel2 is Nivel+1,
     Prof2 is Prof-1, solve_pmaxInv(B,Nivel2,Prof2).
solve_pmaxInv(_,_,0):-write("Limita alcanzado"), nl, !, abort.


