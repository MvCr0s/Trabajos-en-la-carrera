
mueve(q0,a,[],q0,[a]).
mueve(q0,a,[a|Z],q0,[a,a|Z]).
mueve(q0,a,[b|Z],q0,Z).
mueve(q0,a,[c|Z],q0,[a,c|Z]).


mueve(q0,b,[],q0,[b]).
mueve(q0,b,[a|Z],q0,Z).
mueve(q0,b,[b|Z],q0,[b,b|Z]).
mueve(q0,b,[c|Z],q0,Z).


mueve(q0,c,[],q0,[c]).
mueve(q0,c,[a|Z],q0,[c,a|Z]).
mueve(q0,c,[b|Z],q0,Z).
mueve(q0,c,[c|Z],q0,[c,c|Z]).


transita(_,[],[],qf):-!.
transita(Qi,[X|Y],T,Q):-mueve(Qi,X,T,Qf,S), transita(Qf,Y,S,Q).


acepta(Cadena, R):- transita(q0,Cadena,[],Qf), Qf=qf, R is 1, !.



%solve_traza(true):-!.

%solve_traza((A,B)):-
%    solve_traza(A), solve_traza(B).
%solve_traza(A):-
 %   predicate_property(A,built_in), !,
  %  format("Llamada: ~w~n", [A]), call(A).

%solve_traza(A):-
%    clause(A,B), format("Llamada: ~w~n", [A]), solve_traza(B).

solve_traza2(true):-!.
solve_traza2((A, B)) :-!, solve_traza2(A), solve_traza2(B).
solve_traza2(A):-predicate_property(A,built_in), !, call(A).
solve_traza2(A):- write('Call: '), write(A), nl,
 clause(A,B), solve_traza2(B),
 write('Exit: '), write(B), nl.



solve_pmax(A,Prof):- solve_pmax(A,1,Prof).

solve_pmax(true,_,_):-!.

solve_pmax((A,B), Nivel, Prof):-
    solve_pmax(A,Nivel,Prof), solve_pmax(B,Nivel,Prof).

solve_pmax(A,Nivel,Prof):-
    Prof>0,
    predicate_property(A,built_in), !,
    Tab is Nivel*3, format("~t~*|Llamada : ~w~n", [Tab,A]),
    call(A).
solve_pmax(A,Nivel,Prof):-
    Prof>0, clause(A,B),
    Tab is Nivel*3, format("~t~*|Llamada : ~w~n",[Tab,A]),
    NuevoNivel is Nivel+1, NuevaProf is Prof-1,
    solve_pmax(B,NuevoNivel, NuevaProf).

solve_pmax(_,_,0):- write("Limite alcanzado"), nl, !, abort.



solve_pmaxInv(A,Prof):- solve_pmaxInv(A,1,Prof).
solve_pmaxInv(true,_,_):-!.
solve_pmaxInv((A,B),Nivel,Prof):-
    solve_pmaxInv(B,Nivel,Prof), solve_pmaxInv(A,Nivel,Prof).

solve_pmaxInv(A,Nivel,Prof):-
    Prof>0,
    predicate_property(A, built_in),!,
    Tab is Nivel*3, format("~t~*|LLamada : ~w~n", [Tab, A]),
    call(A).
solve_pmaxInv(A,Nivel,Prof):-
    Prof>0, clause(A,B), Tab is Nivel*3,
    format("~t~*|LLamada : ~w~n", [Tab,A]),
    Nivel2 is Nivel+1, Prof2 is Prof-1,
    solve_pmaxInv(B, Nivel2, Prof2).

solve_pmaxInv(_,_,0):-write("Limite alcanzado"), nl, !,abort.

