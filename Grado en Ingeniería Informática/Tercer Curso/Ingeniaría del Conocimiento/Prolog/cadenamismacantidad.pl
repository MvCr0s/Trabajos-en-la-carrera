
% Leer una cadena a^x, b^y, c^z tal que x=y=z
%
% Estados posibles

caso(e0,	a,	[],	e0,	[a]).
caso(e0,	a,	H,	e0,	[a|H]).
caso(e0,	b,	[a|H],	e1,	H).
caso(e1,	b,	[a|H],	e1,	H).
caso(e1,	b,	[],	e1,	[b]).
caso(e1,	b,	H,	e1,	[b|H]).
caso(e1,	c,	H,	e2,	H).
caso(e2,	c,	H,	e2,	H).

caso(e3,	a,	H,	e3,	H).
caso(e3,	b,	[],	e4,	[b]).
caso(e4,	b,	H,	e4,	[b|H]).
caso(e4,	c,	[b|H],	e5,	H).
caso(e5,	c,	[b|H],	e5,	H).
caso(e5,	c,	H,	e6,	H).

mueve(e2, [], [], eF).
mueve(e5, [], [], eF).

mueve(_, [], _, _):- 
	!,
	abort.
 
mueve(Ei, [Li|L], Pi, Ef):-
	caso(Ei, Li, Pi, En, Pn),
	mueve(En, L, Pn, Ef).

comprueba(L, Res):-
	mueve(e0, L, [], Ef),
	Ef = eF,
	mueve(e3, L, [], Eff),
	Eff = eF,
	Res is 1,
	!.

solve(A) :- predicate_property(A,built_in), !, call(A).
solve(true).
solve((A,B)):- solve(A), solve(B).
solve(A):- clause(A,B), solve(B).


solve_traza(A):-
	solve_traza(A,1).

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

solve_trazaInv(A):-
	solve_trazaInv(A,1).

solve_trazaInv(true, _).

solve_trazaInv((A, B), Nivel) :-
    solve_trazaInv(B, Nivel),
    solve_trazaInv(A, Nivel).

solve_trazaInv(A, Nivel) :-
    predicate_property(A, built_in),
    !,
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    call(A).

solve_trazaInv(A, Nivel) :-
    clause(A, B),
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    NivelNuevo is Nivel + 1,
    solve_trazaInv(B, NivelNuevo).


solve_trazaProf(A, Prof):-
	solve_trazaProf(A, 1, Prof).

solve_trazaProf(true, _, _).

solve_trazaProf((A, B), Nivel, Prof) :-
    solve_trazaProf(A, Nivel, Prof),
    solve_trazaProf(B, Nivel, Prof).

solve_trazaProf(A, Nivel, _) :-
    predicate_property(A, built_in),
    !,
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    call(A).

solve_trazaProf(A, Nivel, Prof) :-
    Prof > 0,
    clause(A, B),
    Tab is Nivel * 3,
    format("~t~*|Llamada: ~w~n", [Tab, A]),
    NivelNuevo is Nivel + 1,
    ProfNuevo is Prof - 1,
    solve_trazaProf(B, NivelNuevo, ProfNuevo).

solve_trazaProf(_, _, 0):-
    write('Limite de profundidad alcanzado. Abortando.'), nl,
    abort.
