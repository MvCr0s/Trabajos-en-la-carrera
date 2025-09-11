

% Leer una cadena a^x, b^y, c^z tal que y= x+z
%
% Estados posibles
%
% e0: Lee A hasta encontrar una B
% e1: Lee B hasta encontrar una C, por cada B borra una A de la pila
% e2: Lee C, por cada C borra una B, hasta tener la pila vacia

caso(e0,	a,	[],	e0,	[a]).
caso(e0,	a,	H,	e0,	[a|H]).
caso(e0,	b,	[a|H],	e1,	H).
caso(e1,	b,	[a|H],	e1,	H).
caso(e1,	b,	[],	e1,	[b]).
caso(e1,	b,	H,	e1,	[b|H]).
caso(e1,	c,	[b|H],	e2,	H).
caso(e2,	c,	[b|H],	e2,	H).

mueve(e2, [], [], eF).
mueve(Ei, [Li|L], Pi, Ef):-
	caso(Ei, Li, Pi, En, Pn),
	mueve(En, L, Pn, Ef).

comprueba(L, Res):-
	mueve(e0, L, [], Ef),
	Ef = eF,
	Res is 1,
	!.

solve(A):-
    predicate_property(A,built_in),
    !,
    call(A).
solve(true):-!.
solve((A,B)):-
        !,
        solve(A),
        solve(B).
solve(A):-
        clause(A,B),
        solve(B).


