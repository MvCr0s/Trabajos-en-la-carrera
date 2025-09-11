%borrar_repes([],[]).
%borrar_repes([X|Cola],[X|R]):-borrar(X,Cola,R1),borrar_repes(R1,R).

%borrar(_,[],[]).
%borrar(X,[X|Cola],R):-borrar(X,Cola,R).
%borrar(X,[Y|Cola],[Y|R]):-X\=Y,borrar(X,Cola,R).

% Caso base: una lista vacía da como resultado una lista vacía.
borrar_repes([], []).

% Caso recursivo: procesar el primer elemento y eliminar repeticiones del resto.
borrar_repes([X|Cola], [X|R]) :-
    borrar(Cola, X, NuevaCola),
    borrar_repes(NuevaCola, R).

% Predicado auxiliar borrar/3 elimina todas las apariciones de X en la lista de entrada.
borrar([], _, []).

borrar([X|Cola], X, R) :-
    borrar(Cola, X, R).

borrar([Y|Cola], X, [Y|R]) :-
    X \= Y,
    borrar(Cola, X, R),!.


