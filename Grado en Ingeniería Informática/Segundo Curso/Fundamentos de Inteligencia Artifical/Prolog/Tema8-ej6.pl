% Caso base: rotar una lista vacía no cambia la lista.
rotar_derecha([], []).

% Caso base: rotar una lista con un solo elemento no cambia la lista.
rotar_derecha([X], [X]).

% Caso general: obtener el último elemento y la parte inicial de la lista, y luego combinar.
rotar_derecha(Lista, [Ultimo | Resto]) :-
    ultimo(Lista, Ultimo),
    inicial(Lista, Resto).



ultimo([X], X).
ultimo([_|T], X) :-
    ultimo(T, X).



inicial([_], []).
inicial([H|T], [H|R]) :-
    inicial(T, R).
