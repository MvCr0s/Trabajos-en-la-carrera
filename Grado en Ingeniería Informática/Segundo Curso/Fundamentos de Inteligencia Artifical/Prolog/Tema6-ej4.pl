% Termina la recursión si la lista es vacía.
borrar(_, [], []).

% Caso en el que el elemento en la cabeza de la lista es igual a S.
% Se devuelve el resto de la lista, sin añadir la cabeza a la lista de resultado.
borrar(S, [S|Cola], L) :-
    borrar(S, Cola, L), !.

% Caso en el que el elemento en la cabeza de la lista es diferente a S.
% Añade el elemento cabeza a la lista de resultado y continua la recursión.
borrar(S, [X|Cola], [X|L]) :-
    S \= X,
    borrar(S, Cola, L).
