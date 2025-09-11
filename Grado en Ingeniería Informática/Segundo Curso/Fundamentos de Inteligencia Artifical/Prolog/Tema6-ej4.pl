% Termina la recursi�n si la lista es vac�a.
borrar(_, [], []).

% Caso en el que el elemento en la cabeza de la lista es igual a S.
% Se devuelve el resto de la lista, sin a�adir la cabeza a la lista de resultado.
borrar(S, [S|Cola], L) :-
    borrar(S, Cola, L), !.

% Caso en el que el elemento en la cabeza de la lista es diferente a S.
% A�ade el elemento cabeza a la lista de resultado y continua la recursi�n.
borrar(S, [X|Cola], [X|L]) :-
    S \= X,
    borrar(S, Cola, L).
