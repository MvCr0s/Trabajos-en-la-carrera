% Definición de la regla para calcular la media de dos números
media(X, Y, Res) :-
    suma(X, Y, Sum),
    Res is Sum / 2.

% Definición de la regla para calcular la suma de dos números
suma(X, Y, Sum) :-
    Sum is X + Y.
