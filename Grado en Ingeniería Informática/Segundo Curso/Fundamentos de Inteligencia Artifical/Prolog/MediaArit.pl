% Definici�n de la regla para calcular la media de dos n�meros
media(X, Y, Res) :-
    suma(X, Y, Sum),
    Res is Sum / 2.

% Definici�n de la regla para calcular la suma de dos n�meros
suma(X, Y, Sum) :-
    Sum is X + Y.
