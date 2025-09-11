% Definición del predicado mcd
mcd(X, 0, X) :- X > 0, !.
mcd(0, Y, Y) :- Y > 0, !.
mcd(X, Y, MCD) :- X >= Y, Z is X - Y, mcd(Y, Z, MCD).
mcd(X, Y, MCD) :- X < Y, Z is Y - X, mcd(X, Z, MCD).



