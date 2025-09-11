potenciasDeMulti5(I, S, Exp, Num, R) :-
    between(I, S, Num),             % Num está en el intervalo [I, S]
    0 is Num mod 5,                % Num es múltiplo de 5
    potencia(Num, Exp, R).         % Calcula la potencia de Num con Exp y guarda el resultado en R

potencia(_, 0, 1).
potencia(X, Y, R) :-
    Y1 is Y - 1,
    Y > 0,
    potencia(X, Y1, R1),
    R is R1 * X.

