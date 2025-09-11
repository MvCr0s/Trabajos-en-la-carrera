suma([],X):-X is 0.
suma([Elem1|Cola],X):-suma(Cola,X1), X is X1 + Elem1.
