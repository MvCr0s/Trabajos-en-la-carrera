suma([],X):-X is 0.
suma([Y|Cola],X):-suma(Cola,X1),X is X1+Y.
