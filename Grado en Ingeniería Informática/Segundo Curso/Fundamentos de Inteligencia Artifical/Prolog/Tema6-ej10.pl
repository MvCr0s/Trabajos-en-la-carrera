
consecut([]).
consecut([X|[Y|_]],X,Y).
consecut([_|[Y|Cola]],A,B):-consecut([Y|Cola],A,B).
