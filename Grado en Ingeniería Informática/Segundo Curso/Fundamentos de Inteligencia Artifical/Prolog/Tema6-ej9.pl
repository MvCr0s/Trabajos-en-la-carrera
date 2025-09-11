maximo([],_).
maximo([X],X).

maximo([Y|Cola],Max):-maximo_aux(Cola,Y,Max).

maximo_aux([], Max, Max):-!.
maximo_aux([Y|Cola1],CurrentMax,Max):-CurrentMax<Y,
    NewMax=Y,
    maximo_aux(Cola1,NewMax,Max).

maximo_aux([_|Cola1],CurrentMax,Max):-
    maximo_aux(Cola1,CurrentMax,Max).
