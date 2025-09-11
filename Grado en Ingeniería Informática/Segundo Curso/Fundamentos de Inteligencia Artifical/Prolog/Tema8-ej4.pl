todos_iguales([]).
todos_iguales([X,X|[]]).
todos_iguales([X,X|Cola]):-todos_iguales([X|Cola]).
