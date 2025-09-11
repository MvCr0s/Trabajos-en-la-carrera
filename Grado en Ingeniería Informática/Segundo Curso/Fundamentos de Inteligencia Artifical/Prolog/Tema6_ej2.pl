longitud([],X):-X is 0.
longitud([_|Cola],X):-is_list(Cola),longitud(Cola,X1),X is X1+1.


