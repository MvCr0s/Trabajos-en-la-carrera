lista([],X):-X is 0.
lista([_|Cola],X):-lista(Cola,X1),X is X1+1.
