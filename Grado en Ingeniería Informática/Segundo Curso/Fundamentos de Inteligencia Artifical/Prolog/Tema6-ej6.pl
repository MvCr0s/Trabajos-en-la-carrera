
creciente([_]).
creciente([X|[Y|Z]]):-X<Y,creciente([Y|Z]).
