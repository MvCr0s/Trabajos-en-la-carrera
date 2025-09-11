borrartodos(_,[],[]).
borrartodos(X,[X|Cola],L):-borrartodos(X,Cola,L).
borrartodos(X,[Y|Cola],[Y|L]):-X\=Y,borrartodos(X,Cola,L).
