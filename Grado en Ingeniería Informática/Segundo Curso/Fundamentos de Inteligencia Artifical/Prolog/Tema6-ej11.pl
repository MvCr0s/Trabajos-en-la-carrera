invertir([],_).
invertir([X|Cola],L):-invertir([X|Cola],[],L).

invertir([],L1,L1).
invertir([X|Cola],L1,L):-invertir(Cola,[X|L1],L).
