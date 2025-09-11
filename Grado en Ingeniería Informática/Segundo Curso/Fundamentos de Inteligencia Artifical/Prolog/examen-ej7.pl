fila(Num,_):-Num<1,false.
fila(1,[1]).

fila(Num,[1|L]):-Num>1, N1 is Num-1,fila(N1,L1),sumar2en2(L1,L).


sumar2en2([X],[X]).
sumar2en2([X,Y|Cola],[Z|L]):-Z is X+Y,sumar2en2([Y|Cola],L).
