sumar2en2([],[]).
sumar2en2([X],[X]).
sumar2en2([X,Y|Cola],[Z|L]):-Z is X+Y,sumar2en2(Cola,L).
