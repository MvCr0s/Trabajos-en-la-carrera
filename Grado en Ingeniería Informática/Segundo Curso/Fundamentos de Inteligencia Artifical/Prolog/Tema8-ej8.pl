repetir(_,0,[]).
repetir(Repetir,Num,R):-repetir2(Repetir,Num,R).

repetir2(_,Num,[]):-Num =:= 0,!.
repetir2(Repetir,Num,[Repetir|R]):-Num1 is Num-1,repetir(Repetir,Num1,R).
