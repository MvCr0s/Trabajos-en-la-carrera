
mueve(q0,a,[],q0,[a]).
mueve(q0,a,[a|Z],q0,[a,a|Z]).
mueve(q0,a,[b|Z],q0,Z).
mueve(q0,a,[c|Z],q0,Z).


mueve(q0,b,[],q0,[b]).
mueve(q0,b,[a|Z],q0,Z).
mueve(q0,b,[b|Z],q0,[b,b|Z]).
mueve(q0,b,[c|Z],q0,[b,c|Z]).


mueve(q0,c,[],q0,[b]).
mueve(q0,c,[a|Z],q0,Z).
mueve(q0,c,[b|Z],q0,[c,b|Z]).
mueve(q0,c,[c|Z],q0,[c,c|Z]).




transita(_,[],[],qf):-!.
transita(Qi,[X|Y],P, Qf ):- mueve(Qi,X,P,Q,Pila2), transita(Q,Y,Pila2,Qf).


acepta(X,Resultado):-transita(q0,X,[],Q), Q=qf, Resultado is 1,!.



solve_traza(A):-solve_traza(A,1).

solve_traza(true,_):-!.

solve_traza(A,Nivel):-
      Tab is Nivel * 3,
      format("~t~*|llamada: ~w~n",[Tab,A]), call(A).

solve_traza(A,Nivel):-
    clause(A,B),
    Tab is Nivel *3,
    format("~t~*|llamada: ~w~n", [Tab,A]),
    nuevoNivel is Nivel +1,
    solve_traza(B,nuevoNivel).


solve_pmax(A,Proof):-solve_pmax(A,1,Proof).
solve_pmax(true,_,_).

solve_pmax((A,B),Nivel,Proof):-
   solve_pmax(A,Nivel,Proof),
   solve_pmax(B,Nivel,Proof).

solve_pmax(A,Nivel,_):-
      Tab is Nivel * 3,
      format("~t~*|llamada: ~w~n",[Tab,A]), call(A).

solve_pmax(A,Nivel,Prof):-
    Prof>0,
    clause(A,B),
    Tab is Nivel *3,
    format("~t~*|llamada: ~w~ñ", [Tab,A]),
    nuevoNivel is Nivel +1,
    ProfNuevo is Prof-1,
    solve_traza(B,nuevoNivel,ProfNuevo).

solve_pmax(_,_,0):-write('Limite de profundidad alcanzado'),nl,abort.
