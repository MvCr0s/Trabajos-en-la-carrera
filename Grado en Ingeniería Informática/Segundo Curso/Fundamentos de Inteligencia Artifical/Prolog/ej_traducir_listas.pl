traducir([], []).
traducir([X|Cola1], [Y|Cola2], Idioma):-
traducir_palabra(X,Y,Idioma),
traducir(Cola1,Cola2,Idioma).
