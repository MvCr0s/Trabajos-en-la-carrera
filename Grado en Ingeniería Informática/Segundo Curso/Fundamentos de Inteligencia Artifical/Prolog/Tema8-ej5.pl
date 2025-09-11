%cifras_lista(0,[]).
% cifras_lista(Num,[Resto|L]):-Resto is Num mod 10,Num1 is Num
% //10,cifras_lista(Num1,L).

% Caso base: la representación de 0 es una lista vacía.
cifras_lista(0, []) :- !.

% Caso recursivo principal que inicia la acumulación.
cifras_lista(Num, Lista) :-
    cifras_lista_aux(Num, [], Lista).

% Caso base del predicado auxiliar: cuando el número es 0, la lista acumulada es la lista resultante.
cifras_lista_aux(0, Acc, Acc) :- !.

% Caso recursivo del predicado auxiliar: agrega el resto de la división al acumulador y continúa con el cociente.
cifras_lista_aux(Num, Acc, Lista) :-
    Num > 0,
    Resto is Num mod 10,
    Num1 is Num // 10,
    cifras_lista_aux(Num1, [Resto | Acc], Lista).

