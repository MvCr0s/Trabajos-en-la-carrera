

live(outside).
light(l1).
light(l2).
down(s1).
up(s2).
up(s3).
ok(_).

% Diferimos la evaluación de las conexiones basadas en el estado de los interruptores
delay(up(_)).
delay(down(_)).

% Definimos el predicado `live/1` diferido
live(W):- connected_to(W,W1), dsolve(live(W1)).

% Definimos las luces
lit(l1):- light(l1), ok(l1), dsolve(live(w0)).
lit(l2):- light(l2), ok(l2), dsolve(live(w4)).

% Conexiones
connected_to(w5, outside).
connected_to(cb1, outside).
connected_to(w3, cb1).
connected_to(p1, w3).
connected_to(w4, cb1):- dsolve(up(s3)).
connected_to(w2, cb2):- dsolve(down(s1)).
connected_to(w1, cb1):- dsolve(up(s1)).
connected_to(w0, w1):- dsolve(up(s2)).
connected_to(w0, w2):- dsolve(down(s2)).
connected_to(cb2, w5).
connected_to(w6, cb2).
connected_to(p2, w6).







% Declarar metas diferidas
delay(ok(_)).

% Base case: cuando se encuentra 'true', la lista de resultados es la misma
dsolve(true, D, D) :- !.

% Si A y B son conjunciones, resolvemos ambos
dsolve((A & B), D1, D2) :- !,
    dsolve(A, D1, D3),
    dsolve(B, D3, D2).

% Si A debe ser diferida, la agregamos a la lista
dsolve(A, D, [A|D]) :-
    delay(A), !.  % Si A está marcado como delay, se agrega a la lista de resultados

% Caso general para resolver A
dsolve(A, D1, D2) :-
    call(A),  % Llamamos a A
    dsolve(true, D1, D2).  % Continuamos la resolución
