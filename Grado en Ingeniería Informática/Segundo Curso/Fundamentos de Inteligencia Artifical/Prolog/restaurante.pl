%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Programa restaurante  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% menu

entrada(paella).
entrada(gazpacho).
entrada(consome).

carne(filete_de_cerdo).
carne(pollo_asado).

pescado(trucha).
pescado(bacalao).

postre(flan).
postre(nueces_con_miel).
postre(naranja).

bebida(vino).
bebida(cerveza).
bebida(agua_mineral).

% Valor calorico de una raci�n

calorias(paella, 200).
calorias(gazpacho, 150).
calorias(consome, 300).
calorias(filete_de_cerdo, 400).
calorias(pollo_asado, 280).
calorias(trucha, 160).
calorias(bacalao, 300).
calorias(flan, 200).
calorias(nueces_con_miel, 500).
calorias(naranja, 50).
calorias(vino,50).
calorias(agua_mineral,0).
calorias(cerveza,30).

% plato_principal(P) P es un plato principal si es carne o pescado

plato_principal(P):- carne(P).
plato_principal(P):- pescado(P).

% comida(Entrada, Principal, Postre, Bebida)

comida(Entrada, Principal, Postre, Bebida):-
        entrada(Entrada),
        plato_principal(Principal),
        postre(Postre).
        bebida(Bebida).

% Valor calorico de una comida

valor(Entrada, Principal, Postre,Bebida ,Valor):-
        calorias(Entrada, X),
        calorias(Principal, Y),
        calorias(Postre, Z),
        bebida(Bebida,W).
        sumar(X, Y, Z, W,Valor).

% comida_equilibrada(Entrada, Principal, Postre)

comida_equilibrada(Entrada, Principal, Postre, Bebida):-
        comida(Entrada, Principal, Postre, Bebida),
        valor(Entrada, Principal, Postre,Bebida, Valor),
        menor(Valor, 600).


% Conceptos auxiliares

sumar(X, Y, Z,W ,Res):-
        Res is X + Y + Z + W.             % El predicado "is" se satisface si Res se puede unificar
                                      % con el resultado de evaluar la expresi�n X + Y + Z
menor(X, Y):-
        X < Y.                        % "menor" numerico

dif(X, Y):-
        X =\= Y.                      % desigualdad numerica



