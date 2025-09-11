derecha_de(perro, bombilla).
derecha_de(lupa, perro).
derecha_de(casa, coche).
derecha_de(taza, casa).
derecha_de(balanza, taza).
derecha_de(llave, tortuga).
derecha_de(semaforo,llave).
derecha_de(martillo,semaforo).


encima_de(bombilla,coche).
encima_de(coche,tortuga).
encima_de(perro,casa).
encima_de(casa,llave).
encima_de(lupa,taza).
encima_de(taza,semaforo).
encima_de(semaforo,cuchara).
encima_de(cuchara,tenedor).
encima_de(balanza,martillo).


izquierda_de(X,Y):- derecha_de(X,Y).
debajo_de(X,Y):- encima_de(Y,X).

derecha(X,Y):- derecha_de(X,Y).
derecha(X,Y):- derecha_de(Z,Y),derecha(X,Z).

encima(X,Y):- encima_de(X,Y).
encima(X,Y):- encima_de(Z,Y), encima(X,Z).
