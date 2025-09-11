padre(pedro,juan).
padre(pedro,margarita).
padre(pedro,paula).
padre(juan,isabel).
padre(juan,fernando).
padre(jose,felipe).
padre(manuel,javier).
padre(manuel,eva).
padre(manuel,alicia).

madre(maria,juan).
madre(maria,margarita).
madre(maria,paula).
madre(virginia,isabel).
madre(virginia,fernando).
madre(margarita,felipe).
madre(paula,javier).
madre(paula,eva).
madre(paula,alicia).

hijo(X,Y):- padre(X,Y); madre(X,Y).
abuelo(X,Y):- hijo(X,Z),hijo(Z,Y).
abuela(X, Y):- hijo(X,Z),hijo(Z,Y).

