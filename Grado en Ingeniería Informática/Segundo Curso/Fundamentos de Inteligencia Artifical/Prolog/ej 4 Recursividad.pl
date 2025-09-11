enlace(a,b,3).
enlace(a,c,2).
enlace(c,d,4).
enlace(c,e,5).
enlace(e,f,6).
enlace(c,g,3).
enlace(g,h,7).

ruta(X,Y,D):- enlace(X,Y,D).
ruta(X,Y,D):- enlace(X,Z,L),ruta(Z,Y,M), D is L+M.
