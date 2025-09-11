transporte(roma,200).
transporte(londres,250).
transporte(tunez,150).

hotel(roma,250).
hotel(londres,150).
hotel(tunez,100).

hostal(roma,150).
hostal(londres,100).
hostal(tunez,80).

camping(roma,100).
camping(londres,50).
camping(tunez,50).

precio_hotel(Z,Ciudad,Precio):-
    transporte(Ciudad,X),
    hotel(Ciudad,Y),
    Precio is 2*X+Y*Z.

precio_hostal(Z,Ciudad,Precio):-
    transporte(Ciudad,X),
    hostal(Ciudad,Y),
    Precio is 2*X+Y*Z.


precio_camping(Z,Ciudad,Precio):-
    transporte(Ciudad,X),
    camping(Ciudad,Y),
    Precio is 2*X+Y*Z.



