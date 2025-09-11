le_gusta_a(juan, maria).
le_gusta_a(pedro, coche).
le_gusta_a(maria, libro).
le_gusta_a(maria, juan).
le_gusta_a(jose, maria).
le_gusta_a(jose, coche).
le_gusta_a(jose, pescado).
es_amigo_de(juan, Y):- le_gusta_a(Y, coche).
vertical(seg(punto(X,Y), punto(X,Y1))).
horizontal(seg(punto(X,Y),punto(X1,Y))).
