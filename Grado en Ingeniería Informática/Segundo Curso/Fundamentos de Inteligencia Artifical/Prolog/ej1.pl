% Hechos
asignatura(alg, matematicas).
asignatura(ia, informatica).
ense�a(juan, alg).
ense�a(la_mujer_de_juan, ia).

inteligente(Persona) :- ense�a(Persona, Asignatura), asignatura(Asignatura, informatica).
