% Hechos
asignatura(alg, matematicas).
asignatura(ia, informatica).
enseña(juan, alg).
enseña(la_mujer_de_juan, ia).

inteligente(Persona) :- enseña(Persona, Asignatura), asignatura(Asignatura, informatica).
