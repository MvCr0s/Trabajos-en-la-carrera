familia(persona(juan,perez,50),
persona(maria,alonso,45),
[persona(carlos,perez,20),
persona(andres,perez,18),
persona(elena,perez,12)]).

familia(persona(pedro,lopez,40),
persona(carmen,ruiz,39),
[persona(carlos,lopez,19),
persona(teresa,lopez,8)]).

familia(persona(carlos,martinez,25),
persona(lola,garcia,22),
[]).

edad(persona(_,_,E),E).

%Hijo menor
ultimo(X,Y,Z):-familia(X,Y,Z1),ultimo(Z,Z1).

%Nº de hijos par
