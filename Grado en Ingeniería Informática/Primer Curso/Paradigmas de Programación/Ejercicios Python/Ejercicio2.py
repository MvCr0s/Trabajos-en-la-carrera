#Ejercicio2
piedras=50
while piedras != 0:
    print("Jugador1 coja entre 1 y 5 piedras")
    jugador1=int(input())
    while 1<jugador1>5:
        print("No puedes coger ese numero de piedras")
        jugador1=int(input())
    piedras=piedras-jugador1
    print("Quedan",piedras)
    if piedras==0:
        print("El jugador1 ha ganado")
        break
    print("Jugador2 coja entre 1 y 5 piedras")
    jugador2=int(input())
    while 1<jugador2>5:
        print("No puedes coger ese numero de piedras")
        jugador2=int(input())
    piedras=piedras-jugador2
    print("Quedan",piedras)
    if piedras==0:
        print("El jugador2 ha ganado")
        break
print("Se acabo el juego")