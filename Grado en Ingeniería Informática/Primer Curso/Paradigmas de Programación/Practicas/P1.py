import random

#Funciones

def crearFichero():     #Cada vez que se ejecute se obtendrá un nuevo fichero
    movimientos=open('Listado.txt','w')
    movimientos.write('')
    movimientos.close()
    i=0
    while i<36:
        movimientos=open('Listado.txt','a')
        valor=random.randint(1, 72)
        v=''
        if(valor<46):
            v='.'
        if(45<valor and valor<64):
            v='a'
        if(63<valor and valor<68):
            v='b'
        if(67<valor and valor<71):
            v='c'
        if(70<valor and valor<73):    
            v='1'
        if((i)%6==0 and i!=0):
            movimientos.write("\n")
        i+=1
        movimientos.write(v)
    movimientos.close()

def printTablero(a,actual,almacenado,turno,puntuacion):    #Mostramos el tablero y todo lo relacionado con el(turnos,actual...)
    i=0
    j=0
    #printeamos el tablero
    print("┌───┬───┬───┬───┬───┬───┐")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1 
    print("├───┼───┼───┼───┼───┼───┤")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1
    print("├───┼───┼───┼───┼───┼───┤")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1 
    print("├───┼───┼───┼───┼───┼───┤")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1 
    print("├───┼───┼───┼───┼───┼───┤")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1
    print("├───┼───┼───┼───┼───┼───┤")
    print("│ "+a[i][j]+" │ "+a[i][j+1]+" │ "+a[i][j+2]+" │ "+a[i][j+3]+" │ "+a[i][j+4]+" │ "+a[i][j+5]+" │")
    i=i+1 
    print("└───┴───┴───┴───┴───┴───┘")

    print('Turno: '+str(turno)+'             Puntuacion: '+str(puntuacion))
    print('Actual: '+actual+'            Almacenado: '+almacenado)

def ValidPuntos(a): #Mira que todavia halla mas casillas en blanco en el tabler0
    i=0
    j=0
    while i<6:
        while j<6:
            if a[i][j]==".": return False
            j=j+1             
        i=i+1
        j=0
    return True          

def Actual(): #Saca el siguiente objeto
        valor=random.randint(1, 43)
        v='0'
        if(valor<31):
            v='a'
        if(30<valor and valor<36):
            v='b'
        if(35<valor and valor<37):
            v='c'
        if(36<valor and valor<43):
            v='1'
        if(42<valor and valor<44):
            v='w'
        return v

def Almacen(almacenado, actual,salto):  #funcion para almacenar/cambiar valores
    if almacenado != '.':
        actual = almacenado
        salto=False
    else:
        if almacenado == '.':
            almacenado = actual
            salto=True
            print('se ha actualizado el almacen: ' + almacenado)
    return almacenado, actual, salto

def Entrada(almacenado,actual): #Funcion para recoger una entrada del usuario y manejarla
    salto=False
    entradaX=0
    entradaY=0
    entrada =(input("Ingrese la fila y la columna como un solo número de dos dígitos o un * en caso de que quiera guardarlo: "))    #obtenemos la posiciion seleccionada por el jugador

    if entrada=="*": #El usuario desea utilizar el almacen
        almacenado, actual, salto=Almacen(almacenado,actual,salto)   
    else:
        entrada=int(entrada)
        entradaX = entrada // 10 -1# Obtiene el primer dígito
        entradaY = entrada % 10 -1 # Obtiene el segundo dígito
        while(ValidEntrada(entradaX,entradaY,a)==False):
            entrada =input("introce un valor valido: ")    #obtenemos la posiciion seleccionada por el jugador
            if entrada=="*": #El usuario desea utilizar el almacen
                almacenado, actual, salto=Almacen(almacenado,actual,salto) 
            else:
                entrada=int(entrada)
                entradaX = entrada // 10 -1  # Obtiene el primer dígito
                entradaY = entrada % 10 -1  # Obtiene el segundo dígito
    return entrada,entradaX,entradaY,almacenado,actual,salto

def Puntuacion(a): #Mira que todavia halla mas casillas en blanco en el tablero
    valor=''
    puntaje=0
    puntuacion=0
    i=0
    j=0
    while i<6:
        while j<6:
            valor=a[i][j]
            puntaje=SacarPuntaje(valor)
            puntuacion+=puntaje
            j=j+1             
        i=i+1
        j=0
    return puntuacion

#Validaciones varias.. 

def ValidCasilla(entradaX,entradaY,a): #Mira si la entrada que ha seleccionado el jugador esta vacia
    if a[entradaX][entradaY]==".":
        return True
    else:
        return False

def ValidEntrada(entradaX,entradaY,a):  #Mira si la entrada del usuario no se excede del rango de la matriz
    if -1<entradaX<6:
        if -1<entradaY<6:
            return True
        else:
            return False
    else:
        return False
    
def Movicion(a,entradaX,entradaY,actual,cambio): #almacen es el valor (a,b,c,.) recursion para colapsar
    if entradaX-1 >= 0 and a[entradaX-1][entradaY]==actual: #(mirar si esta en el borde)
        a[entradaX-1][entradaY]="."
        cambio=True
        Movicion(a,entradaX-1,entradaY,actual,cambio)
    if entradaY-1 >= 0 and a[entradaX][entradaY-1]==actual: #...
        a[entradaX][entradaY-1]="."
        cambio=True
        Movicion(a,entradaX,entradaY-1,actual,cambio)
    if entradaX+1 <= 5 and a[entradaX+1][entradaY]==actual: #...
        a[entradaX+1][entradaY]="."
        cambio=True
        Movicion(a,entradaX+1,entradaY,actual,cambio)
    if entradaY+1 <= 5 and a[entradaX][entradaY+1]==actual:#...
        a[entradaX][entradaY+1]="."
        cambio=True
        Movicion(a,entradaX,entradaY+1,actual,cambio)
    return cambio

def copiaMatriz(a): #copia el tablero para contar los elementos que pueden colapsar
    filas = len(a)
    columnas = len(a[0])
    b = [[0 for j in range(columnas)] for i in range(filas)]
    for i in range(filas):
        for j in range(columnas):
            b[i][j] = a[i][j]
    return b

def cuentaElem(b, entradaX, entradaY, actual, cuentaelem): #Cuenta los elementos que hay para colapsar en otra matriz igual a "a"

    if entradaX-1 >= 0 and b[entradaX-1][entradaY] == actual:
        cuentaelem += 1
        b[entradaX-1][entradaY] = "."
        cuentaelem = cuentaElem(b, entradaX-1, entradaY, actual, cuentaelem)
    if entradaY-1 >= 0 and b[entradaX][entradaY-1] == actual:
        cuentaelem += 1
        b[entradaX][entradaY-1] = "."
        cuentaelem = cuentaElem(b, entradaX, entradaY-1, actual, cuentaelem)
    if entradaX+1 <= 5 and b[entradaX+1][entradaY] == actual:
        cuentaelem += 1
        b[entradaX+1][entradaY] = "."
        cuentaelem = cuentaElem(b, entradaX+1, entradaY, actual, cuentaelem)
    if entradaY+1 <= 5 and b[entradaX][entradaY+1] == actual:
        cuentaelem += 1
        b[entradaX][entradaY+1] = "."
        cuentaelem = cuentaElem(b, entradaX, entradaY+1, actual, cuentaelem)
    return cuentaelem  

def movBigfoot(a, i, j, edades, moved): 
    if i-1 >= 0 and a[i-1][j] == '.' and not moved:  # Arriba
        a[i-1][j] = '1'
        a[i][j] = '.'
        edades[i-1][j] = edades[i][j]
        if edades[i][j] > 10:
            a[i][j] = 'X'
            edades[i][j] = 0
        else:
            edades[i][j] = 0
    elif j+1 <= 5 and a[i][j+1] == '.' and not moved:  # Derecha
        a[i][j+1] = '1'
        a[i][j] = '.'
        edades[i][j+1] = edades[i][j]
        moved=True
        if edades[i][j] > 10:
            a[i][j] = 'X'
            edades[i][j] = 0
        else:
            edades[i][j] = 0
    elif i+1 <= 5 and a[i+1][j] == '.' and not moved:  # Abajo
        a[i+1][j] = '1'
        a[i][j] = '.'
        edades[i+1][j] = edades[i][j]
        moved=True
        if edades[i][j] > 10:
            a[i][j] = 'X'
            edades[i][j] = 0
        else:
            edades[i][j] = 0
    elif j-1 >= 0 and a[i][j-1] == '.' and not moved:  # Izquierda
        a[i][j-1] = '1'
        a[i][j] = '.'
        edades[i][j-1] = edades[i][j]
        if edades[i][j] > 10:
            a[i][j] = 'X'
            edades[i][j] = 0
        else:
            edades[i][j] = 0
    else:
        a[i][j] = '2'

    return a, moved

def SacarPuntaje(valor):    #Obtenemos la puntuacion en base a la variable seleccionada
    puntaje=0
    if valor=='.':
        puntaje=0
    if valor=='a':
        puntaje=1
    if valor=='b':
        puntaje=5
    if valor=='c':
        puntaje=25
    if valor=='d':
        puntaje=125
    if valor=='e':
        puntaje=625
    if valor=='1':
        puntaje=-25
    if valor=='2':
        puntaje=-5
    if valor=='3':
        puntaje=50
    if valor=='4':
        puntaje=500
    if valor=='X':
        puntaje=-50
    return puntaje

def creaMatrizBigfoot(a): #hacemos una matriz para ir contando edades
    filas = len(a)
    columnas = len(a[0])
    c = [[0 for j in range(columnas)] for i in range(filas)]
    for i in range(filas):
        for j in range(columnas):
            c[i][j]== 0
    return c

def Simplificar2(a):    #Funcion utilizada para colapsar doses (puede que tenga un poco de delay, si se crean muchos doses en un mismo turno es posible que falle)
    def Buscar2(i, j, doses):
        doses.add((i, j))   #Conjunto en el que se encuentran todos los posibles doses juntos, para no repetir...
        if i > 0 and a[i-1][j] == '2' and (i-1, j) not in doses:
            Buscar2(i-1, j, doses)
        if j > 0 and a[i][j-1] == '2' and (i, j-1) not in doses:
            Buscar2(i, j-1, doses)
        if i < len(a)-1 and a[i+1][j] == '2' and (i+1, j) not in doses:
            Buscar2(i+1, j, doses)
        if j < len(a[i])-1 and a[i][j+1] == '2' and (i, j+1) not in doses:
            Buscar2(i, j+1, doses)
            
        return doses
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] == '2':
                doses = Buscar2(i, j, set())    #Utilizamos set() para iniciar el conjunto de doses (vacio)
                
                if len(doses) >= 3:
                    x, y = random.choice(list(doses))
                    a[x][y] = '3'
                    for p, q in doses:
                        if (p, q) != (x, y):
                            a[p][q] = '.'
                
    return a

def Simplificar3(a):    #Funcion utilizada para colapsar treses, funciona de forma similar a la anterior
    def Buscar3(i, j, treses):
        treses.add((i, j))
        
        if i > 0 and a[i-1][j] == '3' and (i-1, j) not in treses:
            Buscar3(i-1, j, treses)
        if j > 0 and a[i][j-1] == '3' and (i, j-1) not in treses:
            Buscar3(i, j-1, treses)
        if i < len(a)-1 and a[i+1][j] == '3' and (i+1, j) not in treses:
            Buscar3(i+1, j, treses)
        if j < len(a[i])-1 and a[i][j+1] == '3' and (i, j+1) not in treses:
            Buscar3(i, j+1, treses)
        
        return treses
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] == '3':
                treses = Buscar3(i, j, set())
                
                if len(treses) >= 3:
                    x, y = random.choice(list(treses))
                    a[x][y] = '4'
                    for p, q in treses:
                        if (p, q) != (x, y):
                            a[p][q] = '.'
    
    return a
    

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#


crearFichero()
a = []      #Creamos la TREMENDA lista de listas, llamada 'a'
with open('Listado.txt', 'r') as movimientos:
    for linea in movimientos.read().splitlines():
        lista_de_caracteres = list(linea)
        a.append(lista_de_caracteres)

#Inicializamos cosas..
almacenado='.'    #Iniciamos el almacen (almacenado) 
turno=1    #Iniciamos el contador de turnos
edades=creaMatrizBigfoot(a)
yasemovio=False


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////#


while ValidPuntos(a)==False:

    actualizado=False   #Variable para trabajar con el almacen
    cambio=False    #Variable para evolucionar los valores cuando se colapsan

    actual=Actual() #Obtenemos el valor que tendra que manejar el usuario (actual) 
    puntuacion=Puntuacion(a)   
    printTablero(a,actual,almacenado,turno,puntuacion)   #cada turno mostramos el tablero actualizado
    
    entrada,entradaX,entradaY,almacenado,actual,salto=Entrada(almacenado,actual)    #obtenemos la entrada del usuario (entradaX y entradaY)

    if entrada=='*':    #Si el usuario a introducido '*' y no ha almacenado nada, entonces tendremos que actualizar 'actual' con el valor almacenado
        if salto == False:
            almacenado, actual,salto = Almacen(almacenado, actual, salto)
            # Si se actualizó el valor de actual, podemos imprimirlo para verificar que se cambió correctamente
            print("Nuevo valor de actual: ", actual)
            actualizado=True    #Si actualizamos 'actual' es True    

        if salto==True:     #salto es una variable que utilizamos para el almacen, si es True el usuario a guardado un valor, si es False no lo ha hecho
            continue

    if actual!='w':  
        while ValidCasilla(entradaX,entradaY,a)==False:
            if actualizado!=True:   #Para evitar errores con el almacen(el mensaje salia en ocasiones innecesarias)
                print("Esa casilla ya esta ocupada")
                entrada,entradaX,entradaY,almacenado,actual,salto=Entrada(almacenado,actual)    #obtenemos la entrada del usuario (entradaX y entradaY)
        if ValidCasilla(entradaX,entradaY,a)==True:
            a[entradaX][entradaY] = actual
    
    elif actual=='w':
        while ValidCasilla(entradaX,entradaY,a)==True:
                print("Esa casilla esta vacia")
                entrada,entradaX,entradaY,almacenado,actual,salto=Entrada(almacenado,actual)    #obtenemos la entrada del usuario (entradaX y entradaY)
        if ValidCasilla(entradaX,entradaY,a)==False:
            a[entradaX][entradaY] = '.'

    b=copiaMatriz(a)
    cuentaelem=0
    cuentaelem=cuentaElem(b, entradaX, entradaY, actual,cuentaelem)
    print (cuentaelem)
    if cuentaelem>=3:   #Si no son tres no colapsamos
        cambio=Movicion(a,entradaX,entradaY,actual,cambio)
        if cambio==True:    #Al colapsar los valores evolucionan a un nivel superior , si no colapsan no evolucionan
            if actual=='a':
                a[entradaX][entradaY]='b'
            if actual=='b':
                a[entradaX][entradaY]='c'
            if actual=='c':
                a[entradaX][entradaY]='d'
            if actual=='d':
                a[entradaX][entradaY]='e'

    else:
        if actual!='w':
            a[entradaX][entradaY]=actual


    #edad de los bigfoots
    i=0
    j=0
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]=='1':
                edades[i][j]=edades[i][j]+1

    #Miramos los bigfoots y les movemos
    i=0
    j=0
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j]=='1' and a[i][j]!=a[entradaX][entradaY]:
                if yasemovio==False:
                    yasemovio=movBigfoot(a,i,j,edades,yasemovio)
                else:
                    yasemovio=False
        yasemovio=False #Al cambiar de Fila tambien se cambia el yasemovio

    #Juntamos los posibles doses y treses agrupados
    a=Simplificar2(a)   
    a=Simplificar3(a)
            
    if salto==False and actualizado==True:      #Si el usuario ha utilizado un valor en el almacen, lo vaciamos
            almacenado='.'

    turno+=1
