def esPalindromo(palabra):
    palabra=palabra.lower() #Ponemos todo en minusculas
    palabra=palabra.replace(" ","") #Reemplazamos los espacios (los quitamos)
    a=0
    b=len(palabra)-1
    for i in range(0,len(palabra)):
        if palabra[a]==palabra[b]:
            a+=a
            b-=b
        else:
            return False
    return True


palabra=input("Introduzca la palabra/frase que quiera comprobrar: ")

if esPalindromo(palabra)==True:
    print("Esa palabra/frase es un palindromo")
else:
    print("No es un palindromo")



