print("Mete un numero")
Num=int(input())
x=2
while x<Num:
    if  Num%x==0:
        print("El numero no es primo")
        x=Num+x
    x+=1
    if x==Num:
        print("Es primo")
    