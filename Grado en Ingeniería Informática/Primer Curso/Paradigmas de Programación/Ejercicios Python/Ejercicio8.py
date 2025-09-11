def enComun(lista1,lista2):
    lista3=[]
    for i in lista1:
        if i in lista2:
            lista3.append(i)
    return lista3
    
lista1=[]
lista2=[]

a=input("Introduce lo que quieras en la lista1:")   
while a!="":
    lista1.append(a)
    a=input("otro")
   
b=input("Introduce lo que quieras en la lista2:")
while b!="":
    lista2.append(b)
    b=input()

for i in range(len(lista1)):
    print(lista1[i],end=" ")
print()
for j in range(len(lista2)):
    print(lista2[j],end=" ")
    a=len(lista2)

lista3=enComun(lista1,lista2)

for j in range(len(lista3)):
    print(lista3[j])